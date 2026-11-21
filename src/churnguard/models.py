"""Churn-risk model, T-learner uplift model, and honest evaluation of targeting policies."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .data import OFFER_COST, clv
from .features import FEATURES


class ChurnGuardModels:
    """risk  : P(churn | no offer), trained on the CONTROL arm only (treated outcomes are altered by the offer)
    uplift: T-learner, two outcome models (treated / control); uplift = P(churn|control) - P(churn|treated)
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y, treated):
        X, y, t = X[FEATURES], np.asarray(y), np.asarray(treated)
        ctrl = t == 0
        Xc, yc = X[ctrl], y[ctrl]
        Xtr, Xva, ytr, yva = train_test_split(Xc, yc, test_size=0.2, random_state=self.seed, stratify=yc)
        self.risk = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.04, subsample=.8, colsample_bytree=.8,
                                  min_child_weight=10, early_stopping_rounds=40, eval_metric="logloss",
                                  n_jobs=4, random_state=self.seed)
        self.risk.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(self.risk.predict_proba(Xva)[:, 1], yva)
        self.baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)

        kw = dict(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=.8, colsample_bytree=.8,
                  min_child_weight=30, reg_lambda=5.0, n_jobs=4, random_state=self.seed)
        # --- uplift: choose between T-learner and transformed-outcome learner on a VALIDATION split of A
        idx = np.arange(len(X))
        fit_i, val_i = train_test_split(idx, test_size=0.25, random_state=self.seed)
        cand = {}
        for kind in ("t_learner", "transformed_outcome"):
            self._fit_uplift(X.iloc[fit_i], y[fit_i], t[fit_i], kind, kw)
            sc = self._uplift(X.iloc[val_i], kind)
            cand[kind] = qini_coefficient(sc, 1 - y[val_i], t[val_i])
        self.uplift_selection = cand
        self.uplift_kind = max(cand, key=cand.get)
        self._fit_uplift(X, y, t, self.uplift_kind, kw)          # refit the winner on all of cohort A
        return self

    def _fit_uplift(self, X, y, t, kind, kw):
        if kind == "t_learner":
            self.m_ctrl = XGBClassifier(**kw).fit(X[t == 0], y[t == 0])
            self.m_trt = XGBClassifier(**kw).fit(X[t == 1], y[t == 1])
        else:   # Z = retained * (T - e) / (e (1 - e)),  E[Z | X] = uplift in retention (e known: randomised)
            e = t.mean()
            z = (1 - y) * (t - e) / (e * (1 - e))
            self.m_to = XGBRegressor(**{**kw, "min_child_weight": 50}).fit(X, z)

    def _uplift(self, X, kind):
        if kind == "t_learner":
            return self.m_ctrl.predict_proba(X)[:, 1] - self.m_trt.predict_proba(X)[:, 1]
        return self.m_to.predict(X)                              # = reduction in churn probability

    def risk_score(self, X):
        return self.calibrator.predict(self.risk.predict_proba(X[FEATURES])[:, 1])

    def baseline_score(self, X):
        return self.baseline.predict_proba(X[FEATURES])[:, 1]

    def uplift_score(self, X, kind=None):
        return self._uplift(X[FEATURES], kind or self.uplift_kind)

    def net_value(self, X, arpu):
        """Expected net value (INR) of making the offer: P(churn) reduction x CLV - offer cost."""
        return self.uplift_score(X) * clv(arpu) - OFFER_COST

    def drivers(self, X, top: int = 3):
        """Per-customer local explanation from XGBoost's built-in TreeSHAP (pred_contribs)."""
        import xgboost as xgb
        contrib = self.risk.get_booster().predict(xgb.DMatrix(X[FEATURES]), pred_contribs=True)[:, :-1]
        names = np.array(FEATURES)
        out = []
        for row in contrib:
            o = np.argsort(-row)[:top]
            out.append([(names[i], float(row[i])) for i in o if row[i] > 0])
        return out


# ---------------------------------------------------------------- evaluation helpers
def risk_report(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "lift_top10pct": float(y[top].mean() / y.mean()), "Brier": brier_score_loss(y, p), "ECE": float(ece),
            "base_rate": float(y.mean())}


def two_proportion_test(y, t) -> dict:
    """Randomised campaign: difference in churn rate, two-sided z-test, 95% CI."""
    y, t = np.asarray(y), np.asarray(t)
    p1, p0, n1, n0 = y[t == 1].mean(), y[t == 0].mean(), (t == 1).sum(), (t == 0).sum()
    pp = y.mean()
    z = (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {"churn_treated": p1, "churn_control": p0, "diff_pp": 100 * (p1 - p0),
            "CI95_pp": [100 * (p1 - p0 - 1.96 * se), 100 * (p1 - p0 + 1.96 * se)],
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}


def qini_curve(score, retained, treated):
    """Cumulative incremental retained customers when treating the top-k by `score`."""
    o = np.argsort(-np.asarray(score))
    r, t = np.asarray(retained)[o], np.asarray(treated)[o]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    rt, rc = np.cumsum(r * t), np.cumsum(r * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (rt / np.maximum(nt, 1) - rc / np.maximum(nc, 1)) * (nt + nc)
    return np.arange(1, len(r) + 1), q


def qini_coefficient(score, retained, treated) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal, / n^2 (comparable across models)."""
    k, q = qini_curve(score, retained, treated)
    diag = q[-1] * k / len(k)
    return float(np.trapezoid(q - diag, k) / len(k) ** 2)


def policy_value(selected, treated, churned, value, cost=OFFER_COST) -> float:
    """Unbiased (randomised data) estimate of net INR value of offering to `selected`:
    |S| * (value-weighted churn loss in control - in treated) - cost * |S|."""
    s = np.asarray(selected, bool)
    t, c = s & (np.asarray(treated) == 1), s & (np.asarray(treated) == 0)
    if t.sum() == 0 or c.sum() == 0:
        return 0.0
    loss = lambda m: (np.asarray(churned)[m] * np.asarray(value)[m]).mean()
    return float(s.sum() * (loss(c) - loss(t)) - cost * s.sum())


def top_fraction(score, frac):
    m = np.zeros(len(score), bool)
    m[np.argsort(-np.asarray(score))[: int(round(frac * len(score)))]] = True
    return m


def bootstrap_policy_diff(sel_a, sel_b, treated, churned, value, n=500, seed=0):
    """Paired bootstrap CI of value(A) - value(B)."""
    rng = np.random.default_rng(seed)
    N = len(treated)
    d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(policy_value(sel_a[i], treated[i], churned[i], value[i]) - policy_value(sel_b[i], treated[i], churned[i], value[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
