"""End to end: simulate -> SQL features -> train on cohort A -> evaluate OUT-OF-TIME on cohort B.

    python scripts/run_all.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churnguard.data import OFFER_COST, clv, simulate  # noqa: E402
from churnguard.features import build_features  # noqa: E402
from churnguard.models import (ChurnGuardModels, bootstrap_policy_diff, policy_value, qini_coefficient,  # noqa: E402
                               qini_curve, risk_report, top_fraction, two_proportion_test)

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def main():
    subs, usage, tix = simulate()
    X = build_features(subs, usage, tix)
    subs = subs.set_index("sub_id").loc[X.index]
    A, B = (subs.cohort == "A").values, (subs.cohort == "B").values
    print(f"{len(subs)} subscribers | usage rows {len(usage):,} | A={A.sum()} B={B.sum()}")
    R = {}

    # 1) campaign A/B test on the training cohort (randomised) --------------------------------
    R["campaign_AB_test_cohortA"] = two_proportion_test(subs.churned[A], subs.treated[A])
    print("campaign A/B:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["campaign_AB_test_cohortA"].items()})

    # 2) train on cohort A ----------------------------------------------------------------------
    m = ChurnGuardModels().fit(X[A], subs.churned[A], subs.treated[A])

    # 3) risk model, out-of-time on cohort B control arm -----------------------------------------
    Bc = B & (subs.treated == 0).values
    yB = subs.churned[Bc].values
    R["risk_model_cohortB_control"] = {"XGBoost+isotonic": risk_report(yB, m.risk_score(X[Bc])),
                                       "LogisticRegression": risk_report(yB, m.baseline_score(X[Bc]))}
    for k, v in R["risk_model_cohortB_control"].items():
        print(f"risk  {k:20s}", {a: round(b, 3) for a, b in v.items()})

    # 4) uplift & targeting policies on cohort B (randomised treatment) --------------------------
    XB, sB = X[B], subs[B]
    t, y = sB.treated.values, sB.churned.values
    value = clv(sB.arpu.values)
    ret = 1 - y
    s_risk, s_up = m.risk_score(XB), m.uplift_score(XB)
    s_val = m.net_value(XB, sB.arpu.values)
    R["uplift"] = {
        "selected_on_cohortA_validation": {"chosen": m.uplift_kind, "val_qini": m.uplift_selection},
        "qini_coef": {f"uplift_model({m.uplift_kind})": qini_coefficient(s_up, ret, t),
                      "churn_risk_ranking": qini_coefficient(s_risk, ret, t),
                      "random": qini_coefficient(np.random.default_rng(0).random(len(t)), ret, t)},
        "spearman_pred_vs_TRUE_uplift(sim only)": {"uplift_model": float(spearmanr(s_up, sB.true_uplift).statistic),
                                                   "churn_risk": float(spearmanr(s_risk, sB.true_uplift).statistic)}}
    print("uplift:", json.dumps(R["uplift"], indent=1, default=float))

    budgets = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(0)
    curve = {"random": [], "churn-risk": [], "uplift": [], "uplift x CLV (ours)": []}
    for b in budgets:
        curve["random"].append(np.mean([policy_value(top_fraction(rng.random(len(t)), b), t, y, value) for _ in range(20)]))
        curve["churn-risk"].append(policy_value(top_fraction(s_risk, b), t, y, value))
        curve["uplift"].append(policy_value(top_fraction(s_up, b), t, y, value))
        curve["uplift x CLV (ours)"].append(policy_value(top_fraction(s_val, b), t, y, value))
    R["policy_net_value_INR_by_budget"] = {"budgets": budgets, **curve}

    rule = s_val > 0                                         # treat only if expected net value > 0
    n_all = np.ones(len(t), bool)
    top20_risk = top_fraction(s_risk, 0.2)
    R["policy_headline"] = {
        "treat_all_INR": policy_value(n_all, t, y, value),
        "top20pct_by_churn_risk_INR": policy_value(top20_risk, t, y, value),
        "net_value_rule_INR": policy_value(rule, t, y, value), "net_value_rule_share_treated": float(rule.mean()),
        "rule_vs_no_campaign": bootstrap_policy_diff(rule, np.zeros(len(t), bool), t, y, value),
        "rule_minus_treat_all": bootstrap_policy_diff(rule, n_all, t, y, value),
        "rule_minus_top20_risk": bootstrap_policy_diff(rule, top20_risk, t, y, value)}
    print("policy:", json.dumps(R["policy_headline"], indent=1, default=float))
    print("net value by budget:", {k: [round(x) for x in v] for k, v in curve.items()})

    # 5) plots -------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in (("uplift model", s_up), ("churn-risk ranking", s_risk), ("random", np.random.default_rng(0).random(len(t)))):
        k, q = qini_curve(s, ret, t); ax[0].plot(k / len(k), q, label=name)
    ax[0].set_xlabel("fraction of base targeted"); ax[0].set_ylabel("incremental retained customers")
    ax[0].set_title("Qini curves (cohort B, out-of-time)"); ax[0].legend()
    for name, v in curve.items():
        ax[1].plot(np.array(budgets) * 100, v, marker="o", label=name)
    ax[1].axhline(0, c="k", lw=.5); ax[1].set_xlabel("% of base offered a retention deal"); ax[1].set_ylabel("net value (INR)")
    ax[1].set_title("Campaign net value vs budget"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "uplift_and_policy.png", dpi=140)

    # 6) persist ---------------------------------------------------------------------------------
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump({"models": m, "features": XB, "subs": sB.drop(columns=["treated", "churned", "true_uplift"])}, "artifacts/churnguard.joblib")
    print("wrote results/results.json and artifacts/churnguard.joblib")


if __name__ == "__main__":
    main()
