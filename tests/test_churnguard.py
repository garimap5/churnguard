import numpy as np
import pytest
from fastapi.testclient import TestClient

from churnguard.agent import PolicyRetriever, RetentionCopilot, validate_message
from churnguard.api import create_app
from churnguard.data import clv, simulate_cohort
from churnguard.features import FEATURES, LEAKY, build_features
from churnguard.models import (ChurnGuardModels, policy_value, qini_coefficient, top_fraction, two_proportion_test)


@pytest.fixture(scope="module")
def world():
    subs, usage, tix = simulate_cohort(6000, "A", seed=1)
    X = build_features(subs, usage, tix)
    s = subs.set_index("sub_id").loc[X.index]
    m = ChurnGuardModels().fit(X, s.churned, s.treated)
    return subs, usage, tix, X, s, m


def test_sql_features_match_pandas(world):
    subs, usage, _, X, _, _ = world
    u = usage[usage.sub_id == subs.sub_id[0]].sort_values("month")
    row = X.loc[subs.sub_id[0]]
    assert row.data_gb_3m == pytest.approx(u.data_gb.iloc[-3:].mean(), rel=1e-6)
    assert row.data_trend == pytest.approx(u.data_gb.iloc[-3:].mean() / u.data_gb.iloc[:3].mean(), rel=1e-6)
    assert row.drop_rate_max == pytest.approx(u.drop_rate.max())


def test_no_label_leakage_in_features(world):
    X = world[3]
    assert not LEAKY & set(X.columns) and set(FEATURES) <= set(X.columns)
    assert not X[FEATURES].isna().any().any()


def test_campaign_is_randomised(world):
    s = world[4]
    assert abs(s.treated.mean() - 0.30) < 0.02
    # treatment independent of observable risk: treated/control churn-risk proxies similar
    assert abs(s[s.treated == 1].arpu.mean() - s[s.treated == 0].arpu.mean()) < 0.1 * s.arpu.mean()
    assert two_proportion_test(s.churned, s.treated)["diff_pp"] < 0   # offer reduces churn on average


def test_policy_value_estimator_is_unbiased_on_simulator():
    """With known true uplift, the randomised-data estimator should track the oracle value."""
    subs, _, _ = simulate_cohort(40000, "B", seed=5)
    sel = subs.true_uplift.values * clv(subs.arpu.values) > 120
    est = policy_value(sel, subs.treated.values, subs.churned.values, clv(subs.arpu.values))
    oracle = float((subs.true_uplift.values * clv(subs.arpu.values) - 120)[sel].sum())
    assert est == pytest.approx(oracle, abs=0.35 * abs(oracle) + 20000)


def test_uplift_ranking_beats_random_in_sample_cohort(world):
    _, _, _, X, s, m = world
    rnd = np.random.default_rng(0).random(len(s))
    assert qini_coefficient(m.uplift_score(X), 1 - s.churned, s.treated) > qini_coefficient(rnd, 1 - s.churned, s.treated) - 1e-3
    assert 0 <= m.risk_score(X).min() and m.risk_score(X).max() <= 1


def test_top_fraction_size():
    assert top_fraction(np.arange(100), 0.2).sum() == 20


# ---------------------------------------------------------------- agent
def test_validate_message_blocks_unapproved_figures():
    label = "10% cashback on your next two recharges (max Rs 100)"
    assert validate_message("Enjoy 10% cashback on your next two recharges (max Rs 100)!", label)
    assert not validate_message("Enjoy 50% cashback on your next recharge", label)
    assert not validate_message("We will give you Rs 500 off", label)


def test_policy_retrieval_routes_drivers():
    r = PolicyRetriever()
    assert r.search("high dropped call rate poor network quality", 1)[0][0]["id"] == "P2"
    assert r.search("price sensitive frequent offer redemptions falling recharge value", 1)[0][0]["id"] in {"P3", "P4"}


def test_copilot_guardrails(world):
    subs, _, _, X, s, m = world
    cp = RetentionCopilot(m, X, s.drop(columns=["treated", "churned", "true_uplift"]))
    decisions = [cp.decide(i) for i in X.index[:400]]
    acts = {d.action for d in decisions}
    assert "no_action" in acts
    for d in decisions:
        assert d.policies, "every decision must cite a policy"
        if d.action == "offer":
            row = cp.subs.loc[d.sub_id]
            assert d.net_value > 0 and d.uplift_pp > 0                      # never offer at non-positive value
            assert d.offer["cost"] <= 0.15 * row.arpu * 12                  # spend cap
            if row.dnd or not row.marketing_consent:
                assert d.channel == "in_app_or_inbound_only"                # consent / DND respected
            assert X.loc[d.sub_id].unresolved_tickets == 0                  # tickets first
            assert row.plan_type in {"prepaid", "postpaid"} and d.message
    # a hostile LLM cannot smuggle in unapproved figures
    cp2 = RetentionCopilot(m, X, cp.subs, complete=lambda p: "Amazing! Get 90% off everything, Rs 9999 credit!")
    for i in X.index[:400]:
        d = cp2.decide(i)
        if d.action == "offer":
            assert "9999" not in d.message and "90%" not in d.message
            break


def test_api(world):
    subs, _, _, X, s, m = world
    c = TestClient(create_app({"models": m, "features": X, "subs": s.drop(columns=["treated", "churned", "true_uplift"])}))
    sid = X.index[0]
    r = c.get(f"/score/{sid}")
    assert r.status_code == 200 and 0 <= r.json()["churn_risk"] <= 1
    assert c.get(f"/copilot/{sid}").json()["policies"]
    assert c.get("/score/DOESNOTEXIST").status_code == 404
    assert c.get("/health").json()["loaded"]
