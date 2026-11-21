"""Telecom subscriber simulator with a *randomised* retention campaign and known true uplift.

Why simulate: real telco data with a randomised treatment arm is not public. A simulator whose
ground truth is known lets us (a) test whether uplift modelling recovers heterogeneous effects,
(b) evaluate targeting policies honestly, and (c) ship a repo anyone can run. The schema mirrors
what a telco holds: a subscriber master, monthly usage, and support tickets.

Cohort A (train) and cohort B (test, later period) differ by a base-rate shift and a higher 5G share,
so evaluation is out-of-time, not a random split.

Built-in structure (what the models must discover):
  * churn rises with price sensitivity, network problems, disengagement, short tenure, prepaid
  * the offer helps *price-sensitive* customers, does nothing for *network-driven* churners,
    and slightly backfires ("sleeping dogs") for long-tenure low-sensitivity postpaid customers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MONTHS = 6
MARGIN, OFFER_COST = 0.35, 120.0          # INR: contribution margin on ARPU; cost of one retention offer


def _sig(x):
    return 1 / (1 + np.exp(-x))


def simulate_cohort(n: int, cohort: str, seed: int):
    rng = np.random.default_rng(seed)
    shift, p5g = (0.0, 0.25) if cohort == "A" else (0.25, 0.35)
    price_sens, net_issue, eng = rng.normal(size=(3, n))          # latent traits (never observed)
    postpaid = rng.random(n) < 0.25
    tenure = np.clip(1 + rng.exponential(30, n).astype(int), 1, 180)
    circle = rng.integers(0, 10, n)
    circle_eff = np.random.default_rng(123).permutation(np.linspace(-.3, .3, 10))[circle]
    device = rng.choice([0, 1, 2], n, p=[.45, .40, .15])
    is5g = rng.random(n) < p5g
    arpu = np.where(postpaid, 700 * np.exp(.35 * rng.normal(size=n)), 230 * np.exp(.30 * rng.normal(size=n))) * np.exp(.15 * eng)
    redeem = rng.poisson(0.8 + 1.6 * _sig(price_sens))            # observable proxy of price sensitivity
    consent = rng.random(n) < 0.70
    dnd = rng.random(n) < 0.20

    intent = 0.45 * price_sens + 0.5 * net_issue - 0.4 * eng + 0.6 * rng.normal(size=n)  # disengagement
    m = np.arange(N_MONTHS)[None, :]
    base_gb = np.exp(1.4 + .45 * eng + .2 * is5g)[:, None]
    data_gb = base_gb * np.exp(-0.09 * intent[:, None] * m) * np.exp(rng.normal(0, .15, (n, N_MONTHS)))
    voice = np.exp(5 + .3 * eng)[:, None] * np.exp(-0.06 * intent[:, None] * m) * np.exp(rng.normal(0, .2, (n, N_MONTHS)))
    rch = np.where(postpaid[:, None], arpu[:, None] * np.ones((1, N_MONTHS)),
                   arpu[:, None] * np.exp(-0.07 * intent[:, None] * m) * np.exp(rng.normal(0, .2, (n, N_MONTHS))))
    drop = np.clip(_sig(-3.3 + .75 * net_issue[:, None] + .3 * rng.normal(size=(n, N_MONTHS))), 0, 1)
    tickets = rng.poisson(0.25 + .5 * _sig(1.5 * net_issue) + .2 * (intent > 1))
    unresolved = rng.binomial(tickets, .3)

    logit0 = (-2.45 + .50 * price_sens + .55 * net_issue - .55 * eng + .35 * (~postpaid) - .012 * np.minimum(tenure, 60)
              + .12 * tickets + circle_eff + shift + .35 * rng.normal(size=n))
    tau = (-1.15 * _sig(2 * price_sens) * (1 - _sig(2 * (net_issue - .5)))      # persuadables
           + .45 * (postpaid & (tenure > 48) & (price_sens < 0)))               # sleeping dogs
    treated = rng.random(n) < 0.30                                              # RANDOMISED campaign
    churned = rng.random(n) < _sig(logit0 + treated * tau)

    ids = np.array([f"{cohort}{i:06d}" for i in range(n)])
    subs = pd.DataFrame(dict(
        sub_id=ids, cohort=cohort, plan_type=np.where(postpaid, "postpaid", "prepaid"), tenure_months=tenure,
        arpu=arpu.round(1), device_tier=device, is_5g=is5g.astype(int), circle=circle,
        offer_redemptions_6m=redeem, marketing_consent=consent.astype(int), dnd=dnd.astype(int),
        # outcome / campaign columns: kept OUT of the SQL feature layer
        treated=treated.astype(int), churned=churned.astype(int),
        true_uplift=(_sig(logit0) - _sig(logit0 + tau)),                      # simulator ground truth
    ))
    usage = pd.DataFrame(dict(
        sub_id=np.repeat(ids, N_MONTHS), month=np.tile(np.arange(1, N_MONTHS + 1), n),
        data_gb=data_gb.ravel().round(3), voice_min=voice.ravel().round(1),
        recharge_amt=rch.ravel().round(1), drop_rate=drop.ravel().round(4)))
    tix = pd.DataFrame(dict(sub_id=ids, tickets_90d=tickets, unresolved_tickets=unresolved))
    return subs, usage, tix


def simulate(n_per_cohort: int = 30000, seed: int = 7):
    parts = [simulate_cohort(n_per_cohort, c, seed + i) for i, c in enumerate("AB")]
    return tuple(pd.concat(x, ignore_index=True) for x in zip(*parts))


def clv(arpu) -> np.ndarray:
    """12-month contribution value of keeping a subscriber."""
    return np.asarray(arpu) * 12 * MARGIN
