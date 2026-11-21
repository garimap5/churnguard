"""Retention copilot: a tool-using workflow agent with retrieval over policy documents and hard guardrails.

Design stance: *models decide, policies constrain, the LLM only words the result.*
  tools   : get_profile -> score (risk, uplift, CLV, drivers) -> retrieve_policy (RAG) -> choose_offer -> guardrails
  RAG     : TF-IDF retrieval over a policy knowledge base; the offer is read from the retrieved policy's
            structured fields, never invented, and every decision cites its policy ids.
  LLM     : optional (`complete` callable). It may only rephrase a customer message; `validate_message`
            rejects any text that mentions a rupee amount / percentage not present in the approved offer.
            Without an LLM a template is used, so the whole thing runs and is tested offline.
The policy texts below are ILLUSTRATIVE (written for this demo), not any operator's real policies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import OFFER_COST, clv

KB = [
    dict(id="P1", title="Usage-decline data booster",
         text="Customers whose data usage and voice usage are falling month over month, low usage trend, declining engagement, "
              "may be offered a data booster pack of 2GB per day for 14 days.",
         offer=dict(type="data_booster", cost=60, label="a 2GB/day data booster for 14 days"), plans=["prepaid", "postpaid"]),
    dict(id="P2", title="Network-quality remediation",
         text="High dropped call rate, poor network quality, network complaints. Discounts are NOT effective for network-driven churn. "
              "Raise a network ticket, schedule a fix callback within 24 hours and give a goodwill credit.",
         offer=dict(type="network_credit", cost=50, label="a Rs 50 goodwill credit and a network-fix callback within 24 hours"),
         plans=["prepaid", "postpaid"], discount=False),
    dict(id="P3", title="Price-sensitive prepaid cashback",
         text="Price sensitive prepaid customers with frequent offer redemptions and falling recharge value may receive "
              "10 percent cashback on the next two recharges capped at Rs 100.",
         offer=dict(type="cashback", cost=100, label="10% cashback on your next two recharges (max Rs 100)"), plans=["prepaid"], discount=True),
    dict(id="P4", title="Postpaid loyalty bill credit",
         text="Postpaid customers with long tenure, loyalty, high ARPU and falling usage can receive a loyalty bill credit of up to Rs 150. "
              "Credits above Rs 100 need agent approval.",
         offer=dict(type="bill_credit", cost=150, label="a loyalty bill credit of up to Rs 150"), plans=["postpaid"], discount=True,
         needs_approval_above=100),
    dict(id="P5", title="Resolve tickets first",
         text="Open or unresolved support tickets, repeated complaints. Resolve open tickets before any commercial offer; "
              "escalate to L2 if there are more than 3 tickets in 90 days.", offer=None),
    dict(id="P6", title="Consent and DND",
         text="Outbound promotional contact is allowed only with marketing consent and not on the DND registry. "
              "Otherwise use in-app or inbound-only channels. Never call or SMS DND customers with promotions.", offer=None),
    dict(id="P7", title="Spend guardrail",
         text="Retention spend per customer must not exceed 15 percent of annual ARPU and expected net value must be positive.", offer=None),
    dict(id="P8", title="Do not contact when uplift is not positive",
         text="If the predicted uplift is zero or negative, do not contact: an offer may increase churn (sleeping dogs).", offer=None),
]
PHRASES = {"data_trend": "falling data usage", "voice_trend": "falling voice usage", "recharge_trend": "falling recharge value",
           "recharge_3m": "low recharge value", "data_gb_3m": "low data usage", "voice_min_3m": "low voice usage",
           "drop_rate_3m": "high dropped call rate poor network quality", "drop_rate_max": "high dropped call rate poor network quality",
           "tickets_90d": "support tickets complaints", "unresolved_tickets": "unresolved open support tickets",
           "offer_redemptions_6m": "price sensitive frequent offer redemptions", "tenure_months": "short tenure",
           "plan_type_postpaid": "prepaid plan", "arpu": "low arpu"}


class PolicyRetriever:
    def __init__(self, kb=KB):
        self.kb = kb
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.M = self.vec.fit_transform([d["title"] + ". " + d["text"] for d in kb])

    def search(self, query: str, k: int = 3):
        s = cosine_similarity(self.vec.transform([query]), self.M)[0]
        return [(self.kb[i], float(s[i])) for i in np.argsort(-s)[:k] if s[i] > 0]


_NUM = re.compile(r"(?:rs\.?\s*|₹\s*)?\d+(?:\.\d+)?\s*(?:%|percent|gb|rs)?", re.I)


def validate_message(msg: str, offer_label: str) -> bool:
    """Every number in the customer message must appear in the approved offer wording."""
    allowed = {n.strip().lower() for n in _NUM.findall(offer_label)}
    allowed |= {re.sub(r"\D", "", a) for a in allowed}
    nums = [n.strip().lower() for n in _NUM.findall(msg)]
    return all(n in allowed or re.sub(r"\D", "", n) in allowed for n in nums)


@dataclass
class Decision:
    sub_id: str
    action: str                     # no_action | offer | resolve_ticket_first | escalate
    channel: str | None = None
    offer: dict | None = None
    risk: float = 0.0
    uplift_pp: float = 0.0
    clv: float = 0.0
    net_value: float = 0.0
    drivers: list = field(default_factory=list)
    policies: list = field(default_factory=list)       # cited policy ids
    needs_approval: bool = False
    message: str | None = None
    trace: list = field(default_factory=list)

    def dict(self):
        return self.__dict__.copy()


class RetentionCopilot:
    def __init__(self, models, features, subs, complete: Callable[[str], str] | None = None):
        self.m, self.X, self.subs, self.complete = models, features, subs, complete
        self.ret = PolicyRetriever()

    # ---- tools ------------------------------------------------------------------------------
    def get_profile(self, sub_id):
        return self.subs.loc[sub_id], self.X.loc[[sub_id]]

    def score(self, sub_id):
        s, x = self.get_profile(sub_id)
        up = float(self.m.uplift_score(x)[0])
        return dict(risk=float(self.m.risk_score(x)[0]), uplift=up, clv=float(clv(s.arpu)),
                    net_value=up * float(clv(s.arpu)) - OFFER_COST, drivers=self.m.drivers(x)[0])

    # ---- orchestration ---------------------------------------------------------------------------
    def decide(self, sub_id: str) -> Decision:
        s, x = self.get_profile(sub_id)
        sc = self.score(sub_id)
        d = Decision(sub_id, "no_action", risk=sc["risk"], uplift_pp=100 * sc["uplift"], clv=sc["clv"], net_value=sc["net_value"],
                     drivers=[n for n, _ in sc["drivers"]])
        d.trace.append(f"score: risk={sc['risk']:.2f} uplift={100 * sc['uplift']:.1f}pp clv=Rs{sc['clv']:.0f} net=Rs{sc['net_value']:.0f}")

        if sc["uplift"] <= 0:
            d.policies, d.trace = ["P8"], d.trace + ["P8: uplift <= 0 -> do not contact"]
            return d
        if sc["net_value"] <= 0:
            d.policies, d.trace = ["P7"], d.trace + ["P7: expected net value <= 0 -> no offer"]
            return d
        if int(x.unresolved_tickets.iloc[0]) > 0 or int(x.tickets_90d.iloc[0]) > 3:
            esc = int(x.tickets_90d.iloc[0]) > 3
            d.action, d.policies = ("escalate" if esc else "resolve_ticket_first"), ["P5"]
            d.trace.append("P5: open tickets -> resolve before any commercial offer" + (" (escalate to L2)" if esc else ""))
            return d

        q = " ".join(PHRASES.get(n, n) for n in d.drivers) or "declining usage"
        hits = [(p, sim) for p, sim in self.ret.search(q, 4) if p["offer"] and s.plan_type in p["plans"]]
        d.trace.append(f"retrieve({q!r}) -> {[(p['id'], round(sim, 2)) for p, sim in self.ret.search(q, 4)]}")
        if not hits:
            d.policies, d.trace = ["P8"], d.trace + ["no eligible offer policy -> no action"]
            return d
        pol = hits[0][0]
        # P7 spend cap
        cap = 0.15 * float(s.arpu) * 12
        if pol["offer"]["cost"] > cap or pol["offer"]["cost"] >= sc["uplift"] * sc["clv"]:
            d.policies, d.trace = ["P7", pol["id"]], d.trace + [f"P7: offer cost {pol['offer']['cost']} breaches spend/ROI guardrail"]
            return d
        d.action, d.offer, d.policies = "offer", pol["offer"], [pol["id"]]
        d.needs_approval = pol.get("needs_approval_above", 1e9) < pol["offer"]["cost"]
        # P6 consent / DND -> channel
        ok = bool(s.marketing_consent) and not bool(s.dnd)
        d.channel = "outbound_sms_or_call" if ok else "in_app_or_inbound_only"
        d.policies.append("P6")
        d.trace.append(f"{pol['id']}: {pol['title']} -> {pol['offer']['label']}; P6 channel={d.channel}")
        d.message = self._message(d)
        return d

    def _message(self, d: Decision) -> str:
        template = f"As a valued customer, we'd like to give you {d.offer['label']}. Reply YES or visit the app to activate."
        if not self.complete:
            return template
        prompt = ("Rewrite this customer message in a warm, concise tone (max 2 sentences). Keep the offer wording EXACTLY; "
                  f"do not add any discount, number or promise.\nMessage: {template}")
        try:
            out = self.complete(prompt).strip()
        except Exception:                                   # LLM failure must never block the workflow
            return template
        if validate_message(out, d.offer["label"]):
            return out
        d.trace.append("guardrail: LLM message introduced unapproved figures -> used template")
        return template
