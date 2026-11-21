from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_PATH = Path(__file__).with_name("features.sql")

FEATURES = ["plan_type_postpaid", "tenure_months", "arpu", "device_tier", "is_5g", "circle",
            "offer_redemptions_6m", "data_gb_3m", "data_trend", "voice_min_3m", "voice_trend",
            "recharge_3m", "recharge_trend", "drop_rate_3m", "drop_rate_max", "tickets_90d",
            "unresolved_tickets"]
LEAKY = {"churned", "treated", "true_uplift"}


def build_features(subs: pd.DataFrame, usage: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """Run the SQL feature layer in an in-memory SQLite database. Outcome columns never enter it."""
    con = sqlite3.connect(":memory:")
    subs.drop(columns=[c for c in LEAKY if c in subs]).to_sql("subscribers", con, index=False)
    usage.to_sql("usage_monthly", con, index=False)
    tickets.to_sql("tickets", con, index=False)
    con.execute("CREATE INDEX i1 ON usage_monthly(sub_id, month)")
    df = pd.read_sql_query(SQL_PATH.read_text(), con)
    con.close()
    df["plan_type_postpaid"] = (df.pop("plan_type") == "postpaid").astype(int)
    return df.set_index("sub_id")
