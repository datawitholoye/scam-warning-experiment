"""Turn payment logs into the customer-level table an experiment is analysed on.

Randomisation was by customer, so the customer is the unit of analysis: one
row per customer, including customers who made no risky payments at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400


def customer_table(customers: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    p = payments.assign(
        genuine_warned=payments.warned & ~payments.is_scam,
        scam_completed=payments.is_scam & payments.completed,
    )
    agg = p.groupby("customer_id").agg(
        payments=("payment_id", "size"),
        flagged=("flagged", "sum"),
        warned=("warned", "sum"),
        genuine_warned=("genuine_warned", "sum"),
        abandoned_genuine=("abandoned_genuine", "sum"),
        support_contacts=("support_contact", "sum"),
        scam_attempts=("is_scam", "sum"),
        scams_completed=("scam_completed", "sum"),
        scam_loss=("scam_loss", "sum"),
    )
    df = customers.set_index("customer_id").join(agg).fillna(
        {c: 0 for c in agg.columns}).reset_index()
    df["treat"] = df["arm"].eq("treatment")
    # "Triggered" = would have been warned. The risk score runs in BOTH arms;
    # control simply doesn't show the warning (counterfactual logging).
    df["triggered"] = df["flagged"] > 0
    return df


def daily_cumulative(customers: pd.DataFrame, payments: pd.DataFrame, column: str = "scam_loss",
                     days: int | None = None) -> np.ndarray:
    """customers x days matrix of the metric accumulated up to the end of each day.
    Used to replay the experiment day by day for the peeking analysis."""
    day = (payments["seconds_since_start"] // SECONDS_PER_DAY).astype(int)
    days = days or int(day.max()) + 1
    pos = pd.Series(np.arange(len(customers)), index=customers["customer_id"])
    rows = pos.loc[payments["customer_id"]].to_numpy()
    m = np.zeros((len(customers), days))
    np.add.at(m, (rows, day.to_numpy()), payments[column].to_numpy(float))
    return np.cumsum(m, axis=1)
