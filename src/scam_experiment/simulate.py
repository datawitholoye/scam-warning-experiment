"""Simulate a bank running a scam-warning A/B test, with the true answer known.

For every payment the warning could touch, BOTH potential outcomes are drawn
(what happens if warned, what happens if not). Only one is "observed",
depending on the customer's arm, exactly like real life. The other is kept in
a separate truth file so the Day 2-3 analysis can be graded.

    python -m scam_experiment.simulate             # full run -> data/
    python -m scam_experiment.simulate --n 20000   # smaller, faster
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assign import assign
from .config import load_config, resolve

SCAM_TYPES = ["purchase", "impersonation", "investment"]
SECONDS_PER_WEEK = 7 * 86400


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    return np.log(p / (1 - p))


def simulate_customers(cfg: dict, rng: np.random.Generator, n: int | None = None) -> pd.DataFrame:
    c = cfg["customers"]
    n = n or c["n"]
    bands = list(c["age_bands"])
    age = rng.choice(bands, n, p=list(c["age_bands"].values()))
    conf = np.array([c["digital_confidence_mean"][a] for a in age]) + rng.normal(0, 1, n)
    rate = rng.lognormal(np.log(c["payments_per_week_median"]), 0.6, n)       # payments per week
    new_share = rng.beta(*cfg["payments"]["new_payee_share_beta"], n)
    vulnerability = -conf + rng.normal(0, 0.5, n)

    # Pre-period behaviour: observable history, correlated with in-test behaviour
    w = c["pre_period_weeks"]
    pre_payments = rng.poisson(rate * w)
    pre_new_payees = rng.binomial(pre_payments, new_share)
    pre_spend = pre_payments * cfg["payments"]["genuine_amount_median"] * rng.lognormal(0.5, 0.4, n)

    ids = np.char.add("C", np.char.zfill(np.arange(1, n + 1).astype(str), 6))
    return pd.DataFrame({
        "customer_id": ids,
        "age_band": age,
        "tenure_months": rng.integers(1, 121, n),
        "pre_payments": pre_payments,
        "pre_new_payees": pre_new_payees,
        "pre_spend": pre_spend.round(2),
        "arm": assign(ids, cfg["experiment"]["salt"], cfg["experiment"]["treatment_share"]),
        # Latent traits: NOT observable by the bank. Kept for the truth file only.
        "_digital_confidence": conf,
        "_vulnerability": vulnerability,
        "_payments_per_week": rate,
        "_new_payee_share": new_share,
    })


def simulate_payments(customers: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    weeks = cfg["experiment"]["weeks"]
    n_per = rng.poisson(customers["_payments_per_week"].to_numpy() * weeks)
    idx = np.repeat(np.arange(len(customers)), n_per)
    cust = customers.iloc[idx].reset_index(drop=True)
    m = len(idx)

    # ── Scam or genuine? ─────────────────────────────────────────────────────
    s = cfg["scams"]
    new_payee = rng.random(m) < cust["_new_payee_share"].to_numpy()
    p_scam = _sigmoid(s["base_logit"] + s["vulnerability_weight"] * cust["_vulnerability"].to_numpy()
                      + s["new_payee_weight"] * new_payee)
    is_scam = rng.random(m) < p_scam
    new_payee = np.where(is_scam, rng.random(m) < 0.95, new_payee)   # scam payees are almost always new

    scam_type = np.full(m, "", dtype=object)
    age = cust["age_band"].to_numpy()
    for band, mix in s["type_mix"].items():
        sel = is_scam & (age == band)
        scam_type[sel] = rng.choice(list(mix), sel.sum(), p=list(mix.values()))

    p = cfg["payments"]
    amount = rng.lognormal(np.log(p["genuine_amount_median"]), p["genuine_amount_sigma"], m)
    for t in SCAM_TYPES:
        sel = scam_type == t
        amount[sel] = rng.lognormal(np.log(s["amount_median"][t]), s["amount_sigma"], sel.sum())
    amount = np.clip(amount, 1, 50000).round(2)

    # ── Existing risk score -> flagged payments are eligible for a warning ───
    r = cfg["risk_model"]
    logit = (-4 + r["scam_signal"] * is_scam + r["new_payee_signal"] * new_payee
             + r["amount_signal"] * np.log(amount / p["genuine_amount_median"]) + rng.normal(0, r["noise_sd"], m))
    risk_score = _sigmoid(logit)
    threshold = np.quantile(risk_score, 1 - r["flag_rate"])
    flagged = risk_score >= threshold

    # ── Potential outcomes IF warned (drawn for every flagged payment) ───────
    w = cfg["warning"]
    conf = cust["_digital_confidence"].to_numpy()
    base_stop = np.array([w["stop_rate"].get(t, 0.5) for t in scam_type])
    p_stop = np.where(is_scam, _sigmoid(_logit(np.clip(base_stop, 1e-6, 1 - 1e-6)) + w["confidence_effect"] * conf), 0.0)
    p_abandon = np.where(~is_scam, _sigmoid(_logit(w["genuine_abandon_rate"])
                                            + w["genuine_abandon_confidence_effect"] * conf), 0.0)
    stop_if_warned = flagged & (rng.random(m) < p_stop)
    abandon_if_warned = flagged & (rng.random(m) < p_abandon)
    sc = w["support_contact_rate"]
    p_contact = np.where(~is_scam, sc["genuine"], np.where(stop_if_warned, sc["scam_stopped"], sc["scam_completed"]))
    contact_if_warned = flagged & (rng.random(m) < p_contact)

    loss_if_not = np.where(is_scam, amount, 0.0)                      # no warning: every scam goes through
    loss_if_warned = np.where(is_scam & ~stop_if_warned, amount, 0.0)

    # ── What we actually observe depends on the arm ──────────────────────────
    warned = flagged & (cust["arm"].to_numpy() == "treatment")
    completed = ~(warned & (stop_if_warned | abandon_if_warned))
    ts = rng.integers(0, weeks * SECONDS_PER_WEEK, m)

    payments = pd.DataFrame({
        "payment_id": np.arange(1, m + 1),
        "customer_id": cust["customer_id"].to_numpy(),
        "arm": cust["arm"].to_numpy(),
        "seconds_since_start": ts,
        "amount": amount,
        "new_payee": new_payee,
        "risk_score": risk_score.round(5),
        "flagged": flagged,
        "warned": warned,
        "completed": completed,
        "abandoned_genuine": warned & abandon_if_warned,
        "support_contact": warned & contact_if_warned,
        # Fraud labels arrive later (customer report / bank investigation); observed after the test
        "is_scam": is_scam,
        "scam_type": scam_type,
        "scam_loss": np.where(warned, loss_if_warned, loss_if_not),
    }).sort_values(["seconds_since_start", "payment_id"]).reset_index(drop=True)

    truth = pd.DataFrame({
        "payment_id": np.arange(1, m + 1),
        "loss_if_warned": loss_if_warned,
        "loss_if_not_warned": loss_if_not,
        "abandon_if_warned": abandon_if_warned,
        "contact_if_warned": contact_if_warned,
    })
    return payments, truth, float(threshold)


def ground_truth(customers: pd.DataFrame, payments: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """The answers the analysis should recover (sample effects on this population)."""
    df = payments[["payment_id", "customer_id", "is_scam", "scam_type", "flagged"]].merge(truth, on="payment_id")
    df["loss_effect"] = df["loss_if_warned"] - df["loss_if_not_warned"]
    per_cust = df.groupby("customer_id").agg(
        loss_effect=("loss_effect", "sum"),
        loss_if_not=("loss_if_not_warned", "sum"),
        abandon=("abandon_if_warned", "sum"),
        contacts=("contact_if_warned", "sum"),
    ).reindex(customers["customer_id"], fill_value=0)
    per_cust = per_cust.join(customers.set_index("customer_id")[["age_band"]])

    scams_flagged = df[df.is_scam & df.flagged]
    genuine_flagged = df[~df.is_scam & df.flagged]
    by_type = {t: float(1 - g.loss_if_warned.sum() / g.loss_if_not_warned.sum())
               for t, g in scams_flagged.groupby("scam_type")}
    by_age = {a: float(g.loss_effect.mean()) for a, g in per_cust.groupby("age_band")}
    return {
        "note": "Sample average treatment effects if EVERY customer were warned vs none. Hidden from the analysis.",
        "ate_scam_loss_per_customer": float(per_cust.loss_effect.mean()),
        "relative_reduction_in_scam_loss": float(-per_cust.loss_effect.sum() / per_cust.loss_if_not.sum()),
        "ate_abandoned_genuine_per_customer": float(per_cust.abandon.mean()),
        "genuine_flagged_abandon_rate": float(genuine_flagged.abandon_if_warned.mean()),
        "ate_support_contacts_per_customer": float(per_cust.contacts.mean()),
        "share_of_scam_loss_stopped_by_type": by_type,
        "ate_scam_loss_per_customer_by_age": by_age,
    }


def summary(customers: pd.DataFrame, payments: pd.DataFrame, threshold: float, truth: dict) -> str:
    """Sanity checks a reviewer would ask for before trusting the simulation."""
    from sklearn.metrics import roc_auc_score

    scams = payments[payments.is_scam]
    bal = customers.groupby("arm")[["pre_payments", "pre_new_payees", "pre_spend", "tenure_months"]].mean()
    arms = customers["arm"].value_counts()
    lines = [
        "# Simulation summary",
        "",
        f"- Customers: **{len(customers):,}** (treatment {arms.get('treatment', 0):,} / control {arms.get('control', 0):,})",
        f"- Payments in the {payments.seconds_since_start.max() // 604800 + 1}-week test: **{len(payments):,}**",
        f"- Scam payments: **{len(scams):,}** ({len(scams) / len(payments):.3%} of payments), "
        f"£{scams.amount.sum():,.0f} attempted",
        "- Scam mix: " + ", ".join(f"{t} {n:,}" for t, n in scams.scam_type.value_counts().items()),
        f"- Existing risk score ROC-AUC: **{roc_auc_score(payments.is_scam, payments.risk_score):.3f}**; "
        f"flag threshold {threshold:.3f} flags {payments.flagged.mean():.1%} of payments and "
        f"**{scams.flagged.mean():.0%} of scams**",
        f"- Warnings shown (treatment only): {payments.warned.sum():,}",
        "",
        "## Arm balance on pre-period behaviour (should be near-identical)",
        "",
        "| Arm | Pre payments | Pre new payees | Pre spend (£) | Tenure (months) |",
        "|---|---|---|---|---|",
    ]
    for arm, r in bal.iterrows():
        lines.append(f"| {arm} | {r.pre_payments:.2f} | {r.pre_new_payees:.2f} | {r.pre_spend:,.0f} | {r.tenure_months:.1f} |")
    lines += [
        "",
        "## Ground truth (hidden from the analysis)",
        "",
        f"- Warning everyone would cut scam losses by **{truth['relative_reduction_in_scam_loss']:.1%}** "
        f"(£{-truth['ate_scam_loss_per_customer']:.2f} per customer over the test)",
        f"- Cost: {truth['genuine_flagged_abandon_rate']:.1%} of warned genuine payments abandoned; "
        f"{truth['ate_support_contacts_per_customer']:.4f} extra support contacts per customer",
        "- Share of flagged scam £ stopped, by type: " + ", ".join(
            f"{k} {v:.0%}" for k, v in truth["share_of_scam_loss_stopped_by_type"].items()),
    ]
    return "\n".join(lines)


def run(config_path=None, n=None, out_dir=None) -> dict:
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])
    customers = simulate_customers(cfg, rng, n)
    payments, truth_payments, threshold = simulate_payments(customers, cfg, rng)
    truth = ground_truth(customers, payments, truth_payments)

    data = Path(out_dir) if out_dir else resolve(cfg["output"]["data_dir"])
    truth_dir = data / "_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    observable = [c for c in customers.columns if not c.startswith("_")]
    customers[observable].to_parquet(data / "customers.parquet", index=False)
    payments.to_parquet(data / "payments.parquet", index=False)
    customers.to_parquet(truth_dir / "customers_latent.parquet", index=False)
    truth_payments.to_parquet(truth_dir / "potential_outcomes.parquet", index=False)
    (truth_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    report = summary(customers, payments, threshold, truth)
    reports = (data.parent / "reports") if out_dir else resolve(cfg["output"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "simulation_summary.md").write_text(report + "\n")
    print(report)
    return {"customers": customers, "payments": payments, "truth": truth, "threshold": threshold}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, help="number of customers (default from config)")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    run(a.config, a.n, a.out_dir)


if __name__ == "__main__":
    main()
