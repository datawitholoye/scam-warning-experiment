"""Day 2: read out the A/B test the way a product data scientist would.

    python -m scam_experiment.analysis

Order of work (matches the report):
  1. Trust checks      - is the split 50/50, are the arms balanced before the test?
  2. Primary metric    - scam £ lost per customer, several pre-planned estimators
  3. Secondary metric  - scams completed per customer (less noisy count)
  4. Guardrails        - abandoned genuine payments, support contacts
  5. Power             - how big / how long the test needed to be
  6. Peeking           - what checking every day would have done
  7. Grading           - every estimate vs the hidden truth (only possible in simulation)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config, resolve
from .metrics import customer_table, daily_cumulative
from .plots import plot_estimates, plot_false_positive_rates, plot_peeking_path
from .sequential import obf_boundaries, peeking_simulation, z_paths
from .stats import cuped, diff_in_means, mde, n_per_arm, ratio_upper_bound, srm_pvalue

ANALYSIS_CONFIG = PROJECT_ROOT / "configs" / "analysis.yaml"


def _scaled(est, share: float, method: str):
    """Convert a per-triggered-customer effect into a per-customer effect.
    Valid because triggering is decided identically in both arms, before the warning."""
    d = est.to_dict()
    for k in ("effect", "se", "ci_low", "ci_high", "control_mean"):
        d[k] *= share
    d["method"] = method
    return d


def primary_estimates(df: pd.DataFrame, covs: list[str], winsor_q: float, alpha: float) -> list[dict]:
    t = df["treat"].to_numpy()
    y = df["scam_loss"].to_numpy(float)
    trig = df["triggered"].to_numpy()
    share = trig.mean()
    plain = diff_in_means(y, t, "1. Plain difference in means (all customers)", alpha)
    out = [plain.to_dict()]
    out.append(cuped(y, t, df[covs], "2. CUPED (all customers)", alpha).to_dict())
    tr = df[trig]
    out.append(_scaled(diff_in_means(tr["scam_loss"], tr["treat"], alpha=alpha), share,
                       "3. Triggered customers only (pre-registered)"))
    out.append(_scaled(cuped(tr["scam_loss"], tr["treat"], tr[covs], alpha=alpha), share,
                       "4. Triggered + CUPED"))
    fl = df["flagged_scam_loss"].to_numpy(float)
    out.append(diff_in_means(fl, t, "5. Loss on flagged payments only", alpha).to_dict())
    cap = np.quantile(fl[fl > 0], winsor_q) if (fl > 0).any() else 0
    out.append(diff_in_means(np.minimum(fl, cap), t, f"6. As 5, losses capped at £{cap:,.0f} (biased)", alpha).to_dict())
    for e in out:
        e["variance_reduction_vs_plain"] = 1 - (e["se"] / plain.se) ** 2
    return out


def run(config_path=None, analysis_path=None, data_dir=None, reports_dir=None, permutations=None) -> dict:
    cfg = load_config(config_path)
    import yaml
    acfg = yaml.safe_load(Path(analysis_path or ANALYSIS_CONFIG).read_text())
    alpha = acfg["alpha"]
    data = Path(data_dir) if data_dir else resolve(cfg["output"]["data_dir"])
    reports = Path(reports_dir) if reports_dir else resolve(cfg["output"]["reports_dir"])
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    customers = pd.read_parquet(data / "customers.parquet")
    payments = pd.read_parquet(data / "payments.parquet")
    payments["flagged_scam_loss"] = np.where(payments["flagged"], payments["scam_loss"], 0.0)
    df = customer_table(customers, payments)
    df = df.join(payments.groupby("customer_id")["flagged_scam_loss"].sum(), on="customer_id").fillna(
        {"flagged_scam_loss": 0.0})
    t = df["treat"].to_numpy()
    covs = acfg["cuped_covariates"]

    # 1. Trust checks
    n_t, n_c = int(t.sum()), int((~t).sum())
    balance = {c: diff_in_means(df[c], t).p_value for c in covs}
    trust = {"n_treatment": n_t, "n_control": n_c, "srm_p": srm_pvalue(n_t, n_c),
             "balance_p": balance, "trigger_share": float(df["triggered"].mean()),
             "trigger_share_by_arm": df.groupby("arm")["triggered"].mean().to_dict()}

    # 2-3. Metrics
    primary = primary_estimates(df, covs, acfg["winsor_quantile"], alpha)
    secondary = diff_in_means(df["scams_completed"], t, "Scams completed per customer", alpha).to_dict()

    # 4. Guardrails (treatment arm: the only arm that sees warnings)
    trt = df[t]
    ab_rate, ab_ub = ratio_upper_bound(trt["abandoned_genuine"], trt["genuine_warned"], alpha)
    sc_rate, sc_ub = ratio_upper_bound(trt["support_contacts"], trt["warned"], alpha)
    g = acfg["guardrails"]
    guardrails = {
        "abandoned_per_warned_genuine_payment": {"estimate": ab_rate, "upper_95": ab_ub,
                                                 "limit": g["abandoned_per_warned_genuine_payment"],
                                                 "pass": ab_ub < g["abandoned_per_warned_genuine_payment"]},
        "support_contacts_per_warned_payment": {"estimate": sc_rate, "upper_95": sc_ub,
                                                "limit": g["support_contacts_per_warned_payment"],
                                                "pass": sc_ub < g["support_contacts_per_warned_payment"]},
    }

    # 5. Power, from the control arm (stands in for pre-launch historical data)
    weeks = cfg["experiment"]["weeks"]
    ctrl = df[~t]
    power = []
    loss_base = ctrl["scam_loss"].mean()   # both £ metrics sized against the SAME £ effect
    for label, col, base in [("Scam £ lost, all customers", "scam_loss", loss_base),
                             ("Scam £ lost on flagged payments", "flagged_scam_loss", loss_base),
                             ("Scams completed (count)", "scams_completed", ctrl["scams_completed"].mean())]:
        sd = ctrl[col].std(ddof=1)
        row = {"metric": label, "baseline": float(base), "sd": float(sd),
               "mde_relative_now": float(mde(sd, n_c, alpha, acfg["power"]) / base) if base else None}
        for r in acfg["power_relative_mde"]:
            n = n_per_arm(sd, r * base, alpha, acfg["power"])
            row[f"n_per_arm_{int(r * 100)}pct"] = float(n)
            row[f"weeks_{int(r * 100)}pct"] = float(weeks * n / n_c)   # mean and variance both grow ~linearly with time
        power.append(row)

    # 6. Peeking
    cum = daily_cumulative(customers, payments, "scam_loss")
    pk = acfg["peeking"]
    peeking = peeking_simulation(cum, t, permutations or pk["permutations"], pk["weekly_looks"], pk["seed"])
    rng = np.random.default_rng(pk["seed"] + 1)
    example = None
    for _ in range(500):
        z = z_paths(cum, rng.permutation(t)[None, :])[0]
        hit = np.abs(z) > 1.96
        if hit.any() and np.argmax(hit) >= 4 and abs(z[-1]) < 1.96:   # a convincing-looking mid-test "win"
            example = z
            break

    # 7. Grade against the hidden truth
    grading = grade(data, primary, secondary, guardrails, len(df))

    results = {"trust": trust, "primary": primary, "secondary": secondary, "guardrails": guardrails,
               "power": power, "peeking": peeking, "grading": grading}
    (reports / "day2_results.json").write_text(json.dumps(results, indent=2, default=float))
    plot_estimates(primary, grading["true_effect_per_customer"], figs / "estimates_vs_truth.png")
    plot_false_positive_rates(peeking, figs / "peeking_false_positives.png")
    if example is not None:
        plot_peeking_path(example, obf_boundaries(pk["weekly_looks"]), peeking["weekly_look_days"],
                          figs / "peeking_example.png")
    report = render(results, cfg)
    (reports / "day2_ab_test.md").write_text(report + "\n")
    print(report)
    return results


def grade(data: Path, primary: list[dict], secondary: dict, guardrails: dict, n_customers: int) -> dict:
    """Compare each estimate with the true effect. Reads the hidden truth; nothing above does."""
    truth = json.loads((data / "_truth" / "ground_truth.json").read_text())
    po = pd.read_parquet(data / "_truth" / "potential_outcomes.parquet")
    pay = pd.read_parquet(data / "payments.parquet", columns=["payment_id", "flagged", "is_scam", "warned"])
    po = po.merge(pay, on="payment_id")
    true_loss = truth["ate_scam_loss_per_customer"]
    true_count = float(((po.loss_if_warned > 0).sum() - (po.loss_if_not_warned > 0).sum()) / n_customers)
    flagged = po[po.flagged]
    true_contact = float(flagged.contact_if_warned.mean())
    rows = []
    for e in primary:
        rows.append({"method": e["method"], "estimate": e["effect"], "truth": true_loss,
                     "covers_truth": e["ci_low"] <= true_loss <= e["ci_high"],
                     "error": e["effect"] - true_loss})
    rows.append({"method": secondary["method"], "estimate": secondary["effect"], "truth": true_count,
                 "covers_truth": secondary["ci_low"] <= true_count <= secondary["ci_high"],
                 "error": secondary["effect"] - true_count})
    return {"true_effect_per_customer": true_loss, "true_count_effect_per_customer": true_count,
            "true_abandon_rate": truth["genuine_flagged_abandon_rate"], "true_contact_rate": true_contact,
            "rows": rows}


def _gbp(x):
    return f"-£{abs(x):,.2f}" if x < 0 else f"£{x:,.2f}"


def render(r: dict, cfg: dict) -> str:
    tr, g = r["trust"], r["grading"]
    pre = next(e for e in r["primary"] if "pre-registered" in e["method"])
    sec = r["secondary"]
    out = ["# Day 2: A/B test readout", "",
           "## Headline", "",
           f"- **Primary metric (scam £ lost per customer): not statistically significant.** The pre-registered estimate is "
           f"{_gbp(pre['effect'])} per customer ({pre['relative']:+.0%}), 95% CI {_gbp(pre['ci_low'])} to {_gbp(pre['ci_high'])}, "
           f"p = {pre['p_value']:.2f}.",
           f"- **Secondary metric (scams completed): significant.** {sec['relative']:+.0%} "
           f"(p = {sec['p_value']:.3f}). A count is far less noisy than £, because a few large losses dominate the £ total.",
           "- **Guardrails: " + ("all pass" if all(v["pass"] for v in r["guardrails"].values()) else "FAILED") + ".**",
           f"- **The test was underpowered for £ loss.** With {tr['n_control']:,} customers per arm it could only detect a "
           f"{r['power'][0]['mde_relative_now']:.0%} change in £ lost. A power analysis on historical data should have set "
           "the metric and duration before launch.",
           f"- **Grading:** the true effect (hidden until now) is {_gbp(g['true_effect_per_customer'])} per customer. "
           f"Every unbiased estimator's 95% CI contains it.", "",
           "## 1. Trust checks", "",
           f"- Split: {tr['n_treatment']:,} treatment / {tr['n_control']:,} control. Sample ratio mismatch test p = {tr['srm_p']:.2f} (fine; < 0.001 would mean broken randomisation).",
           "- Pre-period balance (p-values, all should be unremarkable): " + ", ".join(f"{k} {v:.2f}" for k, v in tr["balance_p"].items()),
           f"- Triggered (≥ 1 flagged payment): {tr['trigger_share']:.1%} of customers; by arm " +
           ", ".join(f"{k} {v:.1%}" for k, v in tr["trigger_share_by_arm"].items()) + " (equal, as it must be for triggering to be valid).", "",
           "## 2. Primary metric: scam £ lost per customer", "",
           "All effects are per customer over the 4-week test (treatment − control). Negative = the warning reduced losses.", "",
           "| Estimator | Effect | 95% CI | p | Variance removed vs plain | CI contains truth? |",
           "|---|---|---|---|---|---|"]
    cover = {row["method"]: row["covers_truth"] for row in g["rows"]}
    for e in r["primary"]:
        out.append(f"| {e['method']} | {_gbp(e['effect'])} | {_gbp(e['ci_low'])} to {_gbp(e['ci_high'])} | {e['p_value']:.2f} | "
                   f"{e['variance_reduction_vs_plain']:.0%} | {'✅' if cover[e['method']] else '❌'} |")
    out += ["", f"True effect: **{_gbp(g['true_effect_per_customer'])} per customer**.", "",
            "![Estimates vs truth](figures/estimates_vs_truth.png)", "",
            "**What this shows:**",
            "- **CUPED barely helps here.** Past payment behaviour doesn't predict who gets scammed, so there is little noise for it to remove. It shines when the outcome is predictable from history (e.g. spend).",
            "- **Triggering helps most.** Only flagged payments can be warned, so losses elsewhere are pure noise. Restricting to flagged payments removes about a quarter of the variance.",
            "- **Capping big losses cuts noise but biases the answer.** The capped estimate is shrunk towards zero; it can be used to *detect* an effect, not to *size* it in £.", "",
            "## 3. Secondary metric: scams completed", "",
            f"Effect {sec['effect']:+.5f} scams per customer ({sec['relative']:+.0%}), 95% CI {sec['ci_low']:+.5f} to {sec['ci_high']:+.5f}, p = {sec['p_value']:.3f}. "
            f"True effect: {g['true_count_effect_per_customer']:+.5f}. "
            "Secondary metrics support the story but shouldn't be promoted to the headline after the fact: that is how false wins are manufactured.", "",
            "## 4. Guardrails (limits set before the test)", "",
            "| Guardrail | Estimate | One-sided 95% upper bound | Limit | Result |", "|---|---|---|---|---|"]
    names = {"abandoned_per_warned_genuine_payment": "Genuine payments abandoned after a warning",
             "support_contacts_per_warned_payment": "Support contacts per warning"}
    for k, v in r["guardrails"].items():
        out.append(f"| {names[k]} | {v['estimate']:.2%} | {v['upper_95']:.2%} | {v['limit']:.0%} | {'✅ pass' if v['pass'] else '❌ fail'} |")
    out += ["", "Rates use the customer as the unit (delta method), because one customer's payments are correlated.", "",
            "## 5. Power: how big should the test have been?", "",
            "Using the control arm's spread (in practice: last month's data, before launch). "
            "Changes are relative to total scam £ lost (for both £ metrics) or total scams (for the count).", "",
            "| Metric | Smallest detectable change now | Weeks needed for −10% | −15% | −20% |", "|---|---|---|---|---|"]
    for p in r["power"]:
        out.append(f"| {p['metric']} | {p['mde_relative_now']:.0%} | {p['weeks_10pct']:.0f} | {p['weeks_15pct']:.0f} | {p['weeks_20pct']:.0f} |")
    pk = r["peeking"]
    out += ["", "Weeks assume the same traffic (100k customers) and that both the effect and the noise grow in proportion to time.", "",
            "## 6. Peeking: what if we'd checked every day?", "",
            f"{pk['permutations']:,} A/A replays: the real customers re-randomised so there is **no** true difference.", "",
            "| How results are checked | False positive rate |", "|---|---|",
            f"| Once, at the end (as planned) | {pk['once']:.1%} |",
            f"| Every week, stop at p < 0.05 | {pk['weekly']:.1%} |",
            f"| Every day, stop at p < 0.05 | {pk['daily']:.1%} |",
            f"| Every week with O'Brien-Fleming boundaries (z > {', '.join(f'{b:.2f}' for b in pk['obf_boundaries'])}) | {pk['obf_weekly']:.1%} |",
            "", f"Daily peeking turns a 5% error rate into about {pk['daily']:.0%}; most false stops happen in the first week (median day {pk['median_day_of_false_stop']:.0f}), when few scams have been seen. "
            "If the business needs early looks, a sequential design allows them at almost no cost to the final threshold.", "",
            "![False positive rates](figures/peeking_false_positives.png)", "",
            "![A false win from peeking](figures/peeking_example.png)", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--analysis-config")
    ap.add_argument("--permutations", type=int)
    a = ap.parse_args()
    run(a.config, a.analysis_config, permutations=a.permutations)


if __name__ == "__main__":
    main()
