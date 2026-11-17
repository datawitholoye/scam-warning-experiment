"""What happens if you check the results every day?

An A/A test (no real difference between arms) is replayed thousands of times by
re-randomising the real customers. Each replay is "checked" at several points
during the 4 weeks. Stopping the first time p < 0.05 inflates the false
positive rate far above 5%; an O'Brien-Fleming boundary keeps it at 5%.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Classic O'Brien-Fleming constant for K equally spaced looks, two-sided alpha = 0.05
OBF_CONSTANT = {1: 1.960, 2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040}


def obf_boundaries(k_looks: int) -> np.ndarray:
    """z boundary at look k of K: c * sqrt(K / k). Strict early, close to 1.96 at the end."""
    c = OBF_CONSTANT[k_looks]
    k = np.arange(1, k_looks + 1)
    return c * np.sqrt(k_looks / k)


def z_paths(cum: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """z-statistic of treatment - control at every day, for each assignment vector.

    cum: customers x days cumulative metric. assignments: B x customers (bool).
    Returns B x days.
    """
    A = assignments.astype(float)
    n_t = A.sum(axis=1, keepdims=True)
    n_c = A.shape[1] - n_t
    s1, s2 = cum.sum(axis=0), (cum ** 2).sum(axis=0)
    t1, t2 = A @ cum, A @ (cum ** 2)
    c1, c2 = s1 - t1, s2 - t2
    mt, mc = t1 / n_t, c1 / n_c
    vt = (t2 - n_t * mt ** 2) / (n_t - 1)
    vc = (c2 - n_c * mc ** 2) / (n_c - 1)
    se = np.sqrt(vt / n_t + vc / n_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mt - mc) / se
    return np.nan_to_num(z)


def peeking_simulation(cum: np.ndarray, treat: np.ndarray, permutations: int = 2000,
                       weekly_looks: int = 4, seed: int = 0, batch: int = 100) -> dict:
    """False positive rate of different checking strategies under a true null."""
    rng = np.random.default_rng(seed)
    days = cum.shape[1]
    week_days = np.linspace(days / weekly_looks, days, weekly_looks).round().astype(int) - 1
    z_crit = stats.norm.ppf(0.975)
    obf = obf_boundaries(weekly_looks)
    hits = {"once": 0, "weekly": 0, "daily": 0, "obf_weekly": 0}
    first_daily_stop = []
    done = 0
    while done < permutations:
        b = min(batch, permutations - done)
        assign = np.array([rng.permutation(treat) for _ in range(b)])
        z = np.abs(z_paths(cum, assign))
        hits["once"] += (z[:, -1] > z_crit).sum()
        hits["weekly"] += (z[:, week_days] > z_crit).any(axis=1).sum()
        daily_hit = z > z_crit
        hits["daily"] += daily_hit.any(axis=1).sum()
        hits["obf_weekly"] += (z[:, week_days] > obf).any(axis=1).sum()
        first_daily_stop += [int(np.argmax(r)) + 1 for r in daily_hit if r.any()]
        done += b
    out = {k: v / permutations for k, v in hits.items()}
    out["permutations"] = permutations
    out["weekly_look_days"] = (week_days + 1).tolist()
    out["obf_boundaries"] = obf.round(3).tolist()
    out["median_day_of_false_stop"] = float(np.median(first_daily_stop)) if first_daily_stop else None
    return out
