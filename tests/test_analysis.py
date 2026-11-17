import json

import numpy as np
import pytest
from scipy import stats as sps

from scam_experiment.analysis import run as run_analysis
from scam_experiment.sequential import obf_boundaries, peeking_simulation
from scam_experiment.simulate import run as run_sim
from scam_experiment.stats import cuped, diff_in_means, mde, n_per_arm, ratio_upper_bound, srm_pvalue


def test_diff_in_means_matches_welch_t_test():
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(1.0, 2, 4000), rng.normal(0.8, 1, 4000)]
    t = np.r_[np.ones(4000, bool), np.zeros(4000, bool)]
    e = diff_in_means(y, t)
    ref = sps.ttest_ind(y[t], y[~t], equal_var=False)
    assert e.effect == pytest.approx(y[t].mean() - y[~t].mean())
    assert e.p_value == pytest.approx(ref.pvalue, rel=0.02)


def test_cuped_removes_noise_without_bias():
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.normal(0, 1, n)                       # pre-period covariate
    t = rng.random(n) < 0.5
    y = 3 * x + 0.5 * t + rng.normal(0, 1, n)     # outcome mostly predicted by x
    plain, adj = diff_in_means(y, t), cuped(y, t, x)
    assert adj.se < plain.se * 0.5
    assert adj.ci_low < 0.5 < adj.ci_high


def test_ratio_upper_bound_accounts_for_clustering():
    rng = np.random.default_rng(2)
    n = 5000
    warned = rng.integers(1, 20, n)
    p_i = rng.beta(1, 20, n)                      # some customers abandon far more than others
    abandoned = rng.binomial(warned, p_i)
    r, ub = ratio_upper_bound(abandoned, warned)
    naive_ub = r + 1.645 * np.sqrt(r * (1 - r) / warned.sum())
    assert r == pytest.approx(abandoned.sum() / warned.sum())
    assert ub > naive_ub                          # clustered SE is wider than the naive binomial one


def test_srm_detects_broken_split():
    assert srm_pvalue(5000, 5050) > 0.05
    assert srm_pvalue(5000, 5600) < 0.001


def test_power_and_mde_are_inverse():
    n = n_per_arm(sd=10, delta=1)
    assert mde(10, n) == pytest.approx(1, rel=1e-6)


def test_obf_boundaries_shape():
    b = obf_boundaries(4)
    assert b[-1] == pytest.approx(2.024)
    assert (np.diff(b) < 0).all()                 # strict early, relaxed at the end


def test_peeking_inflates_false_positives_and_obf_controls_them():
    rng = np.random.default_rng(3)
    n, days = 4000, 28
    cum = np.cumsum(rng.exponential(1, (n, days)), axis=1)  # no treatment effect anywhere
    treat = np.r_[np.ones(n // 2, bool), np.zeros(n // 2, bool)]
    res = peeking_simulation(cum, treat, permutations=600, seed=0)
    assert res["daily"] > 2 * res["once"]
    assert res["obf_weekly"] < 0.09
    assert 0.02 < res["once"] < 0.09


def test_full_analysis_runs_and_grades(tmp_path):
    data = tmp_path / "data"
    run_sim(n=15000, out_dir=data)
    res = run_analysis(data_dir=data, reports_dir=tmp_path / "reports", permutations=200)
    assert (tmp_path / "reports" / "day2_ab_test.md").exists()
    assert (tmp_path / "reports" / "figures" / "estimates_vs_truth.png").exists()
    assert res["trust"]["srm_p"] > 0.001
    assert all(v["pass"] for v in res["guardrails"].values())
    saved = json.loads((tmp_path / "reports" / "day2_results.json").read_text())
    assert len(saved["primary"]) == 6
