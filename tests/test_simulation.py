import json

import numpy as np
import pandas as pd
import pytest

from scam_experiment.assign import assign, bucket
from scam_experiment.simulate import run


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim") / "data"
    res = run(n=20000, out_dir=out)
    res["out"] = out
    return res


def test_assignment_is_deterministic_and_salted():
    ids = [f"C{i:06d}" for i in range(2000)]
    a1, a2 = assign(ids, "s1"), assign(ids, "s1")
    assert (a1 == a2).all()                               # same customer, same arm, every time
    assert (assign(ids, "s2") != a1).mean() > 0.3         # new salt -> independent split
    assert 0 <= bucket("C000001", "s1") < 1


def test_split_is_close_to_50_50(sim):
    share = (sim["customers"].arm == "treatment").mean()
    assert abs(share - 0.5) < 0.02


def test_arms_balanced_on_pre_period(sim):
    m = sim["customers"].groupby("arm")[["pre_payments", "pre_spend"]].mean()
    rel = (m.loc["treatment"] - m.loc["control"]).abs() / m.mean()
    assert (rel < 0.05).all()


def test_only_treatment_flagged_payments_are_warned(sim):
    p = sim["payments"]
    assert not p[p.arm == "control"].warned.any()
    assert (p[p.arm == "treatment"].warned == p[p.arm == "treatment"].flagged).all()
    assert not p[~p.warned].abandoned_genuine.any()
    assert p[~p.warned].completed.all()


def test_observed_outcomes_match_potential_outcomes(sim):
    p = sim["payments"]
    po = pd.read_parquet(sim["out"] / "_truth" / "potential_outcomes.parquet")
    df = p.merge(po, on="payment_id")
    expected = np.where(df.warned, df.loss_if_warned, df.loss_if_not_warned)
    assert np.allclose(df.scam_loss, expected)
    assert (df.loss_if_warned <= df.loss_if_not_warned).all()   # a warning never increases a loss


def test_genuine_payments_never_lose_money(sim):
    p = sim["payments"]
    assert (p.loc[~p.is_scam, "scam_loss"] == 0).all()


def test_truth_is_hidden_from_observable_files(sim):
    cust = pd.read_parquet(sim["out"] / "customers.parquet")
    assert not [c for c in cust.columns if c.startswith("_")]
    truth = json.loads((sim["out"] / "_truth" / "ground_truth.json").read_text())
    assert truth["ate_scam_loss_per_customer"] < 0            # warnings reduce loss
    assert 0 < truth["relative_reduction_in_scam_loss"] < 1


def test_reproducible_with_seed(tmp_path):
    a = run(n=3000, out_dir=tmp_path / "a" / "data")["payments"]
    b = run(n=3000, out_dir=tmp_path / "b" / "data")["payments"]
    pd.testing.assert_frame_equal(a, b)
