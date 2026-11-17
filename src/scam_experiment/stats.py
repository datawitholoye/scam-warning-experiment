"""Small, transparent statistics toolkit for experiment readouts.

Each function is a few lines of numpy so every number in the report can be
traced back to a formula, rather than to a black-box library call.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    method: str
    effect: float          # treatment - control
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_treatment: int
    n_control: int
    control_mean: float

    @property
    def relative(self) -> float:
        return self.effect / self.control_mean if self.control_mean else float("nan")

    def to_dict(self) -> dict:
        return {**asdict(self), "relative": self.relative}


def _finish(method, effect, se, n_t, n_c, control_mean, alpha=0.05) -> Estimate:
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(effect / se)) if se > 0 else float("nan")
    return Estimate(method, float(effect), float(se), float(effect - z * se), float(effect + z * se),
                    float(p), int(n_t), int(n_c), float(control_mean))


def diff_in_means(y, treat, method="Difference in means", alpha=0.05) -> Estimate:
    """Welch (unequal-variance) difference in means."""
    y, treat = np.asarray(y, float), np.asarray(treat, bool)
    yt, yc = y[treat], y[~treat]
    se = math.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return _finish(method, yt.mean() - yc.mean(), se, len(yt), len(yc), yc.mean(), alpha)


def cuped(y, treat, X, method="CUPED", alpha=0.05) -> Estimate:
    """Regression adjustment on pre-period covariates (multi-covariate CUPED).

    Fits y ~ 1 + treat + (X - mean(X)) by OLS with heteroskedasticity-robust (HC1)
    standard errors. Because X was measured before the test, it is independent
    of the arm, so the adjustment removes noise without introducing bias.
    """
    y, treat = np.asarray(y, float), np.asarray(treat, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    Xc = (X - X.mean(axis=0)) / X.std(axis=0).clip(1e-12)
    D = np.column_stack([np.ones_like(y), treat, Xc])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    n, k = D.shape
    bread = np.linalg.inv(D.T @ D)
    meat = (D * resid[:, None] ** 2).T @ D
    cov = bread @ meat @ bread * n / (n - k)
    control_mean = y[treat == 0].mean()
    return _finish(method, beta[1], math.sqrt(cov[1, 1]), (treat == 1).sum(), (treat == 0).sum(), control_mean, alpha)


def variance_reduction(base: Estimate, other: Estimate) -> float:
    """Share of variance removed relative to a baseline estimate (0.3 = 30% less variance)."""
    return 1 - (other.se / base.se) ** 2


def ratio_upper_bound(numerator, denominator, alpha=0.05) -> tuple[float, float]:
    """Ratio metric (e.g. abandoned / warned payments) with customers as the unit.

    Payments from the same customer are correlated, so a naive binomial SE is too
    small. The delta method linearises the ratio per customer instead.
    Returns (ratio, one-sided upper bound).
    """
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    keep = b > 0
    a, b = a[keep], b[keep]
    r = a.sum() / b.sum()
    n = len(a)
    se = math.sqrt(np.var(a - r * b, ddof=1) / n) / b.mean()
    return float(r), float(r + stats.norm.ppf(1 - alpha) * se)


def srm_pvalue(n_treatment: int, n_control: int, expected_share: float = 0.5) -> float:
    """Sample ratio mismatch: chi-square test that the split matches the design.
    A tiny p-value (< 0.001) means the randomisation or logging is broken."""
    total = n_treatment + n_control
    exp = [total * expected_share, total * (1 - expected_share)]
    return float(stats.chisquare([n_treatment, n_control], exp).pvalue)


def n_per_arm(sd: float, delta: float, alpha=0.05, power=0.8) -> float:
    """Customers per arm to detect an absolute difference `delta` in a mean."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return 2 * (z * sd / delta) ** 2 if delta else math.inf


def mde(sd: float, n_each: float, alpha=0.05, power=0.8) -> float:
    """Smallest absolute effect detectable with n customers per arm."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * sd * math.sqrt(2 / n_each)
