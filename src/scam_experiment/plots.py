"""Report figures (static PNGs for GitHub)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#d9622b", "#9a9893"


def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_estimates(primary: list[dict], truth: float, path: Path) -> None:
    """Each estimator's effect and 95% CI against the true effect."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    _style(ax, "x")
    labels = [e["method"] for e in primary][::-1]
    for i, e in enumerate(primary[::-1]):
        biased = "biased" in e["method"]
        c = GREY if biased else BLUE
        ax.plot([e["ci_low"], e["ci_high"]], [i, i], color=c, linewidth=2)
        ax.scatter([e["effect"]], [i], s=40, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axvline(truth, color=ORANGE, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(truth, len(primary) - 0.4, f" true effect {truth:+.2f}", color=ORANGE, fontsize=9, va="bottom")
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5, color=INK)
    ax.set_ylim(-0.6, len(primary) - 0.1)
    ax.set_xlabel("Effect on scam £ lost per customer (treatment − control), 95% CI", color=MUTED, fontsize=9)
    ax.set_title("Every unbiased estimate covers the truth; none is significant", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_false_positive_rates(pk: dict, path: Path) -> None:
    rows = [("Check once\n(as planned)", pk["once"]), ("Check weekly,\np < 0.05", pk["weekly"]),
            ("Check daily,\np < 0.05", pk["daily"]), ("Weekly with\nO'Brien-Fleming", pk["obf_weekly"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    _style(ax)
    colors = [BLUE, ORANGE, ORANGE, BLUE]
    bars = ax.bar([r[0] for r in rows], [100 * r[1] for r in rows], color=colors, width=0.55)
    for b, (_, v) in zip(bars, rows, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, 100 * v + 0.6, f"{100 * v:.0f}%", ha="center", color=INK, fontsize=10)
    ax.axhline(5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(3.45, 5.4, "5% target", color=MUTED, fontsize=8.5, ha="right")
    ax.set_ylabel("False positive rate (%)", color=MUTED, fontsize=9)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.set_title(f"A/A tests with no real effect ({pk['permutations']:,} replays): how often a 'win' is declared",
                 loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_peeking_path(z: np.ndarray, obf: np.ndarray, look_days: list[int], path: Path) -> None:
    """One A/A replay whose daily z-statistic crosses 1.96 but ends insignificant."""
    days = np.arange(1, len(z) + 1)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    _style(ax)
    ax.axhspan(-1.96, 1.96, color="#f3f2ee")
    cross = np.where(np.abs(z) > 1.96)[0]
    sign = np.sign(z[cross[0]]) if len(cross) else 1
    # label the "not significant" band on the side away from the crossing
    ax.text(1, -sign * 1.85, " not significant (|z| < 1.96)", color=MUTED, fontsize=8,
            va="bottom" if sign > 0 else "top")
    ax.plot(days, z, color=BLUE, linewidth=2)
    if len(cross):
        d = cross[0]
        ax.scatter([days[d]], [z[d]], s=50, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"day {days[d]}: 'significant!' (p < 0.05)", (days[d], z[d]),
                    xytext=(-20, 30 if sign > 0 else -34), textcoords="offset points",
                    color=INK, fontsize=9, ha="right",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.scatter(look_days, sign * obf, marker="_", s=300, color=MUTED, linewidth=2,
               label="O'Brien-Fleming weekly boundary")
    ax.set_ylim(-4.6, 4.6)
    ax.set_xlabel("Day of test", color=MUTED, fontsize=9)
    ax.set_ylabel("z-statistic", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right" if sign < 0 else "lower right")
    ax.set_title("An A/A test with no real effect: daily peeking would have stopped early",
                 loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
