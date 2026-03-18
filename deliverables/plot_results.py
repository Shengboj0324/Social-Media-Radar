"""
plot_results.py — Generate performance plots from benchmark CSV files.

Reads deliverables/results/*.csv and writes four PNG plots to
deliverables/plots/.  Each plot shows:
  • Measured data points (mean ± 1 std error bar)
  • Best-fit curve (from scipy curve_fit)
  • Theoretical complexity label and R² value

Usage:
    python deliverables/plot_results.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RESULTS = Path(__file__).parent / "results"
PLOTS   = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

# ── curve models ──────────────────────────────────────────────────────────────
def _constant(n, a):  return a * np.ones_like(n, dtype=float)
def _linear(n, a):    return a * n
def _log(n, a):       return a * np.log(n)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def load_csv(name):
    rows = []
    with open(RESULTS / name) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def fit(ns, ts, model_fn):
    popt, _ = curve_fit(model_fn, ns, ts, p0=[1e-4], maxfev=10_000)
    pred = model_fn(ns, *popt)
    return popt[0], r_squared(ts, pred), pred

# ── generic plot helper ───────────────────────────────────────────────────────
def make_plot(rows, xcol, ycol, ecol,
              model_fn, theory_label, fit_label,
              title, xlabel, ylabel, filename,
              extra_series=None):
    ns = np.array([r[xcol] for r in rows])
    ts = np.array([r[ycol] for r in rows])
    es = np.array([r[ecol] for r in rows])

    a, r2, pred = fit(ns, ts, model_fn)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ns, ts, yerr=es, fmt="o", color="#2563EB",
                capsize=4, label="Measured (mean ± 1σ)", zorder=3)

    ns_fine = np.linspace(ns.min(), ns.max(), 500)
    ax.plot(ns_fine, model_fn(ns_fine, a), "--", color="#DC2626",
            linewidth=1.8, label=f"Fit: {fit_label}  (R²={r2:.4f})")

    if extra_series:
        for label, xs, ys, style in extra_series:
            ax.plot(xs, ys, style, label=label, linewidth=1.4)

    ax.set_title(f"{title}\n(Theoretical: {theory_label})", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = PLOTS / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}  (R²={r2:.6f})")
    return a, r2

# ── Plot 1: BloomFilter total time  O(n·k) ──────────────────────────────────
def plot_bloom():
    rows = load_csv("bloom.csv")
    # Derive per-operation constant line: each operation costs ~12.9µs
    ns = np.array([r["n"] for r in rows])
    ts = np.array([r["mean_ms"] for r in rows])
    per_op_us = (ts / ns) * 1000          # µs per item

    # Insert per-op as annotation series
    ax2_series = []  # added inside make_plot wouldn't access ax2 — handle separately

    a, r2 = make_plot(
        rows, "n", "mean_ms", "std_ms",
        _linear,
        "O(k) per-op = O(1),  O(n·k) total",
        "a·n",
        "BloomFilter: Total time for n insert+lookup ops",
        "n (items)", "Wall-clock time (ms)", "bloom.png",
    )
    # Annotate per-op constancy with a second figure panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(ns, ts, fmt="o", color="#2563EB", capsize=4,
                     label="Total time")
    xs_fine = np.linspace(ns.min(), ns.max(), 300)
    axes[0].plot(xs_fine, a * xs_fine, "--r", label=f"a·n (R²={r2:.4f})")
    axes[0].set_title("Total time vs n"); axes[0].set_xlabel("n"); axes[0].set_ylabel("ms")
    axes[0].legend(fontsize=8); axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(ns, per_op_us, "s-", color="#059669", label="Per-op time (µs)")
    axes[1].axhline(np.mean(per_op_us), linestyle="--", color="#DC2626",
                    label=f"Mean = {np.mean(per_op_us):.2f} µs")
    axes[1].set_title("Per-operation time (constant → O(1))")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("µs per item")
    axes[1].legend(fontsize=8); axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("BloomFilter Complexity Analysis", fontsize=12)
    fig.tight_layout()
    path = PLOTS / "bloom_dual.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved → {path}")

# ── Plot 2: ReservoirSampler O(n) ────────────────────────────────────────────
def plot_reservoir():
    make_plot(load_csv("reservoir.csv"), "n","mean_ms","std_ms",
              _linear, "O(n)",  "a·n",
              "ReservoirSampler: Stream of n items (Algorithm R)",
              "n (stream size)", "Wall-clock time (ms)", "reservoir.png")

# ── Plot 3: ConfidenceCalibrator O(m) ────────────────────────────────────────
def plot_calibrator():
    make_plot(load_csv("calibrator.csv"), "m","mean_ms","std_ms",
              _linear, "O(m)",  "a·m",
              "ConfidenceCalibrator: m gradient updates",
              "m (training examples)", "Wall-clock time (ms)", "calibrator.png")

# ── Plot 4: BFS O(V+E) = O(n) ────────────────────────────────────────────────
def plot_bfs():
    make_plot(load_csv("bfs.csv"), "n","mean_ms","std_ms",
              _linear, "O(V+E) = O(n)",  "a·n",
              "BFS: Degree-4 ring graph with n nodes",
              "n (nodes)", "Wall-clock time (ms)", "bfs.png")

def main():
    print("=" * 52)
    print("  Generating performance plots …")
    print("=" * 52)
    plot_bloom()
    plot_reservoir()
    plot_calibrator()
    plot_bfs()
    print("\nAll plots written to deliverables/plots/")

if __name__ == "__main__":
    main()

