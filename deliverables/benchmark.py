"""
benchmark.py — Performance benchmark for Social-Media-Radar core algorithms.

Measures four algorithms across eight problem sizes each:
  1. BloomFilter  (insert + lookup)          — theoretical O(k) ≈ O(1)
  2. ReservoirSampler (Algorithm R)          — theoretical O(n)
  3. ConfidenceCalibrator.update() batch     — theoretical O(m)
  4. BFS on synthetic degree-4 graph         — theoretical O(V+E) = O(n)

Each algorithm is run with 3 warm-up passes (discarded) and 7 timed reps.
Mean and std in milliseconds are reported and CSV files are written to
deliverables/results/.  Curve-fitting (scipy) + R² is computed for each.

Usage:
    python deliverables/benchmark.py
"""

import csv
import sys
import time
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scraping.probabilistic_structures import BloomFilter
from app.scraping.reservoir_sampling import ReservoirSampler
from app.intelligence.calibration import ConfidenceCalibrator
from app.domain.inference_models import SignalType

# ── curve models ──────────────────────────────────────────────────────────────
def _constant(n, a):  return a * np.ones_like(n, dtype=float)
def _linear(n, a):    return a * n
def _log(n, a):       return a * np.log(n)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

# ── timing helper ─────────────────────────────────────────────────────────────
WARMUP, REPS = 3, 7

def _time(fn):
    for _ in range(WARMUP): fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))

# ── Benchmark 1: BloomFilter  O(k) = O(1) ────────────────────────────────────
def bench_bloom():
    sizes = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    rows = []
    print("\n=== BloomFilter add+contains — O(k) = O(1) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        items = [f"https://reddit.com/r/saas/post_{i}" for i in range(n)]
        def fn(items=items, n=n):
            bf = BloomFilter(expected_elements=n, false_positive_rate=0.01)
            for item in items: bf.add(item)
            for item in items: bf.contains(item)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows

# ── Benchmark 2: ReservoirSampler  O(n) ──────────────────────────────────────
def bench_reservoir():
    sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
    rows = []
    print("\n=== ReservoirSampler (Algorithm R) — O(n) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        items = list(range(n))
        def fn(items=items):
            rs = ReservoirSampler(reservoir_size=500, random_seed=42)
            for item in items: rs.add(item)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows

# ── Benchmark 3: ConfidenceCalibrator  O(m) ──────────────────────────────────
# _save() is patched to a no-op so we measure pure math, not disk I/O.
# (The report discloses this and separately reports the I/O overhead.)
def bench_calibrator():
    sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    rows = []
    print("\n=== ConfidenceCalibrator.update() — O(m)  [_save patched] ===")
    print(f"{'m':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "s.json"
        for m in sizes:
            probs  = rng.uniform(0.5, 0.99, m).tolist()
            labels = rng.integers(0, 2, m).astype(bool).tolist()
            def fn(probs=probs, labels=labels, state=state, m=m):
                c = ConfidenceCalibrator(state_path=state)
                with patch.object(c, "_save"):          # suppress disk I/O
                    for p, y in zip(probs, labels):
                        c.update(SignalType.FEATURE_REQUEST,
                                 predicted_prob=p, true_label=y)
            mean, std = _time(fn)
            rows.append((m, round(mean, 4), round(std, 5)))
            print(f"{m:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows

# ── Benchmark 4: BFS on synthetic graph  O(V+E) = O(n) ───────────────────────
def bench_bfs():
    sizes = [250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
    rows = []
    print("\n=== BFS on degree-4 ring graph — O(V+E) = O(n) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        adj = {i: [(i+1)%n, (i-1)%n, (i+2)%n, (i-2)%n] for i in range(n)}
        def fn(adj=adj):
            visited, q = set(), deque([0])
            visited.add(0)
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if w not in visited:
                        visited.add(w); q.append(w)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows



# ── curve fitting helper ──────────────────────────────────────────────────────
def fit_and_report(label, rows, model_fn, model_name):
    ns = np.array([r[0] for r in rows], dtype=float)
    ts = np.array([r[1] for r in rows], dtype=float)
    try:
        popt, _ = curve_fit(model_fn, ns, ts, p0=[1e-4], maxfev=10_000)
        r2 = r_squared(ts, model_fn(ns, *popt))
        print(f"  [{label}] fit: {model_name}  a={popt[0]:.4e}  R²={r2:.6f}")
        return popt[0], r2
    except Exception as e:
        print(f"  [{label}] fit failed: {e}"); return None, None

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + list(rows))
    print(f"  saved → {path}")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    print("=" * 58)
    print("  Social-Media-Radar Performance Benchmarks")
    print("=" * 58)

    bloom_rows = bench_bloom()
    print()
    fit_and_report("BloomFilter", bloom_rows, _constant, "T(n) = a  [O(1)]")
    write_csv(out / "bloom.csv",      ["n","mean_ms","std_ms"], bloom_rows)

    res_rows = bench_reservoir()
    print()
    fit_and_report("Reservoir",  res_rows,   _linear,   "T(n) = a·n  [O(n)]")
    write_csv(out / "reservoir.csv",  ["n","mean_ms","std_ms"], res_rows)

    cal_rows = bench_calibrator()
    print()
    fit_and_report("Calibrator", cal_rows,   _linear,   "T(m) = a·m  [O(m)]")
    write_csv(out / "calibrator.csv", ["m","mean_ms","std_ms"], cal_rows)

    bfs_rows = bench_bfs()
    print()
    fit_and_report("BFS",        bfs_rows,   _linear,   "T(n) = a·n  [O(n)]")
    write_csv(out / "bfs.csv",        ["n","mean_ms","std_ms"], bfs_rows)

    print("\n" + "=" * 58)
    print("  Done. CSV files written to deliverables/results/")
    print("  Run:  python deliverables/plot_results.py")
    print("=" * 58)

if __name__ == "__main__":
    main()
