"""
stress_tests.py — Adversarial and edge-case stress tests for Social-Media-Radar
core algorithms.

Tests:
  1a. BloomFilter at exactly declared capacity — empirical FPR vs theory
  1b. BloomFilter at 10× over-capacity — FPR near 1 confirmed
  2a. ReservoirSampler with reservoir_size=1 — chi-squared uniformity
  2b. ReservoirSampler with reservoir_size=n (full stream) — all items retained
  3a. ConfidenceCalibrator at boundary probability 1e-9 — no NaN/Inf
  3b. ConfidenceCalibrator at boundary probability 1−1e-9 — T stays in [T_MIN, ∞)
  4.  BFS on disconnected graph — only source component visited

Usage:
    python deliverables/stress_tests.py
"""

import math
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scraping.probabilistic_structures import BloomFilter
from app.scraping.reservoir_sampling import ReservoirSampler
from app.intelligence.calibration import ConfidenceCalibrator
from app.domain.inference_models import SignalType

_T_MIN = 0.1          # mirrors app/intelligence/calibration.py _T_MIN

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def _verdict(ok: bool, label: str, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return ok

# ── 1a. BloomFilter at declared capacity ─────────────────────────────────────
def test_bloom_at_capacity():
    """Insert n_train unique items into a filter declared for n_train.
    Then test n_test UNSEEN items and measure empirical FPR.
    The theoretical FPR formula is (1 − e^(−k·n/m))^k ≈ ε = 0.01.
    Acceptance: empirical FPR within ±0.005 of theoretical.
    """
    n_train, n_test = 10_000, 10_000
    eps = 0.01
    bf = BloomFilter(expected_elements=n_train, false_positive_rate=eps)
    k, m = bf.num_hashes, bf.size

    for i in range(n_train):
        bf.add(f"train_{i}")

    fp = sum(1 for i in range(n_test) if bf.contains(f"test_{i}"))
    empirical_fpr = fp / n_test

    load = k * n_train / m
    theoretical_fpr = (1 - math.exp(-load)) ** k
    delta = abs(empirical_fpr - theoretical_fpr)

    detail = (f"empirical={empirical_fpr:.4f}  theoretical={theoretical_fpr:.4f}  "
              f"Δ={delta:.4f}  k={k}  m={m:,}")
    return _verdict(delta <= 0.005, "BloomFilter @ capacity: FPR ≈ ε", detail)

# ── 1b. BloomFilter at 10× over-capacity ─────────────────────────────────────
def test_bloom_over_capacity():
    """Insert 10× more items than declared capacity.
    Expected FPR ≈ 99.5% (nearly all bits saturated).
    Acceptance: empirical FPR > 0.80.
    """
    n_declared, n_train, n_test = 10_000, 100_000, 10_000
    bf = BloomFilter(expected_elements=n_declared, false_positive_rate=0.01)
    k, m = bf.num_hashes, bf.size

    for i in range(n_train):
        bf.add(f"train_{i}")

    fp = sum(1 for i in range(n_test) if bf.contains(f"test_{i}"))
    empirical_fpr = fp / n_test

    load = k * n_train / m
    theoretical_fpr = (1 - math.exp(-load)) ** k

    detail = (f"empirical={empirical_fpr:.4f}  theoretical={theoretical_fpr:.4f}  "
              f"load_factor={n_train/n_declared:.0f}×")
    return _verdict(empirical_fpr > 0.80, "BloomFilter 10× over-capacity: FPR > 80%", detail)

# ── 2a. ReservoirSampler k=1 uniformity chi-squared ──────────────────────────
def test_reservoir_k1_uniformity():
    """reservoir_size=1 on a stream of n items.
    Each item should be selected with probability 1/n.
    Repeat TRIALS times; chi-squared test at α=0.01.
    """
    import random as _random
    n, TRIALS = 20, 10_000
    counts = [0] * n
    for trial in range(TRIALS):
        rs = ReservoirSampler(reservoir_size=1, random_seed=trial)
        for item in range(n):
            rs.add(item)
        selected = rs.reservoir[0]
        counts[selected] += 1

    expected = TRIALS / n
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    # chi-squared critical value for df=19, α=0.01 is 36.19
    chi2_critical = 36.19
    detail = f"chi²={chi2:.2f}  critical={chi2_critical}  df={n-1}"
    return _verdict(chi2 < chi2_critical,
                    "ReservoirSampler k=1 uniformity (chi²)", detail)

# ── 2b. ReservoirSampler k=n full stream ─────────────────────────────────────
def test_reservoir_full_stream():
    """reservoir_size == stream length: every item must be retained."""
    n = 500
    rs = ReservoirSampler(reservoir_size=n, random_seed=0)
    for i in range(n):
        rs.add(i)
    retained = set(rs.reservoir)
    all_present = all(i in retained for i in range(n))
    detail = f"reservoir size={len(rs.reservoir)}  all {n} items present={all_present}"
    return _verdict(all_present and len(rs.reservoir) == n,
                    "ReservoirSampler k=n: all items retained", detail)

# ── 3a. ConfidenceCalibrator boundary probability 1e-9 ───────────────────────
def test_calibrator_boundary_low():
    """predicted_prob=1e-9 (very low): clamp → no NaN/Inf, T stays ≥ T_MIN."""
    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "s.json")
        with patch.object(c, "_save"):
            c.update(SignalType.CHURN_RISK, predicted_prob=1e-9, true_label=True)
            c.update(SignalType.CHURN_RISK, predicted_prob=1e-9, true_label=False)
        t = c._scalars.get(SignalType.CHURN_RISK.value, 1.0)
        ok = math.isfinite(t) and t >= _T_MIN
        detail = f"T={t:.6f}  isfinite={math.isfinite(t)}  ≥T_MIN({_T_MIN})={t >= _T_MIN}"
    return _verdict(ok, "Calibrator boundary p=1e-9: no NaN/Inf, T≥T_MIN", detail)

# ── 3b. ConfidenceCalibrator boundary probability 1−1e-9 ─────────────────────
def test_calibrator_boundary_high():
    """predicted_prob=1−1e-9 (very high): clamp → no NaN/Inf, T stays ≥ T_MIN."""
    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "s.json")
        with patch.object(c, "_save"):
            c.update(SignalType.CHURN_RISK, predicted_prob=1 - 1e-9, true_label=True)
            c.update(SignalType.CHURN_RISK, predicted_prob=1 - 1e-9, true_label=False)
        t = c._scalars.get(SignalType.CHURN_RISK.value, 1.0)
        ok = math.isfinite(t) and t >= _T_MIN
        detail = f"T={t:.6f}  isfinite={math.isfinite(t)}  ≥T_MIN({_T_MIN})={t >= _T_MIN}"
    return _verdict(ok, "Calibrator boundary p=1−1e-9: no NaN/Inf, T≥T_MIN", detail)

# ── 4. BFS on disconnected graph ──────────────────────────────────────────────
def test_bfs_disconnected():
    """Graph has two equal-sized components (0..n/2-1 and n/2..n-1).
    BFS from node 0 must visit exactly n/2 nodes — never crossing the gap.
    """
    n = 1_000
    half = n // 2
    # Component A: ring over 0..half-1
    adj_a = {i: [(i+1) % half, (i-1) % half] for i in range(half)}
    # Component B: ring over half..n-1 (isolated — not connected to A)
    adj_b = {i: [(half + (i - half + 1) % half),
                 (half + (i - half - 1) % half)] for i in range(half, n)}
    adj = {**adj_a, **adj_b}

    visited, q = set(), deque([0])
    visited.add(0)
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in visited:
                visited.add(w); q.append(w)

    in_A = all(v < half for v in visited)
    detail = (f"visited={len(visited)}  expected={half}  "
              f"all_in_component_A={in_A}")
    return _verdict(len(visited) == half and in_A,
                    "BFS disconnected graph: only source component visited", detail)

# ── 5a. Calibrator: large batch (m=10⁶ updates) ───────────────────────────────
def test_calibrator_large_batch():
    """m=1,000,000 gradient updates in one session with _save patched.
    All T values must remain in [T_MIN, T_MAX] and be finite throughout.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    m = 1_000_000
    probs  = rng.uniform(0.01, 0.99, m).tolist()
    labels = rng.integers(0, 2, m).astype(bool).tolist()

    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "s.json")
        with patch.object(c, "_save"):
            for p, y in zip(probs, labels):
                c.update(SignalType.FEATURE_REQUEST, predicted_prob=p, true_label=y)

        t = c._scalars.get(SignalType.FEATURE_REQUEST.value, 1.0)
        ok = math.isfinite(t) and _T_MIN <= t <= 100.0
        detail = f"m={m:,}  T={t:.6f}  finite={math.isfinite(t)}  in_bounds={_T_MIN}≤{t:.4f}≤100.0"
    return _verdict(ok, "Calibrator large batch (m=10⁶): T finite and in [T_MIN, T_MAX]", detail)


# ── 5b. Calibrator: all 18 SignalTypes updated simultaneously ─────────────────
def test_calibrator_all_signal_types():
    """Update every SignalType once and verify all scalars are finite and bounded."""
    from app.domain.inference_models import SignalType as ST
    all_types = list(ST)
    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "s.json")
        with patch.object(c, "_save"):
            for st in all_types:
                c.update(st, predicted_prob=0.75, true_label=True)

        bad = [
            st for st in all_types
            if not math.isfinite(c._scalars.get(st.value, 1.0))
            or not (_T_MIN <= c._scalars.get(st.value, 1.0) <= 100.0)
        ]
        ok = len(bad) == 0
        detail = (f"updated={len(all_types)} types  bad={len(bad)}"
                  + (f"  failed={[s.value for s in bad]}" if bad else ""))
    return _verdict(ok, "Calibrator all-18 SignalTypes: all T finite and bounded", detail)


# ── 5c. Calibrator: alternating extreme inputs ────────────────────────────────
def test_calibrator_alternating_extremes():
    """10,000 updates alternating p=1e-9 and p=1-1e-9 (will be clamped).
    T must stay finite and in [T_MIN, T_MAX] throughout.
    """
    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "s.json")
        with patch.object(c, "_save"):
            for i in range(10_000):
                p = 1e-9 if i % 2 == 0 else 1 - 1e-9
                label = (i % 3 == 0)
                c.update(SignalType.CHURN_RISK, predicted_prob=p, true_label=label)

        t = c._scalars.get(SignalType.CHURN_RISK.value, 1.0)
        ok = math.isfinite(t) and _T_MIN <= t <= 100.0
        detail = (f"10,000 alternating extremes  T={t:.6f}  "
                  f"finite={math.isfinite(t)}  in_bounds={ok}")
    return _verdict(ok, "Calibrator alternating extremes (10K updates): T bounded", detail)


# ── 5d. Calibrator: concurrent thread safety ──────────────────────────────────
def test_calibrator_thread_safety():
    """Launch 8 threads each performing 500 updates on different SignalTypes.
    Assert: no exception raised, no T outside [T_MIN, T_MAX], no NaN/Inf.
    Note: Python's GIL makes dict updates atomic at the bytecode level, but the
    read-modify-write sequence in update() is NOT atomic.  The test verifies that
    the safety clamps prevent T from escaping valid bounds even under races.
    """
    import numpy as np
    N_THREADS, N_UPDATES = 8, 500
    signal_types = [
        SignalType.FEATURE_REQUEST, SignalType.BUG_REPORT, SignalType.CHURN_RISK,
        SignalType.PRAISE, SignalType.COMPETITOR_MENTION, SignalType.LEGAL_RISK,
        SignalType.SECURITY_CONCERN, SignalType.REPUTATION_RISK,
    ]
    errors = []

    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "s.json")
        rng = np.random.default_rng(7)
        probs  = rng.uniform(0.01, 0.99, N_THREADS * N_UPDATES).tolist()
        labels = rng.integers(0, 2, N_THREADS * N_UPDATES).astype(bool).tolist()

        def worker(thread_idx: int):
            st = signal_types[thread_idx]
            base = thread_idx * N_UPDATES
            try:
                with patch.object(c, "_save"):
                    for i in range(N_UPDATES):
                        c.update(st, predicted_prob=probs[base + i],
                                 true_label=labels[base + i])
            except Exception as exc:
                errors.append(f"thread-{thread_idx}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()

        bad_scalars = [
            (k, v) for k, v in c._scalars.items()
            if not math.isfinite(v) or not (_T_MIN <= v <= 100.0)
        ]

    ok = len(errors) == 0 and len(bad_scalars) == 0
    detail = (f"threads={N_THREADS}  updates_each={N_UPDATES}  "
              f"errors={len(errors)}  bad_scalars={len(bad_scalars)}")
    return _verdict(ok, "Calibrator concurrent threads: no exceptions, all T bounded", detail)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Social-Media-Radar Adversarial & Edge-Case Stress Tests")
    print("=" * 62)

    tests = [
        ("1a", test_bloom_at_capacity),
        ("1b", test_bloom_over_capacity),
        ("2a", test_reservoir_k1_uniformity),
        ("2b", test_reservoir_full_stream),
        ("3a", test_calibrator_boundary_low),
        ("3b", test_calibrator_boundary_high),
        ("4 ", test_bfs_disconnected),
        ("5a", test_calibrator_large_batch),
        ("5b", test_calibrator_all_signal_types),
        ("5c", test_calibrator_alternating_extremes),
        ("5d", test_calibrator_thread_safety),
    ]

    results = []
    for tag, fn in tests:
        print(f"\nTest {tag}:")
        results.append(fn())

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 62)
    print(f"  Results: {n_pass}/{len(results)} PASSED  |  {n_fail} FAILED")
    print("=" * 62)
    sys.exit(0 if n_fail == 0 else 1)

if __name__ == "__main__":
    main()

