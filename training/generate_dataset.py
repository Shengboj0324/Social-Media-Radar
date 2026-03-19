#!/usr/bin/env python3
"""training/generate_dataset.py — Synthetic dataset generator for ConfidenceCalibrator.

Generates a balanced, reproducible JSONL training dataset with at least
10,000 labelled examples spread evenly across all 18 ``SignalType`` values.

Each record contains:
    signal_type     : str   — one of the 18 SignalType enum values
    confidence      : float — predicted probability in [CONF_MIN, CONF_MAX]
    is_hard_negative: bool  — True when model is confidently wrong (conf ≥ HN_CONF_MIN)
    platform        : str   — one of the five supported platforms
    text_length     : int   — synthetic post length in [TEXT_LEN_MIN, TEXT_LEN_MAX]
    source_quality  : float — estimated source reliability in [0.0, 1.0]

Hard negatives account for HN_FRACTION of each class (20–25 %) and have
confidence ≥ HN_CONF_MIN, modelling cases where the upstream model is
confidently wrong — exactly the examples that push calibration error up.

Usage::

    python training/generate_dataset.py            # writes default output path
    python training/generate_dataset.py --out path/to/file.jsonl
    python training/generate_dataset.py --seed 0  # different RNG seed
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Allow execution from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.inference_models import SignalType

# ── Dataset shape constants ───────────────────────────────────────────────────
#: Fixed seed guarantees identical datasets across runs.
_DEFAULT_SEED: int = 42

#: Target examples per SignalType (18 × 556 = 10,008 ≥ 10,000).
_N_PER_CLASS: int = 556

#: Fraction of each class that are hard negatives (22.1 % → 123 / 556).
#: Kept between 20 % and 25 % as required.
_HN_FRACTION: float = 0.221

#: Minimum / maximum allowed confidence for any example.
_CONF_MIN: float = 0.05
_CONF_MAX: float = 0.99

#: Hard negatives must have confidence ≥ this threshold (model is overconfident).
_HN_CONF_MIN: float = 0.75

#: Synthetic text-length range in characters.
_TEXT_LEN_MIN: int = 50
_TEXT_LEN_MAX: int = 800

#: Platforms from which social-media content is sourced.
_PLATFORMS: List[str] = ["reddit", "youtube", "rss", "twitter", "linkedin"]

#: Default output location (overwritten on each run).
_DEFAULT_OUTPUT: Path = Path("training/signal_classification_dataset.jsonl")


# ── Per-class generation ──────────────────────────────────────────────────────

def _generate_class(
    signal_type: SignalType,
    n: int,
    n_hard: int,
    rng: random.Random,
) -> List[Dict]:
    """Generate ``n`` labelled examples for one ``SignalType``.

    Args:
        signal_type: The target signal class for all examples in this batch.
        n: Total examples to generate for this class.
        n_hard: Number of those examples that are hard negatives.
        rng: Seeded RNG instance (mutated in place for reproducibility).

    Returns:
        List of ``n`` record dicts, shuffled.

    Raises:
        ValueError: If ``n_hard > n`` or ``n <= 0``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if n_hard > n:
        raise ValueError(f"n_hard ({n_hard}) must not exceed n ({n})")

    records: List[Dict] = []

    # Hard negatives: high confidence, is_hard_negative=True
    for _ in range(n_hard):
        records.append({
            "signal_type":      signal_type.value,
            "confidence":       round(rng.uniform(_HN_CONF_MIN, _CONF_MAX), 4),
            "is_hard_negative": True,
            "platform":         rng.choice(_PLATFORMS),
            "text_length":      rng.randint(_TEXT_LEN_MIN, _TEXT_LEN_MAX),
            "source_quality":   round(rng.random(), 4),
        })

    # Regular examples: any confidence, is_hard_negative=False
    n_easy: int = n - n_hard
    for _ in range(n_easy):
        records.append({
            "signal_type":      signal_type.value,
            "confidence":       round(rng.uniform(_CONF_MIN, _CONF_MAX), 4),
            "is_hard_negative": False,
            "platform":         rng.choice(_PLATFORMS),
            "text_length":      rng.randint(_TEXT_LEN_MIN, _TEXT_LEN_MAX),
            "source_quality":   round(rng.random(), 4),
        })

    rng.shuffle(records)
    return records


# ── Dataset assembly and I/O ──────────────────────────────────────────────────

def generate(
    output_path: Path = _DEFAULT_OUTPUT,
    seed: int = _DEFAULT_SEED,
) -> List[Dict]:
    """Generate the full balanced dataset and write it to ``output_path``.

    Args:
        output_path: Destination ``.jsonl`` file (created or overwritten).
        seed: RNG seed for full reproducibility.

    Returns:
        All generated records as a list of dicts (interleaved across classes).

    Raises:
        ValueError: If ``seed`` is not an integer.
    """
    if not isinstance(seed, int):
        raise ValueError(f"seed must be an int; got {type(seed).__name__!r}")

    rng: random.Random = random.Random(seed)
    n_hard: int = round(_N_PER_CLASS * _HN_FRACTION)   # = 123

    all_records: List[Dict] = []
    for signal_type in SignalType:
        class_records = _generate_class(signal_type, _N_PER_CLASS, n_hard, rng)
        all_records.extend(class_records)

    # Interleave classes so every window of the JSONL is class-diverse.
    rng.shuffle(all_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record) + "\n")

    return all_records


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic ConfidenceCalibrator training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUTPUT,
        help="Output .jsonl path.",
    )
    parser.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate dataset and print a per-class count table."""
    args = _parse_args()

    t0: float = time.perf_counter()
    records = generate(output_path=args.out, seed=args.seed)
    elapsed: float = time.perf_counter() - t0

    counts: Counter = Counter(r["signal_type"] for r in records)
    hn_counts: Counter = Counter(
        r["signal_type"] for r in records if r["is_hard_negative"]
    )

    header = f"\n{'Signal Type':30s} {'Count':>6}  {'Hard Neg':>8}  {'HN %':>5}"
    separator = "─" * 56
    print(header)
    print(separator)
    for st in sorted(counts.keys()):
        total = counts[st]
        hn = hn_counts.get(st, 0)
        pct = hn / total * 100.0
        print(f"{st:30s} {total:>6}  {hn:>8}  {pct:>4.1f}%")
    print(separator)
    total_all = sum(counts.values())
    total_hn = sum(hn_counts.values())
    print(f"{'TOTAL':30s} {total_all:>6}  {total_hn:>8}  "
          f"{total_hn/total_all*100:.1f}%")
    print(f"\nWritten {total_all:,} records → {args.out}  ({elapsed:.2f}s)\n")


if __name__ == "__main__":
    main()

