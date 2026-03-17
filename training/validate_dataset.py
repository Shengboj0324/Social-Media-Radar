#!/usr/bin/env python3
"""Standalone validation script for training artefacts.

Implements docs/competitive_analysis.md §5.6 — Training Readiness Gate.

Usage::

    python training/validate_dataset.py

The script validates both JSONL files in ``training/`` against the following
rules:

1. Every ``signal_type`` value is a valid ``SignalType`` enum member.
2. Every class has ≥ 5 examples in ``signal_classification_dataset.jsonl``.
3. Hard-negative rate (unclear + not_actionable examples) is ≥ 20%.
4. The adversarial augmentation file is non-empty and all entries parse
   without error.

Prints per-class counts, max/min class ratio, and hard-negative rate.
Exits with code 1 and a descriptive error message on any assertion failure.
Exits with code 0 on success.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to sys.path so app imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.domain.inference_models import SignalType  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_TRAINING_DIR = Path(__file__).parent
_MAIN_DATASET = _TRAINING_DIR / "signal_classification_dataset.jsonl"
_ADVERSARIAL_DATASET = _TRAINING_DIR / "adversarial_augmentation.jsonl"

_MIN_EXAMPLES_PER_CLASS = 5
_MIN_HARD_NEGATIVE_RATE = 0.20
_HARD_NEGATIVE_TYPES = {"unclear", "not_actionable"}

# All valid signal type values
_VALID_SIGNAL_TYPES = {st.value for st in SignalType}


def _load_jsonl(path: Path) -> list:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        SystemExit: If the file does not exist or a line fails to parse.
    """
    if not path.exists():
        _fail(f"File not found: {path}")

    records = []
    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                _fail(f"JSON parse error at {path.name}:{lineno}: {exc}")
    return records


def _fail(message: str) -> None:
    """Print an error message and exit with code 1.

    Args:
        message: Human-readable error description.
    """
    print(f"\n❌  VALIDATION FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def _validate_signal_types(records: list, source: str) -> None:
    """Assert every signal_type in records is a valid SignalType.

    Args:
        records: List of JSONL dicts from the dataset.
        source: Filename string used in error messages.

    Raises:
        SystemExit: On first invalid signal_type found.
    """
    for i, rec in enumerate(records):
        st = rec.get("signal_type")
        if st not in _VALID_SIGNAL_TYPES:
            _fail(
                f"{source} record #{i + 1}: invalid signal_type='{st}'. "
                f"Must be one of: {sorted(_VALID_SIGNAL_TYPES)}"
            )


def validate_main_dataset() -> None:
    """Run all validation rules against signal_classification_dataset.jsonl.

    Raises:
        SystemExit: On any validation failure.
    """
    print(f"=== Validating {_MAIN_DATASET.name} ===")
    records = _load_jsonl(_MAIN_DATASET)
    total = len(records)
    print(f"  Total examples: {total}")

    # 1. All signal_types are valid enum members
    _validate_signal_types(records, _MAIN_DATASET.name)

    # 2. Per-class counts
    counts: Counter = Counter(r["signal_type"] for r in records)
    print("\n  Per-class counts:")
    for st in sorted(counts):
        bar = "█" * counts[st]
        print(f"    {st:<30} {counts[st]:>3}  {bar}")

    # 3. Every class has ≥ MIN_EXAMPLES_PER_CLASS
    under_threshold = {k: v for k, v in counts.items() if v < _MIN_EXAMPLES_PER_CLASS}
    if under_threshold:
        _fail(
            f"The following signal types have fewer than "
            f"{_MIN_EXAMPLES_PER_CLASS} examples: {under_threshold}"
        )

    # 4. Max/min class ratio
    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count if min_count else float("inf")
    print(f"\n  Max class count : {max_count}")
    print(f"  Min class count : {min_count}")
    print(f"  Max/min ratio   : {ratio:.2f}")

    # 5. Hard-negative rate ≥ 20%
    hard_neg_count = sum(v for k, v in counts.items() if k in _HARD_NEGATIVE_TYPES)
    hard_neg_rate = hard_neg_count / total if total else 0.0
    print(f"\n  Hard negatives  : {hard_neg_count}/{total} = {hard_neg_rate:.1%}")
    if hard_neg_rate < _MIN_HARD_NEGATIVE_RATE:
        _fail(
            f"Hard-negative rate {hard_neg_rate:.1%} is below the required "
            f"{_MIN_HARD_NEGATIVE_RATE:.0%}. "
            f"Add more 'unclear' or 'not_actionable' examples."
        )

    print(f"\n  ✅ {_MAIN_DATASET.name} passed all checks.")


def validate_adversarial_dataset() -> None:
    """Run basic sanity checks on adversarial_augmentation.jsonl.

    Raises:
        SystemExit: If the file is empty or contains invalid signal_types.
    """
    print(f"\n=== Validating {_ADVERSARIAL_DATASET.name} ===")
    records = _load_jsonl(_ADVERSARIAL_DATASET)
    total = len(records)
    print(f"  Total examples: {total}")

    if total == 0:
        _fail(f"{_ADVERSARIAL_DATASET.name} is empty.")

    # Adversarial entries use expected_signal_type — validate that field
    for i, rec in enumerate(records):
        est = rec.get("expected_signal_type")
        if est is not None and est not in _VALID_SIGNAL_TYPES:
            _fail(
                f"{_ADVERSARIAL_DATASET.name} record #{i + 1}: "
                f"invalid expected_signal_type='{est}'"
            )

    abstain_count = sum(1 for r in records if r.get("expected_abstain"))
    print(f"  Expected-abstain count: {abstain_count}/{total}")
    print(f"  ✅ {_ADVERSARIAL_DATASET.name} passed all checks.")


def main() -> None:
    """Entry point: run all validations and report overall result.

    Exits with code 0 on success, code 1 on any failure.
    """
    print("Social-Media-Radar training artefact validator")
    print("=" * 50)

    validate_main_dataset()
    validate_adversarial_dataset()

    print("\n" + "=" * 50)
    print("✅  All training artefacts passed validation.")


if __name__ == "__main__":
    main()

