#!/usr/bin/env python3
"""training/train.py — Production-grade ConfidenceCalibrator training loop.

Features
--------
* Multi-epoch training with configurable ``--epochs`` (default 5).
* Stratified 80/20 train/validation split by ``SignalType``.
* Per-epoch validation ECE and MCE across all 18 signal types.
* Step-decay LR schedule: lr halves every LR_DECAY_EPOCHS epochs.
* Early stopping: halts when ECE fails to improve by EARLY_STOP_MIN_DELTA
  for EARLY_STOP_PATIENCE consecutive epochs.
* Reproducible per-epoch shuffling via seeded RNG.
* Checkpoint saved on every ECE improvement; at most MAX_CHECKPOINTS kept.
* Full per-epoch summary table and final per-SignalType calibration report.

Usage::

    python training/train.py                        # 5 epochs, default paths
    python training/train.py --epochs 10 --lr 0.005
    python training/train.py --dry-run              # validate dataset only
"""

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.inference_models import SignalType
from app.intelligence.calibration import ConfidenceCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger: logging.Logger = logging.getLogger(__name__)

# ── Hyper-parameters (all named; no magic numbers in code below) ─────────────
#: Default passes over the training set.
DEFAULT_EPOCHS: int = 5
#: Starting learning rate.
DEFAULT_BASE_LR: float = 0.01
#: LR is multiplied by this value every LR_DECAY_EPOCHS epochs.
LR_DECAY_FACTOR: float = 0.5
#: Epochs between each LR decay step.
LR_DECAY_EPOCHS: int = 2
#: Fraction of each class reserved for validation.
VAL_FRACTION: float = 0.20
#: Consecutive non-improving epochs before early stop.
EARLY_STOP_PATIENCE: int = 2
#: Minimum ECE reduction that counts as an improvement.
EARLY_STOP_MIN_DELTA: float = 0.001
#: Equal-width probability bins for ECE/MCE.
ECE_N_BINS: int = 15
#: Maximum checkpoint files to keep in the checkpoint directory.
MAX_CHECKPOINTS: int = 3
#: epoch shuffle seed = SHUFFLE_SEED_BASE + epoch_index.
SHUFFLE_SEED_BASE: int = 42
#: Clamp epsilon for logit conversion (avoids log(0)).
LOGIT_EPS: float = 1e-7

# ── Default file paths ────────────────────────────────────────────────────────
DEFAULT_DATASET: Path = Path("training/signal_classification_dataset.jsonl")
DEFAULT_STATE: Path = Path("training/calibration_state.json")
DEFAULT_CKPT_DIR: Path = Path("training/checkpoints")


# ── Dataset I/O ───────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, skipping blank lines and malformed entries.

    Args:
        path: Filesystem path to the ``.jsonl`` file.

    Returns:
        List of record dicts.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Line %d skipped (bad JSON): %s", lineno, exc)
    return records


# ── Stratified split ──────────────────────────────────────────────────────────

def stratified_split(
    records: List[Dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into train/val sets, stratified by ``signal_type``.

    Args:
        records: Full labelled dataset.
        val_fraction: Proportion to reserve for validation; in (0, 1).
        seed: RNG seed for deterministic splits.

    Returns:
        Tuple of ``(train_records, val_records)``.

    Raises:
        ValueError: If ``val_fraction`` is not in (0, 1).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction!r}")
    rng: random.Random = random.Random(seed)
    by_class: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_class[rec.get("signal_type", "__unknown__")].append(rec)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for cls_recs in by_class.values():
        shuffled = cls_recs[:]
        rng.shuffle(shuffled)
        n_val: int = max(1, round(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(epoch: int, base_lr: float, decay_factor: float, decay_epochs: int) -> float:
    """Step-decay LR for zero-indexed epoch: ``base_lr × decay_factor^(epoch//decay_epochs)``.

    Args:
        epoch: Zero-based epoch index.
        base_lr: Learning rate at epoch 0.
        decay_factor: Multiplier applied every ``decay_epochs`` epochs.
        decay_epochs: Epochs between decay steps.

    Returns:
        Effective learning rate (always positive).

    Raises:
        ValueError: If any numeric argument is out of range.
    """
    if base_lr <= 0.0 or not math.isfinite(base_lr):
        raise ValueError(f"base_lr must be finite and positive; got {base_lr!r}")
    if not (0.0 < decay_factor <= 1.0):
        raise ValueError(f"decay_factor must be in (0, 1]; got {decay_factor!r}")
    if decay_epochs <= 0:
        raise ValueError(f"decay_epochs must be positive; got {decay_epochs!r}")
    return base_lr * (decay_factor ** (epoch // decay_epochs))


# ── ECE / MCE computation ─────────────────────────────────────────────────────

def compute_ece_mce(
    calibrator: ConfidenceCalibrator,
    val_records: List[Dict[str, Any]],
    n_bins: int,
) -> Tuple[float, float]:
    """Compute Expected Calibration Error and Maximum Calibration Error.

    Each validation record's ``confidence`` is converted to a raw log-odds
    score, fed through ``calibrator.calibrate()``, and the resulting probability
    and binary ground-truth label are placed into one of ``n_bins`` equal-width
    buckets.

        ECE = Σ_b (|B_b| / n) × |avg_conf_b − avg_acc_b|
        MCE = max_b |avg_conf_b − avg_acc_b|  (non-empty bins only)

    Args:
        calibrator: Fitted ``ConfidenceCalibrator`` instance.
        val_records: Held-out validation records.
        n_bins: Number of equal-width probability bins.

    Returns:
        ``(ece, mce)`` both in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``n_bins < 1``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be ≥ 1; got {n_bins!r}")

    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for record in val_records:
        raw_conf: float = float(record.get("confidence", 0.7))
        clamped: float = max(LOGIT_EPS, min(1.0 - LOGIT_EPS, raw_conf))
        raw_logit: float = math.log(clamped / (1.0 - clamped))
        try:
            sig: SignalType = SignalType(record["signal_type"])
        except (KeyError, ValueError):
            continue
        p_cal: float = calibrator.calibrate(raw_logit, sig)
        y: float = 0.0 if record.get("is_hard_negative", False) else 1.0
        bin_idx: int = min(int(p_cal * n_bins), n_bins - 1)
        bins[bin_idx].append((p_cal, y))

    n: int = len(val_records)
    if n == 0:
        return 0.0, 0.0
    ece: float = 0.0
    mce: float = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf: float = sum(p for p, _ in b) / len(b)
        avg_acc: float = sum(y for _, y in b) / len(b)
        err: float = abs(avg_conf - avg_acc)
        ece += (len(b) / n) * err
        mce = max(mce, err)
    return ece, mce


# ── Checkpoint management ─────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    ece: float,
    scalars: Dict[str, float],
    ckpt_dir: Path,
    max_checkpoints: int,
) -> Path:
    """Write a checkpoint JSON and prune older files beyond ``max_checkpoints``.

    Args:
        epoch: One-based epoch number embedded in the filename.
        ece: Validation ECE value that triggered this save.
        scalars: Snapshot of ``calibrator._scalars`` to serialise.
        ckpt_dir: Directory to write checkpoint files into.
        max_checkpoints: Maximum files retained; oldest deleted first.

    Returns:
        Path of the newly written checkpoint file.

    Raises:
        ValueError: If ``max_checkpoints < 1``.
    """
    if max_checkpoints < 1:
        raise ValueError(f"max_checkpoints must be ≥ 1; got {max_checkpoints!r}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = ckpt_dir / f"epoch_{epoch:03d}_ece_{ece:.4f}.json"
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "val_ece": ece,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "scalars": scalars,
    }
    with ckpt_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    existing: List[Path] = sorted(ckpt_dir.glob("epoch_*.json"))
    for stale in existing[:-max_checkpoints]:
        try:
            stale.unlink()
            logger.debug("Pruned checkpoint: %s", stale.name)
        except OSError as exc:
            logger.warning("Could not delete checkpoint %s: %s", stale, exc)
    return ckpt_path


# ── Table helpers ─────────────────────────────────────────────────────────────

def _print_table_header() -> None:
    """Print column headers for the per-epoch training summary table."""
    print(
        f"\n{'Epoch':>5}  {'LR':>8}  {'Train ex':>9}  "
        f"{'Val ECE':>8}  {'Val MCE':>8}  {'Best ECE':>8}  Status"
    )
    print("─" * 74)


def _print_table_row(
    epoch: int,
    lr: float,
    n_train: int,
    ece: float,
    mce: float,
    best_ece: float,
    status: str,
) -> None:
    """Print one row of the per-epoch training summary table."""
    print(
        f"{epoch:>5}  {lr:>8.5f}  {n_train:>9,}  "
        f"{ece:>8.4f}  {mce:>8.4f}  {best_ece:>8.4f}  {status}"
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Production-grade ConfidenceCalibrator training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",  type=Path,  default=DEFAULT_DATASET,
                        help="Labelled JSONL dataset path.")
    parser.add_argument("--state",    type=Path,  default=DEFAULT_STATE,
                        help="Calibration state JSON path.")
    parser.add_argument("--ckpt-dir", type=Path,  default=DEFAULT_CKPT_DIR,
                        help="Checkpoint directory.")
    parser.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--lr",       type=float, default=DEFAULT_BASE_LR,
                        help="Initial learning rate.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Validate dataset and split, then exit.")
    return parser.parse_args()


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> int:
    """Run the full production training loop and return an exit code.

    Returns:
        ``0`` on success, ``1`` on unrecoverable error.
    """
    args = _parse_args()

    logger.info("=" * 64)
    logger.info("Social-Media-Radar ConfidenceCalibrator — Production Training")
    logger.info("  dataset  : %s", args.dataset)
    logger.info("  state    : %s", args.state)
    logger.info("  ckpt-dir : %s", args.ckpt_dir)
    logger.info("  epochs   : %d", args.epochs)
    logger.info("  lr       : %.5f", args.lr)
    logger.info("=" * 64)

    try:
        all_records = load_dataset(args.dataset)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Loaded %d records", len(all_records))

    train_records, val_records = stratified_split(
        all_records, VAL_FRACTION, SHUFFLE_SEED_BASE
    )
    logger.info(
        "Split → train=%d  val=%d  (%.0f%% held out)",
        len(train_records), len(val_records), VAL_FRACTION * 100,
    )

    if args.dry_run:
        logger.info("Dry run complete — exiting.")
        return 0

    calibrator = ConfidenceCalibrator(state_path=args.state)

    best_ece: float = float("inf")
    best_ckpt: Optional[Path] = None
    no_improve_count: int = 0
    total_updates: int = 0
    total_skipped: int = 0

    _print_table_header()

    for epoch in range(args.epochs):
        epoch_num: int = epoch + 1
        lr: float = get_lr(epoch, args.lr, LR_DECAY_FACTOR, LR_DECAY_EPOCHS)

        epoch_train: List[Dict[str, Any]] = train_records[:]
        random.Random(SHUFFLE_SEED_BASE + epoch).shuffle(epoch_train)

        for record in epoch_train:
            try:
                sig: SignalType = SignalType(record["signal_type"])
            except (KeyError, ValueError):
                total_skipped += 1
                continue
            predicted_prob: float = float(record.get("confidence", 0.7))
            true_label: bool = not bool(record.get("is_hard_negative", False))
            try:
                calibrator.update(sig, predicted_prob, true_label, lr=lr)
                total_updates += 1
            except ValueError as exc:
                logger.warning("Skipping record at epoch %d: %s", epoch_num, exc)
                total_skipped += 1

        val_ece, val_mce = compute_ece_mce(calibrator, val_records, ECE_N_BINS)
        improved: bool = val_ece < best_ece - EARLY_STOP_MIN_DELTA
        if improved:
            best_ece = val_ece
            no_improve_count = 0
            best_ckpt = save_checkpoint(
                epoch_num, val_ece, dict(calibrator._scalars),
                args.ckpt_dir, MAX_CHECKPOINTS,
            )
            status = "IMPROVED"
        else:
            no_improve_count += 1
            status = "EARLY STOP" if no_improve_count >= EARLY_STOP_PATIENCE else "NO IMPROVEMENT"

        _print_table_row(
            epoch_num, lr, len(epoch_train), val_ece, val_mce, best_ece, status
        )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            logger.info(
                "Early stopping: no ECE improvement ≥ %.4f for %d epochs.",
                EARLY_STOP_MIN_DELTA, EARLY_STOP_PATIENCE,
            )
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(" Training Complete")
    print("═" * 64)
    print(f"  Total gradient updates : {total_updates:,}")
    print(f"  Skipped (invalid)      : {total_skipped:,}")
    print(f"  Best validation ECE    : {best_ece:.6f}")
    print(f"  Best checkpoint        : {best_ckpt or '(none saved)'}")

    print(f"\n  {'Signal Type':30s}  {'Mean |p_cal − y|':>18}")
    print("  " + "─" * 52)
    by_type: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    for record in val_records:
        raw_conf: float = float(record.get("confidence", 0.7))
        clamped: float = max(LOGIT_EPS, min(1.0 - LOGIT_EPS, raw_conf))
        raw_logit: float = math.log(clamped / (1.0 - clamped))
        try:
            sig = SignalType(record["signal_type"])
        except (KeyError, ValueError):
            continue
        p_cal: float = calibrator.calibrate(raw_logit, sig)
        y: float = 0.0 if record.get("is_hard_negative", False) else 1.0
        by_type[sig.value].append((p_cal, y))
    for st_val, pairs in sorted(by_type.items()):
        mae: float = sum(abs(p - y) for p, y in pairs) / len(pairs)
        print(f"  {st_val:30s}  {mae:>18.6f}")
    print("═" * 64 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

