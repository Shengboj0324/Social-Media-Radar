"""Calibration evaluation metrics.

Blueprint §8 — Calibration metrics:
- Expected Calibration Error (ECE)
- Brier score
- Reliability diagram data (bin_center, accuracy, confidence, count)
- Overconfidence rate

Uses sklearn's calibration_curve for reliability diagram data and
brier_score_loss for the Brier score.  ECE is computed from the
calibration_curve output for bin-size weighting accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityBin:
    """One calibration bin for a reliability diagram."""
    bin_lower: float
    bin_upper: float
    bin_center: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float  # avg_confidence - avg_accuracy  (positive = overconfident)


@dataclass
class CalibrationReport:
    ece: float                          # Expected Calibration Error ∈ [0, 1]
    brier_score: float                  # Mean squared error of probability ∈ [0, 1]
    overconfidence_rate: float          # Fraction of bins where gap > 0.1
    reliability_bins: List[ReliabilityBin]
    total_samples: int
    n_bins: int


class CalibrationEvaluator:
    """Compute calibration metrics over a batch of probabilistic predictions.

    Parameters
    ----------
    n_bins : int
        Number of equal-width confidence bins (default 10 → 0.1 width each).
    overconfidence_gap : float
        Gap threshold above which a bin is considered overconfident.
    """

    def __init__(self, n_bins: int = 10, overconfidence_gap: float = 0.1) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self.overconfidence_gap = overconfidence_gap

    def evaluate(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
    ) -> CalibrationReport:
        """Compute calibration metrics via sklearn.

        Parameters
        ----------
        probabilities : sequence of float in [0, 1]
            Predicted probability that the prediction is correct.
        labels : sequence of int (0 or 1)
            1 if the predicted class was correct, 0 otherwise.
        """
        probs = np.array(probabilities, dtype=float)
        labs  = np.array(labels, dtype=int)
        n = len(probs)

        if n == 0:
            raise ValueError("Empty calibration batch")
        if len(labs) != n:
            raise ValueError("probabilities and labels must have the same length")
        if not np.all((labs == 0) | (labs == 1)):
            raise ValueError("labels must be binary (0 or 1)")

        # ---- Brier score via sklearn --------------------------------------
        brier = float(brier_score_loss(labs, probs))

        # ---- Reliability diagram via sklearn calibration_curve -----------
        # fraction_of_positives: actual accuracy per bin
        # mean_predicted_value:  average predicted confidence per bin
        frac_pos, mean_pred = calibration_curve(
            labs, probs, n_bins=self.n_bins, strategy="uniform"
        )

        # Build per-bin records; some bins may be empty (calibration_curve skips them)
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        reliability_bins: List[ReliabilityBin] = []
        ece_numerator = 0.0

        # Map sklearn bins back to edge ranges
        bin_width = 1.0 / self.n_bins
        for i, (avg_conf, avg_acc) in enumerate(zip(mean_pred, frac_pos)):
            lo = i * bin_width
            hi = lo + bin_width
            # Count samples in this bin (uniform strategy)
            mask = (probs >= lo) & (probs <= hi if i == self.n_bins - 1 else probs < hi)
            count = int(mask.sum())
            gap = float(avg_conf - avg_acc)
            ece_numerator += count * abs(gap)
            reliability_bins.append(ReliabilityBin(
                bin_lower=lo, bin_upper=hi,
                bin_center=(lo + hi) / 2,
                avg_confidence=float(avg_conf),
                avg_accuracy=float(avg_acc),
                count=count,
                gap=gap,
            ))

        ece = ece_numerator / n
        occupied = [b for b in reliability_bins if b.count > 0]
        overconfidence_rate = (
            sum(1 for b in occupied if b.gap > self.overconfidence_gap) / len(occupied)
        ) if occupied else 0.0

        report = CalibrationReport(
            ece=ece,
            brier_score=brier,
            overconfidence_rate=overconfidence_rate,
            reliability_bins=reliability_bins,
            total_samples=n,
            n_bins=self.n_bins,
        )
        logger.info(
            "CalibrationEvaluator: n=%d n_bins=%d ECE=%.4f Brier=%.4f overconf_rate=%.3f",
            n, self.n_bins, ece, brier, overconfidence_rate,
        )
        return report

