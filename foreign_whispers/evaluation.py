"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics:        list[SegmentMetrics],
    aligned:        list[AlignedSegment],
    align_report:   dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard. All scores in [0, 1] (1 = perfect).

    Dimensions
    ----------
    timing_score : float
        How well TTS durations fit source time windows.
        Based on mean absolute duration error, capped at 5s worst case.
        1.0 = zero error, 0.0 = ≥5s mean error per segment.

    stretch_score : float
        Absence of severe time-stretching.
        Severe stretch (>1.4x) degrades audio quality noticeably.
        1.0 = no severe stretches, 0.0 = all segments severely stretched.

    coverage_score : float
        Fraction of segments that were successfully aligned without
        requiring re-ranking or failing entirely.
        1.0 = all segments accepted/mild-stretched, 0.0 = all failed.

    naturalness_score : float
        Consistency of speaking rate across segments (low variance = natural).
        Measured as coefficient of variation of predicted TTS durations
        relative to source durations. 1.0 = perfectly consistent rate,
        0.0 = wildly inconsistent (some segments rush, others drag).

    drift_score : float
        Absence of cumulative timeline drift from gap-shifts.
        Drift > 10s means the dubbed audio is significantly de-synced.
        1.0 = zero drift, 0.0 = ≥10s drift.

    overall : float
        Weighted average of all five dimensions.
        Weights: timing=0.25, stretch=0.20, coverage=0.30, naturalness=0.15, drift=0.10.
        Coverage weighted highest — a clip where half the segments fail is
        fundamentally unusable regardless of timing accuracy.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        aligned: Aligned segments from ``global_align`` or ``global_align_dp``.
        align_report: Optional pre-computed dict from ``clip_evaluation_report``.
            If None, computed automatically.

    Returns:
        Dict with keys: timing_score, stretch_score, coverage_score,
        naturalness_score, drift_score, overall, and a breakdown sub-dict
        with the raw values used to compute each score.
    """
    if not metrics:
        perfect = {
            "timing_score":      1.0,
            "stretch_score":     1.0,
            "coverage_score":    1.0,
            "naturalness_score": 1.0,
            "drift_score":       1.0,
            "overall":           1.0,
            "breakdown":         {},
        }
        return perfect

    report = align_report or clip_evaluation_report(metrics, aligned)

    # ------------------------------------------------------------------ #
    # 1. Timing score — how accurate are duration predictions?
    # ------------------------------------------------------------------ #
    _TIMING_WORST_CASE_S = 5.0  # 5s mean error = score of 0
    mean_err = report["mean_abs_duration_error_s"]
    timing_score = max(0.0, 1.0 - mean_err / _TIMING_WORST_CASE_S)

    # ------------------------------------------------------------------ #
    # 2. Stretch score — absence of severe stretching (>1.4x)
    # ------------------------------------------------------------------ #
    pct_severe = report["pct_severe_stretch"] / 100.0  # convert % to fraction
    stretch_score = 1.0 - pct_severe

    # ------------------------------------------------------------------ #
    # 3. Coverage score — fraction of segments without fatal failures
    # ------------------------------------------------------------------ #
    n = len(metrics)
    n_bad = sum(
        1 for m in metrics
        if decide_action(m) in (AlignAction.REQUEST_SHORTER, AlignAction.FAIL)
    )
    coverage_score = 1.0 - (n_bad / n)

    # ------------------------------------------------------------------ #
    # 4. Naturalness score — speaking rate consistency across segments
    #    Low coefficient of variation (CV) = consistent rate = natural
    # ------------------------------------------------------------------ #
    stretch_factors = [
        m.predicted_tts_s / m.source_duration_s
        for m in metrics
        if m.source_duration_s > 0
    ]
    if len(stretch_factors) >= 2:
        mean_sf = _stats.mean(stretch_factors)
        std_sf  = _stats.stdev(stretch_factors)
        cv = std_sf / mean_sf if mean_sf > 0 else 0.0
        # CV of 0 = perfect consistency; CV > 1 = very inconsistent
        # Clamp CV to [0, 1] for scoring (CV > 1 is extremely rare)
        naturalness_score = max(0.0, 1.0 - min(cv, 1.0))
    else:
        naturalness_score = 1.0

    # ------------------------------------------------------------------ #
    # 5. Drift score — absence of cumulative timeline drift
    # ------------------------------------------------------------------ #
    _DRIFT_WORST_CASE_S = 10.0  # 10s drift = score of 0
    drift = abs(report["total_cumulative_drift_s"])
    drift_score = max(0.0, 1.0 - drift / _DRIFT_WORST_CASE_S)

    # ------------------------------------------------------------------ #
    # Weighted overall score
    # ------------------------------------------------------------------ #
    weights = {
        "timing":      0.25,
        "stretch":     0.20,
        "coverage":    0.30,
        "naturalness": 0.15,
        "drift":       0.10,
    }
    overall = (
        weights["timing"]      * timing_score
        + weights["stretch"]     * stretch_score
        + weights["coverage"]    * coverage_score
        + weights["naturalness"] * naturalness_score
        + weights["drift"]       * drift_score
    )

    return {
        "timing_score":      round(timing_score,      3),
        "stretch_score":     round(stretch_score,     3),
        "coverage_score":    round(coverage_score,    3),
        "naturalness_score": round(naturalness_score, 3),
        "drift_score":       round(drift_score,       3),
        "overall":           round(overall,            3),
        "breakdown": {
            "mean_abs_duration_error_s": mean_err,
            "pct_severe_stretch":        report["pct_severe_stretch"],
            "n_bad_segments":            n_bad,
            "n_total_segments":          n,
            "stretch_cv":                round(cv if len(stretch_factors) >= 2 else 0.0, 3),
            "total_drift_s":             drift,
        },
    }