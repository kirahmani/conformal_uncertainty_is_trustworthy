"""
Conformal calibration layers.

SETTING. Split conformal outlier detection. A calibration set of inlier
windows yields nonconformity scores s_1..s_n. For a test window with score s,
the conformal p-value is

    p = (1 + #{i : s_i >= s}) / (n + 1)

and the window is flagged when p <= alpha. Under EXCHANGEABILITY of the test
inlier with the calibration inliers,

    P(p <= alpha | window is an inlier) <= alpha

so alpha is a guaranteed bound on the false-alarm rate. This is the guarantee
the paper interrogates. It is a MARGINAL statement: it averages over the test
distribution. It says nothing about the false-alarm rate conditional on the
system being in a particular operational phase, and that conditional rate is
what an operator actually experiences.

METHODS COMPARED
  split       classical split conformal, calibration fixed from steady state
  rolling     calibration set = most recent k inlier windows (sliding)
  aci         Adaptive Conformal Inference: alpha_t updated online from
              realised error, after Gibbs & Candes
  mondrian    per-phase calibration using ORACLE phase labels. Not deployable
              - it is the upper bound showing what phase conditioning buys if
              the phase were known perfectly.
  mondrian_d  Mondrian using DETECTED phase from an online change detector.
              This is the deployable version, and the gap between it and the
              oracle is the cost of not knowing the phase.

All methods return per-window p-values; flagging is p <= alpha.
"""

from __future__ import annotations
import numpy as np


def conformal_p(cal_scores: np.ndarray, s: float) -> float:
    n = len(cal_scores)
    if n == 0:
        return 1.0
    return (1.0 + np.sum(cal_scores >= s)) / (n + 1.0)


def conformal_p_vec(cal_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Vectorised p-values for a fixed calibration set."""
    cal = np.sort(cal_scores)
    n = len(cal)
    if n == 0:
        return np.ones_like(scores)
    # #{cal >= s} = n - searchsorted(cal, s, 'left')
    ge = n - np.searchsorted(cal, scores, side='left')
    return (1.0 + ge) / (n + 1.0)


# ----------------------------------------------------------------------
def split_conformal(scores, cal_idx, **_):
    """Classical split conformal with a fixed calibration set."""
    cal = scores[cal_idx]
    return conformal_p_vec(cal, scores)


# ----------------------------------------------------------------------
def rolling_conformal(scores, cal_idx, is_inlier_known=None, k=200, **_):
    """Sliding-window calibration over recent windows.

    is_inlier_known: boolean mask of windows whose inlier status is available
    at calibration time. In deployment an operator does not know which recent
    windows were inliers; passing None uses ALL recent windows, which is the
    honest deployable version and is what we report. Passing the oracle mask
    is available for a sensitivity check.
    """
    n = len(scores)
    p = np.ones(n)
    init = scores[cal_idx]
    buf = list(init[-k:])
    for t in range(n):
        p[t] = conformal_p(np.asarray(buf), scores[t])
        usable = True if is_inlier_known is None else bool(is_inlier_known[t])
        if usable:
            buf.append(scores[t])
            if len(buf) > k:
                buf.pop(0)
    return p


# ----------------------------------------------------------------------
def rolling_frozen_conformal(scores, cal_idx, k=200, alpha=0.05, **_):
    """Rolling calibration that refuses to admit windows it just flagged.

    Plain rolling calibration absorbs the fault regime into its own
    calibration set, which is why it loses detection power: after a few
    degraded windows the detector has recalibrated to treat the fault as
    normal. Freezing the buffer while an alarm is active is the obvious
    operational fix. It is self-referential - the filter depends on the
    decisions it produces - so it carries no formal guarantee, and measuring
    what it costs empirically is one of the things this paper is for.
    """
    n = len(scores)
    p = np.ones(n)
    buf = list(scores[cal_idx][-k:])
    for t in range(n):
        p[t] = conformal_p(np.asarray(buf), scores[t])
        if p[t] > alpha:                 # admit only unflagged windows
            buf.append(scores[t])
            if len(buf) > k:
                buf.pop(0)
    return p


# ----------------------------------------------------------------------
def aci_conformal(scores, cal_idx, alpha=0.05, gamma=0.02, **_):
    """Adaptive Conformal Inference.

    Keeps the calibration set fixed but adapts the working alpha_t from the
    realised error stream:  alpha_{t+1} = alpha_t + gamma * (alpha - err_t).
    Returns p-values AND the alpha_t trajectory, since the trajectory is a
    result in its own right: it shows how long the method takes to react to
    a phase change and how far it must overshoot to recover.
    """
    cal = np.sort(scores[cal_idx])
    n = len(scores)
    p = conformal_p_vec(cal, scores)
    alpha_t = np.zeros(n)
    a = alpha
    for t in range(n):
        alpha_t[t] = a
        err = 1.0 if p[t] <= a else 0.0
        a = float(np.clip(a + gamma * (alpha - err), 1e-4, 0.5))
    return p, alpha_t


# ----------------------------------------------------------------------
def mondrian_conformal(scores, cal_idx, taxonomy, **_):
    """Phase-conditional (Mondrian) conformal.

    `taxonomy` is a per-window integer group label. Calibration scores are
    partitioned by group and a test window is compared only against
    calibration windows of its own group, which restores exchangeability
    within the group and therefore restores conditional validity - provided
    the group label is correct and the group has enough calibration data.
    """
    n = len(scores)
    p = np.ones(n)
    cal_mask = np.zeros(n, dtype=bool)
    cal_mask[cal_idx] = True
    groups = np.unique(taxonomy)
    per_group = {}
    for g in groups:
        sel = cal_mask & (taxonomy == g)
        per_group[g] = np.sort(scores[sel])
    # Fallback pool for groups with too little calibration data: this is a
    # real deployment problem, not a simulation artefact - rare phases have
    # few calibration windows, which is exactly when the guarantee is
    # weakest.
    pooled = np.sort(scores[cal_mask])
    for t in range(n):
        g = taxonomy[t]
        cal = per_group.get(g, pooled)
        if len(cal) < 20:
            cal = pooled
        p[t] = conformal_p(cal, scores[t])
    return p


# ----------------------------------------------------------------------
def detect_phase_online(scores, window=12, z=2.5, min_run=3):
    """Online regime-change detector over the score stream.

    Deliberately simple and causal: it uses only past scores. It emits an
    integer regime id that increments whenever the running mean shifts by
    more than z robust standard deviations and the shift persists.

    This is what makes the Mondrian variant deployable: the operator does not
    know the true phase, but a change detector can tell them that the regime
    moved, which is enough to switch calibration pools.
    """
    n = len(scores)
    regime = np.zeros(n, dtype=int)
    cur = 0
    ref = scores[:window]
    mu, sd = np.median(ref), 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-9
    run = 0
    for t in range(n):
        if t < window:
            regime[t] = cur
            continue
        recent = scores[t - window:t]
        m = np.median(recent)
        if abs(m - mu) > z * sd:
            run += 1
            if run >= min_run:
                cur += 1
                ref = recent
                mu = np.median(ref)
                sd = 1.4826 * np.median(np.abs(ref - mu)) + 1e-9
                run = 0
        else:
            run = 0
        regime[t] = cur
    return regime
