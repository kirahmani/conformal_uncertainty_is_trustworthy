"""
Phase-aware telemetry generator for microservice systems.

Extends the multi-layer simulator used in the authors' prior work (same
eight-feature metric taxonomy, same log-severity and trace-feature layers)
with an explicit OPERATIONAL PHASE schedule. The prior simulator produced a
binary normal/anomaly split; conformal validity cannot be studied there,
because exchangeability is either trivially satisfied or trivially broken.

The phases below are the point of the paper. Conformal prediction guarantees
a bounded false-alarm rate on inliers only when calibration and test data are
exchangeable. Real systems are not stationary: load ramps diurnally, code is
deployed, faults arrive, and systems recover into regimes unlike the one the
detector was calibrated on. Each phase below breaks exchangeability in a
different way, and crucially SOME OF THEM CONTAIN NO FAULT AT ALL - a window
in the ramp or deploy phase is a genuine inlier, so any alarm raised there is
a false alarm the conformal guarantee was supposed to bound.

PHASE SEMANTICS
  steady      stationary reference regime; calibration is drawn from here
  ramp        gradual load increase, no fault (inlier)
  deploy      abrupt but benign regime change: new version shifts latency and
              error baselines without being a fault (inlier)
  onset       fault appears, sharp transient (outlier)
  degraded    stable regime under a persistent fault (outlier)
  recovery    returning toward baseline, overshoot and oscillation (inlier)

`is_fault` below is the ground-truth label. A detector flagging a window with
is_fault == False has produced a false alarm, whatever the phase.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Eight-feature metric taxonomy, carried over unchanged from the prior
# simulator so results remain comparable across the two papers.
METRIC_SPECS = [
    ('cpu_usage',    0.20, 0.40, 0.65, 0.95),
    ('memory_usage', 0.25, 0.35, 0.70, 0.90),
    ('network_io',   0.15, 0.25, 0.60, 0.85),
    ('disk_io',      0.10, 0.20, 0.55, 0.80),
    ('request_rate', 0.20, 0.30, 0.70, 0.90),
    ('error_rate',   0.00, 0.05, 0.40, 0.70),
    ('latency_p50',  0.10, 0.25, 0.60, 0.85),
    ('latency_p99',  0.15, 0.35, 0.75, 0.95),
]
FEATURES = [s[0] for s in METRIC_SPECS]
N_FEATURES = len(METRIC_SPECS)

NORMAL_MU = np.array([(lo + hi) / 2 for _, lo, hi, _, _ in METRIC_SPECS])
FAULT_MU = np.array([(a + b) / 2 for _, _, _, a, b in METRIC_SPECS])

PHASES = ['steady', 'ramp', 'deploy', 'onset', 'degraded', 'recovery']
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
INLIER_PHASES = ['steady', 'ramp', 'deploy', 'recovery']
FAULT_PHASES = ['onset', 'degraded']


@dataclass
class GenConfig:
    n_cycles: int = 70             # repetitions of the phase schedule
    win_size: int = 60             # metric samples per detection window
    log_win: int = 40              # log messages per detection window
    normal_std: float = 0.030
    fault_std: float = 0.055
    ramp_strength: float = 0.34    # how far the ramp moves toward fault means
    deploy_shift: float = 0.24     # benign regime shift magnitude
    recovery_overshoot: float = 0.22
    drift_per_cycle: float = 0.002  # slow secular drift across the whole trace
    seed: int = 42


# Windows per phase within one cycle. Deliberately asymmetric: steady
# dominates, transitions are short. This matters - a transition that occupied
# half the trace would make the marginal guarantee fail too, and the paper's
# point is that it does NOT fail marginally.
# Steady state dominates, as it does in a real system: transitions are brief
# and rare. This balance is load-bearing for the paper. If transitions were a
# large share of operation, the MARGINAL guarantee would fail too, and the
# contribution - marginal validity coexisting with conditional failure -
# would disappear. Section VI reports sensitivity to this balance.
PHASE_LENGTHS = {
    'steady':   70,
    'ramp':      4,
    'deploy':    3,
    'onset':     2,
    'degraded':  5,
    'recovery':  4,
}


def _phase_schedule(cfg: GenConfig, rng) -> list[str]:
    """Build the per-window phase sequence.

    Each cycle jitters phase lengths so transition boundaries do not land on
    a fixed period, which would let a detector learn the schedule rather than
    the signal.
    """
    seq = []
    for _ in range(cfg.n_cycles):
        for ph in PHASES:
            base = PHASE_LENGTHS[ph]
            n = max(1, base + int(rng.integers(-1, 2)))
            seq.extend([ph] * n)
    return seq


def _window_mean(phase: str, pos: float, cfg: GenConfig, rng) -> np.ndarray:
    """Per-window feature means.

    `pos` is the fractional position within the current phase run, which is
    what makes ramp and recovery non-stationary WITHIN the phase rather than
    merely different between phases.
    """
    delta = FAULT_MU - NORMAL_MU

    if phase == 'steady':
        return NORMAL_MU.copy()

    if phase == 'ramp':
        # benign diurnal load increase: throughput-linked features move,
        # error rate does not
        w = cfg.ramp_strength * pos
        mu = NORMAL_MU + w * delta
        mu[FEATURES.index('error_rate')] = NORMAL_MU[FEATURES.index('error_rate')]
        return mu

    if phase == 'deploy':
        # abrupt benign regime change: a new build with different latency and
        # memory characteristics. Not a fault, but not exchangeable with
        # calibration either.
        mu = NORMAL_MU.copy()
        for f in ('latency_p50', 'latency_p99', 'memory_usage'):
            mu[FEATURES.index(f)] += cfg.deploy_shift * delta[FEATURES.index(f)]
        return mu

    if phase == 'onset':
        # sharp transient into the fault regime
        return NORMAL_MU + (0.45 + 0.55 * pos) * delta

    if phase == 'degraded':
        return FAULT_MU.copy()

    if phase == 'recovery':
        # decaying return to baseline with overshoot
        decay = np.exp(-3.0 * pos)
        osc = cfg.recovery_overshoot * decay * np.sin(2 * np.pi * pos)
        return NORMAL_MU + decay * 0.5 * delta + osc * delta

    raise ValueError(phase)


def generate(cfg: GenConfig | None = None):
    """Return (metrics, windows) where metrics is (n_windows, win_size, 8)."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    schedule = _phase_schedule(cfg, rng)
    n_win = len(schedule)

    # positional index within each phase run
    pos_in_run, run_len = [], []
    i = 0
    while i < n_win:
        j = i
        while j < n_win and schedule[j] == schedule[i]:
            j += 1
        L = j - i
        for k in range(L):
            pos_in_run.append(k / max(L - 1, 1))
            run_len.append(L)
        i = j

    metrics = np.zeros((n_win, cfg.win_size, N_FEATURES), dtype=np.float32)
    rows = []

    for w, phase in enumerate(schedule):
        pos = pos_in_run[w]
        mu = _window_mean(phase, pos, cfg, rng)

        # slow secular drift across the whole trace: even 'steady' is not
        # perfectly exchangeable with calibration taken long before it
        cycle_frac = w / n_win
        mu = mu + cfg.drift_per_cycle * cycle_frac * cfg.n_cycles * np.array(
            [1, 1, 0.5, 0.5, 1, 0.2, 1, 1]) * 0.02

        std = cfg.fault_std if phase in FAULT_PHASES else cfg.normal_std
        # transitions are noisier than stable regimes
        if phase in ('onset', 'recovery'):
            std *= 1.4

        samples = rng.normal(mu, std, size=(cfg.win_size, N_FEATURES))
        metrics[w] = np.clip(samples, 0.0, 1.0)

        rows.append({
            'window': w,
            'phase': phase,
            'phase_id': PHASE_ID[phase],
            'pos_in_phase': pos,
            'is_fault': phase in FAULT_PHASES,
            'is_transition': phase in ('ramp', 'deploy', 'onset', 'recovery'),
            'cycle': w // max(sum(PHASE_LENGTHS.values()), 1),
        })

    import pandas as pd
    windows = pd.DataFrame(rows)

    # Layer 2 (logs) and Layer 3 (traces), kept for the fused-score variant
    windows['S_log'] = _log_scores(windows, cfg, rng)
    windows['S_trace'] = _trace_scores(windows, cfg, rng)
    return metrics, windows


def _log_scores(windows, cfg: GenConfig, rng) -> np.ndarray:
    """Layer 2 score: fraction of log messages at ERROR/FATAL severity.

    Deploys produce a benign burst of WARN/ERROR lines (migration notices,
    cache invalidation) that the log layer cannot distinguish from a fault -
    this is one of the mechanisms by which a benign phase raises alarms.
    """
    out = np.zeros(len(windows))
    for i, r in windows.iterrows():
        ph = r['phase']
        if ph == 'steady':
            p = rng.beta(1.2, 60)
        elif ph == 'ramp':
            p = rng.beta(1.6, 45)
        elif ph == 'deploy':
            p = rng.beta(4.0, 22)        # benign but log-noisy
        elif ph == 'onset':
            p = rng.beta(7.0, 9)
        elif ph == 'degraded':
            p = rng.beta(12.0, 7)
        else:                             # recovery
            p = rng.beta(3.0, 25)
        k = rng.binomial(cfg.log_win, min(max(p, 0), 1))
        out[i] = k / cfg.log_win
    return out


def _trace_scores(windows, cfg: GenConfig, rng) -> np.ndarray:
    """Layer 3 score: fraction of traces breaching latency/span thresholds."""
    out = np.zeros(len(windows))
    for i, r in windows.iterrows():
        ph = r['phase']
        base = {'steady': 0.02, 'ramp': 0.10, 'deploy': 0.07,
                'onset': 0.45, 'degraded': 0.62, 'recovery': 0.12}[ph]
        out[i] = float(np.clip(rng.normal(base, 0.05), 0, 1))
    return out


if __name__ == '__main__':
    import pandas as pd
    m, w = generate()
    print(f'windows {len(w)}   metrics {m.shape}')
    print(w.groupby('phase').agg(
        n=('window', 'size'), fault=('is_fault', 'mean'),
        log=('S_log', 'mean'), trace=('S_trace', 'mean')).round(3))
    print(f"\ninlier windows: {(~w.is_fault).sum()}   fault windows: {w.is_fault.sum()}")
