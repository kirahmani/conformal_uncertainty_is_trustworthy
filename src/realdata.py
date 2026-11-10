"""
Real-telemetry replication on RCAEval multi-source data.

DATA
----
Online Boutique deployed on Kubernetes, instrumented at 1 Hz, from the
RCAEval benchmark (multi-source release asset). One fault case:

    1,441 s of telemetry, 73 metric channels over 11 services
    (cpu, mem, diskio, socket, latency-50, latency-90, error, workload),
    plus log and trace time series.
    Fault injected at t = 720 s (documented in inject_time.txt).

WHY THIS TRACE ANSWERS THE QUESTION
-----------------------------------
The 720 s preceding injection are documented fault-free. Any alarm raised
there is unambiguously false, with no reliance on a synthetic label. That
segment is also NOT stationary: 19 of 72 metric channels drift by more than
0.5 standard deviations between its first and second half, memory channels
by up to 1.8. Benign non-stationarity of exactly the kind the phase model
posits occurs here naturally, in real telemetry, with no fault present.

The real-data analysis therefore asks a narrower question than the synthetic
study, and one that needs no synthetic phase labels:

    Within genuinely fault-free real telemetry, does split conformal
    calibration hold its nominal rate marginally while failing conditional
    on observable operating regime?

REGIME VARIABLES
----------------
Two, both observable at prediction time and neither derived from the
detector's own output — which matters, because conditioning on a variable
computed from the score would be circular.

  elapsed   time since the calibration period ended, in strata.
            Tests whether calibration goes stale as the system drifts.
  workload  tertile of aggregate request volume across services, read from
            the *_workload channels. Tests whether coverage depends on how
            busy the system is.

DEPENDENCE
----------
Samples are 1 Hz and strongly autocorrelated: the aggregate workload series
has acf(1 s) = 0.96, acf(10 s) = 0.39, acf(30 s) = -0.03. Windows are
therefore NOT independent replicates, and intervals are computed by a moving
block bootstrap with a 30 s block (6 windows at W = 5 s), which is the lag at
which the autocorrelation has decayed. Treating windows as independent would
understate the intervals substantially.
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import DETECTORS            # noqa: E402
import conformal as cf                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'rcaeval_multisource')
RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)

WIN = 4             # seconds per detection window
BLOCK = 8           # bootstrap block length in windows (= 32 s)
ALPHA = 0.05
N_BOOT = 2000

# Channel families admitted as detector inputs. Specified on methodological
# grounds BEFORE any conditional result was examined, for three reasons:
#   - cpu/mem/latency/error are the families conventionally used for
#     microservice anomaly detection;
#   - socket and diskio counters span eight orders of magnitude and are
#     near-constant within calibration, which makes standardised features
#     numerically unstable;
#   - workload is EXCLUDED because it serves as a regime variable. Using it
#     both to detect and to condition would make the conditional analysis
#     circular.
DETECT_FAMILIES = ('cpu', 'mem', 'latency-50', 'latency-90', 'error')


# ----------------------------------------------------------------------
def load_trace():
    m = pd.read_csv(os.path.join(DATA, 'metrics.csv'))
    with open(os.path.join(DATA, 'inject_time.txt')) as f:
        inject = int(f.read().strip())
    t0 = int(m['time'].iloc[0])
    m = m.copy()
    m['rel'] = m['time'] - t0
    inject_rel = inject - t0

    chans = [c for c in m.columns if c not in ('time', 'rel')]
    pre = m[m.rel < inject_rel]
    keep = [c for c in chans
            if c.rsplit('_', 1)[-1] in DETECT_FAMILIES
            and pre[c].std() > 0 and np.isfinite(pre[c]).all()]
    dropped = len(chans) - len(keep)
    return m, keep, inject_rel, dropped


def sequence_context(m, chans, win=WIN, ctx=32):
    """Causal context sequences for the sequence autoencoder.

    A 4 s window holds only four samples, which is too short for a recurrent
    model to represent. Each window is therefore given the `ctx` raw 1 Hz
    samples ENDING at that window, so the model sees ~32 s of history. The
    context extends backwards only, so nothing from the future of a window
    enters its own score. Consecutive contexts overlap, which raises the
    dependence between neighbouring windows; the block bootstrap used for all
    real-trace intervals already accounts for that.
    """
    V = m[chans].to_numpy(dtype=float)
    n_win = int(len(m) // win)
    out = np.zeros((n_win, ctx, len(chans)), dtype=np.float32)
    for w in range(n_win):
        end = (w + 1) * win
        start = end - ctx
        if start < 0:                      # left-pad early windows by repeating
            pad = np.repeat(V[:1], -start, axis=0)
            out[w] = np.vstack([pad, V[:end]])
        else:
            out[w] = V[start:end]
    return out


def windowise(m, chans, inject_rel, win=WIN):
    """Non-overlapping windows -> (features, meta).

    One statistic per channel (the window mean). The synthetic study used
    mean, sd and 95th percentile, but that representation yields 3x as many
    features as there are calibration windows on this trace, which makes
    every held-out window look anomalous regardless of calibration method.
    The reduction is a sample-size constraint of the real trace, not a tuning
    choice, and it is applied identically to all detectors.
    """
    n_win = int(len(m) // win)
    X, rows = [], []
    for w in range(n_win):
        seg = m.iloc[w * win:(w + 1) * win]
        v = seg[chans].to_numpy(dtype=float)
        X.append(v.mean(0))
        mid = float(seg['rel'].mean())
        rows.append({'window': w, 'rel_mid': mid, 'is_fault': mid >= inject_rel})
    return np.asarray(X), pd.DataFrame(rows)


def robust_scale(X, cal_idx):
    """Median/IQR scaling fitted on calibration windows only."""
    c = X[cal_idx]
    med = np.median(c, axis=0)
    iqr = np.percentile(c, 75, axis=0) - np.percentile(c, 25, axis=0)
    iqr = np.where(iqr < 1e-9, 1.0, iqr)
    return (X - med) / iqr


def add_regimes(meta, m, chans, win=WIN, cal_end_window=None):
    """Attach the two regime variables. Both are label-free and causal."""
    wl_cols = [c for c in m.columns if c.endswith('_workload')]
    agg = []
    for w in range(len(meta)):
        seg = m.iloc[w * win:(w + 1) * win]
        agg.append(float(seg[wl_cols].sum(axis=1).mean()) if wl_cols else np.nan)
    meta = meta.copy()
    meta['workload'] = agg

    # workload tertiles fixed from the CALIBRATION portion only, so the cut
    # points do not depend on test data
    cal = meta.iloc[:cal_end_window]
    q = np.quantile(cal['workload'], [1 / 3, 2 / 3])
    meta['wl_regime'] = np.digitize(meta['workload'], q)
    meta['wl_regime_name'] = meta['wl_regime'].map(
        {0: 'low load', 1: 'medium load', 2: 'high load'})

    # Elapsed-time strata over the held-out period, in equal thirds. Using
    # thirds rather than fixed clock bins keeps the strata balanced whatever
    # the held-out period's length.
    since = (meta['window'] - cal_end_window).astype(float)
    # thirds computed over held-out INLIER windows only; including the fault
    # period would push the cut points outside the range being stratified
    held = since[(since >= 0) & (~meta['is_fault'])]
    if len(held):
        c1, c2 = np.quantile(held, [1 / 3, 2 / 3])
        lab = pd.Series(index=meta.index, dtype=object)
        lab[since < 0] = 'calibration'
        lab[(since >= 0) & (since < c1)] = 'early (1st third)'
        lab[(since >= c1) & (since < c2)] = 'mid (2nd third)'
        lab[since >= c2] = 'late (3rd third)'
        meta['elapsed'] = lab
        meta['since_cal_s'] = since * WIN
    return meta


# ----------------------------------------------------------------------
def block_bootstrap_rate(flags, block=BLOCK, n_boot=N_BOOT, seed=0):
    """Moving block bootstrap CI for a rate over a dependent sequence."""
    rng = np.random.default_rng(seed)
    x = np.asarray(flags, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    n_blocks = int(np.ceil(n / block))
    starts_max = max(n - block, 0)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, starts_max + 1, size=n_blocks)
        samp = np.concatenate([x[i:i + block] for i in s])[:n]
        out[b] = samp.mean()
    return float(x.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------------
def run(detector_name='pca', alpha=ALPHA, seed=0):
    m, chans, inject_rel, dropped = load_trace()
    X, meta = windowise(m, chans, inject_rel)

    inlier = ~meta.is_fault.to_numpy()
    inlier_idx = np.where(inlier)[0]
    # THREE disjoint sets, in temporal order. Fitting the detector on the
    # same windows used as the conformal calibration set makes calibration
    # scores in-sample and therefore optimistically low, so every
    # out-of-sample window appears anomalous. Split conformal requires a
    # proper training set distinct from the calibration set.
    n = len(inlier_idx)
    n_tr, n_ca = n // 3, n // 3
    train_idx = inlier_idx[:n_tr]
    cal_idx = inlier_idx[n_tr:n_tr + n_ca]
    cal_end_window = int(meta.loc[cal_idx[-1], 'window']) + 1
    meta = add_regimes(meta, m, chans, cal_end_window=cal_end_window)

    if detector_name == 'lstm_ae':
        # sequence model: raw causal context rather than summary statistics
        Seq = sequence_context(m, chans)
        med = np.median(Seq[train_idx].reshape(-1, Seq.shape[2]), axis=0)
        iqr = np.subtract(*np.percentile(
            Seq[train_idx].reshape(-1, Seq.shape[2]), [75, 25], axis=0))
        iqr = np.where(iqr < 1e-9, 1.0, iqr)
        Seq = ((Seq - med) / iqr).astype(np.float32)
        det = DETECTORS[detector_name](seed=seed, max_len=Seq.shape[1])
        det.fit(Seq[train_idx])
        scores = det.score(Seq)
    else:
        Xs = robust_scale(X, train_idx)
        det = DETECTORS[detector_name](seed=seed)
        det.fit(Xs[train_idx][:, None, :])
        scores = det.score(Xs[:, None, :])

    p = cf.split_conformal(scores, cal_idx)
    flag = p <= alpha

    # evaluation uses held-out inlier windows only
    test_mask = inlier.copy()
    test_mask[train_idx] = False
    test_mask[cal_idx] = False

    rows = []

    def rec(label, mask, kind):
        if mask.sum() == 0:
            return
        r, lo, hi = block_bootstrap_rate(flag[mask], seed=seed)
        rows.append({'detector': detector_name, 'kind': kind, 'regime': label,
                     'n_windows': int(mask.sum()), 'fpr': r, 'ci_lo': lo, 'ci_hi': hi})

    rec('marginal (all held-out inliers)', test_mask, 'marginal')
    for nm in ['low load', 'medium load', 'high load']:
        rec(nm, test_mask & (meta.wl_regime_name == nm).to_numpy(), 'workload')
    for nm in ['early (1st third)', 'mid (2nd third)', 'late (3rd third)']:
        rec(nm, test_mask & (meta.elapsed == nm).to_numpy(), 'elapsed')

    df = pd.DataFrame(rows)
    power = float(flag[meta.is_fault.to_numpy()].mean())
    info = {'detector': detector_name, 'power_tpr': power,
            'n_train': len(train_idx), 'n_cal': len(cal_idx),
            'n_test_inlier': int(test_mask.sum()),
            'n_fault': int(meta.is_fault.sum()), 'channels_used': len(chans),
            'channels_dropped': dropped, 'win_s': WIN, 'alpha': alpha,
            'inject_rel_s': int(inject_rel), 'block_windows': BLOCK}
    return df, info, meta, scores, p


def main(detectors=('pca', 'iforest', 'dense_ae', 'lstm_ae')):
    all_df, infos = [], []
    for d in detectors:
        df, info, meta, scores, p = run(d)
        all_df.append(df)
        infos.append(info)
        marg = df[df.kind == 'marginal'].iloc[0]
        worst = df[df.kind != 'marginal'].fpr.max()
        print(f"  {d:8s} marginal={marg.fpr:.4f} "
              f"[{marg.ci_lo:.4f},{marg.ci_hi:.4f}]  "
              f"worst-regime={worst:.4f}  power={info['power_tpr']:.4f}", flush=True)
        if d == 'pca':
            out = meta.copy()
            out['score'], out['p'] = scores, p
            out.to_csv(f'{RES}/real_trace_pca.csv', index=False)

    res = pd.concat(all_df, ignore_index=True)
    res.to_csv(f'{RES}/real_results.csv', index=False)
    with open(f'{RES}/real_info.json', 'w') as f:
        json.dump(infos, f, indent=2)
    return res, infos


if __name__ == '__main__':
    print('RCAEval multi-source replication (real Online Boutique telemetry)')
    res, infos = main()
    print()
    print(res.to_string(index=False))
