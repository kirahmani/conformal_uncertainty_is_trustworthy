"""
Main experiment.

RQ1  Does split conformal achieve its nominal marginal false-alarm rate on a
     non-stationary microservice trace?
RQ2  Does it achieve that rate CONDITIONAL on operational phase?
RQ3  How long does the violation last after a phase boundary?
RQ4  Do adaptive and phase-conditional variants restore conditional validity,
     and at what cost in detection power?

All results are written to results/ as CSV and JSON. No number is printed in
the paper that is not produced here.
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phases import generate, GenConfig, PHASES, INLIER_PHASES, PHASE_ID  # noqa
from detectors import DETECTORS                                          # noqa
import conformal as cf                                                   # noqa

ALPHA = 0.05
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)


def calibration_index(windows, n_cal=1500, mode='steady'):
    """Select the calibration set.

    Two regimes, because which one an operator uses turns out to decide
    whether the marginal guarantee survives at all:

    steady  calibrate on a quiet period only. This is standard practice -
            you switch the detector on after observing a stable window - and
            it is the regime in which conformal validity fails most severely.
    mixed   calibrate on every inlier window in an initial observation period,
            transitions included. Harder to arrange operationally (you must
            know the period was fault-free) but much better behaved.

    Calibration always comes from the START of the trace. Drawing it from
    throughout would leak future regimes into calibration and conceal the
    effect under study.
    """
    if mode == 'steady':
        idx = windows.index[windows.phase == 'steady'].to_numpy()
    elif mode == 'mixed':
        idx = windows.index[~windows.is_fault].to_numpy()
    else:
        raise ValueError(mode)
    # Two disjoint sets in temporal order. The detector is fitted on the
    # training half and the conformal quantiles are computed on the
    # calibration half. Sharing them makes calibration scores in-sample and
    # therefore optimistically low, which inflates the apparent anomaly of
    # every out-of-sample window.
    sel = idx[:2 * n_cal]
    return sel[:n_cal], sel[n_cal:2 * n_cal]


def evaluate(p, windows, alpha=ALPHA, exclude=None):
    """False-alarm rate on held-out inliers (marginal and per phase), plus power."""
    flag = p <= alpha
    inl = ~windows.is_fault.to_numpy()
    if exclude is not None:
        inl = inl & ~exclude
    flt = windows.is_fault.to_numpy()
    if exclude is not None:
        flt = flt & ~exclude
    out = {
        'marginal_fpr': float(flag[inl].mean()),
        'power_tpr': float(flag[flt].mean()),
        'n_inlier': int(inl.sum()),
        'n_fault': int(flt.sum()),
    }
    for ph in PHASES:
        m = (windows.phase == ph).to_numpy() & inl
        out[f'fpr_{ph}'] = float(flag[m].mean()) if m.sum() else np.nan
        out[f'n_{ph}'] = int(m.sum())
    inlier_phase_fprs = [out[f'fpr_{ph}'] for ph in INLIER_PHASES
                         if not np.isnan(out[f'fpr_{ph}'])]
    out['worst_phase_fpr'] = float(np.nanmax(inlier_phase_fprs))
    out['conditional_gap'] = float(out['worst_phase_fpr'] - alpha)
    out['violation_ratio'] = float(out['worst_phase_fpr'] / alpha)
    return out


def recovery_profile(p, windows, alpha=ALPHA, horizon=12, exclude=None):
    """False-alarm rate as a function of windows since a phase boundary.

    Answers RQ3 directly: how long after a benign regime change does the
    detector keep crying wolf? Only inlier windows are counted, so every
    alarm here is a false one.
    """
    flag = (p <= alpha)
    phase = windows.phase.to_numpy()
    inl = ~windows.is_fault.to_numpy()
    if exclude is not None:
        inl = inl & ~exclude
    boundaries = np.where(phase[1:] != phase[:-1])[0] + 1
    acc = {k: [] for k in range(horizon)}
    for b in boundaries:
        for k in range(horizon):
            t = b + k
            if t >= len(phase) or phase[t] != phase[b]:
                break
            if inl[t]:
                acc[k].append(bool(flag[t]))
    return pd.DataFrame([
        {'since_boundary': k,
         'fpr': float(np.mean(v)) if v else np.nan,
         'n': len(v)} for k, v in acc.items()])


def run_once(seed, detector_name, n_cal=1500, alpha=ALPHA, cal_mode='steady'):
    metrics, windows = generate(GenConfig(seed=seed))
    train_idx, cal_idx = calibration_index(windows, n_cal, cal_mode)

    det = DETECTORS[detector_name](seed=seed)
    det.fit(metrics[train_idx])
    scores = det.score(metrics)

    fitted = np.zeros(len(windows), dtype=bool)
    fitted[train_idx] = True
    fitted[cal_idx] = True

    rows, extras = [], {}

    # --- split conformal ---------------------------------------------------
    p_split = cf.split_conformal(scores, cal_idx)
    r = evaluate(p_split, windows, alpha, exclude=fitted)
    r.update(method='split', detector=detector_name, seed=seed, cal_mode=cal_mode)
    rows.append(r)
    extras['split_p'] = p_split

    # --- rolling calibration ----------------------------------------------
    p_roll = cf.rolling_conformal(scores, cal_idx, None, k=200)
    r = evaluate(p_roll, windows, alpha, exclude=fitted)
    r.update(method='rolling', detector=detector_name, seed=seed, cal_mode=cal_mode)
    rows.append(r)

    # --- rolling with frozen buffer ---------------------------------------
    p_rfz = cf.rolling_frozen_conformal(scores, cal_idx, k=200, alpha=alpha)
    r = evaluate(p_rfz, windows, alpha, exclude=fitted)
    r.update(method='rolling_frozen', detector=detector_name, seed=seed,
             cal_mode=cal_mode)
    rows.append(r)

    # --- adaptive conformal inference -------------------------------------
    p_aci, alpha_t = cf.aci_conformal(scores, cal_idx, alpha=alpha, gamma=0.02)
    flag = p_aci <= alpha_t
    inl = (~windows.is_fault.to_numpy()) & ~fitted
    flt_a = windows.is_fault.to_numpy() & ~fitted
    r = {'marginal_fpr': float(flag[inl].mean()),
         'power_tpr': float(flag[flt_a].mean()), 'n_inlier': int(inl.sum()),
         'n_fault': int(flt_a.sum())}
    for ph in PHASES:
        m = (windows.phase == ph).to_numpy() & inl
        r[f'fpr_{ph}'] = float(flag[m].mean()) if m.sum() else np.nan
        r[f'n_{ph}'] = int(m.sum())
    vals = [r[f'fpr_{ph}'] for ph in INLIER_PHASES if not np.isnan(r[f'fpr_{ph}'])]
    r['worst_phase_fpr'] = float(np.nanmax(vals))
    r['conditional_gap'] = float(r['worst_phase_fpr'] - alpha)
    r['violation_ratio'] = float(r['worst_phase_fpr'] / alpha)
    r.update(method='aci', detector=detector_name, seed=seed, cal_mode=cal_mode)
    rows.append(r)
    extras['alpha_t'] = alpha_t

    # --- Mondrian, oracle phase -------------------------------------------
    tax_oracle = windows.phase_id.to_numpy()
    p_mo = cf.mondrian_conformal(scores, cal_idx, tax_oracle)
    r = evaluate(p_mo, windows, alpha, exclude=fitted)
    r.update(method='mondrian_oracle', detector=detector_name, seed=seed, cal_mode=cal_mode)
    rows.append(r)

    # --- Mondrian, detected regime ----------------------------------------
    regime = cf.detect_phase_online(scores)
    p_md = cf.mondrian_conformal(scores, cal_idx, regime)
    r = evaluate(p_md, windows, alpha, exclude=fitted)
    r.update(method='mondrian_detected', detector=detector_name, seed=seed, cal_mode=cal_mode)
    rows.append(r)

    extras['fitted'] = fitted
    extras['windows'] = windows
    extras['scores'] = scores
    extras['regime'] = regime
    return pd.DataFrame(rows), extras


def main(seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
         detectors=('pca', 'iforest', 'dense_ae', 'lstm_ae'),
         cal_modes=('steady', 'mixed')):
    all_rows, rec_rows = [], []
    keep = None
    for dname in detectors:
        for mode in cal_modes:
            for sd in seeds:
                df, ex = run_once(sd, dname, cal_mode=mode)
                all_rows.append(df)
                rp = recovery_profile(ex['split_p'], ex['windows'], exclude=ex['fitted'])
                rp['detector'], rp['seed'], rp['cal_mode'] = dname, sd, mode
                rec_rows.append(rp)
                if dname == 'pca' and sd == 1 and mode == 'mixed':
                    keep = ex
            sub = pd.concat(all_rows[-len(seeds):])
            sp = sub[sub.method == 'split']
            print(f'  {dname:8s} cal={mode:6s}  '
                  f'marginal={sp.marginal_fpr.mean():.4f}  '
                  f'worst_phase={sp.worst_phase_fpr.mean():.4f}', flush=True)

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(f'{RESULTS}/raw_results.csv', index=False)
    pd.concat(rec_rows, ignore_index=True).to_csv(
        f'{RESULTS}/recovery_profile.csv', index=False)

    w = keep['windows'].copy()
    w['score'] = keep['scores']
    w['p_split'] = keep['split_p']
    w['alpha_t'] = keep['alpha_t']
    w['regime'] = keep['regime']
    w.to_csv(f'{RESULTS}/trace_lstm_seed1.csv', index=False)

    with open(f'{RESULTS}/config.json', 'w') as f:
        json.dump({'alpha': ALPHA, 'seeds': list(seeds),
                   'detectors': list(detectors), 'cal_modes': list(cal_modes),
                   'n_cal': 1500, 'n_windows_approx': 6000}, f, indent=2)
    return res


if __name__ == '__main__':
    print('Running phase-conditional conformal validity experiment...')
    res = main()
    print('\nDone. Rows:', len(res))
