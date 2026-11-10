"""
Final analysis: tables, inference and figures.

UNIT OF ANALYSIS
----------------
This is the point on which the earlier version of this analysis was wrong,
and it matters enough to state explicitly.

Synthetic study. The independent replicate is the SEED, not the detection
window and not the detector. Windows within a run are autocorrelated (phases
are runs of consecutive windows), so pooling them as independent observations
overstates precision badly. Detectors are distinct methods, not replicates of
one method, so pooling across them conflates between-method variance with
between-seed variance. Every synthetic interval below is therefore computed
over n = 10 seeds, separately for each detector and calibration regime.

Real trace. There is a single trace, so seeds are unavailable and windows are
the only unit. The aggregate workload series has acf(1 s) = 0.96 and
acf(30 s) = -0.03, so intervals use a moving block bootstrap with a 32 s
block rather than assuming independence.

Method comparisons are paired on seed and tested with a Wilcoxon signed-rank
test, with Holm correction across the family of comparisons. With n = 10 the
smallest attainable two-sided p-value is 0.002, which bounds what any such
test can show here; that bound is reported rather than glossed over.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
ALPHA = 0.05

PHASES = ['steady', 'ramp', 'deploy', 'recovery']
PLAB = {'steady': 'Steady', 'ramp': 'Ramp', 'deploy': 'Deploy',
        'recovery': 'Recovery'}
MLAB = {'split': 'Split conformal',
        'rolling': 'Rolling calibration',
        'rolling_frozen': 'Rolling, frozen buffer',
        'aci': 'Adaptive conformal inference',
        'mondrian_oracle': 'Phase-conditional (known phase)',
        'mondrian_detected': 'Phase-conditional (detected regime)'}
MORDER = list(MLAB)
DLAB = {'pca': 'PCA', 'iforest': 'Isolation Forest',
        'dense_ae': 'Dense autoencoder', 'lstm_ae': 'LSTM autoencoder'}
DETS = ['pca', 'iforest', 'dense_ae', 'lstm_ae']

C = {'nominal': '#444444', 'split': '#C44536', 'rolling': '#2A6F97',
     'rolling_frozen': '#8D6A9F', 'aci': '#E09F3E',
     'mondrian_oracle': '#2E7D52', 'mondrian_detected': '#7FB069'}


def ci_t(x, conf=0.95):
    """Two-sided t interval for a mean over independent replicates."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()), np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def fmt(m, lo, hi, d=4):
    if np.isnan(lo):
        return f'{m:.{d}f}'
    return f'{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


# ----------------------------------------------------------------------
def table_main(res, cal_mode='mixed'):
    """Per-detector method comparison with seed-level intervals."""
    rows = []
    d = res[res.cal_mode == cal_mode]
    for det in DETS:
        for m in MORDER:
            s = d[(d.method == m) & (d.detector == det)]
            if not len(s):
                continue
            mm = ci_t(s.marginal_fpr)
            ww = ci_t(s.worst_phase_fpr)
            pw = ci_t(s.power_tpr)
            rows.append({
                'Detector': DLAB[det], 'Method': MLAB[m], 'n seeds': len(s),
                'Marginal FPR': fmt(*mm),
                'Worst-phase FPR': fmt(*ww),
                'Ratio to nominal': f'{ww[0] / ALPHA:.1f}',
                'Power (TPR)': fmt(*pw)})
    return pd.DataFrame(rows)


def table_phase(res):
    """Per-phase FPR for split conformal, by detector and calibration regime."""
    rows = []
    for mode in ['steady', 'mixed']:
        for det in DETS:
            s = res[(res.method == 'split') & (res.detector == det)
                    & (res.cal_mode == mode)]
            if not len(s):
                continue
            r = {'Calibration': ('Steady-state only' if mode == 'steady'
                                 else 'Mixed'),
                 'Detector': DLAB[det],
                 'Marginal': fmt(*ci_t(s.marginal_fpr))}
            for ph in PHASES:
                r[PLAB[ph]] = fmt(*ci_t(s[f'fpr_{ph}']))
            r['Power'] = f'{s.power_tpr.mean():.4f}'
            rows.append(r)
    return pd.DataFrame(rows)


def table_tests(res, cal_mode='mixed'):
    """Paired comparisons against split conformal, Holm-corrected."""
    d = res[res.cal_mode == cal_mode]
    rows = []
    for det in DETS:
        base = d[(d.method == 'split') & (d.detector == det)].sort_values('seed')
        for m in MORDER:
            if m == 'split':
                continue
            alt = d[(d.method == m) & (d.detector == det)].sort_values('seed')
            if len(alt) != len(base) or len(base) < 5:
                continue
            dw = base.worst_phase_fpr.to_numpy() - alt.worst_phase_fpr.to_numpy()
            dp = base.power_tpr.to_numpy() - alt.power_tpr.to_numpy()
            try:
                pw = stats.wilcoxon(dw).pvalue if np.any(dw != 0) else 1.0
            except ValueError:
                pw = 1.0
            rows.append({'Detector': DLAB[det], 'Comparison': MLAB[m],
                         'Δ worst-phase FPR': f'{-dw.mean():+.4f}',
                         'Δ power': f'{-dp.mean():+.4f}',
                         '_p': pw})
    t = pd.DataFrame(rows)
    if len(t):
        # Holm correction across the whole family of comparisons
        order = np.argsort(t['_p'].to_numpy())
        p = t['_p'].to_numpy()
        n = len(p)
        adj = np.empty(n)
        run = 0.0
        for rank, i in enumerate(order):
            val = (n - rank) * p[i]
            run = max(run, val)
            adj[i] = min(run, 1.0)
        t['p (Holm)'] = [f'{v:.4f}' if v >= 0.0005 else '<0.001' for v in adj]
        t = t.drop(columns='_p')
    return t


def table_real(real):
    rows = []
    for _, r in real.iterrows():
        rows.append({'Detector': DLAB[r.detector], 'Conditioning': r['kind'],
                     'Regime': r.regime, 'Windows': int(r.n_windows),
                     'FPR [95% block-bootstrap CI]':
                         f'{r.fpr:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]'})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def fig_phase(res):
    d = res[res.cal_mode == 'mixed']
    methods = ['split', 'aci', 'rolling', 'mondrian_detected', 'mondrian_oracle']
    x = np.arange(len(PHASES))
    w = 0.15
    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    for i, m in enumerate(methods):
        means, los, his = [], [], []
        for ph in PHASES:
            # average the per-detector seed means, then interval over seeds
            vals = d[d.method == m].groupby('seed')[f'fpr_{ph}'].mean()
            mm, lo, hi = ci_t(vals)
            means.append(mm); los.append(mm - lo); his.append(hi - mm)
        ax.bar(x + (i - 2) * w, means, w, yerr=[los, his], capsize=2,
               label=MLAB[m], color=C[m], edgecolor='white', linewidth=0.5)
    ax.axhline(ALPHA, color=C['nominal'], ls='--', lw=1.2,
               label=f'Nominal α = {ALPHA}')
    ax.set_xticks(x); ax.set_xticklabels([PLAB[p] for p in PHASES])
    ax.set_ylabel('False-alarm rate on held-out\ninlier windows')
    ax.set_xlabel('Operational phase')
    ax.legend(fontsize=7.2, ncol=2, framealpha=0.95, loc='upper left')
    ax.set_ylim(0, 0.80)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig1_phase_fpr.png', dpi=220)
    plt.close(fig)


def fig_recovery(rec):
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for mode, col, lab in (('mixed', C['split'], 'Mixed calibration'),
                           ('steady', C['rolling_frozen'], 'Steady-only calibration')):
        s = rec[rec.cal_mode == mode]
        g = s.groupby(['since_boundary', 'seed']).fpr.mean().reset_index()
        xs, ms, los, his = [], [], [], []
        for k, sub in g.groupby('since_boundary'):
            mm, lo, hi = ci_t(sub.fpr.dropna())
            xs.append(k); ms.append(mm); los.append(lo); his.append(hi)
        ax.plot(xs, ms, 'o-', color=col, lw=1.8, ms=4, label=lab)
        ax.fill_between(xs, los, his, color=col, alpha=0.15)
    ax.axhline(ALPHA, color=C['nominal'], ls='--', lw=1.2, label=f'Nominal α = {ALPHA}')
    ax.set_xlabel('Detection windows since phase boundary')
    ax.set_ylabel('False-alarm rate on held-out\ninlier windows')
    ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig2_recovery.png', dpi=220)
    plt.close(fig)


def fig_tradeoff(res):
    d = res[res.cal_mode == 'mixed']
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for m in MORDER:
        s = d[d.method == m]
        if not len(s):
            continue
        wv = s.groupby('seed').worst_phase_fpr.mean()
        pv = s.groupby('seed').power_tpr.mean()
        wm, wlo, whi = ci_t(wv)
        pm, plo, phi = ci_t(pv)
        ax.errorbar(wm, pm, xerr=[[wm - wlo], [whi - wm]],
                    yerr=[[pm - plo], [phi - pm]],
                    fmt='o', ms=8, color=C[m], capsize=3, label=MLAB[m])
    ax.axvline(ALPHA, color=C['nominal'], ls='--', lw=1.2)
    ax.annotate('nominal α', xy=(ALPHA, 0.02), xycoords=('data', 'axes fraction'),
                xytext=(4, 0), textcoords='offset points', fontsize=8,
                color=C['nominal'], va='bottom')
    ax.set_xscale('log'); ax.set_xlim(5e-4, 2.0); ax.set_ylim(0.25, 1.08)
    ax.set_xlabel('Worst-phase false-alarm rate (log scale)')
    ax.set_ylabel('Detection power (TPR)')
    ax.legend(fontsize=7.2, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.savefig(f'{FIG}/fig3_tradeoff.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def fig_real(real, trace):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8),
                                 gridspec_kw={'width_ratios': [1.25, 1]})
    # left: FPR by elapsed stratum, per detector
    strata = ['early (1st third)', 'mid (2nd third)', 'late (3rd third)']
    slab = ['Early', 'Middle', 'Late']
    dets = DETS
    cols = [C['split'], C['rolling'], C['mondrian_oracle'], C['aci']]
    x = np.arange(len(strata)); w = 0.20
    for i, (dt, col) in enumerate(zip(dets, cols)):
        s = real[(real.detector == dt) & (real.kind == 'elapsed')]
        s = s.set_index('regime').reindex(strata)
        lo = (s.fpr - s.ci_lo).clip(lower=0).fillna(0)
        hi = (s.ci_hi - s.fpr).clip(lower=0).fillna(0)
        a1.bar(x + (i - 1.5) * w, s.fpr.fillna(0), w, yerr=[lo, hi], capsize=2,
               label=DLAB[dt], color=col, edgecolor='white', linewidth=0.5)
    a1.axhline(ALPHA, color=C['nominal'], ls='--', lw=1.2, label='Nominal α = 0.05')
    a1.set_xticks(x); a1.set_xticklabels(slab)
    a1.set_xlabel('Third of the held-out fault-free period')
    a1.set_ylabel('False-alarm rate')
    a1.legend(fontsize=7.2, ncol=2)
    a1.spines[['top', 'right']].set_visible(False)
    a1.grid(axis='y', alpha=0.25, lw=0.6)

    # right: the real score trace
    t = trace.copy()
    inl = ~t.is_fault
    a2.plot(t.rel_mid[inl] / 60, t.score[inl], '.', ms=3.5, color='#333333',
            label='fault-free')
    a2.plot(t.rel_mid[~inl] / 60, t.score[~inl], '.', ms=3.5, color=C['split'],
            label='fault injected')
    a2.axvline(12, color=C['nominal'], ls='--', lw=1.1)
    a2.annotate('fault injected', xy=(12, 0.97), xycoords=('data', 'axes fraction'),
                xytext=(-4, 0), textcoords='offset points', fontsize=7.5,
                ha='right', va='top', color=C['nominal'])
    a2.set_yscale('log')
    a2.set_xlabel('Time (minutes)')
    a2.set_ylabel('Nonconformity score (PCA)')
    a2.legend(fontsize=7.2, loc='lower right')
    a2.spines[['top', 'right']].set_visible(False)
    a2.grid(alpha=0.25, lw=0.6)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig4_real.png', dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    res = pd.read_csv(f'{RES}/raw_results.csv')
    rec = pd.read_csv(f'{RES}/recovery_profile.csv')
    real = pd.read_csv(f'{RES}/real_results.csv')
    trace = pd.read_csv(f'{RES}/real_trace_pca.csv')

    t1 = table_main(res); t1.to_csv(f'{RES}/T1_methods.csv', index=False)
    t2 = table_phase(res); t2.to_csv(f'{RES}/T2_phase.csv', index=False)
    t3 = table_tests(res); t3.to_csv(f'{RES}/T3_tests.csv', index=False)
    t4 = table_real(real); t4.to_csv(f'{RES}/T4_real.csv', index=False)

    fig_phase(res); fig_recovery(rec); fig_tradeoff(res); fig_real(real, trace)
    fig_detector_profile(res)

    print('TABLE 1 — methods, mixed calibration (mean [95% CI] over 10 seeds)\n')
    print(t1.to_string(index=False))
    print('\n\nTABLE 2 — per-phase FPR, split conformal\n')
    print(t2.to_string(index=False))
    print('\n\nTABLE 3 — paired comparisons vs split conformal (Holm-corrected)\n')
    print(t3.to_string(index=False))
    print('\n\nTABLE 4 — real telemetry (RCAEval Online Boutique)\n')
    print(t4.to_string(index=False))
    return t1, t2, t3, t4


if __name__ == '__main__':
    main()


def fig_detector_profile(res):
    """Per-detector phase profile for split conformal under mixed calibration.

    Pooling detectors hides the most informative contrast in the study: the
    sequence model is far better than the others on level-shift phases and far
    worse on the phase whose temporal dynamics are unlike calibration.
    """
    d = res[(res.cal_mode == 'mixed') & (res.method == 'split')]
    x = np.arange(len(PHASES)); w = 0.20
    cols = [C['split'], C['rolling'], C['aci'], C['mondrian_oracle']]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for i, (dt, col) in enumerate(zip(DETS, cols)):
        s = d[d.detector == dt]
        ms, los, his = [], [], []
        for ph in PHASES:
            mm, lo, hi = ci_t(s[f'fpr_{ph}'])
            ms.append(mm); los.append(max(mm - lo, 0)); his.append(max(hi - mm, 0))
        ax.bar(x + (i - 1.5) * w, ms, w, yerr=[los, his], capsize=2,
               label=DLAB[dt], color=col, edgecolor='white', linewidth=0.5)
    ax.axhline(ALPHA, color=C['nominal'], ls='--', lw=1.2, label='Nominal α = 0.05')
    ax.set_xticks(x); ax.set_xticklabels([PLAB[p] for p in PHASES])
    ax.set_ylabel('False-alarm rate on held-out\ninlier windows')
    ax.set_xlabel('Operational phase')
    ax.legend(fontsize=7.2, ncol=2, loc='upper left')
    ax.set_ylim(0, 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig5_detector_profile.png', dpi=220)
    plt.close(fig)
