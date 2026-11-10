"""
Base anomaly detectors producing per-window nonconformity scores.

Three detectors are included so that the paper's finding can be shown to be a
property of CONFORMAL CALIBRATION UNDER PHASE CHANGE rather than an artefact
of one model. If only the LSTM-AE were used, a reviewer would reasonably ask
whether the coverage collapse is really a reconstruction-error pathology.

Every detector exposes the same interface:
    fit(metrics_train)            trains on inlier windows only
    score(metrics)  -> (n,)       higher = more anomalous

The score is the nonconformity measure fed to the conformal layer. Nothing in
the conformal machinery depends on which detector produced it.
"""

from __future__ import annotations
import numpy as np


def _windows_to_sequences(metrics, seq_len, stride):
    """(n_win, T, F) -> (n_win, n_seq, seq_len, F)"""
    n_win, T, F = metrics.shape
    idx = list(range(0, T - seq_len + 1, stride))
    seqs = np.stack([metrics[:, i:i + seq_len, :] for i in idx], axis=1)
    return seqs


# ----------------------------------------------------------------------
class LSTMAutoencoder:
    """Carried over from the authors' prior multi-layer architecture.

    Kept deliberately close to the published configuration so the detector
    is the same object the earlier paper evaluated; only the surrounding
    calibration changes.
    """
    name = 'lstm_ae'

    def __init__(self, n_features=8, hidden=48, seq_len=20, stride=10,
                 epochs=12, lr=1e-3, batch=128, seed=42):
        self.seq_len, self.stride = seq_len, stride
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.n_features, self.hidden, self.seed = n_features, hidden, seed
        self.model = None

    def _build(self):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)

        class AE(nn.Module):
            def __init__(s, f, h):
                super().__init__()
                s.enc = nn.LSTM(f, h, 1, batch_first=True)
                s.dec = nn.LSTM(h, h, 1, batch_first=True)
                s.out = nn.Linear(h, f)

            def forward(s, x):
                _, (hn, _) = s.enc(x)
                lat = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)
                y, _ = s.dec(lat)
                return s.out(y)

        return AE(self.n_features, self.hidden)

    def fit(self, metrics):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        seqs = _windows_to_sequences(metrics, self.seq_len, self.stride)
        X = torch.FloatTensor(seqs.reshape(-1, self.seq_len, self.n_features))
        self.model = self._build()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss()
        dl = DataLoader(TensorDataset(X), batch_size=self.batch, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for (b,) in dl:
                opt.zero_grad()
                lossf(self.model(b), b).backward()
                opt.step()
        return self

    def score(self, metrics):
        import torch
        seqs = _windows_to_sequences(metrics, self.seq_len, self.stride)
        n_win, n_seq = seqs.shape[0], seqs.shape[1]
        X = torch.FloatTensor(seqs.reshape(-1, self.seq_len, self.n_features))
        self.model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                b = X[i:i + 2048]
                errs.append(((b - self.model(b)) ** 2).mean(dim=(1, 2)).numpy())
        e = np.concatenate(errs).reshape(n_win, n_seq)
        return e.mean(axis=1)


# ----------------------------------------------------------------------
class MLPAutoencoder:
    """Dense autoencoder over per-window summary statistics (scikit-learn).

    Implemented with MLPRegressor trained to reconstruct its own input, which
    keeps the dependency footprint to numpy/scikit-learn. The prior work's
    LSTM-AE is retained separately for continuity, but the paper's finding
    must not depend on a deep-learning stack being available.
    """
    name = 'mlp_ae'

    def __init__(self, hidden=(32, 12, 32), max_iter=400, seed=42):
        self.hidden, self.max_iter, self.seed = hidden, max_iter, seed
        self.model, self.mu, self.sd = None, None, None

    @staticmethod
    def _feats(metrics):
        return np.concatenate([metrics.mean(1), metrics.std(1),
                               np.percentile(metrics, 95, axis=1)], axis=1)

    def fit(self, metrics):
        from sklearn.neural_network import MLPRegressor
        F = self._feats(metrics)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        X = (F - self.mu) / self.sd
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden,
                                  max_iter=self.max_iter, random_state=self.seed,
                                  early_stopping=False).fit(X, X)
        return self

    def score(self, metrics):
        X = (self._feats(metrics) - self.mu) / self.sd
        return ((X - self.model.predict(X)) ** 2).mean(axis=1)


# ----------------------------------------------------------------------
class PCAReconstruction:
    """Linear reconstruction-error detector.

    The simplest possible baseline. Including it matters: if the coverage
    collapse appears even here, it cannot be blamed on model capacity,
    overfitting, or any property of neural reconstruction.
    """
    name = 'pca'

    def __init__(self, n_components=4, seed=42):
        self.k, self.seed = n_components, seed
        self.model, self.mu, self.sd = None, None, None

    @staticmethod
    def _feats(metrics):
        return np.concatenate([metrics.mean(1), metrics.std(1),
                               np.percentile(metrics, 95, axis=1)], axis=1)

    def fit(self, metrics):
        from sklearn.decomposition import PCA
        F = self._feats(metrics)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        X = (F - self.mu) / self.sd
        self.model = PCA(n_components=self.k, random_state=self.seed).fit(X)
        return self

    def score(self, metrics):
        X = (self._feats(metrics) - self.mu) / self.sd
        R = self.model.inverse_transform(self.model.transform(X))
        return ((X - R) ** 2).mean(axis=1)


# ----------------------------------------------------------------------
class IsolationForestDetector:
    name = 'iforest'

    def __init__(self, n_estimators=200, seed=42):
        self.n_estimators, self.seed = n_estimators, seed
        self.model = None

    @staticmethod
    def _feats(metrics):
        return np.concatenate([metrics.mean(1), metrics.std(1),
                               np.percentile(metrics, 95, axis=1)], axis=1)

    def fit(self, metrics):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed,
            contamination='auto').fit(self._feats(metrics))
        return self

    def score(self, metrics):
        # sklearn returns higher = more normal; negate so higher = anomalous
        return -self.model.score_samples(self._feats(metrics))


# The two neural detectors live in neural.py and need PyTorch. They are
# registered only if the import succeeds, so the study still runs end-to-end
# on PCA and Isolation Forest alone.
DETECTORS = {
    'iforest': IsolationForestDetector,
    'pca': PCAReconstruction,
    'mlp_ae': MLPAutoencoder,          # scikit-learn fallback, retained for reference
}

try:
    from neural import DenseAutoencoder, LSTMAutoencoder as TorchLSTMAutoencoder
    DETECTORS['dense_ae'] = DenseAutoencoder
    DETECTORS['lstm_ae'] = TorchLSTMAutoencoder
except Exception:
    pass
