"""
Neural autoencoder detectors (PyTorch).

Two models, differing in what they are allowed to see:

  DenseAutoencoder   per-window summary statistics (mean, sd, p95 per channel).
                     An explicit encoder-bottleneck-decoder, trained with Adam
                     and early stopping on a held-out slice of the training
                     windows. This replaces the scikit-learn MLPRegressor
                     stand-in used earlier, which had no controlled bottleneck
                     and no validation-based stopping.

  LSTMAutoencoder    the RAW multivariate sequence inside each window. This is
                     the only detector in the study with access to temporal
                     structure within a window; the other three see summary
                     statistics and are blind to ordering. Including it tests
                     whether the phase-conditional result is an artefact of
                     discarding within-window dynamics.

Both score a window by mean squared reconstruction error, so the conformal
layer above them is unchanged.

Training uses inlier windows from the proper training split only. The
calibration split is never seen during fitting - see Methods on the
train/calibration separation.
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH = True
except Exception:                                    # pragma: no cover
    TORCH = False


def _require_torch():
    if not TORCH:
        raise ImportError(
            'PyTorch is required for the neural detectors. Install with '
            '`pip install torch`, or run the study with the pca/iforest '
            'detectors, which reproduce the finding without it.')


# ----------------------------------------------------------------------
class DenseAutoencoder:
    """Undercomplete autoencoder over per-window summary statistics."""

    name = 'dense_ae'

    def __init__(self, bottleneck=8, hidden=48, epochs=300, lr=1e-3,
                 batch=128, val_frac=0.15, patience=25, seed=42):
        self.bottleneck, self.hidden = bottleneck, hidden
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.val_frac, self.patience, self.seed = val_frac, patience, seed
        self.model = self.mu = self.sd = None

    @staticmethod
    def _feats(metrics):
        return np.concatenate([metrics.mean(1), metrics.std(1),
                               np.percentile(metrics, 95, axis=1)], axis=1)

    def _build(self, d):
        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(d, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.bottleneck), nn.ReLU(),
            nn.Linear(self.bottleneck, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, d))

    def fit(self, metrics):
        _require_torch()
        from torch.utils.data import DataLoader, TensorDataset
        F = self._feats(metrics)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        X = torch.FloatTensor((F - self.mu) / self.sd)

        g = torch.Generator().manual_seed(self.seed)
        n_val = max(8, int(len(X) * self.val_frac))
        perm = torch.randperm(len(X), generator=g)
        val, tr = X[perm[:n_val]], X[perm[n_val:]]

        self.model = self._build(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = nn.MSELoss()
        dl = DataLoader(TensorDataset(tr), batch_size=self.batch, shuffle=True,
                        generator=g)

        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.model.train()
            for (b,) in dl:
                opt.zero_grad(); lossf(self.model(b), b).backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                v = float(lossf(self.model(val), val))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, metrics):
        X = torch.FloatTensor((self._feats(metrics) - self.mu) / self.sd)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                b = X[i:i + 4096]
                out.append(((b - self.model(b)) ** 2).mean(dim=1).numpy())
        return np.concatenate(out)


# ----------------------------------------------------------------------
class LSTMAutoencoder:
    """Sequence-to-sequence LSTM autoencoder over the raw window.

    The encoder compresses the window to its final hidden state; that state is
    repeated across the window length and decoded back to the original
    channels. Reconstruction error over the whole sequence is the score.

    Unlike the other detectors this one sees the ordering of samples inside a
    window, so a phase-conditional result that survives here cannot be blamed
    on summary statistics discarding within-window dynamics.
    """

    name = 'lstm_ae'

    def __init__(self, hidden=32, layers=1, epochs=60, lr=1e-3, batch=64,
                 max_len=60, val_frac=0.15, patience=8, seed=42):
        self.hidden, self.layers = hidden, layers
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.max_len, self.val_frac, self.patience = max_len, val_frac, patience
        self.seed = seed
        self.model = self.mu = self.sd = None

    def _build(self, n_feat):
        torch.manual_seed(self.seed)
        hidden, layers = self.hidden, self.layers

        class AE(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.LSTM(n_feat, hidden, layers, batch_first=True)
                s.dec = nn.LSTM(hidden, hidden, layers, batch_first=True)
                s.out = nn.Linear(hidden, n_feat)

            def forward(s, x):
                _, (h, _) = s.enc(x)
                lat = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
                y, _ = s.dec(lat)
                return s.out(y)

        return AE()

    def _prep(self, metrics):
        """Subsample long windows so training stays tractable."""
        T = metrics.shape[1]
        if T > self.max_len:
            idx = np.linspace(0, T - 1, self.max_len).astype(int)
            return metrics[:, idx, :]
        return metrics

    def fit(self, metrics):
        _require_torch()
        from torch.utils.data import DataLoader, TensorDataset
        M = self._prep(np.asarray(metrics, dtype=np.float32))
        flat = M.reshape(-1, M.shape[2])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        X = torch.FloatTensor((M - self.mu) / self.sd)

        g = torch.Generator().manual_seed(self.seed)
        n_val = max(8, int(len(X) * self.val_frac))
        perm = torch.randperm(len(X), generator=g)
        val, tr = X[perm[:n_val]], X[perm[n_val:]]

        self.model = self._build(M.shape[2])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = nn.MSELoss()
        dl = DataLoader(TensorDataset(tr), batch_size=self.batch, shuffle=True,
                        generator=g)

        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.model.train()
            for (b,) in dl:
                opt.zero_grad(); lossf(self.model(b), b).backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                v = float(lossf(self.model(val), val))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, metrics):
        M = self._prep(np.asarray(metrics, dtype=np.float32))
        X = torch.FloatTensor((M - self.mu) / self.sd)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                b = X[i:i + 512]
                out.append(((b - self.model(b)) ** 2).mean(dim=(1, 2)).numpy())
        return np.concatenate(out)
