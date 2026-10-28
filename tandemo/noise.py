"""Stage 1 interface, and a numpy empirical noise model (baseline, and for running without PyTorch).

A Stage 1 model gives, for each read and each candidate true length L on a locus grid,
log p(Y_read | L, x_read): the read's likelihood curve. Stage 2 only ever sees these curves.

EmpiricalNoise: per stratum (motif-period class x allele-length class), a histogram over the same
DeltaBins as the network, fitted by EM with the same mixture-over-alleles objective as network 1:
each read's allele is weighted by its phase weight and by how well each allele explains its length.
It has no read-level covariates, so it is a baseline for network 1, not a replacement.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .features import DeltaBins

PERIOD_EDGES = [1, 2, 3, 7]            # classes: 1, 2, 3-6, >=7
LEN_EDGES = [0, 50, 200, 1000]          # classes: <50, 50-200, 200-1000, >=1000


def stratum(period, L) -> np.ndarray:
    pc = np.searchsorted(PERIOD_EDGES, np.asarray(period), side="right") - 1
    lc = np.searchsorted(LEN_EDGES, np.asarray(L), side="right") - 1
    return np.clip(pc, 0, 3) * 4 + np.clip(lc, 0, 3)


class NoiseModel:
    bins: DeltaBins

    def log_curves(self, reads: pd.DataFrame, grid: np.ndarray) -> np.ndarray:  # [n_reads, n_grid]
        raise NotImplementedError


class EmpiricalNoise(NoiseModel):
    def __init__(self, bins: DeltaBins | None = None, logp: np.ndarray | None = None):
        self.bins = bins or DeltaBins()
        self.logp = logp                    # [16 strata, n bins], log probabilities

    def fit(self, df: pd.DataFrame, iters: int = 15, pseudo: float = 0.5):
        b = self.bins
        Y = df["Y"].to_numpy()
        La = np.stack([df["L1"].to_numpy(), df["L2"].to_numpy()], 1)
        w = df["w1_train"].to_numpy()
        W = np.stack([w, 1 - w], 1)
        per = df["period"].to_numpy()
        strata = np.stack([stratum(per, La[:, 0]), stratum(per, La[:, 1])], 1)
        idx, corr, valid = b.lookup(Y[:, None] - La)
        # start: Laplace-like around 0
        d = np.arange(-b.D, b.D + 1)
        base = np.concatenate([[1e-4], np.exp(-np.abs(d)), [1e-4]])
        logp = np.tile(np.log(base / base.sum()), (16, 1))
        for _ in range(iters):
            ll = logp[strata, idx] + corr
            ll = np.where(valid, ll, b.floor) + np.log(np.clip(W, 1e-12, 1))
            r = np.exp(ll - ll.max(1, keepdims=True))
            r /= r.sum(1, keepdims=True)
            counts = np.full((16, b.n), pseudo / b.n)
            for h in (0, 1):
                m = valid[:, h]
                np.add.at(counts, (strata[m, h], idx[m, h]), r[m, h])
            glob = counts.sum(0)
            # back-off: strata with few reads borrow from the pooled histogram (200 pseudo-reads)
            counts = counts + 200.0 * glob / glob.sum()
            logp = np.log(counts / counts.sum(1, keepdims=True))
        self.logp = logp
        return self

    def log_curves(self, reads: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
        b = self.bins
        Y = reads["Y"].to_numpy()[:, None]
        per = reads["period"].to_numpy()[:, None]
        G = np.asarray(grid)[None, :]
        idx, corr, valid = b.lookup(Y - G)
        st = stratum(np.broadcast_to(per, idx.shape), np.broadcast_to(G, idx.shape))
        out = self.logp[st, idx] + corr
        return np.where(valid, out, b.floor)

    def save(self, path: str):
        np.savez(path, logp=self.logp, bins=json.dumps(self.bins.state()), kind="empirical")

    @classmethod
    def load(cls, path: str):
        z = np.load(path, allow_pickle=False)
        return cls(DeltaBins(**json.loads(str(z["bins"]))), z["logp"])


def load_model(path: str) -> NoiseModel:
    if path.endswith(".npz"):
        return EmpiricalNoise.load(path)
    from .network1 import TorchNoise
    return TorchNoise.load(path)
