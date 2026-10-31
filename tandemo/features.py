"""Covariates x of the Stage 1 noise model, and the bins of the observed-length error.

Read features describe how noisy this read is; locus features describe how hard this repeat is to
read; L features describe the candidate true length. None of them encodes the read's own repeat
length Y (no base counts, no quality sums over the repeat), so the network cannot read Y off its
input.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# name: (source column, transform, description) — the table in the docs is generated from this
READ_FEATURES = {
    "rq_phred": ("rq", "phred", "HiFi read quality as Phred, -10 log10(1 - rq), capped at 60"),
    "np_log": ("np", "log1p", "CCS passes, log(1 + np)"),
    "read_len_log": ("read_len", "log10", "read length, log10 bp"),
    "mapq": ("mapq", "div60", "mapping quality / 60"),
    "has_sa": ("n_sa", "pos", "read has supplementary alignments"),
    "bq_rep": ("bq_rep", "div40", "mean base quality over the read's repeat bases / 40"),
    "bq_flank": ("bq_flank", "div40", "mean base quality over the two 128 bp flanks / 40"),
    "dist_end_log": ("dist_end", "log1p", "bases from the repeat to the nearer read end, log"),
    "flank_err": ("flank_id_min", "err", "error rate of the worse flank, 1 - identity, x100"),
}
LOCUS_FEATURES = {
    "period_log": ("period", "log2", "motif period, log2 bp"),
    "ref_len_log": ("ref_len", "log1p", "reference repeat length, log bp"),
    "gc": ("gc", "id", "GC fraction of the reference repeat"),
    "hp_frac": ("hp_frac", "id", "fraction of repeat bases in homopolymer runs >= 4"),
    "max_hp_log": ("max_hp", "log1p", "longest homopolymer in the repeat, log"),
    "flank_cx": ("flank_cx", "id", "3-mer complexity of the lower-complexity flank"),
    "segdup": ("segdup", "id", "fraction of the repeat inside a segmental duplication (paralogous reads may collapse)"),
    "elem_gc": ("elem_gc", "id", "GC fraction of the locus's motif elements"),
    "elem_entropy": ("elem_entropy", "id", "Shannon entropy of the motif elements' base composition, bits (0-2)"),
    "purity": ("purity", "id", "fraction of repeat positions whose base recurs one period later (1 = pure, no interruptions)"),
    "n_motifs_log": ("n_motifs", "log1p", "number of distinct motif elements at the locus, log"),
}
# inputs of earlier models only: still computed for them, not used for new training
LEGACY_FEATURES = {
    "cx3": ("cx3", "id", "3-mer complexity of the repeat (distinct / possible); replaced by elem_entropy and purity"),
}
# 5mC at CpGs (MM/ML C+m calls; the PacBio model calls CpG context only). Means and fractions only, never counts
# (a CpG count over the repeat would scale with Y). Optional at inference: missing values get the training median
# and mC_avail = 0, and training masks them for a share of reads so the network also learns to do without.
METH_FEATURES = {
    "mC_flank": ("mC_flank", "id", "mean 5mC probability over CpG calls in the read's two 128 bp flanks"),
    "mC_rep": ("mC_rep", "id", "mean 5mC probability over CpG calls in the read's repeat bases"),
    "mC_cert": ("mC_cert", "id", "share of the read's CpG calls (flanks + repeat) that are decisive (p <= 0.2 or >= 0.8)"),
    "mC_avail": ("mC_avail", "avail", "1 when the read carries 5mC calls at the locus, else 0 (masked in training)"),
}
METH_NAMES = list(METH_FEATURES)
L_FEATURES = {
    "L_log": "candidate true length, log(1 + L), standardised",
    "L_vs_ref": "log(1 + L) - log(1 + reference length): distance of the allele from the reference",
    "L_units_log": "log(1 + L / period): candidate length in motif units",
}
X_NAMES = list(READ_FEATURES) + list(METH_FEATURES) + list(LOCUS_FEATURES)
X_NAMES_NO_METH = list(READ_FEATURES) + list(LOCUS_FEATURES)          # training with --no-methylation
# the 16 inputs of the first models, which did not store their feature list
FIRST_MODEL_NAMES = list(READ_FEATURES) + ["period_log", "ref_len_log", "gc", "hp_frac", "max_hp_log", "cx3", "flank_cx"]
ALL_FEATURES = {**READ_FEATURES, **METH_FEATURES, **LOCUS_FEATURES, **LEGACY_FEATURES}


def _tf(v: np.ndarray, kind: str) -> np.ndarray:
    v = v.astype(float)
    if kind == "phred":
        return np.minimum(-10 * np.log10(np.clip(1 - v, 1e-6, 1)), 60)
    if kind == "log1p":
        return np.log1p(np.clip(v, 0, None))
    if kind == "log10":
        return np.log10(np.clip(v, 1, None))
    if kind == "log2":
        return np.log2(np.clip(v, 1, None))
    if kind == "div60":
        return v / 60
    if kind == "div40":
        return v / 40
    if kind == "pos":
        return (v > 0).astype(float)
    if kind == "err":
        return 100 * (1 - v)
    return v


def raw_x(df: pd.DataFrame, names=None) -> np.ndarray:
    """Raw feature matrix [n, len(names)] (default X_NAMES). A source column the table lacks (e.g. methylation in
    an older extraction) gives NaN, and mC_avail = 0."""
    cols = []
    for name in (names or X_NAMES):
        src, kind = ALL_FEATURES[name][0], ALL_FEATURES[name][1]
        if kind == "avail":
            have = [df[c].to_numpy(dtype=float) for c in ("mC_flank", "mC_rep") if c in df]
            v = np.zeros(len(df)) if not have else np.any([~np.isnan(h) for h in have], axis=0).astype(float)
            cols.append(v)
            continue
        v = df[src].to_numpy(dtype=float).copy() if src in df else np.full(len(df), np.nan)
        if src == "np":
            v[v < 0] = np.nan          # tag missing
        cols.append(_tf(v, kind))
    return np.stack(cols, axis=1)


class Scaler:
    """Median fill and standardisation, fitted on the training reads and stored with the model."""

    def __init__(self, med=None, mu=None, sd=None, lmu=0.0, lsd=1.0):
        self.med, self.mu, self.sd, self.lmu, self.lsd = med, mu, sd, lmu, lsd

    def fit(self, X: np.ndarray, L: np.ndarray):
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        Xf = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-6
        ll = np.log1p(L[L >= 0])
        self.lmu, self.lsd = float(ll.mean()), float(ll.std() + 1e-6)
        return self

    def x(self, X: np.ndarray) -> np.ndarray:
        return ((np.where(np.isnan(X), self.med, X) - self.mu) / self.sd).astype(np.float32)

    def l_feats(self, L, ref_len, period) -> np.ndarray:
        """L features for candidate lengths L (any shape; ref_len and period broadcast)."""
        L = np.asarray(L, dtype=float)
        L = np.maximum(L, 0)                 # a pad deletion longer than the allele: no negative lengths
        a = (np.log1p(L) - self.lmu) / self.lsd
        b = np.log1p(L) - np.log1p(ref_len)
        c = np.log1p(L / np.maximum(period, 1))
        return np.stack(np.broadcast_arrays(a, b, c), axis=-1).astype(np.float32)

    def state(self) -> dict:
        return dict(med=self.med.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist(), lmu=self.lmu,
                    lsd=self.lsd)

    @classmethod
    def from_state(cls, s: dict):
        return cls(np.array(s["med"]), np.array(s["mu"]), np.array(s["sd"]), s["lmu"], s["lsd"])


class DeltaBins:
    """Bins of the error Delta = Y - L: one bin per bp for |Delta| <= D, plus a left and a right tail
    bin whose mass is spread uniformly over D < |Delta| <= T. The model's softmax over these bins
    is a normalised distribution of Y for each candidate L; a Delta beyond T gets log `floor`."""

    def __init__(self, D: int = 50, T: int = 5000, floor: float = -30.0):
        self.D, self.T, self.floor = D, T, floor

    @property
    def n(self) -> int:
        return 2 * self.D + 3

    def lookup(self, delta: np.ndarray):
        """(bin index, log-width correction to add, valid mask)."""
        d = np.asarray(delta)
        idx = np.clip(d, -self.D, self.D) + self.D + 1
        idx = np.where(d < -self.D, 0, np.where(d > self.D, self.n - 1, idx)).astype(np.int64)
        corr = np.where(np.abs(d) > self.D, -math.log(self.T - self.D), 0.0)
        valid = np.abs(d) <= self.T
        return idx, corr, valid

    def state(self):
        return dict(D=self.D, T=self.T, floor=self.floor)
