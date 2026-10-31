"""Network 1 report: how the learned read likelihood p(Y | L, x) behaves, as figures and an HTML page.

Everything is computed on reads of held-out loci (the loci network1.train validated on), heterozygous and
homozygous alike, with each read's own features. Each read is compared with the predictive distribution training
optimised, the mixture over its locus' two alleles (TRGT alleles plus pad indels, on the read's Y scale),

    p(Y) = w1 p(Y | L1, x) + (1 - w1) p(Y | L2, x),     w1 = the read's phase weight (0.5 without evidence),

on the axis Delta = Y - L_ref, L_ref the allele the read more likely comes from; at a heterozygous locus the
other allele's term is shifted onto the same axis. Observed Delta of the same reads is drawn with every
prediction. Cells with fewer than MIN_LOCI held-out loci are marked as low-confidence.

  curves.svg        predicted (line) and observed (points) Delta by period class, one colour per allele length L
  summary.svg       heatmaps over period class x L: predicted and observed P(Delta = 0) and mean Delta
  features.svg      partial dependence: each feature set to its 5th ... 95th percentile for every read of a
                    reference stratum, predictions averaged
  importance.svg    change of the averaged P(Delta = 0) from the 5th to the 95th percentile of each feature
  expansion.svg     the same reads with both alleles scaled 0.5 ... 4x against an unchanged reference length
                    (a counterfactual: no observed points)
  calibration.svg   observed vs predicted per period class, with the empirical baseline when given
  index.html        the figures, numbers and a per-stratum held-out NLL table (the mixture likelihood)
"""

from __future__ import annotations

import html
import json
import os

import numpy as np
import pandas as pd

from .data import TRGT_READ_TOLERANCE, UNTRUSTED_TRGT, training_reads
from .features import raw_x
from .noise import load_model, stratum

PERIOD_CLASSES = [(1, 1, "1 bp"), (2, 2, "2 bp"), (3, 3, "3 bp"), (4, 4, "4 bp"), (5, 6, "5-6 bp"),
                  (7, 15, "7-15 bp"), (16, 50, "16-50 bp"), (51, 10_000, "> 50 bp")]
L_VALUES = [10, 20, 40, 80, 160, 320, 640, 1280]

# reference palette (dataviz skill): chrome, ordinal blue ramp, categorical slots, diverging poles
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _ordinal(n: int) -> list[str]:
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "lines.linewidth": 1.8, "legend.frameon": False})
    return plt


# ---------------------------------------------------------------- model access

class Probe:
    """p(Delta | L, x) over the model's bins for arbitrary raw features."""

    def __init__(self, model):
        import torch
        self.torch, self.m = torch, model
        self.D = max(model.bins.D, WINDOW)                  # display window: >= +-WINDOW bp whatever the core D
        self.delta = np.arange(-self.D, self.D + 1)

    def probs(self, X: np.ndarray, L, ref_len, period) -> np.ndarray:
        """X [n, n_features] raw; L, ref_len, period broadcast to [n, K]. -> [n, K, nbins] probabilities."""
        t = self.torch
        with t.no_grad():
            h = self.m.net.enc(t.as_tensor(self.m.scaler.x(X), device=self.m.dev))
            L = np.atleast_2d(np.asarray(L, dtype=float))
            L = np.broadcast_to(L, (len(X), L.shape[-1]))
            lf = self.m.scaler.l_feats(L, np.broadcast_to(np.asarray(ref_len, dtype=float), L.shape),
                                       np.broadcast_to(np.asarray(period, dtype=float), L.shape))
            out = []
            for i in range(0, len(X), 4096):
                lp = self.m.net.logp(h[i:i + 4096], t.as_tensor(lf[i:i + 4096], device=self.m.dev))
                out.append(lp.exp().cpu().numpy())
        return np.concatenate(out)

    def delta_probs(self, X: np.ndarray, L, ref_len, period, deltas) -> np.ndarray:
        """p(Delta) under the model's own tail law (Pareto or uniform), any Delta. X [n, n_features] raw;
        L, ref_len, period [n]; deltas [n, M] integers -> [n, M] probabilities."""
        t = self.torch
        L, ref_len, period = (np.asarray(v, dtype=float) for v in (L, ref_len, period))
        deltas = np.asarray(deltas)
        out = []
        with t.no_grad():
            for i in range(0, len(X), 4096):
                sl = slice(i, i + 4096)
                h = self.m.net.enc(t.as_tensor(self.m.scaler.x(X[sl]), device=self.m.dev))
                lf = self.m.scaler.l_feats(L[sl, None], ref_len[sl, None], period[sl, None])          # [b, 1, 3]
                lp, tp = self.m.net.dist(h, t.as_tensor(lf, device=self.m.dev))
                d = t.as_tensor(np.rint(deltas[sl])[:, None, :].astype(np.int64), device=self.m.dev)
                out.append(self.m.net.delta_logp(lp, tp, d, self.m.bins)[:, 0].exp().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, deltas.shape[-1]))

    def core(self, p: np.ndarray) -> np.ndarray:
        return p[..., 1:-1]                        # Delta = -D .. D

    def summary(self, p: np.ndarray, period: float) -> dict:
        c = self.core(p)
        d = self.delta
        cm = c.sum(-1)
        mean = (c * d).sum(-1) / cm
        sd = np.sqrt(np.clip((c * d ** 2).sum(-1) / cm - mean ** 2, 0, None))
        return dict(p0=c[..., self.D], p_unit=c[..., np.abs(d) <= max(period, 1)].sum(-1), mean=mean, sd=sd,
                    tail=p[..., 0] + p[..., -1])


def _period_class(period: np.ndarray) -> np.ndarray:
    out = np.full(len(period), -1)
    for i, (a, b, _) in enumerate(PERIOD_CLASSES):
        out[(period >= a) & (period <= b)] = i
    return out


WINDOW = 30            # figures span at least +-30 bp of Delta (network 1's core can be as narrow as 5 bp)
MIN_LOCI = 20           # cells with fewer held-out loci are drawn dashed / hatched as low-confidence


def held_out(df: pd.DataFrame, val_chroms=("chr20", "chr21", "chr22")) -> np.ndarray:
    """The training split of network1.train: the validation chromosomes when present, else 10% of loci by hash."""
    v = df["chrom"].isin(val_chroms).to_numpy()
    if v.sum() == 0 or v.all():
        v = (pd.util.hash_pandas_object(df["locus"], index=False).to_numpy() % 10) == 0
    return v


def reference_allele(df: pd.DataFrame):
    """(L_ref, L_other, w_ref) per read: the allele with the larger phase weight (L1 on ties)."""
    w = df["w1_train"].to_numpy(dtype=float)
    L1, L2 = df["L1"].to_numpy(dtype=float), df["L2"].to_numpy(dtype=float)
    one = w >= 0.5
    return np.where(one, L1, L2), np.where(one, L2, L1), np.where(one, w, 1 - w)


def predictive(probe, sub: pd.DataFrame, X: np.ndarray, scale: float = 1.0, ref_len=None, period=None):
    """Mixture predictive of each read on Delta = Y - L_ref over the display window (-probe.D .. probe.D), each
    allele under the model's own core and tail law, and the observed Delta. `scale` multiplies both alleles,
    `ref_len` / `period` replace the locus values (counterfactuals)."""
    D = probe.D
    Lr, Lo, wr = reference_allele(sub)
    Lr, Lo = Lr * scale, Lo * scale
    rl = sub["ref_len"].to_numpy(dtype=float) if ref_len is None else np.broadcast_to(ref_len, len(sub)).astype(float)
    pe = sub["period"].to_numpy(dtype=float) if period is None else np.broadcast_to(period, len(sub)).astype(float)
    win = np.arange(-D, D + 1)
    het = Lr != Lo
    shift = np.rint(Lr - Lo).astype(int)                   # Delta from L_other = Delta from L_ref + shift
    ref = probe.delta_probs(X, Lr, rl, pe, np.broadcast_to(win, (len(sub), len(win))))
    other = ref.copy()
    if het.any():                                          # the other allele on the L_ref axis
        other[het] = probe.delta_probs(X[het], Lo[het], rl[het], pe[het], win[None, :] + shift[het, None])
    P = wr[:, None] * ref + (1 - wr[:, None]) * other
    return P, sub["Y"].to_numpy() - Lr, Lr, (None, None, wr, shift)


def _observed(delta: np.ndarray, D: int) -> np.ndarray:
    d = np.rint(delta).astype(int)
    d = d[np.abs(d) <= D]
    return np.bincount(d + D, minlength=2 * D + 1) / max(len(delta), 1)


def _cells(df, pc, mask, rng, n_max=3000):
    """{(period class, L class): read indices} of held-out reads whose L_ref is within x1.6 of the L class."""
    Lr, _, _ = reference_allele(df)
    out = {}
    for i in range(len(PERIOD_CLASSES)):
        for j, L in enumerate(L_VALUES):
            idx = np.nonzero(mask & (pc == i) & (Lr >= L / 1.6) & (Lr <= L * 1.6))[0]
            if len(idx) >= 50:
                out[(i, j)] = rng.choice(idx, min(len(idx), n_max), replace=False)
    return out


def _span(period) -> int:
    return int(max(8, 3 * np.median(period) + 4))


# ---------------------------------------------------------------- figures

def fig_curves(plt, probe, df, X, pc, cells, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey=True)
    cols = _ordinal(len(L_VALUES))
    D = probe.D
    grid = {}
    for i, (_, _, name) in enumerate(PERIOD_CLASSES):
        ax = axes.flat[i]
        spans = []
        for j, L in enumerate(L_VALUES):
            if (i, j) not in cells:
                continue
            idx = cells[(i, j)]
            sub = df.iloc[idx]
            P, delta, _, _ = predictive(probe, sub, X[idx])
            pred, obs = P.mean(0), _observed(delta, D)
            n_loci = int(sub["locus"].nunique())
            grid[(i, j)] = dict(pred=pred, obs=obs, delta=delta, n_reads=len(idx), n_loci=n_loci,
                                period=float(np.median(sub["period"])))
            spans.append(_span(sub["period"]))
            few = n_loci < MIN_LOCI
            ax.plot(probe.delta, np.log10(np.clip(pred, 1e-12, 1)), color=cols[j], ls="--" if few else "-",
                    label=f"L ≈ {L} bp")
            ok = obs > 0
            ax.plot(probe.delta[ok], np.log10(obs[ok]), color=cols[j], marker="o", ms=2.5, lw=0)
        span = min(D, max(spans) if spans else 8)
        ax.set_xlim(-span, span); ax.set_ylim(-5, 0.1)
        ax.set_title(f"period {name}")
        ax.set_xlabel("Δ = Y − L_ref (bp)")
        if i % 4 == 0:
            ax.set_ylabel("log10 p(Δ), held-out reads")
    _legend(fig, axes, len(L_VALUES))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "curves")
    return grid


def fig_summary(plt, probe, grid, out):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    seq = LinearSegmentedColormap.from_list("blue", ["#cde2fb"] + BLUE_RAMP[2:])
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    D, d = probe.D, probe.delta
    nP, nL = len(PERIOD_CLASSES), len(L_VALUES)
    M = {k: np.full((nP, nL), np.nan) for k in ("p0_pred", "p0_obs", "mean_pred", "mean_obs", "n_loci")}
    for (i, j), c in grid.items():
        M["p0_pred"][i, j], M["p0_obs"][i, j] = c["pred"][D], c["obs"][D]
        M["mean_pred"][i, j] = (c["pred"] * d).sum() / c["pred"].sum()
        core = c["delta"][np.abs(c["delta"]) <= D]
        M["mean_obs"][i, j] = core.mean() if len(core) else np.nan
        M["n_loci"][i, j] = c["n_loci"]
    panels = [("p0_pred", "P(Δ = 0), predicted", seq), ("p0_obs", "P(Δ = 0), observed", seq),
              ("mean_pred", "mean Δ (bp), predicted", div), ("mean_obs", "mean Δ (bp), observed", div)]
    lim = np.nanmax(np.abs(np.concatenate([M["mean_pred"].ravel(), M["mean_obs"].ravel(), [1e-3]])))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (key, title, cmap) in zip(axes, panels):
        V = M[key]
        if key.startswith("mean"):
            im = ax.imshow(V, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        else:
            im = ax.imshow(V, cmap=cmap, vmin=np.nanmin(np.append(M["p0_obs"], M["p0_pred"])) if np.isfinite(V).any()
                           else 0, vmax=1, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(nL), [str(L) for L in L_VALUES], rotation=45)
        ax.set_yticks(range(nP), [c[2] for c in PERIOD_CLASSES])
        ax.set_xlabel("allele length L (bp)")
        ax.set_title(title)
        for i in range(nP):
            for j in range(nL):
                if np.isnan(V[i, j]):
                    continue
                few = M["n_loci"][i, j] < MIN_LOCI
                ax.text(j, i, f"{V[i, j]:.2f}" + ("*" if few else ""), ha="center", va="center", fontsize=6.5,
                        color=MUTED if few else INK2)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("motif period")
    fig.tight_layout()
    _save(fig, out, "summary")
    return M


def fig_features(plt, probe, df, X, pc, mask, rng, out, ref_class=(2, 4), ref_len=(30, 80), n_max=4000):
    """Partial dependence on held-out reads of a reference stratum (period 3 bp to 5-6 bp, L_ref 30-80 bp):
    one feature set to the same value for every read, the other features as they are, predictions averaged."""
    Lr, _, _ = reference_allele(df)
    m = mask & np.isin(pc, list(range(ref_class[0], ref_class[1] + 1))) & (Lr >= ref_len[0]) & (Lr <= ref_len[1])
    if m.sum() < 200:
        m = mask
    idx = rng.choice(np.nonzero(m)[0], min(int(m.sum()), n_max), replace=False)
    sub, Xs = df.iloc[idx], X[idx]
    qs = [5, 25, 50, 75, 95]
    cols = _ordinal(len(qs))
    D = probe.D
    nf = len(probe.m.features)
    rows_ = int(np.ceil(nf / 4))
    fig, axes = plt.subplots(rows_, 4, figsize=(15, 3 * rows_), sharey=True)
    eff = {}
    for k, name in enumerate(probe.m.features):
        ax = axes.flat[k]
        vals = np.nanpercentile(X[m][:, k], qs)
        p0s = []
        for q, v, c in zip(qs, vals, cols):
            X2 = Xs.copy(); X2[:, k] = v
            kw = {}
            if name == "period_log":
                kw["period"] = float(2 ** v)
            if name == "ref_len_log":                       # a longer locus: alleles scale with the reference
                kw["ref_len"] = float(np.expm1(v))
                kw["scale"] = float(np.expm1(v) / np.median(sub["ref_len"]))
            P = predictive(probe, sub, X2, **kw)[0].mean(0)
            p0s.append(float(P[D]))
            ax.plot(probe.delta, np.log10(np.clip(P, 1e-12, 1)), color=c, label=f"p{q}")
        eff[name] = (p0s[0], p0s[-1], float(vals[0]), float(vals[-1]))
        ax.set_xlim(-10, 10); ax.set_ylim(-6, 0.1)
        ax.set_title(name)
        if k % 4 == 0:
            ax.set_ylabel("log10 p(Δ), averaged")
        if k >= nf - 4:
            ax.set_xlabel("Δ = Y − L_ref (bp)")
    for ax in axes.flat[nf:]:
        ax.set_visible(False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, [f"feature at its {x} percentile" for x in l], loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, out, "features")

    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted(eff, key=lambda n: abs(eff[n][1] - eff[n][0]))
    dd = [eff[n][1] - eff[n][0] for n in order]
    ax.barh(range(len(order)), dd, color=[SERIES[0] if v >= 0 else SERIES[1] for v in dd], height=0.6)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("averaged P(Δ = 0), feature at 95th − 5th percentile")
    ax.set_title(f"Partial dependence, {len(idx):,} held-out reads at {sub['locus'].nunique():,} loci")
    fig.tight_layout()
    _save(fig, out, "importance")
    return eff, dict(n=len(idx), loci=int(sub["locus"].nunique()))


def fig_expansion(plt, probe, df, X, pc, mask, rng, out, n_max=3000):
    ratios = [0.5, 1, 1.5, 2, 4]
    cols = _ordinal(len(ratios))
    cases = [(0, 20), (1, 30), (2, 45), (3, 60), (4, 60), (5, 120)]
    Lr, _, _ = reference_allele(df)
    rl_all = df["ref_len"].to_numpy()
    fig, axes = plt.subplots(1, len(cases), figsize=(18, 3.8), sharey=True)
    for ax, (i, ref) in zip(axes, cases):
        m = mask & (pc == i) & (Lr >= ref / 1.6) & (Lr <= ref * 1.6) & (np.abs(Lr - rl_all) <= 2)
        if m.sum() < 50:
            ax.set_visible(False); continue
        idx = rng.choice(np.nonzero(m)[0], min(int(m.sum()), n_max), replace=False)
        sub = df.iloc[idx]
        for r, c in zip(ratios, cols):
            P = predictive(probe, sub, X[idx], scale=r)[0].mean(0)
            ax.plot(probe.delta, np.log10(np.clip(P, 1e-12, 1)), color=c, label=f"L = {r:g} × observed")
        span = min(probe.D, _span(sub["period"]))
        ax.set_xlim(-span, span); ax.set_ylim(-8, 0.1)
        ax.set_title(f"period {PERIOD_CLASSES[i][2]}, ~{ref} bp · {sub['locus'].nunique()} loci")
        ax.set_xlabel("Δ = Y − L_ref (bp)")
    axes[0].set_ylabel("log10 p(Δ), averaged")
    _legend(fig, axes, len(ratios))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, out, "expansion")


def _mix_nll(bins, p1, p2, w, y_minus_l1, y_minus_l2) -> float:
    """Mean negative log mixture likelihood of Y, the training objective, for a binned model with uniform tails
    (the empirical baseline; network 1 is scored with Probe.delta_probs and its own tail law)."""
    def lp(p, d):
        idx, corr, valid = bins.lookup(d)
        v = np.log(np.clip(p[np.arange(len(p)), idx], 1e-30, 1)) + corr
        return np.where(valid, v, bins.floor)
    a = lp(p1, y_minus_l1) + np.log(np.clip(w, 1e-12, 1))
    b = lp(p2, y_minus_l2) + np.log(np.clip(1 - w, 1e-12, 1))
    return float(-np.logaddexp(a, b).mean())


def fig_calibration(plt, probe, df, X, pc, mask, out, baseline=None, max_reads=60_000):
    """Every held-out read (heterozygous and homozygous loci): observed Delta vs the mean mixture predictive,
    per period class; NLL is the training objective on these reads."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey=True)
    rows = []
    D = probe.D
    for i, (_, _, name) in enumerate(PERIOD_CLASSES):
        ax = axes.flat[i]
        idx = np.nonzero(mask & (pc == i))[0]
        if len(idx) < 200:
            ax.set_visible(False); continue
        idx = rng.choice(idx, min(len(idx), max_reads), replace=False)
        sub = df.iloc[idx]
        P, delta, Lr, (pr, po, wr, shift) = predictive(probe, sub, X[idx])
        Lo = Lr - shift
        Y = sub["Y"].to_numpy()
        span = min(D, _span(sub["period"]))
        xs = np.arange(-span, span + 1)
        obs, pred = _observed(delta, D)[D - span:D + span + 1], P.mean(0)[D - span:D + span + 1]
        ok = obs > 0
        ax.plot(xs[ok], np.log10(obs[ok]), color=SERIES[1], marker="o", ms=3, lw=0, label="observed")
        ax.plot(xs, np.log10(np.clip(pred, 1e-12, 1)), color=SERIES[0], label="network 1")
        het = sub["L1"].to_numpy() != sub["L2"].to_numpy()
        rl, pe = sub["ref_len"].to_numpy(), sub["period"].to_numpy()
        lr = np.log(np.clip(probe.delta_probs(X[idx], Lr, rl, pe, (Y - Lr)[:, None])[:, 0], 1e-300, 1))
        lo = np.log(np.clip(probe.delta_probs(X[idx], Lo, rl, pe, (Y - Lo)[:, None])[:, 0], 1e-300, 1))
        nll_net = float(-np.logaddexp(lr + np.log(np.clip(wr, 1e-12, 1)), lo + np.log(np.clip(1 - wr, 1e-12, 1))).mean())
        row = dict(period=name, reads=len(idx), loci=int(sub["locus"].nunique()), het_reads=float(het.mean()),
                   nll_net=nll_net,
                   obs_exact=float((delta == 0).mean()), pred_exact=float(P[:, D].mean()))
        if baseline is not None:                           # its own bins, at any Delta (its core D may differ)
            per = sub["period"].to_numpy()
            b1, b2 = np.exp(baseline.logp[stratum(per, Lr)]), np.exp(baseline.logp[stratum(per, Lo)])
            row["nll_baseline"] = _mix_nll(baseline.bins, b1, b2, wr, Y - Lr, Y - Lo)
            rows_ = np.arange(len(sub))[:, None]

            def at(b, d):                                  # baseline p(Delta = d) per read, d [n, G]
                ii, cc, vv = baseline.bins.lookup(d)
                return np.where(vv, b[rows_, ii] * np.exp(cc), 0.0)
            pb = (wr[:, None] * at(b1, np.broadcast_to(xs, (len(sub), len(xs))))
                  + (1 - wr[:, None]) * at(b2, xs[None, :] + shift[:, None])).mean(0)
            ax.plot(xs, np.log10(np.clip(pb, 1e-12, 1)), color=SERIES[2], label="empirical baseline")
        rows.append(row)
        ax.set_xlim(-span, span); ax.set_ylim(-6, 0.1)
        ax.set_title(f"period {name} · {len(idx):,} reads, {het.mean():.0%} het")
        ax.set_xlabel("Δ = Y − L_ref (bp)")
        if i % 4 == 0:
            ax.set_ylabel("log10 frequency")
    _legend(fig, axes, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "calibration")
    return rows


def _legend(fig, axes, ncol):
    """One legend for the figure, from the panel with the most series (panels may be empty)."""
    best = max(axes.flat, key=lambda a: len(a.get_legend_handles_labels()[0]) if a.get_visible() else -1)
    h, l = best.get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=ncol)
    else:
        fig.text(0.5, 0.5, "too few reads in every stratum", ha="center", color=MUTED)


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".svg"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------- driver

def run(model_path: str, prefixes: list[str], out: str, baseline_path: str | None = None,
        max_reads: int = 2_000_000, per_stratum: int | None = None, seed: int = 0, log=print,
        tolerance: float = TRGT_READ_TOLERANCE, untrusted: str = UNTRUSTED_TRGT) -> str:
    plt = _style()
    os.makedirs(out, exist_ok=True)
    model = load_model(model_path)
    if not hasattr(model, "net"):
        raise SystemExit("report needs a network 1 model (.pt)")
    baseline = load_model(baseline_path) if baseline_path else None
    log("loading reads")
    df = training_reads(prefixes, per_stratum=per_stratum, log=log, tolerance=tolerance, untrusted=untrusted)
    if len(df) > max_reads:
        df = df.sample(max_reads, random_state=seed).reset_index(drop=True)
    X = raw_x(df, model.features)
    pc = _period_class(df["period"].to_numpy())
    mask = held_out(df)
    log(f"{int(mask.sum())} held-out reads at {df.loc[mask, 'locus'].nunique()} loci")
    rng = np.random.default_rng(seed)
    probe = Probe(model)
    cells = _cells(df, pc, mask, rng)
    log("curves"); grid = fig_curves(plt, probe, df, X, pc, cells, out)
    log("summary"); table = fig_summary(plt, probe, grid, out)
    log("features"); eff, ref = fig_features(plt, probe, df, X, pc, mask, rng, out)
    log("expansion"); fig_expansion(plt, probe, df, X, pc, mask, rng, out)
    log("calibration"); cal = fig_calibration(plt, probe, df, X, pc, mask, out, baseline)
    meta = dict(model=model_path, inputs=prefixes, reads=len(df), loci=int(df["locus"].nunique()),
                held_out_reads=int(mask.sum()), held_out_loci=int(df.loc[mask, "locus"].nunique()),
                features=model.features,
                **{k: v for k, v in model.meta.items() if k in ("val_nll", "n_train", "n_val", "tail", "meth_mask")})
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(dict(meta=meta, calibration=cal, feature_effects={k: [float(x) for x in v] for k, v in eff.items()},
                       summary={k: np.where(np.isnan(v), None, v).tolist() for k, v in table.items()}), fh, indent=1)
    return _html(out, meta, cal, ref)


def _html(out, meta, cal, ref) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(r['period'])}</td><td>{r['reads']:,}</td><td>{r['loci']:,}</td>"
        f"<td>{r['obs_exact']:.3f}</td><td>{r['pred_exact']:.3f}</td><td>{r['nll_net']:.3f}</td>"
        f"<td>{r.get('nll_baseline', float('nan')):.3f}</td></tr>" for r in cal)
    figs = [("curves", "Read-error distributions by motif period",
             "Held-out reads (heterozygous and homozygous loci), binned by the length of the allele each read "
             "more likely comes from: the mean mixture predictive (lines) and the observed Δ = Y − L_ref of the "
             f"same reads (points). Dashed: fewer than {MIN_LOCI} held-out loci."),
            ("summary", "Summary over period × length",
             "Predicted and observed P(Δ = 0) and mean Δ (diverging, grey = 0) of the same held-out reads. "
             f"Values marked * come from fewer than {MIN_LOCI} loci; empty cells: fewer than 50 reads."),
            ("features", "What moves the curve",
             f"Partial dependence on {ref['n']:,} held-out reads at {ref['loci']:,} loci (period 3 to 6 bp, "
             "allele 30-80 bp): each feature set to its 5th to 95th percentile for every read, the other features "
             "as they are, predictions averaged."),
            ("importance", "Feature effects on exact-length reads",
             "Change of the averaged P(Δ = 0) between the 95th and 5th percentile of each feature."),
            ("expansion", "Alleles away from the reference",
             "The same kind of held-out reads (alleles at the reference length) with both alleles scaled 0.5 to 4 "
             "times and the reference unchanged: the network's extrapolation, so there are no observed points."),
            ("calibration", "Calibration on held-out loci",
             "Every held-out read: observed Δ (points) against the mean mixture predictive of the same reads "
             "(line), and the empirical baseline. NLL in the table is the training objective on these reads.")]
    body = "".join(f"<h2>{t}</h2><p>{html.escape(c)}</p><img src='{n}.svg' alt='{html.escape(t)}'>" for n, t, c in figs)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Network 1 report</title>
<style>body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px auto;max-width:1400px;color:#0b0b0b;
background:#f9f9f7;padding:0 16px}}img{{max-width:100%;background:#fcfcfb;border:1px solid #e1e0d9}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}td,th{{padding:3px 10px;border-bottom:1px solid #e1e0d9;
text-align:right}}td:first-child,th:first-child{{text-align:left}}p{{color:#52514e;max-width:80ch}}</style></head><body>
<h1>Network 1 report</h1><p>Model {html.escape(meta['model'])} · {meta['reads']:,} reads at {meta['loci']:,} loci
(sampled), of which {meta['held_out_reads']:,} reads at {meta['held_out_loci']:,} held-out loci · validation NLL {meta.get('val_nll', float('nan')):.4f} · {meta.get('n_train', 0):,} training / {meta.get('n_val', 0):,}
validation reads</p>
<h2>Held-out calibration</h2><table><tr><th>period</th><th>reads</th><th>loci</th><th>observed P(Δ=0)</th>
<th>predicted P(Δ=0)</th><th>NLL network 1</th><th>NLL baseline</th></tr>{rows}</table>{body}</body></html>"""
    path = os.path.join(out, "index.html")
    with open(path, "w") as fh:
        fh.write(doc)
    return path
