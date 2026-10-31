"""Tails of the per-read error Delta = Y - L_ref at the longest alleles: observed vs network 1.

For reads at alleles >= 320 bp (TRGT allele + pad indels), Delta is measured from the allele the read more likely
comes from (phase weight), and network 1's predictive is the two-allele mixture with each read's own features
(as in tandemo report). Panels: allele length class x motif size; log scale out to +-SPAN bp. Prints tail masses.

    python scripts/long_tails.py --input ex/HG002.chr4.pad --model net1.chr4.pad.pt --out long_tails
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tandemo.data import assign_alleles, load  # noqa: E402
from tandemo.data import TOLERANCE_HELP, TRGT_READ_TOLERANCE, UNTRUSTED_HELP, UNTRUSTED_TRGT  # noqa: E402
from tandemo.noise import load_model  # noqa: E402
from tandemo.store import read_table  # noqa: E402
from tandemo.features import raw_x  # noqa: E402
from tandemo.report import SERIES, Probe, _save, _style, held_out, predictive, reference_allele  # noqa: E402

LENGTHS = [(320, 640, "320-640 bp"), (640, 1280, "640-1280 bp"), (1280, 1e9, "> 1280 bp")]
MOTIFS = [(1, 1, "homopolymers (1 bp)"), (2, 6, "motif 2-6 bp"), (7, 10_000, "motif >= 7 bp (VNTR)")]
SPAN = 150
FINE = 10              # per-bp resolution within +-FINE bp; beyond, BIN-bp bins (per-bp density in each bin)
BIN = 10


def binned(xs: np.ndarray, per_bp: np.ndarray):
    """(x positions, per-bp density): per bp within +-FINE, the mean over BIN-bp bins beyond."""
    xo, yo = [], []
    for x, v in zip(xs, per_bp):
        if abs(x) <= FINE:
            xo.append(x); yo.append(v)
    for side in (-1, 1):
        for a in range(FINE + 1, SPAN + 1, BIN):
            m = (np.abs(xs) >= a) & (np.abs(xs) < a + BIN) & (np.sign(xs) == side)
            if m.any():
                xo.append(side * (a + BIN / 2 - 0.5)); yo.append(per_bp[m].mean())
    o = np.argsort(xo)
    return np.asarray(xo)[o], np.asarray(yo)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    ap.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-reads", type=int, default=4000, help="reads per panel for the prediction (4000)")
    a = ap.parse_args()
    L = read_table(a.input, "loci", ["locus", "al1", "al2", "pad_d1", "pad_d2", "pad_status", "n_span"])
    L = L[(L.al1 >= 0) & (L.al2 >= 0) & (L.pad_status == 0) & (L.n_span > 0)
          & (np.maximum(L.al1 + L.pad_d1, L.al2 + L.pad_d2) >= LENGTHS[0][0])]
    reads, _ = load(a.input, loci_keep=set(L.locus))
    reads = assign_alleles(reads, tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt)
    reads = reads[reads["label_ok"]].copy()            # TRGT allele not trusted and not assembled: left out
    reads["w1_train"] = reads["w1"].fillna(0.5)
    Lr, Lo, wr = reference_allele(reads)
    reads["L_ref"], reads["L_oth"], reads["w_ref"] = Lr, Lo, wr
    reads["delta"] = reads["Y"] - reads["L_ref"]
    reads["held_out"] = held_out(reads)
    model = load_model(a.model)
    probe = Probe(model)
    rng = np.random.default_rng(0)
    plt = _style()
    fig, axes = plt.subplots(len(MOTIFS), len(LENGTHS), figsize=(16, 11), sharex=True, sharey=True)
    xs = np.arange(-SPAN, SPAN + 1)
    rows = []
    for i, (m0, m1, mname) in enumerate(MOTIFS):
        for j, (l0, l1, lname) in enumerate(LENGTHS):
            ax = axes[i, j]
            sub = reads[(reads.period >= m0) & (reads.period <= m1) & (reads.L_ref >= l0) & (reads.L_ref < l1)]
            if len(sub) < 50:
                ax.set_visible(False)
                continue
            d = sub["delta"].to_numpy()
            obs = np.array([(d == x).mean() for x in xs])
            # predicted: each read's two-allele mixture at Delta = x from L_ref (x + shift from the other allele),
            # each allele under the model's own tail law
            pick = sub.iloc[rng.choice(len(sub), min(len(sub), a.max_reads), replace=False)]
            Xp, rl, pe = raw_x(pick, model.features), pick.ref_len.to_numpy(), pick.period.to_numpy()
            Lr_, Lo_, w = pick.L_ref.to_numpy(), pick.L_oth.to_numpy(), pick.w_ref.to_numpy()
            shift = np.rint(Lr_ - Lo_).astype(int)
            p_ref = probe.delta_probs(Xp, Lr_, rl, pe, np.broadcast_to(xs, (len(pick), len(xs))))
            p_oth = probe.delta_probs(Xp, Lo_, rl, pe, xs[None, :] + shift[:, None])
            pred = (w[:, None] * p_ref + (1 - w[:, None]) * p_oth).mean(0)
            ok = obs > 0
            bx_, bo = binned(xs, obs)
            _, bp = binned(xs, pred)
            ok = bo > 0
            ax.plot(bx_[ok], np.log10(bo[ok]), color=SERIES[1], marker="o", ms=3.2, lw=0, label="observed reads")
            ax.plot(bx_, np.log10(np.clip(bp, 1e-12, 1)), color=SERIES[0], lw=1.6, label="network 1 (mixture)")
            one = 1 / len(sub)                              # the smallest observable fraction: one read
            ax.axhline(np.log10(one), color="#898781", lw=0.9, ls="--", label="one read (1/N)")
            ax.axhline(np.log10(one / BIN), color="#898781", lw=0.6, ls=":", label="one read in a 10 bp bin")
            ax.axvspan(-SPAN, -model.bins.D, color="#e1e0d9", alpha=0.35, lw=0)                 # tail region
            ax.axvspan(model.bins.D, SPAN, color="#e1e0d9", alpha=0.35, lw=0)
            n_loci = sub.locus.nunique()
            ax.set_title(f"{mname} · alleles {lname}\n{len(sub):,} reads, {n_loci:,} loci "
                         f"({sub.held_out.mean():.0%} held out)", fontsize=9)
            ax.set_ylim(-6, 0.05)
            if i == len(MOTIFS) - 1:
                ax.set_xlabel("Δ = Y − L_ref (bp)")
            if j == 0:
                ax.set_ylabel("log10 fraction of reads per bp")
            row = dict(motif=mname, alleles=lname, reads=len(sub), loci=n_loci)
            for t in (1, 5, 10, 25, 50):
                row[f"obs |Δ|>{t}"] = float((np.abs(d) > t).mean())
                row[f"pred |Δ|>{t}"] = float(pred[np.abs(xs) > t].sum() + max(0.0, 1 - pred.sum()))
            row["obs |Δ|>150"] = float((np.abs(d) > SPAN).mean())
            rows.append(row)
    h, l = next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
    fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color="#e1e0d9")], l + [f"beyond ±D: network tail ({model.net.tail})"],
               loc="upper center", ncol=5)
    fig.text(0.5, 0.005, f"per bp within ±{FINE} bp; beyond, the mean per bp over {BIN} bp bins, so single reads and "
             "the prediction are compared on the same footing", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, os.path.dirname(os.path.abspath(a.out)) or ".", os.path.basename(a.out))
    t = pd.DataFrame(rows)
    t.to_parquet(a.out + ".parquet", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
