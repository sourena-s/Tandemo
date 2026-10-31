#!/usr/bin/env python
"""Candidate true repeat instability: loci where high-quality, confidently phased reads sit far from their own
allele (and not at the other allele), drawn as the outlier scan's two-panel figures (read lengths over the pileup).

Reads (all must hold):
  confident phase       >= --min-support (3) het SNVs / indels carry the read's haplotype and fewer than
                        --max-conflict-frac (20%) the other one (the scan's rule)
  far from its allele   |Y - L_own| > max(--min-delta, one unit), and not within max(5 bp, one unit) of the other allele
  clean read            MAPQ 60, flank identity >= --min-flank-id (read accuracy: the extraction's rq >= 0.99, and
                        network 1 takes rq as an input, so no further cut)
  surprising            log10 p(Y | L_own) < --threshold under network 1
Reads with supplementary alignments are kept and counted (sa_unstable, n_sa), marked "SA" in the figures.
Loci: an allele >= --min-allele bp (the long-allele tail set), motif >= --min-period bp, >= --min-reads spanning
reads, TRGT alleles supported by the reads (tandemo trgt-check). Ranked by recurrent reads (>= 2 at one locus
first), then by their number. Writes <out>.parquet (loci), <out>.reads.parquet (the reads) and the figures.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deviant_vntr as dv  # noqa: E402
from tandemo.data import assign_alleles, local_haplotypes, load, trgt_read_summary  # noqa: E402
from tandemo.data import TOLERANCE_HELP, TRGT_READ_TOLERANCE, UNTRUSTED_HELP, UNTRUSTED_TRGT  # noqa: E402
from tandemo.noise import load_model  # noqa: E402
from tandemo.phasing import PhaseParams, read_phase  # noqa: E402
from tandemo.store import read_table  # noqa: E402
from tandemo.extract import Params  # noqa: E402
from tandemo.genes import context as gene_context, for_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-allele", type=int, default=320)
    ap.add_argument("--min-period", type=int, default=2)
    ap.add_argument("--min-reads", type=int, default=12)
    ap.add_argument("--min-delta", type=int, default=10)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--max-conflict-frac", type=float, default=0.2)
    ap.add_argument("--min-flank-id", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=-3.5, help="log10 p(Y) below which a read deviates (-3.5)")
    ap.add_argument("--plot", type=int, default=16, help="loci drawn (16)")
    ap.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    ap.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--phased-vcf")
    ap.add_argument("--trgt-vcf")
    ap.add_argument("--sample")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--catalog", default="/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/"
                                         "chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz",
                    help="catalog of the extraction, for the pileup panels (Platinum TRs on T2T)")
    a = ap.parse_args()

    L = read_table(a.input, "loci")
    sel = L[(L.period >= a.min_period) & (L.al1 >= 0) & (L.al2 >= 0) & (L.pad_status == 0) & (L.n_span >= a.min_reads)
            & (np.maximum(L.al1, L.al2) >= a.min_allele)]
    print(f"{len(sel)} loci with an allele >= {a.min_allele} bp, motif >= {a.min_period} bp, >= {a.min_reads} "
          f"spanning reads", file=sys.stderr)
    reads, _ = load(a.input, loci_keep=set(sel.locus))
    reads = assign_alleles(reads, tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt)
    local = local_haplotypes(reads.attrs["trgt_read"])       # haplotypes with a read-assembled allele
    print(trgt_read_summary(reads.attrs["trgt_read"]), file=sys.stderr)
    reads = reads[reads["label_ok"]].copy()
    reads["w1_train"] = reads["w1"].fillna(0.5)
    pp = PhaseParams(min_support=a.min_support, max_conflict_frac=a.max_conflict_frac)
    reads["phase"] = read_phase(reads["n_hap1"], reads["n_hap2"], pp)[0]      # > 2 vars or <= 2 vars, none conflicting
    reads["n_support"] = np.where(reads["phase"] == 2, reads["n_hap2"], reads["n_hap1"])
    reads["n_conflict"] = reads["n_het"] - reads["n_support"]
    model = load_model(a.model)
    keep, rows, read_rows = {}, [], []
    for lid, g in reads.groupby("locus", sort=False):
        L1, L2, per = int(g.L1.iloc[0]), int(g.L2.iloc[0]), int(g.period.iloc[0])
        lc = model.log_curves(g, np.array([L1, L2])).astype(float) / np.log(10)
        ph = g.phase.to_numpy()
        Y = g.Y.to_numpy()
        own = np.where(ph == 2, L2, L1)
        oth = np.where(ph == 2, L1, L2)
        lp10 = np.where(ph == 2, lc[:, 1], lc[:, 0])
        d = Y - own
        units = d / max(per, 1)
        dev = ((ph > 0) & (np.abs(d) > max(a.min_delta, per)) & ~((np.abs(Y - oth) <= max(5, per)) & (L1 != L2))
               & (g.mapq.to_numpy() >= 60) & (g.flank_id_min.to_numpy() >= a.min_flank_id) & (lp10 < a.threshold))
        if not dev.any():
            continue
        keep[lid] = (g, lp10, dev)
        rows.append(dict(locus=lid, chrom=g.chrom.iloc[0], start=int(g.start.iloc[0]), end=int(g.end.iloc[0]),
                         motif=g.motif.iloc[0], period=per, L1=L1, L2=L2, reads=len(g),
                         confident=int((ph > 0).sum()), unstable=int(dev.sum()),
                         unstable_frac=float(dev.sum() / max((ph > 0).sum(), 1)),
                         hap1=int((dev & (ph == 1)).sum()), hap2=int((dev & (ph == 2)).sum()),
                         delta_units=",".join(f"{u:+.1f}" for u in sorted(units[dev])),
                         min_log10p=float(lp10[dev].min()),
                         segdup=float(g.segdup.iloc[0]) if "segdup" in g else float("nan"),
                         sa_unstable=int((g.n_sa.to_numpy()[dev] > 0).sum())))
        read_rows.append(pd.DataFrame(dict(locus=lid, read=g.read.to_numpy()[dev], phase=ph[dev], Y=Y[dev],
                                           L_own=own[dev], delta=d[dev], delta_units=units[dev],
                                           log10p=lp10[dev], rq=g.rq.to_numpy()[dev],
                                           n_support=g.n_support.to_numpy()[dev], n_sa=g.n_sa.to_numpy()[dev])))
    res = pd.DataFrame(rows)
    if res.empty:
        print("no candidate unstable reads", file=sys.stderr)
        return
    res["recurrent"] = res.unstable >= 2
    res = res.sort_values(["recurrent", "unstable", "min_log10p"], ascending=[False, False, True])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res.to_parquet(a.out + ".parquet", index=False)
    pd.concat(read_rows, ignore_index=True).to_parquet(a.out + ".reads.parquet", index=False)
    print(f"{len(res)} loci with candidate unstable reads, {res.recurrent.sum()} with >= 2; "
          f"{int(res.unstable.sum())} reads", file=sys.stderr)
    print(res.head(25).drop(columns=["chrom", "start", "end"]).to_string(index=False), file=sys.stderr)

    top = list(res.locus.head(a.plot))
    sub = {lid: keep[lid] for lid in top}
    cls = dv.classify(sub, a)
    plt = dv._style()
    genes = for_reference(a.reference)
    spikes = {}
    for lid in top:
        g, lp10, dev = keep[lid]
        c = cls.get(lid)
        k = c.loc[dev, "class"].mode().iloc[0] if c is not None and dev.any() else None
        spikes[lid] = dv.plot(plt, model, g, lp10, dev, a.out, c, k,
                              dv._title(genes, g))
    from tandemo.pileup import run as pileup
    pdir = a.out + "_pileups"
    pileup(a.bam, a.reference, a.catalog, pdir, a.phased_vcf, a.trgt_vcf, loci_names=top,
           regions=sorted({keep[x][0].chrom.iloc[0] for x in top}), sample=a.sample, max_reads=None,
           exclude_catalog=a.catalog, params=Params(),
           local=local, pads={x: int(keep[x][0].pad.iloc[0]) for x in top},
           labels={x: {1: int(keep[x][0].L1.iloc[0]), 2: int(keep[x][0].L2.iloc[0])} for x in top},
           highlight={x: set(keep[x][0].read.to_numpy()[keep[x][2]]) for x in top},
           log=lambda m: print(m, file=sys.stderr))
    for rank, lid in enumerate(top, 1):
        pu = os.path.join(pdir, f"{lid}.svg")
        if os.path.exists(pu):
            out = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"{rank:02d}.{lid}.panels.svg")
            dv.compose(spikes[lid], pu, out)
            print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
