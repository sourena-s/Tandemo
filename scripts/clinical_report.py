#!/usr/bin/env python
"""Report of the disease-associated tandem repeats (the genome's pathogenic catalog, 68 Platinum loci on T2T):
one page per locus with the outlier scan's two panels (read lengths against network 1, over the pileup), after a
summary table. Input: any extraction that includes those loci (genome-wide Platinum keeps them through every
catalog filter; only their reads are loaded) and network 1.

Writes <out>.summary.svg, <out>.<locus>.panels.svg, <out>.pages.txt (page order: summary, then loci by
chromosome) and <out>.summary.tsv. Join the pages into one PDF (each page at its own size) with
    bash scripts/pages_to_pdf.sh <out>.pages.txt <out>.clinical_repeats.pdf
Every locus is shown, whatever trgt-check says; its status is in the table.
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
from tandemo.data import CORRECTED, EXCLUDED, LOCAL, assign_alleles, local_haplotypes, load  # noqa: E402
from tandemo.data import TOLERANCE_HELP, TRGT_READ_TOLERANCE, UNTRUSTED_HELP, UNTRUSTED_TRGT  # noqa: E402
from tandemo.extract import Params  # noqa: E402
from tandemo.genes import context as gene_context, for_reference  # noqa: E402
from tandemo.genomes import resource  # noqa: E402
from tandemo.noise import load_model  # noqa: E402
from tandemo.phasing import PhaseParams, read_phase  # noqa: E402
from tandemo.pileup import _label, run as pileup  # noqa: E402

CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="extraction prefixes (e.g. one per chromosome)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--catalog", default=resource("t2t", "pathogenic"))
    ap.add_argument("--phased-vcf")
    ap.add_argument("--trgt-vcf")
    ap.add_argument("--sample")
    ap.add_argument("--threshold", type=float, default=-3.5, help="log10 p(Y) below which a read deviates (-3.5)")
    ap.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    ap.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)
    a = ap.parse_args()

    want = set(_catalog_ids(a.catalog))                 # only the catalog's loci, from every input
    parts = [load(p, loci_keep=want)[0] for p in a.input]
    reads = pd.concat([x for x in parts if len(x)], ignore_index=True)
    reads = assign_alleles(reads, tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt)
    local = local_haplotypes(reads.attrs["trgt_read"])       # haplotypes with a read-assembled allele
    rec = reads.attrs["trgt_read"]
    pp = PhaseParams()
    reads["phase"] = read_phase(reads["n_hap1"], reads["n_hap2"], pp)[0]      # > 2 vars or <= 2 vars, none conflicting
    reads["w1_train"] = reads["w1"].fillna(0.5)
    model = load_model(a.model)
    genes = for_reference(a.reference)
    plt = dv._style()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    rows, keep, spikes = [], {}, {}
    for lid, g in reads.groupby("locus", sort=False):
        L1, L2, per = int(g.L1.iloc[0]), int(g.L2.iloc[0]), int(g.period.iloc[0])
        if L1 < 0 or L2 < 0:
            continue
        lc = model.log_curves(g, np.array([L1, L2])).astype(float)
        w = g.w1_train.to_numpy()
        mix = np.logaddexp(lc[:, 0] + np.log(np.clip(w, 1e-12, 1)), lc[:, 1] + np.log(np.clip(1 - w, 1e-12, 1)))
        ph = g.phase.to_numpy()
        lp10 = np.where(ph == 1, lc[:, 0], np.where(ph == 2, lc[:, 1], mix)) / np.log(10)
        cons10 = lc.max(1) / np.log(10)
        far = np.minimum(np.abs(g.Y - L1), np.abs(g.Y - L2)).to_numpy() >= per
        dev = (lp10 < a.threshold) & (cons10 < a.threshold) & far
        keep[lid] = (g, lp10, dev)
        ctx = gene_context(genes, g.chrom.iloc[0], int(g.start.iloc[0]), int(g.end.iloc[0]))
        spikes[lid] = dv.plot(plt, model, g, lp10, dev, a.out, None, None, dv._title(genes, g))
        r = rec[rec.locus == lid]
        status = ("excluded: TRGT allele not trusted" if (r.action == EXCLUDED).any() else
                  "local allele from the reads" if (r.action == LOCAL).any() else
                  "pad indel corrected" if (r.action == CORRECTED).any() else "")
        Y = g.Y.to_numpy()
        med = {h: float(np.median(Y[ph == h])) if (ph == h).any() else np.nan for h in (1, 2)}
        rows.append(dict(locus=lid, gene_disease=_label(lid), chrom=g.chrom.iloc[0], start=int(g.start.iloc[0]),
                         end=int(g.end.iloc[0]), gene_context=ctx, motif=g.motif.iloc[0], period=per,
                         ref_len=int(g.ref_len.iloc[0]), L1=L1, L2=L2, units1=L1 / per, units2=L2 / per,
                         median1=med[1], median2=med[2], reads=len(g), phased=int((ph > 0).sum()),
                         outliers=int(dev.sum()), status=status,
                         segdup=float(g.segdup.iloc[0]) if "segdup" in g else float("nan"),
                         sa_reads=int((g.n_sa.to_numpy() > 0).sum())))
    res = pd.DataFrame(rows)
    res["c"] = res.chrom.map({c: i for i, c in enumerate(CHROMS)}).fillna(99)
    res = res.sort_values(["c", "start"]).drop(columns="c").reset_index(drop=True)
    res.to_csv(a.out + ".summary.tsv", sep="\t", index=False, float_format="%.1f")

    lids = list(res.locus)
    pdir = a.out + "_pileups"
    pileup(a.bam, a.reference, a.catalog, pdir, a.phased_vcf, a.trgt_vcf, loci_names=lids,
           regions=sorted(set(res.chrom)), sample=a.sample, max_reads=None, exclude_catalog=a.catalog,
           params=Params(), highlight={x: set(keep[x][0].read.to_numpy()[keep[x][2]]) for x in lids},
           local=local, pads={x: int(keep[x][0].pad.iloc[0]) for x in lids},
           labels={x: {1: int(keep[x][0].L1.iloc[0]), 2: int(keep[x][0].L2.iloc[0])} for x in lids},
           log=lambda m: print(m, file=sys.stderr))
    pages = [summary_page(plt, res, a.out, a.sample or "")]
    for lid in lids:
        pu = os.path.join(pdir, f"{lid}.svg")
        if os.path.exists(pu):
            out = f"{a.out}.{lid}.panels.svg"
            dv.compose(spikes[lid], pu, out)
            pages.append(out)
        else:
            pages.append(spikes[lid])
    missing = sorted(set(_catalog_ids(a.catalog)) - set(lids))
    with open(a.out + ".pages.txt", "w") as fh:
        fh.write("\n".join(os.path.abspath(x) for x in pages) + "\n")     # absolute: usable from anywhere
    print(f"{len(lids)} loci with spanning reads; without: {', '.join(missing) or 'none'}", file=sys.stderr)
    print(f"pages in {a.out}.pages.txt; PDF: bash scripts/pages_to_pdf.sh {a.out}.pages.txt "
          f"{a.out}.clinical_repeats.pdf", file=sys.stderr)


def _catalog_ids(path):
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return [ln.split("\t")[3].split(";")[0].removeprefix("ID=") for ln in fh if ln.strip()]


def summary_page(plt, res: pd.DataFrame, out: str, sample: str) -> str:
    """The table: one row per locus, alleles as bp and motif units, read medians per haplotype, outliers."""
    fmt = lambda x: "–" if pd.isna(x) else f"{x:.0f}"
    cells = [[r.gene_disease, f"{r.chrom}:{r.start + 1:,}", r.gene_context[:34], (r.motif[:14] + "…") if len(r.motif) > 15 else r.motif,
              str(r.ref_len), f"{r.L1} ({r.units1:.1f})", f"{r.L2} ({r.units2:.1f})", fmt(r.median1), fmt(r.median2),
              f"{r.phased}/{r.reads}", str(r.sa_reads), "–" if not r.segdup > 0 else f"{r.segdup:.0%}",
              str(r.outliers), r.status[:22]] for r in res.itertuples()]
    cols = ["gene (disease)", "T2T position", "gene context", "motif", "ref bp", "hap 1 bp (units)",
            "hap 2 bp (units)", "hap 1 median", "hap 2 median", "phased / reads", "SA reads", "segdup", "outliers",
            "TRGT-read check"]
    h = 1.4 + 0.235 * (len(cells) + 1)                # rows at font 7.5 and scale 1.18, with the title
    fig = plt.figure(figsize=(15, h))
    ax = fig.add_axes([0.01, 0.01, 0.98, 1 - 1.0 / h])
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="left", colLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.18)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#d9d8d3")
        c.set_linewidth(0.5)
        if i == 0:
            c.set_text_props(weight="bold", color="#0b0b0b")
            c.set_facecolor("#eeede8")
        elif cells[i - 1][12] != "0":
            c.set_facecolor("#fbe9e5")
    t.auto_set_column_width(list(range(len(cols))))
    fig.suptitle(f"{sample} · disease-associated tandem repeats ({len(res)} loci with spanning reads) · "
                 "alleles: TRGT, pad-corrected (motif units); medians: confidently phased reads; "
                 "outliers: −log10 p > 3.5 under network 1 and ≥ 1 unit from both alleles", fontsize=9, x=0.01,
                 ha="left", y=1 - 0.35 / h)
    name = out + ".summary.svg"
    fig.savefig(name)
    plt.close(fig)
    return name


if __name__ == "__main__":
    main()
