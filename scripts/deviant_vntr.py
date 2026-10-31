"""Heterozygous VNTR loci with reads the network 1 mixture cannot explain, and a figure of the clearest ones.

For every read at a heterozygous VNTR locus (motif >= 7 bp, L1 != L2 after the pad correction, pad unambiguous)
two scores of its length Y under network 1:
  phase-based   a confidently phased read (>= --phase-min-support het SNVs / indels carry one haplotype's alleles
                and fewer than --phase-max-conflict-frac of its informative sites carry the other's) is scored
                against its own haplotype's allele only,
                p(Y | L_h, x); a read with uncertain phase (fewer supporting sites, more conflicts, or none)
                uses the phase-weighted mixture w1 p(Y | L1, x) + (1 - w1) p(Y | L2, x)
  phase-free    max(p(Y | L1, x), p(Y | L2, x)): the allele that explains the read best, whatever its phase
A read deviates when the phase-based score is below --threshold and, unless --no-conservative, the phase-free one
is too, so a phase error alone never makes a read deviant. Both are reported. Loci are
ranked by the number of deviant reads that sit at least one motif unit from both alleles, among loci where most
reads fit. The figures plot -log10 p(Y), so outliers rise above the threshold line.

With --bam, each read's repeat sequence (the bases Y counts) is compared with the reference repeat: a read base is
"ref-like" when an 8-mer of the reference repeat covers it, else "novel" (a point difference costs one base, so
VNTR unit variation stays ref-like). Deviant reads are then classed as
  contraction                 shorter than both alleles
  expansion, ref-like units   longer, and at most a quarter of the extra length is novel (beyond its allele)
  insertion, novel repeat     a novel block that is itself periodic (its unit and period are reported)
  insertion, non-repetitive   a novel block without periodicity
and reads of normal length are flagged as composition outliers when they carry >= max(10 bp, 2 units) more
novel sequence than the median non-deviant read of the same allele (so germline allele content, e.g. an
insertion TRGT genotyped, is not an outlier).

Writes <out>.parquet (loci with deviant or composition-outlier reads), <out>.reads.parquet (those reads, with class,
novel bases and novel unit) and <out>.<locus>.png/.svg for the clearest locus of each class (--plot per class).

    python scripts/deviant_vntr.py --input ex/HG002.chr4.pad --model net1.chr4.pad.pt --out deviant/chr4 \
        --bam HG002.haplotagged.bam --reference chm13v2.0_maskedY_rCRS.fa
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import pysam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tandemo.aln import Aln  # noqa: E402
from tandemo.data import assign_alleles, local_haplotypes, load, trgt_read_summary  # noqa: E402
from tandemo.data import TOLERANCE_HELP, TRGT_READ_TOLERANCE, UNTRUSTED_HELP, UNTRUSTED_TRGT  # noqa: E402
from tandemo.extract import Params, measured_period, self_match, usable  # noqa: E402
from tandemo.noise import load_model  # noqa: E402
from tandemo.phasing import PhaseParams, read_phase  # noqa: E402
from tandemo.store import read_table  # noqa: E402
from tandemo.genes import context as gene_context, for_reference  # noqa: E402
from tandemo.pileup import HAP_COLOUR as HAP_COLOURS  # noqa: E402
from tandemo.report import _save, _style  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="extraction prefix (with pad columns)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-period", type=int, default=7)
    ap.add_argument("--min-reads", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=-3.5, help="log10 p(Y) below which a read deviates (-3.5)")
    ap.add_argument("--plot", type=int, default=2, help="loci plotted per class (2)")
    ap.add_argument("--phase-min-support", type=int, default=3,
                    help="confident phase: at least this many het SNVs / indels carry the read's haplotype (3)")
    ap.add_argument("--phase-max-conflict-frac", type=float, default=0.2,
                    help="confident phase: fewer than this fraction of the read's informative het sites carry the "
                         "other haplotype's allele (0.2: 5:1 is confident, 4:1 is not)")
    ap.add_argument("--no-conservative", action="store_true",
                    help="flag on the phase-weighted mixture alone (default: also require the phase-free score, "
                         "the better of the two alleles, below the threshold)")
    ap.add_argument("--bam", help="the extraction's BAM: classify outliers by repeat sequence")
    ap.add_argument("--reference", help="FASTA (needed with --bam)")
    ap.add_argument("--k", type=int, default=8, help="k-mer length for ref-like bases (8)")
    ap.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    ap.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)
    ap.add_argument("--catalog", default="/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/"
                                         "chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz",
                    help="catalog of the extraction, for the pileup panels (Platinum TRs on T2T)")
    ap.add_argument("--phased-vcf", help="HiPhase phased VCF, for the pileup panels' phase and het markers")
    ap.add_argument("--trgt-vcf", help="TRGT VCF, for the pileup panels' allele labels")
    ap.add_argument("--sample")
    ap.add_argument("--composition", action="store_true",
                    help="also fetch sequences at loci without length-deviant reads, for normal-length composition "
                         "outliers (default: sequences only where length already flags a read)")
    a = ap.parse_args()

    L = read_table(a.input, "loci")
    het = L[(L.period >= a.min_period) & (L.al1 >= 0) & (L.al2 >= 0) & (L.pad_status == 0) & (L.n_span >= a.min_reads)
            & (L.al1 + L.pad_d1 != L.al2 + L.pad_d2)]
    print(f"{len(het)} heterozygous VNTR loci (motif >= {a.min_period} bp, >= {a.min_reads} spanning reads)",
          file=sys.stderr)
    reads, _ = load(a.input, loci_keep=set(het.locus))
    if "n_hap1" not in reads:
        raise SystemExit("this extraction has no per-haplotype site counts (n_hap1 / n_hap2): re-run tandemo extract")
    reads = assign_alleles(reads, tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt)
    local = local_haplotypes(reads.attrs["trgt_read"])       # haplotypes with a read-assembled allele
    print(f"{trgt_read_summary(reads.attrs['trgt_read'])} (of {reads['locus'].nunique()} heterozygous VNTR loci)",
          file=sys.stderr)
    reads = reads[reads["label_ok"]].copy()
    reads["w1_train"] = reads["w1"].fillna(0.5)
    pp = PhaseParams(min_support=a.phase_min_support, max_conflict_frac=a.phase_max_conflict_frac)
    reads["phase"] = read_phase(reads["n_hap1"], reads["n_hap2"], pp)[0]      # > 2 vars or <= 2 vars, none conflicting
    reads["n_support"] = np.where(reads["phase"] == 2, reads["n_hap2"], reads["n_hap1"])
    reads.loc[reads["phase"] == 0, "n_support"] = np.maximum(reads["n_hap1"], reads["n_hap2"])
    reads["n_conflict"] = reads["n_het"] - reads["n_support"]
    print(f"{(reads.phase > 0).mean():.1%} of reads confidently phased (>= {pp.min_support} supporting het sites, "
          f"< {pp.max_conflict_frac:.0%} conflicting)", file=sys.stderr)
    model = load_model(a.model)
    rows, keep, cons = [], {}, {}
    for lid, g in reads.groupby("locus", sort=False):
        L1, L2 = int(g.L1.iloc[0]), int(g.L2.iloc[0])
        lc = model.log_curves(g, np.array([L1, L2])).astype(float)          # [n, 2], natural log
        w = g.w1_train.to_numpy()
        mix = np.logaddexp(lc[:, 0] + np.log(np.clip(w, 1e-12, 1)), lc[:, 1] + np.log(np.clip(1 - w, 1e-12, 1)))
        ph = g.phase.to_numpy()                             # confident: own allele only; uncertain: the mixture
        lp = np.where(ph == 1, lc[:, 0], np.where(ph == 2, lc[:, 1], mix))
        lp10 = lp / np.log(10)
        cons10 = lc.max(1) / np.log(10)                     # phase-free: the allele that explains the read best
        per = int(g.period.iloc[0])
        far = np.minimum(np.abs(g.Y - L1), np.abs(g.Y - L2)).to_numpy() >= per
        dev = (lp10 < a.threshold) & far
        if not a.no_conservative:
            dev &= cons10 < a.threshold
        cons[lid] = cons10
        keep[lid] = (g, lp10, dev)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # stage 2 (sequence) only where stage 1 (length) flags a read, unless --composition
    todo = keep if a.composition else {lid: v for lid, v in keep.items() if v[2].any()}
    if a.bam:
        print(f"sequence classification at {len(todo)} loci", file=sys.stderr)
    cls = classify(todo, a) if a.bam else {}
    rows, read_rows = [], []
    for lid, (g, lp10, dev) in keep.items():
        c = cls.get(lid)
        comp = c["class"].str.startswith("composition").to_numpy() if c is not None else np.zeros(len(g), bool)
        if not dev.any() and not comp.any():
            continue
        L1, L2, per = int(g.L1.iloc[0]), int(g.L2.iloc[0]), int(g.period.iloc[0])
        row = dict(locus=lid, chrom=g.chrom.iloc[0], start=int(g.start.iloc[0]), end=int(g.end.iloc[0]),
                   motif=g.motif.iloc[0], period=per, L1=L1, L2=L2, reads=len(g), deviant=int(dev.sum()),
                   deviant_frac=float(dev.mean()), min_log10p=float(lp10.min()),
                   min_log10p_phase_free=float(cons[lid].min()),
                   deviant_Y=",".join(map(str, sorted(g.Y.to_numpy()[dev]))), composition_outliers=int(comp.sum()),
                   segdup=float(g.segdup.iloc[0]) if "segdup" in g else float("nan"),
                   sa_deviant=int((g.n_sa.to_numpy()[dev] > 0).sum()))
        if c is not None:
            flagged = c[dev | comp]
            for k, v in flagged["class"].value_counts().items():
                row[k] = int(v)
            sel = dev | comp
            read_rows.append(flagged.assign(locus=lid, log10p=lp10[sel], log10p_phase_free=cons[lid][sel],
                                            phase=g.phase.to_numpy()[sel], n_support=g.n_support.to_numpy()[sel],
                                            n_conflict=g.n_conflict.to_numpy()[sel]))
        rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty:
        print("no deviant reads", file=sys.stderr)
        return
    # clearest: most deviant reads while the large majority of reads still fit
    res = res[res.deviant_frac <= 0.3].sort_values(["deviant", "min_log10p"], ascending=[False, True])
    res.to_parquet(a.out + ".parquet", index=False)
    print(res.head(15).to_string(index=False), file=sys.stderr)
    plt = _style()
    genes = for_reference(a.reference)
    jobs = []                                             # (locus, class) to draw
    if read_rows:
        rr = pd.concat(read_rows, ignore_index=True)
        rr = rr[rr["locus"].isin(set(res.locus))]
        rr.to_parquet(a.out + ".reads.parquet", index=False)
        print(rr["class"].value_counts().to_string(), file=sys.stderr)
        for k, sub in rr.groupby("class"):
            top = sub.groupby("locus").size().sort_values(ascending=False).index[:a.plot]
            print(f"{k}: plotting {', '.join(top)}", file=sys.stderr)
            jobs += [(lid, k) for lid in top]
    else:
        jobs = [(lid, None) for lid in res.locus.head(a.plot)]
    spikes = {}
    for lid, k in jobs:
        g, lp10, dev = keep[lid]
        spikes[(lid, k)] = plot(plt, model, g, lp10, dev, a.out, cls.get(lid), k,
                                _title(genes, g))
    if a.bam:                                             # two-panel figures: spike plot over the pileup
        from tandemo.pileup import run as pileup
        pdir = a.out + "_pileups"
        lids = sorted({lid for lid, _ in jobs})
        outliers = {x: set(keep[x][0].read.to_numpy()[keep[x][2]]) for x in lids}   # the flagged reads per locus
        pileup(a.bam, a.reference, a.catalog, pdir, a.phased_vcf, a.trgt_vcf, loci_names=lids,
               regions=sorted({keep[x][0].chrom.iloc[0] for x in lids}), sample=a.sample, max_reads=None,
               exclude_catalog=a.catalog, params=Params(),
           local=local, pads={x: int(keep[x][0].pad.iloc[0]) for x in lids},
           labels={x: {1: int(keep[x][0].L1.iloc[0]), 2: int(keep[x][0].L2.iloc[0])} for x in lids}, highlight=outliers,
               log=lambda m: print(m, file=sys.stderr))
        for (lid, k), spike in spikes.items():
            pu = os.path.join(pdir, f"{lid}.svg")
            if os.path.exists(pu):
                out = spike.replace(".svg", ".panels.svg")
                compose(spike, pu, out)
                print(f"wrote {out}", file=sys.stderr)


def _title(genes, g) -> str:
    """Gene context, plus the segdup tag when the locus is in a segmental duplication."""
    from tandemo.segdup import tag
    ctx = gene_context(genes, g.chrom.iloc[0], int(g.start.iloc[0]), int(g.end.iloc[0]))
    t = tag(float(g.segdup.iloc[0])) if "segdup" in g else ""
    return " · ".join(x for x in (ctx, t) if x)


def compose(a_svg: str, b_svg: str, out: str, W: int = 1240, gap: int = 12, gutter: int = 22):
    """One SVG: panel a (matplotlib spike plot, scaled to width W) above panel b (the pileup), with a left
    gutter for the panel letters."""
    def body(path):
        t = open(path).read()
        t = re.sub(r"<\?xml[^>]*\?>", "", t)
        t = re.sub(r"<!DOCTYPE[^>]*>", "", t, flags=re.S)
        tag = re.search(r"<svg\b[^>]*>", t).group(0)
        vb = re.search(r'viewBox="([^"]+)"', tag)
        if vb:
            w, h = map(float, vb.group(1).split()[2:4])
        else:
            w = float(re.search(r'width="([\d.]+)', tag).group(1))
            h = float(re.search(r'height="([\d.]+)', tag).group(1))
        return t, tag, w, h

    def place(t, tag, w, h, y, H):
        new = re.sub(r'\s(width|height|x|y)="[^"]*"', "", tag)
        if "viewBox" not in new:
            new = new.replace("<svg", f'<svg viewBox="0 0 {w:g} {h:g}"', 1)
        new = new.replace("<svg", f'<svg x="{gutter}" y="{y:.1f}" width="{W}" height="{H:.1f}"', 1)
        return t.replace(tag, new, 1)

    ta, taga, wa, ha = body(a_svg)
    tb, tagb, wb, hb = body(b_svg)
    Ha, Hb = W * ha / wa, W * hb / wb
    tot = Ha + gap + Hb
    doc = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W + gutter}" '
           f'height="{tot:.0f}" viewBox="0 0 {W + gutter} {tot:.0f}">',
           f'<rect width="{W + gutter}" height="{tot:.0f}" fill="#fcfcfb"/>',
           place(ta, taga, wa, ha, 0, Ha), place(tb, tagb, wb, hb, Ha + gap, Hb),
           '<text x="6" y="16" font-family="sans-serif" font-size="15" font-weight="bold">a</text>',
           f'<text x="6" y="{Ha + gap + 16:.0f}" font-family="sans-serif" font-size="15" font-weight="bold">b</text>',
           "</svg>"]
    with open(out, "w") as fh:
        fh.write("\n".join(doc))


def _covered(seq: str, kmers: set, k: int) -> np.ndarray:
    """Bases of seq covered by at least one k-mer of the reference repeat."""
    cov = np.zeros(len(seq), bool)
    for i in range(len(seq) - k + 1):
        if seq[i:i + k] in kmers:
            cov[i:i + k] = True
    return cov


def _blocks(novel: np.ndarray, gap: int):
    """Runs of novel bases, merging runs separated by fewer than `gap` ref-like bases: [(start, end)]."""
    out = []
    for i in np.nonzero(novel)[0]:
        if out and i - out[-1][1] < gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def classify(keep: dict, a) -> dict:
    """{locus: DataFrame per read (same order as its group)}: repeat sequence class and novel content."""
    p = Params()
    out = {}
    with pysam.AlignmentFile(a.bam) as bam, pysam.FastaFile(a.reference) as fa:
        for lid, (g, lp10, dev) in keep.items():
            chrom, s, e = g.chrom.iloc[0], int(g.start.iloc[0]), int(g.end.iloc[0])
            per, L1, L2 = int(g.period.iloc[0]), int(g.L1.iloc[0]), int(g.L2.iloc[0])
            ref = fa.fetch(chrom, s, e).upper()
            k = min(a.k, max(len(ref) - 1, 1))
            kmers = {ref[i:i + k] for i in range(len(ref) - k + 1)}
            want = dict(zip(g.read, range(len(g))))
            seqs = [""] * len(g)
            for r in bam.fetch(chrom, s, e):
                i = want.get(r.query_name)
                if i is None or not usable(r, p) or r.reference_start > s - p.flank or r.reference_end < e + p.flank:
                    continue
                aln = Aln(r)
                q = r.query_sequence.upper()
                P = int(g.pad.iloc[0]) if "pad" in g else p.pad_for(g.motif.iloc[0].split(","))
                lo = aln.q_at(s - P) + P
                seqs[i] = q[lo:lo + int(g.Y.iloc[i])]
            novel = [(~_covered(x, kmers, k)) if x else np.zeros(0, bool) for x in seqs]
            nb = np.array([n.sum() for n in novel])
            w = g.w1_train.to_numpy()
            hap = g.phase.to_numpy()                        # confident phase (1, 2) or uncertain (0)
            Y = g.Y.to_numpy()
            # a confidently phased read is compared with its own allele, an uncertain one with the nearer allele
            near = np.where(hap == 1, L1, np.where(hap == 2, L2,
                            np.where(np.abs(Y - L1) <= np.abs(Y - L2), L1, L2)))
            extra = Y - near
            # baseline: the non-deviant reads of the same allele (germline allele content is not an outlier)
            base = {Lx: np.median(nb[(near == Lx) & ~dev]) if ((near == Lx) & ~dev).any() else 0 for Lx in (L1, L2)}
            rows = []
            for i in range(len(g)):
                excess = nb[i] - base[near[i]]
                unit, period, c = "", 0, "fits"
                blocks = _blocks(novel[i], k) if len(novel[i]) else []
                big = max(blocks, key=lambda b: b[1] - b[0]) if blocks else None
                if big is not None:
                    blk = seqs[i][big[0]:big[1]]
                    period = measured_period(blk, max_period=min(200, len(blk) // 2 or 1))
                    # a new repeat element: a periodic block of >= max(10 bp, 2 units), not a single repeated base
                    if len(blk) >= max(10, 2 * period) and self_match(blk, period) >= 0.75 \
                            and len(set(blk[:period])) > 1:
                        unit = blk[:period]
                if not seqs[i]:
                    c = "no sequence" if dev[i] else "fits"
                elif dev[i]:
                    if extra[i] < 0:
                        c = "contraction"
                    elif excess <= max(0.25 * extra[i], 2):
                        c = "expansion, ref-like units"
                    elif unit:
                        c = "insertion, novel repeat"
                    else:
                        c = "insertion, non-repetitive"
                elif excess >= max(10, 2 * per):
                    c = "composition, novel repeat" if unit else "composition, non-repetitive"
                rows.append(dict(read=g.read.iloc[i], hap=int(hap[i]), Y=int(Y[i]), L_near=int(near[i]),
                                 extra=int(extra[i]), novel_bp=int(nb[i]), novel_excess=float(excess),
                                 novel_block_bp=int(big[1] - big[0]) if big is not None else 0,
                                 novel_unit=unit, novel_period=int(period) if unit else 0, **{"class": c}))
            out[lid] = pd.DataFrame(rows)
    return out


def _ranges(Y: np.ndarray, L1: int, L2: int, per: int):
    """x ranges to draw: one, or two (a broken axis) when the alleles' read clusters are far apart (the empty
    stretch between them > 40% of the span)."""
    pad = 3 * per + 3
    near1 = np.abs(Y - L1) <= np.abs(Y - L2)
    a = (int(min([L1] + list(Y[near1]))) - pad, int(max([L1] + list(Y[near1]))) + pad)
    b = (int(min([L2] + list(Y[~near1]))) - pad, int(max([L2] + list(Y[~near1]))) + pad)
    a, b = sorted([a, b])
    if b[0] - a[1] > 0.4 * (b[1] - a[0]):
        return [a, b]
    return [(min(a[0], b[0]), max(a[1], b[1]))]


def plot(plt, model, g, lp10, dev, out, cls=None, label=None, gene="", atoms=None):
    """Observed Y per haplotype (stacked) against each haplotype's expected counts under network 1 (the
    phase-weighted share of every read), one curve per haplotype in its ribbon colour. When the two
    alleles are far apart the x axis is broken into two sections (each as wide as its range) to drop the empty
    stretch between them.
    atoms: Stage 2 rows (one per haplotype: hap, f_L, f_p, n_eff, U, mad, mean_shift) add a right half on the same
    y scale: each haplotype's true-length atoms at their length L, height f(L) x n_eff (expected molecules among
    the reads), unsmoothed."""
    L1, L2, per = int(g.L1.iloc[0]), int(g.L2.iloc[0]), int(g.period.iloc[0])
    Y = g.Y.to_numpy()
    w = g.w1_train.to_numpy()
    hap = g.phase.to_numpy()                        # confident phase (1, 2) or uncertain (0)
    ranges = _ranges(np.concatenate([Y] + [np.asarray(a["f_L"])[np.asarray(a["f_p"]) >= 1e-3] for a in (atoms or [])]),
                     L1, L2, per)
    widths = [max(b - a, 1) for a, b in ranges]
    widths = [max(x, 0.25 * sum(widths)) for x in widths] if len(widths) == 2 else widths
    k = len(ranges)
    if atoms:                                       # left half reads, a spacer, right half true lengths
        fig, axes = plt.subplots(1, 2 * k + 1, figsize=(15, 4.2), sharey=True,
                                 gridspec_kw=dict(width_ratios=widths + [0.09 * sum(widths)] + widths, wspace=0.04))
        axes[k].axis("off")
        right = list(axes[k + 1:])
        axes = np.array(list(axes[:k]))
    else:
        fig, axes = plt.subplots(1, k, figsize=(14, 4.2), sharey=True,
                                 gridspec_kw=dict(width_ratios=widths, wspace=0.04))
        right = []
    axes = np.atleast_1d(axes)
    top = 1.0
    for ax, (lo, hi) in zip(axes, ranges):
        ys = np.arange(lo, hi + 1)
        # expected count at each Y: sum over reads of their mixture probability (each read with its own features)
        lc = np.stack([model.log_curves(g.assign(Y=y), np.array([L1, L2])) for y in ys], 0)     # [Y, n, 2]
        expect = {1: (w[None, :] * np.exp(lc[..., 0])).sum(1), 2: ((1 - w[None, :]) * np.exp(lc[..., 1])).sum(1)}
        bottom = np.zeros(len(ys))
        for h, name in ((1, "haplotype 1"), (2, "haplotype 2"), (0, "unphased")):
            c = np.array([((Y == y) & (hap == h)).sum() for y in ys])
            if (hap == h).any():
                ax.bar(ys, c, bottom=bottom, width=0.85, color=HAP_COLOURS[h], alpha=0.75,
                       label=f"{name} reads ({(hap == h).sum()})")
                bottom += c
        from matplotlib import patheffects as pe
        for h in (1, 2):                                # each haplotype's expected spike, in its ribbon colour
            ax.plot(ys, expect[h], color=HAP_COLOURS[h], lw=1.8, label=f"Haplotype {h} (network 1)",
                    path_effects=[pe.Stroke(linewidth=3.2, foreground="#ffffff"), pe.Normal()])
        for Lx in (L1, L2):
            if lo <= Lx <= hi:
                ax.axvline(Lx, color="#898781", lw=1, ls=":")
        dy = sorted({y for y in Y[dev] if lo <= y <= hi})
        if dy:
            ax.scatter(dy, [bottom[ys == y][0] * 1.6 + 0.3 for y in dy], marker="v", s=34, color="#e34948",
                       edgecolors="#0b0b0b", linewidths=0.8, zorder=3, label="deviant read (−log10 p > 3.5)")
        top = max(top, float(bottom.max()) if len(bottom) else 1.0)
        ax.set_xlim(lo, hi)
        ax.set_yscale("symlog", linthresh=1)
    for ax, (lo, hi) in zip(right, ranges):          # Stage 2: true-length atoms per haplotype, side by side
        for a in atoms:
            h = int(a["hap"])
            Ls, m = np.asarray(a["f_L"]), np.asarray(a["f_p"]) * float(a["n_eff"])
            keep = (Ls >= lo) & (Ls <= hi) & (m >= 0.01)
            off = -0.18 if h == 1 else 0.18
            ax.vlines(Ls[keep] + off, 0, m[keep], color=HAP_COLOURS[h], lw=2.4,
                      label=f"Haplotype {h} true lengths (Stage 2)")
            ax.plot(Ls[keep] + off, m[keep], "o", ms=3, color=HAP_COLOURS[h])
            top = max(top, float(m.max()))
        for Lx in (L1, L2):
            if lo <= Lx <= hi:
                ax.axvline(Lx, color="#898781", lw=1, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_yscale("symlog", linthresh=1)
        ax.tick_params(axis="y", left=False, labelleft=False)
    if right:
        txt = "\n".join(f"haplotype {int(a['hap'])}: U {a['U']:.2f}, spread E|d| {a['mad']:.1f} bp, "
                         f"direction E[d] {a['mean_shift']:+.1f} bp" for a in atoms)
        right[0].text(0.02, 0.97, txt, transform=right[0].transAxes, va="top", fontsize=7.5, color="#2b2a28")
    axes[0].set_ylabel("reads")
    axes[0].set_ylim(0, top * 3)
    halves = [list(axes)] + ([right] if right else [])
    for half in halves:
        if len(half) == 2:                              # the axis break: a white gap (marks drawn below)
            half[0].spines["right"].set_visible(False)
            half[1].spines["left"].set_visible(False)
            half[1].tick_params(axis="y", left=False)
    motif = g.motif.iloc[0] if len(g.motif.iloc[0]) <= 70 else g.motif.iloc[0][:70] + "…"
    fig.suptitle(f"{g.locus.iloc[0]}{' · ' + gene if gene else ''} · motif {motif} ({per} bp) · read repeat lengths, alleles {L1} / {L2} bp",
                 fontsize=10, x=0.01, ha="left")
    handles, names = [], []
    for ax in list(axes) + right:
        for h_, n_ in zip(*ax.get_legend_handles_labels()):
            if n_ not in names:
                handles.append(h_); names.append(n_)
    notes = []
    if cls is not None:                                   # every flagged read with a new repeat element, named below
        flagged = cls[(dev | cls["class"].str.startswith("composition").to_numpy()) & cls.novel_unit.astype(bool)]
        for _, r in flagged.head(5).iterrows():
            copies = r.novel_block_bp / max(r.novel_period, 1)
            u = r.novel_unit if len(r.novel_unit) <= 40 else r.novel_unit[:40] + "…"
            notes.append(f"{r['class']}: read …{str(r.read)[-22:]} (hap {r.hap}), Y {r.Y} bp ({r.extra:+d} vs "
                         f"allele {r.L_near}) · new repeat {u} ({r.novel_period} bp, ~{copies:.1f} copies in a "
                         f"{r.novel_block_bp} bp block)")
        if len(flagged) > 5:
            notes.append(f"… and {len(flagged) - 5} more reads with a new repeat element")
    # below the plot, bottom up: the notes, the legend in one row (as in the pileup), the x label
    notes_in = 0.2 * len(notes) + (0.1 if notes else 0)
    h_in = 4.7 + notes_in
    fig.set_figheight(h_in)
    leg_in = notes_in + 0.08
    fig.subplots_adjust(top=0.88, bottom=(leg_in + 0.75) / h_in, left=0.06, right=0.99)
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, leg_in / h_in), ncol=len(handles),
               fontsize=8, frameon=False, handlelength=1.6, columnspacing=1.6)
    for half, name in zip(halves, ("read repeat length Y (bp)", "true repeat length L (bp)")):
        x0, x1 = half[0].get_position().x0, half[-1].get_position().x1
        fig.text((x0 + x1) / 2, (leg_in + 0.36) / h_in, name, fontsize=9, color="#52514e", ha="center")
    if right:                                           # what each half shows
        fig.text(axes[0].get_position().x0, 0.885, "reads and network 1 expectation", fontsize=8.5, color="#52514e")
        fig.text(right[0].get_position().x0, 0.885, "true lengths (Stage 2 NPMLE): expected molecules among the reads",
                 fontsize=8.5, color="#52514e")
    if notes:
        fig.text(0.01, 0.02, "\n".join(notes), fontsize=8.5, family="monospace", color="#0b0b0b", va="bottom")
    for half in halves:
        if len(half) != 2:
            continue
        p0, p1 = half[0].get_position(), half[1].get_position()   # two parallel 45-degree strokes in the gap
        xc, yb = (p0.x1 + p1.x0) / 2, p0.y0
        W_in, H_in = fig.get_size_inches()
        hy = 0.018                                      # half height of a stroke, figure fraction
        hx = hy * H_in / W_in                           # the same length in x for a true 45 degrees
        sep = 0.006
        for dx in (-sep, sep):
            fig.add_artist(plt.Line2D([xc + dx - hx, xc + dx + hx], [yb - hy, yb + hy], transform=fig.transFigure,
                                      color="#52514e", lw=1.1))
    tag = "" if label is None else "." + label.split(",")[0].replace(" ", "_") + (
        "_" + label.split(", ")[1].replace(" ", "_").replace("-", "") if ", " in label else "")
    name = os.path.basename(out) + tag + "." + g.locus.iloc[0]
    _save(fig, os.path.dirname(os.path.abspath(out)), name)
    return os.path.join(os.path.dirname(os.path.abspath(out)), name + ".svg")


if __name__ == "__main__":
    main()
