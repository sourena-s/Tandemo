"""tandemo command line: extract, train, fit."""

from __future__ import annotations

import argparse
import sys

from .phasing import PhaseParams
from .data import TOLERANCE_HELP, TRGT_READ_TOLERANCE, UNTRUSTED_HELP, UNTRUSTED_TRGT


def _phase(a) -> PhaseParams:
    return PhaseParams(min_het=a.min_het, min_llr=a.min_llr, eps=a.phase_eps)


def _phase_args(sp):
    sp.add_argument("--min-het", type=int, default=3, help="het SNVs for a confidently phased read (3)")
    sp.add_argument("--min-llr", type=float, default=6.9, help="|log LR| for a confident read (6.9 = 1000:1)")
    sp.add_argument("--phase-eps", type=float, default=0.005, help="weight left on the other haplotype (0.005)")


def _genome_args(sp, default_catalog=None):
    from .genomes import GENOMES
    sp.add_argument("--genome", choices=sorted(GENOMES),
                    help="reference genome: sets --reference and --catalog defaults (see `tandemo genomes`)")
    sp.add_argument("--reference", help="FASTA (default: the --genome's)")
    sp.add_argument("--catalog", help="TRGT BED or plain BED, or a catalog name of the --genome "
                                      f"(default: {default_catalog or 'its main catalog'})")


def _resolve(a, ap, default_catalog=None):
    from .genomes import resolve
    a.reference, a.catalog = resolve(a.genome, a.reference, a.catalog, default_catalog)
    if not a.reference or not a.catalog:
        ap.error("give --genome, or both --reference and --catalog")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tandemo", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="per-read repeat lengths, features and phase evidence")
    e.add_argument("--bam", required=True)
    _genome_args(e)
    e.add_argument("--phased-vcf", help="HiPhase phased small-variant VCF (bgzipped, indexed)")
    e.add_argument("--trgt-vcf", help="TRGT VCF (optionally phased by HiPhase)")
    e.add_argument("--sample", help="sample name in the VCFs (default: first)")
    e.add_argument("--out", required=True, help="output prefix")
    e.add_argument("--regions", nargs="*")
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--flank", type=int, default=128, help="good flank bases needed on each side (128)")
    e.add_argument("--min-mapq", type=int, default=20)
    e.add_argument("--min-rq", type=float, default=0.99)
    e.add_argument("--min-flank-identity", type=float, default=0.95)
    e.add_argument("--min-len", type=int, default=0, help="skip catalog loci shorter than this, bp (0)")
    e.add_argument("--no-homopolymer", action="store_true", help="skip loci whose motifs are all 1 bp")
    e.add_argument("--no-segdup", action="store_true",
                   help="skip loci in segmental duplications (the --genome's, or --segdup-bed)")
    e.add_argument("--segdup-bed", help="segmental duplications BED for --no-segdup")
    e.add_argument("--keep-bed", help="loci overlapping this BED survive the filters "
                                      "(default with --genome: its pathogenic catalog)")
    e.add_argument("--no-indel-phasing", action="store_true",
                   help="phase reads from het SNVs only (default: also het indels outside catalog repeats)")

    t = sub.add_parser("train", help="Stage 1 noise model on stable sample(s)")
    t.add_argument("--input", nargs="+", required=True, help="extraction prefixes of stable samples (need --trgt-vcf)")
    t.add_argument("--out", required=True, help="model file: .pt (network 1) or .npz (--empirical)")
    t.add_argument("--empirical", action="store_true", help="numpy histogram baseline instead of network 1")
    t.add_argument("--val-chroms", nargs="*", default=["chr20", "chr21", "chr22"])
    t.add_argument("--per-stratum", type=int,
                   help="sample at most this many loci per period x length class over all inputs (bounds memory "
                        "for genome-wide training; default: every locus)")
    t.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    t.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without a better validation loss (10); the learning rate halves "
                        "after every 2 such epochs, and the best epoch's weights are kept")
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--D", type=int, help="per-bp core: |Y - L| <= D (default 5 with Pareto tails: most read errors "
                                         "are 1-2 bp and the tail law takes over from 6 bp; 50 with uniform tails or "
                                         "--empirical)")
    t.add_argument("--T", type=int, default=5000, help="uniform tails only: they reach |Y - L| = T (5000)")
    t.add_argument("--no-methylation", action="store_true", help="train without the CpG 5mC features")
    t.add_argument("--meth-mask", type=float, default=0.4,
                   help="share of training reads per batch whose methylation inputs are hidden, so the model also "
                        "works for reads without 5mC calls (0.4)")
    t.add_argument("--tail", choices=["pareto", "uniform"], default="pareto",
                   help="network 1 beyond |Y - L| > D: learned generalised Pareto decay per side (default) or the "
                        "old uniform spread to T")
    t.add_argument("--device")
    _phase_args(t)

    f = sub.add_parser("fit", help="Stage 2: per locus and haplotype length distributions")
    f.add_argument("--input", required=True, help="extraction prefix of the subject")
    f.add_argument("--model", required=True)
    f.add_argument("--out", required=True, help="output prefix")
    f.add_argument("--subject")
    f.add_argument("--min-reads", type=int, default=3)
    f.add_argument("--bootstrap", type=int, default=0, help="read resamples for 95%% intervals of U and mad (0: none)")
    f.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    f.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)
    f.add_argument("--no-selection", action="store_true", help="skip the spanning-selection correction")
    f.add_argument("--flank", type=int, default=128)
    f.add_argument("--threads", type=int, default=1)
    f.add_argument("--write-curves", action="store_true")
    _phase_args(f)

    pu = sub.add_parser("pileup", help="SVG pileups of disease repeats, reads split by haplotype")
    pu.add_argument("--bam", required=True)
    _genome_args(pu, "pathogenic")
    pu.add_argument("--phased-vcf", help="HiPhase phased small-variant VCF")
    pu.add_argument("--trgt-vcf", nargs="*", help="TRGT VCF(s), e.g. one per chromosome (alleles in the title)")
    pu.add_argument("--sample")
    pu.add_argument("--out", required=True, help="output directory (one SVG per locus + index.html)")
    pu.add_argument("--loci", nargs="*", help="catalog IDs or gene names (default: every catalog locus)")
    pu.add_argument("--regions", nargs="*")
    pu.add_argument("--top", type=int, help="keep the N loci furthest from the reference")
    pu.add_argument("--view", type=int, default=10_000, help="flank bp drawn each side (10000)")
    pu.add_argument("--max-reads", type=int, help="reads drawn per haplotype (default: every spanning read)")
    pu.add_argument("--min-reads", type=int, default=4)
    pu.add_argument("--no-indel-phasing", action="store_true")
    _phase_args(pu)

    rp = sub.add_parser("report", help="figures of how network 1 behaves (HTML + SVG/PNG)")
    rp.add_argument("--model", required=True, help="network 1 (.pt)")
    rp.add_argument("--input", nargs="+", required=True, help="extraction prefixes (with --trgt-vcf alleles)")
    rp.add_argument("--out", required=True, help="output directory")
    rp.add_argument("--baseline", help="empirical model (.npz) to compare in the calibration panels")
    rp.add_argument("--max-reads", type=int, default=2_000_000, help="reads sampled for the figures (2 M)")
    rp.add_argument("--per-stratum", type=int, help="as for train: loci sampled per period x length class")
    rp.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    rp.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)

    tc = sub.add_parser("trgt-check", help="list the loci whose TRGT alleles disagree with the reads (tsv.gz)")
    tc.add_argument("--input", required=True, help="extraction prefix")
    tc.add_argument("--out", help="output .tsv.gz (default <input>.trgt_read_discrepancy.tsv.gz)")
    tc.add_argument("--trgt-read-tolerance", type=float, default=TRGT_READ_TOLERANCE, help=TOLERANCE_HELP)
    tc.add_argument("--untrusted-trgt", choices=["drop", "local"], default=UNTRUSTED_TRGT, help=UNTRUSTED_HELP)

    sub.add_parser("genomes", help="list the --genome choices, their references and catalogs")

    c = sub.add_parser("combine", help="merge per-chromosome lengths tables into one")
    c.add_argument("--inputs", nargs="+", required=True, help="<out>.lengths.parquet files")
    c.add_argument("--out", required=True, help="output .parquet")

    a = ap.parse_args(argv)
    if a.cmd == "trgt-check":
        from .data import trgt_read_table
        out = a.out or a.input + ".trgt_read_discrepancy.tsv.gz"
        t = trgt_read_table(a.input, a.trgt_read_tolerance, a.untrusted_trgt, log=lambda m: print(m, file=sys.stderr))
        t.to_csv(out, sep="\t", index=False, float_format="%.1f")
        print(f"wrote {out}: {len(t)} haplotypes at {t['locus'].nunique()} loci", file=sys.stderr)
    elif a.cmd == "genomes":
        from .genomes import describe
        print(describe())
    elif a.cmd == "extract":
        _resolve(a, ap)
        from .genomes import resource
        seg = (a.segdup_bed or resource(a.genome, "segdups")) if a.no_segdup else None
        if a.no_segdup and not seg:
            ap.error("--no-segdup needs --genome or --segdup-bed")
        keep = a.keep_bed or resource(a.genome, "pathogenic")
        from .extract import Params, extract
        nr, nl = extract(a.bam, a.reference, a.catalog, a.out, a.phased_vcf, a.trgt_vcf, a.regions, a.threads,
                         Params(flank=a.flank, min_mapq=a.min_mapq, min_rq=a.min_rq,
                                min_flank_identity=a.min_flank_identity, sample=a.sample,
                                indel_phasing=not a.no_indel_phasing, min_len=a.min_len,
                                no_homopolymer=a.no_homopolymer, exclude_bed=seg, keep_bed=keep))
        print(f"{nr} read observations at {nl} loci", file=sys.stderr)
    elif a.cmd == "pileup":
        _resolve(a, ap, "pathogenic")
        from .extract import Params
        from .genomes import GENOMES, resource
        from .pileup import run as pileup
        excl = resource(a.genome, GENOMES[a.genome]["default_catalog"]) if a.genome else a.catalog
        done = pileup(a.bam, a.reference, a.catalog, a.out, a.phased_vcf, a.trgt_vcf, a.loci, a.regions, a.sample,
                      a.view, a.max_reads, a.min_reads, a.top, excl, Params(indel_phasing=not a.no_indel_phasing),
                      _phase(a), log=lambda m: print(m, file=sys.stderr))
        print(f"{len(done)} pileups in {a.out}", file=sys.stderr)
    elif a.cmd == "report":
        from .report import run as report
        path = report(a.model, a.input, a.out, a.baseline, a.max_reads, a.per_stratum,
                      log=lambda m: print(m, file=sys.stderr), tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt)
        print(f"wrote {path}", file=sys.stderr)
    elif a.cmd == "train":
        from .data import training_reads
        from .features import DeltaBins
        df = training_reads(a.input, _phase(a), a.per_stratum, log=lambda m: print(m, file=sys.stderr),
                            tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt)
        D = a.D if a.D is not None else (50 if a.empirical or a.tail == "uniform" else 5)
        bins = DeltaBins(D, a.T)
        print(f"{len(df)} training reads at {df['locus'].nunique()} loci", file=sys.stderr)
        if a.empirical:
            from .noise import EmpiricalNoise
            EmpiricalNoise(bins).fit(df).save(a.out)
        else:
            from .network1 import train
            train(df, a.out, bins, a.val_chroms, a.epochs, a.batch, a.lr, hidden=a.hidden, patience=a.patience, device=a.device,
                  tail=a.tail, methylation=not a.no_methylation, meth_mask=a.meth_mask,
                  log=lambda m: print(m, file=sys.stderr))
    elif a.cmd == "fit":
        from .fit import run
        from .noise import load_model
        from .stage2 import FitParams
        p = FitParams(min_reads=a.min_reads, bootstrap=a.bootstrap,
                      selection=not a.no_selection, flank=a.flank)
        _, paths = run(a.input, load_model(a.model), a.out, a.subject, p, _phase(a), a.threads,
                       write_curves=a.write_curves, tolerance=a.trgt_read_tolerance, untrusted=a.untrusted_trgt,
                       log=lambda m: print(m, file=sys.stderr))
        print("wrote " + ", ".join(paths), file=sys.stderr)
    elif a.cmd == "combine":
        import pandas as pd
        from .fit import write_table
        parts = [pd.read_parquet(f) for f in a.inputs]
        print("wrote " + write_table(pd.concat(parts, ignore_index=True), a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
