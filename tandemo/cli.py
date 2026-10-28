"""tandemo command line: extract, train, fit."""

from __future__ import annotations

import argparse
import sys

from .phasing import PhaseParams


def _phase(a) -> PhaseParams:
    return PhaseParams(min_het=a.min_het, min_llr=a.min_llr, eps=a.phase_eps)


def _phase_args(sp):
    sp.add_argument("--min-het", type=int, default=3, help="het SNVs for a confidently phased read (3)")
    sp.add_argument("--min-llr", type=float, default=6.9, help="|log LR| for a confident read (6.9 = 1000:1)")
    sp.add_argument("--phase-eps", type=float, default=0.005, help="weight left on the other haplotype (0.005)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tandemo", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="per-read repeat lengths, features and phase evidence")
    e.add_argument("--bam", required=True)
    e.add_argument("--reference", required=True)
    e.add_argument("--catalog", required=True, help="TRGT BED or plain BED")
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

    t = sub.add_parser("train", help="Stage 1 noise model on stable sample(s)")
    t.add_argument("--input", nargs="+", required=True, help="extraction prefixes of stable samples (need --trgt-vcf)")
    t.add_argument("--out", required=True, help="model file: .pt (network 1) or .npz (--empirical)")
    t.add_argument("--empirical", action="store_true", help="numpy histogram baseline instead of network 1")
    t.add_argument("--val-chroms", nargs="*", default=["chr20", "chr21", "chr22"])
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--D", type=int, default=50, help="core error bins: |Y - L| <= D (50)")
    t.add_argument("--T", type=int, default=5000, help="tail bins reach |Y - L| = T (5000)")
    t.add_argument("--device")
    _phase_args(t)

    f = sub.add_parser("fit", help="Stage 2: per locus and haplotype length distributions")
    f.add_argument("--input", required=True, help="extraction prefix of the subject")
    f.add_argument("--model", required=True)
    f.add_argument("--out", required=True, help="output prefix")
    f.add_argument("--subject")
    f.add_argument("--shape", choices=["parametric", "nonparametric"], default="parametric")
    f.add_argument("--min-reads", type=int, default=3)
    f.add_argument("--omega-min", type=float, help="parametric g width floor, bp (default period / 2, at least 1)")
    f.add_argument("--np-smooth", type=float, help="nonparametric smoothing sd, bp (default period / 2, at least 1)")
    f.add_argument("--g-min-shift", type=int, help="g has no mass within this many bp of L0 (default max(2, period))")
    f.add_argument("--no-selection", action="store_true", help="skip the spanning-selection correction")
    f.add_argument("--flank", type=int, default=128)
    f.add_argument("--threads", type=int, default=1)
    f.add_argument("--write-curves", action="store_true")
    _phase_args(f)

    c = sub.add_parser("combine", help="merge per-chromosome lengths tables into one")
    c.add_argument("--inputs", nargs="+", required=True, help="<out>.lengths.parquet or .tsv.gz files")
    c.add_argument("--out", required=True, help="output .parquet (gzip TSV without pyarrow)")

    a = ap.parse_args(argv)
    if a.cmd == "extract":
        from .extract import Params, extract
        nr, nl = extract(a.bam, a.reference, a.catalog, a.out, a.phased_vcf, a.trgt_vcf, a.regions, a.threads,
                         Params(flank=a.flank, min_mapq=a.min_mapq, min_rq=a.min_rq,
                                min_flank_identity=a.min_flank_identity, sample=a.sample))
        print(f"{nr} read observations at {nl} loci", file=sys.stderr)
    elif a.cmd == "train":
        from .data import training_reads
        from .features import DeltaBins
        df = training_reads(a.input, _phase(a))
        bins = DeltaBins(a.D, a.T)
        print(f"{len(df)} training reads at {df['locus'].nunique()} loci", file=sys.stderr)
        if a.empirical:
            from .noise import EmpiricalNoise
            EmpiricalNoise(bins).fit(df).save(a.out)
        else:
            from .network1 import train
            train(df, a.out, bins, a.val_chroms, a.epochs, a.batch, a.lr, hidden=a.hidden, device=a.device,
                  log=lambda m: print(m, file=sys.stderr))
    elif a.cmd == "fit":
        from .fit import run
        from .noise import load_model
        from .stage2 import FitParams
        p = FitParams(shape=a.shape, min_reads=a.min_reads, omega_min=a.omega_min, np_smooth=a.np_smooth,
                      g_min_shift=a.g_min_shift,
                      selection=not a.no_selection, flank=a.flank)
        _, paths = run(a.input, load_model(a.model), a.out, a.subject, p, _phase(a), a.threads,
                       write_curves=a.write_curves, log=lambda m: print(m, file=sys.stderr))
        print("wrote " + ", ".join(paths), file=sys.stderr)
    elif a.cmd == "combine":
        import pandas as pd
        from .fit import write_table
        parts = [pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, sep="\t") for f in a.inputs]
        print("wrote " + write_table(pd.concat(parts, ignore_index=True), a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
