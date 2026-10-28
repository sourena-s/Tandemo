"""Loading extraction outputs, phase weights, and linking TRGT alleles to haplotypes for training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .phasing import PhaseParams, phase_weight

LOCUS_KEEP = ["locus", "motif", "period", "ref_len", "gc", "hp_frac", "max_hp", "cx3", "flank_cx", "locus_ps",
              "al1", "al2", "ploidy", "trgt_phased", "n_overlap", "n_span", "read_lens"]


def load(prefix: str, phase: PhaseParams | None = None, sample: str | None = None):
    """(reads merged with their locus features, loci) for one extraction prefix."""
    reads = pd.read_csv(prefix + ".reads.tsv.gz", sep="\t", dtype={"locus": str, "read": str, "chrom": str})
    loci = pd.read_csv(prefix + ".loci.tsv.gz", sep="\t", dtype={"locus": str, "chrom": str, "read_lens": str})
    reads = reads.merge(loci[LOCUS_KEEP], on="locus", how="left")
    reads["w1"] = phase_weight(reads["n_het"].to_numpy(), reads["llr"].to_numpy(), phase)
    reads["sample"] = sample or prefix
    loci["sample"] = sample or prefix
    return reads, loci


def assign_alleles(reads: pd.DataFrame) -> pd.DataFrame:
    """Columns L1, L2: the TRGT allele lengths of haplotype 1 and 2 of each read's locus.

    Phased TRGT call (HiPhase, same block as the reads): allele k is haplotype k + 1.
    Unphased call: the orientation is taken from the reads with confident phase evidence (the one
    that puts each haplotype's median length nearer its allele). Without such reads the two
    orientations give the same training loss, because every read then has w1 = 0.5.
    Haploid calls (al1 == al2 from one AL value) are simply both alleles equal.
    """
    out = reads.copy()
    out["L1"], out["L2"] = out["al1"], out["al2"]
    for loc, g in out.groupby("locus", sort=False):
        if g["trgt_phased"].iloc[0] == 1 or g["al1"].iloc[0] == g["al2"].iloc[0]:
            continue
        a1, a2 = g["al1"].iloc[0], g["al2"].iloc[0]
        h1 = g.loc[g["w1"] > 0.9, "Y"]
        h2 = g.loc[g["w1"] < 0.1, "Y"]
        if len(h1) == 0 and len(h2) == 0:
            continue
        m1 = h1.median() if len(h1) else None
        m2 = h2.median() if len(h2) else None
        keep = sum(abs(m - a) for m, a in ((m1, a1), (m2, a2)) if m is not None)
        swap = sum(abs(m - a) for m, a in ((m1, a2), (m2, a1)) if m is not None)
        if swap < keep:
            out.loc[g.index, ["L1", "L2"]] = [a2, a1]
    return out


def training_reads(prefixes: list[str], phase: PhaseParams | None = None) -> pd.DataFrame:
    frames = []
    for p in prefixes:
        r, _ = load(p, phase, sample=p)
        r = r[(r["al1"] >= 0) & (r["al2"] >= 0)]
        frames.append(assign_alleles(r))
    df = pd.concat(frames, ignore_index=True)
    df["w1_train"] = df["w1"].fillna(0.5)
    return df


def locus_grid(Y: np.ndarray, anchors, D: int, margin: int = 10, max_size: int = 6000) -> np.ndarray:
    """Candidate true lengths for one locus: every integer from below the shortest to above the
    longest read (and the TRGT alleles), widened by the core error range D."""
    vals = list(np.asarray(Y, dtype=int)) + [int(a) for a in anchors if a is not None and a >= 0]
    lo = max(min(vals) - D - margin, 0)
    hi = max(vals) + D + margin
    if hi - lo + 1 > max_size:          # very spread loci: keep the full range, coarser near the ends
        core = np.arange(lo, hi + 1)
        return core[np.linspace(0, len(core) - 1, max_size).astype(int)]
    return np.arange(lo, hi + 1)
