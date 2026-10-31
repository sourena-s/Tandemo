"""Motif-element inputs of network 1, computed from the catalog motifs and the reference repeat:

  elem_gc       GC fraction of the locus's motif elements (all motifs pooled)
  elem_entropy  Shannon entropy of their base composition, bits (0: one base; 2: A, C, G, T equally)
  purity        fraction of reference repeat positions i with base i = base i + period: 1 for a pure repeat,
                lower with every interruption (no tiling, so it is fast for millions of loci)
  n_motifs      number of distinct motif elements the catalog lists for the locus

They replace the 3-mer complexity of the whole repeat (cx3), which mixed element composition with purity.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .genomes import GENOMES, root
from .segdup import PLATINUM_SUFFIXES

COLUMNS = ("elem_gc", "elem_entropy", "purity", "n_motifs")


def reference_for_loci(ids) -> str | None:
    """$TANDEMO_REFERENCE, else T2T's reference when the IDs are Platinum's."""
    if os.environ.get("TANDEMO_REFERENCE"):
        return os.environ["TANDEMO_REFERENCE"]
    ids = list(ids)[:1000]
    if ids and all(str(i).endswith(PLATINUM_SUFFIXES) for i in ids):
        return os.path.join(root(), GENOMES["t2t"]["reference"])
    return None


def motif_features(motif: str) -> tuple[float, float, int]:
    """(elem_gc, elem_entropy, n_motifs) of a comma-separated motif list."""
    ms = [m for m in str(motif).upper().split(",") if m]
    seq = "".join(ms)
    if not seq:
        return float("nan"), float("nan"), 0
    counts = [seq.count(b) for b in "ACGT"]
    n = max(sum(counts), 1)
    ent = -sum(c / n * math.log2(c / n) for c in counts if c)
    return (counts[1] + counts[2]) / n, ent, len(set(ms))


def purity(seq: str, period: int) -> float:
    """Fraction of positions whose base recurs one period later (NaN when the repeat is shorter than 2 periods)."""
    p = max(int(period), 1)
    if len(seq) < 2 * p:
        return float("nan")
    a = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
    return float(np.mean(a[:-p] == a[p:]))


def annotate(loci, reference: str | None) -> None:
    """Add the COLUMNS to a loci table (chrom, start, end, motif, period) in place."""
    feats = [motif_features(m) for m in loci["motif"]]
    loci["elem_gc"] = [f[0] for f in feats]
    loci["elem_entropy"] = [f[1] for f in feats]
    loci["n_motifs"] = [f[2] for f in feats]
    if reference and os.path.exists(reference):
        import pysam
        with pysam.FastaFile(reference) as fa:
            loci["purity"] = [purity(fa.fetch(c, int(s), int(e)), p) for c, s, e, p in
                              zip(loci["chrom"], loci["start"], loci["end"], loci["period"])]
    else:
        loci["purity"] = float("nan")
