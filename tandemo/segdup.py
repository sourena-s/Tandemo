"""Segmental duplications: how much of a repeat lies in one, and the duplicated stretches near it.

Reads from paralogous copies can collapse onto one locus and look like a third allele or instability, so loci in
segmental duplications are flagged (tables, figure titles, a track under the pileup's reference row) and the
overlap is a network 1 locus input; such loci are never dropped. The track is the genome's segdup BED
(T2T-CHM13v2.0: chm13v2.0_maskedY_rCRS.segdups.bed.gz).
"""

from __future__ import annotations

import bisect
import os

from .catalog import read_intervals
from .genomes import GENOMES, root

_CACHE: dict = {}
PLATINUM_SUFFIXES = ("_trsolve", "_TRF", "_pathogenic")     # Platinum TRs locus IDs: a T2T extraction


def for_reference(reference: str | None) -> str | None:
    """The segdup BED of the genome whose reference this is, or None."""
    if not reference:
        return None
    for g in GENOMES.values():
        if g.get("segdups") and os.path.basename(g["reference"]) == os.path.basename(reference):
            return os.path.join(root(), g["segdups"])
    return None


def for_loci(ids) -> str | None:
    """For a loci table without a segdup column: $TANDEMO_SEGDUPS, else T2T's track when the IDs are Platinum's."""
    if os.environ.get("TANDEMO_SEGDUPS"):
        return os.environ["TANDEMO_SEGDUPS"]
    ids = list(ids)[:1000]
    if ids and all(str(i).endswith(PLATINUM_SUFFIXES) for i in ids):
        return os.path.join(root(), GENOMES["t2t"]["segdups"])
    return None


def _iv(path: str) -> dict:
    if path not in _CACHE:
        _CACHE[path] = read_intervals(path) if os.path.exists(path) else {}
    return _CACHE[path]


def intervals(path: str | None, chrom: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Merged segdup intervals overlapping [lo, hi), clipped to it."""
    if not path:
        return []
    iv = _iv(path).get(chrom, [])
    i = max(bisect.bisect_left(iv, (lo,)) - 1, 0)
    out = []
    while i < len(iv) and iv[i][0] < hi:
        s, e = iv[i]
        if e > lo:
            out.append((max(s, lo), min(e, hi)))
        i += 1
    return out


def overlap(path: str | None, chrom: str, start: int, end: int) -> float:
    """Fraction of [start, end) inside a segmental duplication (NaN without a track)."""
    if not path:
        return float("nan")
    return sum(e - s for s, e in intervals(path, chrom, start, end)) / max(end - start, 1)


def tag(frac: float) -> str:
    """Short label for titles: "" when outside, else "segdup 100%"."""
    return f"segdup {frac:.0%}" if frac == frac and frac > 0 else ""
