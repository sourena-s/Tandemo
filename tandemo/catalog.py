"""Tandem-repeat catalogs: TRGT BED (Platinum TRs, Adotto, PacBio pathogenic) or a plain BED.

TRGT BED: chrom, start (0-based), end, then `ID=...;MOTIFS=A,B;STRUC=...`. A plain BED gives the
motif(s) in column 4 (comma-separated) and an optional ID in column 5.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass


@dataclass(frozen=True)
class Locus:
    id: str
    chrom: str
    start: int          # 0-based, reference repeat start
    end: int            # exclusive
    motifs: tuple[str, ...]

    @property
    def ref_len(self) -> int:
        return self.end - self.start

    @property
    def periods(self) -> tuple[int, ...]:
        return tuple(sorted({len(m) for m in self.motifs if m})) or (1,)

    @property
    def period(self) -> int:
        """The shortest motif length (the unit of a one-motif change)."""
        return self.periods[0]


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_catalog(path, regions=None, max_len: int = 20_000) -> list[Locus]:
    """Loci sorted by (chrom, start). `regions`: contigs or contig:start-end strings."""
    want = _parse_regions(regions)
    out = []
    with _open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, s, e = f[0], int(f[1]), int(f[2])
            if e - s > max_len or e <= s or not _wanted(want, chrom, s, e):
                continue
            info = f[3] if len(f) > 3 else ""
            if "MOTIFS=" in info:
                kv = dict(x.split("=", 1) for x in info.split(";") if "=" in x)
                motifs = tuple(m for m in kv.get("MOTIFS", "").split(",") if m)
                lid = kv.get("ID", f"{chrom}_{s}_{e}")
            else:
                motifs = tuple(m for m in info.split(",") if m)
                lid = f[4] if len(f) > 4 else f"{chrom}_{s}_{e}"
            out.append(Locus(lid, chrom, s, e, motifs))
    out.sort(key=lambda x: (x.chrom, x.start))
    return out


def _parse_regions(regions):
    if not regions:
        return None
    want = {}
    for r in regions:
        if ":" in r:
            c, se = r.split(":", 1)
            s, e = se.replace(",", "").split("-")
            want.setdefault(c, []).append((int(s) - 1, int(e)))
        else:
            want.setdefault(r, []).append((0, 1 << 62))
    return want


def _wanted(want, chrom, s, e):
    if want is None:
        return True
    return any(s < b and e > a for a, b in want.get(chrom, ()))
