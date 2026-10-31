"""Gene context of a repeat for figure titles: which gene body it lies in, and where in that gene.

From the genome's gene tables (T2T-CHM13v2.0: NCBI RefSeq GCF_009914755.1-RS_2025_08, native CHM13 coordinates):
protein-coding genes first, then the other biotypes (lncRNA, pseudogenes, ...). Within a gene the position is
read off its representative transcript (RefSeq Select where there is one): exon k of n or intron k, numbered in
transcript direction. Outside every gene: upstream of the nearest gene within 5 kb (its promoter side, by strand),
else intergenic with the nearest protein-coding gene and its distance.

    HTT · exon 1 of 67 (+)        FXN · intron 1 (+)        ARX · 1.2 kb upstream (−)        intergenic · near GBA1 (34 kb)
"""

from __future__ import annotations

import bisect
import os

import pandas as pd

from .genomes import GENOMES, root

KINDS = ("protein_coding", "other_biotypes")
UPSTREAM = 5_000


def for_reference(reference: str | None) -> str | None:
    """The gene-table prefix of the genome whose reference this is, or None."""
    if not reference:
        return None
    for g in GENOMES.values():
        if g.get("genes") and os.path.basename(g["reference"]) == os.path.basename(reference):
            return os.path.join(root(), g["genes"])
    return None


def _kb(bp: int) -> str:
    return f"{bp / 1000:.1f} kb" if bp >= 1000 else f"{bp} bp"


class Genes:
    def __init__(self, prefix: str):
        self.genes, self.exons = {}, {}
        for kind in KINDS:
            g = pd.read_csv(f"{prefix}.{kind}.genes.tsv.gz", sep="\t",
                            usecols=["#chrom", "start", "end", "name", "strand", "gene_biotype",
                                     "representative_transcript"]).rename(columns={"#chrom": "chrom"})
            e = pd.read_csv(f"{prefix}.{kind}.exons.tsv.gz", sep="\t",
                            usecols=["#chrom", "start", "end", "transcript_id", "exon_number", "n_exons",
                                     "representative"]).rename(columns={"#chrom": "chrom"})
            e = e[e["representative"].astype(str) == "True"]
            self.genes[kind] = {c: d.sort_values("start").reset_index(drop=True) for c, d in g.groupby("chrom")}
            for tx, d in e.groupby("transcript_id"):
                self.exons[tx] = d.sort_values("exon_number")[["start", "end", "exon_number", "n_exons"]].to_numpy()

    def _overlapping(self, kind, chrom, start, end):
        d = self.genes[kind].get(chrom)
        if d is None:
            return d, []
        i = bisect.bisect_left(d["start"].to_numpy(), end)
        sub = d.iloc[:i]
        return d, list(sub[sub["end"] > start].itertuples())

    def _where(self, g, start, end) -> str:
        ex = self.exons.get(g.representative_transcript)
        if ex is None or not len(ex):
            return "gene body"
        hit = [int(k) for s, e, k, n in ex if s < end and e > start]
        n = int(ex[0][3])
        if hit:
            return f"exon {hit[0]} of {n}" if len(hit) == 1 else f"exons {min(hit)}-{max(hit)} of {n}"
        for (s0, e0, k0, _), (s1, e1, k1, _) in zip(ex[:-1], ex[1:]):    # consecutive in transcript direction
            lo, hi = min(e0, e1), max(s0, s1)          # the gap: lower exon's end to upper exon's start
            if start < hi and end > lo:
                return f"intron {int(min(k0, k1))}"
        return "gene body"

    def context(self, chrom: str, start: int, end: int) -> str:
        for kind in KINDS:
            _, hits = self._overlapping(kind, chrom, start, end)
            if hits:
                hits.sort(key=lambda g: (g.end - g.start))
                parts = [f"{g.name} · {self._where(g, start, end)} ({'+' if g.strand == '+' else '−'})"
                         for g in hits[:2]]
                return "; ".join(parts) + (f" (+{len(hits) - 2} more)" if len(hits) > 2 else "")
        best = None                                     # upstream of a gene, on its promoter side
        for kind in KINDS:
            d = self.genes[kind].get(chrom)
            if d is None:
                continue
            near = d[(d["start"] < end + UPSTREAM) & (d["end"] > start - UPSTREAM)]
            for g in near.itertuples():
                up = g.start - end if g.strand == "+" else start - g.end
                if 0 <= up <= UPSTREAM and (best is None or up < best[0]):
                    best = (up, g.name, g.strand)
            if best:
                return f"{best[1]} · {_kb(best[0])} upstream ({'+' if best[2] == '+' else '−'})"
        d = self.genes["protein_coding"].get(chrom)
        if d is None or not len(d):
            return "intergenic"
        dist = (d["start"] - end).clip(lower=0) + (start - d["end"]).clip(lower=0)
        k = int(dist.idxmin())
        return f"intergenic · near {d.loc[k, 'name']} ({_kb(int(dist[k]))})"


_CACHE: dict = {}


def context(prefix: str | None, chrom: str, start: int, end: int) -> str:
    """Gene context, or "" without gene tables (cached per prefix)."""
    if not prefix:
        return ""
    if prefix not in _CACHE:
        try:
            _CACHE[prefix] = Genes(prefix)
        except FileNotFoundError:
            _CACHE[prefix] = None
    g = _CACHE[prefix]
    return g.context(chrom, start, end) if g else ""
