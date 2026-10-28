"""Germline repeat alleles from a TRGT VCF (optionally phased by HiPhase).

TRGT reports, per locus (INFO/TRID), the length of each allele over the catalog interval in
FORMAT/AL, in genotype order. The interval is the same [start, end) the extractor measures reads
over, so AL and a read's length Y are on the same scale.

When HiPhase has phased the TRGT VCF (GT with '|' and a PS), allele k of the genotype belongs to
haplotype k + 1 of that phase block, which is how training links each allele to the read phase
evidence. Otherwise the link is made from the reads (see network1.assign_alleles).
"""

from __future__ import annotations

import pysam


def read_trgt(vcf: str | None, sample: str | None = None) -> dict:
    """{trid: {"al": (len1[, len2]), "phased": bool, "ps": int | None}}; haploid calls have one length."""
    out = {}
    if not vcf:
        return out
    with pysam.VariantFile(vcf) as vf:
        smp = sample or list(vf.header.samples)[0]
        for rec in vf:
            trid = rec.info.get("TRID")
            if trid is None:
                continue
            s = rec.samples[smp]
            al = s.get("AL") if "AL" in s else None
            gt = s.get("GT")
            if al is None or gt is None or None in gt or None in al:
                continue
            al = tuple(int(x) for x in al)
            ps = s.get("PS") if "PS" in s else None
            out[str(trid)] = dict(al=al, phased=bool(s.phased) and len(al) == 2,
                                  ps=int(ps) if ps is not None else None)
    return out
