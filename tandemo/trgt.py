"""Germline repeat alleles from a TRGT VCF (optionally phased by HiPhase).

TRGT reports, per locus (INFO/TRID), the length of each allele over the catalog interval in
FORMAT/AL, in allele-index order (the order of the unphased genotype, e.g. 0/1). HiPhase phases the call by
rewriting GT (to 1|0, say) but leaves AL as it was, so AL is reordered here to follow GT: with GT 1|0 and
AL 439,221, haplotype 1 carries allele 1 (221 bp). The interval is the same [start, end) the extractor measures reads
over, so AL and a read's length Y are on the same scale.

When HiPhase has phased the TRGT VCF (GT with '|' and a PS), the k-th genotype entry belongs to
haplotype k + 1 of that phase block, which is how training links each allele to the read phase
evidence. Otherwise the link is made from the reads (see network1.assign_alleles).
"""

from __future__ import annotations

import pysam


def gt_order(gt, al) -> tuple:
    """AL (allele-index order: one value per genotype entry, sorted by allele index) -> one length per GT entry."""
    al = tuple(int(x) for x in al)
    if len(al) != len(gt):
        return al
    rank = sorted(gt)
    return tuple(al[rank.index(a)] for a in gt)


def read_trgt(vcf: str | None, sample: str | None = None, ids=None) -> dict:
    """{trid: {"al": (len1[, len2]) in GT order, "phased": bool, "ps": int | None, "seq": (allele sequences, same
    order)}};
    haploid calls have one length; a sequence is "" when the VCF record does not carry it.
    `ids`: keep only these loci."""
    out = {}
    if not vcf:
        return out
    with pysam.VariantFile(vcf) as vf:
        smp = sample or list(vf.header.samples)[0]
        for rec in vf:
            trid = rec.info.get("TRID")
            if trid is None or (ids is not None and str(trid) not in ids):
                continue
            s = rec.samples[smp]
            al = s.get("AL") if "AL" in s else None
            gt = s.get("GT")
            if al is None or gt is None or None in gt or None in al:
                continue
            al = gt_order(gt, al)
            ps = s.get("PS") if "PS" in s else None
            # allele sequences in genotype order: VCF REF / ALT carry one leading anchor base before the interval
            seqs = []
            for i, n in zip(gt, al):
                x = (rec.alleles[i] or "").upper()
                seqs.append(x[1:] if len(x) == n + 1 else x if len(x) == n else "")
            out[str(trid)] = dict(al=al, phased=bool(s.phased) and len(al) == 2,
                                  ps=int(ps) if ps is not None else None, seq=tuple(seqs))
    return out
