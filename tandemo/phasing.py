"""Read-level phase evidence from phased germline het SNVs and indels (HiPhase VCF), over the whole read.

For every phased heterozygous SNV the read covers (not only those in the flanks), the base the read
carries and its base quality give

    P(base | allele) = 1 - e        if base == allele
                       e / 3        otherwise,            e = 10^(-Q/10)

and the read's log-likelihood ratio for haplotype 1 over haplotype 2 is the sum over sites of
log P(base | hap1 allele) - log P(base | hap2 allele) (natural log). Sites where the read carries
neither allele add nothing and are not counted.

Het indels: the read's bases between two anchors INDEL_CTX reference bases either side of the indel
are compared with the two haplotype sequences (reference context + allele). An exact match to one
of them adds +-log((1 - e) / e), with e = INDEL_ERR, or INDEL_ERR_HP when the indel sits in a
homopolymer run >= 4 (where HiFi residual errors concentrate); a read matching neither adds nothing.
Indels that overlap a catalog repeat are dropped (`exclude`): their alleles are repeat lengths, the
quantity being estimated, and phase evidence must never come from length.

Haplotype labels are only consistent inside one phase block (PS), so the evidence is kept per PS and
a locus uses the block it sits in (see `locus_ps`). Base qualities are capped at Q50.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
import pysam

QMAX = 50
INDEL_CTX = 10          # reference bases either side of a het indel in the haplotype sequences
INDEL_ERR = 0.01        # chance a read shows the other allele's indel by error
INDEL_ERR_HP = 0.05     # the same inside a homopolymer run >= 4
EXCLUDE_PAD = 5         # indels this close to a catalog repeat count as overlapping it


@dataclass
class HetSites:
    """Phased het SNVs and indels of one chromosome, sorted by position (0-based).

    SNVs: h1 / h2 are the bases, end = pos + 1. Indels: h1 / h2 are the two haplotype sequences
    (INDEL_CTX reference bases + allele + INDEL_CTX reference bases), [pos, end) the reference span
    they replace, and llr the evidence of an exact match."""
    pos: list = field(default_factory=list)
    end: list = field(default_factory=list)
    h1: list = field(default_factory=list)      # haplotype 1 (first GT allele)
    h2: list = field(default_factory=list)
    ps: list = field(default_factory=list)
    indel: list = field(default_factory=list)
    llr: list = field(default_factory=list)

    def __len__(self):
        return len(self.pos)


def _in_homopolymer(ref: str, i: int, j: int, run: int = 4) -> bool:
    """Is any base of ref[i:j] (the indel and its anchor) part of a run of >= `run` equal bases?"""
    k = max(i - run, 0)
    s = ref[k:j + run]
    n = 1
    for a, b in zip(s, s[1:]):
        n = n + 1 if a == b else 1
        if n >= run:
            return True
    return False


def _overlaps(exclude, a: int, b: int) -> bool:
    """exclude: sorted, merged (start, end) intervals; does any overlap [a - pad, b + pad)?"""
    i = bisect.bisect_left(exclude, (b + EXCLUDE_PAD,)) if exclude else 0
    return i > 0 and exclude[i - 1][1] > a - EXCLUDE_PAD


def load_het_sites(vcf: str | None, chrom: str, sample: str | None = None,
                   lo: int | None = None, hi: int | None = None, fasta: str | None = None,
                   exclude=None) -> HetSites:
    """Phased het sites in [lo, hi). Indels are loaded only when `fasta` is given; `exclude`
    (sorted catalog intervals) drops indels inside repeats."""
    out = HetSites()
    if not vcf:
        return out
    fa = pysam.FastaFile(fasta) if fasta else None
    try:
        with pysam.VariantFile(vcf) as vf:
            if chrom not in vf.header.contigs:
                return out
            smp = sample or list(vf.header.samples)[0]
            it = vf.fetch(chrom, lo, hi) if lo is not None else vf.fetch(chrom)
            for rec in it:
                s = rec.samples[smp]
                gt = s.get("GT")
                if not s.phased or gt is None or len(gt) != 2 or None in gt or gt[0] == gt[1]:
                    continue
                alleles = rec.alleles
                a1, a2 = alleles[gt[0]].upper(), alleles[gt[1]].upper()
                ps = s.get("PS") if "PS" in s else None
                ps = int(ps) if ps is not None else 0
                p = rec.pos - 1
                if len(a1) == 1 and len(a2) == 1 and len(rec.ref) == 1:
                    out.pos.append(p); out.end.append(p + 1)
                    out.h1.append(a1); out.h2.append(a2)
                    out.ps.append(ps); out.indel.append(False); out.llr.append(0.0)
                    continue
                if fa is None or len(a1) == len(a2) or a1[0] != a2[0] or "<" in a1 + a2 or "*" in a1 + a2:
                    continue                                   # MNPs, symbolic and unanchored alleles
                rlen = len(rec.ref)
                if _overlaps(exclude, p, p + rlen):
                    continue
                c0, c1 = max(p - INDEL_CTX, 0), p + rlen + INDEL_CTX
                ctx = fa.fetch(chrom, c0, c1 + INDEL_CTX).upper()
                left, right = ctx[:p - c0], ctx[p - c0 + rlen:c1 - c0]
                e = INDEL_ERR_HP if _in_homopolymer(ctx, p - c0, p - c0 + max(rlen, len(a1), len(a2))) \
                    else INDEL_ERR
                out.pos.append(c0); out.end.append(c1)
                out.h1.append(left + a1 + right); out.h2.append(left + a2 + right)
                out.ps.append(ps); out.indel.append(True); out.llr.append(math.log((1 - e) / e))
    finally:
        if fa is not None:
            fa.close()
    if out.indel and any(a > b for a, b in zip(out.pos, out.pos[1:])):      # indels start INDEL_CTX early
        order = sorted(range(len(out.pos)), key=out.pos.__getitem__)
        for name in ("pos", "end", "h1", "h2", "ps", "indel", "llr"):
            v = getattr(out, name)
            setattr(out, name, [v[i] for i in order])
    return out


def site_llr(aln, seq: str, qual, sites: HetSites, k: int) -> float | None:
    """Evidence of the read for haplotype 1 over 2 at het site k, or None when the read does not
    cover the site or carries neither allele."""
    a1, a2 = sites.h1[k], sites.h2[k]
    if sites.indel[k]:
        if sites.pos[k] < aln.start or sites.end[k] > aln.end or aln.query_pos(sites.pos[k]) is None \
                or aln.query_pos(sites.end[k] - 1) is None:
            return None
        r = seq[aln.q_at(sites.pos[k]):aln.q_at(sites.end[k])]
        if r != a1 and r != a2:
            return None
        return sites.llr[k] if r == a1 else -sites.llr[k]
    qp = aln.query_pos(sites.pos[k])
    if qp is None:
        return None
    b = seq[qp]
    if b != a1 and b != a2:
        return None
    q = min(int(qual[qp]), QMAX) if qual is not None else 30
    e = 10 ** (-max(q, 1) / 10)
    lm, lx = math.log(1 - e), math.log(e / 3)
    return (lm - lx) if b == a1 else (lx - lm)


def read_evidence(aln, seq: str, qual, sites: HetSites) -> dict:
    """{ps: (n_informative_sites, llr_hap1_vs_hap2, n_informative_indels, n_hap1_sites, n_hap2_sites)} over
    every phased het site the read covers; n_hap1 / n_hap2 count the sites where the read carries haplotype 1's
    or haplotype 2's allele (the support and the conflict of a phase call)."""
    out: dict = {}
    if not len(sites):
        return out
    i0 = bisect.bisect_left(sites.pos, aln.start)
    i1 = bisect.bisect_left(sites.pos, aln.end)
    for k in range(i0, i1):
        llr = site_llr(aln, seq, qual, sites, k)
        if llr is None:
            continue
        n, s, ni, n1, n2 = out.get(sites.ps[k], (0, 0.0, 0, 0, 0))
        out[sites.ps[k]] = (n + 1, s + llr, ni + sites.indel[k], n1 + (llr > 0), n2 + (llr < 0))
    return out


def locus_ps(sites: HetSites, start: int, end: int, trgt_ps: int | None = None) -> int | None:
    """The phase block a locus belongs to: the phased TRGT record's PS when there is one, else the
    block shared by the nearest het SNVs on both sides, else the nearest het's block."""
    if trgt_ps:
        return int(trgt_ps)
    if not len(sites):
        return None
    i = bisect.bisect_left(sites.pos, start)
    j = bisect.bisect_left(sites.pos, end)
    left = sites.ps[i - 1] if i > 0 else None
    right = sites.ps[j] if j < len(sites) else None
    if left is not None and left == right:
        return left
    dl = start - sites.pos[i - 1] if i > 0 else 1 << 62
    dr = sites.pos[j] - end if j < len(sites) else 1 << 62
    return left if dl <= dr else right


@dataclass
class PhaseParams:
    min_het: int = 3                 # informative sites for a confident read
    min_llr: float = math.log(1000)  # |LLR| for a confident read (1000:1)
    eps: float = 0.005               # cap: a confident read keeps 1 - eps on its haplotype
    min_support: int = 3             # confident phase call: >= this many het sites carry one haplotype's allele
    max_conflict_frac: float = 0.2   # ... and fewer than this fraction of the informative sites carry the other's


def read_phase(n_hap1, n_hap2, p: PhaseParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(haplotype 1, 2 or 0, many): a read is phased when it is confidently phased (confident_phase: > 2 sites, i.e.
    >= min_support, with < max_conflict_frac conflicting; many = True) or when 1 to min_support - 1 het SNVs / indels
    carry one haplotype's alleles and none the other's (many = False: "<= 2 vars"). A single concordant HiFi site is
    already ~1000:1 evidence; a read misphased this way sits at the other allele, which the scans set aside."""
    p = p or PhaseParams()
    a, b = np.asarray(n_hap1, dtype=float), np.asarray(n_hap2, dtype=float)
    conf = confident_phase(a, b, p)
    few = np.where((a >= 1) & (b == 0), 1, np.where((b >= 1) & (a == 0), 2, 0))
    return np.where(conf > 0, conf, few), conf > 0


def confident_phase(n_hap1, n_hap2, p: PhaseParams | None = None) -> np.ndarray:
    """1 or 2: the haplotype a read is confidently phased to (>= min_support het SNVs / indels carry its
    alleles and fewer than max_conflict_frac of its informative sites carry the other haplotype's: 3:0 and 5:1
    are confident, 4:1 is not); 0 when the phase is uncertain, for which the phase-weighted mixture is used."""
    p = p or PhaseParams()
    a, b = np.asarray(n_hap1, dtype=float), np.asarray(n_hap2, dtype=float)
    n = np.maximum(a + b, 1)
    one = (a >= p.min_support) & (b / n < p.max_conflict_frac)
    two = (b >= p.min_support) & (a / n < p.max_conflict_frac)
    return np.where(one, 1, np.where(two, 2, 0))


def phase_weight(n_het: np.ndarray, llr: np.ndarray, p: PhaseParams | None = None) -> np.ndarray:
    """w1 = prior weight of haplotype 1 from het evidence alone (never from length).

    confident (n >= min_het and |llr| >= min_llr):  1 - eps or eps
    some evidence (n >= 1):                         sigmoid(llr), clipped to [eps, 1 - eps]
    no evidence (n == 0):                           NaN; Stage 2 replaces it by the fitted
                                                    haplotype share rho (training uses 0.5)
    """
    p = p or PhaseParams()
    n_het = np.asarray(n_het, dtype=float)
    llr = np.asarray(llr, dtype=float)
    w = 1.0 / (1.0 + np.exp(-np.clip(llr, -50, 50)))
    w = np.clip(w, p.eps, 1 - p.eps)
    conf = (n_het >= p.min_het) & (np.abs(llr) >= p.min_llr)
    w = np.where(conf, np.where(llr > 0, 1 - p.eps, p.eps), w)
    return np.where(n_het >= 1, w, np.nan)


# ---------------------------------------------------------------- germline indels in the measurement pads

PAD_OK, PAD_AMBIGUOUS, PAD_UNPHASED = 0, 1, 2


def _events(ref: str, alt: str, p: int):
    """(kind, a, b, delta) of one allele against REF at 0-based POS p, after stripping the shared prefix:
    an insertion before reference base a (b = a), or a deletion / replacement of reference [a, b)."""
    k = 0
    while k < min(len(ref), len(alt)) and ref[k] == alt[k]:
        k += 1
    return ("ins" if p + k == p + len(ref) else "del", p + k, p + len(ref), len(alt) - len(ref))


def _zone(kind, a, b, start, end, pad):
    """Where an event falls for a locus: None (outside Y), "pad" (in Y, not in the TRGT allele),
    "repeat" (in both) or "ambiguous" (on a boundary). Y spans reference [start - pad, end + pad);
    an insertion before base x counts in Y for start - pad < x <= end + pad (aln.q_at)."""
    lo, hi = start - pad, end + pad
    if kind == "ins":
        if not lo < a <= hi:
            return None
        if a in (start, end):
            return "ambiguous"
        return "repeat" if start < a < end else "pad"
    if b <= lo or a >= hi:
        return None
    if a < lo or b > hi:
        return "ambiguous"
    if start <= a and b <= end:
        return "repeat"
    if b <= start or a >= end:
        return "pad"
    return "ambiguous"


def pad_indels(vcf: str | None, chrom: str, loci, locus_ps: dict, pad, sample: str | None = None) -> dict:
    """{locus id: (d1, d2, status)}: net length change, per VCF haplotype, of the germline indels that lie
    in the measurement pads of a locus (inside Y, outside the TRGT allele interval). Adding d_h to the TRGT
    allele of haplotype h puts it on the read's Y scale. status: PAD_OK; PAD_AMBIGUOUS when an indel sits on
    a repeat boundary or the outer pad edge; PAD_UNPHASED when a het pad indel is unphased or in another
    phase block than the locus. Loci without pad indels are absent (0, 0, PAD_OK).
    pad: bp, or a function of the locus (per-locus pads)."""
    pad_of = pad if callable(pad) else (lambda L, _p=pad: _p)
    out: dict = {}
    if not vcf or not loci:
        return out
    starts = [L.start for L in loci]
    maxlen = max(L.end - L.start for L in loci)
    pmax = max(pad_of(L) for L in loci)
    lo, hi = max(min(starts) - pmax - 1, 0), max(L.end for L in loci) + pmax + 1
    with pysam.VariantFile(vcf) as vf:
        if chrom not in vf.header.contigs:
            return out
        smp = sample or list(vf.header.samples)[0]
        for rec in vf.fetch(chrom, lo, hi):
            s = rec.samples[smp]
            gt = s.get("GT")
            if gt is None or None in gt or all(i == 0 for i in gt):
                continue
            al = rec.alleles
            if any(a is None or a.startswith("<") or a == "*" for a in al):
                continue
            if all(len(a) == len(rec.ref) for a in al):          # SNVs / MNPs change no length
                continue
            p = rec.pos - 1
            ev = [None if i == 0 else _events(rec.ref.upper(), al[i].upper(), p) for i in gt]
            span_lo = min(e[1] for e in ev if e) - 1
            span_hi = max(e[2] for e in ev if e) + 1
            i0 = bisect.bisect_left(starts, span_lo - pmax - maxlen)
            i1 = bisect.bisect_right(starts, span_hi + pmax)
            het = len(set(gt)) > 1
            ps = s.get("PS") if "PS" in s else None
            for L in loci[i0:i1]:
                zones = [None if e is None else _zone(e[0], e[1], e[2], L.start, L.end, pad_of(L)) for e in ev]
                if all(z in (None, "repeat") for z in zones):
                    continue
                d1, d2, st = out.get(L.id, (0, 0, PAD_OK))
                if "ambiguous" in zones:
                    st = max(st, PAD_AMBIGUOUS)
                if het and (not s.phased or ps is None or ps != locus_ps.get(L.id)):
                    st = max(st, PAD_UNPHASED)
                dh = [e[3] if (e is not None and z == "pad") else 0 for e, z in zip(ev, zones)]
                if len(dh) == 1:                                  # haploid call
                    dh = dh * 2
                out[L.id] = (d1 + dh[0], d2 + dh[1], st)
    return out
