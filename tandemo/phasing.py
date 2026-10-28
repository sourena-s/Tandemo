"""Read-level phase evidence from phased germline het SNVs (HiPhase VCF), over the whole read.

For every phased heterozygous SNV the read covers (not only those in the flanks), the base the read
carries and its base quality give

    P(base | allele) = 1 - e        if base == allele
                       e / 3        otherwise,            e = 10^(-Q/10)

and the read's log-likelihood ratio for haplotype 1 over haplotype 2 is the sum over sites of
log P(base | hap1 allele) - log P(base | hap2 allele) (natural log). Sites where the read carries
neither allele add nothing and are not counted.

Haplotype labels are only consistent inside one phase block (PS), so the evidence is kept per PS and
a locus uses the block it sits in (see `locus_ps`).

Indels are not used: HiFi residual errors are mostly homopolymer indels, so het indels are
unreliable phase markers. Base qualities are capped at Q50.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
import pysam

QMAX = 50


@dataclass
class HetSites:
    """Phased het SNVs of one chromosome, sorted by position (0-based)."""
    pos: list = field(default_factory=list)
    h1: list = field(default_factory=list)      # base on haplotype 1 (first GT allele)
    h2: list = field(default_factory=list)
    ps: list = field(default_factory=list)

    def __len__(self):
        return len(self.pos)


def load_het_sites(vcf: str | None, chrom: str, sample: str | None = None,
                   lo: int | None = None, hi: int | None = None) -> HetSites:
    out = HetSites()
    if not vcf:
        return out
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
            a1, a2 = alleles[gt[0]], alleles[gt[1]]
            if len(a1) != 1 or len(a2) != 1:
                continue
            ps = s.get("PS") if "PS" in s else None
            out.pos.append(rec.pos - 1)
            out.h1.append(a1.upper()); out.h2.append(a2.upper())
            out.ps.append(int(ps) if ps is not None else 0)
    return out


def read_evidence(aln, seq: str, qual, sites: HetSites) -> dict:
    """{ps: (n_informative_sites, llr_hap1_vs_hap2)} over every phased het SNV the read covers."""
    out: dict = {}
    if not len(sites):
        return out
    i0 = bisect.bisect_left(sites.pos, aln.start)
    i1 = bisect.bisect_left(sites.pos, aln.end)
    for k in range(i0, i1):
        qp = aln.query_pos(sites.pos[k])
        if qp is None:
            continue
        b = seq[qp]
        a1, a2 = sites.h1[k], sites.h2[k]
        if b != a1 and b != a2:
            continue
        q = min(int(qual[qp]), QMAX) if qual is not None else 30
        e = 10 ** (-max(q, 1) / 10)
        lm, lx = math.log(1 - e), math.log(e / 3)
        llr = (lm - lx) if b == a1 else (lx - lm)
        n, s = out.get(sites.ps[k], (0, 0.0))
        out[sites.ps[k]] = (n + 1, s + llr)
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
