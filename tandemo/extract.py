"""Per-read repeat lengths, read features and phase evidence from a minimap2 HiFi BAM.

Inputs: the aligned BAM (e.g. somatic_sv/scripts/align_hifi.sh, pbmm2 HiFi settings with -y so `rq`
and `np` are kept), the reference, a tandem-repeat catalog (TRGT BED or plain BED), the HiPhase
phased VCF (het SNVs) and, optionally, the TRGT VCF (germline allele lengths).

Reads used at a locus [start, end) (spanning reads only for now):
  - primary, not duplicate / QC-fail / secondary / supplementary;
  - MAPQ >= min_mapq, rq >= min_rq when the tag is present;
  - both flanks: the read covers `flank` (128) reference bases on each side, with alignment
    identity >= min_flank_identity and no clipping inside them.

Repeat length Y = q(end + pad) - q(start - pad) - 2 * pad, q(r) = read bases before reference
position r. Every base the read has between the two anchors counts, including what minimap2 called
insertions, so the indel placement inside the repeat does not matter.

Outputs (gzip TSV): <prefix>.reads.tsv.gz, one row per (locus, read); <prefix>.loci.tsv.gz, one
row per locus with reference features, TRGT alleles, the locus phase block, and the lengths of all
usable reads that overlap the locus (for the spanning-selection correction in Stage 2).
"""

from __future__ import annotations

import bisect
import gzip
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pysam

from .aln import Aln
from .catalog import Locus, read_catalog
from .phasing import HetSites, load_het_sites, locus_ps, read_evidence
from .trgt import read_trgt

READ_COLUMNS = ["locus", "chrom", "start", "end", "read", "hp", "ps_tag", "Y", "rq", "np", "read_len",
                "mapq", "n_sa", "bq_rep", "bq_flank", "dist_end", "flank_id_min", "n_het", "llr"]
LOCUS_COLUMNS = ["locus", "chrom", "start", "end", "motif", "period", "ref_len", "gc", "hp_frac", "max_hp",
                 "cx3", "flank_cx", "locus_ps", "al1", "al2", "ploidy", "trgt_phased", "n_overlap", "n_span",
                 "read_lens"]
READ_SPAN = 100_000          # het sites are loaded this far beyond a window (reads reach that far)


@dataclass
class Params:
    flank: int = 128
    pad: int = 10
    min_mapq: int = 20
    min_rq: float = 0.99
    min_flank_identity: float = 0.95
    max_locus_len: int = 20_000
    window: int = 2_000_000
    sample: str | None = None


# ---------------------------------------------------------------- reference features

def self_match(s: str, p: int) -> float:
    n = len(s) - p
    if n <= 0:
        return 0.0
    a = np.frombuffer(s.encode(), dtype=np.uint8)
    return float(np.mean(a[:-p] == a[p:]))


def measured_period(s: str, max_period: int = 200) -> int:
    top = min(max_period, len(s) // 2)
    if top < 1:
        return 1
    return max(range(1, top + 1), key=lambda p: (round(self_match(s, p), 3), -p))


def homopolymer_stats(s: str, min_run: int = 4) -> tuple[float, int]:
    """(fraction of bases in homopolymer runs >= min_run, longest run)."""
    if not s:
        return 0.0, 0
    runs, cur = [], 1
    for a, b in zip(s, s[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return sum(r for r in runs if r >= min_run) / len(s), max(runs)


def complexity3(s: str) -> float:
    n = len(s) - 2
    if n <= 0:
        return 0.0
    return len({s[i:i + 3] for i in range(n)}) / min(64, n)


def locus_features(locus: Locus, rep: str, fl: str, fr: str) -> dict:
    period = locus.period if locus.motifs else measured_period(rep)
    hpf, hpm = homopolymer_stats(rep)
    gc = (rep.count("G") + rep.count("C")) / len(rep) if rep else 0.0
    return dict(motif=",".join(locus.motifs) or ".", period=period, ref_len=locus.ref_len,
                gc=round(gc, 4), hp_frac=round(hpf, 4), max_hp=hpm, cx3=round(complexity3(rep), 4),
                flank_cx=round(min(complexity3(fl), complexity3(fr)), 4))


# ---------------------------------------------------------------- reads

def usable(read, p: Params) -> bool:
    if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate \
            or read.is_qcfail or read.mapping_quality < p.min_mapq or read.query_sequence is None:
        return False
    if read.has_tag("rq") and float(read.get_tag("rq")) < p.min_rq:
        return False
    return True


def observe(locus: Locus, aln: Aln, seq: str, qual, ref: str, ref_off: int, p: Params) -> dict | None:
    """Spanning observation of `locus` in one read, or None."""
    s, e, F = locus.start, locus.end, p.flank
    if aln.start > s - F or aln.end < e + F:
        return None
    idl = aln.flank_identity(s - F, s, ref, ref_off, seq)
    idr = aln.flank_identity(e, e + F, ref, ref_off, seq)
    if min(idl, idr) < p.min_flank_identity:
        return None
    P = p.pad
    a, b = aln.q_at(s - P) + P, aln.q_at(e + P) - P
    Y = b - a
    if Y < 0:
        return None
    if qual is not None:
        bq_rep = float(qual[a:b].mean()) if b > a else float("nan")
        fq = np.concatenate([qual[aln.q_at(s - F):aln.q_at(s)], qual[aln.q_at(e):aln.q_at(e + F)]])
        bq_flank = float(fq.mean()) if len(fq) else float("nan")
    else:
        bq_rep = bq_flank = float("nan")
    return dict(Y=Y, bq_rep=bq_rep, bq_flank=bq_flank, dist_end=min(a, len(seq) - b),
                flank_id_min=min(idl, idr))


def _windows(loci, size):
    out, cur = [], []
    for L in loci:
        if cur and (L.chrom != cur[0].chrom or L.start - cur[0].start > size):
            out.append(cur); cur = []
        cur.append(L)
    if cur:
        out.append(cur)
    return out


def _fmt(x):
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.6g}"
    return str(x)


def run_window(args):
    bam_path, fasta_path, loci, p, phased_vcf, trgt = args
    chrom = loci[0].chrom
    F = p.flank + p.pad
    lo = max(min(L.start for L in loci) - F, 0)
    hi = max(L.end for L in loci) + F
    read_rows, locus_rows = [], []
    with pysam.AlignmentFile(bam_path) as bam, pysam.FastaFile(fasta_path) as fa:
        if chrom not in fa.references:
            return read_rows, locus_rows
        hi = min(hi, fa.get_reference_length(chrom))
        ref = fa.fetch(chrom, lo, hi).upper()
        sites = load_het_sites(phased_vcf, chrom, p.sample, max(lo - READ_SPAN, 0), hi + READ_SPAN) \
            if phased_vcf else HetSites()
        starts = [L.start for L in loci]
        maxlen = max(L.ref_len for L in loci)
        meta, lens, nspan = {}, {L.id: [] for L in loci}, {L.id: 0 for L in loci}
        for L in loci:
            if L.start - F < lo or L.end + F > hi:
                continue
            rep = ref[L.start - lo:L.end - lo]
            fl, fr = ref[L.start - p.flank - lo:L.start - lo], ref[L.end - lo:L.end + p.flank - lo]
            t = trgt.get(L.id, {})
            meta[L.id] = locus_features(L, rep, fl, fr)
            meta[L.id]["locus_ps"] = locus_ps(sites, L.start, L.end, t.get("ps") if t.get("phased") else None)
            al = t.get("al", ())
            meta[L.id]["al1"] = al[0] if len(al) >= 1 else -1
            meta[L.id]["al2"] = al[1] if len(al) >= 2 else (al[0] if len(al) == 1 else -1)
            meta[L.id]["ploidy"] = 1 if len(al) == 1 else 2
            meta[L.id]["trgt_phased"] = int(bool(t.get("phased")))
        for read in bam.fetch(chrom, lo, hi):
            if not usable(read, p):
                continue
            aln = Aln(read)
            seq = read.query_sequence.upper()
            qual = np.asarray(read.query_qualities, dtype=float) if read.query_qualities is not None else None
            ev = None
            i0 = bisect.bisect_left(starts, aln.start - maxlen)
            i1 = bisect.bisect_right(starts, aln.end)
            for L in loci[i0:i1]:
                if L.id not in meta or L.end <= aln.start or L.start >= aln.end:
                    continue
                lens[L.id].append(read.query_length)
                o = observe(L, aln, seq, qual, ref, lo, p)
                if o is None:
                    continue
                if ev is None:
                    ev = read_evidence(aln, seq, qual, sites)
                nh, llr = ev.get(meta[L.id]["locus_ps"], (0, 0.0))
                rq = float(read.get_tag("rq")) if read.has_tag("rq") else float("nan")
                npass = int(read.get_tag("np")) if read.has_tag("np") else -1
                n_sa = len([x for x in read.get_tag("SA").split(";") if x]) if read.has_tag("SA") else 0
                read_rows.append((L.id, chrom, L.start, L.end, read.query_name,
                                  read.get_tag("HP") if read.has_tag("HP") else 0,
                                  read.get_tag("PS") if read.has_tag("PS") else 0,
                                  o["Y"], rq, npass, read.query_length, read.mapping_quality, n_sa,
                                  round(o["bq_rep"], 2), round(o["bq_flank"], 2), o["dist_end"],
                                  round(o["flank_id_min"], 4), nh, round(llr, 4)))
                nspan[L.id] += 1
        for L in loci:
            if L.id not in meta:
                continue
            m = meta[L.id]
            locus_rows.append((L.id, chrom, L.start, L.end, m["motif"], m["period"], m["ref_len"], m["gc"],
                               m["hp_frac"], m["max_hp"], m["cx3"], m["flank_cx"],
                               m["locus_ps"] if m["locus_ps"] is not None else -1, m["al1"], m["al2"],
                               m["ploidy"], m["trgt_phased"], len(lens[L.id]), nspan[L.id],
                               ",".join(map(str, lens[L.id])) or "."))
    return read_rows, locus_rows


def extract(bam: str, reference: str, catalog: str, out_prefix: str, phased_vcf: str | None = None,
            trgt_vcf: str | None = None, regions=None, threads: int = 1, params: Params | None = None):
    p = params or Params()
    loci = read_catalog(catalog, regions, p.max_locus_len)
    trgt = read_trgt(trgt_vcf, p.sample)
    jobs = [(bam, reference, w, p, phased_vcf, {L.id: trgt[L.id] for L in w if L.id in trgt})
            for w in _windows(loci, p.window)]
    rpath, lpath = out_prefix + ".reads.tsv.gz", out_prefix + ".loci.tsv.gz"
    os.makedirs(os.path.dirname(os.path.abspath(rpath)), exist_ok=True)
    nr = nl = 0
    with gzip.open(rpath + ".tmp", "wt") as fr, gzip.open(lpath + ".tmp", "wt") as fl:
        fr.write("\t".join(READ_COLUMNS) + "\n")
        fl.write("\t".join(LOCUS_COLUMNS) + "\n")
        ex = ProcessPoolExecutor(threads) if threads > 1 else None
        try:
            results = ex.map(run_window, jobs, chunksize=1) if ex else map(run_window, jobs)
            for rr, lr in results:
                for r in rr:
                    fr.write("\t".join(map(_fmt, r)) + "\n")
                for r in lr:
                    fl.write("\t".join(map(_fmt, r)) + "\n")
                nr += len(rr); nl += len(lr)
        finally:
            if ex:
                ex.shutdown()
    os.replace(rpath + ".tmp", rpath)
    os.replace(lpath + ".tmp", lpath)
    return nr, nl
