"""Schematic pileups of disease-associated tandem repeats: one SVG per locus, reads grouped by haplotype.

Each spanning read (the extractor's filters: primary, MAPQ, rq, clean 128 bp flanks) is one row, aligned at the
repeat start:

  left flank  |  repeat (its own length Y, on a bp scale)  |  right flank

The repeat is tiled into motif units, one colour per catalog motif (rotations included), with interruptions
(bases no motif explains) in dark grey. Flanks are grey and drawn on a compressed scale up to `view` bp from the
repeat, as far as the read reaches. On the flanks, every phased het SNV (tick) and het indel (triangle) of the
locus phase block is marked in the colour of the haplotype allele the read carries (hollow when the read carries
neither). Reads are assigned to haplotype 1 / 2 by the same phase weights as Stage 2 (w1 > 0.9 / < 0.1), else
unphased.

`pileup` renders the loci of a catalog (by default the genome's pathogenic catalog) and writes an index.html that
ranks them by how far a haplotype's median length is from the reference, in motif units.
"""

from __future__ import annotations

import bisect
import html
import os
import re

import numpy as np
import pysam

from .aln import Aln
from .catalog import Locus, read_catalog
from .extract import Params, observe, usable
from .phasing import (INDEL_CTX, HetSites, PhaseParams, load_het_sites, locus_ps, phase_weight, read_phase,
                      read_evidence, site_llr)
from .trgt import read_trgt

# categorical slots 3-8 of the reference palette, in its validated order; slots 1-2 are the haplotypes
MOTIF_COLOURS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_MOTIF = "#898781"           # motifs beyond six fold into one muted colour
INTERRUPTION = "#0b0b0b"
FLANK = "#c3c2b7"
MUTED = "#898781"                 # dashed bridge from a read's repeat end to the aligned right flank
HAP_COLOUR = {1: "#2a78d6", 2: "#eb6834", 0: "#898781"}
HAP_NAME = {1: "haplotype 1", 2: "haplotype 2", 0: "unphased"}
OUTLIER = "#e34948"               # outlier reads: the deviant-read triangle of the length panel, rotated
FLANK_BP = 128                    # reference flank either side of the TRGT allele for local realignment


# ---------------------------------------------------------------- data

def _rotations(m: str):
    return {m[i:] + m[:i] for i in range(len(m))}


def _canon(k: str) -> str:
    return min(_rotations(k))


def discover_motifs(seqs, motifs, min_frac: float = 0.1, max_new: int = 4) -> tuple[str, ...]:
    """The catalog motifs, then unit classes (k-mers up to rotation, k = the catalog motif lengths) that
    sit in tandem (a unit followed by another copy of itself), cover >= min_frac of the repeat bases in
    `seqs`, and the catalog does not already name. Catalogs often list a different motif than the one a
    sample carries (RFC1: AAAAG on the reference, AAGGG / ACAGG pathogenic, AAAGG and others benign);
    the tandem condition keeps single copies of non-repeat sequence inside a locus out."""
    known = {_canon(m.upper()) for m in motifs if "N" not in m.upper()}
    ks = sorted({len(m) for m in motifs if m}) or [1]
    total = sum(len(s) for s in seqs) or 1
    found = []
    for k in ks:
        counts: dict = {}
        for s in seqs:
            for f in range(k):                          # every frame: units may start anywhere
                units = [s[i:i + k] for i in range(f, len(s) - k + 1, k)]
                for a, b in zip(units, units[1:]):
                    if a == b:                          # a tandem copy
                        c = _canon(a)
                        counts[c] = counts.get(c, 0) + 1 / k   # each frame sees the run once

        for c, n in counts.items():
            if c not in known and "N" not in c and len(set(c)) > 1 and n * k / total >= min_frac:
                found.append((n * k, c))
    new = tuple(c for _, c in sorted(found, reverse=True)[:max_new])
    return tuple(motifs) + new


def max_mismatches(length: int) -> int:
    """Mismatches a unit of this length may carry and still tile as its motif. Chosen so that random sequence
    tiles as one of ~3 motifs (any rotation) with probability <= ~1e-3 per position: exact below 10 bp
    (one mismatch at 5 bp would already be 0.23), 1 for 10-14 bp, 2 for 15-19 bp, 3 from 20 bp."""
    return 0 if length < 10 else 1 if length < 15 else 2 if length < 20 else 3


MISMATCH_OPACITY = {0: None, 1: 0.55, 2: 0.4, 3: 0.28}   # same colour, lighter with each mismatch


HP_VARIANT_MIN = 6     # motifs this long also tile when a homopolymer run in a unit is one base longer or shorter


def _hp_variants(r: str, run: int = 3) -> set:
    """r with one homopolymer run (>= run bases) lengthened or shortened by one base: the commonest HiFi error."""
    out, i = set(), 0
    while i < len(r):
        j = i
        while j < len(r) and r[j] == r[i]:
            j += 1
        if j - i >= run:
            out.add(r[:j] + r[i] + r[j:])
            out.add(r[:i] + r[i + 1:])
        i = j
    return out - {r}


def tile(seq: str, motifs, allowance=max_mismatches) -> list[tuple[int, int, int, int]]:
    """(start, length, motif index | -1 for an interruption, mismatches) runs covering seq. Greedy: at each base
    the longest motif (or rotation; N matches anything) that matches exactly; else, for motifs >= HP_VARIANT_MIN bp,
    a unit whose only difference is one homopolymer run one base longer or shorter (counted as 1 mismatch); else
    the longest motif within allowance(motif length) mismatches, fewest first; else one interruption base."""
    pats = []
    for k, m in enumerate(motifs):
        for r in sorted(_rotations(m.upper())):
            pats.append((len(r), k, r))
    pats.sort(key=lambda x: -x[0])
    loose = [(ln, k, r, allowance(ln)) for ln, k, r in pats if allowance(ln) > 0]
    hpv = sorted({(len(v), k, v) for ln, k, r in pats if ln >= HP_VARIANT_MIN for v in _hp_variants(r)},
                 key=lambda x: -x[0])
    out, i, n = [], 0, len(seq)
    while i < n:
        hit = None
        for ln, k, r in pats:
            if i + ln <= n and all(a == b or a == "N" for a, b in zip(r, seq[i:i + ln])):
                hit = (ln, k, 0)
                break
        if hit is None:
            for ln, k, v in hpv:
                if seq.startswith(v, i):
                    hit = (ln, k, 1)
                    break
        if hit is None:
            best = None
            for ln, k, r, cap in loose:
                if i + ln > n or (best is not None and ln < best[0]):
                    continue
                mm = 0
                for a, b in zip(r, seq[i:i + ln]):
                    if a != b and a != "N":
                        mm += 1
                        if mm > cap:
                            break
                if mm <= cap and (best is None or (ln, -mm) > (best[0], -best[2])):
                    best = (ln, k, mm)
            hit = best
        if hit is None:
            if out and out[-1][2] == -1 and out[-1][0] + out[-1][1] == i:
                out[-1] = (out[-1][0], out[-1][1] + 1, -1, 0)
            else:
                out.append((i, 1, -1, 0))
            i += 1
        else:
            out.append((i, hit[0], hit[1], hit[2]))
            i += hit[0]
    return out


def align_motifs(motifs) -> list[str]:
    """The motifs as rows of one block: each rotated to its best Hamming match against the first,
    the shorter ones padded with '-' (VNTR units of one locus are usually rotations and variants)."""
    if not motifs:
        return []
    base = motifs[0].upper()
    rows = [base]
    for m in motifs[1:]:
        m = m.upper()
        best = min(sorted(_rotations(m)), key=lambda r: sum(a != b for a, b in zip(r, base)) + abs(len(r) - len(base)))
        rows.append(best)
    w = max(len(r) for r in rows)
    return [r.ljust(w, "-") for r in rows]


def collect(bam, fasta, locus: Locus, sites: HetSites, ps, p: Params, phase: PhaseParams, view: int,
            pad: int | None = None):
    """Spanning reads of one locus: dicts with hap, w1, Y, repeat sequence, flank reach and het-site calls."""
    P = pad if pad is not None else p.pad_for(locus.motifs)   # the extraction's measurement pad
    F = p.flank + P
    lo, hi = max(locus.start - F - view, 0), locus.end + F + view
    ref = fasta.fetch(locus.chrom, lo, hi).upper()
    i0, i1 = bisect.bisect_left(sites.pos, lo), bisect.bisect_left(sites.pos, hi)
    rows = []
    for read in bam.fetch(locus.chrom, locus.start, locus.end):
        if not usable(read, p):
            continue
        aln = Aln(read)
        seq = read.query_sequence.upper()
        qual = np.asarray(read.query_qualities, dtype=float) if read.query_qualities is not None else None
        o = observe(locus, aln, seq, qual, ref, lo, p, pad=P)
        if o is None:
            continue
        nh, llr, ni, n1, n2 = read_evidence(aln, seq, qual, sites).get(ps, (0, 0.0, 0, 0, 0))
        w1 = float(phase_weight(np.array([nh]), np.array([llr]), phase)[0])
        hp_, many_ = read_phase(np.array([n1]), np.array([n2]), phase)          # the scans' rule: 1, 2 or 0
        hap, many = int(hp_[0]), bool(many_[0])
        a = aln.q_at(locus.start - P) + P
        g0 = aln.q_at(locus.start - p.flank)                     # the read's flank + repeat + flank segment
        marks = []
        for k in range(i0, i1):
            if sites.ps[k] != ps:
                continue
            c = sites.pos[k] + (INDEL_CTX if sites.indel[k] else 0)       # the variant, not its context
            if locus.start - view <= c < locus.start or locus.end <= c < locus.end + view:
                s = site_llr(aln, seq, qual, sites, k)
                if sites.pos[k] >= aln.start and sites.end[k] <= aln.end:
                    marks.append((c, bool(sites.indel[k]), 0 if s is None else (1 if s > 0 else 2)))
        n_sa = len([x for x in read.get_tag("SA").split(";") if x]) if read.has_tag("SA") else 0
        cig = read.cigartuples or []                     # soft clips: the parts aligned elsewhere (SA)
        clip_l = cig[0][1] if cig and cig[0][0] == 4 else 0
        clip_r = cig[-1][1] if cig and cig[-1][0] == 4 else 0
        rows.append(dict(name=read.query_name, hap=hap, many=many, w1=w1, n_het=nh, n_indel=ni, Y=o["Y"], n_sa=n_sa,
                         clip_l=clip_l, clip_r=clip_r,
                         rep=seq[a:a + o["Y"]], left=min(view, locus.start - aln.start),
                         right=min(view, aln.end - locus.end), marks=marks,
                         seg=seq[g0:aln.q_at(locus.end + p.flank)], rep0=a - g0))
    return rows


def summarise(rows, locus: Locus, trgt_al=(), ref_rep: str = "") -> dict:
    per = {}
    for h in (1, 2, 0):
        Y = [r["Y"] for r in rows if r["hap"] == h]
        if Y:
            per[h] = dict(n=len(Y), median=float(np.median(Y)), lo=min(Y), hi=max(Y))
    dev = max((abs(v["median"] - locus.ref_len) / locus.period for h, v in per.items() if h), default=0.0)
    motifs = discover_motifs([r["rep"] for r in rows] + [ref_rep], locus.motifs)
    return dict(per=per, n=len(rows), dev_units=dev, trgt=tuple(trgt_al), motifs=motifs)


# ---------------------------------------------------------------- drawing

def _pick(rows, k, must=()):
    """At most k rows (plus any in `must`, e.g. outlier reads), sorted by length, evenly spaced so the shortest
    and longest are always shown; every row when k is None."""
    rows = sorted(rows, key=lambda r: (r["Y"], r["name"]))
    if k is None or len(rows) <= k:
        return rows
    idx = set(np.unique(np.linspace(0, len(rows) - 1, k).round().astype(int)).tolist())
    idx |= {i for i, r in enumerate(rows) if r["name"] in must}
    return [rows[i] for i in sorted(idx)]


# ---------------------------------------------------------------- local realignment to the haplotype's allele

ALN_MATCH, ALN_MISMATCH, ALN_OPEN, ALN_EXTEND = 2, -4, 6, 1     # HiFi-like scores (minimap2 map-hifi spirit)
_MATRIX = []


def realign(seg: str, target: str):
    """Global affine alignment of a read segment to target (TRGT allele + flanks), end to end: (base[t] = the
    read index aligned to target position t, or -1 where the read lacks it; ins {t: read indices inserted
    before t}; score). minimap2 is a mapper: in long arrays it soft-clips at large gaps (z-drop), so a global
    aligner is used; a contraction is then one gap, not scattered edits.

    Indels 3'-shifted: in a repeat a gained or lost unit scores the same wherever it is placed, and the aligner
    places gaps leftmost, so the sequences are aligned reversed and mapped back. Every indel then sits at the 3'
    end of its run (as in HGVS / VCF normalisation): extra units at the repeat end, a contraction as a gap next
    to the right flank, the same position in every read."""
    T, Q = len(target), len(seg)
    b, ins_r, score = _align(seg[::-1], target[::-1])
    base = np.full(T, -1, dtype=np.int64)
    hit = b >= 0
    base[T - 1 - np.nonzero(hit)[0]] = Q - 1 - b[hit]
    ins = {}
    for t, qs in ins_r.items():                          # before reversed t = between forward T-1-t and T-t
        ins[T - t] = sorted(Q - 1 - q for q in qs)
    return base, ins, score


def _align(seg: str, target: str):
    import parasail
    if not _MATRIX:
        _MATRIX.append(parasail.matrix_create("ACGTN", ALN_MATCH, ALN_MISMATCH))
    r = parasail.nw_trace_striped_32(seg, target, ALN_OPEN, ALN_EXTEND, _MATRIX[0])
    cig = r.cigar.decode
    cig = cig.decode() if isinstance(cig, bytes) else cig
    base = np.full(len(target), -1, dtype=np.int64)
    ins: dict = {}
    q = t = 0
    for n, op in re.findall(r"(\d+)([=XIDM])", cig):
        n = int(n)
        if op in "=XM":
            base[t:t + n] = np.arange(q, q + n); q += n; t += n
        elif op == "I":
            ins.setdefault(t, []).extend(range(q, q + n)); q += n
        else:
            t += n
    return base, ins, r.score


def plan_groups(rows, trgt: dict, ps, left: str, right: str, max_reads: int, must=(), period: int = 1,
                local=()) -> list[dict]:
    """Haplotype groups, each realigned to its own TRGT allele (+ FLANK_BP of reference either side): haplotype 1,
    haplotype 2, and the unphased reads. When the two haplotypes' read lengths are clearly apart (medians
    >= max(10 bp, 2 units)), an unphased read within a quarter of that gap of one haplotype's median joins it as
    "phase from length" (group["inferred"]); the rest stay unphased, split by the allele they align to best.
    Without allele sequences (or parasail) every group keeps the reads' own left-aligned sequence (allele None)."""
    seqs = [x for x in trgt.get("seq", ()) if x]
    al = trgt.get("al", ())
    try:
        import parasail  # noqa: F401
        can = len(seqs) == len(al) and len(seqs) > 0
    except ImportError:
        can = False
    by = {h: [r for r in rows if r["hap"] == h] for h in (1, 2, 0)}
    if not can:
        return [dict(hap=h, label=HAP_NAME[h], rows=g, shown=_pick(g, max_reads, must), allele=None)
                for h, g in by.items() if g]
    # which allele each haplotype carries: the phased TRGT call when it shares the locus phase block, else the
    # orientation that puts each haplotype's median read length nearer its allele
    if len(seqs) == 1 or seqs[0] == seqs[1]:
        hap_allele = {1: 0, 2: 0}
    elif trgt.get("phased") and ps is not None and trgt.get("ps") == ps:
        hap_allele = {1: 0, 2: 1}
    else:
        med = {h: np.median([r["Y"] for r in by[h]]) for h in (1, 2) if by[h]}
        keep = sum(abs(m - al[h - 1]) for h, m in med.items())
        swap = sum(abs(m - al[2 - h]) for h, m in med.items())
        hap_allele = {1: 0, 2: 1} if keep <= swap else {1: 1, 2: 0}
    targets = [left + x + right for x in seqs]
    # unphased reads the length separates cleanly join a haplotype ("phase from length")
    inferred = {1: [], 2: []}
    if by[0] and len(set(seqs)) > 1:
        med = {h: (np.median([r["Y"] for r in by[h]]) if by[h] else al[hap_allele[h]]) for h in (1, 2)}
        gap = abs(med[1] - med[2])
        if gap >= max(10, 2 * period):
            rest = []
            for r in by[0]:
                h = 1 if abs(r["Y"] - med[1]) <= abs(r["Y"] - med[2]) else 2
                (inferred[h] if abs(r["Y"] - med[h]) <= gap / 4 else rest).append(r)
            by[0] = rest
    groups = []
    for h in (1, 2):
        if by[h] or inferred[h]:
            k = hap_allele[h]
            seq, target, local_len, core = seqs[k], targets[k], None, set()
            if h in local and by[h]:                    # TRGT length not trusted: the reads' consensus length
                ys = np.array([r["Y"] for r in by[h]])     # (densest +-1 bp window; the label), drawn on the TRGT allele
                cands = np.unique(ys)
                counts = np.array([np.sum(np.abs(ys - c) <= 1) for c in cands])
                best = cands[counts == counts.max()]
                c = best[np.argmin(np.abs(best - len(seqs[k])))]
                win = [r for r in by[h] if abs(r["Y"] - c) <= 1]
                local_len = int(round(np.median([r["Y"] for r in win])))
                core = {r["name"] for r in win}
            shown = _pick(by[h], max_reads, must) if by[h] else []
            inf = _pick(inferred[h], max_reads, must) if inferred[h] else []
            for r in shown + inf:
                r["aln"] = realign(r["seg"], target)[:2]
            groups.append(dict(hap=h, label=HAP_NAME[h], rows=by[h] + inferred[h], shown=shown, inferred=inf,
                               n_inferred=len(inferred[h]), allele=k, seq=seq, local_len=local_len, core=core))
    if by[0]:                                            # unphased: to whichever allele each aligns best
        best = {}
        for r in by[0]:
            alns = [realign(r["seg"], tg) for tg in targets]
            k = int(np.argmax([x[2] for x in alns])) if len(set(seqs)) > 1 else 0
            r["aln"] = alns[k][:2]
            best.setdefault(k, []).append(r)
        for k, g in sorted(best.items()):
            tag = f" · allele {al[k]} bp" if len(set(seqs)) > 1 else ""
            groups.append(dict(hap=0, label=HAP_NAME[0] + tag, rows=g, shown=_pick(g, max_reads, must), allele=k,
                               seq=seqs[k]))
    return groups


def _ribbon(x, y, w, h, colour, kind: str) -> str:
    """Haplotype ribbon segment: > 2 phasing variants solid, <= 2 lighter, assigned by length light and dashed."""
    if kind == "few":
        return f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" fill="{colour}" fill-opacity="0.45"/>'
    if kind == "len":
        return (f'<rect x="{x + 0.5}" y="{y:.1f}" width="{w - 1}" height="{h:.1f}" fill="{colour}" fill-opacity="0.18" '
                f'stroke="{colour}" stroke-width="1" stroke-dasharray="2,1.5"/>')
    return f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" fill="{colour}"/>'


def _base_colours(seq: str, motifs, colour):
    """Per base (fill, opacity) from the motif tiling, and the unit starts [(index, length)]."""
    col = [(INTERRUPTION, None)] * len(seq)
    starts = []
    for st, n, k, mm in tile(seq, motifs):
        c = INTERRUPTION if k < 0 else colour(k)
        op = MISMATCH_OPACITY[min(mm, 3)] if mm else None
        for x in range(st, st + n):
            col[x] = (c, op)
        if k >= 0:
            starts.append((st, n))
    return col, starts


def render(locus: Locus, groups: list[dict], summary: dict, sample: str, view: int, ref_rep: str,
           label: str | None = None, highlight=(), flank: int = FLANK_BP, labels: dict | None = None,
           gene: str = "", segdups: list | None = None, read_tags: dict | None = None) -> str:
    """One SVG. Each row starts with its deviation in bp from its haplotype's allele: `labels` {1: L1, 2: L2}
    (the training labels: TRGT allele, pad-corrected) when given, else the group's TRGT allele; unphased rows
    against the nearer allele. Reads left-aligned at the repeat start; with allele sequences, each haplotype group is drawn in
    the columns of its own TRGT allele (insertions relative to it expanded, gaps dashed), else each read's own
    sequence left-aligned. Right flanks start at one shared x for every row."""
    W, LAB, FLW, ROW, GAP = 1240, 226, 230, 7, 2
    motifs = summary["motifs"]
    block = align_motifs(motifs)
    colour = lambda k: MOTIF_COLOURS[k] if k < len(MOTIF_COLOURS) else OTHER_MOTIF
    for g in groups:                                    # columns of each group
        if g["allele"] is not None:
            A = len(g["seq"])
            # the germline allele defines the columns; a read's insertions extend only its own row to the right
            rows_ = g["shown"] + g.get("inferred", [])
            widths = [A + sum(len(v) for t, v in r["aln"][1].items() if flank <= t <= flank + A)
                      for r in rows_ if r["name"] not in highlight]
            widths += [r["Y"] for r in rows_ if r["name"] in highlight]
            g["ncol"] = max([A] + widths)
        else:
            g["ncol"] = max([r["Y"] for r in g["shown"]] + [1])
    ncol = max([g["ncol"] for g in groups] + [locus.ref_len, 1])
    rep_w = W - LAB - 2 * FLW - 48
    bp = min(6.0, rep_w / ncol)                         # px per repeat column
    fpx = FLW / view                                    # px per flank bp
    x0 = LAB + FLW                                      # repeat start: every row left-aligned here
    xR = x0 + ncol * bp + 8                             # right flank start: every row right-aligned here
    kind = lambda r: "many" if r.get("many", True) else "few"
    for g in groups:                                    # draw order: > 2 vars, <= 2 vars, by length, then outliers
        g.setdefault("inferred", [])
        normal = [(r, kind(r)) for r in g["shown"] if r["name"] not in highlight and kind(r) == "many"] + \
                 [(r, kind(r)) for r in g["shown"] if r["name"] not in highlight and kind(r) == "few"] + \
                 [(r, "len") for r in g["inferred"] if r["name"] not in highlight]
        outl = [(r, kind(r)) for r in g["shown"] if r["name"] in highlight] + \
               [(r, "len") for r in g["inferred"] if r["name"] in highlight]
        g["draw"] = normal + sorted(outl, key=lambda x: x[0]["Y"])
        g["n_out"] = len(outl)
    nrow = sum(len(g["draw"]) + (g["allele"] is not None) for g in groups)
    BLOCK_GAP, OUT_GAP, BOX_PAD = 4, 9, 3.5            # extra space before "phase from length" / outlier rows;
    #                                                    the outlier box clears its rows by BOX_PAD px
    H = 0                                               # set from what is drawn, at the end
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="11" '
           f'fill="#0b0b0b">',
           f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>']

    def dash(xa, xb, y_mid):
        if xb - xa > 1:
            out.append(f'<line x1="{xa:.2f}" y1="{y_mid:.1f}" x2="{xb:.2f}" y2="{y_mid:.1f}" stroke="{MUTED}" '
                       f'stroke-width="0.8" stroke-dasharray="3,2"/>')

    def cells(y, cs, h, seps=()):
        """cs: per column (fill, opacity) or None (a gap); consecutive equal cells merged."""
        c0 = 0
        while c0 < len(cs):
            c1 = c0
            while c1 + 1 < len(cs) and cs[c1 + 1] == cs[c0]:
                c1 += 1
            if cs[c0] is None:
                dash(x0 + c0 * bp, x0 + (c1 + 1) * bp, y + h / 2)
            else:
                fill, op = cs[c0]
                o = f' fill-opacity="{op}"' if op else ""
                out.append(f'<rect x="{x0 + c0 * bp:.2f}" y="{y}" width="{max((c1 - c0 + 1) * bp, 0.6):.2f}" '
                           f'height="{h}" fill="{fill}"{o}/>')
            c0 = c1 + 1
        for c in seps:
            out.append(f'<line x1="{x0 + c * bp:.2f}" y1="{y}" x2="{x0 + c * bp:.2f}" y2="{y + h}" '
                       f'stroke="#ffffff" stroke-width="0.6"/>')
        dash(x0 + len(cs) * bp, xR, y + h / 2)          # repeat end -> aligned right flank

    def row_cells(g, r):
        """(columns, unit-start columns) of one read in its group's layout."""
        if g["allele"] is None:                         # left-aligned, the read's own repeat bases
            colrs, starts = _base_colours(r["rep"], motifs, colour)
            return list(colrs), [st for st, n in starts if n * bp >= 4]
        base, ins = r["aln"]
        A = len(g["seq"])
        gapless = r["name"] in highlight                # outliers: their own bases, no allele columns
        # the read's repeat bases: those this realignment puts in the allele's repeat columns, with insertions at
        # either edge (not the BAM alignment's repeat start: a unit deleted in a pure run is placed differently
        # by the two aligners, and the first repeat bases would be drawn as flank)
        qs = [int(base[t]) for t in range(flank, flank + A) if base[t] >= 0]
        qs += [q for t in range(flank, flank + A + 1) for q in ins.get(t, [])]
        lo, hi = (min(qs), max(qs) + 1) if qs else (r["rep0"], r["rep0"] + r["Y"])
        colrs, starts = _base_colours(r["seg"][lo:hi], motifs, colour)
        if gapless:
            return list(colrs), [st for st, n in starts if n * bp >= 4]
        unit_q = {lo + st for st, n in starts if n * bp >= 4}
        cs, seps = [], []

        def put(q):
            if q is None or q < 0:
                cs.append(None)
                return
            if q in unit_q:
                seps.append(len(cs))
            cs.append(colrs[q - lo] if lo <= q < hi else (FLANK, None))
        for jj in range(A + 1):
            t = flank + jj
            for q in ins.get(t, []):                    # inserted bases inline: this row alone gets longer
                put(q)
            if jj < A:
                put(int(base[t]))                       # a germline allele base: the read's, or a deletion dash
        return cs, seps

    def allele_cells(g):
        colrs, starts = _base_colours(g["seq"], motifs, colour)
        unit = {st for st, n in starts if n * bp >= 4}
        cs, seps = [], []
        for jj in range(len(g["seq"])):
            if jj in unit:
                seps.append(len(cs))
            cs.append(colrs[jj])
        return cs, seps

    t = html.escape(label or locus.id)
    seg_in = sum(min(e, locus.end) - max(s, locus.start) for s, e in (segdups or []) if s < locus.end and e > locus.start)
    seg_frac = seg_in / max(locus.ref_len, 1)
    gtxt = f'<tspan font-size="13" font-weight="normal" fill="#52514e">  ·  {html.escape(gene)}</tspan>' if gene else ""
    if seg_frac > 0:
        gtxt += f'<tspan font-size="13" font-weight="bold" fill="#6250d6">  ·  segmental duplication {seg_frac:.0%}</tspan>'
    out.append(f'<text x="16" y="24" font-size="16" font-weight="bold">{t}{gtxt}</text>')
    trg = ("  TRGT alleles " + " / ".join(map(str, summary["trgt"])) + " bp") if summary["trgt"] else ""
    mtxt = ",".join(locus.motifs)
    mtxt = mtxt if len(mtxt) <= 60 else mtxt[:60] + f"… ({len(locus.motifs)} motifs)"
    out.append(f'<text x="16" y="42" fill="#444">{html.escape(sample)} · {locus.chrom}:{locus.start + 1:,}-'
               f'{locus.end:,} · motifs {html.escape(mtxt)} · reference {locus.ref_len} bp '
               f'({locus.ref_len / locus.period:.1f} units){html.escape(trg)} · {summary["n"]} spanning reads</text>')

    # reference row (the catalog interval, left-aligned; not in any group's columns)
    y = 66
    out.append(f'<text x="{LAB - 8}" y="{y + 8}" text-anchor="end" fill="#444">reference</text>')
    out.append(f'<rect x="{LAB}" y="{y + 2}" width="{FLW}" height="6" fill="{FLANK}"/>')
    out.append(f'<rect x="{xR:.2f}" y="{y + 2}" width="{FLW}" height="6" fill="{FLANK}"/>')
    rc, rs = _base_colours(ref_rep, motifs, colour)
    cells(y, rc, 10, [st for st, n in rs if n * bp >= 4])
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if s * bp >= 60)
    out.append(f'<line x1="{x0}" y1="{y - 8}" x2="{x0 + step * bp:.2f}" y2="{y - 8}" stroke="#222"/>')
    out.append(f'<text x="{x0 + step * bp + 4:.2f}" y="{y - 5}" fill="#222">{step} bp</text>')
    out.append(f'<text x="{LAB}" y="{y - 5}" fill="#666">← {view:,} bp flank</text>')
    out.append(f'<text x="{xR + FLW:.2f}" y="{y - 5}" text-anchor="end" fill="#666">{view:,} bp flank →</text>')
    if segdups:                                         # segmental duplications in view, under the reference
        ty = y + 14
        out.append(f'<text x="{LAB - 8}" y="{ty + 6}" text-anchor="end" font-size="9" fill="#6250d6">segdup</text>')
        out.append(f'<line x1="{LAB}" y1="{ty + 3}" x2="{xR + FLW:.2f}" y2="{ty + 3}" stroke="#e4e3df" stroke-width="0.8"/>')

        def xmap(c):                                    # reference coordinate -> x on the reference row
            if c < locus.start:
                return x0 - min(locus.start - c, view) * fpx
            if c >= locus.end:
                return xR + min(c - locus.end, view) * fpx
            return x0 + (c - locus.start) * bp
        for s_, e_ in segdups:
            xa, xb = xmap(s_), xmap(e_)
            if locus.start < e_ and s_ < locus.end and e_ >= locus.end:   # spans the repeat end: bridge the dash
                xb = max(xb, xR)
            out.append(f'<rect x="{xa:.2f}" y="{ty}" width="{max(xb - xa, 1):.2f}" height="6" fill="#6250d6" '
                       f'fill-opacity="0.75"/>')

    y = 96 + (10 if segdups else 0)
    for g in groups:
        h = g["hap"]
        st = dict(n=len(g["rows"]), median=float(np.median([r["Y"] for r in g["rows"]])),
                  lo=min(r["Y"] for r in g["rows"]), hi=max(r["Y"] for r in g["rows"]))
        y += 20
        ribbon = []                                     # (top, bottom, phase from length?) per drawn row
        out.append(f'<text x="24" y="{y - 1}" font-weight="bold" fill="{HAP_COLOUR[h]}">{html.escape(g["label"])}</text>')
        nshow = len(g["draw"])
        shown = f" (showing {nshow})" if nshow < st["n"] else ""
        # the group's numbers in the label column, one line each; the group is at least as tall as they are
        lines = [(f'{st["n"]} reads{shown}', "#444"),
                 (f'median {st["median"]:.0f} bp ({st["median"] / locus.period:.1f} u)', "#444"),
                 (f'range {st["lo"]}-{st["hi"]} bp', "#444")]
        n_sa = sum(1 for r in g["rows"] if r.get("n_sa"))
        if n_sa:
            lines.append((f'{n_sa} with supplementary alignments (SA)', "#6250d6"))
        for k, (txt, fill) in enumerate(lines):
            out.append(f'<text x="24" y="{y + 12 + 13 * k}" fill="{fill}">{html.escape(txt)}</text>')
        label_bottom = y + 16 + 13 * len(lines)
        if g["allele"] is not None:                     # the group's TRGT allele, in its own columns
            y += 10                                     # room for the scale labels above it
            ry = y + 2
            AH = ROW + 5                                # the allele row: thicker than a read row
            mid = ry + AH / 2
            # the allele, not a read: its flanks as dark lines at the full flank width (the reads' flank scale),
            # labelled with the flank length; the repeat labelled with the allele length
            for xa, xb in ((x0 - FLW, x0 - 1), (xR, xR + FLW)):
                out.append(f'<line x1="{xa:.1f}" y1="{mid:.1f}" x2="{xb:.1f}" y2="{mid:.1f}" stroke="#2b2a28" '
                           f'stroke-width="1.6"/>')
            # the row's name over the left flank, the flank scale over the right one
            out.append(f'<text x="{x0 - FLW / 2:.1f}" y="{ry - 2:.1f}" text-anchor="middle" font-size="8" '
                       f'font-weight="bold" fill="#2b2a28">Germline allele (TRGT)</text>')
            if g.get("local_len"):                      # TRGT failed the read check: the length the model uses
                out.append(f'<text x="{x0:.1f}" y="{ry - 2:.1f}" font-size="8" font-weight="bold" fill="#2b2a28">'
                           f'{len(g["seq"])} bp · length mismatch: updated to consensus haplotype read length '
                           f'{g["local_len"]} bp (reads with a black line in their flanks)</text>')
            else:                                       # the allele length, over the repeat
                out.append(f'<text x="{x0 + len(g["seq"]) * bp / 2:.1f}" y="{ry - 2:.1f}" text-anchor="middle" '
                           f'font-size="8" font-weight="bold" fill="#2b2a28">{len(g["seq"])} bp</text>')
            out.append(f'<text x="{xR + FLW / 2:.1f}" y="{ry - 2:.1f}" text-anchor="middle" font-size="8" '
                       f'fill="#52514e">{view:,} bp flank</text>')
            ac, asp = allele_cells(g)
            cells(ry, ac, AH, asp)
            out.append(f'<rect x="{x0:.2f}" y="{ry}" width="{len(g["seq"]) * bp:.2f}" '
                       f'height="{AH}" fill="none" stroke="#0b0b0b" stroke-width="1"/>')     # the TRGT repeat, outlined
            ribbon.append((ry - 1, ry + AH + 1, "many"))
            y += AH + GAP + 6                           # gap between the allele and the reads
        box_top = None
        prev_inf = False
        if labels and h in labels:                      # the allele each row's deviation is taken from
            ref_L = [labels[h]]
        elif h in (1, 2) and g.get("seq"):
            ref_L = [len(g["seq"])]
        else:
            ref_L = list(labels.values()) if labels else [len(g["seq"])] if g.get("seq") else list(summary["trgt"])
        for i_row, (r, rk) in enumerate(g["draw"]):
            if g["n_out"] and i_row == len(g["draw"]) - g["n_out"]:
                y += OUT_GAP                            # outlier rows start: a gap, then a dashed box
                box_top = y + 2 - 1.5 - BOX_PAD
            elif prev_inf is not False and rk != prev_inf:
                y += BLOCK_GAP                          # a new phase-evidence block starts
            prev_inf = rk
            ry = y + 2
            ribbon.append((ry - 1, ry + ROW + 1, rk))
            box_bot = ry + ROW + 1.5 + BOX_PAD
            if r["name"] in highlight:                  # outlier read: the length panel's triangle, rotated
                out.append(f'<path d="M{LAB - 38:.1f},{ry - 1.5:.1f} L{LAB - 38:.1f},{ry + ROW + 1.5:.1f} '
                           f'L{LAB - 28:.1f},{ry + ROW / 2:.1f} Z" fill="{OUTLIER}" stroke="#0b0b0b" stroke-width="0.8"/>')
            if ref_L:                                   # deviation from the allele, after the triangle if any
                dv = int(r["Y"] - min(ref_L, key=lambda a: abs(r["Y"] - a)))
                bold = ' font-weight="bold"' if r["name"] in highlight else ""
                out.append(f'<text x="{LAB - 4}" y="{ry + ROW - 0.3:.1f}" text-anchor="end" font-size="7.5" '
                           f'fill="#52514e"{bold}>{dv:+d}</text>' if dv else
                           f'<text x="{LAB - 4}" y="{ry + ROW - 0.3:.1f}" text-anchor="end" font-size="7.5" '
                           f'fill="#9a9893">0</text>')
            out.append(f'<rect x="{x0 - r["left"] * fpx:.2f}" y="{ry + 1}" width="{r["left"] * fpx:.2f}" '
                       f'height="{ROW - 2}" fill="{FLANK}"/>')
            out.append(f'<rect x="{xR:.2f}" y="{ry + 1}" width="{r["right"] * fpx:.2f}" height="{ROW - 2}" fill="{FLANK}"/>')
            if r["name"] in g.get("core", ()):          # read behind the local length call: black line in its flanks
                for xa, w in ((x0 - r["left"] * fpx, r["left"] * fpx), (xR, r["right"] * fpx)):
                    out.append(f'<line x1="{xa:.2f}" y1="{ry + ROW / 2:.2f}" x2="{xa + w:.2f}" y2="{ry + ROW / 2:.2f}" '
                               f'stroke="#0b0b0b" stroke-width="1.2"/>')
            cs, seps = row_cells(g, r)
            cells(ry, cs, ROW, seps)
            tag = (read_tags or {}).get(r["name"])      # an optional per-read tag, e.g. tissue: (short, long, colour)
            if tag:
                out.append(f'<text x="{xR + FLW + 18:.1f}" y="{ry + ROW - 0.3:.1f}" font-size="7.5" font-weight="bold" '
                           f'fill="{tag[2]}">{html.escape(tag[0])}</text>')
            if r.get("n_sa"):                           # read with supplementary alignments: its soft-clipped
                xl = x0 - r["left"] * fpx                   # ends in violet beyond its flanks, same flank scale
                wl = min(r.get("clip_l", 0), view - r["left"]) * fpx
                if wl > 0:
                    out.append(f'<rect x="{xl - wl:.2f}" y="{ry + 1}" width="{wl:.2f}" height="{ROW - 2}" fill="#6250d6" '
                               f'fill-opacity="0.75"/>')
                xr_ = xR + r["right"] * fpx
                wr = min(r.get("clip_r", 0), view - r["right"]) * fpx
                if wr > 0:
                    out.append(f'<rect x="{xr_:.2f}" y="{ry + 1}" width="{wr:.2f}" height="{ROW - 2}" fill="#6250d6" '
                               f'fill-opacity="0.75"/>')
                out.append(f'<text x="{xR + FLW + 4:.1f}" y="{ry + ROW - 0.3:.1f}" font-size="7" font-weight="bold" '
                           f'fill="#6250d6">SA</text>')
            xr = xR
            for c, indel, call in r["marks"]:
                cx = x0 - (locus.start - c) * fpx if c < locus.start else xr + (c - locus.end) * fpx
                col = HAP_COLOUR.get(call) if call else "none"
                stroke = HAP_COLOUR[0] if not call else col
                if indel:
                    out.append(f'<path d="M{cx - 3:.1f},{ry + ROW} L{cx + 3:.1f},{ry + ROW} L{cx:.1f},{ry - 1} Z" '
                               f'fill="{col}" stroke="{stroke}" stroke-width="0.8"/>')
                elif call:
                    out.append(f'<rect x="{cx - 0.6:.1f}" y="{ry - 1}" width="1.2" height="{ROW + 2}" fill="{col}"/>')
                else:                                   # neither allele: hollow, with a visible border
                    out.append(f'<rect x="{cx - 1.2:.1f}" y="{ry - 0.5}" width="2.4" height="{ROW + 1}" fill="none" '
                               f'stroke="{stroke}" stroke-width="1"/>')
            y += ROW + GAP
        if box_top is not None:                         # same columns inside and outside the box
            out.append(f'<rect x="{LAB - 42:.1f}" y="{box_top:.1f}" width="{xR + FLW + 6 - (LAB - 42):.1f}" '
                       f'height="{box_bot - box_top:.1f}" fill="none" stroke="#52514e" stroke-width="1.1" '
                       f'stroke-dasharray="4,2.5"/>')
            y = max(y, box_bot - 2 + 1)
        y = max(y, label_bottom)
        # the haplotype ribbon, one column, split where the rows' phase evidence changes: > 2 vars solid,
        # <= 2 vars lighter, by length light and dashed
        runs = []
        for top_, bot_, inf_ in ribbon:
            if runs and runs[-1][2] == inf_:
                runs[-1][1] = bot_
            else:
                runs.append([top_, bot_, inf_])
        for top_, bot_, inf_ in runs:
            out.append(_ribbon(12, top_, 6, bot_ - top_, HAP_COLOUR[h], inf_ if h in (1, 2) else "many"))
        y += 14

    # motif block: the units in rows, rotated to line up; columns where all agree in plain ink, the
    # differing characters bold in their motif's colour
    y += 6
    out.append(f'<text x="16" y="{y + 9}" fill="#444">motif units</text>')
    cols = list(zip(*block)) if block else []
    agree = [len({c for c in col if c != "-"}) <= 1 for col in cols]
    for k, row in enumerate(block):
        out.append(f'<rect x="96" y="{y}" width="12" height="10" fill="{colour(k)}"/>')
        shown = row if len(row) <= 150 else row[:150] + "…"
        spans = "".join(html.escape(c) if (j >= len(agree) or agree[j]) else
                        f'<tspan font-weight="bold" fill="{colour(k)}">{html.escape(c)}</tspan>'
                        for j, c in enumerate(shown))
        tag = "  (found in reads)" if k >= len(locus.motifs) else ""
        out.append(f'<text x="114" y="{y + 9}" font-family="DejaVu Sans Mono, Menlo, Consolas, monospace" '
                   f'font-size="11" xml:space="preserve">{spans}<tspan fill="#898781">{tag}</tspan></text>')
        y += 14
    if len(motifs) > len(MOTIF_COLOURS):
        out.append(f'<text x="114" y="{y + 9}" fill="#898781">further motifs drawn in grey</text>')
        y += 14

    # legend
    y += 8
    x = 16
    caps = sorted({max_mismatches(len(m)) for m in motifs} - {0})
    hp = any(len(m) >= HP_VARIANT_MIN for m in motifs)
    caps = sorted(set(caps) | ({1} if hp else set()))
    items = [("light", colour(0), "unit within " + "/".join(map(str, range(1, max(caps) + 1))) +
              " mismatches" + (" or ±1 homopolymer base" if hp else "") + " (lighter)")] if caps else []
    items += [
             ("rect", INTERRUPTION, "interruption"), ("rect", FLANK, "read flank"),
             ("dash", MUTED, "gap / repeat end → aligned flank"), ("out", OUTLIER, "outlier read (dashed box)"),
             ("rib_many", HAP_COLOUR[1], "Haplotype (>2 vars)"), ("rib_few", HAP_COLOUR[1], "Haplotype (≤2 vars)"),
             ("rib_len", HAP_COLOUR[1], "Haplotype (by length)"),
             ("tick", HAP_COLOUR[1], "het SNV, hap 1 allele"), ("tick", HAP_COLOUR[2], "het SNV, hap 2 allele"),
             ("tri", HAP_COLOUR[1], "het indel, hap 1 allele"), ("tri", HAP_COLOUR[2], "het indel, hap 2 allele"),
             ("tick", "none", "neither allele"),
             ("num", "#52514e", "+n / −n: read length minus its haplotype's allele (bp)"),
             ("sa", "#6250d6", "SA: read has supplementary alignments; its soft-clipped part in violet"),
             ("seg", "#6250d6", "segmental duplication (track under the reference)"),
             ("core", "#0b0b0b", "read behind a local length call (TRGT length mismatch)")]
    items += [("tag:" + short, col, f"{long_} (per-read tag)") for short, long_, col in sorted(set((read_tags or {}).values()))]
    for kind, col, name in items:
        if kind.startswith("tag:"):
            out.append(f'<text x="{x + 1}" y="{y + 8.5}" font-size="8" font-weight="bold" fill="{col}">'
                       f'{html.escape(kind[4:])}</text>')
        elif kind == "core":
            out.append(f'<rect x="{x}" y="{y}" width="12" height="10" fill="{FLANK}"/>'
                       f'<line x1="{x}" y1="{y + 5}" x2="{x + 12}" y2="{y + 5}" stroke="{col}" stroke-width="1.2"/>')
        elif kind == "rect":
            out.append(f'<rect x="{x}" y="{y}" width="12" height="10" fill="{col}"/>')
        elif kind == "light":
            for j, c in enumerate(range(1, max(caps) + 1)):
                out.append(f'<rect x="{x + 7 * j}" y="{y}" width="6" height="10" fill="{col}" '
                           f'fill-opacity="{MISMATCH_OPACITY[c]}"/>')
            x += 7 * (max(caps) - 1)
        elif kind.startswith("rib_"):
            out.append(_ribbon(x + 3, y, 5, 10, col, kind[4:]))
        elif kind == "out":
            out.append(f'<path d="M{x + 1},{y - 0.5} L{x + 1},{y + 10.5} L{x + 11},{y + 5} Z" fill="{col}" '
                       f'stroke="#0b0b0b" stroke-width="0.8"/>')
        elif kind == "sa":
            out.append(f'<text x="{x + 1}" y="{y + 8.5}" font-size="8" font-weight="bold" fill="{col}">SA</text>')
        elif kind == "seg":
            out.append(f'<rect x="{x}" y="{y + 2}" width="12" height="6" fill="{col}" fill-opacity="0.75"/>')
        elif kind == "num":
            out.append(f'<text x="{x + 12}" y="{y + 8.5}" text-anchor="end" font-size="8" fill="{col}">+3</text>')
        elif kind == "dash":
            out.append(f'<line x1="{x}" y1="{y + 5}" x2="{x + 12}" y2="{y + 5}" stroke="{col}" stroke-width="0.8" '
                       f'stroke-dasharray="3,2"/>')
        elif kind == "tick" and col != "none":
            out.append(f'<rect x="{x + 5.4}" y="{y - 1}" width="1.2" height="12" fill="{col}"/>')
        elif kind == "tick":
            out.append(f'<rect x="{x + 4.8}" y="{y - 0.5}" width="2.4" height="11" fill="none" '
                       f'stroke="{HAP_COLOUR[0]}" stroke-width="1"/>')
        else:
            out.append(f'<path d="M{x},{y + 10} L{x + 10},{y + 10} L{x + 5},{y} Z" fill="{col}"/>')
        out.append(f'<text x="{x + 16}" y="{y + 9}">{name}</text>')
        x += 34 + 6.2 * len(name)
        if x > W - 160:
            x, y = 16, y + 18
    H = int(y + 46)                                     # the page ends below the legend
    out[0] = out[0].replace(f'height="0" viewBox="0 0 {W} 0"', f'height="{H}" viewBox="0 0 {W} {H}"')
    out[1] = f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>'
    out.append(f'<text x="16" y="{H - 10}" fill="#888">tandemo pileup · repeat on a bp scale ({bp:.2f} px/bp), '
               f'flanks compressed ({view:,} bp = {FLW} px) · reads aligned at the repeat start' +
               (f'; reads aligned to their germline (TRGT) allele ± {flank} bp (global affine): germline allele as the columns, deletions dashed, insertions inline in their own row (it extends right); outliers gapless'
                if any(g["allele"] is not None for g in groups) else '') + '; right flanks aligned</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- driver

def _label(lid: str) -> str:
    """'HD_HTT_pathogenic' -> 'HTT (HD)'; gene IDs stay as they are."""
    if lid.endswith("_pathogenic"):
        parts = lid[: -len("_pathogenic")].split("_")
        if len(parts) >= 2:
            return f"{parts[-1]} ({'_'.join(parts[:-1])})"
    return lid


def run(bam: str, reference: str, catalog: str, out_dir: str, phased_vcf: str | None = None,
        trgt_vcf: str | None = None, loci_names=None, regions=None, sample: str | None = None,
        view: int = 10_000, max_reads: int | None = None, min_reads: int = 4, top: int | None = None,
        exclude_catalog: str | None = None, params: Params | None = None, phase: PhaseParams | None = None,
        highlight: dict | None = None, labels: dict | None = None, local: dict | None = None,
        read_tags: dict | None = None, pads: dict | None = None,
        genes: str | None = None,
        log=print) -> list[dict]:
    """highlight: {locus id: read names} drawn with the outlier triangle (and always among the rows shown).
    labels: {locus id: {1: L1, 2: L2}}, the haplotype alleles each row's deviation is measured from.
    local: {locus id: haplotypes} whose TRGT allele is not trusted: their allele row stays the TRGT allele, labelled with the
             reads' consensus length, and the reads behind it get a black line in their flanks.
    pads: {locus id: bp}, the extraction's measurement pad per locus, so each row's Y matches its extraction.
    read_tags: {read name: (short, long, colour)}: short drawn after each row, long in the legend
    (e.g. the tissue a read came from).
    genes: gene-table prefix for the title's gene context (default: the reference's genome, if it has one)."""
    from .genes import context as gene_context, for_reference
    from . import segdup as sd
    genes = genes or for_reference(reference)
    seg_path = sd.for_reference(reference)
    highlight = highlight or {}
    p = params or Params()
    phase = phase or PhaseParams()
    loci = read_catalog(catalog, regions, max_len=1 << 62)
    if loci_names:
        want = {n.upper() for n in loci_names}
        loci = [L for L in loci if L.id.upper() in want or _label(L.id).split(" ")[0].upper() in want]
    trgt = {}
    for v in ([trgt_vcf] if isinstance(trgt_vcf, str) else trgt_vcf or []):
        trgt.update(read_trgt(v, sample, {L.id for L in loci}))
    os.makedirs(out_dir, exist_ok=True)
    smp = sample or os.path.basename(bam).split(".")[0]
    excl = {}
    if exclude_catalog and p.indel_phasing:            # het indels inside any catalog repeat are not markers
        for L in read_catalog(exclude_catalog, sorted({L.chrom for L in loci}), max_len=1 << 62):
            iv = excl.setdefault(L.chrom, [])
            if iv and L.start <= iv[-1][1]:
                iv[-1] = (iv[-1][0], max(iv[-1][1], L.end))
            else:
                iv.append((L.start, L.end))
    done = []
    with pysam.AlignmentFile(bam) as bf, pysam.FastaFile(reference) as fa:
        for L in loci:
            if L.chrom not in fa.references:
                continue
            span = 100_000
            sites = load_het_sites(phased_vcf, L.chrom, sample, max(L.start - span, 0), L.end + span,
                                   reference if p.indel_phasing else None, excl.get(L.chrom)) \
                if phased_vcf else HetSites()
            t = trgt.get(L.id, {})
            ps = locus_ps(sites, L.start, L.end, t.get("ps") if t.get("phased") else None)
            rows = collect(bf, fa, L, sites, ps, p, phase, view, (pads or {}).get(L.id))
            if len(rows) < min_reads:
                log(f"{L.id}: {len(rows)} spanning reads, skipped")
                continue
            ref_rep = fa.fetch(L.chrom, L.start, L.end).upper()
            summ = summarise(rows, L, t.get("al", ()), ref_rep)
            fn = f"{L.id}.svg"
            hl = set(highlight.get(L.id, ()))
            left = fa.fetch(L.chrom, max(L.start - p.flank, 0), L.start).upper()
            right = fa.fetch(L.chrom, L.end, L.end + p.flank).upper()
            groups = plan_groups(rows, t, ps, left, right, max_reads, hl, L.period, (local or {}).get(L.id, ()))
            with open(os.path.join(out_dir, fn), "w") as fh:
                fh.write(render(L, groups, summ, smp, view, ref_rep, _label(L.id), hl, p.flank,
                                (labels or {}).get(L.id), gene_context(genes, L.chrom, L.start, L.end),
                                sd.intervals(seg_path, L.chrom, L.start - view, L.end + view), read_tags))
            done.append(dict(locus=L, label=_label(L.id), file=fn, **summ))
            log(f"{L.id}: {len(rows)} reads -> {fn}")
    done.sort(key=lambda d: -d["dev_units"])
    if top:
        keep = {d["file"] for d in done[:top]}
        for d in done[top:]:
            os.remove(os.path.join(out_dir, d["file"]))
        done = [d for d in done if d["file"] in keep]
    _index(done, out_dir, smp)
    return done


def _index(done, out_dir, sample):
    rows = []
    for d in done:
        L, per = d["locus"], d["per"]
        cell = lambda h: (f'{per[h]["median"]:.0f} ({per[h]["n"]})' if h in per else "–")
        rows.append(f'<tr><td><a href="#{html.escape(L.id)}">{html.escape(d["label"])}</a></td>'
                    f'<td>{L.chrom}:{L.start + 1:,}</td><td>{html.escape(",".join(d["motifs"]))}</td>'
                    f'<td>{L.ref_len}</td><td>{cell(1)}</td><td>{cell(2)}</td><td>{cell(0)}</td>'
                    f'<td>{d["dev_units"]:.1f}</td></tr>')
    figs = "\n".join(f'<h2 id="{html.escape(d["locus"].id)}">{html.escape(d["label"])}</h2>'
                     f'<img src="{html.escape(d["file"])}" alt="{html.escape(d["label"])} pileup">' for d in done)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Disease repeat pileups</title>
<style>body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;color:#0b0b0b;background:#f9f9f7}}table{{border-collapse:collapse}}
td,th{{padding:3px 10px;border-bottom:1px solid #e1e0d9;text-align:left;font-variant-numeric:tabular-nums}}img{{max-width:100%;border:1px solid #eee}}</style>
</head><body><h1>Disease repeat pileups · {html.escape(sample)}</h1>
<p>Ranked by the largest distance of a haplotype's median length from the reference, in motif units.
Median bp (reads) per haplotype.</p>
<table><tr><th>locus</th><th>position</th><th>motifs</th><th>ref bp</th><th>hap 1</th><th>hap 2</th>
<th>unphased</th><th>max |Δ| units</th></tr>{''.join(rows)}</table>{figs}</body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write(doc)
