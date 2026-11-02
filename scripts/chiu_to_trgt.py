"""Convert the Chiu et al. 2026 tandem repeat catalogue (T2T-CHM13) into a TRGT catalogue.

Source: R. Chiu, I.-S. Rajan-Babu, ... I. Birol, "A comprehensive tandem repeat catalog of the human genome",
Nature Communications 17:1106 (2026), https://doi.org/10.1038/s41467-025-66153-5; catalogue v2.0, Zenodo record
14597629, file chm13.v2.bed.gz. Built from 272 long-read genomes: TRF on haplotype-resolved assemblies, loci placed on
the reference by their flanks and clustered across samples.

Input (tab-separated, no header; one line per locus):
    0 chrom   1 start   2 end           1-based and inclusive (a CACCACCAC locus of 9 bp reads 202680 202688)
    3 motif                             the locus's consensus motif (TRF)
    4 copy_numbers  5 sizes             per haplotype, ';'-separated, '-' where a haplotype has no call
    6 max_change    7 num_samples       8 num_calls (haplotypes with a call)
    9 motif_frequency                   haplotypes per observed motif, e.g. CAG(10);CAA(2)
    10 feature                          gene|transcript|element, or NA

Output (TRGT catalogue BED, 0-based half-open, sorted, bgzipped and tabix-indexed):
    chrom  start  end  ID=<id>;MOTIFS=<m1>[,<m2>...];STRUC=(<m1>)n[(<m2>)n...]

One record per source locus, except where loci sit side by side (see "Merging"):
    ID      chiu2026_<chrom>_<start0>_<end>
    MOTIFS  the consensus motif, then every other observed motif carried by at least --min-motif-frac of the called
            haplotypes (most frequent first); rotations of a motif already listed are folded into it (TRGT counts
            motifs in any rotation), and motifs of other lengths are kept
    STRUC   the motifs in that order, each (motif)n, as PlatinumTRs writes multi-motif loci; the catalogue gives no
            order along the repeat, so the order is consensus-first, not positional

Merging: TRF calls two tracts that touch as two loci (HTT's CAG and CCG tracts are two records that overlap by 35
bp; SCA8's CTA and CTG tracts end and start at the same base), so one disease repeat would get two partial calls and
the bases they share would be counted twice. Records on a chromosome are read in order of start (the longer first on
a tie); a record that starts inside or right at the end of the current group and ends past it (a partial overlap, or
adjacent) joins the group, unless the group would then exceed --max-merged bp; the group becomes one locus:
    ID      chiu2026m_<chrom>_<start0>_<end>
    MOTIFS, STRUC  from tr-solve on the merged interval's reference sequence, restricted to the members' motifs:
            each motif once, in the order it first appears along the repeat (HTT: (CAG)n(CCG)n, as PlatinumTRs
            writes it, although interruptions split the tract into many short spans); where tr-solve finds no
            structure, the members' motifs in member order
A record that lies wholly inside the current group (nested: FXN's (AAG)n inside an A-rich record, GIPC1's (CCG)n
inside a (CCGCCT)n record) is not merged: both it and the record around it stay, since the inner one is often the
disease repeat and the outer one its context; the pair is listed in the merges table. At a disease locus
(--pathogenic: a TRGT catalogue whose disease loci have IDs ending in _pathogenic, e.g. PlatinumTRs) whose motifs
the inner record shares (up to rotation or strand: FXN's AAG is its GAA), the inner record is the disease repeat and
the outer one, reaching into the flanks, gets "_nested" appended to its ID; where the inner record's motifs are not
the disease's (AFF3: the disease's GCC is the outer record's), nothing is renamed.

PacBio disease loci (--pacbio-pathogenic: their T2T placements from scripts/pacbio_pathogenic_t2t.py, for the loci of
TRGT's pathogenic_repeats.hg38.bed that PlatinumTRs has no _pathogenic locus for): the record over a placement that
carries its motif (up to rotation or strand; the most overlap if several) gets "_<GENE>_pacbio_pathogenic" appended
to its ID. Where no record does (NOTCH2NLA, absent from Chiu et al.), the --pathogenic catalogue's record(s) over the
placement are added, their ID with "_<GENE>_platinum_pacbio_pathogenic" appended. Groups that would
pass --max-merged stay as separate records, so a chain of overlaps in a repeat-dense region does not become one
long locus.

Side outputs (next to --out, the .bed.gz name with .merges.tsv / .summary.tsv):
    merges.tsv   kind (merged | nested | nested at a disease locus), the locus written (merged ID, or the outer
                 record, as written), its members (IDs)
    summary.tsv  counts: loci read, records written, merged loci and the records in them, nested pairs, groups left
                 unmerged by the cap, motifs per record

Usage (on Snellius: slurm/chiu_to_trgt.sh):
    python scripts/chiu_to_trgt.py --catalog chm13.v2.bed.gz --reference chm13v2.0_maskedY_rCRS.fa \
        --out chm13v2.0.chiu2026.v2.trgt.bed.gz [--min-motif-frac 0.1] [--max-merged 1000] [--threads 16] \
        [--pathogenic chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz] [--pacbio-pathogenic pacbio.t2t.tsv]
"""
import argparse
import collections
import gzip
import os
import re
import sys
from multiprocessing import Pool

import pysam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MOTIF_FREQ = re.compile(r"([ACGTN]+)\((\d+)\)")          # one entry of motif_frequency: MOTIF(haplotypes)
REFERENCE = None                                         # the FASTA path, for the worker processes


def rotations(m: str) -> set:
    """Every rotation of a motif: CAG, AGC, GCA are one motif to TRGT."""
    return {m[i:] + m[:i] for i in range(len(m))}


def motifs_of(consensus: str, freq: str, calls: int, min_frac: float) -> list[str]:
    """A source locus's motif list: the consensus first, then the other motifs seen in at least min_frac of the
    called haplotypes, most frequent first, each once up to rotation."""
    out = [consensus.upper()]
    seen = rotations(out[0])
    alt = sorted(((int(n), m.upper()) for m, n in MOTIF_FREQ.findall(freq)), key=lambda x: (-x[0], x[1]))
    for n, m in alt:
        if "N" in m or m in seen or calls <= 0 or n < min_frac * calls:
            continue
        out.append(m)
        seen |= rotations(m)
    return out


def same_motif(a: str, b: str) -> bool:
    """One motif up to rotation or strand (CAG = AGC = CTG)."""
    rc = b[::-1].translate(str.maketrans("ACGT", "TGCA"))
    return len(a) == len(b) and (a in rotations(b) or a in rotations(rc))


def disease_loci(path: str) -> dict:
    """{chrom: [(start, end, ID, motifs)]} of the loci whose ID ends in _pathogenic in a TRGT catalogue."""
    out = collections.defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            c, s, e, inf = line.rstrip("\n").split("\t")[:4]
            kv = dict(x.split("=", 1) for x in inf.split(";") if "=" in x)
            if kv.get("ID", "").endswith("_pathogenic"):
                out[c].append((int(s), int(e), kv["ID"], kv.get("MOTIFS", "").upper().split(",")))
    return out


def info(lid: str, ms: list[str], struc: list[str] | None = None) -> str:
    """The TRGT INFO column: ID, MOTIFS and STRUC ((m)n per entry of struc, by default per motif)."""
    return f"ID={lid};MOTIFS={','.join(ms)};STRUC={''.join(f'({m})n' for m in (struc or ms))}"


def groups(recs: list, max_merged: int):
    """Walk one chromosome's records (chrom, s0, e1, id, motifs), sorted by start and longer first: yield
    ("merged", [members]) for side-by-side records joined into one locus, ("single", [rec]) for a record left as it
    is, and ("nested", [outer, inner]) for each record lying wholly inside the group before it (kept, both)."""
    cur, outer = [], None                                # the open group, and its widest member (for nested)
    for r in recs:
        if cur:
            g0, g1 = cur[0][1], max(x[2] for x in cur)
            if r[2] <= g1:                               # wholly inside the group: nested, kept as it is
                yield "nested", [outer, r]
                yield "single", [r]
                continue
            if r[1] <= g1 and r[2] - g0 <= max_merged:   # overlapping its end or touching it: joins the group
                cur.append(r)
                outer = max(cur, key=lambda x: x[2] - x[1])
                continue
            yield ("merged" if len(cur) > 1 else "single"), cur
            if r[1] <= g1:                               # would pass the cap: left as its own record
                yield "capped", [cur[-1], r]
        cur, outer = [r], r
    if cur:
        yield ("merged" if len(cur) > 1 else "single"), cur


def structure(job):
    """MOTIFS of a merged locus from tr-solve on its reference sequence, restricted to its members' motifs: each
    motif once, in the order it first appears along the repeat (STRUC lists them in that order). None when tr-solve
    finds nothing."""
    from tandemo.pileup import resolve_structures
    chrom, s0, e1, motifs = job
    seq = pysam.FastaFile(REFERENCE).fetch(chrom, s0, e1).upper()
    st = resolve_structures([seq], [tuple(motifs)])[0]
    if not st:
        return None
    first = list(dict.fromkeys(m for m, _a, _b in sorted(st, key=lambda x: x[1])))
    return first, first


def main():
    global REFERENCE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", required=True, help="Chiu et al. chm13.v2.bed.gz (1-based, inclusive)")
    ap.add_argument("--reference", required=True, help="T2T-CHM13v2.0 FASTA (indexed): lengths, merged sequences")
    ap.add_argument("--out", required=True, help="output TRGT catalogue, .bed.gz (bgzipped and tabix-indexed)")
    ap.add_argument("--min-motif-frac", type=float, default=0.1,
                    help="an alternative motif joins MOTIFS when this fraction of called haplotypes carries it")
    ap.add_argument("--max-merged", type=int, default=1000, help="longest merged locus, bp")
    ap.add_argument("--threads", type=int, default=16, help="tr-solve workers for the merged loci")
    ap.add_argument("--pathogenic", help="TRGT catalogue with disease loci (IDs ending _pathogenic): at a nested "
                                         "pair whose inner record carries the disease motif, the outer gets _nested")
    ap.add_argument("--pacbio-pathogenic", help="T2T placements of PacBio disease loci (pacbio_pathogenic_t2t.py): "
                                                "their records get _<GENE>_pacbio_pathogenic; where none, the "
                                                "--pathogenic catalogue's records are added (_platinum_pacbio_pathogenic)")
    a = ap.parse_args()
    REFERENCE = a.reference

    clen = dict(zip(*(lambda f: (f.references, f.lengths))(pysam.FastaFile(a.reference))))
    by_chrom, stats = collections.defaultdict(list), collections.Counter()
    with gzip.open(a.catalog, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            chrom, s1, e1, motif = f[0], int(f[1]), int(f[2]), f[3]
            s0 = s1 - 1                                  # 1-based inclusive -> 0-based half-open
            stats["loci read"] += 1
            if chrom not in clen or e1 > clen[chrom] or s0 < 0 or e1 <= s0:
                stats["dropped: outside the reference"] += 1
                continue
            ms = motifs_of(motif, f[9], int(f[8]), a.min_motif_frac)
            by_chrom[chrom].append((chrom, s0, e1, f"chiu2026_{chrom}_{s0}_{e1}", ms))

    dis = disease_loci(a.pathogenic) if a.pathogenic else {}
    rows, table, merged = [], [], []                     # records out (chrom, s0, e1, ID, MOTIFS, STRUC), the
    nested_out = set()                                   # merges table, merged loci to solve; outer records at a
    for chrom in sorted(by_chrom, key=list(clen).index): # disease locus (renamed _nested)
        recs = sorted(by_chrom[chrom], key=lambda r: (r[1], -r[2]))
        for kind, mem in groups(recs, a.max_merged):
            if kind == "nested":
                outer, inner = mem
                hit = [d for d in dis.get(chrom, ()) if d[0] < inner[2] and inner[1] < d[1] and
                       any(same_motif(m, dm) for m in inner[4] for dm in d[3])]
                if hit:                                  # the inner record is the disease repeat
                    nested_out.add(outer[3])
                    table.append((f"nested at {hit[0][2]}", outer[3], [inner[3]]))
                    stats["nested pairs at a disease locus (outer renamed _nested)"] += 1
                else:
                    table.append(("nested", outer[3], [inner[3]]))
                stats["nested pairs (both kept)"] += 1
            elif kind == "capped":
                stats["side-by-side records left apart (over --max-merged)"] += 1
            elif kind == "single":
                rows.append((chrom, mem[0][1], mem[0][2], mem[0][3], mem[0][4], None))
            else:
                s0, e1 = mem[0][1], max(x[2] for x in mem)
                ms = list(dict.fromkeys(m for x in mem for m in x[4]))
                merged.append((chrom, s0, e1, f"chiu2026m_{chrom}_{s0}_{e1}", ms, [x[3] for x in mem]))
    with Pool(a.threads) as p:                           # merged loci: their structure, from the reference
        solved = p.map(structure, [(c, s, e, ms) for c, s, e, _i, ms, _m in merged], chunksize=200)
    for (chrom, s0, e1, lid, ms, mem), sv in zip(merged, solved):
        if any(x in nested_out for x in mem):            # a merged locus around a disease repeat
            nested_out.add(lid)
        if sv:
            rows.append((chrom, s0, e1, lid, sv[0], sv[1]))
        else:                                            # no structure: the members' motifs in member order
            rows.append((chrom, s0, e1, lid, ms, None))
            stats["merged loci without a tr-solve structure"] += 1
        table.append(("merged", lid, mem))
        stats["merged loci"] += 1
        stats["records merged into them"] += len(mem)

    renamed = {}                                         # ID -> ID with a PacBio disease label
    if a.pacbio_pathogenic:
        place = [line.rstrip("\n").split("\t") for line in open(a.pacbio_pathogenic)][1:]
        plat = collections.defaultdict(list)             # the --pathogenic catalogue's records on those chromosomes
        if a.pathogenic:
            chroms = {p_[1] for p_ in place}
            with gzip.open(a.pathogenic, "rt") as fh:
                for line in fh:
                    f = line.rstrip("\n").split("\t")
                    if f[0] in chroms:
                        kv = dict(x.split("=", 1) for x in f[3].split(";") if "=" in x)
                        plat[f[0]].append((int(f[1]), int(f[2]), kv["ID"], kv["MOTIFS"].split(","),
                                           re.findall(r"\(([ACGT]+)\)n", kv["STRUC"])))
        for gene, chrom, ps, pe, _strand, _k, pm in place:
            ps, pe, pm = int(ps), int(pe), pm.upper().split(",")
            ov = lambda s_, e_: min(e_, max(pe, ps + 1)) - max(s_, ps)   # overlap (a 0-bp placement counts as 1 bp)
            cand = [r for r in rows if r[0] == chrom and ov(r[1], r[2]) > 0 and
                    any(same_motif(m, x) for m in r[4] for x in pm)]
            if cand:
                r = max(cand, key=lambda r: ov(r[1], r[2]))
                renamed[r[3]] = f"{r[3]}_{gene}_pacbio_pathogenic"
                table.append((f"pacbio pathogenic {gene}", r[3], [f"{chrom}:{ps}-{pe}"]))
                stats["PacBio disease loci labelled on a record"] += 1
            else:                                        # none: the --pathogenic catalogue's record(s) there
                for s_, e_, pid, ms, st in plat.get(chrom, ()):
                    if ov(s_, e_) > 0:
                        lid = f"{pid}_{gene}_platinum_pacbio_pathogenic"
                        rows.append((chrom, s_, e_, lid, ms, st or None))
                        table.append((f"pacbio pathogenic {gene} (added from --pathogenic)", lid, [f"{chrom}:{ps}-{pe}"]))
                        stats["PacBio disease records added from --pathogenic"] += 1

    order = {c: i for i, c in enumerate(clen)}           # the reference's chromosome order, then position
    rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))
    name = lambda lid: renamed.get(lid, lid + "_nested" if lid in nested_out else lid)
    for _c, _s, _e, _i, ms, _st in rows:
        stats[f"motifs per record: {min(len(ms), 4)}{'+' if len(ms) >= 4 else ''}"] += 1
    plain = a.out[:-3] if a.out.endswith(".gz") else a.out
    with open(plain, "w") as fo:
        for chrom, s0, e1, lid, ms, st in rows:
            fo.write(f"{chrom}\t{s0}\t{e1}\t{info(name(lid), ms, st)}\n")
    pysam.tabix_index(plain, preset="bed", force=True)   # bgzip (plain -> plain.gz) and .tbi; removes plain
    stats["records written"] = len(rows)
    stem = a.out[:-7] if a.out.endswith(".bed.gz") else a.out
    with open(stem + ".merges.tsv", "w") as fm:
        fm.write("kind\tlocus\tmembers\n")
        for kind, lid, mem in table:
            fm.write(f"{kind}\t{name(lid)}\t{','.join(mem)}\n")
    with open(stem + ".summary.tsv", "w") as fs:
        for k, v in sorted(stats.items()):
            fs.write(f"{k}\t{v}\n")
    for k, v in sorted(stats.items()):
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
