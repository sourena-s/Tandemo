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
    chrom  start-1  end  ID=<id>;MOTIFS=<m1>[,<m2>...];STRUC=(<m1>)n[(<m2>)n...]
    ID      chiu2026_<chrom>_<start0>_<end>
    MOTIFS  the consensus motif, then every other observed motif carried by at least --min-motif-frac of the called
            haplotypes (most frequent first); rotations of a motif already listed are folded into it (TRGT counts
            motifs in any rotation), and motifs of other lengths are kept
    STRUC   the motifs in that order, each (motif)n, as PlatinumTRs writes multi-motif loci; the catalogue gives no
            order along the repeat, so the order is consensus-first, not positional

Kept as they are, and reported in the summary: loci overlapping another locus (about 2% sit inside a larger one;
TRGT genotypes each locus separately, so their reads are counted twice) and loci whose end runs past the chromosome
(none expected; such a locus would be dropped).

Usage (on Snellius: slurm/chiu_to_trgt.sh):
    python scripts/chiu_to_trgt.py --catalog chm13.v2.bed.gz --reference chm13v2.0_maskedY_rCRS.fa \
        --out chm13v2.0.chiu2026.v2.trgt.bed.gz [--min-motif-frac 0.1]
"""
import argparse
import collections
import gzip
import os
import re

import pysam

MOTIF_FREQ = re.compile(r"([ACGTN]+)\((\d+)\)")          # one entry of motif_frequency: MOTIF(haplotypes)


def rotations(m: str) -> set:
    """Every rotation of a motif: CAG, AGC, GCA are one motif to TRGT."""
    return {m[i:] + m[:i] for i in range(len(m))}


def motifs_of(consensus: str, freq: str, calls: int, min_frac: float) -> list[str]:
    """The locus's motif list: the consensus first, then the other motifs seen in at least min_frac of the called
    haplotypes, most frequent first, each once up to rotation."""
    out = [consensus.upper()]
    seen = rotations(out[0])
    alt = sorted(((int(n), m.upper()) for m, n in MOTIF_FREQ.findall(freq)), key=lambda x: (-x[0], x[1]))
    for n, m in alt:
        if "N" in m or m in seen or calls <= 0 or n < min_frac * calls:
            continue
        out.append(m)
        seen |= rotations(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", required=True, help="Chiu et al. chm13.v2.bed.gz (1-based, inclusive)")
    ap.add_argument("--reference", required=True, help="T2T-CHM13v2.0 FASTA (indexed), for the chromosome lengths")
    ap.add_argument("--out", required=True, help="output TRGT catalogue, .bed.gz (bgzipped and tabix-indexed)")
    ap.add_argument("--min-motif-frac", type=float, default=0.1,
                    help="an alternative motif joins MOTIFS when this fraction of called haplotypes carries it")
    a = ap.parse_args()

    clen = dict(zip(*(lambda f: (f.references, f.lengths))(pysam.FastaFile(a.reference))))
    rows, stats = [], collections.Counter()
    with gzip.open(a.catalog, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            chrom, s1, e1, motif = f[0], int(f[1]), int(f[2]), f[3]
            calls, freq = int(f[8]), f[9]
            s0 = s1 - 1                                  # 1-based inclusive -> 0-based half-open
            stats["loci"] += 1
            if chrom not in clen or e1 > clen[chrom] or s0 < 0 or e1 <= s0:
                stats["dropped: outside the reference"] += 1
                continue
            ms = motifs_of(motif, freq, calls, a.min_motif_frac)
            stats[f"motifs: {min(len(ms), 4)}{'+' if len(ms) >= 4 else ''}"] += 1
            info = (f"ID=chiu2026_{chrom}_{s0}_{e1};MOTIFS={','.join(ms)};"
                    f"STRUC={''.join(f'({m})n' for m in ms)}")
            rows.append((chrom, s0, e1, info))

    order = {c: i for i, c in enumerate(clen)}           # the reference's chromosome order, then position
    rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))
    last_end = {}                                        # loci overlapping an earlier one on the same chromosome
    for chrom, s0, e1, _info in rows:
        if s0 < last_end.get(chrom, -1):
            stats["overlapping a previous locus (kept)"] += 1
        last_end[chrom] = max(last_end.get(chrom, -1), e1)

    plain = a.out[:-3] if a.out.endswith(".gz") else a.out
    with open(plain, "w") as fo:
        for chrom, s0, e1, info in rows:
            fo.write(f"{chrom}\t{s0}\t{e1}\t{info}\n")
    pysam.tabix_index(plain, preset="bed", force=True)   # bgzip (plain -> plain.gz) and .tbi; removes plain
    stats["written"] = len(rows)
    with open(a.out[:-7] + ".summary.tsv" if a.out.endswith(".bed.gz") else a.out + ".summary.tsv", "w") as fs:
        for k, v in sorted(stats.items()):
            fs.write(f"{k}\t{v}\n")
    for k, v in sorted(stats.items()):
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
