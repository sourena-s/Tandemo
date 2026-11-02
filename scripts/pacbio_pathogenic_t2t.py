"""Place PacBio's GRCh38 disease repeats (TRGT's pathogenic_repeats.hg38.bed) on T2T-CHM13 by their flanking sequence.

No liftOver: for each locus, the GRCh38 bases on either side of the repeat are searched for, exactly, in the T2T
chromosome (both strands); a placement is a left-flank match followed within 5 kb by the right-flank match, and the
repeat is what lies between them. Flanks of 60 bp are tried first, then 30, 20 and 15 bp, so a locus whose GRCh38
flank differs by a base, or whose GRCh38 interval covers only part of the tract (TMEM185A: the bases left of it are
GCC copies, fewer on T2T), still places. Every placement is kept: a locus in a segmental duplication places once per
copy (NOTCH2NLA: two copies on 1q21), and the flanks cannot say which copy is the gene.

Output (TSV, one line per placement): gene, chrom, start, end (0-based, half-open, the bases between the flanks),
strand, flank_bp, motifs (PacBio's). Read by scripts/chiu_to_trgt.py --pacbio-pathogenic.

Usage:
    python scripts/pacbio_pathogenic_t2t.py --bed pathogenic_repeats.hg38.bed --grch38 GRCh38.fasta \
        --t2t chm13v2.0.fa --genes C11ORF80,FRA10AC1,TMEM185A,NOTCH2NLA --out pacbio_pathogenic.t2t.tsv
"""
import argparse
import re

import pysam

FLANKS = (60, 30, 20, 15)                                # flank lengths tried, longest first
WINDOW = 5000                                            # the right flank's search window after the left one


def rc(s: str) -> str:
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def place(chrom_seq: str, left: str, right: str) -> list[tuple[int, int, str]]:
    """(start, end, strand) of every left-flank match followed within WINDOW bp by the right flank, either strand
    (on the minus strand the reverse complement of the right flank comes first)."""
    out = []
    for strand, l_, r_ in (("+", left, right), ("-", rc(right), rc(left))):
        for m in re.finditer(re.escape(l_), chrom_seq):
            b = chrom_seq.find(r_, m.end(), m.end() + WINDOW)
            if b >= 0:
                out.append((m.end(), b, strand))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bed", required=True, help="PacBio's pathogenic_repeats.hg38.bed (TRGT format, ID = gene)")
    ap.add_argument("--grch38", required=True, help="the GRCh38 FASTA the BED refers to (indexed)")
    ap.add_argument("--t2t", required=True, help="T2T-CHM13v2.0 FASTA (indexed)")
    ap.add_argument("--genes", required=True, help="comma-separated IDs from the BED to place")
    ap.add_argument("--out", required=True, help="output TSV")
    a = ap.parse_args()

    want = a.genes.split(",")
    h38, t2t = pysam.FastaFile(a.grch38), pysam.FastaFile(a.t2t)
    loci = {}
    for line in open(a.bed):
        f = line.rstrip("\n").split("\t")
        kv = dict(x.split("=", 1) for x in f[3].split(";"))
        if kv["ID"] in want:
            loci[kv["ID"]] = (f[0], int(f[1]), int(f[2]), kv["MOTIFS"])
    seqs = {}
    with open(a.out, "w") as fo:
        fo.write("gene\tchrom\tstart\tend\tstrand\tflank_bp\tmotifs\n")
        for gene in want:
            if gene not in loci:
                print(f"{gene}: not in {a.bed}")
                continue
            c, s, e, motifs = loci[gene]
            if c not in seqs:
                seqs[c] = t2t.fetch(c).upper()           # the same chromosome on T2T
            hits, used = [], None
            for k in FLANKS:                             # longest flanks that place it
                hits = place(seqs[c], h38.fetch(c, s - k, s).upper(), h38.fetch(c, e, e + k).upper())
                if hits:
                    used = k
                    break
            if not hits:
                print(f"{gene}: flanks not found on T2T {c}")
            for a0, b0, strand in hits:
                fo.write(f"{gene}\t{c}\t{a0}\t{b0}\t{strand}\t{used}\t{motifs}\n")
                print(f"{gene}\t{c}:{a0}-{b0}\t{strand}\tflanks {used} bp")


if __name__ == "__main__":
    main()
