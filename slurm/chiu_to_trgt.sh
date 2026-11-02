#!/bin/bash
# Chiu et al. 2026 tandem repeat catalogue (T2T-CHM13, Zenodo 14597629) -> TRGT catalogue, next to the source:
#   $D/chm13v2.0.chiu2026.v2.trgt.bed.gz (+ .tbi) and $D/chm13v2.0.chiu2026.v2.trgt.summary.tsv
# then a check that every record parses and its coordinates sit inside the reference.
#   sbatch slurm/chiu_to_trgt.sh
#SBATCH -J chiu2trgt
#SBATCH -p rome
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /projects/0/prjs2178/seed/pacbio/logs/chiu_to_trgt.%j.out
set -euo pipefail
R=/projects/prjs2178/repo_clones/Tandemo
PY=/projects/prjs2178/software/venvs/tandemo/bin/python
T2T=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0
D=$T2T/chiu2026_tr_catalog
REF=$T2T/chm13v2.0_maskedY_rCRS.fa
OUT=$D/chm13v2.0.chiu2026.v2.trgt.bed.gz
$PY $R/scripts/chiu_to_trgt.py --catalog $D/chm13.v2.bed.gz --reference $REF --out $OUT --min-motif-frac 0.1
# check: every record has ID, MOTIFS and STRUC, motifs are ACGT, and the interval fits its chromosome
$PY - "$OUT" "$REF" <<'EOF'
import gzip, re, sys, pysam
out, ref = sys.argv[1:]
L = dict(zip(*(lambda f: (f.references, f.lengths))(pysam.FastaFile(ref))))
n = bad = 0
for line in gzip.open(out, "rt"):
    c, s, e, info = line.rstrip("\n").split("\t")
    kv = dict(x.split("=", 1) for x in info.split(";"))
    ms = kv["MOTIFS"].split(",")
    ok = (set(kv) == {"ID", "MOTIFS", "STRUC"} and all(re.fullmatch("[ACGT]+", m) for m in ms)
          and kv["STRUC"] == "".join(f"({m})n" for m in ms) and 0 <= int(s) < int(e) <= L[c])
    n += 1; bad += not ok
print(f"checked {n} records, {bad} malformed")
sys.exit(1 if bad else 0)
EOF
ls -la $D
