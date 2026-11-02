#!/bin/bash
# Chiu et al. 2026 tandem repeat catalogue (T2T-CHM13, Zenodo 14597629) -> TRGT catalogue, next to the source:
#   $D/chm13v2.0.chiu2026.v2.trgt.bed.gz (+ .tbi), .merges.tsv (merged and nested loci) and .summary.tsv
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
export TANDEMO_TRSOLVE=/projects/prjs2178/software/trtools/tr-solve-v0.2.0-linux_x86_64   # merged loci's STRUC
$PY $R/scripts/chiu_to_trgt.py --catalog $D/chm13.v2.bed.gz --reference $REF --out $OUT --min-motif-frac 0.1 \
    --max-merged 1000 --threads 16 \
    --pathogenic $T2T/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz     # disease loci: _nested
# check: every record has ID, MOTIFS and STRUC, motifs are ACGT, STRUC is (m)n entries of MOTIFS, the interval fits
# its chromosome
$PY - "$OUT" "$REF" <<'EOF'
import gzip, re, sys, pysam
out, ref = sys.argv[1:]
L = dict(zip(*(lambda f: (f.references, f.lengths))(pysam.FastaFile(ref))))
n = bad = 0
for line in gzip.open(out, "rt"):
    c, s, e, info = line.rstrip("\n").split("\t")
    kv = dict(x.split("=", 1) for x in info.split(";"))
    ms = kv["MOTIFS"].split(",")
    st = re.findall(r"\(([ACGT]+)\)n", kv.get("STRUC", ""))
    ok = (set(kv) == {"ID", "MOTIFS", "STRUC"} and re.fullmatch(r"chiu2026m?_chr\w+_\d+_\d+(_nested)?", kv["ID"]) and all(re.fullmatch("[ACGT]+", m) for m in ms) and st
          and "".join(f"({m})n" for m in st) == kv["STRUC"] and set(st) <= set(ms) and 0 <= int(s) < int(e) <= L[c])
    n += 1; bad += not ok
print(f"checked {n} records, {bad} malformed")
sys.exit(1 if bad else 0)
EOF
ls -la $D
