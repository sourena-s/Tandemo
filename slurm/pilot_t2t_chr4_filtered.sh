#!/usr/bin/env bash
# Stage 2 on the filtered chr4 catalog of the pilot (slurm/pilot_t2t_chr4.sh): >= 50 bp, no homopolymer-only loci,
# no segmental duplications, the 68 Platinum disease loci always kept (~14% of the loci). Network 1 is the one the
# pilot trained on all chr4 loci; pass its job id so the fit waits for it.
#   extract  genoa, 16 cores     catalog read + ~70 k loci                                  ~1 SBU
#   fit      genoa, 48 cores     ~1 core-s per locus                                       ~20 SBU
#   bash slurm/pilot_t2t_chr4_filtered.sh [train job id] [pilot dir]
set -euo pipefail
V=/projects/prjs2178/software/venvs/tandemo
H=/scratch-shared/$USER/hg002_t2t
P=${2:-/scratch-shared/$USER/tandemo_pilot_t2t/chr4}
cd $P
mkdir -p logs ex fit
ex=$(sbatch --parsable -J tdm-extract-chr4f -p genoa -c 16 --mem=$((16 * 1792))M -t 00:45:00 -o logs/extract_filtered.%j.out \
    --wrap "set -euo pipefail; $V/bin/tandemo extract --genome t2t --bam $H/hiphase/HG002.haplotagged.bam \
        --phased-vcf $H/hiphase/HG002.deepvariant.phased.vcf.gz \
        --trgt-vcf /scratch-shared/$USER/trgt_t2t/HG002/tandems.chr4.sorted.vcf.gz \
        --regions chr4 --min-len 50 --no-homopolymer --no-segdup --out ex/HG002.chr4.filtered --threads 16")
fit=$(sbatch --parsable -J tdm-fit-chr4f -p genoa -c 48 --mem=$((48 * 1792))M -t 02:00:00 \
    --dependency=afterok:$ex${1:+:$1} -o logs/fit_filtered.%j.out \
    --wrap "set -euo pipefail; $V/bin/tandemo fit --input ex/HG002.chr4.filtered --model net1.chr4.pt \
        --out fit/HG002.chr4.filtered --subject HG002 --threads 48")
echo -e "extract_filtered\t$ex\nfit_filtered\t$fit" | tee -a jobs.tsv
