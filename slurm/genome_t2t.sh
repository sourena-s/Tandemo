#!/usr/bin/env bash
# tandemo genome-wide on one sample, T2T-CHM13v2.0 + Platinum TRs v1.0. Inputs: the T2T alignment + HiPhase run
# and per-chromosome TRGT VCFs (slurm/trgt_t2t_genome.sh). Jobs, each on the smallest allocation that fits:
#   extract-train  genoa array, 16 cores/chrom   full catalog (7.8 M loci): network 1 sees every class   ~25 SBU
#   extract-fit    genoa array, 16 cores/chrom   >= 50 bp, no homopolymers, no segdups, + disease loci     ~5 SBU
#   train          gpu_h100, 1 GPU + 16 cores    <= PER_STRATUM loci per period x length class; network 1
#                                                then the empirical baseline (192 SBU/h, GPU budget)     ~1-2 h
#   report         genoa, 16 cores, 64 GB        network 1 figures                                        ~15 SBU
#   fit (FIT=1)    genoa array, 48 cores/chrom   ~1.8 core-s per filtered locus x 1.07 M loci            ~500 SBU
# SBUs are charged for time used; limits are ~2x the estimates.
#   [FIT=1] bash slurm/genome_t2t.sh <sample> <bam> <phased vcf> <trgt dir> [out dir]
set -euo pipefail
S=${1:?sample}; BAM=${2:?bam}; PHASED=${3:?phased vcf}; TRGT=${4:?trgt dir with tandems.<chrom>.sorted.vcf.gz}
O=${5:-/scratch-shared/$USER/tandemo_t2t/$S}
PER_STRATUM=${PER_STRATUM:-12000}
V=/projects/prjs2178/software/venvs/tandemo/bin
CHR=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
     chr21 chr22 chrX chrY)
N=$(( ${#CHR[@]} - 1 ))
mkdir -p $O/{logs,ex_train,ex_fit,fit,report}
cd $O
sb() { sbatch --parsable "$@" | grep -v '^sbatch:'; }
pick='C=('"${CHR[*]}"'); c=${C[$SLURM_ARRAY_TASK_ID]}'
common="--genome t2t --bam $BAM --phased-vcf $PHASED --sample $S --threads 16"

ext=$(sb -J tdm-ext-train-$S -p genoa -c 16 --mem=$((16 * 1792))M -t 01:30:00 --array=0-$N -o logs/ext_train.%A_%a.out \
    --wrap "set -euo pipefail; $pick
        $V/tandemo extract $common --trgt-vcf $TRGT/tandems.\$c.sorted.vcf.gz --regions \$c --out ex_train/$S.\$c")
exf=$(sb -J tdm-ext-fit-$S -p genoa -c 16 --mem=$((16 * 1792))M -t 01:00:00 --array=0-$N -o logs/ext_fit.%A_%a.out \
    --wrap "set -euo pipefail; $pick
        $V/tandemo extract $common --trgt-vcf $TRGT/tandems.\$c.sorted.vcf.gz --regions \$c \
            --min-len 50 --no-homopolymer --no-segdup --out ex_fit/$S.\$c")
inputs=$(printf "ex_train/$S.%s " "${CHR[@]}")
tr=$(sb -J tdm-train-$S -p gpu_h100 --gpus=1 -c 16 -t 03:00:00 --dependency=afterok:$ext -o logs/train.%j.out \
    --wrap "set -euo pipefail; nvidia-smi --query-gpu=name,driver_version --format=csv
        $V/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"'
        $V/tandemo train --input $inputs --out net1.pt --per-stratum $PER_STRATUM --device cuda
        $V/tandemo train --input $inputs --out noise.npz --empirical --per-stratum $PER_STRATUM")
rp=$(sb -J tdm-report-$S -p genoa -c 16 --mem=64G -t 01:30:00 --dependency=afterok:$tr -o logs/report.%j.out \
    --wrap "set -euo pipefail; $V/tandemo report --model net1.pt --baseline noise.npz --input $inputs \
        --per-stratum $PER_STRATUM --out report")
echo -e "extract_train\t$ext\nextract_fit\t$exf\ntrain\t$tr\nreport\t$rp" | tee -a jobs.tsv
if [ "${FIT:-0}" = 1 ]; then
    fi_=$(sb -J tdm-fit-$S -p genoa -c 48 --mem=$((48 * 1792))M -t 06:00:00 --array=0-$N \
        --dependency=afterok:$exf:$tr -o logs/fit.%A_%a.out \
        --wrap "set -euo pipefail; $pick
            $V/tandemo fit --input ex_fit/$S.\$c --model net1.pt --out fit/$S.\$c --subject $S --threads 48")
    echo -e "fit\t$fi_" | tee -a jobs.tsv
fi
