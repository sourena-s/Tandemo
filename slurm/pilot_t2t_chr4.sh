#!/usr/bin/env bash
# tandemo pilot on Snellius: HG002 (Revio SPRQ, ~74x, XY) on T2T-CHM13v2.0, chr4, Platinum TRs v1.0.
# Inputs already exist: the T2T alignment / DeepVariant / HiPhase run (scratch-shared/$USER/hg002_t2t) and TRGT
# on chr4 (scratch-shared/$USER/trgt_t2t). Three jobs, each on the smallest allocation that fits it:
#   extract  genoa, 48 cores (1/4 node)       ~1.5 core-h measured on 1 Mb x 193      ~2 SBU used
#   train    gpu_h100, 1 GPU + 16 cores       network 1 on ~40 M reads, early stop     GPU budget
#   fit      genoa, 96 cores (1/2 node)       ~1 core-s per locus x 505 k loci         ~140 SBU used
# SBUs are charged for the time used, not the limit; limits are set ~2x the estimate.
#   bash slurm/pilot_t2t_chr4.sh [out dir]
set -euo pipefail
V=/projects/prjs2178/software/venvs/tandemo
T=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0
REF=$T/chm13v2.0_maskedY_rCRS.fa
H=/scratch-shared/$USER/hg002_t2t
BAM=$H/hiphase/HG002.haplotagged.bam
PHASED=$H/hiphase/HG002.deepvariant.phased.vcf.gz
TRGT=/scratch-shared/$USER/trgt_t2t/HG002/tandems.chr4.sorted.vcf.gz
CAT=$T/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz
O=${1:-/scratch-shared/$USER/tandemo_pilot_t2t/chr4}
mkdir -p $O/logs $O/ex $O/fit
cd $O

ex=$(sbatch --parsable -J tdm-extract-chr4 -p genoa -c 48 --mem=$((48 * 1792))M -t 01:00:00 -o logs/extract.%j.out \
    --wrap "set -euo pipefail; $V/bin/tandemo extract --bam $BAM --reference $REF --catalog $CAT \
        --phased-vcf $PHASED --trgt-vcf $TRGT --regions chr4 --out ex/HG002.chr4 --threads 48")
# torch 2.15 is the CUDA 13.2 build: fail at once (not after burning GPU hours on the CPU) if the driver lacks it
tr=$(sbatch --parsable -J tdm-train-chr4 -p gpu_h100 --gpus=1 -c 16 -t 02:00:00 --dependency=afterok:$ex \
    -o logs/train.%j.out \
    --wrap "set -euo pipefail; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        $V/bin/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"; print(torch.cuda.get_device_name())'
        $V/bin/tandemo train --input ex/HG002.chr4 --out net1.chr4.pt --device cuda
        $V/bin/tandemo train --input ex/HG002.chr4 --out noise.chr4.npz --empirical")
fi_=$(sbatch --parsable -J tdm-fit-chr4 -p genoa -c 96 --mem=$((96 * 1792))M -t 04:00:00 --dependency=afterok:$tr \
    -o logs/fit.%j.out \
    --wrap "set -euo pipefail; $V/bin/tandemo fit --input ex/HG002.chr4 --model net1.chr4.pt --out fit/HG002.chr4 \
        --subject HG002 --threads 96")
echo -e "extract\t$ex\ntrain\t$tr\nfit\t$fi_" | tee jobs.tsv
