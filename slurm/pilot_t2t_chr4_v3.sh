#!/usr/bin/env bash
# chr4 pilot with the current network 1: per-bp core D = 5 with generalised Pareto tails, CpG 5mC inputs (flanks,
# repeat, call certainty; masked for 40% of reads per batch so they are optional at inference), labels on the
# read's Y scale (TRGT allele + pad indels), het SNV + indel phase evidence with per-haplotype site counts.
#   extract   genoa, 16 cores, all chr4 loci (pads, n_hap1 / n_hap2, 5mC)              ~3 SBU
#   train     gpu_h100, 1 GPU + 16 cores: network 1, then the empirical baseline         ~60 SBU (GPU budget)
#   report    genoa, 16 cores, 64 GB: held-out figures                                  ~5 SBU
#   scan      genoa, 8 cores: heterozygous VNTR outliers (confident phase >= 3 supporting sites, < 1/5 conflicting;
#             phase-based and phase-free scores) with two-panel figures                ~3 SBU
#   tails     genoa, 24 cores: per-read error tails at alleles >= 320 bp                ~2 SBU
#   bash slurm/pilot_t2t_chr4_v3.sh [pilot dir]
set -euo pipefail
V=/projects/prjs2178/software/venvs/tandemo/bin
R=/projects/prjs2178/repo_clones/Tandemo
H=/scratch-shared/$USER/hg002_t2t
TV=/scratch-shared/$USER/trgt_t2t/HG002/tandems.chr4.sorted.vcf.gz
REF=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa
P=${1:-/scratch-shared/$USER/tandemo_pilot_t2t/chr4}
cd $P
mkdir -p logs ex v3
sb() { sbatch --parsable "$@" | grep -v '^sbatch:'; }
EX=ex/HG002.chr4.v3
ex=$(sb -J tdm-v3-extract -p genoa -c 16 --mem=$((16 * 1792))M -t 01:00:00 -o logs/v3_extract.%j.out \
    --wrap "set -euo pipefail; $V/tandemo extract --genome t2t --bam $H/hiphase/HG002.haplotagged.bam \
        --phased-vcf $H/hiphase/HG002.deepvariant.phased.vcf.gz --trgt-vcf $TV --regions chr4 --out $EX --threads 16")
tr=$(sb -J tdm-v3-train -p gpu_h100 --gpus=1 -c 16 -t 02:30:00 --dependency=afterok:$ex -o logs/v3_train.%j.out \
    --wrap "set -euo pipefail; nvidia-smi --query-gpu=name,driver_version --format=csv
        $V/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"'
        $V/tandemo train --input $EX --out v3/net1.pt --device cuda
        $V/tandemo train --input $EX --out v3/noise.npz --empirical")
rp=$(sb -J tdm-v3-report -p genoa -c 16 --mem=64G -t 01:00:00 --dependency=afterok:$tr -o logs/v3_report.%j.out \
    --wrap "set -euo pipefail; $V/tandemo report --model v3/net1.pt --baseline v3/noise.npz --input $EX \
        --per-stratum 12000 --out v3/report")
sc=$(sb -J tdm-v3-scan -p genoa -c 8 --mem=$((24 * 1792))M -t 01:00:00 --dependency=afterok:$tr -o logs/v3_scan.%j.out \
    --wrap "set -euo pipefail; mkdir -p v3/deviant; $V/python $R/scripts/deviant_vntr.py --input $EX --model v3/net1.pt \
        --out v3/deviant/chr4 --bam $H/hiphase/HG002.haplotagged.bam --reference $REF \
        --phased-vcf $H/hiphase/HG002.deepvariant.phased.vcf.gz --trgt-vcf $TV --sample HG002")
tl=$(sb -J tdm-v3-tails -p genoa -c 24 --mem=$((24 * 1792))M -t 00:30:00 --dependency=afterok:$tr -o logs/v3_tails.%j.out \
    --wrap "set -euo pipefail; mkdir -p v3/tails; $V/python $R/scripts/long_tails.py --input $EX --model v3/net1.pt \
        --out v3/tails/chr4_long")
echo -e "v3_extract\t$ex\nv3_train\t$tr\nv3_report\t$rp\nv3_scan\t$sc\nv3_tails\t$tl" | tee -a jobs.tsv
