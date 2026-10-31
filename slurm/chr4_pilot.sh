#!/usr/bin/env bash
# chr4 pilot of the current network 1 for any T2T sample (pilot_t2t_chr4_v3.sh with the inputs as arguments):
# extraction (pads, per-haplotype phase counts, CpG 5mC) -> network 1 on one H100 (D = 5 + Pareto tails, 5mC inputs
# masked for 40% of reads) + the empirical baseline -> held-out report, heterozygous VNTR outlier scan with
# two-panel figures, and the long-allele tail plot.
#   extract   genoa, 16 cores                                                           ~3 SBU
#   trgt-check genoa: <ex>.trgt_read_discrepancy.tsv.gz, loci to keep out of results     ~3 SBU
#   train     gpu_h100, 1 GPU + 16 cores                                                ~60-100 SBU (GPU budget)
#   report / scan / tails   genoa                                                       ~10 SBU together
#   bash slurm/chr4_pilot.sh <sample> <haplotagged bam> <phased small-variant vcf> <trgt vcf> <out dir>
set -euo pipefail
S=${1:?sample}; BAM=${2:?haplotagged bam}; PHASED=${3:?phased vcf}; TV=${4:?trgt vcf}; P=${5:?out dir}
V=/projects/prjs2178/software/venvs/tandemo/bin
R=/projects/prjs2178/repo_clones/Tandemo
REF=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa
for f in $BAM $BAM.bai $PHASED $PHASED.tbi $TV $TV.tbi; do [ -s $f ] || { echo "missing $f"; exit 1; }; done
mkdir -p $P/{logs,ex,model,report,deviant,tails}
cd $P
sb() { sbatch --parsable "$@" | grep -v '^sbatch:'; }
EX=ex/$S.chr4
ex=$(sb -J tdm-$S-chr4-extract -p genoa -c 16 --mem=$((16 * 1792))M -t 02:00:00 -o logs/extract.%j.out \
    --wrap "set -euo pipefail; $V/tandemo extract --genome t2t --bam $BAM --phased-vcf $PHASED --trgt-vcf $TV \
        --sample $S --regions chr4 --out $EX --threads 16")
tc=$(sb -J tdm-$S-chr4-trgtcheck -p genoa -c 16 --mem=64G -t 00:30:00 --dependency=afterok:$ex -o logs/trgtcheck.%j.out \
    --wrap "set -euo pipefail; $V/tandemo trgt-check --input $EX")
tr=$(sb -J tdm-$S-chr4-train -p gpu_h100 --gpus=1 -c 16 -t 04:00:00 --dependency=afterok:$ex -o logs/train.%j.out \
    --wrap "set -euo pipefail; nvidia-smi --query-gpu=name,driver_version --format=csv
        $V/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"'
        $V/tandemo train --input $EX --out model/net1.pt --device cuda
        $V/tandemo train --input $EX --out model/noise.npz --empirical")
rp=$(sb -J tdm-$S-chr4-report -p genoa -c 16 --mem=64G -t 01:00:00 --dependency=afterok:$tr -o logs/report.%j.out \
    --wrap "set -euo pipefail; $V/tandemo report --model model/net1.pt --baseline model/noise.npz --input $EX \
        --per-stratum 12000 --out report")
sc=$(sb -J tdm-$S-chr4-scan -p genoa -c 8 --mem=$((24 * 1792))M -t 01:00:00 --dependency=afterok:$tr -o logs/scan.%j.out \
    --wrap "set -euo pipefail; $V/python $R/scripts/deviant_vntr.py --input $EX --model model/net1.pt \
        --out deviant/chr4 --plot 50 --bam $BAM --reference $REF --phased-vcf $PHASED --trgt-vcf $TV --sample $S")
tl=$(sb -J tdm-$S-chr4-tails -p genoa -c 24 --mem=$((24 * 1792))M -t 00:30:00 --dependency=afterok:$tr -o logs/tails.%j.out \
    --wrap "set -euo pipefail; $V/python $R/scripts/long_tails.py --input $EX --model model/net1.pt --out tails/chr4_long")
echo -e "extract\t$ex\ntrgt-check\t$tc\ntrain\t$tr\nreport\t$rp\nscan\t$sc\ntails\t$tl" | tee -a jobs.tsv
