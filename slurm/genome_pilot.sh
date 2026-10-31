#!/usr/bin/env bash
# tandemo on every chromosome of one T2T sample (the chr4 pilot chain, genome-wide). Training samples loci per
# period x length class (PER_STRATUM, default 15000: rare classes such as long VNTRs whole, short STRs capped) so
# the reads fit one H100's host and GPU memory; chr20-22 are the validation chromosomes.
#   extract + trgt-check  genoa array, 48 cores per chromosome (all threads used; 84 GB for trgt-check)  ~60 SBU
#   train                 gpu_h100, 1 GPU + 16 cores, wall time 12 h (charged for time used)            ~1-3 h
#   report                genoa, 24 cores, 64 GB                                                       ~10 SBU
#   scan, instability     genoa arrays, 24 cores per chromosome                                         ~100 SBU
#   clinical              genoa, 24 cores: 68 disease loci                                             ~5 SBU
#   clinical PDF          staging, 1 core: scripts/pages_to_pdf.sh (svgpdf env: librsvg + poppler)
#   [WAIT=<job id>] bash slurm/genome_pilot.sh <sample> <haplotagged bam> <phased small-variant vcf> <trgt vcf> <out dir> [XX|XY]
#   WAIT: the inputs are still being made by that job (e.g. HiPhase); extraction starts after it succeeds
#   MODEL_DIR=<dir with net1.pt, noise.npz> [MODEL_WAIT=<job id>]: no training; apply that model (e.g. the germline
#     pangenome model to brain tissue, whose somatic instability must not be learnt as read noise), after the job
#     that is still training it
#   EXTRACT_JOB=<job id | done>: the extraction is already submitted (or finished); only the later steps are submitted
#   NO_CLINICAL=1: skip the clinical report
set -euo pipefail
S=${1:?sample}; BAM=${2:?haplotagged bam}; PHASED=${3:?phased vcf}; TV=${4:?trgt vcf}; P=${5:?out dir}; K=${6:-XY}
PER_STRATUM=${PER_STRATUM:-15000}
V=/projects/prjs2178/software/venvs/tandemo/bin
R=/projects/prjs2178/repo_clones/Tandemo
REF=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa
CHR=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
     chr21 chr22 chrX)
[ "$K" = XY ] && CHR+=(chrY)
N=$(( ${#CHR[@]} - 1 ))
if [ -z "${WAIT:-}" ]; then
    for f in $BAM $BAM.bai $PHASED $PHASED.tbi $TV $TV.tbi; do [ -s $f ] || { echo "missing $f"; exit 1; }; done
fi
dep=${WAIT:+--dependency=afterok:$WAIT}
mkdir -p $P/{logs,ex,model,report,deviant,instability,clinical}
cd $P
sb() { sbatch --parsable "$@" | grep -v '^sbatch:'; }
pick='C=('"${CHR[*]}"'); c=${C[$SLURM_ARRAY_TASK_ID]}'
inputs=$(printf "ex/$S.%s " "${CHR[@]}")
if [ -n "${EXTRACT_JOB:-}" ]; then
    ex=$EXTRACT_JOB
else
ex=$(sb -J tdm-$S-extract -p genoa -c 48 -t 03:00:00 --array=0-$N $dep -o logs/extract.%A_%a.out \
    --wrap "set -euo pipefail; $pick
        $V/tandemo extract --genome t2t --bam $BAM --phased-vcf $PHASED --trgt-vcf $TV --sample $S \
            --regions \$c --out ex/$S.\$c --threads 48
        $V/tandemo trgt-check --input ex/$S.\$c")
fi
# dependency flag on the given jobs ("done" and empty ids dropped)
afterok() { local j=""; for x in "$@"; do [ -n "$x" ] && [ "$x" != done ] && j="$j:$x"; done
            [ -n "$j" ] && echo "--dependency=afterok$j"; true; }
if [ -n "${MODEL_DIR:-}" ]; then
    NET=$MODEL_DIR/net1.pt; NOISE=$MODEL_DIR/noise.npz
    after=$(afterok $ex ${MODEL_WAIT:-}); tr="${MODEL_WAIT:-none} (model $MODEL_DIR)"
else
NET=model/net1.pt; NOISE=model/noise.npz
tr=$(sb -J tdm-$S-train -p gpu_h100 --gpus=1 -c 16 -t 12:00:00 $(afterok $ex) -o logs/train.%j.out \
    --wrap "set -euo pipefail; nvidia-smi --query-gpu=name,driver_version --format=csv
        $V/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"'
        /usr/bin/time -v $V/tandemo train --input $inputs --out model/net1.pt --per-stratum $PER_STRATUM --device cuda
        $V/tandemo train --input $inputs --out model/noise.npz --empirical --per-stratum $PER_STRATUM")
after=$(afterok $tr)
fi
rp=$(sb -J tdm-$S-report -p genoa -c 24 --mem=64G -t 02:00:00 $after -o logs/report.%j.out \
    --wrap "set -euo pipefail; $V/tandemo report --model $NET --baseline $NOISE --input $inputs \
        --per-stratum 12000 --out report")
sc=$(sb -J tdm-$S-scan -p genoa -c 24 -t 02:00:00 --array=0-$N $after -o logs/scan.%A_%a.out \
    --wrap "set -euo pipefail; $pick
        $V/python $R/scripts/deviant_vntr.py --input ex/$S.\$c --model $NET --out deviant/\$c --plot 10 \
            --bam $BAM --reference $REF --phased-vcf $PHASED --trgt-vcf $TV --sample $S")
in_=$(sb -J tdm-$S-instab -p genoa -c 24 -t 02:00:00 --array=0-$N $after -o logs/instab.%A_%a.out \
    --wrap "set -euo pipefail; $pick; mkdir -p instability/\$c
        $V/python $R/scripts/instability.py --input ex/$S.\$c --model $NET --out instability/\$c/\$c \
            --plot 8 --bam $BAM --reference $REF --phased-vcf $PHASED --trgt-vcf $TV --sample $S")
cl=skipped
[ "${NO_CLINICAL:-0}" = 1 ] || cl=$(sb -J tdm-$S-clinical -p genoa -c 24 -t 02:00:00 $after -o logs/clinical.%j.out \
    --wrap "set -euo pipefail; $V/python $R/scripts/clinical_report.py --input $inputs --model $NET \
        --out clinical/$S --bam $BAM --reference $REF --phased-vcf $PHASED --trgt-vcf $TV --sample $S")
pdf=skipped
[ "$cl" = skipped ] || pdf=$(sb -J tdm-$S-clinical-pdf -p staging -c 1 -t 01:00:00 $(afterok $cl) -o logs/clinical_pdf.%j.out \
    --wrap "set -euo pipefail; bash $R/scripts/pages_to_pdf.sh clinical/$S.pages.txt clinical/$S.clinical_repeats.pdf")
echo -e "extract\t$ex\ntrain\t$tr\nreport\t$rp\nscan\t$sc\ninstability\t$in_\nclinical\t$cl\nclinical_pdf\t$pdf" | tee -a jobs.tsv
