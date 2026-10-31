#!/usr/bin/env bash
# chr4 pilot with Y and L on one definition: the extraction records germline indels in the 10 bp measurement pads
# (pad_d1 / pad_d2 / pad_status) and training adds them to the TRGT alleles. Retrains network 1 exactly as the
# first pilot (all chr4 reads, H100) and redraws the report, to see whether the +1 / -1 cells disappear.
#   extract   genoa, 16 cores           ~2 SBU
#   pads      genoa, 4 cores            how many loci / reads the correction touches, by stratum   <1 SBU
#   train     gpu_h100, 1 GPU + 16 cores, 17 min last time                                          ~55 SBU (GPU budget)
#   report    genoa, 16 cores, 64 GB                                                                 ~15 SBU
#   bash slurm/pilot_t2t_chr4_pad.sh [pilot dir]
set -euo pipefail
V=/projects/prjs2178/software/venvs/tandemo/bin
H=/scratch-shared/$USER/hg002_t2t
P=${1:-/scratch-shared/$USER/tandemo_pilot_t2t/chr4}
cd $P
mkdir -p logs ex
sb() { sbatch --parsable "$@" | grep -v '^sbatch:'; }
ex=$(sb -J tdm-extract-chr4pad -p genoa -c 16 --mem=$((16 * 1792))M -t 01:00:00 -o logs/extract_pad.%j.out \
    --wrap "set -euo pipefail; $V/tandemo extract --genome t2t --bam $H/hiphase/HG002.haplotagged.bam \
        --phased-vcf $H/hiphase/HG002.deepvariant.phased.vcf.gz \
        --trgt-vcf /scratch-shared/$USER/trgt_t2t/HG002/tandems.chr4.sorted.vcf.gz \
        --regions chr4 --out ex/HG002.chr4.pad --threads 16")
pd_=$(sb -J tdm-pads-chr4 -p genoa -c 4 --mem=24G -t 00:30:00 --dependency=afterok:$ex -o logs/pads.%j.out \
    --wrap "set -euo pipefail; $V/python - <<'PY'
import numpy as np, pandas as pd
L = pd.read_csv('ex/HG002.chr4.pad.loci.tsv.gz', sep='\t')
L = L[(L.al1 >= 0) & (L.n_span > 0)]
L['pc'] = pd.cut(L.period, [0, 1, 2, 3, 4, 6, 15, 50, 1e9], labels=['1', '2', '3', '4', '5-6', '7-15', '16-50', '>50'])
L['lc'] = pd.cut(L.ref_len, [0, 15, 25, 64, 128, 256, 1e9], labels=['<15', '15-25', '25-64', '64-128', '128-256', '>256'])
L['moved'] = (L.pad_d1 != 0) | (L.pad_d2 != 0)
print('loci', len(L), 'pad status', L.pad_status.value_counts().to_dict(), 'with a pad indel', int(L.moved.sum()))
t = L.groupby(['pc', 'lc'], observed=True).agg(loci=('locus', 'size'), moved=('moved', 'mean'),
                                                ambiguous=('pad_status', lambda s: (s == 1).mean()),
                                                unphased=('pad_status', lambda s: (s == 2).mean()))
print(t.round(3).to_string())
PY")
tr=$(sb -J tdm-train-chr4pad -p gpu_h100 --gpus=1 -c 16 -t 02:00:00 --dependency=afterok:$ex -o logs/train_pad.%j.out \
    --wrap "set -euo pipefail; nvidia-smi --query-gpu=name,driver_version --format=csv
        $V/tandemo train --input ex/HG002.chr4.pad --out net1.chr4.pad.pt --device cuda
        $V/tandemo train --input ex/HG002.chr4.pad --out noise.chr4.pad.npz --empirical")
rp=$(sb -J tdm-report-chr4pad -p genoa -c 16 --mem=64G -t 01:00:00 --dependency=afterok:$tr -o logs/report_pad.%j.out \
    --wrap "set -euo pipefail; $V/tandemo report --model net1.chr4.pad.pt --baseline noise.chr4.pad.npz \
        --input ex/HG002.chr4.pad --per-stratum 12000 --out report_pad")
echo -e "extract_pad\t$ex\npads\t$pd_\ntrain_pad\t$tr\nreport_pad\t$rp" | tee -a jobs.tsv
