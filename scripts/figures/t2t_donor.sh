#!/usr/bin/env bash
# PacBio HiFi donor on T2T-CHM13v2.0 (chm13v2.0_maskedY_rCRS): pbmm2 per SMRT cell -> merge + depth + karyotype ->
# DeepVariant (SNV/indel), sawfish (SV), TRGT with Platinum TRs v1.0 (per chromosome) -> HiPhase phasing all three
# together and haplotagging the reads. Same tools and settings as the HG002 T2T run (hg002_t2t/submit.sh) and the
# GRCh38 pipeline (seed/pacbio/pipeline/stages.sh), sized from HG002 at 74x (align 46 min x 72 cores per cell,
# DeepVariant 69 min x 120 cores / 196 GB, sawfish 40 min x 48 cores / 102 GB, HiPhase 47 min x 48 cores).
#
# Partitions: the heavy steps run on fat_genoa (1.5 SBU per core-hour, the large EINF-19323/L3 budget) so they do not
# drain the small genoa budget; TRGT, merge and checks run on genoa (1 SBU per core-hour).
# pbmm2 --strip drops the raw kinetics (fi/ri/fp/rp) but keeps the 5mC calls (MM/ML); the raw BAMs keep everything.
# The karyotype (XX/XY) is called from chrY vs autosomal depth after the merge and used by DeepVariant, sawfish, TRGT.
#
#   bash t2t_donor.sh <sample> <dir with raw *.bam> <out dir> [job id the raw BAMs wait for]
#   FROM=callers: align/<sample>.bam and karyotype.txt exist; run DeepVariant, sawfish, TRGT, HiPhase
#   FROM=trgt: DeepVariant and sawfish exist as well; run TRGT, its concat and HiPhase only (the raw BAM dir is
#     not read; pass the job id these files wait for, e.g. a copy, as the 4th argument)
#   HP_OUT=<dir>: HiPhase output directory (default hiphase), to keep an earlier phasing
#   NO_MD5=1: raw BAMs not downloaded by this pipeline (no md5_check.txt); check them with samtools quickcheck -u first
# Cores: genoa bills whole 24-core units and every partition gives a fixed memory per core, so each step asks for a
# multiple of 24 cores (or what its tool can use on fat_genoa) and runs its tool on all of them.
set -euo pipefail
S=${1:?sample}; RAW=${2:?raw bam dir}; O=${3:?out dir}; WAIT=${4:-}
T=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0
REF=$T/chm13v2.0_maskedY_rCRS.fa; MMI=$T/chm13v2.0_maskedY_rCRS.mmi; PAR=$T/chm13v2.0_maskedY_rCRS.PAR.bed
CAT=$T/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz
CNV_EXCL=$T/cnv_excluded_regions.chm13v2.0_maskedY_rCRS.bed.gz
ENVB=/projects/prjs2178/software/miniforge3/envs/somatic-sv/bin                 # samtools bcftools pbmm2 sawfish hiphase
TRGT=/home/$USER/.conda/envs/seqlab/bin/trgt                                    # trgt 5.1.0
DV_SIF=/projects/prjs2178/software/deepvariant_1.10.0.sif
CHR=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
     chr21 chr22 chrX chrY)
HP=${HP_OUT:-hiphase}
mkdir -p $O/{align,deepvariant,sawfish,trgt/catalog,$HP,logs}
cd $O
if [ -z "${FROM:-}" ]; then
    mapfile -t CELLS < <(cd $RAW && ls *.bam | sed 's/\.bam$//')
    [ ${#CELLS[@]} -gt 0 ] || { echo "no BAMs in $RAW"; exit 1; }
    echo "${CELLS[*]}" | tr ' ' '\n' > cells.txt
fi
[ -s trgt/catalog/platinum.chr1.bed ] || zcat $CAT | awk -v d=trgt/catalog '{ print > (d "/platinum." $1 ".bed") }'

run() {  # name partition cpus time dependency array command   (memory: the partition's default per core)
    local d="" a=""
    [[ -n $5 ]] && d="--dependency=afterok:$5"
    [[ -n $6 ]] && a="--array=$6"
    sbatch --parsable -J "$S-t2t-$1" -p "$2" -c "$3" -t "$4" $d $a -o "logs/$1.%A_%a.out" \
        --wrap "set -euo pipefail; export PATH=$ENVB:\$PATH; cd $O; $7" | grep -v '^sbatch:'
}

# FROM=callers: align/$S.bam(.bai) and karyotype.txt exist; submit only steps 3-6
if [ "${FROM:-}" = callers ] || [ "${FROM:-}" = trgt ]; then
    if [ -z "$WAIT" ]; then                       # inputs present now (else they arrive with the job we wait for)
        [ -s align/$S.bam.bai ] && [ -s karyotype.txt ] || { echo "FROM=$FROM needs align/$S.bam.bai and karyotype.txt"; exit 1; }
    fi
    ck=""; al=""; mg=$WAIT
else
# 0. the download must have passed its md5 checks (two OK lines per cell: BAM and .pbi)
if [ "${NO_MD5:-0}" = 1 ]; then ck=$WAIT; else
ck=$(run check genoa 1 00:10:00 "$WAIT" "" "
    n=\$(grep -c '^OK ' $RAW/md5_check.txt); echo \"\$n OK of ${#CELLS[@]} cells x 2\"
    [ \$n -eq $(( 2 * ${#CELLS[@]} )) ] && ! grep -q '^BAD' $RAW/md5_check.txt")
fi

# 1. align each cell (one sample name for all cells, e.g. NA19909 and NA19909_Fiber_Seq read groups)
al=$(run align fat_genoa 72 08:00:00 "$ck" "0-$(( ${#CELLS[@]} - 1 ))" "
    C=(${CELLS[*]}); c=\${C[\$SLURM_ARRAY_TASK_ID]}
    pbmm2 align $MMI $RAW/\$c.bam align/\$c.bam --preset HIFI --sort --strip -j 64 -J 8 -m 2G --sample $S \
        --log-level INFO
    samtools flagstat -@ 8 align/\$c.bam > align/\$c.flagstat.txt")

# 2. merge (24 cores = one genoa allocation unit) and karyotype from the index: samtools idxstats takes seconds;
#    chrY vs autosomal read density is ~0.5 for XY, ~0 for XX on the chrY-PAR-masked reference
mg=$(run merge genoa 24 04:00:00 "$al" "" "
    samtools merge -@ 24 -f align/$S.bam align/*.hifi_reads.*.bam
    samtools index -@ 24 align/$S.bam
    samtools idxstats align/$S.bam > align/$S.idxstats.tsv
    awk '\$1 ~ /^chr([0-9]+)\$/ { r += \$3; n += \$2 } \$1 == \"chrY\" { yr = \$3; yn = \$2 }
         END { q = (yr / yn) / (r / n); printf \"chrY / autosomal read density %.2f\n\", q > \"/dev/stderr\";
               print (q > 0.2 ? \"XY\" : \"XX\") }' align/$S.idxstats.tsv > karyotype.txt
    echo karyotype \$(cat karyotype.txt)")
fi
# depth summary, opt-in (COVERAGE=1): samtools coverage is single-threaded and takes hours on a 200x BAM, and genoa
# charges whole 24-core units, so even a 1-core job costs 24 SBU per hour. Nothing downstream needs it.
[ "${COVERAGE:-0}" = 1 ] && run coverage genoa 1 12:00:00 "$mg" "" \
    "samtools coverage align/$S.bam > align/$S.coverage.tsv" > /dev/null

dv=""; sw=""
if [ "${FROM:-}" != trgt ]; then
# 3. DeepVariant (CPU); XY: chrX/chrY haploid outside the PARs
dv=$(run deepvariant fat_genoa 128 12:00:00 "$mg" "" "
    K=\$(cat karyotype.txt); hap=''; [ \$K = XY ] && hap='--haploid_contigs=chrX,chrY --par_regions_bed=$PAR'
    mkdir -p /scratch-local/\$USER.\$SLURM_JOB_ID/dv
    apptainer exec --bind /gpfs,/projects,/scratch-shared,/scratch-local $DV_SIF /opt/deepvariant/bin/run_deepvariant \
        --model_type=PACBIO --ref=$REF --reads=align/$S.bam \$hap \
        --output_vcf=deepvariant/$S.deepvariant.vcf.gz --output_gvcf=deepvariant/$S.deepvariant.g.vcf.gz \
        --num_shards=120 --intermediate_results_dir=/scratch-local/\$USER.\$SLURM_JOB_ID/dv")

# 4. sawfish (SV); its peak memory grows with depth (102 GB at 74x), fat_genoa gives 7.5 GB per core
sw=$(run sawfish fat_genoa 64 12:00:00 "$mg" "" "
    K=\$(cat karyotype.txt)
    sawfish discover --clobber --threads 64 --ref $REF --bam align/$S.bam \
        --expected-cn $T/expected_cn.chm13v2.0_maskedY_rCRS.\$K.bed --cnv-excluded-regions $CNV_EXCL \
        --output-dir sawfish/discover
    sawfish joint-call --clobber --threads 64 --sample sawfish/discover --report-supporting-reads \
        --output-dir sawfish/joint")
fi

# 5. TRGT, Platinum TRs, one task per chromosome (chrY skipped for XX), then one sorted VCF for HiPhase
tr=$(run trgt genoa 24 03:00:00 "$mg" "0-$(( ${#CHR[@]} - 1 ))" "
    C=(${CHR[*]}); c=\${C[\$SLURM_ARRAY_TASK_ID]}; K=\$(cat karyotype.txt)
    if [ \$K = XX ] && [ \$c = chrY ]; then echo 'XX: chrY skipped'; exit 0; fi
    mkdir -p /scratch-local/\$USER.\$SLURM_JOB_ID
    $TRGT genotype --genome $REF --reads align/$S.bam --repeats trgt/catalog/platinum.\$c.bed \
        --output-prefix trgt/tandems.\$c --karyotype \$K --threads 24 --preset wgs
    bcftools sort -Oz -m 4G -T /scratch-local/\$USER.\$SLURM_JOB_ID -o trgt/tandems.\$c.sorted.vcf.gz trgt/tandems.\$c.vcf.gz
    bcftools index -t trgt/tandems.\$c.sorted.vcf.gz
    rm -f trgt/tandems.\$c.vcf.gz trgt/tandems.\$c.spanning.bam")
tc=$(run trgt_concat genoa 24 01:00:00 "$tr" "" "
    bcftools concat -a --threads 24 -Oz -o trgt/tandems.sorted.vcf.gz \$(ls trgt/tandems.chr*.sorted.vcf.gz | sort -V)
    bcftools index -t trgt/tandems.sorted.vcf.gz")

# 6. HiPhase: small variants, SVs and tandem repeats phased together; haplotagged BAM
hp=$(run hiphase fat_genoa 48 12:00:00 "$(echo "$dv:$sw:$tc" | sed 's/::*/:/g; s/^://; s/:$//')" "" "
    hiphase --threads 48 --reference $REF \
        --vcf deepvariant/$S.deepvariant.vcf.gz --output-vcf $HP/$S.deepvariant.phased.vcf.gz \
        --vcf sawfish/joint/genotyped.sv.vcf.gz --output-vcf $HP/$S.sawfish.phased.vcf.gz \
        --vcf trgt/tandems.sorted.vcf.gz --output-vcf $HP/$S.trgt.phased.vcf.gz \
        --bam align/$S.bam --output-bam $HP/$S.haplotagged.bam \
        --stats-file $HP/stats.csv --blocks-file $HP/blocks.tsv --summary-file $HP/summary.tsv
    [ -e $HP/$S.haplotagged.bam.bai ] || samtools index -@ 48 $HP/$S.haplotagged.bam
    samtools quickcheck $HP/$S.haplotagged.bam")

echo -e "check\t${ck:-}\nalign\t$al\nmerge\t$mg\ndeepvariant\t$dv\nsawfish\t$sw\ntrgt\t$tr\ntrgt_concat\t$tc\nhiphase\t$hp" \
    | tee -a jobs.tsv
