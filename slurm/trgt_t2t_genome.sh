#!/usr/bin/env bash
# TRGT 5.1.0 genotypes on T2T-CHM13v2.0 with the full Platinum TRs v1.0 catalog, one array task per chromosome:
# the germline alleles network 1 trains on. chr4 took 6.3 min on 24 cores for 505 k loci, so the genome
# (7.8 M loci) is ~40 core-hours; 24 genoa cores per task (one allocation unit), ~40 SBU in all.
# The unfiltered catalog is used so any later catalog filter finds its loci genotyped.
#   bash slurm/trgt_t2t_genome.sh <sample> <bam> <karyotype XX|XY> [out dir]
set -euo pipefail
S=${1:?sample}; BAM=${2:?bam}; K=${3:?karyotype}
O=${4:-/scratch-shared/$USER/trgt_t2t/$S.genome}
T=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0
REF=$T/chm13v2.0_maskedY_rCRS.fa
CAT=$T/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz
TRGT=/home/$USER/.conda/envs/seqlab/bin/trgt
ENVB=/projects/prjs2178/software/miniforge3/envs/somatic-sv/bin              # bcftools, samtools
CHR=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
     chr21 chr22 chrX chrY)
[ $K = XX ] && unset 'CHR[23]'
mkdir -p $O/logs $O/catalog
[ -s $O/catalog/platinum.chr1.bed ] ||       # per-chromosome catalogs, one pass over the file
    zcat $CAT | awk -v d=$O/catalog '{ f = d "/platinum." $1 ".bed"; print > f }'
sbatch --parsable -J trgt-t2t-$S -p genoa -c 24 --mem=42G -t 01:00:00 --array=0-$(( ${#CHR[@]} - 1 )) \
    -o $O/logs/trgt.%A_%a.out --wrap "set -euo pipefail; export PATH=$ENVB:\$PATH; cd $O
    C=(${CHR[*]}); c=\${C[\$SLURM_ARRAY_TASK_ID]}
    $TRGT genotype --genome $REF --reads $BAM --repeats catalog/platinum.\$c.bed --output-prefix tandems.\$c \
        --karyotype $K --threads 24 --preset wgs
    bcftools sort -Oz -m 4G -o tandems.\$c.sorted.vcf.gz tandems.\$c.vcf.gz
    bcftools index -t tandems.\$c.sorted.vcf.gz
    rm -f tandems.\$c.vcf.gz tandems.\$c.spanning.bam" | tee $O/jobs.txt
