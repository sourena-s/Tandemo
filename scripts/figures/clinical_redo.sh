#!/usr/bin/env bash
# rebuild the NA19909 clinical report (all reads, pileup repeat-boundary fix) on the login node; extraction reused
set -euo pipefail
N=/projects/0/prjs2178/seed/pacbio/pangenome-NA19909; S=NA19909
V=/projects/prjs2178/software/venvs/tandemo/bin; R=/projects/prjs2178/repo_clones/Tandemo
(cd $R && TMPDIR=$(mktemp -d) $V/python -m pytest -q 2>&1 | tail -1) | tee /tmp/$USER.pt; grep -q failed /tmp/$USER.pt && exit 1
O=$N/tandemo/clinical_chr4model; cd $O
rm -rf ${S}_pileups
nice $V/python $R/scripts/clinical_report.py --input ex/$S.pathogenic \
    --model /scratch-shared/ssoheilinezhad/tandemo_pilot_t2t/NA19909_chr4/model/net1.pt --out $O/$S \
    --bam $N/aligned_t2t_phased/$S.haplotagged.bam --reference /projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa \
    --phased-vcf $N/deepvariant/$S.deepvariant.phased.vcf.gz --trgt-vcf $N/trgt/$S.trgt.phased.vcf.gz --sample $S 2>&1 | tail -3
bash $R/scripts/pages_to_pdf.sh $S.pages.txt $S.clinical_repeats.pdf; grep -E "PRDM12|TRGT allele already" $S.summary.tsv | cut -f2,19 | head -12
