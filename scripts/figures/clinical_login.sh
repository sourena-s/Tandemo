#!/usr/bin/env bash
# NA19909 clinical repeat report on the login node: extraction of the 68 disease loci, report, PDF
set -euo pipefail
N=/projects/0/prjs2178/seed/pacbio/pangenome-NA19909; S=NA19909
V=/projects/prjs2178/software/venvs/tandemo/bin; R=/projects/prjs2178/repo_clones/Tandemo
REF=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa
M=/scratch-shared/ssoheilinezhad/tandemo_pilot_t2t/NA19909_chr4/model/net1.pt      # corrected chr4 model
O=$N/tandemo/clinical_chr4model
mkdir -p $O/ex && cd $O
nice $V/tandemo extract --genome t2t --catalog pathogenic --bam $N/aligned_t2t_phased/$S.haplotagged.bam \
    --phased-vcf $N/deepvariant/$S.deepvariant.phased.vcf.gz --trgt-vcf $N/trgt/$S.trgt.phased.vcf.gz \
    --sample $S --out ex/$S.pathogenic --threads 4
nice $V/python $R/scripts/clinical_report.py --input ex/$S.pathogenic --model $M --out $O/$S \
    --bam $N/aligned_t2t_phased/$S.haplotagged.bam --reference $REF \
    --phased-vcf $N/deepvariant/$S.deepvariant.phased.vcf.gz --trgt-vcf $N/trgt/$S.trgt.phased.vcf.gz --sample $S
rsvg-convert -f pdf $(cat $S.pages.txt) -o $S.clinical_repeats.pdf
ls -la $S.clinical_repeats.pdf; echo "pages: $(wc -l < $S.pages.txt)"
