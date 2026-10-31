#!/usr/bin/env bash
# Disease-associated tandem repeats on T2T-CHM13v2.0: the 68 loci Platinum TRs v1.0 names `<DISEASE>_<GENE>_pathogenic`
# (STRchive loci, curated by the Platinum TRs authors). A subset of the Platinum catalog, same coordinates and IDs,
# so tandemo results on Platinum and the pileups refer to the same locus. (Lifting PacBio's 56-locus GRCh38
# pathogenic catalog with the T2T chain instead loses DAB1 and DMPK and finds nothing Platinum lacks.)
#   bash scripts/t2t_pathogenic_catalog.sh
set -euo pipefail
T2T=/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0
PLAT=$T2T/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz
O=$T2T/pathogenic_repeats
mkdir -p $O
zcat $PLAT | awk -F'\t' '$4 ~ /^ID=[^;]*_pathogenic;/' > $O/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.pathogenic.trgt.bed
md5sum $O/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.pathogenic.trgt.bed > $O/md5sums.txt
echo "$(wc -l < $O/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.pathogenic.trgt.bed) loci"
