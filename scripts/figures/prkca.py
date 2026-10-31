import glob, os, sys
import pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo/scripts"); sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
import deviant_vntr as dv
from tandemo.pileup import run as pileup
from tandemo.extract import Params
N = "/projects/0/prjs2178/seed/pacbio/pangenome-NA19909"; I = f"{N}/tandemo/instability/chr17"
lid = "chr17_67457221_67457499_trsolve"
L = pd.read_parquet(f"{I}/chr17.parquet").set_index("locus").loc[lid]
hl = set(pd.read_parquet(f"{I}/chr17.reads.parquet").query("locus == @lid").read)
cat = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz"
out = f"{N}/tandemo/instability/prkca"
pileup(f"{N}/aligned_t2t_phased/NA19909.haplotagged.bam", "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa",
       cat, out, f"{N}/deepvariant/NA19909.deepvariant.phased.vcf.gz", f"{N}/trgt/NA19909.trgt.phased.vcf.gz", loci_names=[lid],
       regions=["chr17"], sample="NA19909", max_reads=None, exclude_catalog=cat, params=Params(), highlight={lid: hl},
       labels={lid: {1: int(L.L1), 2: int(L.L2)}}, log=print)
spike = [f for f in glob.glob(f"{I}/chr17.*{lid}.svg") if not f.endswith("panels.svg")][0]
dv.compose(spike, f"{out}/{lid}.svg", f"{out}/{lid}.panels.svg")
os.system(f"rsvg-convert -z 1.3 -b white {out}/{lid}.panels.svg -o {out}/{lid}.panels.png")
print("done", spike)
