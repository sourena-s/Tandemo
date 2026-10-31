import glob, sys, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.genes import context, for_reference
P = "/projects/0/prjs2178/seed/pacbio/pangenome-NA19909/tandemo/instability"
parts = [pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/chr*/chr*.parquet")) if not f.endswith(".reads.parquet")]
t = pd.concat(parts, ignore_index=True)
g = for_reference("/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa")
t = t.sort_values(["recurrent", "unstable", "min_log10p"], ascending=[False, False, True]).reset_index(drop=True)
t["gene_context"] = [context(g, r.chrom, r.start, r.end) for r in t.itertuples()]
t["figure"] = [next(iter(glob.glob(f"{P}/{r.chrom}/[0-9][0-9].{r.locus}.panels.svg")), "") for r in t.itertuples()]
t.to_csv(f"{P}/NA19909.instability_genome.tsv", sep="\t", index=False)
print(f"{len(t)} loci with candidate unstable reads, {int(t.recurrent.sum())} recurrent (>= 2 reads), {int(t.unstable.sum())} reads")
pd.set_option("display.width", 250)
print(t.head(30)[["locus", "period", "L1", "L2", "confident", "unstable", "hap1", "hap2", "delta_units", "min_log10p", "gene_context"]].to_string(index=False))
