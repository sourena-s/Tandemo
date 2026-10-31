import glob, sys, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.genes import context, for_reference
from tandemo import segdup
g = for_reference("/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa")
sp = segdup.for_reference("/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
for s, d in (("UCL", "revio_ucl_t2t"), ("leiden", "leiden_t2t")):
    P = f"/projects/0/prjs2178/seed/pacbio/{d}/tandemo/instability"
    t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/chr*/chr*.parquet")) if not f.endswith(".reads.parquet")],
                  ignore_index=True)
    t = t.sort_values(["recurrent", "unstable", "min_log10p"], ascending=[False, False, True]).reset_index(drop=True)
    t["gene"] = [context(g, r.chrom, r.start, r.end) for r in t.itertuples()]
    t["segdup"] = [segdup.overlap(sp, r.chrom, r.start, r.end) for r in t.itertuples()]
    t["shift"] = t.delta_units.map(lambda x: "one shared" if len(set(x.split(","))) == 1 else "varied")
    t.to_csv(f"{P}/{s}.instability_genome.tsv", sep="\t", index=False)
    print(f"== {s}: {len(t)} loci, {int(t.recurrent.sum())} recurrent, {int(t.unstable.sum())} reads")
    print(t.head(15)[["locus", "period", "L1", "L2", "confident", "unstable", "hap1", "hap2", "delta_units", "shift",
                      "segdup", "gene"]].to_string(index=False))
