import sys, numpy as np, pandas as pd, pysam
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.data import load, assign_alleles
from tandemo.phasing import PhaseParams, confident_phase
N = "/projects/0/prjs2178/seed/pacbio/pangenome-NA19909"
lid = "chr17_67457221_67457499_trsolve"
g, _ = load(f"{N}/tandemo/ex/NA19909.chr17", loci_keep={lid})
g = assign_alleles(g)
g["phase"] = confident_phase(g.n_hap1, g.n_hap2, PhaseParams())
g["cell"] = g.read.str.split("/").str[0]
# read group description per movie: which cells are Fiber-seq
rg = {}
with pysam.AlignmentFile(f"{N}/aligned_t2t_phased/NA19909.haplotagged.bam") as b:
    for r in b.header.to_dict().get("RG", []):
        mv = r.get("PU") or r.get("ID")
        rg[r.get("ID")] = (r.get("SM", ""), r.get("DS", "")[:0], mv)
names = {}
for f in __import__("glob").glob("/scratch-shared/ssoheilinezhad/pacgenome/NA19909/*.bam") + __import__("glob").glob("/projects/0/prjs2178/seed/pacbio/pangenome-NA19909/*/*.hifi_reads*.bam"):
    pass
h2 = g[g.phase == 2]
short = h2[(h2.Y - h2.L2) <= -20]
t = pd.DataFrame({"hap2 reads": h2.cell.value_counts(), "short (-24..-27 bp)": short.cell.value_counts(),
                  "all reads at locus": g.cell.value_counts()}).fillna(0).astype(int).sort_index()
t["short / hap2"] = (t["short (-24..-27 bp)"] / t["hap2 reads"].replace(0, np.nan)).round(2)
print(t.to_string()); print("total", len(short), "short of", len(h2), "haplotype 2 reads")
