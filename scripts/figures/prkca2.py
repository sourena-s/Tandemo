"""PRKCA (chr17_67457221_67457499_trsolve) with the relaxed instability rules (scripts/instability.py defaults)."""
import os, sys
import numpy as np
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo/scripts"); sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
import deviant_vntr as dv
from tandemo.data import load, assign_alleles
from tandemo.phasing import PhaseParams, confident_phase
from tandemo.noise import load_model
from tandemo.genes import for_reference
from tandemo.pileup import run as pileup
from tandemo.extract import Params
N = "/projects/0/prjs2178/seed/pacbio/pangenome-NA19909"; T = f"{N}/tandemo"
REF = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa"
CAT = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz"
lid = "chr17_67457221_67457499_trsolve"
g, _ = load(f"{T}/ex/NA19909.chr17", loci_keep={lid})
g = assign_alleles(g)
g["w1_train"] = g["w1"].fillna(0.5)
g["phase"] = confident_phase(g.n_hap1, g.n_hap2, PhaseParams(min_support=3, max_conflict_frac=0.2))
model = load_model(f"{T}/model/net1.pt")
L1, L2, per = int(g.L1.iloc[0]), int(g.L2.iloc[0]), int(g.period.iloc[0])
lc = model.log_curves(g, np.array([L1, L2])).astype(float) / np.log(10)
ph, Y = g.phase.to_numpy(), g.Y.to_numpy()
own, oth = np.where(ph == 2, L2, L1), np.where(ph == 2, L1, L2)
lp10 = np.where(ph == 2, lc[:, 1], lc[:, 0])
dev = ((ph > 0) & (np.abs(Y - own) > max(10, per)) & ~((np.abs(Y - oth) <= max(5, per)) & (L1 != L2))
       & (g.mapq.to_numpy() >= 60) & (g.flank_id_min.to_numpy() >= 0.99) & (lp10 < -3.5))
print(f"flagged {dev.sum()} reads; haplotype 2 reads at <= -20 bp: {int(((ph == 2) & (Y - L2 <= -20)).sum())}; "
      f"with SA among flagged: {int((g.n_sa.to_numpy()[dev] > 0).sum())}; segdup {float(g.segdup.iloc[0]):.2f}")
out = f"{T}/instability/prkca_v2"; os.makedirs(out, exist_ok=True)
plt = dv._style()
spike = dv.plot(plt, model, g, lp10, dev, f"{out}/prkca", None, None, dv._title(for_reference(REF), g))
pileup(f"{N}/aligned_t2t_phased/NA19909.haplotagged.bam", REF, CAT, out, f"{N}/deepvariant/NA19909.deepvariant.phased.vcf.gz",
       f"{N}/trgt/NA19909.trgt.phased.vcf.gz", loci_names=[lid], regions=["chr17"], sample="NA19909", max_reads=None,
       exclude_catalog=CAT, params=Params(), highlight={lid: set(g.read.to_numpy()[dev])}, labels={lid: {1: L1, 2: L2}},
       log=print)
dv.compose(spike, f"{out}/{lid}.svg", f"{out}/PRKCA.panels.svg")
os.system(f"rsvg-convert -z 1.3 -b white {out}/PRKCA.panels.svg -o {out}/PRKCA.panels.png")
