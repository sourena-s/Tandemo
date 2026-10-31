import sys, numpy as np, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.data import load, assign_alleles
from tandemo.phasing import PhaseParams, confident_phase
from tandemo.noise import load_model
N = "/projects/0/prjs2178/seed/pacbio/pangenome-NA19909/tandemo"
lid = "chr17_67457221_67457499_trsolve"
g, _ = load(f"{N}/ex/NA19909.chr17", loci_keep={lid})
g = assign_alleles(g)
g["phase"] = confident_phase(g.n_hap1, g.n_hap2, PhaseParams(min_support=5, max_conflict_frac=1e-9))
g["phase_plot"] = confident_phase(g.n_hap1, g.n_hap2, PhaseParams())
m = load_model(f"{N}/model/net1.pt")
L1, L2 = int(g.L1.iloc[0]), int(g.L2.iloc[0])
lc = m.log_curves(g, np.array([L1, L2])) / np.log(10)
g["d"] = g.Y - L2
g["log10p"] = lc[:, 1]
s = g[(g.phase_plot == 2) & (g.d <= -20)].copy()
s["units_off"] = (s.d / 2 - np.round(s.d / 2)).abs()
s["fails"] = [", ".join(n for n, bad in (
    ("phase (<5 support or any conflict)", r.phase != 2), ("rq < 0.999", r.rq < 0.999), ("MAPQ < 60", r.mapq < 60),
    ("SA", r.n_sa > 0), ("flank identity < 0.99", r.flank_id_min < 0.99), ("log10 p >= -4", r.log10p >= -4),
    ("not whole units", r.units_off > 0.2)) if bad) or "none (flagged)" for r in s.itertuples()]
pd.set_option("display.width", 220)
print(s.sort_values("d")[["d", "n_hap1", "n_hap2", "rq", "mapq", "n_sa", "flank_id_min", "log10p", "fails"]].to_string(index=False))
print(s.fails.str.split(", ").explode().value_counts().to_string())
