"""Stage 2 (free f, NPMLE) at the brain-unstable loci in UCL, Leiden and HG002: spread and direction per sample."""
import sys, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.fit import run
from tandemo.noise import load_model
from tandemo.stage2 import FitParams
B = "/projects/0/prjs2178/seed/pacbio"
EX = {"UCL": f"{B}/revio_ucl_t2t/tandemo/ex/UCL", "Leiden": f"{B}/leiden_t2t/tandemo/ex/leiden",
      "HG002": f"{B}/hg002_t2t_tandemo/tandemo/ex/HG002"}
model = load_model(f"{B}/pangenome-NA19909/tandemo/model/net1.pt")
T = pd.read_csv(f"{B}/brain_vs_hg002_unstable.tsv", sep="\t")
loci = {f"{c}_{int(s.replace(',', '')) - 1}_{e.replace(',', '')}_trsolve"
        for c, s, e in (x.replace(":", "-").split("-") for x in T.locus)}
gene = {f"{c}_{int(s.replace(',', '')) - 1}_{e.replace(',', '')}_trsolve": g
        for (c, s, e), g in zip((x.replace(":", "-").split("-") for x in T.locus), T.gene)}
out = []
for smp, pre in EX.items():
    for chrom in sorted({l.split("_")[0] for l in loci}):
        keep = {l for l in loci if l.split("_")[0] == chrom}
        res, _ = run(f"{pre}.{chrom}", model, f"/tmp/ssoh_s2_{smp}_{chrom}", smp, FitParams(bootstrap=200), untrusted="local",
                     loci_keep=keep, log=lambda *_: None)
        out.append(res)
r = pd.concat(out, ignore_index=True)
r["gene"] = r.locus.map(gene)
r.to_parquet(f"{B}/brain_vs_hg002_stage2.parquet", index=False)
pd.set_option("display.width", 250)
for c in ("U", "mad", "mean_shift"):
    r[c] = r[c].round(3 if c == "U" else 2)
r["U_ci"] = r.U_lo.round(2).astype(str) + "-" + r.U_hi.round(2).astype(str)
cols = ["gene", "subject", "hap", "L0", "n_eff", "U", "U_ci", "p_exp", "p_con", "mad", "mean_shift", "n_atoms"]
print(r.sort_values(["gene", "locus", "subject", "hap"])[cols].round(2).to_string(index=False))

# ---- plots: f on the deviation scale d = L - L0 (bp), unsmoothed stems; one panel per locus x allele
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
COL = {"UCL": "#2a78d6", "Leiden": "#e8663d", "HG002": "#5b5a56"}
OFF = {"UCL": -0.25, "Leiden": 0.0, "HG002": 0.25}
r = pd.read_parquet(f"{B}/brain_vs_hg002_stage2.parquet")
r = r[r.status == "ok"]
order = list(dict.fromkeys(T.locus))
key = {f"{c}_{int(s.replace(',', '')) - 1}_{e.replace(',', '')}_trsolve": x
       for x, (c, s, e) in zip(T.locus, (x.replace(":", "-").split("-") for x in T.locus))}
lids = sorted(r.locus.unique(), key=lambda l: order.index(key[l]))
with PdfPages(f"{B}/brain_vs_hg002_stage2.pdf") as pdf:
    for p0 in range(0, len(lids), 4):
        fig, axes = plt.subplots(4, 2, figsize=(11, 12), squeeze=False)
        for i, lid in enumerate(lids[p0:p0 + 4]):
            for j, rank in enumerate(("short", "long")):
                ax = axes[i, j]
                lab = []
                for smp in COL:
                    q = r[(r.locus == lid) & (r.subject == smp)]
                    q = q[q.hap_rank.isin([rank, "tie", "haploid"])] if rank in set(q.hap_rank) or not len(q) else \
                        q[q.hap_rank.isin(["tie", "haploid"])]
                    if not len(q):
                        continue
                    q = q.iloc[0 if rank == "short" else -1]
                    d = np.array(q.f_L) - q.L0
                    ax.vlines(d + OFF[smp], 0, q.f_p, color=COL[smp], lw=2.2)
                    ax.plot(d + OFF[smp], q.f_p, "o", ms=2.5, color=COL[smp])
                    lab.append(f"{smp}: L0 {q.L0} bp, n {q.n_eff:.0f}, U {q.U:.2f} [{q.U_lo:.2f}-{q.U_hi:.2f}], "
                               f"E|d| {q.mad:.1f} bp, E[d] {q.mean_shift:+.1f} bp")
                ax.set_yscale("symlog", linthresh=0.01); ax.set_ylim(0, 1.3)
                ax.axvspan(-r[r.locus == lid].delta.iloc[0] + 0.5, r[r.locus == lid].delta.iloc[0] - 0.5,
                           color="#eeeeec", zorder=0)
                ax.set_xlabel("d = L − L0 (bp)", fontsize=8); ax.tick_params(labelsize=7)
                ax.set_title(f"{key[lid]} · {gene[lid]} · {rank} allele", fontsize=8)
                ax.text(0.01, 0.97, "\n".join(lab), transform=ax.transAxes, va="top", fontsize=6.2)
                if j == 0:
                    ax.set_ylabel("f (mass)", fontsize=8)
        for ax in axes.ravel()[2 * len(lids[p0:p0 + 4]):]:
            ax.axis("off")
        fig.suptitle("Stage 2 NPMLE of the true-length distribution, on each allele's deviation scale "
                     "(grey band: |d| < Δ; UCL blue, Leiden orange, HG002 grey)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.98)); pdf.savefig(fig); plt.close(fig)
print("wrote", f"{B}/brain_vs_hg002_stage2.pdf")
