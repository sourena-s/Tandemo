"""Brain-sample candidate unstable loci, redrawn with the current instability rules (NA19909 genome-wide model):
page per locus (UCL CDON, Leiden CDON, the other Leiden loci, the other UCL loci), then the summary tables."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo/scripts"); sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
import deviant_vntr as dv
from tandemo.data import load, assign_alleles
from tandemo.phasing import PhaseParams, confident_phase
from tandemo.noise import load_model
from tandemo.genes import for_reference, context
from tandemo.pileup import run as pileup
from tandemo.extract import Params
B = "/projects/0/prjs2178/seed/pacbio"
REF = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa"
CAT = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz"
MODEL = f"{B}/pangenome-NA19909/tandemo/model/net1.pt"
OUT = f"{B}/brain_instability_pdf"; os.makedirs(OUT, exist_ok=True)
S = {"UCL": dict(ex=f"{B}/revio_ucl_t2t/tandemo/ex/UCL", bam=f"{B}/revio_ucl_t2t/hiphase/UCL.haplotagged.bam",
                 ph=f"{B}/revio_ucl_t2t/hiphase/UCL.deepvariant.phased.vcf.gz", tv=f"{B}/revio_ucl_t2t/hiphase/UCL.trgt.phased.vcf.gz", vcf="UCL"),
     "Leiden": dict(ex=f"{B}/leiden_t2t/tandemo/ex/leiden", bam=f"{B}/leiden_t2t/hiphase_trgt/leiden.haplotagged.bam",
                    ph=f"{B}/leiden_t2t/hiphase_trgt/leiden.deepvariant.phased.vcf.gz",
                    tv=f"{B}/leiden_t2t/hiphase_trgt/leiden.trgt.phased.vcf.gz", vcf="leiden")}
CDON = "chr11_126093829_126094756_trsolve"
PAGES = [("UCL", CDON), ("Leiden", CDON)] + [("Leiden", x) for x in (
    "chr9_148330481_148331298_trsolve", "chr9_3507918_3509888_trsolve", "chr7_47272929_47273721_trsolve",
    "chr9_25295804_25296702_trsolve", "chr2_54194105_54194228_trsolve", "chr4_153744094_153745194_trsolve")] + \
    [("UCL", x) for x in ("chr12_119907035_119907158_trsolve", "chr9_150464855_150465288_trsolve",
                          "chr9_138690232_138691160_trsolve", "chr4_2397291_2397946_trsolve")]
THR = -3.5
model = load_model(MODEL); genes = for_reference(REF); plt = dv._style()
pages, rows = [], []
for k, (smp, lid) in enumerate(PAGES, 1):
    s = S[smp]; chrom = lid.split("_")[0]
    g, _ = load(f"{s['ex']}.{chrom}", loci_keep={lid})
    g = assign_alleles(g); g["w1_train"] = g["w1"].fillna(0.5)
    g["phase"] = confident_phase(g.n_hap1, g.n_hap2, PhaseParams())          # >= 3 supporting, < 20% conflicting
    L1, L2, per = int(g.L1.iloc[0]), int(g.L2.iloc[0]), int(g.period.iloc[0])
    lc = model.log_curves(g, np.array([L1, L2])).astype(float) / np.log(10)
    ph, Y = g.phase.to_numpy(), g.Y.to_numpy()
    own, oth = np.where(ph == 2, L2, L1), np.where(ph == 2, L1, L2)
    lp10 = np.where(ph == 2, lc[:, 1], lc[:, 0])
    dev = ((ph > 0) & (np.abs(Y - own) > max(10, per)) & ~((np.abs(Y - oth) <= max(5, per)) & (L1 != L2))
           & (g.mapq.to_numpy() >= 60) & (g.flank_id_min.to_numpy() >= 0.99) & (lp10 < THR))
    units = (Y - own)[dev] / max(per, 1)
    d = os.path.join(OUT, f"{k:02d}_{smp}_{lid}"); os.makedirs(d, exist_ok=True)
    spike = dv.plot(plt, model, g, lp10, dev, f"{d}/len", None, None, f"{smp} · " + dv._title(genes, g))
    pileup(s["bam"], REF, CAT, d, s["ph"], s["tv"], loci_names=[lid], regions=[chrom], sample=s["vcf"], max_reads=None,
           exclude_catalog=CAT, params=Params(), highlight={lid: set(g.read.to_numpy()[dev])}, labels={lid: {1: L1, 2: L2}},
           log=print)
    page = f"{d}/{smp}.{lid}.panels.svg"
    dv.compose(spike, f"{d}/{lid}.svg", page); pages.append(page)
    rows.append(dict(sample=smp, locus=lid, gene=context(genes, chrom, int(g.start.iloc[0]), int(g.end.iloc[0])),
                     motif_bp=per, alleles=f"{L1} / {L2}", phased=int((ph > 0).sum()), flagged=int(dev.sum()),
                     hap1=int((dev & (ph == 1)).sum()), hap2=int((dev & (ph == 2)).sum()),
                     shifts=",".join(f"{u:+.1f}" for u in sorted(units)),
                     pattern="one shared" if len(set(np.round(units, 1))) == 1 else "varied",
                     segdup=f"{float(g.segdup.iloc[0]):.0%}", SA=int((g.n_sa.to_numpy()[dev] > 0).sum())))
    print(k, smp, lid, "flagged", int(dev.sum()))
t = pd.DataFrame(rows); t.to_csv(f"{OUT}/brain_loci.tsv", sep="\t", index=False)
# last page: the tables (one per sample, in page order)
cols = ["locus", "gene", "motif_bp", "alleles", "phased", "flagged", "hap1", "hap2", "shifts", "pattern", "segdup", "SA"]
fig = plt.figure(figsize=(16, 7.5)); y0 = 0.93
fig.text(0.01, 0.975, "Candidate unstable loci, current rules: confident phase (>= 3 supporting, < 20% conflicting het sites), "
         ">= max(10 bp, 1 unit) from own allele and not at the other, MAPQ 60, flank identity >= 0.99, log10 p < -3.5 under "
         "the NA19909 genome-wide network 1", fontsize=8.5, color="#52514e")
for smp in ("Leiden", "UCL"):
    sub = t[t["sample"] == smp][cols].copy()
    sub["shifts"] = sub["shifts"].str.slice(0, 48); sub["gene"] = sub["gene"].str.slice(0, 40)
    h = 0.05 * (len(sub) + 1)
    fig.text(0.01, y0, smp, fontsize=11, weight="bold")
    ax = fig.add_axes([0.01, y0 - 0.01 - h, 0.98, h]); ax.axis("off")
    tb = ax.table(cellText=sub.values.tolist(), colLabels=cols, loc="upper center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7.5)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#d9d8d3"); c.set_linewidth(0.5)
        if i == 0:
            c.set_text_props(weight="bold"); c.set_facecolor("#eeede8")
    tb.auto_set_column_width(list(range(len(cols))))
    y0 -= h + 0.09
fig.savefig(f"{OUT}/tables.svg"); pages.append(f"{OUT}/tables.svg")
with open(f"{OUT}/pages.txt", "w") as fh:
    fh.write("\n".join(pages) + "\n")
print(t.drop(columns=["locus"]).to_string(index=False))
