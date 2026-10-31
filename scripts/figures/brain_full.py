"""Brain-sample candidate unstable loci, redrawn with the current instability rules (NA19909 genome-wide model):
page per locus (UCL CDON, Leiden CDON, the other Leiden loci, the other UCL loci), then the summary tables."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo/scripts"); sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
import deviant_vntr as dv
from tandemo.data import load, assign_alleles, local_haplotypes
from tandemo.phasing import PhaseParams, read_phase
from tandemo.noise import load_model
from tandemo.genes import for_reference, context
from tandemo.pileup import run as pileup
from tandemo.extract import Params
B = "/projects/0/prjs2178/seed/pacbio"
REF = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa"
CAT = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz"
MODEL = f"{B}/pangenome-NA19909/tandemo/model/net1.pt"
OUT = f"{B}/brain_unstable_full"; os.makedirs(OUT, exist_ok=True)
S = {"NA19909": dict(ex=f"{B}/pangenome-NA19909/tandemo/ex/NA19909", bam=f"{B}/pangenome-NA19909/aligned_t2t_phased/NA19909.haplotagged.bam",
                     ph=f"{B}/pangenome-NA19909/deepvariant/NA19909.deepvariant.phased.vcf.gz",
                     tv=f"{B}/pangenome-NA19909/trgt/NA19909.trgt.phased.vcf.gz", vcf="NA19909"),
     "UCL": dict(ex=f"{B}/revio_ucl_t2t/tandemo/ex/UCL", bam=f"{B}/revio_ucl_t2t/hiphase/UCL.haplotagged.bam",
                 ph=f"{B}/revio_ucl_t2t/hiphase/UCL.deepvariant.phased.vcf.gz", tv=f"{B}/revio_ucl_t2t/hiphase/UCL.trgt.phased.vcf.gz", vcf="UCL"),
     "Leiden": dict(ex=f"{B}/leiden_t2t/tandemo/ex/leiden", bam=f"{B}/leiden_t2t/hiphase_trgt/leiden.haplotagged.bam",
                    ph=f"{B}/leiden_t2t/hiphase_trgt/leiden.deepvariant.phased.vcf.gz",
                    tv=f"{B}/leiden_t2t/hiphase_trgt/leiden.trgt.phased.vcf.gz", vcf="leiden")}
CDON = "chr11_126093829_126094756_trsolve"
CDON = "chr11_126093829_126094756_trsolve"
RFC1, FGF14 = "CANVAS_RFC1_pathogenic", "SCA27B_FGF14_pathogenic"
LEIDEN = ["chr9_148330481_148331298_trsolve", "chr9_3507918_3509888_trsolve", "chr7_47272929_47273721_trsolve",
          "chr9_25295804_25296702_trsolve", "chr2_54194105_54194228_trsolve", "chr4_153744094_153745194_trsolve"]
UCLL = ["chr12_119907035_119907158_trsolve", "chr9_150464855_150465288_trsolve", "chr9_138690232_138691160_trsolve",
        "chr4_2397291_2397946_trsolve"]
PAGES = ([("NA19909", CDON), ("UCL", CDON), ("Leiden", CDON)] + [("Leiden", x) for x in LEIDEN + [RFC1, FGF14]]
         + [("UCL", x) for x in UCLL + [RFC1, FGF14]] + [("NA19909", RFC1), ("NA19909", FGF14)])
CHROM = {RFC1: "chr4", FGF14: "chr13"}
TISSUE = {"m84230_260701_144835_s1": ("A", "amygdala", "#1b7a3d"), "m84230_260629_141223_s4": ("H", "hippocampus", "#a86b00")}
THR = -3.5
model = load_model(MODEL); genes = for_reference(REF); plt = dv._style()
pages, rows = [], []
for k, (smp, lid) in enumerate(PAGES, 1):
    s = S[smp]; chrom = CHROM.get(lid, lid.split("_")[0])
    g, _ = load(f"{s['ex']}.{chrom}", loci_keep={lid})
    if len(g) == 0:
        from tandemo.trgt import read_trgt
        t_ = read_trgt(s["tv"], s["vcf"], {lid}).get(lid, {})
        page = os.path.join(OUT, f"{k:02d}_{smp}_{lid}.svg")
        msg = (f"TRGT call: {' / '.join(map(str, t_['al']))} bp" if t_ else "no TRGT call")
        open(page, "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="220" font-family="sans-serif">'
                              f'<rect width="1240" height="220" fill="#fcfcfb"/><text x="24" y="50" font-size="18" font-weight="bold">'
                              f'{lid} · {smp}</text><text x="24" y="90" font-size="14">No read spans the repeat with clean flanks '
                              f'in this sample, so there is no read-length or pileup panel. {msg}.</text></svg>')
        pages.append(page); print(k, smp, lid, "no spanning reads;", msg); continue
    g = assign_alleles(g, untrusted="local"); g["w1_train"] = g["w1"].fillna(0.5)
    local = local_haplotypes(g.attrs["trgt_read"])
    src = {h: ("local" if h in local.get(lid, ()) else "TRGT") for h in (1, 2)}
    TAGS = {r: TISSUE[r.split("/")[0]] for r in g.read if r.split("/")[0] in TISSUE} if smp == "Leiden" else None
    g["phase"] = read_phase(g.n_hap1, g.n_hap2, PhaseParams())[0]          # >= 3 supporting, < 20% conflicting
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
           exclude_catalog=CAT, params=Params(), highlight={lid: set(g.read.to_numpy()[dev])}, labels={lid: {1: L1, 2: L2}}, local=local, read_tags=TAGS, pads={lid: int(g.pad.iloc[0])},
           log=print)
    page = f"{d}/{smp}.{lid}.panels.svg"
    dv.compose(spike, f"{d}/{lid}.svg", page); pages.append(page)
    tis = ""
    if TAGS:
        tg = np.array([TAGS.get(r, ("?", "?", ""))[0] for r in g.read])
        tis = " / ".join(f"{t_} {int((dev & (tg == t_)).sum())} of {int(((ph > 0) & (tg == t_)).sum())}" for t_ in ("A", "H"))
    rows.append(dict(sample=smp, locus=lid, gene=context(genes, chrom, int(g.start.iloc[0]), int(g.end.iloc[0])),
                     motif_bp=per, alleles=f"{L1} / {L2}", allele_source=f"{src[1]} / {src[2]}",
                     phased=int((ph > 0).sum()), flagged=int(dev.sum()),
                     hap1=int((dev & (ph == 1)).sum()), hap2=int((dev & (ph == 2)).sum()),
                     shifts_units=",".join(f"{u:+.1f}" for u in sorted(units)),
                     shifts_bp=",".join(f"{u * per:+.0f}" for u in sorted(units)),
                     pattern=("" if not len(units) else "one shared" if len(set(np.round(units, 1))) == 1 else "varied"),
                     segdup=f"{float(g.segdup.iloc[0]):.0%}", SA=int((g.n_sa.to_numpy()[dev] > 0).sum()),
                     flagged_by_tissue=tis))
    print(k, smp, lid, "flagged", int(dev.sum()))
t = pd.DataFrame(rows); t.to_csv(f"{OUT}/brain_loci.tsv", sep="\t", index=False)
cols = ["locus", "gene", "motif (bp)", "alleles (bp)", "allele source", "phased reads", "flagged reads", "flagged hap 1",
        "flagged hap 2", "shifts (motif units)", "shifts (bp)", "shift pattern", "segdup", "SA reads", "flagged by tissue"]
tt = t.rename(columns=dict(zip(["motif_bp", "alleles", "allele_source", "phased", "flagged", "hap1", "hap2", "shifts_units",
                                "shifts_bp", "pattern", "SA", "flagged_by_tissue"], cols[2:4] + cols[4:12] + ["SA reads", "flagged by tissue"])))
fig = plt.figure(figsize=(21, 10)); y0 = 0.95
fig.text(0.01, 0.98, "Candidate unstable loci and RFC1 / FGF14, current rules: confident phase (>= 3 supporting, < 20% conflicting het "
         "sites), >= max(10 bp, 1 unit) from own allele and not at the other, MAPQ 60, flank identity >= 0.99, log10 p < -3.5 under "
         "the NA19909 genome-wide network 1; TRGT allele trusted within 3 bp of the reads, else assembled from the reads (local). "
         "Shifts: each flagged read's length minus its own allele. Tissue: A amygdala, H hippocampus (flagged of phased reads).",
         fontsize=8, color="#52514e")
for smp in ("NA19909", "Leiden", "UCL"):
    sub = tt[tt["sample"] == smp][cols].copy()
    for c in ("shifts (motif units)", "shifts (bp)"):
        sub[c] = sub[c].astype(str).str.slice(0, 40)
    sub["gene"] = sub["gene"].astype(str).str.slice(0, 36)
    h = 0.04 * (len(sub) + 1)
    fig.text(0.01, y0, smp, fontsize=11, weight="bold")
    ax = fig.add_axes([0.01, y0 - 0.01 - h, 0.98, h]); ax.axis("off")
    tb = ax.table(cellText=sub.values.tolist(), colLabels=cols, loc="upper center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7)
    for (i_, j_), c in tb.get_celld().items():
        c.set_edgecolor("#d9d8d3"); c.set_linewidth(0.5)
        if i_ == 0:
            c.set_text_props(weight="bold"); c.set_facecolor("#eeede8")
    tb.auto_set_column_width(list(range(len(cols))))
    y0 -= h + 0.07
fig.savefig(f"{OUT}/tables.svg"); pages.append(f"{OUT}/tables.svg")
with open(f"{OUT}/pages.txt", "w") as fh:
    fh.write("\n".join(pages) + "\n")
print(pd.DataFrame(rows).drop(columns=["locus"]).to_string(index=False))
