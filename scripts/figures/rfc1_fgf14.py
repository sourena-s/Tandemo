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
OUT = f"{B}/rfc1_fgf14_pdf"; os.makedirs(OUT, exist_ok=True)
S = {"NA19909": dict(ex=f"{B}/pangenome-NA19909/tandemo/ex/NA19909", bam=f"{B}/pangenome-NA19909/aligned_t2t_phased/NA19909.haplotagged.bam",
                     ph=f"{B}/pangenome-NA19909/deepvariant/NA19909.deepvariant.phased.vcf.gz",
                     tv=f"{B}/pangenome-NA19909/trgt/NA19909.trgt.phased.vcf.gz", vcf="NA19909"),
     "UCL": dict(ex=f"{B}/revio_ucl_t2t/tandemo/ex/UCL", bam=f"{B}/revio_ucl_t2t/hiphase/UCL.haplotagged.bam",
                 ph=f"{B}/revio_ucl_t2t/hiphase/UCL.deepvariant.phased.vcf.gz", tv=f"{B}/revio_ucl_t2t/hiphase/UCL.trgt.phased.vcf.gz", vcf="UCL"),
     "Leiden": dict(ex=f"{B}/leiden_t2t/tandemo/ex/leiden", bam=f"{B}/leiden_t2t/hiphase_trgt/leiden.haplotagged.bam",
                    ph=f"{B}/leiden_t2t/hiphase_trgt/leiden.deepvariant.phased.vcf.gz",
                    tv=f"{B}/leiden_t2t/hiphase_trgt/leiden.trgt.phased.vcf.gz", vcf="leiden")}
CDON = "chr11_126093829_126094756_trsolve"
PAGES = [(smp, lid) for lid in ("CANVAS_RFC1_pathogenic", "SCA27B_FGF14_pathogenic")
         for smp in ("NA19909", "UCL", "Leiden")]
CHROM = {"CANVAS_RFC1_pathogenic": "chr4", "SCA27B_FGF14_pathogenic": "chr13"}
THR = -3.5
model = load_model(MODEL); genes = for_reference(REF); plt = dv._style()
pages, rows = [], []
for k, (smp, lid) in enumerate(PAGES, 1):
    s = S[smp]; chrom = CHROM[lid]
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
pd.DataFrame(rows).to_csv(f"{OUT}/rfc1_fgf14.tsv", sep="\t", index=False)
with open(f"{OUT}/pages.txt", "w") as fh:
    fh.write("\n".join(pages) + "\n")
print(pd.DataFrame(rows).drop(columns=["locus"]).to_string(index=False))
