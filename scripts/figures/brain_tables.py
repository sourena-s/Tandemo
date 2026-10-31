"""Rebuild the last page of the brain-loci PDF with explicit units: alleles in bp, shifts in motif units and in bp."""
import sys
import pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo/scripts"); sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
import deviant_vntr as dv
OUT = "/projects/0/prjs2178/seed/pacbio/brain_instability_pdf"
t = pd.read_csv(f"{OUT}/brain_loci.tsv", sep="\t", keep_default_na=False)
units = t["shifts"].astype(str).map(lambda x: [float(v) for v in x.split(",") if v])
t["shifts_bp"] = [",".join(f"{u * p:+.0f}" for u in us) for us, p in zip(units, t["motif_bp"])]
t = t.rename(columns={"motif_bp": "motif (bp)", "alleles": "alleles (bp)", "phased": "phased reads",
                      "flagged": "flagged reads", "hap1": "flagged hap 1", "hap2": "flagged hap 2",
                      "shifts": "shifts (motif units)", "shifts_bp": "shifts (bp)", "pattern": "shift pattern",
                      "SA": "SA reads"})
cols = ["locus", "gene", "motif (bp)", "alleles (bp)", "phased reads", "flagged reads", "flagged hap 1", "flagged hap 2",
        "shifts (motif units)", "shifts (bp)", "shift pattern", "segdup", "SA reads"]
plt = dv._style()
fig = plt.figure(figsize=(19, 7.5)); y0 = 0.93
fig.text(0.01, 0.975, "Candidate unstable loci, current rules: confident phase (>= 3 supporting, < 20% conflicting het sites), "
         ">= max(10 bp, 1 unit) from own allele and not at the other, MAPQ 60, flank identity >= 0.99, log10 p < -3.5 "
         "under the NA19909 genome-wide network 1. Shifts: each flagged read's length minus its own allele.",
         fontsize=8.5, color="#52514e")
for smp in ("Leiden", "UCL"):
    sub = t[t["sample"] == smp][cols].copy()
    for c in ("shifts (motif units)", "shifts (bp)"):
        sub[c] = sub[c].astype(str).str.slice(0, 40)
    sub["gene"] = sub["gene"].astype(str).str.slice(0, 38)
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
fig.savefig(f"{OUT}/tables.svg")
t.to_csv(f"{OUT}/brain_loci.tsv", sep="\t", index=False)
print("tables page rebuilt")
