import glob, pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
B = "/projects/0/prjs2178/seed/pacbio"
def load(d):
    P = f"{B}/{d}/tandemo/instability"
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/chr*/chr*.parquet"))
                      if not f.endswith(".reads.parquet")], ignore_index=True)
T = {"UCL": load("revio_ucl_t2t"), "Leiden": load("leiden_t2t"), "HG002": load("hg002_t2t_tandemo")}
print(list(T["HG002"].columns))
for s, t in T.items():
    t = t[t.chrom != "chrY"]                     # HG002 chrY scan missing
    print(f"{s:7s} loci {len(t):7d}  median conf reads {t.confident.median():5.0f}  flagged reads {int(t.unstable.sum()):6d}"
          f"  per 1k conf {1000*t.unstable.sum()/t.confident.sum():.2f}  loci>=1 {int((t.unstable>0).sum()):6d}"
          f"  recurrent {int(t.recurrent.sum()):5d}")
H = T["HG002"].set_index("locus")
for s in ("UCL", "Leiden"):
    t = T[s].sort_values(["recurrent", "unstable", "min_log10p"], ascending=[False, False, True])
    rec = t[t.recurrent.astype(bool)]
    j = rec.join(H[["confident", "unstable", "recurrent", "L1", "L2"]], on="locus", rsuffix="_hg")
    print(f"\n== {s}: {len(rec)} recurrent loci; in HG002 table {j.confident_hg.notna().sum()}; also recurrent in HG002 "
          f"{int(j.recurrent_hg.fillna(0).astype(bool).sum())}; any HG002 flagged read {int((j.unstable_hg.fillna(0)>0).sum())}")
    print(j.head(20)[["locus", "period", "L1", "L2", "confident", "unstable", "L1_hg", "L2_hg", "confident_hg",
                      "unstable_hg", "recurrent_hg"]].to_string(index=False))
# HG002 own top, and whether brains share it
t = T["HG002"].sort_values(["recurrent", "unstable", "min_log10p"], ascending=[False, False, True])
rec = t[t.recurrent.astype(bool)]
print(f"\n== HG002: {len(rec)} recurrent")
for s in ("UCL", "Leiden"):
    o = T[s].set_index("locus")
    print(f"   recurrent also in {s}: {int(o.reindex(rec.locus).recurrent.fillna(0).astype(bool).sum())}")
print(rec.head(15)[["locus", "period", "L1", "L2", "confident", "unstable", "delta_units"]].to_string(index=False))
# shared loci: flagged-read fraction correlation
for s in ("UCL", "Leiden"):
    m = T[s].merge(T["HG002"], on="locus", suffixes=("", "_hg"))
    m = m[(m.confident >= 20) & (m.confident_hg >= 20)]
    a, b = m.unstable / m.confident, m.unstable_hg / m.confident_hg
    print(f"{s} vs HG002 shared loci (>=20 conf reads each) {len(m)}: flagged frac mean {a.mean():.4f} vs {b.mean():.4f}; "
          f"spearman {a.corr(b, method='spearman'):.3f}; brain-only flagged>=3 & HG002 0: {int(((m.unstable>=3)&(m.unstable_hg==0)).sum())}"
          f"; HG002-only: {int(((m.unstable_hg>=3)&(m.unstable==0)).sum())}")
