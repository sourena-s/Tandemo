import glob, pandas as pd, pyarrow.parquet as pq
pd.set_option("display.width", 250)
B = "/projects/0/prjs2178/seed/pacbio"
def inst(d):
    P = f"{B}/{d}/tandemo/instability"
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/chr*/chr*.parquet"))
                      if not f.endswith(".reads.parquet")], ignore_index=True)
D = {"UCL": "revio_ucl_t2t", "Leiden": "leiden_t2t", "HG002": "hg002_t2t_tandemo"}
T = {s: inst(d) for s, d in D.items()}
# genotyped loci per sample (denominator)
def nloci(d):
    fs = glob.glob(f"{B}/{d}/tandemo/ex/*.loci.parquet")
    return sum(pq.ParquetFile(f).metadata.num_rows for f in fs if "chrX" not in f and "chrY" not in f)
print("autosomes only:")
for s, t in T.items():
    a = t[~t.chrom.isin(["chrX", "chrY"])].copy(); a["recurrent"] = a.recurrent.astype(bool)
    n = nloci(D[s])
    print(f"{s:7s} extracted loci {n:7d}  flagged loci {len(a):5d} ({1e4*len(a)/n:.1f}/10k)  flagged reads {int(a.unstable.sum()):5d}"
          f"  recurrent {int(a.recurrent.sum()):4d} ({1e4*a.recurrent.sum()/n:.2f}/10k)")
x = T["HG002"][T["HG002"].chrom == "chrX"]
print(f"HG002 chrX flagged {len(x)}, recurrent {int(x.recurrent.sum())}, in PAR1 (<2.78Mb) {int((x.recurrent & (x.end<2781479)).sum())}")
# HG002 depth at brain recurrent loci
loc = pd.concat([pd.read_parquet(f, columns=["locus", "read_lens"]) for f in glob.glob(f"{B}/hg002_t2t_tandemo/tandemo/ex/*.loci.parquet")])
loc["n"] = loc.read_lens.map(lambda v: 0 if v is None else len(str(v).split(",")))
loc = loc.set_index("locus").n
for s in ("UCL", "Leiden"):
    r = T[s][T[s].recurrent.astype(bool)]
    print(f"\n{s} recurrent loci -> HG002 reads at locus (0 = not extracted) and HG002 flagged:")
    h = T["HG002"].set_index("locus")
    for q in r.itertuples():
        print(f"  {q.locus:38s} {s} {q.unstable}/{q.confident}  HG002 reads {int(loc.get(q.locus, 0)):4d}  "
              f"flagged {int(h.unstable.get(q.locus, 0))}/{int(h.confident.get(q.locus, 0))}")
