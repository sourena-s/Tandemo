import glob, pandas as pd, pyarrow.parquet as pq, numpy as np
B = "/projects/0/prjs2178/seed/pacbio"
D = {"UCL": "revio_ucl_t2t", "Leiden": "leiden_t2t", "HG002": "hg002_t2t_tandemo"}
for s, d in D.items():
    nr = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{B}/{d}/tandemo/ex/*.reads.parquet")
             if "chrX" not in f and "chrY" not in f)
    P = f"{B}/{d}/tandemo/instability"
    t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/chr*/chr*.parquet")) if not f.endswith(".reads.parquet")])
    t = t[~t.chrom.isin(["chrX", "chrY"])]; t["recurrent"] = t.recurrent.astype(bool)
    r = t[t.recurrent]
    d_ = r.delta_units.map(lambda x: np.array([float(v) for v in x.split(",")]))
    fixed = d_.map(lambda a: a.max() - a.min() <= 1.0)
    big = r.unstable / r.confident >= 0.15
    exp = t.unstable.map(lambda x: 0)
    print(f"{s:7s} reads {nr/1e6:6.1f}M  flagged reads/M {1e6*t.unstable.sum()/nr:6.1f}  recurrent/M reads {1e6*len(r)/nr:5.2f}"
          f"  recurrent: same-shift {int(fixed.sum())}, >=15% of reads {int(big.sum())}, sparse varied {int((~fixed & ~big).sum())}"
          f"  contractions {int(d_.map(lambda a: (a<0).mean()>0.5).sum())}")
