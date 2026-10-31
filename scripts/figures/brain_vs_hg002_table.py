import glob, sys, pandas as pd
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.genes import context, for_reference
from tandemo import segdup
REF = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa"
B = "/projects/0/prjs2178/seed/pacbio"
D = {"UCL": "revio_ucl_t2t", "Leiden": "leiden_t2t", "HG002": "hg002_t2t_tandemo"}
I, R = {}, {}
for s, d in D.items():
    P = f"{B}/{d}/tandemo/instability"
    t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/chr*/chr*.parquet")) if not f.endswith(".reads.parquet")])
    t["recurrent"] = t.recurrent.astype(bool); I[s] = t.set_index("locus")
brain = sorted(set(I["UCL"].index[I["UCL"].recurrent]) | set(I["Leiden"].index[I["Leiden"].recurrent]))
for s, d in D.items():
    l = pd.concat([pd.read_parquet(f, columns=["locus", "read_lens"]) for f in glob.glob(f"{B}/{d}/tandemo/ex/*.loci.parquet")])
    l = l[l.locus.isin(brain)].set_index("locus").read_lens
    R[s] = l.map(lambda v: 0 if v is None or len(v) == 0 else len(v) if not isinstance(v, str) else len(v.split(",")))
g, sp = for_reference(REF), segdup.for_reference(REF)
rows = []
for lid in brain:
    c, a, b = lid.split("_")[:3]; a, b = int(a), int(b)
    src = I["Leiden"] if lid in I["Leiden"].index else I["UCL"]
    row = {"locus": f"{c}:{a+1:,}-{b:,}", "period": int(src.loc[lid, "period"]), "gene": context(g, c, a, b),
           "segdup": round(segdup.overlap(sp, c, a, b), 2)}
    for s in D:
        t = I[s]
        if lid in t.index:
            q = t.loc[lid]
            row[s] = f"{int(q.unstable)}/{int(q.confident)}" + (" *" if q.recurrent else "")
            row[f"{s}_shift"] = q.delta_units
        else:
            row[s] = f"0/{int(R[s].get(lid, 0))} reads"; row[f"{s}_shift"] = ""
    rows.append(row)
o = pd.DataFrame(rows)
o.to_csv(f"{B}/brain_vs_hg002_unstable.tsv", sep="\t", index=False)
print(o.drop(columns=[c for c in o if c.endswith("_shift")]).to_string(index=False))
print("\nshifts (motif units):")
for r in o.itertuples():
    print(r.locus, "| UCL", r.UCL_shift or "-", "| Leiden", r.Leiden_shift or "-", "| HG002", (r.HG002_shift[:60] if r.HG002_shift else "-"))
