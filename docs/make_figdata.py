"""Data for the plotted figures of the method document, produced by the package itself on a simulated
locus (hap1: 30% of molecules expanded around +8 units; hap2: stable). Run from the repository root:
python docs/make_figdata.py"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tandemo.data import locus_grid  # noqa: E402
from tandemo.features import DeltaBins  # noqa: E402
from tandemo.noise import EmpiricalNoise  # noqa: E402
from tandemo.stage2 import FitParams, LocusFit, selection_curve  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figs", "data")
os.makedirs(OUT, exist_ok=True)

# HiFi-like noise: 90% exact, 4% +-1, 0.5% +-2, rare beyond
b = DeltaBins(D=30, T=2000)
d = np.arange(-30, 31)
core = np.where(d == 0, 0.9, np.where(abs(d) == 1, 0.04, np.where(abs(d) == 2, 0.005, 1e-6)))
v = np.concatenate([[1e-5], core, [1e-5]])
v = v / v.sum()
noise = EmpiricalNoise(b, np.tile(np.log(v), (16, 1)))
pd.DataFrame(dict(delta=d[20:41], p=v[1:-1][20:41])).to_csv(f"{OUT}/noise.csv", index=False)

rng = np.random.default_rng(4)
Y, w1 = [], []
for h, (L0, pi) in enumerate([(60, 0.7), (69, 1.0)]):
    for _ in range(45):
        L = L0 if rng.random() < pi else int(round(rng.normal(84, 6)))
        e = rng.choice(len(v), p=v)
        Y.append(L + (e - 31 if 0 < e < len(v) - 1 else 0))
        w1.append(np.nan if rng.random() < 0.3 else (0.995 if h == 0 else 0.005))
reads = pd.DataFrame(dict(Y=Y, period=3))
w1 = np.array(w1)
pd.DataFrame(dict(Y=Y, hap=[1] * 45 + [2] * 45, phased=~np.isnan(w1))).to_csv(f"{OUT}/reads.csv", index=False)

grid = locus_grid(reads["Y"].to_numpy(), [], b.D)
logl = noise.log_curves(reads, grid)
# three example curves: a read at L0, one at L0 + 1, one expanded
ex = [int(np.where(reads.Y == 60)[0][0]), int(np.where(reads.Y == 61)[0][0]) if (reads.Y == 61).any() else 1,
      int(np.argmax(reads.Y))]
cur = pd.DataFrame({"L": grid})
for j, i in enumerate(ex):
    cur[f"r{j}"] = np.exp(logl[i] - logl[i].max())
cur[(cur.L >= 50) & (cur.L <= 110)].to_csv(f"{OUT}/curves.csv", index=False)
with open(f"{OUT}/curves_y.tex", "w") as fh:
    fh.write("".join(f"\\def\\curveY{'abc'[j]}{{{int(reads.Y[i])}}}" for j, i in enumerate(ex)))

res = {}
for shape in ("parametric", "nonparametric"):
    rows, f, post = LocusFit(logl, grid, w1, 2, 3, None, (), FitParams(shape=shape)).run()
    res[shape] = rows
    df = pd.DataFrame({"L": grid, "f1": f[0], "f2": f[1]})
    df[(df.L >= 50) & (df.L <= 110)].to_csv(f"{OUT}/f_{shape}.csv", index=False)
with open(f"{OUT}/fit_numbers.tex", "w") as fh:
    for shape, key in (("parametric", "P"), ("nonparametric", "N")):
        r1, r2 = res[shape]
        fh.write(f"\\def\\fit{key}varA{{{r1['var_weight']:.2f}}}\\def\\fit{key}varB{{{r2['var_weight']:.2f}}}"
                 f"\\def\\fit{key}expA{{{r1['p_exp']:.2f}}}\\def\\fit{key}dll{{{r1['d_ll']:.1f}}}"
                 f"\\def\\fit{key}LA{{{r1['L0']}}}\\def\\fit{key}LB{{{r2['L0']}}}\n")

g = np.arange(0, 15001, 250)
s = selection_curve(g, ",".join(str(int(x)) for x in rng.normal(16000, 3000, 60).clip(5000)), 128)
pd.DataFrame(dict(L=g, s=s)).to_csv(f"{OUT}/selection.csv", index=False)
print("wrote", OUT)

# detection: k expanded reads among 40 per haplotype, with a sharp and a heavy-tailed noise model
heavy_core = np.where(d == 0, 0.88, np.where(abs(d) == 1, 0.04, np.where(abs(d) == 2, 0.005, 6e-4)))
hv = np.concatenate([[1e-3], heavy_core, [1e-3]])
heavy = EmpiricalNoise(b, np.tile(np.log(hv / hv.sum()), (16, 1)))
det = []
for name, m, vv in (("sharp", noise, v), ("heavy", heavy, hv / hv.sum())):
    for k in (0, 1, 2, 3, 5, 8):
        vw, dl = [], []
        for seed in range(8):
            r = np.random.default_rng(1000 + seed)
            Yk, wk = [], []
            for h, L0 in enumerate((60, 69)):
                for j in range(40):
                    L = L0 + (3 * int(r.integers(6, 16)) if (h == 0 and j < k) else 0)
                    e = r.choice(len(vv), p=vv)
                    Yk.append(L + (e - 31 if 0 < e < len(vv) - 1 else int(r.integers(31, 300)) * (1 if e else -1)))
                    wk.append(0.995 if h == 0 else 0.005)
            rd = pd.DataFrame(dict(Y=Yk, period=3))
            gr = locus_grid(rd["Y"].to_numpy(), [], b.D)
            rows, _, _ = LocusFit(m.log_curves(rd, gr), gr, np.array(wk), 2, 3, None, (), FitParams()).run()
            vw.append(rows[0]["var_weight"]); dl.append(rows[0]["d_ll"])
        det.append(dict(noise=name, k=k, var_weight=float(np.median(vw)), d_ll=float(np.median(dl))))
det = pd.DataFrame(det)
for name in ("sharp", "heavy"):
    det[det.noise == name].to_csv(f"{OUT}/detect_{name}.csv", index=False)
print(det)
