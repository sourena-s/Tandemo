"""Run Stage 2 over every locus of one subject: Stage 1 curves, then the per-locus fits.

Outputs
  <out>.lengths.parquet   one row per subject x locus x haplotype (see stage2 for the columns).
                          Haplotype labels from HiPhase are arbitrary per phase block, so group
                          comparisons should use hap_rank (short / long / tie / haploid).
  <out>.curves.parquet    optional (--write-curves): each read's Stage 1 log-likelihood curve,
                          kept where it is within 20 log units of its maximum, with the read's phase
                          weight. Enough to refit Stage 2, or a group-level model, without network 1.
Parquet needs pyarrow; without it the tables are written as gzip TSV.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import load, locus_grid
from .noise import NoiseModel
from .phasing import PhaseParams
from .stage2 import FitParams, fit_locus

LEAD = ["subject", "locus", "chrom", "start", "end", "motif", "period", "hap", "hap_rank", "status"]


def write_table(df: pd.DataFrame, path: str) -> str:
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.replace(".parquet", ".tsv.gz")
        df.to_csv(alt, sep="\t", index=False)
        return alt


def run(prefix: str, model: NoiseModel, out: str, subject: str | None = None, p: FitParams | None = None,
        phase: PhaseParams | None = None, threads: int = 1, chunk: int = 500, write_curves: bool = False,
        log=print):
    p = p or FitParams()
    subject = subject or os.path.basename(prefix)
    reads, loci = load(prefix, phase, subject)
    loci = loci.set_index("locus", drop=False)
    groups = [(lid, g) for lid, g in reads.groupby("locus", sort=False) if len(g) >= p.min_reads]
    rows, curves = [], []
    ex = ProcessPoolExecutor(threads) if threads > 1 else None
    try:
        for c0 in range(0, len(groups), chunk):
            jobs = []
            for lid, g in groups[c0:c0 + chunk]:
                L = loci.loc[lid]
                grid = locus_grid(g["Y"].to_numpy(), [L["al1"], L["al2"]], model.bins.D)
                logl = model.log_curves(g, grid).astype(np.float64)
                H = 1 if int(L.get("ploidy", 2)) == 1 else 2
                jobs.append((lid, logl, grid, g["w1"].to_numpy(), H, int(L["period"]), L["read_lens"],
                             (L["al1"], L["al2"]), p))
                if write_curves:
                    keep = logl >= logl.max(1, keepdims=True) - 20
                    ii, kk = np.nonzero(keep)
                    curves.append(pd.DataFrame(dict(locus=lid, read=g["read"].to_numpy()[ii], L=grid[kk],
                                                    logl=logl[ii, kk].astype(np.float32),
                                                    w1=g["w1"].to_numpy()[ii])))
            res = ex.map(fit_locus, jobs) if ex else map(fit_locus, jobs)
            for r in res:
                rows.extend(r)
            log(f"{min(c0 + chunk, len(groups))}/{len(groups)} loci")
    finally:
        if ex:
            ex.shutdown()
    res = pd.DataFrame(rows)
    if res.empty:
        res = pd.DataFrame(columns=["locus", "hap", "status"])
    res["subject"] = subject
    meta = loci[["locus", "chrom", "start", "end", "motif", "period"]].reset_index(drop=True)
    res = res.merge(meta, on="locus", how="left")
    res = res[[c for c in LEAD if c in res] + [c for c in res if c not in LEAD]]
    paths = [write_table(res, out + ".lengths.parquet")]
    if write_curves and curves:
        cv = pd.concat(curves, ignore_index=True)
        cv.insert(0, "subject", subject)
        paths.append(write_table(cv, out + ".curves.parquet"))
    return res, paths
