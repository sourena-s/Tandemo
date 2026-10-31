"""Loading extraction outputs, phase weights, and linking TRGT alleles to haplotypes for training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .phasing import PhaseParams, phase_weight
from .store import read_table, to_pandas

LOCUS_KEEP = ["locus", "motif", "period", "ref_len", "gc", "hp_frac", "max_hp", "cx3", "flank_cx", "locus_ps",
              "al1", "al2", "ploidy", "trgt_phased", "n_overlap", "n_span", "read_lens", "pad_d1", "pad_d2",
              "pad_status", "pad", "segdup", "elem_gc", "elem_entropy", "purity", "n_motifs"]


TRGT_READ_TOLERANCE = 3.0     # bp: largest gap between a haplotype's TRGT allele (after the pad correction) and the
#                               median length of its reads; beyond it the TRGT allele is not trusted
UNTRUSTED_TRGT = "drop"       # what then: "drop" the locus, or "local": the allele from the reads (median +- 1 bp)
CORRECTED = "corrected: TRGT allele already includes the pad indel, label = TRGT allele"
EXCLUDED = "excluded: TRGT allele not supported by the reads"
LOCAL = "local allele: TRGT allele disregarded, label = median of the haplotype's main read group (densest 3 bp window)"

TOLERANCE_HELP = ("largest gap in bp between a haplotype's TRGT allele (after the pad double-count correction) and the "
                  "median length of its reads; beyond it the TRGT allele is not trusted (default 3)")
UNTRUSTED_HELP = ("what to do with a haplotype whose TRGT allele is not trusted (see --trgt-read-tolerance): 'drop' "
                  "leaves the locus out; 'local' assembles the allele from the haplotype's main read group, the densest "
                  "+-1 bp window of read lengths (>= 3 reads and >= a third of the haplotype), and uses it from training on, "
                  "else drops it (default drop)")


# what training takes from the loci table: no motif string or read-length list per read
TRAIN_LOCUS_COLUMNS = [c for c in LOCUS_KEEP if c not in ("motif", "read_lens")]

TRAIN_READ_COLUMNS = ["locus", "chrom", "Y", "rq", "np", "read_len", "mapq", "n_sa", "bq_rep", "bq_flank",
                      "dist_end", "flank_id_min", "n_het", "llr", "mC_flank", "mC_rep", "mC_cert"]


def load(prefix: str, phase: PhaseParams | None = None, sample: str | None = None, loci_keep=None, columns=None,
         locus_columns=None, genotyped_only: bool = False):
    """(reads with their locus features, loci) for one extraction prefix.
    `loci_keep`: only these loci, filtered while reading; `columns`: only these read columns (training needs no
    read names); `locus_columns`: the locus features attached to each read (LOCUS_KEEP); `genotyped_only`: only
    loci with both TRGT alleles and unambiguous pad indels (training).
    Locus features are attached by each read's row in the loci table (one Arrow lookup), not a pandas merge, so
    100 M reads cost one copy of the attached columns."""
    import pyarrow as pa
    import pyarrow.compute as pc
    loci = read_table(prefix, "loci", None, loci_keep)
    for c in ("pad_d1", "pad_d2", "pad_status"):          # extractions made before the pad columns
        if c not in loci:
            loci[c] = 0
    if "pad" not in loci:                                 # extractions made with the fixed 10 bp pad
        loci["pad"] = 10
    if "segdup" not in loci:                              # extractions made before the segdup column
        from .segdup import for_loci, overlap
        path = for_loci(loci["locus"])
        loci["segdup"] = [overlap(path, c, s, e) for c, s, e in zip(loci["chrom"], loci["start"], loci["end"])]
    if "purity" not in loci:                              # motif-element inputs, from the motifs and the reference
        from .elements import annotate, reference_for_loci
        annotate(loci, reference_for_loci(loci["locus"]))
    if genotyped_only:
        loci = loci[(loci["al1"] >= 0) & (loci["al2"] >= 0) & (loci["pad_status"] == 0)]
        loci_keep = set(loci["locus"])
    loci = loci.reset_index(drop=True)
    t = read_table(prefix, "reads", columns, loci_keep, arrow=True)
    ids = pa.array(loci["locus"].to_numpy(dtype=object), type=pa.string())
    idx = pc.index_in(t["locus"], value_set=ids).to_numpy(zero_copy_only=False)
    reads = to_pandas(t)
    del t
    for c in (locus_columns or LOCUS_KEEP):
        if c != "locus":
            reads[c] = loci[c].to_numpy()[idx]
    reads["w1"] = phase_weight(reads["n_het"].to_numpy(), reads["llr"].to_numpy(), phase)
    reads["sample"] = sample or prefix
    loci["sample"] = sample or prefix
    return reads, loci


def assign_alleles(reads: pd.DataFrame, inplace: bool = False, tolerance: float = TRGT_READ_TOLERANCE,
                   untrusted: str = UNTRUSTED_TRGT) -> pd.DataFrame:
    """Columns L1, L2: the TRGT allele lengths of haplotype 1 and 2 of each read's locus, on the read's Y
    scale: plus the germline indels of that haplotype in the measurement pads (pad_d1, pad_d2).

    Phased TRGT call (HiPhase, same block as the reads): allele k is haplotype k + 1.
    Unphased call: the orientation is taken from the reads with confident phase evidence (the one
    that puts each haplotype's median length nearer its allele). Without such reads the two
    orientations give the same training loss, because every read then has w1 = 0.5.
    Haploid calls (al1 == al2 from one AL value) are simply both alleles equal.
    Then trgt_read_check: pad indels TRGT already counted are taken out of the label, and loci whose TRGT allele
    the reads do not support are marked (label_ok False); the per-haplotype record is in .attrs["trgt_read"].
    """
    out = reads if inplace else reads.copy()
    out["L1"] = out["al1"] + out["pad_d1"]
    out["L2"] = out["al2"] + out["pad_d2"]
    # unphased diploid calls: orient by the medians of the confidently phased reads of each haplotype
    unph = (out["trgt_phased"] != 1) & (out["al1"] != out["al2"])
    if unph.any():
        sub = out.loc[unph, ["locus", "Y", "w1", "al1", "al2", "pad_d1", "pad_d2"]]
        m1 = sub[sub["w1"] > 0.9].groupby("locus", observed=True)["Y"].median()
        m2 = sub[sub["w1"] < 0.1].groupby("locus", observed=True)["Y"].median()
        L = sub.groupby("locus", observed=True)[["al1", "al2", "pad_d1", "pad_d2"]].first()
        m1, m2 = m1.reindex(L.index), m2.reindex(L.index)
        a1e1, a2e2 = L["al1"] + L["pad_d1"], L["al2"] + L["pad_d2"]
        a2e1, a1e2 = L["al2"] + L["pad_d1"], L["al1"] + L["pad_d2"]
        keep = (m1 - a1e1).abs().fillna(0) + (m2 - a2e2).abs().fillna(0)
        swap = (m1 - a2e1).abs().fillna(0) + (m2 - a1e2).abs().fillna(0)
        flip = set(L.index[(swap < keep) & (m1.notna() | m2.notna())])
        if flip:
            rows = out["locus"].isin(flip).to_numpy() & unph.to_numpy()
            al1, al2 = out.loc[rows, "al1"].to_numpy(), out.loc[rows, "al2"].to_numpy()
            out.loc[rows, "L1"] = al2 + out.loc[rows, "pad_d1"].to_numpy()
            out.loc[rows, "L2"] = al1 + out.loc[rows, "pad_d2"].to_numpy()
    return trgt_read_check(out, tolerance, untrusted)


def trgt_read_check(out: pd.DataFrame, tolerance: float = TRGT_READ_TOLERANCE, untrusted: str = UNTRUSTED_TRGT,
                    min_reads: int = 5, min_votes: int = 3, vote_frac: float = 0.75,
                    local_frac: float = 1 / 3) -> pd.DataFrame:
    """TRGT-read discrepancy per haplotype, from the median read length Y of its confidently phased reads; at a
    homozygous locus (same label and pad indel on both haplotypes) from all of its reads, phased or not:

    pad indel counted twice: the label is TRGT allele + pad indel d, but where TRGT has already absorbed a
      repeat-like germline indel next to the repeat into its allele, the reads (which count d once) sit at the
      TRGT allele itself. A vote: each read is nearer one of the two labels, TRGT allele or allele + d; when
      |d| > 2 bp and >= vote_frac (75%) of >= min_votes (3) reads are nearer the TRGT allele, the label becomes
      the TRGT allele. A fraction, so depth does not matter and a few unstable reads cannot swing it; a floor,
      so one or two reads cannot decide. The read lengths are never changed.
    TRGT allele not trusted: when the median is still more than `tolerance` bp from the label (>= min_reads reads),
      untrusted = "drop": the locus is marked label_ok = False and left out of training and the outlier scan;
      untrusted = "local": the label becomes the median of the haplotype's main read group: the +-1 bp window of
      read lengths holding the most reads (ties: nearer the TRGT allele), when it has >= min_votes reads and >=
      local_frac (a third) of the haplotype (a second group, e.g. a paralog or a mosaic allele, cannot pull it);
      else dropped.
    Columns added: label_resid (largest |median - label| of the two haplotypes after the correction), label_ok.
    out.attrs["trgt_read"]: one row per corrected or excluded haplotype."""
    loci = out.groupby("locus", sort=False)[["L1", "L2", "pad_d1", "pad_d2"]].first()
    resid = pd.Series(0.0, index=loci.index)
    bad = pd.Series(False, index=loci.index)
    rows = []
    pooled = out.groupby("locus", sort=False)["Y"].agg(["median", "size"])
    hom = (loci["L1"] == loci["L2"]) & (loci["pad_d1"] == loci["pad_d2"])
    for h, sel in ((1, out["w1"] > 0.9), (2, out["w1"] < 0.1)):
        m = out[sel].groupby("locus")["Y"].agg(["median", "size"]).reindex(loci.index)
        m.loc[hom] = pooled.loc[hom[hom].index]
        voters = out[sel.to_numpy() | out["locus"].map(hom).to_numpy()]
        Lv, dv = voters[f"L{h}"], voters[f"pad_d{h}"]
        near_al = ((voters["Y"] - (Lv - dv)).abs() < (voters["Y"] - Lv).abs()).groupby(voters["locus"]).agg(
            ["sum", "size"]).set_axis(["votes_al", "votes"], axis=1)
        m = m.join(near_al)
        m = m[m["votes"] >= min_votes]
        lab, d = loci.loc[m.index, f"L{h}"], loci.loc[m.index, f"pad_d{h}"]
        al = lab - d
        double = (d.abs() > 2) & (m["votes_al"] >= vote_frac * m["votes"])
        if double.any():
            fix = out["locus"].isin(set(double.index[double])).to_numpy()
            out.loc[fix, f"L{h}"] = out.loc[fix, f"L{h}"] - out.loc[fix, f"pad_d{h}"]
        new = lab.where(~double, al)
        diff = (m["median"] - new).abs()
        excl = (diff > tolerance) & (m["size"].fillna(0) >= min_reads)
        local = pd.Series(False, index=m.index)
        if untrusted == "local" and excl.any():         # the allele from the reads: median +- 1 bp
            ids = excl.index[excl]
            v = voters[voters["locus"].isin(set(ids))]
            found = {}
            for lid_, ys in v.groupby("locus")["Y"]:      # the densest +-1 bp window of this haplotype's lengths
                ys = ys.to_numpy()
                cands = np.unique(ys)
                counts = np.array([np.sum(np.abs(ys - c) <= 1) for c in cands])
                best = cands[counts == counts.max()]
                c = best[np.argmin(np.abs(best - new[lid_]))]
                core = ys[np.abs(ys - c) <= 1]
                if len(core) >= min_votes and len(core) >= local_frac * len(ys):
                    found[lid_] = int(round(float(np.median(core))))
            if found:
                lab_local = pd.Series(found)
                fix = out["locus"].isin(set(lab_local.index)).to_numpy()
                out.loc[fix, f"L{h}"] = out.loc[fix, "locus"].map(lab_local).to_numpy().astype(out[f"L{h}"].dtype)
                new.loc[lab_local.index] = lab_local.astype(new.dtype)
                local.loc[lab_local.index] = True
                excl.loc[lab_local.index] = False
                diff = (m["median"] - new).abs()
        resid.loc[diff.index] = np.maximum(resid.loc[diff.index], diff)
        bad.loc[excl.index[excl]] = True
        flag = double | excl | local
        rows.append(pd.DataFrame({"locus": m.index[flag], "haplotype": h, "trgt_allele_bp": al[flag].astype(int),
                                  "pad_indel_bp": d[flag].astype(int), "label_bp": new[flag].astype(int),
                                  "read_median_bp": m["median"][flag], "n_reads": m["votes"][flag].astype(int),
                                  "votes_trgt_allele": (m["votes_al"] / m["votes"])[flag].round(2),
                                  "trgt_read_diff_bp": (m["median"] - new)[flag],
                                  "action": np.select([excl[flag], local[flag]], [EXCLUDED, LOCAL], CORRECTED)}))
    out["label_resid"] = out["locus"].map(resid).astype(float)
    out["label_ok"] = ~out["locus"].map(bad).astype(bool)
    out.attrs["trgt_read"] = pd.concat(rows, ignore_index=True)
    return out


def local_haplotypes(rec: pd.DataFrame) -> dict:
    """{locus: haplotypes} whose allele was assembled from the reads (untrusted = "local")."""
    sub = rec[rec["action"] == LOCAL] if len(rec) else rec
    return {lid: set(g["haplotype"].astype(int)) for lid, g in sub.groupby("locus")} if len(sub) else {}


def trgt_read_summary(rec: pd.DataFrame) -> str:
    n_c = rec.loc[rec["action"] == CORRECTED, "locus"].nunique()
    n_x = rec.loc[rec["action"] == EXCLUDED, "locus"].nunique()
    n_l = rec.loc[rec["action"] == LOCAL, "locus"].nunique()
    return (f"TRGT-read discrepancy: {n_c} loci with a pad indel TRGT already counted (label corrected), "
            f"{n_l} loci with a local allele from the reads, {n_x} loci left out (TRGT allele not trusted)")


def trgt_read_table(prefix: str, tolerance: float = TRGT_READ_TOLERANCE, untrusted: str = UNTRUSTED_TRGT,
                    log=None) -> pd.DataFrame:
    """Every genotyped locus of an extraction through trgt_read_check: the corrected and the excluded haplotypes,
    with the locus coordinates and motif (the list to keep out of final results)."""
    info = read_table(prefix, "loci", ["locus", "chrom", "start", "end", "motif", "period", "al1", "al2",
                                       "pad_status"])
    ok = info[(info.al1 >= 0) & (info.al2 >= 0) & (info.pad_status == 0)]
    recs = []
    for chrom, ids in ok.groupby("chrom", sort=False)["locus"]:   # one chromosome at a time: bounded memory
        r, _ = load(prefix, columns=["locus", "chrom", "Y", "n_het", "llr"], locus_columns=TRAIN_LOCUS_COLUMNS,
                    loci_keep=set(ids), genotyped_only=True)
        if len(r):
            recs.append(assign_alleles(r, inplace=True, tolerance=tolerance, untrusted=untrusted).attrs["trgt_read"])
        del r
    rec = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["locus", "action"])
    if log:
        log(f"{prefix}: {trgt_read_summary(rec)}")
    info = info[["locus", "chrom", "start", "end", "motif", "period"]]
    return (info.merge(rec, on="locus").sort_values(["chrom", "start", "haplotype"])
            [["chrom", "start", "end", "locus", "motif", "period", "haplotype", "trgt_allele_bp", "pad_indel_bp",
              "label_bp", "read_median_bp", "n_reads", "votes_trgt_allele", "trgt_read_diff_bp", "action"]])


PERIOD_STRATA = [1, 2, 3, 4, 5, 7, 16, 51]          # lower edges: 1, 2, 3, 4, 5-6, 7-15, 16-50, > 50 bp
LENGTH_STRATA = [0, 20, 50, 100, 200, 500, 1000]     # reference length classes, bp


def sample_loci(prefixes: list[str], per_stratum: int, seed: int = 0, log=None) -> dict:
    """{prefix: kept locus IDs}: at most `per_stratum` genotyped loci (TRGT alleles, >= 1 spanning read)
    per period x reference-length class over all inputs together, so rare classes (long VNTRs, long
    periods) are kept whole and the abundant short STRs are capped."""
    frames = []
    for p in prefixes:
        L = read_table(p, "loci", ["locus", "period", "ref_len", "al1", "al2", "n_span", "pad_status"])
        L = L[(L["al1"] >= 0) & (L["al2"] >= 0) & (L["n_span"] > 0)]
        if "pad_status" in L:
            L = L[L["pad_status"] == 0]
        frames.append(L.assign(prefix=p))
    L = pd.concat(frames, ignore_index=True)
    L["stratum"] = (np.searchsorted(PERIOD_STRATA, L["period"], side="right") * 10
                    + np.searchsorted(LENGTH_STRATA, L["ref_len"], side="right"))
    keep = L.groupby("stratum", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_stratum), random_state=seed))
    if log:
        log(f"sampled {len(keep)} of {len(L)} genotyped loci ({L['stratum'].nunique()} strata, "
            f"<= {per_stratum} each), {int(keep['n_span'].sum())} spanning reads")
    return {p: set(g["locus"]) for p, g in keep.groupby("prefix")}


def training_reads(prefixes: list[str], phase: PhaseParams | None = None, per_stratum: int | None = None,
                   seed: int = 0, log=None, tolerance: float = TRGT_READ_TOLERANCE,
                   untrusted: str = UNTRUSTED_TRGT) -> pd.DataFrame:
    want = sample_loci(prefixes, per_stratum, seed, log) if per_stratum else None
    frames = []
    for p in prefixes:
        if want is not None and not want.get(p):
            continue
        r, _ = load(p, phase, sample=p, loci_keep=want[p] if want is not None else None, columns=TRAIN_READ_COLUMNS,
                    locus_columns=TRAIN_LOCUS_COLUMNS, genotyped_only=True)   # pad indels unambiguous
        r = assign_alleles(r, inplace=True, tolerance=tolerance, untrusted=untrusted)
        if log:
            log(f"{p}: {trgt_read_summary(r.attrs['trgt_read'])} (tolerance {tolerance:g} bp, untrusted TRGT: {untrusted}, "
                f"of {r['locus'].nunique()} loci)")
        neg = (r["L1"] < 1) | (r["L2"] < 1)               # pad deletion longer than the TRGT allele
        if neg.any():
            if log:
                log(f"{p}: {r.loc[neg, 'locus'].nunique()} loci left out: a haplotype label below 1 bp "
                    f"(pad deletion longer than the TRGT allele)")
            r["label_ok"] &= ~neg
        r.attrs.clear()                                   # the TRGT-read record is logged; pd.concat compares attrs
        ok = r["label_ok"].to_numpy()
        frames.append(r if ok.all() else r[ok].reset_index(drop=True))
        del r
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    del frames
    df["w1_train"] = df["w1"].fillna(0.5)
    return df


def locus_grid(Y: np.ndarray, anchors, D: int, margin: int = 10, max_size: int = 6000) -> np.ndarray:
    """Candidate true lengths for one locus: every integer from below the shortest to above the
    longest read (and the TRGT alleles), widened by the core error range D."""
    vals = list(np.asarray(Y, dtype=int)) + [int(a) for a in anchors if a is not None and a >= 0]
    lo = max(min(vals) - D - margin, 0)
    hi = max(vals) + D + margin
    if hi - lo + 1 > max_size:          # very spread loci: keep the full range, coarser near the ends
        core = np.arange(lo, hi + 1)
        return core[np.linspace(0, len(core) - 1, max_size).astype(int)]
    return np.arange(lo, hi + 1)
