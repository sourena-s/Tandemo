"""UCL LOC101928932 haplotype 2: consensus of the +3 bp reads (repeat + 128 bp flanks) against TRGT's allele."""
import sys, collections
import pysam
sys.path.insert(0, "/projects/prjs2178/repo_clones/Tandemo")
from tandemo.catalog import read_catalog
from tandemo.pileup import collect, realign, FLANK_BP
from tandemo.phasing import load_het_sites, locus_ps, PhaseParams
from tandemo.trgt import read_trgt
from tandemo.extract import Params
B = "/projects/0/prjs2178/seed/pacbio/revio_ucl_t2t"
REF = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa"
CAT = "/projects/prjs2178/genomic_refs/human/T2T-CHM13v2.0/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz"
lid = "chr9_150464855_150465288_trsolve"
L = [x for x in read_catalog(CAT, ["chr9"], max_len=1 << 62) if x.id == lid][0]
t = read_trgt(f"{B}/hiphase/UCL.trgt.phased.vcf.gz", "UCL", {lid})[lid]
p = Params()
with pysam.AlignmentFile(f"{B}/hiphase/UCL.haplotagged.bam") as bf, pysam.FastaFile(REF) as fa:
    sites = load_het_sites(f"{B}/hiphase/UCL.deepvariant.phased.vcf.gz", "chr9", "UCL", L.start - 100000, L.end + 100000, REF)
    ps = locus_ps(sites, L.start, L.end, t.get("ps") if t.get("phased") else None)
    rows = collect(bf, fa, L, sites, ps, p, PhaseParams(), 10000)
    left = fa.fetch("chr9", L.start - p.flank, L.start).upper()
    right = fa.fetch("chr9", L.end, L.end + p.flank).upper()
    ref_rep = fa.fetch("chr9", L.start, L.end).upper()
print(f"TRGT alleles (GT order) {t['al']}, phased {t.get('phased')}; flank {p.flank} bp; reference repeat {len(ref_rep)} bp")
# haplotype 2's allele: the TRGT allele whose length is 409
k = [i for i, a in enumerate(t["al"]) if a == 409][0]
allele = t["seq"][k]
target = left + allele + right
A0, A1 = len(left), len(left) + len(allele)
sel = [r for r in rows if r["hap"] == 2 and r["Y"] - 409 == 3]
print(f"haplotype 2 reads: {sum(r['hap'] == 2 for r in rows)}; at +3 bp: {len(sel)}")
# per target column: majority base (or deletion); per gap: majority inserted string
cols = [collections.Counter() for _ in target]
gaps = [collections.Counter() for _ in range(len(target) + 1)]
for r in sel:
    base, ins, _ = realign(r["seg"], target)
    for i in range(len(target)):
        cols[i][r["seg"][base[i]] if base[i] >= 0 else "-"] += 1
    for i in range(len(target) + 1):
        gaps[i]["".join(r["seg"][q] for q in ins.get(i, []))] += 1
n = len(sel)
cons, diffs = [], []
where = lambda i: f"left flank {i - A0:+d}" if i < A0 else (f"repeat pos {i - A0 + 1}" if i < A1 else f"right flank +{i - A1 + 1}")
for i in range(len(target) + 1):
    g, gn = gaps[i].most_common(1)[0]
    if g and gn > n / 2:
        cons.append(g); diffs.append((i, "insertion", "", g, gn, where(i) if i < len(target) else "end"))
    if i < len(target):
        b, bn = cols[i].most_common(1)[0]
        if b != "-":
            cons.append(b)
        if b != target[i]:
            diffs.append((i, "deletion" if b == "-" else "substitution", target[i], b, bn, where(i)))
cons = "".join(cons)
print(f"consensus {len(cons)} bp vs TRGT allele + flanks {len(target)} bp (difference {len(cons) - len(target):+d})\n")
print("differences of the +3 reads' consensus from TRGT's allele (with 128 bp reference flanks):")
for i, kind, a, b, cnt, w in diffs:
    ctx = target[max(i - 12, 0):i] + "[" + (a or "^") + "]" + target[i + (1 if a else 0):i + 12 + (1 if a else 0)]
    print(f"  {w:20s} {kind:12s} TRGT '{a}' -> reads '{b}' ({cnt}/{n} reads)   context {ctx}")
# is TRGT's allele just the reference here, or its own sequence?
same_as_ref = allele == ref_rep
print(f"\nTRGT haplotype-2 allele identical to the reference repeat: {same_as_ref} (reference {len(ref_rep)} bp, allele {len(allele)} bp)")
print("TRGT allele  :", allele[:120] + "…")
c_rep = cons[A0:len(cons) - (len(target) - A1)]
print("reads' repeat:", c_rep[:120] + "…")

# which reads carry TRGT's bases at the three repeat substitutions and the left-flank site?
print("\nbase at each differing site, by read group (TRGT base / reads' consensus base):")
sites_i = [i for i, kind, *_ in diffs if kind == "substitution"]
groups = {"hap 2, +2..+4": [r for r in rows if r["hap"] == 2 and 2 <= r["Y"] - 409 <= 4],
          "hap 2, +12..+20 (incl. the outlier cluster)": [r for r in rows if r["hap"] == 2 and r["Y"] - 409 >= 12],
          "hap 1": [r for r in rows if r["hap"] == 1]}
for name, rs in groups.items():
    tally = []
    for i in sites_i:
        c = collections.Counter()
        for r in rs:
            base, ins, _ = realign(r["seg"], target)
            c[r["seg"][base[i]] if base[i] >= 0 else "-"] += 1
        tally.append(f"{where(i)}: TRGT {target[i]} {c.get(target[i], 0)}, other {sum(c.values()) - c.get(target[i], 0)}")
    print(f"  {name} ({len(rs)} reads): " + " | ".join(tally))

# anchoring: does each read reach past the segmental duplication around the locus into unique sequence?
from tandemo import segdup
import numpy as np
sp = segdup.for_reference(REF)
iv = segdup.intervals(sp, "chr9", L.start - 2_000_000, L.end + 2_000_000)
blk = [(s, e) for s, e in iv if s <= L.start and e >= L.end]
seg_s, seg_e = blk[0] if blk else (L.start, L.end)
print(f"\nsegmental duplication around the locus: chr9:{seg_s:,}-{seg_e:,} ({(seg_e - seg_s) / 1000:.1f} kb; "
      f"repeat at {L.start - seg_s:,} bp from its start, {seg_e - L.end:,} bp from its end)")
info = {}
with pysam.AlignmentFile(f"{B}/hiphase/UCL.haplotagged.bam") as bf:
    for a in bf.fetch("chr9", L.start, L.end):
        if a.is_secondary or a.is_supplementary:
            continue
        nm = a.get_tag("NM") if a.has_tag("NM") else -1
        info[a.query_name] = dict(mapq=a.mapping_quality, start=a.reference_start, end=a.reference_end,
                                  err=nm / max(a.query_alignment_length, 1),
                                  anchored=a.reference_start < seg_s or a.reference_end > seg_e,
                                  out_bp=max(seg_s - a.reference_start, 0) + max(a.reference_end - seg_e, 0))
for name, rs in groups.items():
    d = [info[r["name"]] for r in rs if r["name"] in info]
    mq = [x["mapq"] for x in d]; er = [100 * x["err"] for x in d]
    anc = sum(x["anchored"] for x in d)
    print(f"  {name}: {len(d)} reads | MAPQ median {np.median(mq):.0f} (min {min(mq)}) | NM per 100 aligned bp "
          f"median {np.median(er):.2f} | reach unique sequence beyond the segdup: {anc}/{len(d)} "
          f"(median {np.median([x['out_bp'] for x in d]):,.0f} bp beyond)")
