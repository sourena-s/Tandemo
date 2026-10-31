"""Synthetic tests: a CAG reference, a phased het-SNV VCF, a TRGT VCF and a BAM of spanning reads.

Extraction is checked on the BAM. Stage 2 is checked on simulated reads drawn from known length
distributions, with an empirical noise model trained on stable synthetic reads. Network 1 is
trained briefly when PyTorch is installed.
"""

import gzip

import numpy as np
import pandas as pd
import pysam
import pytest

from tandemo.data import assign_alleles, load, locus_grid, training_reads
from tandemo.extract import Params, extract
from tandemo.features import DeltaBins
from tandemo.noise import EmpiricalNoise
from tandemo.phasing import PhaseParams, phase_weight
from tandemo.stage2 import FitParams, LocusFit, selection_curve

UNIT = "CAG"
REF_UNITS = 20
SPACING = 8000
N_LOCI = 10
FL = 2500                  # flank bases of long reads (cover the het SNVs)
SHORT_FL = 400             # flank bases of short reads (no het SNVs)
SNP_OFF = (-1500, -900, 900, 1500)   # het SNV offsets from the repeat start / end


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _flip(b):
    return {"A": "C", "C": "G", "G": "T", "T": "A"}[b]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(3)
    d = tmp_path_factory.mktemp("tdm")
    parts, loci, pos = [], [], 0
    for i in range(N_LOCI):
        left = _rand(rng, SPACING // 2); parts.append(left); pos += len(left)
        rep = UNIT * REF_UNITS; loci.append((pos, pos + len(rep))); parts.append(rep); pos += len(rep)
        right = _rand(rng, SPACING // 2); parts.append(right); pos += len(right)
    ref = "".join(parts)
    fa = d / "ref.fa"
    fa.write_text(">chr1\n" + "\n".join(ref[i:i + 80] for i in range(0, len(ref), 80)) + "\n")
    pysam.faidx(str(fa))
    with open(d / "cat.bed", "w") as fh:
        for i, (s, e) in enumerate(loci):
            fh.write(f"chr1\t{s}\t{e}\tID=L{i};MOTIFS={UNIT};STRUC=({UNIT})n\n")

    # het SNVs: hap1 = reference base, hap2 = flipped; one phase block per locus
    snps = []
    for i, (s, e) in enumerate(loci):
        for off in SNP_OFF:
            p = s + off if off < 0 else e + off
            snps.append((p, ref[p], _flip(ref[p]), 1000 + i))
    vcf = d / "phased.vcf"
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=%d>\n" % len(ref))
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n')
        fh.write('##FORMAT=<ID=PS,Number=1,Type=Integer,Description="ps">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
        for p, r, a, ps in snps:
            fh.write(f"chr1\t{p + 1}\t.\t{r}\t{a}\t50\tPASS\t.\tGT:PS\t0|1:{ps}\n")
    pysam.tabix_index(str(vcf), preset="vcf", force=True)

    # germline alleles: odd loci heterozygous (+3 units on hap2)
    alleles = {i: (REF_UNITS * 3, (REF_UNITS + (3 if i % 2 else 0)) * 3) for i in range(N_LOCI)}
    tv = d / "trgt.vcf"
    with open(tv, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=%d>\n" % len(ref))
        fh.write('##INFO=<ID=TRID,Number=1,Type=String,Description="t">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n')
        fh.write('##FORMAT=<ID=AL,Number=.,Type=Integer,Description="a">\n')
        fh.write('##FORMAT=<ID=PS,Number=1,Type=Integer,Description="ps">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
        for i, (s, e) in enumerate(loci):
            a1, a2 = alleles[i]
            fh.write(f"chr1\t{s + 1}\t.\tN\t<TR>\t.\tPASS\tTRID=L{i}\tGT:AL:PS\t0|1:{a1},{a2}:{1000 + i}\n")
    pysam.tabix_index(str(tv), preset="vcf", force=True)

    reads, truth = [], {}
    for i, (s, e) in enumerate(loci):
        for hp in (1, 2):
            for r in range(14):
                L = alleles[i][hp - 1]
                u = rng.random()
                L_obs = L + (1 if u < 0.06 else -1 if u < 0.12 else 0)
                fl = SHORT_FL if r >= 10 else FL
                left, right = list(ref[s - fl:s]), list(ref[e:e + fl])
                if hp == 2:
                    for p, rb, ab, _ in snps:
                        if s - fl <= p < s:
                            left[p - (s - fl)] = ab
                        elif e <= p < e + fl:
                            right[p - e] = ab
                rep = (UNIT * (L_obs // 3 + 1))[:L_obs]
                seq = "".join(left) + rep + "".join(right)
                A = e - s
                cig = [(0, fl), (0, min(A, L_obs))]
                cig += [(1, L_obs - A)] if L_obs > A else [(2, A - L_obs)] if L_obs < A else []
                cig += [(0, fl)]
                name = f"L{i}_h{hp}_{r}"
                truth[name] = (L_obs, hp, fl == FL)
                reads.append((name, s - fl, cig, seq))
    header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": "chr1", "LN": len(ref)}]}
    ub = d / "u.bam"
    with pysam.AlignmentFile(str(ub), "wb", header=header) as out:
        for name, start, cig, seq in reads:
            a = pysam.AlignedSegment(out.header)
            a.query_name = name; a.flag = 0; a.reference_id = 0; a.reference_start = start
            a.mapping_quality = 60; a.cigartuples = cig; a.query_sequence = seq
            a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
            a.set_tag("rq", 0.999); a.set_tag("np", 12)
            out.write(a)
    bam = d / "s.bam"
    pysam.sort("-o", str(bam), str(ub))
    pysam.index(str(bam))
    prefix = str(d / "out" / "s1")
    extract(str(bam), str(fa), str(d / "cat.bed"), prefix, str(vcf) + ".gz", str(tv) + ".gz",
            params=Params())
    return dict(prefix=prefix, truth=truth, alleles=alleles, bam=str(bam), fa=str(fa), cat=str(d / "cat.bed"),
                vcf=str(vcf) + ".gz", trgt=str(tv) + ".gz")


def test_extract_lengths_and_phase(data):
    reads, loci = load(data["prefix"])
    assert len(loci) == N_LOCI
    assert len(reads) == N_LOCI * 28
    for _, r in reads.iterrows():
        L, hp, long_read = data["truth"][r["read"]]
        assert r["Y"] == L
        if long_read:
            assert r["n_het"] == len(SNP_OFF)
            assert (r["llr"] > 0) == (hp == 1)
            assert (r["w1"] > 0.99) == (hp == 1)
        else:
            assert r["n_het"] == 0 and np.isnan(r["w1"])
    assert set(loci["trgt_phased"]) == {1}


def test_phase_weight_tiers():
    w = phase_weight(np.array([0, 1, 4, 4]), np.array([0.0, 2.0, 20.0, -20.0]), PhaseParams())
    assert np.isnan(w[0])
    assert 0.8 < w[1] < 0.95
    assert w[2] == pytest.approx(0.995) and w[3] == pytest.approx(0.005)


def test_assign_alleles_orients_unphased_calls(data):
    df, _ = load(data["prefix"])
    df["trgt_phased"] = 0
    g = df[df["locus"] == "L1"].copy()
    g["al1"], g["al2"] = g["al2"].iloc[0], g["al1"].iloc[0]       # swapped call
    out = assign_alleles(g)
    assert out["L1"].iloc[0] == data["alleles"][1][0]


def _noise(data):
    df = training_reads([data["prefix"]])
    return EmpiricalNoise(DeltaBins(D=30, T=2000)).fit(df)


def test_empirical_noise_peaks_at_zero(data):
    m = _noise(data)
    st = 1 * 4 + 1                                       # period 3, length 50-200
    D = m.bins.D
    assert np.argmax(m.logp[st]) == D + 1
    assert np.exp(m.logp[st, D + 1]) > 0.7


def _simulate(rng, model, period, comp, n_per_hap=40, phased_frac=0.7):
    """comp: per haplotype (L0, pi, sampler of variable lengths). Noise drawn from the model's own
    period-3 / 50-200 bp histogram."""
    st = 1 * 4 + 1
    p = np.exp(model.logp[st]); p /= p.sum()
    D = model.bins.D
    Ys, w1s = [], []
    for h, (L0, pi, var) in enumerate(comp):
        for _ in range(n_per_hap):
            L = L0 if rng.random() < pi else var(rng)
            b = rng.choice(len(p), p=p)
            err = b - D - 1 if 0 < b < len(p) - 1 else 0
            Ys.append(L + err)
            w1s.append(np.nan if rng.random() > phased_frac else (0.995 if h == 0 else 0.005))
    reads = pd.DataFrame(dict(Y=np.array(Ys), period=period))
    return reads, np.array(w1s)


def _fit(model, reads, w1, period, **kw):
    grid = locus_grid(reads["Y"].to_numpy(), [], model.bins.D)
    logl = model.log_curves(reads, grid)
    rows, f, post = LocusFit(logl, grid, w1, 2, period, None, (), FitParams(**kw)).run()
    return {r["hap"]: r for r in rows}


def test_stage2_recovers_expansion_on_one_haplotype(data):
    m = _noise(data)
    rng = np.random.default_rng(11)
    comp = [(60, 0.7, lambda r: int(round(r.normal(84, 6)))),   # hap1: 30% expanded
            (69, 1.0, None)]                                     # hap2: stable
    reads, w1 = _simulate(rng, m, 3, comp, n_per_hap=60)
    r = _fit(m, reads, w1, 3)
    assert r[1]["L0"] == 60 and r[2]["L0"] == 69
    assert 0.15 < r[1]["U"] < 0.45
    assert r[2]["U"] < 0.1
    assert r[1]["p_exp"] > 0.15 and r[1]["p_con"] < 0.05
    assert r[1]["mad"] > 3 * r[2]["mad"]
    assert r[1]["d_ll"] > 10
    assert 0.1 < r[1]["pooled_U"] < 0.3
    assert abs(sum(r[1]["f_p"]) - 1) < 0.01


def test_stage2_stable_locus_stays_narrow(data):
    m = _noise(data)
    rng = np.random.default_rng(5)
    reads, w1 = _simulate(rng, m, 3, [(60, 1.0, None), (66, 1.0, None)], n_per_hap=40)
    r = _fit(m, reads, w1, 3)
    assert r[1]["U"] < 0.15 and r[2]["U"] < 0.15                     # stable floor, 40 reads/haplotype
    assert {r[1]["hap_rank"], r[2]["hap_rank"]} == {"short", "long"}


def test_stage2_bootstrap_interval(data):
    m = _noise(data)
    rng = np.random.default_rng(11)
    comp = [(60, 0.7, lambda r: int(round(r.normal(84, 6)))), (69, 1.0, None)]
    reads, w1 = _simulate(rng, m, 3, comp, n_per_hap=40)
    r = _fit(m, reads, w1, 3, bootstrap=20)
    assert r[1]["U_lo"] <= r[1]["U"] <= r[1]["U_hi"] and r[1]["U_hi"] > r[1]["U_lo"]
    assert r[2]["U_hi"] < r[1]["U_lo"] + 0.2


def test_stage2_measures_from_the_germline_anchor(data):
    """Spread mass whose mode wanders off the germline is still measured from the germline allele."""
    m = _noise(data)
    rng = np.random.default_rng(4)
    comp = [(60, 0.3, lambda r: int(r.choice([51, 69]))), (90, 1.0, None)]
    reads, w1 = _simulate(rng, m, 3, comp, n_per_hap=60)
    grid = locus_grid(reads["Y"].to_numpy(), [60, 90], m.bins.D)
    rows, _, _ = LocusFit(m.log_curves(reads, grid), grid, w1, 2, 3, None, (60, 90), FitParams()).run()
    assert rows[0]["L0"] == 60 and rows[1]["L0"] == 90 and rows[0]["U"] > 0.5


def test_stage2_short_grid_long_period(data):
    """A grid shorter than the start bump (few reads, long period) must still fit."""
    m = _noise(data)
    rng = np.random.default_rng(3)
    reads, w1 = _simulate(rng, m, 30, [(600, 1.0, None), (600, 1.0, None)], n_per_hap=5)
    grid = np.arange(590, 611)
    rows, f, _ = LocusFit(m.log_curves(reads, grid), grid, w1, 2, 30, None, (), FitParams()).run()
    assert f.shape == (2, len(grid)) and rows[0]["L0"] == 600


def test_unphased_reads_follow_length(data):
    """All reads unphased: haplotypes are separated by length alone and rho is fitted."""
    m = _noise(data)
    rng = np.random.default_rng(9)
    reads, w1 = _simulate(rng, m, 3, [(60, 1.0, None), (75, 1.0, None)], n_per_hap=30, phased_frac=0.0)
    r = _fit(m, reads, w1, 3)
    assert sorted([r[1]["L0"], r[2]["L0"]]) == [60, 75]
    assert 0.35 < r[1]["rho"] < 0.65


def test_selection_curve_decreases():
    s = selection_curve(np.arange(0, 3000, 100), "4000,5000,6000", 128)
    assert s[0] == 1.0 and np.all(np.diff(s) <= 0)


def test_cli_fit_writes_table(data, tmp_path):
    from tandemo.fit import run
    m = _noise(data)
    res, paths = run(data["prefix"], m, str(tmp_path / "fit"), "S1", FitParams(), write_curves=True,
                     log=lambda *_: None)
    assert set(res["status"]) == {"ok"}
    assert len(res) == 2 * N_LOCI
    assert res["U"].mean() < 0.1
    het = res[res["locus"] == "L1"].set_index("hap")
    assert het.loc[1, "L0"] == 60 and het.loc[2, "L0"] == 69


def test_network1_trains(data, tmp_path):
    torch = pytest.importorskip("torch")
    from tandemo.network1 import TorchNoise, train
    df = training_reads([data["prefix"]])
    model = train(df, str(tmp_path / "n1.pt"), DeltaBins(D=30, T=2000), val_chroms=(), epochs=60, batch=64,
                  lr=3e-3, log=lambda *_: None)
    m2 = TorchNoise.load(str(tmp_path / "n1.pt"))
    g = df[df["locus"] == "L0"].head(5)
    grid = np.arange(40, 80)
    lc = m2.log_curves(g, grid)
    assert lc.shape == (5, len(grid))
    assert (grid[lc.argmax(1)] == g["Y"].to_numpy()).mean() >= 0.8


def _hifi_like():
    """90% exact, 4% +-1, 0.5% +-2, rare beyond: the sharp noise a well-trained network 1 gives."""
    b = DeltaBins(D=30, T=2000)
    d = np.arange(-30, 31)
    core = np.where(d == 0, 0.9, np.where(abs(d) == 1, 0.04, np.where(abs(d) == 2, 0.005, 1e-6)))
    v = np.concatenate([[1e-5], core, [1e-5]])
    return EmpiricalNoise(b, np.tile(np.log(v / v.sum()), (16, 1)))


@pytest.mark.parametrize("extra", [[90], [300], [90, 96]])
def test_one_or_two_expanded_reads_are_captured(extra):
    """A single expanded read (near or far) must get its own mass in f: nothing may hide it."""
    m = _hifi_like()
    rng = np.random.default_rng(1)
    reads, w1 = _simulate(rng, m, 3, [(60, 1.0, None), (66, 1.0, None)], n_per_hap=40)
    reads = pd.concat([reads, pd.DataFrame(dict(Y=extra, period=3))], ignore_index=True)
    w1 = np.concatenate([w1, [0.995] * len(extra)])
    r = _fit(m, reads, w1, 3)
    assert r[1]["n_support"] == len(extra)
    assert r[1]["U"] == pytest.approx(len(extra) / (40 + len(extra)), abs=0.01)
    assert r[1]["p_exp"] == pytest.approx(r[1]["U"]) and r[1]["mean_shift"] > 0
    assert r[1]["d_ll"] > 5 and r[2]["U"] < 0.01


def test_het_indels_phase_reads_outside_repeats(tmp_path):
    """A het deletion outside any repeat phases reads; one inside a catalog repeat is ignored."""
    from tandemo.aln import Aln
    from tandemo.phasing import load_het_sites, read_evidence
    rng = np.random.default_rng(7)
    ref = "".join(rng.choice(list("ACGT"), 600))
    fa = tmp_path / "r.fa"
    fa.write_text(">chr1\n" + ref + "\n")
    pysam.faidx(str(fa))
    vcf = tmp_path / "v.vcf"
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=600>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n')
        fh.write('##FORMAT=<ID=PS,Number=1,Type=Integer,Description="ps">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
        for p in (200, 400):        # 0-based anchors; deletion of 2 bases after the anchor on hap 2
            fh.write(f"chr1\t{p + 1}\t.\t{ref[p:p + 3]}\t{ref[p]}\t50\tPASS\t.\tGT:PS\t0|1:7\n")
    pysam.tabix_index(str(vcf), preset="vcf", force=True)
    sites = load_het_sites(str(vcf) + ".gz", "chr1", fasta=str(fa), exclude=[(395, 420)])
    assert len(sites) == 1 and sites.indel == [True]
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 600}]})
    for hp in (1, 2):
        a = pysam.AlignedSegment(header)
        a.query_name, a.reference_id, a.reference_start, a.mapping_quality = f"h{hp}", 0, 50, 60
        if hp == 1:
            a.query_sequence, a.cigartuples = ref[50:550], [(0, 500)]
        else:
            a.query_sequence = ref[50:201] + ref[203:550]
            a.cigartuples = [(0, 151), (2, 2), (0, 347)]
        a.query_qualities = pysam.qualitystring_to_array("I" * len(a.query_sequence))
        n, llr, ni, n1, n2 = read_evidence(Aln(a), a.query_sequence, np.array(a.query_qualities), sites)[7]
        assert n == 1 and ni == 1 and (n1, n2) == ((1, 0) if hp == 1 else (0, 1))
        assert (llr > 0) == (hp == 1) and abs(llr) > 2


def test_catalog_filter_keeps_disease_loci():
    from tandemo.catalog import Locus, filter_loci
    loci = [Locus("short", "chr1", 100, 120, ("CA",)), Locus("hp", "chr1", 200, 300, ("A",)),
            Locus("sd", "chr1", 1000, 1100, ("CAG",)), Locus("ok", "chr1", 2000, 2100, ("CAG",)),
            Locus("HD_HTT_pathogenic", "chr1", 3000, 3010, ("CAG",)), Locus("gene", "chr1", 4000, 4010, ("A",))]
    out = filter_loci(loci, 50, True, {"chr1": [(1050, 1060)]}, {"chr1": [(4005, 4006)]})
    assert [L.id for L in out] == ["ok", "HD_HTT_pathogenic", "gene"]


def test_tile_motifs_and_interruptions():
    from tandemo.pileup import tile
    t = tile("CAGCAGCAACAGCCGCCG", ("CAG", "CCG"))
    assert [(n, k, mm) for _, n, k, mm in t] == [(3, 0, 0), (3, 0, 0), (3, -1, 0), (3, 0, 0), (3, 1, 0), (3, 1, 0)]
    assert sum(x[1] for x in t) == 18
    assert tile("AGCAGC", ("CAG",))[0] == (0, 3, 0, 0)       # rotations count as the motif
    # short units tile exactly only; 10-14 bp within 1 mismatch, 15-19 bp within 2, >= 20 bp within 3
    t = tile("AAAAG" + "AACAG", ("AAAAG",))
    assert t[0][1:] == (5, 0, 0) and t[1][2] == -1
    m10 = "AAAAGAAAGG"
    t = tile(m10 + "AAACGAAAGG" + "AAACGAATGG", (m10,))
    assert [(n, k, mm) for _, n, k, mm in t][:2] == [(10, 0, 0), (10, 0, 1)] and t[2][2] == -1
    m20 = "ACGTTGCAAGTCCATGGACT"
    var3 = "ACGATGCAAGTCCTTGGAGT"                                  # three substitutions
    assert tile(m20 + var3, (m20,))[1][1:] == (20, 0, 3)


def test_align_motifs_rotates_to_line_up():
    from tandemo.pileup import align_motifs
    assert align_motifs(["CCCTTTCCTACACCAA", "CCTACACCACCCTTT"]) == ["CCCTTTCCTACACCAA", "CCCTTTCCTACACCA-"]


def test_pileup_svgs_split_haplotypes(data, tmp_path):
    from tandemo.pileup import run
    done = run(data["bam"], data["fa"], data["cat"], str(tmp_path / "pu"), data["vcf"], data["trgt"],
               loci_names=["L1"], view=3000, log=lambda *_: None)
    assert len(done) == 1 and done[0]["per"][1]["n"] == 10 and done[0]["per"][2]["n"] == 10
    assert done[0]["per"][0]["n"] == 8                          # short reads carry no het SNVs
    svg = (tmp_path / "pu" / "L1.svg").read_text()
    assert svg.startswith("<svg") and "haplotype 2" in svg and (tmp_path / "pu" / "index.html").exists()


def test_discover_motifs_adds_units_the_catalog_misses():
    from tandemo.pileup import discover_motifs
    m = discover_motifs(["AAAGG" * 40, "AAAAG" * 12], ("AAGGG", "ACAGG"))
    assert m[:2] == ("AAGGG", "ACAGG") and "AAAGG" in m and "AAAAG" in m


def test_report_writes_figures(data, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from tandemo.network1 import train
    from tandemo.report import run
    df = training_reads([data["prefix"]])
    train(df, str(tmp_path / "n1.pt"), DeltaBins(D=30, T=2000), val_chroms=(), epochs=3, batch=64, log=lambda *_: None)
    EmpiricalNoise(DeltaBins(D=30, T=2000)).fit(df).save(str(tmp_path / "b.npz"))
    path = run(str(tmp_path / "n1.pt"), [data["prefix"]], str(tmp_path / "rep"), str(tmp_path / "b.npz"),
               log=lambda *_: None)
    for f in ("curves", "summary", "features", "importance", "expansion"):
        assert (tmp_path / "rep" / f"{f}.svg").exists()
    assert "Network 1 report" in open(path).read()


def test_stratified_training_sample(data):
    df_all = training_reads([data["prefix"]])
    df = training_reads([data["prefix"]], per_stratum=3)
    assert df["locus"].nunique() == 3 and len(df) == len(df_all[df_all["locus"].isin(df["locus"].unique())])


def test_discover_motifs_ignores_single_copies():
    from tandemo.pileup import discover_motifs
    assert discover_motifs(["GGCCCC" * 2 + "CCCGCGACGCCCACCCCG" + "GGCCCC" * 3] * 10, ("GGCCCC",)) == ("GGCCCC",)
    assert discover_motifs(["TTTTA" * 10 + "ATGTT" * 6 + "TTTTA" * 3] * 10, ("TTTTA", "TTTCA")) == \
        ("TTTTA", "TTTCA", "ATGTT")


def test_pad_indels_put_trgt_alleles_on_the_read_scale(tmp_path):
    """Y counts germline indels in the 10 bp pads, TRGT's allele does not: they are added per haplotype."""
    from tandemo.catalog import Locus
    from tandemo.phasing import PAD_AMBIGUOUS, PAD_OK, PAD_UNPHASED, pad_indels
    rng = np.random.default_rng(2)
    ref = "".join(rng.choice(list("ACGT"), 400))
    vcf = tmp_path / "p.vcf"
    recs = [(134, ref[134], ref[134] + "T", "1|1", 7),          # hom insertion before 135: right pad of A
            (91, ref[91:94], ref[91], "0|1", 7),                # het deletion of [92, 94): left pad of A, hap 2
            (226, ref[226:232], ref[226], "1|0", 7),            # deletion [227, 232) across B's start (230)
            (291, ref[291:293], ref[291], "0|1", 9)]            # het deletion in C's pad, other phase block
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=400>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n')
        fh.write('##FORMAT=<ID=PS,Number=1,Type=Integer,Description="ps">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
        for p, r, a, gt, ps in sorted(recs):
            fh.write(f"chr1\t{p + 1}\t.\t{r}\t{a}\t50\tPASS\t.\tGT:PS\t{gt}:{ps}\n")
    pysam.tabix_index(str(vcf), preset="vcf", force=True)
    loci = [Locus("A", "chr1", 100, 130, ("CA",)), Locus("B", "chr1", 230, 260, ("CA",)),
            Locus("C", "chr1", 300, 330, ("CA",)), Locus("D", "chr1", 360, 380, ("CA",))]
    out = pad_indels(str(vcf) + ".gz", "chr1", loci, {"A": 7, "B": 7, "C": 7, "D": 7}, 10)
    assert out["A"] == (1, -1, PAD_OK)
    assert out["B"][2] == PAD_AMBIGUOUS
    assert out["C"] == (0, -1, PAD_UNPHASED)
    assert "D" not in out


def test_report_mixture_predictive_shifts_the_other_allele():
    """A het read at alleles 60 / 63 with w1 = 0.8 and an exact-length network: mass 0.8 at Delta 0 and
    0.2 at Delta +3 (the other allele, seen from L_ref = 60); a homozygous read keeps all mass at 0."""
    pytest.importorskip("matplotlib")
    from tandemo.report import predictive

    class Stub:
        D = 5
        delta = np.arange(-5, 6)

        class m:
            bins = DeltaBins(D=5, T=100)

        def probs(self, X, L, ref_len, period):
            p = np.zeros((len(X), 1, 2 * 5 + 3)); p[:, :, 5 + 1] = 1.0      # all mass at Delta = 0
            return p

        def delta_probs(self, X, L, ref_len, period, deltas):
            return (np.asarray(deltas) == 0).astype(float)                    # the same, at any Delta

    sub = pd.DataFrame(dict(L1=[60, 60], L2=[63, 60], w1_train=[0.8, 0.5], Y=[60, 61], ref_len=[60, 60],
                            period=[3, 3]))
    P, delta, Lr, _ = predictive(Stub(), sub, np.zeros((2, 16)))
    assert P[0, 5] == pytest.approx(0.8) and P[0, 5 + 3] == pytest.approx(0.2)
    assert P[1, 5] == pytest.approx(1.0) and list(delta) == [0, 1] and list(Lr) == [60, 60]


def test_pareto_tail_is_a_proper_distribution():
    """The discretised generalised Pareto sums to 1 over k >= 1 and decays with distance; a full network
    output (core + both tails) sums to 1 over Delta."""
    torch = pytest.importorskip("torch")
    from tandemo.network1 import Net1, gpd_logpmf
    k = torch.arange(1, 200_001, dtype=torch.float64)
    for sigma, xi in ((3.0, 0.3), (20.0, 0.05), (1.0, 1e-3)):
        p = gpd_logpmf(k, torch.tensor(sigma, dtype=torch.float64), torch.tensor(xi, dtype=torch.float64)).exp()
        assert abs(p.sum().item() - 1) < 2e-3 and bool((p[1:] <= p[:-1]).all())
    bins = DeltaBins(D=10, T=2000)
    net = Net1(16, bins.n, 16, tail="pareto")
    torch.manual_seed(0)
    lp, tp = net.dist(torch.randn(3, 16), torch.randn(3, 1, 3))
    d = torch.arange(-20_000, 20_001)[None, None, :].expand(3, 1, -1)
    tot = net.delta_logp(lp, tp, d, bins).exp().sum(-1)
    assert torch.allclose(tot, torch.ones_like(tot), atol=2e-2)


def test_uniform_tail_models_still_load(data, tmp_path):
    """Models saved before the Pareto tails keep their uniform tails."""
    pytest.importorskip("torch")
    from tandemo.network1 import TorchNoise, train
    df = training_reads([data["prefix"]])
    train(df, str(tmp_path / "u.pt"), DeltaBins(D=30, T=2000), val_chroms=(), epochs=2, batch=64, tail="uniform",
          log=lambda *_: None)
    m = TorchNoise.load(str(tmp_path / "u.pt"))
    assert m.net.tail == "uniform"
    lc = m.log_curves(df.head(3), np.array([40, 60, 1000]))
    assert lc.shape == (3, 3) and np.isfinite(lc).all()


def test_confident_phase_counts_support_and_conflict(data):
    from tandemo.phasing import confident_phase
    p = PhaseParams()                                       # >= 3 supporting, < 1/5 of the sites conflicting
    assert list(confident_phase([3, 5, 4, 3, 2, 0, 1, 1, 4], [0, 1, 1, 2, 0, 3, 5, 4, 4], p)) == \
        [1, 1, 0, 0, 0, 2, 2, 0, 0]                          # 5:1 confident, 4:1 not (exactly 1/5), 2:0 too few
    reads, _ = load(data["prefix"])
    long_reads = reads[reads["n_het"] > 0]
    assert (long_reads["n_hap1"] + long_reads["n_hap2"] == long_reads["n_het"]).all()
    assert set(confident_phase(long_reads["n_hap1"], long_reads["n_hap2"])) == {1, 2}


def test_methylation_features_are_means_and_optional():
    from tandemo.extract import meth_features
    from tandemo.features import METH_NAMES, X_NAMES, raw_x
    pos = np.array([5, 20, 60, 70, 150, 300])
    pr = np.array([0.9, 0.1, 0.95, 0.5, 0.85, 0.99])
    o = dict(q_left=(0, 50), q_rep=(50, 100), q_right=(100, 200))
    mf, mr, mc = meth_features((pos, pr), o)
    assert mf == pytest.approx((0.9 + 0.1 + 0.85) / 3) and mr == pytest.approx((0.95 + 0.5) / 2)
    assert mc == pytest.approx(4 / 5)                              # 0.5 is not decisive
    assert all(np.isnan(meth_features(None, o)))
    df = pd.DataFrame(dict(rq=[0.999] * 2, np=[10, 10], read_len=[15000] * 2, mapq=[60] * 2, n_sa=[0] * 2,
                           bq_rep=[30] * 2, bq_flank=[35] * 2, dist_end=[1000] * 2, flank_id_min=[1.0] * 2,
                           period=[3] * 2, ref_len=[60] * 2, gc=[0.5] * 2, hp_frac=[0] * 2, max_hp=[2] * 2,
                           cx3=[0.5] * 2, flank_cx=[0.8] * 2, mC_flank=[0.8, np.nan], mC_rep=[np.nan, np.nan],
                           mC_cert=[0.9, np.nan]))
    X = raw_x(df)
    assert X[:, X_NAMES.index("mC_avail")].tolist() == [1.0, 0.0]
    old = df.drop(columns=["mC_flank", "mC_rep", "mC_cert"])      # an extraction from before methylation
    Xo = raw_x(old)
    assert np.isnan(Xo[:, [X_NAMES.index(m) for m in METH_NAMES if m != "mC_avail"]]).all()
    assert (Xo[:, X_NAMES.index("mC_avail")] == 0).all()


def test_models_with_and_without_methylation(data, tmp_path):
    pytest.importorskip("torch")
    from tandemo.features import X_NAMES, X_NAMES_NO_METH
    from tandemo.network1 import TorchNoise, train
    df = training_reads([data["prefix"]])
    for meth, names in ((True, X_NAMES), (False, X_NAMES_NO_METH)):
        path = str(tmp_path / f"m{int(meth)}.pt")
        train(df, path, DeltaBins(D=5), val_chroms=(), epochs=2, batch=64, methylation=meth, log=lambda *_: None)
        m = TorchNoise.load(path)
        assert m.features == names
        lc = m.log_curves(df.drop(columns=[c for c in ("mC_flank", "mC_rep", "mC_cert") if c in df]).head(3),
                          np.array([50, 60, 70]))                  # reads without 5mC columns still score
        assert lc.shape == (3, 3) and np.isfinite(lc).all()


def test_realign_puts_a_contraction_in_one_gap():
    pytest.importorskip("parasail")
    from tandemo.pileup import realign
    rng = np.random.default_rng(4)
    L, R = "".join(rng.choice(list("ACGT"), 128)), "".join(rng.choice(list("ACGT"), 128))
    unit = "CCCTTTCCTACACCAA"
    target, read = L + unit * 40 + R, L + unit * 32 + R          # 8 units (128 bp) shorter
    base, ins, _ = realign(read, target)
    assert not ins and (base < 0).sum() == 8 * len(unit)           # one deletion, no insertions
    gaps = np.nonzero(base < 0)[0]
    assert gaps.max() - gaps.min() + 1 == len(gaps)               # contiguous


def test_pileup_aligns_reads_to_their_trgt_allele(data, tmp_path, monkeypatch):
    pytest.importorskip("parasail")
    import tandemo.pileup as pu
    real = pu.read_trgt
    def with_seqs(vcf, sample=None, ids=None):                    # the synthetic VCF has symbolic alleles
        out = real(vcf, sample, ids)
        for t in out.values():
            t["seq"] = tuple("CAG" * (n // 3) for n in t["al"])
        return out
    monkeypatch.setattr(pu, "read_trgt", with_seqs)
    reads, _ = load(data["prefix"])
    one = reads[(reads.locus == "L1") & (reads.Y == 69)].read.iloc[0]
    done = pu.run(data["bam"], data["fa"], data["cat"], str(tmp_path / "pu"), data["vcf"], data["trgt"],
                  loci_names=["L1"], view=3000, highlight={"L1": {one}}, log=lambda *_: None)
    svg = (tmp_path / "pu" / "L1.svg").read_text()
    assert done and "Germline allele (TRGT)</text>" in svg and ">60 bp</text>" in svg and ">69 bp</text>" in svg
    assert svg.count(f'fill="{pu.OUTLIER}" stroke="#0b0b0b"') == 2   # the read's triangle + the legend's


def test_tile_tolerates_one_homopolymer_base_in_long_units():
    from tandemo.pileup import tile
    seq = "AGAAAAA" + "AGAAAA" + "AGAAAAA" + "AGAAAAA"                  # one unit with a short A run
    t = tile(seq, ("AGAAAAA",))
    assert all(k == 0 for _, _, k, _ in t) and sum(n for _, n, _, _ in t) == len(seq)
    assert any(mm == 1 for _, _, _, mm in t)                            # the variant unit, drawn lighter
    assert tile("CAGCAACAG", ("CAG",))[1][2] == -1                      # short units stay exact


def test_trgt_al_follows_phased_gt():
    # HiPhase rewrites GT (0/1 -> 1|0) but keeps AL in allele-index order: AL 439,221 with GT 1|0 means
    # haplotype 1 carries allele 1 = 221 bp (NA19909 chr4_18735494)
    from tandemo.trgt import gt_order
    assert gt_order((1, 0), (439, 221)) == (221, 439)
    assert gt_order((0, 1), (439, 221)) == (439, 221)
    assert gt_order((2, 1), (404, 439)) == (439, 404)
    assert gt_order((1, 1), (50, 50)) == (50, 50)
    assert gt_order((1,), (30,)) == (30,)


def test_trgt_read_check_corrects_double_counted_pad_indels_and_drops_unsupported_alleles():
    from tandemo.data import trgt_read_check, CORRECTED, EXCLUDED
    rows = []
    # locus, L1 = AL1 + d1, d1, L2, d2, read median hap 1, hap 2
    for locus, L1, d1, L2, d2, y1, y2 in (("ok", 40, 0, 44, 0, 40, 44),
                                           ("double", 116, 8, 100, 0, 108, 100),    # TRGT 108 already has the 8 bp
                                           ("small_gap", 30, 0, 20, 0, 28, 20),     # 2 bp off: kept (limit 3 bp)
                                           ("unsupported", 3155, 0, 53, 0, 53, 53),  # TRGT allele no read shows
                                           ("pad_ok", 108, 8, 100, 0, 108, 100)):   # reads at AL + d: no change
        for w, y in ((1.0, y1), (0.0, y2)):
            rows += [dict(locus=locus, L1=L1, L2=L2, pad_d1=d1, pad_d2=d2, w1=w, Y=y)] * 5
    out = trgt_read_check(pd.DataFrame(rows))
    first = out.groupby("locus")[["L1", "label_ok"]].first()
    assert first.loc["double", "L1"] == 108 and first.loc["pad_ok", "L1"] == 108   # reads are never changed
    assert first["label_ok"].to_dict() == {"double": True, "ok": True, "pad_ok": True, "small_gap": True,
                                           "unsupported": False}
    rec = out.attrs["trgt_read"].set_index("locus")
    assert set(rec.index) == {"double", "unsupported"}
    assert rec.loc["double", "action"] == CORRECTED and rec.loc["unsupported", "action"] == EXCLUDED


def test_training_reads_from_several_extractions(data):
    one = training_reads([data["prefix"]])
    two = training_reads([data["prefix"], data["prefix"]])          # e.g. one extraction per chromosome
    assert len(two) == 2 * len(one) and not two.attrs


def test_pileup_realignment_puts_repeat_indels_at_the_3prime_end():
    pytest.importorskip("parasail")
    from tandemo.pileup import realign
    target = "GGATCC" + "CAG" * 4 + "TTGACA"                 # 6 bp flank, 4 units, 6 bp flank
    base, ins, _ = realign("GGATCC" + "CAG" * 5 + "TTGACA", target)      # one unit gained
    assert ins == {18: [18, 19, 20]} and base[17] == 17 and base[18] == 21
    base, ins, _ = realign("GGATCC" + "CAG" * 3 + "TTGACA", target)      # one unit lost
    assert not ins and list(base[15:18]) == [-1, -1, -1] and base[14] == 14 and base[18] == 15


def test_trgt_read_check_low_coverage_homozygous_double_count():
    # PRDM12 in NA19909: TRGT 38 / 38 bp already holds a homozygous 12 bp pad insertion; 3 + 4 phased reads + 1
    from tandemo.data import trgt_read_check, CORRECTED
    rows = [dict(locus="prdm12", L1=50, L2=50, pad_d1=12, pad_d2=12, w1=w, Y=y)
            for w, y in [(1.0, 37), (1.0, 37), (1.0, 38), (0.0, 36), (0.0, 38), (0.0, 38), (0.0, 39), (0.5, 39)]]
    out = trgt_read_check(pd.DataFrame(rows))
    assert set(out["L1"]) == {38} and set(out["L2"]) == {38} and out["label_ok"].all()
    assert set(out.attrs["trgt_read"]["action"]) == {CORRECTED}


def test_trgt_read_check_vote_needs_75_percent_of_at_least_3_reads():
    from tandemo.data import trgt_read_check
    def run(ys):                                   # one haplotype: label 116 = TRGT 108 + 8 bp pad insertion
        rows = [dict(locus="x", L1=116, L2=100, pad_d1=8, pad_d2=0, w1=1.0, Y=y) for y in ys]
        rows += [dict(locus="x", L1=116, L2=100, pad_d1=8, pad_d2=0, w1=0.0, Y=100)] * 5
        return trgt_read_check(pd.DataFrame(rows))["L1"].iloc[0]
    assert run([108, 108]) == 116                  # 2 reads: below the floor, default label kept
    assert run([108, 108, 108]) == 108             # 3 of 3 for the TRGT allele
    assert run([108, 108, 108, 116]) == 108        # 75 %
    assert run([108, 108, 116, 116]) == 116        # 50 %: kept


def test_segdup_overlap_and_feature(tmp_path):
    from tandemo import segdup
    from tandemo.features import X_NAMES
    bed = tmp_path / "segdups.bed"
    bed.write_text("chr1\t100\t200\nchr1\t180\t260\nchr1\t1000\t1100\n")
    assert segdup.overlap(str(bed), "chr1", 150, 250) == 1.0
    assert segdup.overlap(str(bed), "chr1", 250, 350) == 0.1
    assert segdup.overlap(str(bed), "chr2", 0, 10) == 0.0
    assert segdup.intervals(str(bed), "chr1", 240, 1050) == [(240, 260), (1000, 1050)]
    assert segdup.tag(0.5) == "segdup 50%" and segdup.tag(0.0) == "" and segdup.tag(float("nan")) == ""
    assert "segdup" in X_NAMES and "has_sa" in X_NAMES      # inputs of the next network 1 training


def test_element_features_replace_cx3():
    from tandemo.elements import motif_features, purity
    from tandemo.features import X_NAMES, ALL_FEATURES, FIRST_MODEL_NAMES
    gc, ent, n = motif_features("CAG")
    assert abs(gc - 2 / 3) < 1e-9 and abs(ent - 1.585) < 1e-3 and n == 1
    gc, ent, n = motif_features("AT,TA")
    assert gc == 0 and ent == 1.0 and n == 2
    assert purity("CAG" * 10, 3) == 1.0
    assert 0.8 < purity("CAG" * 5 + "CAT" + "CAG" * 4, 3) < 1.0          # one interruption
    assert purity("CA", 3) != purity("CA", 3)                              # NaN: shorter than two periods
    assert "cx3" not in X_NAMES and "cx3" in ALL_FEATURES and "cx3" in FIRST_MODEL_NAMES
    assert {"elem_gc", "elem_entropy", "purity", "n_motifs_log", "segdup"} <= set(X_NAMES)


def test_untrusted_trgt_drop_or_local_allele():
    from tandemo.data import trgt_read_check, EXCLUDED, LOCAL
    # haplotype 2: TRGT 409 bp, 10 reads at 412-414 bp (median 413) + 4 at 425-429 (a second group)
    ys = [412, 413, 413, 413, 413, 414, 413, 412, 413, 414, 425, 427, 427, 429]
    rows = [dict(locus="x", L1=395, L2=409, pad_d1=0, pad_d2=0, w1=0.0, Y=y) for y in ys]
    rows += [dict(locus="x", L1=395, L2=409, pad_d1=0, pad_d2=0, w1=1.0, Y=395)] * 8
    df = pd.DataFrame(rows).astype({"L1": "int32", "L2": "int32", "Y": "int32"})   # the extraction's dtypes
    drop = trgt_read_check(df.copy(), untrusted="drop")
    assert not drop["label_ok"].any() and set(drop.attrs["trgt_read"]["action"]) == {EXCLUDED}
    loc = trgt_read_check(df.copy(), untrusted="local")
    assert loc["label_ok"].all() and set(loc["L2"]) == {413} and set(loc["L1"]) == {395}
    assert set(loc.attrs["trgt_read"]["action"]) == {LOCAL}
    # a second group pulls the median between the groups: the densest window still finds the main one
    two = df.copy()
    two.loc[(two.w1 == 0.0) & (two.Y >= 425), "Y"] = np.array([425, 426, 427, 427], dtype="int32")
    two = pd.concat([two, pd.DataFrame([dict(locus="x", L1=395, L2=409, pad_d1=0, pad_d2=0, w1=0.0, Y=y)
                                        for y in (426, 427, 428, 426, 427)])], ignore_index=True)
    two = trgt_read_check(two, untrusted="local")
    assert set(two["L2"]) == {413}                       # 10 of 19 reads at 412-414, not the 9 at 425-428
    few = pd.DataFrame([dict(locus="x", L1=395, L2=409, pad_d1=0, pad_d2=0, w1=0.0, Y=y) for y in (412, 416, 420, 425, 429)]
                       + [dict(locus="x", L1=395, L2=409, pad_d1=0, pad_d2=0, w1=1.0, Y=395)] * 5)
    assert not trgt_read_check(few, untrusted="local")["label_ok"].any()   # no window with >= 3 reads: dropped


def test_measurement_pad_follows_the_longest_element():
    from tandemo.extract import Params
    p = Params()
    assert p.pad_for(("CAG",)) == 10                       # STRs keep 10 bp
    assert p.pad_for(("TAGTATTAAGAGGTGA", "AT")) == 16      # one VNTR element
    assert p.pad_for(("A" * 1458,)) == 100                  # capped
