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
    return dict(prefix=prefix, truth=truth, alleles=alleles)


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


def _fit(model, reads, w1, period, shape):
    grid = locus_grid(reads["Y"].to_numpy(), [], model.bins.D)
    logl = model.log_curves(reads, grid)
    rows, f, post = LocusFit(logl, grid, w1, 2, period, None, (), FitParams(shape=shape)).run()
    return {r["hap"]: r for r in rows}


@pytest.mark.parametrize("shape", ["parametric", "nonparametric"])
def test_stage2_recovers_expansion_on_one_haplotype(data, shape):
    m = _noise(data)
    rng = np.random.default_rng(11)
    comp = [(60, 0.7, lambda r: int(round(r.normal(84, 6)))),   # hap1: 30% expanded
            (69, 1.0, None)]                                     # hap2: stable
    reads, w1 = _simulate(rng, m, 3, comp, n_per_hap=60)
    r = _fit(m, reads, w1, 3, shape)
    assert r[1]["L0"] == 60 and r[2]["L0"] == 69
    assert 0.15 < r[1]["var_weight"] < 0.45
    assert r[2]["var_weight"] < 0.1
    assert r[1]["p_exp"] > 0.15 and r[1]["p_con"] < 0.05
    assert r[1]["d_ll"] > 10
    assert 0.1 < r[1]["pooled_var_weight"] < 0.3


def test_stage2_stable_locus_stays_point_mass(data):
    m = _noise(data)
    rng = np.random.default_rng(5)
    reads, w1 = _simulate(rng, m, 3, [(60, 1.0, None), (66, 1.0, None)], n_per_hap=40)
    r = _fit(m, reads, w1, 3, "parametric")
    assert r[1]["var_weight"] < 0.15 and r[2]["var_weight"] < 0.15   # stable floor, 40 reads/haplotype
    assert {r[1]["hap_rank"], r[2]["hap_rank"]} == {"short", "long"}


def test_unphased_reads_follow_length(data):
    """All reads unphased: haplotypes are separated by length alone and rho is fitted."""
    m = _noise(data)
    rng = np.random.default_rng(9)
    reads, w1 = _simulate(rng, m, 3, [(60, 1.0, None), (75, 1.0, None)], n_per_hap=30, phased_frac=0.0)
    r = _fit(m, reads, w1, 3, "parametric")
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
    assert res["var_weight"].mean() < 0.1
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


@pytest.mark.parametrize("shape", ["parametric", "nonparametric"])
@pytest.mark.parametrize("extra", [[90], [300], [90, 96]])
def test_one_or_two_expanded_reads_are_captured(shape, extra):
    """A single expanded read (near or far) must reach g: no regularisation or start may hide it."""
    m = _hifi_like()
    rng = np.random.default_rng(1)
    reads, w1 = _simulate(rng, m, 3, [(60, 1.0, None), (66, 1.0, None)], n_per_hap=40)
    reads = pd.concat([reads, pd.DataFrame(dict(Y=extra, period=3))], ignore_index=True)
    w1 = np.concatenate([w1, [0.995] * len(extra)])
    r = _fit(m, reads, w1, 3, shape)
    assert r[1]["n_support"] == len(extra)
    assert r[1]["var_weight"] == pytest.approx(len(extra) / 41 if len(extra) == 1 else len(extra) / 42, abs=0.01)
    assert r[1]["d_ll"] > 5 and r[2]["var_weight"] < 0.01
