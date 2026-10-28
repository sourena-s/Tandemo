# tandemo

Distributions of **true** tandem-repeat lengths per locus and haplotype from PacBio HiFi reads.

Observed read lengths at a repeat mix sequencing noise with real variation between molecules. tandemo separates
the two:

- **Network 1** (PyTorch) learns the read-generation likelihood p(Y | L, x) on a sample without repeat instability:
  how a molecule of true length L turns into an observed repeat length Y in a read with features x. TRGT
  supplies the germline lengths, and HiPhase the phase evidence.
- **Stage 2** estimates, per locus and haplotype of each subject, the distribution of true lengths
  f_h(L) = π δ(L − L0) + (1 − π) g(L). It sums over each read's unknown length and assigns reads to haplotypes
  softly, using het-SNV evidence along the whole read. g is a skew-normal by default, or free with
  `--shape nonparametric`.

The method, with read schematics, the statistical choices and the feature tables, is in
[docs/tandemo.pdf](docs/tandemo.pdf).

## Install

```
pip install -e ".[torch,test]"      # torch is only needed to train or apply network 1
```

## Usage

```
# per sample, after alignment (minimap2 with pbmm2 HiFi settings, e.g. somatic_sv/scripts/align_hifi.sh),
# HiPhase (phased small variants; optionally also phasing the TRGT VCF) and TRGT
tandemo extract --bam s.bam --reference ref.fa --catalog catalog.trgt.bed.gz \
    --phased-vcf s.hiphase.vcf.gz --trgt-vcf s.trgt.vcf.gz --out ex/s --threads 32

# once, on one or more stable samples
tandemo train --input ex/stable1 ex/stable2 --out net1.pt
tandemo train --input ex/stable1 --out noise.npz --empirical     # numpy baseline, no read covariates

# per subject
tandemo fit --input ex/s --model net1.pt --out fit/s --threads 32 [--shape nonparametric] [--write-curves]
```

## What each step does

**extract**: keeps primary reads with MAPQ ≥ 20 and `rq` ≥ 0.99 that cover 128 good flank bases on each side of
the repeat (identity ≥ 0.95, no clipping). Y is every read base between two anchors in the unique flanks, so
where minimap2 placed indels inside the repeat does not matter. Phase evidence is a log-likelihood ratio over every
phased het SNV on the read, within the locus's phase block. Truncated reads (one flank) are not used yet.

**train**: the loss is the negative log of a mixture over the two TRGT alleles,
−log[w1 p(Y | L1, x) + (1 − w1) p(Y | L2, x)], so reads are never hard-assigned. The output is a softmax over bins
of Y − L: one bin per bp for |Y − L| ≤ 50, plus two tail bins to 5000 bp. Features: 9 read features (rq, passes,
read length, MAPQ, supplementary flag, base quality over the repeat and the flanks, distance to the read end, flank
error), 7 locus features (period, reference length, GC, homopolymer content, complexity) and 3 features of the
candidate L.

**fit**: computes each read's curve ℓ_i(L) on an integer grid, then fits per locus:
- M0, one length per haplotype;
- M1, a point mass plus g, with g kept at least max(2 bp, one motif unit) away from L0.

Phase weights: confident reads 0.995/0.005, partly informative reads σ(LLR), reads without het SNVs the fitted
share ρ. A spanning-selection term corrects for long molecules being spanned less often.

## Output: `<out>.lengths.parquet`

One row per subject × locus × haplotype:

| column | meaning |
|---|---|
| `L0`, `pi`, `var_weight` | germline length, stable fraction, 1 − π |
| `g_xi`, `g_omega`, `g_alpha`, `g_mean`, `g_sd` | the variable component |
| `p_exp`, `p_con` | P(L ≥ L0 + Δ), P(L ≤ L0 − Δ), Δ = max(3, period) |
| `mean_abs_dev` | E\|L − L0\| |
| `n_eff`, `n_phased_eff`, `n_var_eff`, `n_support` | reads received; with het evidence; expected from g; favouring g |
| `unphased_var_frac` | share of the variable part carried by reads without het evidence |
| `pooled_var_weight`, `pooled_p_exp`, `pooled_p_con` | locus level, unaffected by leakage between haplotypes |
| `ll_m0`, `ll_m1`, `d_ll` | log-likelihoods and gain (no p-values) |
| `hap`, `hap_rank` | HiPhase label; `short` / `long` / `tie` / `haploid` by L0 (use this across subjects) |

`--write-curves` adds `<out>.curves.parquet`: every read's log-likelihood curve near its peak, with its phase
weight. That is enough to refit Stage 2, or a group-level model, without network 1. Without pyarrow, tables are
written as gzip TSV.

Read instability as an excess over held-out stable samples at matched depth and locus class. How much one or two
deviant reads count depends on how sharp network 1's tails are (see the PDF, section 6).

## Tests

`pytest` builds a synthetic reference, phased VCF, TRGT VCF and BAM, and checks:
- extraction and phase weights;
- recovery of a 30% expansion on one haplotype;
- capture of one or two expanded reads;
- separation of unphased haplotypes;
- the CLI end to end;
- network 1 training (when torch is installed).

The PDF's figure data and feature table are regenerated with `python docs/make_figdata.py` and
`python docs/make_tables.py`.
