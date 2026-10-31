# tandemo

Distributions of **true** tandem-repeat lengths per locus and haplotype from PacBio HiFi reads.

Observed read lengths at a repeat mix sequencing noise with real variation between molecules. tandemo separates
the two:

- **Network 1** (PyTorch) learns the read-generation likelihood p(Y | L, x) on a sample without repeat instability:
  how a molecule of true length L turns into an observed repeat length Y in a read with features x. TRGT
  supplies the germline lengths, and HiPhase the phase evidence.
- **Stage 2** estimates, per locus and haplotype of each subject, the distribution of true lengths f_h(L) as a
  free distribution: the nonparametric maximum-likelihood estimate (NPMLE) of Σ_i log Σ_L l_i(L) f(L), by EM
  (concave, so a global maximum). It assigns reads to haplotypes softly, using het-SNV evidence along the whole
  read. The NPMLE has a few atoms and is not smoothed (smoothing would inflate the spread).

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
tandemo fit --input ex/s --model net1.pt --out fit/s --threads 32 [--bootstrap 100] [--write-curves]

# merge per-chromosome results
tandemo combine --inputs fit/chr*.lengths.parquet --out fit/s.lengths.parquet

# reference genome by name instead of paths (tandemo genomes lists them): t2t = T2T-CHM13v2.0 + Platinum TRs,
# grch38 = the PacBio analysis set + Adotto; --catalog may name another catalog of the genome
tandemo extract --genome t2t --bam s.bam --phased-vcf s.hiphase.vcf.gz --trgt-vcf s.trgt.vcf.gz --out ex/s \
    --min-len 50 --no-homopolymer --no-segdup       # catalog filters; disease loci always kept

# genome-wide training: sample at most N loci per period x length class (bounds memory)
tandemo train --input ex/s.chr* --out net1.pt --per-stratum 12000

# figures of how network 1 behaves, and SVG pileups of the disease repeats split by haplotype
tandemo report --model net1.pt --baseline noise.npz --input ex/s.chr* --per-stratum 12000 --out report
tandemo pileup --genome t2t --bam s.bam --phased-vcf s.hiphase.vcf.gz --trgt-vcf trgt/*.vcf.gz --out pileups
```

On Snellius, `slurm/` holds the job scripts (`trgt_t2t_genome.sh`, `genome_t2t.sh`, the chr4 pilot) with the
allocation and SBU estimate of each step, and `scripts/t2t_pathogenic_catalog.sh` builds the T2T disease catalog.

## What each step does

**extract**: keeps primary reads with MAPQ ≥ 20 and `rq` ≥ 0.99 that cover 128 good flank bases on each side of
the repeat (identity ≥ 0.95, no clipping). Y is every read base between two anchors in the unique flanks, so
where minimap2 placed indels inside the repeat does not matter. Phase evidence is a log-likelihood ratio over every
phased het SNV and het indel on the read, within the locus's phase block. An indel counts when the read's sequence
around it matches one haplotype exactly (weaker in homopolymers); indels inside any catalog repeat are not used, so
phase never comes from repeat length (`--no-indel-phasing` for SNVs only). Truncated reads (one flank) are not used
yet. Catalog filters (`--min-len`, `--no-homopolymer`, `--no-segdup`) keep the genome's pathogenic loci.

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
| `L0` | germline length: the TRGT allele nearest the mode of f (the mode without TRGT); d = L − L0, Δ = max(3, period) |
| `U` | P(\|d\| ≥ Δ): the unstable fraction |
| `p_exp`, `p_con` | P(d ≥ Δ), P(d ≤ −Δ) |
| `mad`, `mean_shift` | spread E\|d\| and direction E[d], bp (mad: Wasserstein-1 from a stable allele) |
| `n_atoms` | grid points with ≥ 1% of f |
| `U_lo`, `U_hi`, `mad_lo`, `mad_hi` | 95% bootstrap intervals (`--bootstrap B`) |
| `f_L`, `f_p` | f itself (points with mass ≥ 10⁻⁴), to compare samples without refitting |
| `n_eff`, `n_phased_eff`, `n_var_eff`, `n_support` | reads received; with het evidence; expected from \|d\| ≥ Δ; favouring it |
| `unphased_var_frac` | share of the unstable part carried by reads without het evidence |
| `pooled_U`, `pooled_p_exp`, `pooled_p_con`, `pooled_mad`, `pooled_mean_shift` | locus level: both haplotypes on their own d scale, mixed by read share (germline lengths drop out) |
| `ll_m0`, `ll`, `d_ll` | log-likelihood of one length per haplotype, of f, and the gain (no p-values) |
| `hap`, `hap_rank` | HiPhase label; `short` / `long` / `tie` / `haploid` by L0 (use this across subjects) |

`--write-curves` adds `<out>.curves.parquet`: every read's log-likelihood curve near its peak, with its phase
weight. That is enough to refit Stage 2, or a group-level model, without network 1. All tables tandemo
writes (extraction, Stage 2, scan and tail summaries) are Parquet.

Read instability as an excess over held-out stable samples at matched depth and locus class. How much one or two
deviant reads count depends on how sharp network 1's tails are (see the PDF, section 6).

**pileup**: one SVG per disease locus. Reads are grouped into haplotype 1, haplotype 2 and unphased by the same
phase weights as Stage 2, and drawn aligned at the repeat start: the repeat on a bp scale, tiled into motif units
(catalog motifs, plus units found in the reads that the catalog does not list), interruptions in black, flanks
in grey on a compressed scale with each phased het SNV (tick) and indel (triangle) in the colour of the allele the
read carries. `index.html` ranks the loci by distance of a haplotype's median length from the reference.

**report**: network 1's error distribution p(Y − L) by motif period and true length, summary heatmaps, the effect
of each feature, alleles away from the reference, and held-out calibration against the empirical baseline.

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
