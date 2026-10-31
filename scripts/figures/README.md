# Figure and analysis scripts

One-off scripts behind the figures and tables of the T2T runs (NA19909 pangenome, UCL and Leiden brain, HG002).
They hard-code the Snellius paths of those runs (`/projects/0/prjs2178/seed/pacbio/...`); copy and edit the
paths to reuse them. `t2t_donor.sh` is the alignment-to-TRGT pipeline for one donor.

- `brain_full.py`, `brain_pdf.py`, `brain_tables.py`: the brain unstable-loci PDF (pileups, length panels, tables)
- `stage2_brain.py`: Stage 2 (NPMLE) at the brain loci in UCL, Leiden and HG002, and the per-allele deviation plots
- `acyp2_stage2.py`: the two-part length panel (reads | Stage 2 true lengths) above a pileup, ACYP2 in Leiden
- `loc101*.py`, `consensus_loc101.py`: LOC101928932 (local allele, read consensus)
- `cdon_*.py`, `rfc1_fgf14.py`, `prkca*.py`: single-locus figures
- `compare_hg002*.py`, `brain_vs_hg002_table.py`, `top_*.py`: brain vs HG002 tables, top unstable loci
