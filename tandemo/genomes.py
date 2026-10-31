"""Reference genomes and their tandem-repeat catalogs, selected with `--genome grch38|t2t`.

Paths are relative to TANDEMO_REFDIR (default: the prjs2178 genomic_refs/human tree on Snellius). `--reference`
and `--catalog` given as paths always win; `--catalog` may also name one of the genome's catalogs.

Catalog filters (`extract --min-len --no-homopolymer --no-segdup`) use the genome's segmental duplications and
always keep its pathogenic loci.

grch38  human_GRCh38_no_alt_analysis_set (PacBio's analysis set)
        adotto      Adotto TR catalog in TRGT format, 937 k loci (default)
        trgt        PacBio's TRGT repeat catalog v0.3.4, 171 k loci
        pathogenic  PacBio's pathogenic repeats, 56 disease loci (ID = gene)
t2t     T2T-CHM13v2.0, chm13v2.0_maskedY_rCRS analysis set (chrY PARs masked, rCRS chrM)
        platinum    Platinum TRs v1.0 (TRGT format), 7.8 M loci (default)
        pathogenic  the 68 Platinum loci named <DISEASE>_<GENE>_pathogenic (scripts/t2t_pathogenic_catalog.sh)
"""

from __future__ import annotations

import os

DEFAULT_ROOT = "/projects/prjs2178/genomic_refs/human"

GENOMES = {
    "grch38": dict(
        reference="GRCh38/reference/human_GRCh38_no_alt_analysis_set.fasta",
        catalogs=dict(
            adotto="GRCh38/reference/human_GRCh38_no_alt_analysis_set.adotto_repeat_catalog.bed.gz",
            trgt="GRCh38/reference/human_GRCh38_no_alt_analysis_set.repeat_catolog.bed",
            pathogenic="GRCh38/reference/human_GRCh38_no_alt_analysis_set.pathogenic_repeats.bed"),
        default_catalog="adotto",
        segdups="GRCh38/pipeline_resources/GRCh38/annotation/GRCh38.segdups.bed.gz"),
    "t2t": dict(
        reference="T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.fa",
        catalogs=dict(
            platinum="T2T-CHM13v2.0/platinum_trs_v1.0/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.trgt.bed.gz",
            pathogenic="T2T-CHM13v2.0/pathogenic_repeats/chm13v2.0_maskedY_rCRS.platinumTRs-v1.0.pathogenic.trgt.bed"),
        default_catalog="platinum",
        segdups="T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.segdups.bed.gz",
        genes="T2T-CHM13v2.0/chm13v2.0_maskedY_rCRS.refseq_RS_2025_08"),     # NCBI RefSeq RS_2025_08 gene tables
}


def root() -> str:
    return os.environ.get("TANDEMO_REFDIR", DEFAULT_ROOT)


def resolve(genome: str | None, reference: str | None = None, catalog: str | None = None,
            default_catalog: str | None = None) -> tuple[str | None, str | None]:
    """(reference, catalog) paths. Without a genome both must be paths (or None)."""
    if genome is None:
        return reference, catalog
    g = GENOMES[genome]
    ref = reference or os.path.join(root(), g["reference"])
    name = catalog or default_catalog or g["default_catalog"]
    if name in g["catalogs"]:
        cat = os.path.join(root(), g["catalogs"][name])
    elif os.path.exists(name):
        cat = name
    else:
        raise SystemExit(f"--catalog {name}: not a file, nor one of {genome}'s catalogs "
                         f"({', '.join(g['catalogs'])})")
    return ref, cat


def resource(genome: str | None, key: str) -> str | None:
    """Path of a genome resource ("segdups") or catalog ("pathogenic"), or None."""
    if genome is None:
        return None
    g = GENOMES[genome]
    rel = g.get(key) or g["catalogs"].get(key)
    return os.path.join(root(), rel) if rel else None


def describe() -> str:
    lines = [f"TANDEMO_REFDIR = {root()}"]
    for name, g in GENOMES.items():
        lines.append(f"{name}\n  reference  {os.path.join(root(), g['reference'])}")
        for c, p in g["catalogs"].items():
            mark = " (default)" if c == g["default_catalog"] else ""
            ok = "" if os.path.exists(os.path.join(root(), p)) else "  [missing]"
            lines.append(f"  {c:<10} {os.path.join(root(), p)}{mark}{ok}")
        lines.append(f"  {'segdups':<10} {os.path.join(root(), g['segdups'])}")
    return "\n".join(lines)
