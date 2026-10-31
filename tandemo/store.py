"""Extraction tables on disk, as Parquet.

<prefix>.reads.parquet / .loci.parquet hold typed, zstd-compressed columns; readers take only the columns they
need, filter loci while reading, and keep strings Arrow-backed (no Python objects per read).
"""

from __future__ import annotations

import os

import pandas as pd

KINDS = ("reads", "loci")

# column -> Arrow type name; columns not listed are float64
TYPES = {
    "reads": dict(locus="string", chrom="string", start="int64", end="int64", read="string", hp="int16",
                  ps_tag="int64", Y="int32", rq="float32", np="int16", read_len="int32", mapq="int16", n_sa="int16",
                  bq_rep="float32", bq_flank="float32", dist_end="int32", flank_id_min="float32", n_het="int16",
                  llr="float32", n_indel="int16", n_hap1="int16", n_hap2="int16", mC_flank="float32",
                  mC_rep="float32", mC_cert="float32"),
    "loci": dict(locus="string", chrom="string", start="int64", end="int64", motif="string", period="int16",
                 ref_len="int32", gc="float32", hp_frac="float32", max_hp="int16", cx3="float32", flank_cx="float32",
                 locus_ps="int64", al1="int32", al2="int32", ploidy="int8", trgt_phased="int8", n_overlap="int32",
                 n_span="int32", read_lens="string", pad_d1="int32", pad_d2="int32", pad_status="int8",
                 pad="int16"),
}


def parquet_path(prefix: str, kind: str) -> str:
    return f"{prefix}.{kind}.parquet"


def schema(kind: str, columns):
    import pyarrow as pa
    t = TYPES[kind]
    return pa.schema([(c, getattr(pa, t.get(c, "float64"))()) for c in columns])


def table_from_rows(kind: str, columns, rows):
    """An Arrow table from row tuples (the extractor's rows), typed by TYPES."""
    import pyarrow as pa
    sch = schema(kind, columns)
    cols = list(zip(*rows)) if rows else [[] for _ in columns]
    return pa.Table.from_arrays([pa.array(list(v), type=f.type) for v, f in zip(cols, sch)], schema=sch)


class Writer:
    """Row groups appended one window at a time; the file appears under its final name only when closed."""

    def __init__(self, path: str, kind: str, columns):
        import pyarrow.parquet as pq
        self.path, self.tmp = path, path + ".tmp"
        self.w = pq.ParquetWriter(self.tmp, schema(kind, columns), compression="zstd")

    def write(self, table):
        if table.num_rows:
            self.w.write_table(table)

    def close(self):
        self.w.close()
        os.replace(self.tmp, self.path)


def _arrow_strings():
    import pyarrow as pa
    return {pa.string(): pd.StringDtype("pyarrow"), pa.large_string(): pd.StringDtype("pyarrow")}.get


def read_table(prefix: str, kind: str, columns=None, loci_keep=None, arrow: bool = False):
    """One extraction table as a DataFrame (an Arrow table with arrow=True): only `columns` (all when None), only the
    loci in `loci_keep`."""
    import pyarrow.dataset as ds
    d = ds.dataset(parquet_path(prefix, kind), format="parquet")
    cols = [c for c in columns if c in d.schema.names] if columns is not None else None
    filt = ds.field("locus").isin(list(loci_keep)) if loci_keep is not None else None
    t = d.to_table(columns=cols, filter=filt)
    return t if arrow else to_pandas(t)


def to_pandas(t) -> pd.DataFrame:
    return t.to_pandas(types_mapper=_arrow_strings(), self_destruct=True)



def exists(prefix: str, kind: str = "reads") -> bool:
    return os.path.exists(parquet_path(prefix, kind))
