"""Write the network-1 feature table for the method document from tandemo.features, so the
document and the code cannot drift apart. Run from the repository root: python docs/make_tables.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tandemo.features import L_FEATURES, LOCUS_FEATURES, READ_FEATURES  # noqa: E402

TF = {"phred": "Phred", "log1p": "log(1+x)", "log10": "log10", "log2": "log2", "div60": "x/60",
      "div40": "x/40", "pos": "$x>0$", "err": "100(1-x)", "id": "x"}


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace(">=", r"$\geq$").replace("<=", r"$\leq$")


rows = []
for group, d in (("read", READ_FEATURES), ("locus", LOCUS_FEATURES)):
    for name, (src, kind, desc) in d.items():
        rows.append(f"{group} & \\texttt{{{esc(name)}}} & \\texttt{{{esc(src)}}} & {TF[kind]} & {esc(desc)} \\\\")
for name, desc in L_FEATURES.items():
    rows.append(f"candidate $L$ & \\texttt{{{esc(name)}}} & -- & -- & {esc(desc)} \\\\")

out = os.path.join(os.path.dirname(__file__), "figs", "features_table.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write("\\begin{tabularx}{\\textwidth}{@{}l l l l L@{}}\n\\toprule\n")
    fh.write("group & feature & source column & transform & meaning \\\\\n\\midrule\n")
    fh.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")
print("wrote", out)
