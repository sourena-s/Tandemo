#!/usr/bin/env bash
# SVG pages -> one PDF, each page at its own size. (rsvg-convert given several SVGs sizes every page like the
# first one, which crops taller pages.) Uses the svgpdf env: librsvg + poppler from conda-forge, on any node.
#   bash scripts/pages_to_pdf.sh <pages.txt: one SVG path per line> <out.pdf>
set -euo pipefail
E=${SVGPDF_ENV:-/projects/prjs2178/software/miniforge3/envs/svgpdf/bin}
LIST=${1:?pages.txt}; OUT=${2:?out.pdf}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
n=0
while read -r svg; do
    [ -n "$svg" ] || continue
    n=$((n + 1)); "$E/rsvg-convert" -f pdf "$svg" -o "$T/$(printf %04d $n).pdf"
done < "$LIST"
"$E/pdfunite" "$T"/*.pdf "$OUT"
echo "$OUT: $n pages"
