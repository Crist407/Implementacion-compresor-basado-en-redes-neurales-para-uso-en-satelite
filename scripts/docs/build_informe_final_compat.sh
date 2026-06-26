#!/usr/bin/env bash
set -euo pipefail

# Compila la copia de trabajo de la plantilla institucional.
# La plantilla original de INFORME FINAL/ no se modifica.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_DOC="$ROOT/docs/informe_final/main.tex"

clean_aux() {
  local dir="$1"
  local job="$2"
  rm -f \
    "$dir/$job.aux" \
    "$dir/$job.bbl" \
    "$dir/$job.blg" \
    "$dir/$job.fdb_latexmk" \
    "$dir/$job.fls" \
    "$dir/$job.lof" \
    "$dir/$job.log" \
    "$dir/$job.lot" \
    "$dir/$job.out" \
    "$dir/$job.synctex.gz" \
    "$dir/$job.toc"
  find "$dir" -mindepth 2 -name '*.aux' -delete
}

ACTION="build"
if [[ "${1:-}" == "--clean" || "${1:-}" == "--clean-all" ]]; then
  ACTION="$1"
  shift
fi

DOC="${1:-$DEFAULT_DOC}"
if [[ "$DOC" != /* ]]; then
  DOC="$ROOT/$DOC"
fi
if [[ ! -f "$DOC" && -f "$DOC.tex" ]]; then
  DOC="$DOC.tex"
fi
[[ -f "$DOC" ]] || { echo "[ERROR] LaTeX root not found: $DOC" >&2; exit 1; }

DOC_DIR="$(cd "$(dirname "$DOC")" && pwd)"
DOC_FILE="$(basename "$DOC")"
JOB="${DOC_FILE%.tex}"

if [[ "$ACTION" == "--clean" || "$ACTION" == "--clean-all" ]]; then
  clean_aux "$DOC_DIR" "$JOB"
  if [[ "$ACTION" == "--clean-all" ]]; then
    rm -f "$DOC_DIR/$JOB.pdf"
  fi
  echo "[OK] Cleaned LaTeX outputs under $DOC_DIR"
  exit 0
fi

cd "$DOC_DIR"
if [[ "${LATEX_LEGACY_RELEASE:-0}" == "1" ]]; then
  TEX_INPUT="\\RequirePackage[2020-10-01]{latexrelease}\\input{$DOC_FILE}"
else
  TEX_INPUT="$DOC_FILE"
fi
PDFLATEX=(
  pdflatex
  "-jobname=$JOB"
  -interaction=nonstopmode
  -file-line-error
  -halt-on-error
  -synctex=1
  "$TEX_INPUT"
)

"${PDFLATEX[@]}"
if [[ -f "$JOB.aux" ]] && grep -q '\\bibdata' "$JOB.aux"; then
  bibtex "$JOB"
fi
"${PDFLATEX[@]}"
"${PDFLATEX[@]}"

[[ -f "$JOB.pdf" ]] || { echo "[ERROR] PDF was not generated: $DOC_DIR/$JOB.pdf" >&2; exit 1; }
echo "[OK] Informe final compiled: $DOC_DIR/$JOB.pdf"
