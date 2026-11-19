#!/usr/bin/env bash
# =============================================================================
# scripts/reproduce.sh — regenerate every figure in results/ from configs/.
#
#   bash scripts/reproduce.sh                 # GPU (Colab/Kaggle): real models
#   bash scripts/reproduce.sh --mock          # CPU smoke test, no model downloads
#
# Options
#   --mock               CPU placeholder generation; results go to <out>/results
#                        (never overwrites the real figures in results/)
#   --experiments "IDS"  space-separated experiment IDs (default: "E1 E2 E6",
#                        the set behind the committed figures; "all" = E1–E8)
#   --out DIR            scratch dir for raw outputs (default: outputs/reproduce)
#   --results DIR        where figures are written (default: results/, or
#                        <out>/results with --mock)
#
# Steps: environment check -> sketch preprocessing -> experiment sweeps
# (+ hero figure) -> metrics/plots/ANALYSIS.md -> SHA-256 manifest.
#
# <out> is DELETED first: run_experiments.py is resumable and would otherwise
# silently reuse images from an earlier run instead of regenerating them.
# Compare two runs with:  diff <outA>/checksums.sha256 <outB>/checksums.sha256
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
MOCK=0
EXPERIMENTS="E1 E2 E6"
OUT="outputs/reproduce"
RESULTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MOCK=1; shift ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$RESULTS" ]]; then
  if [[ $MOCK -eq 1 ]]; then RESULTS="$OUT/results"; else RESULTS="results"; fi
fi

# Refuse to wipe anything outside the repo's outputs/ tree or the given scratch dir.
case "$OUT" in
  ""|"/"|"."|".."|"$ROOT") echo "Refusing to use --out '$OUT'" >&2; exit 2 ;;
esac

# Determinism knobs (see README "Reproducibility"):
export PYTHONHASHSEED=0                 # stable str hashing across processes
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS kernels on CUDA
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONIOENCODING=utf-8           # ✓/★ in output would crash cp1252 pipes on Windows

MOCK_FLAG=()
ANALYZE_FLAGS=()
if [[ $MOCK -eq 1 ]]; then
  MOCK_FLAG=(--mock)
  ANALYZE_FLAGS=(--edge-methods canny --no-clip)   # no model downloads on CPU
elif ! "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  # Without CUDA the generator silently falls back to placeholder images; a real
  # run would then overwrite results/ with fakes. Refuse before deleting anything.
  echo "No CUDA GPU detected. Run this on a GPU (Colab/Kaggle), or use --mock" \
       "for a CPU smoke test (writes to $OUT/results, never results/)." >&2
  exit 3
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Fresh output directory: $OUT"
rm -rf "$OUT"
mkdir -p "$OUT" "$RESULTS"

step "1/5 Environment"
"$PY" scripts/check_env.py | tee "$OUT/environment.txt"

step "2/5 Preprocess sketches -> $OUT/conditions"
"$PY" -m sketch2image.preprocessing --input sketches --out "$OUT/conditions" --modes clean,canny

step "3/5 Experiments: $EXPERIMENTS (+ hero figure from E6)"
# shellcheck disable=SC2086
"$PY" scripts/run_experiments.py -e $EXPERIMENTS \
  --out-dir "$OUT/experiments" --results-dir "$RESULTS" "${MOCK_FLAG[@]}"

step "4/5 Metrics, plots and ANALYSIS.md"
"$PY" scripts/analyze.py --outputs-dir "$OUT/experiments" --results-dir "$RESULTS" \
  "${ANALYZE_FLAGS[@]}"

step "5/5 SHA-256 manifest -> $OUT/checksums.sha256"
# Hash the raw generated images and condition maps (pixel-level reproducibility) and
# every figure. Paths are relative so manifests from different --out dirs diff cleanly.
(
  cd "$OUT"
  find conditions experiments -type f -name '*.png' | LC_ALL=C sort | xargs sha256sum
) > "$OUT/checksums.sha256"
# metrics.csv and backend_comparison.png contain wall-clock runtime, so they differ
# run to run by design and are left out of the manifest.
(
  cd "$RESULTS"
  find . -maxdepth 1 -type f \( -name '*.png' -o -name '*.csv' \) \
    ! -name metrics.csv ! -name backend_comparison.png | LC_ALL=C sort \
    | xargs sha256sum | sed -E 's#^([0-9a-f]+) [ *]\./#\1  results/#'
  echo "# excluded (contain runtime): results/metrics.csv results/backend_comparison.png"
) >> "$OUT/checksums.sha256"
echo "$(wc -l < "$OUT/checksums.sha256") files hashed."

step "Done"
echo "Figures:  $RESULTS"
echo "Raw:      $OUT/experiments"
[[ $MOCK -eq 1 ]] && echo "NOTE: --mock run — placeholder images, not model outputs."
exit 0
