#!/usr/bin/env bash
# Build the offline vLLM wheelhouse dataset that the Kaggle rerun needs.
#
# The rerun has NO internet, so `pip install vllm` cannot work there. The
# notebook installs from an attached wheelhouse dataset with
# `pip install --no-index --find-links <dir> vllm` -- the same pattern the Duck's
# own Kaggle notebook uses for the ARC runtime.
#
# Run this on a machine with internet AND the same platform/Python as Kaggle's
# notebook VM (linux x86_64, CPython 3.12 as of writing -- the kernel log shows
# /usr/local/lib/python3.12/dist-packages). Wheels are platform-specific, so
# building from Windows or from a different CPython minor version produces files
# pip will refuse at rerun.
#
#   ./scripts/build_wheelhouse.sh [output_dir] [vllm_version]
#
# Then publish it:
#   kaggle datasets create -p <output_dir>   # or `kaggle datasets version`
# and add the resulting `owner/slug` to dataset_data_sources in
# notebook/kernel-metadata.json (and re-run scripts/build_notebook.py).
set -euo pipefail

OUT="${1:-wheelhouse}"
VLLM_VERSION="${2:-}"
PY_TAG="${PY_TAG:-cp312}"
PLATFORM="${PLATFORM:-manylinux2014_x86_64}"

mkdir -p "$OUT"

PIN=""
if [[ -n "$VLLM_VERSION" ]]; then PIN="==$VLLM_VERSION"; fi

echo "==> downloading vllm${PIN} + deps for ${PY_TAG}/${PLATFORM} into ${OUT}"

# --only-binary=:all: with an explicit platform is what makes this reproducible
# from a machine that is not the target; without it pip resolves for the host.
python -m pip download \
  --dest "$OUT" \
  --only-binary=:all: \
  --python-version "${PY_TAG#cp}" \
  --implementation cp \
  --abi "${PY_TAG}" \
  --platform "$PLATFORM" \
  --platform manylinux_2_17_x86_64 \
  --platform linux_x86_64 \
  "vllm${PIN}"

# Pillow renders the multimodal panel and numpy is what model-generated transforms
# use inside the sandbox. Both are usually already on the Kaggle base image, but
# including them costs a few MB and removes an assumption.
python -m pip download \
  --dest "$OUT" \
  --only-binary=:all: \
  --python-version "${PY_TAG#cp}" \
  --implementation cp \
  --abi "${PY_TAG}" \
  --platform "$PLATFORM" \
  --platform manylinux_2_17_x86_64 \
  --platform linux_x86_64 \
  pillow numpy || echo "WARNING: pillow/numpy download failed; the base image probably has them"

echo
echo "==> $(ls -1 "$OUT" | wc -l) files, $(du -sh "$OUT" | cut -f1)"
echo
echo "Sanity-check before publishing (a vLLM that cannot import is worse than none,"
echo "because the notebook would then try to serve and burn the startup reserve):"
echo "  docker run --rm -v \"\$PWD/$OUT:/wh:ro\" python:3.12-slim sh -c \\"
echo "    'pip install --no-index --find-links /wh vllm && python -c \"import vllm; print(vllm.__version__)\"'"
echo
echo "Note: vLLM pulls torch + CUDA wheels, so expect several GB. Kaggle datasets cap"
echo "at 100 GB, which is fine, but the mount is read-only and pip needs the"
echo "--no-index/--find-links form the notebook already uses."
