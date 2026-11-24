#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  printf '%s\n' 'Activate kvcohort-dev first.' >&2
  exit 2
fi
cmake -S "$project_dir" -B "$project_dir/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_BUILD_RPATH="$CONDA_PREFIX/lib" -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib"
cmake --build "$project_dir/build" --parallel 2
ctest --test-dir "$project_dir/build" --output-on-failure
