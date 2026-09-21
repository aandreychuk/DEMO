#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_executable="${PYTHON_EXECUTABLE:-python3}"
build_dir="${NATIVE_BUILD_DIR:-${runtime_dir}/native-runner/build}"

cmake -S "${runtime_dir}/native-runner" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="${python_executable}"
cmake --build "${build_dir}" -j "${BUILD_JOBS:-6}"
