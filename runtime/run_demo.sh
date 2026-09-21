#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_executable="${PYTHON_EXECUTABLE:-python3}"

model="${DMM_MODEL:-}"
if [[ -z "${model}" ]]; then
  if [[ $# -eq 0 ]]; then
    echo "usage: runtime/run_demo.sh MODEL.pt2 [bridge options]" >&2
    echo "   or: DMM_MODEL=MODEL.pt2 runtime/run_demo.sh [bridge options]" >&2
    exit 2
  fi
  model="$1"
  shift
fi

"${python_executable}" "${runtime_dir}/generate_lifelong_warehouse.py"
exec "${python_executable}" "${runtime_dir}/bridge.py" --model "${model}" "$@"
