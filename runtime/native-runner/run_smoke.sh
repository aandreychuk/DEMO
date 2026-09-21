#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON_EXECUTABLE:-python3}"
MODEL="${DMM_MODEL:-$ROOT/../models/dmm-7m/aoti/grpo-500/policy.pt2}"

if [[ ! -f "$MODEL" ]]; then
  echo "DMM AOTI package not found: $MODEL" >&2
  echo "Set DMM_MODEL to a compatible .pt2 package." >&2
  exit 2
fi

export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export TORCH_PREFIX
TORCH_PREFIX="$($PY -c 'import pathlib, torch; assert str(torch.__version__) == "2.13.0+cu126", torch.__version__; print(pathlib.Path(torch.__file__).parent)')"
export LD_LIBRARY_PATH="$($PY -c 'import pathlib, site; p=pathlib.Path(site.getsitepackages()[0]); print(":".join([str(p/"torch/lib"), *map(str, (p/"nvidia").glob("*/lib"))]))'):${LD_LIBRARY_PATH:-}"

cmake -S "$ROOT" -B "$ROOT/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PY" \
  -DTORCH_PREFIX="$TORCH_PREFIX"
cmake --build "$ROOT/build" -j2

rm -f "$ROOT/smoke."{summary.json,trajectory.tsv,decisions.tsv}
"$ROOT/build/dmm_native_runner" \
  --map "$ROOT/examples/test01.map" \
  --scen "$ROOT/examples/test01.scen" \
  --model "$MODEL" \
  --num-agents 3 \
  --max-steps 128 \
  --mode pibt \
  --sampling deterministic \
  --escape-repeated-states \
  --max-repeat-retries 16 \
  --output-prefix "$ROOT/smoke" \
  >"$ROOT/smoke.log" 2>&1

"$PY" - "$ROOT" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "smoke.summary.json").read_text())
rows = list(csv.DictReader((root / "smoke.trajectory.tsv").open(), delimiter="\t"))
assert summary["status"] in {"solved", "no_solution"}
assert summary["num_agents"] == 3
assert rows and {int(row["agent"]) for row in rows} == {0, 1, 2}
assert len(rows) == (summary["episode_steps"] + 1) * 3

by_step = {}
for row in rows:
    by_step.setdefault(int(row["step"]), []).append(
        (int(row["agent"]), int(row["x"]), int(row["y"])))
for step, positions in by_step.items():
    coords = [(x, y) for _, x, y in positions]
    assert len(coords) == len(set(coords)), f"vertex collision at step {step}"
for step in range(1, max(by_step) + 1):
    previous = {agent: (x, y) for agent, x, y in by_step[step - 1]}
    current = {agent: (x, y) for agent, x, y in by_step[step]}
    for a in previous:
        for b in previous:
            if a < b:
                assert not (
                    previous[a] == current[b] and previous[b] == current[a]
                ), f"swap collision at step {step}: {a}, {b}"

(root / "smoke.validation.json").write_text(
    json.dumps(
        {
            "passed": True,
            "status": summary["status"],
            "episode_steps": summary["episode_steps"],
            "trajectory_rows": len(rows),
            "torch": "2.13.0+cu126",
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"event": "native_runner_smoke_passed", **summary}))
PY
