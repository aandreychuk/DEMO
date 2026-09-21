# Minimal native DMM runner

This directory is the small standalone C++ runtime needed to execute the
archived DMM/FastDMM/iDMM AOTI policies on a MovingAI map and scenario. It
contains the complete inference path rather than a model-only smoke test:

- MovingAI `.map` and `.scen` parsing;
- exact per-agent BFS goal-distance (`cost-to-go`) tables;
- the 256-token local observation and 13-slot neighbour/chat tensor;
- five-action DMM AOTI inference;
- native soft collision handling or PIBT shielding;
- optional repeat-state escape (RSE);
- trajectory, decision trace, and summary metrics.

The implementation is intentionally single-environment. For batched GPU
evaluation use POGEMA-GPU; this runner is the compact reproducibility path for
one explicit MAPF task.

## Requirements

- Linux and an NVIDIA GPU compatible with the selected `.pt2` package;
- Python environment with **PyTorch 2.13** and CUDA support;
- CMake 3.20+, Ninja, GCC/G++ 11, C++20, and OpenMP.

The CMake configure step fails early when the active Python environment is not
PyTorch 2.13. AOTI packages are not portable across arbitrary Torch/CUDA/GPU
versions; rebuild the package when the target architecture differs.

## Build

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
cmake -S native-runner -B native-runner/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build native-runner/build -j
```

## Run DMM-7M GRPO-500 with PIBT and RSE

Run from the root of this model repository:

```bash
native-runner/build/dmm_native_runner \
  --map native-runner/examples/test01.map \
  --scen native-runner/examples/test01.scen \
  --model models/dmm-7m/aoti/grpo-500/policy.pt2 \
  --num-agents 8 \
  --max-steps 5000 \
  --mode pibt \
  --sampling deterministic \
  --escape-repeated-states \
  --max-repeat-retries 16 \
  --output-prefix result
```

Outputs:

- `result.summary.json`: machine-readable status and metrics;
- stdout: the same status, CSR/ISR, SoC, makespan, runtime, inference and
  PIBT/RSE/collision diagnostics;
- `result.trajectory.tsv`: position of every agent at every step;
- `result.decisions.tsv`: preferred action ranking and executed action.

The trajectory is the solution. Action ids are `0=wait`, `1=up`, `2=down`,
`3=left`, `4=right`. A solved run has `status=solved`; for an unsolved run SoC
and makespan are reported as `-1`.

For an interactive lifelong simulation, add `--stream` and set
`--max-steps 0`. The process emits its initial `MAPF_FRAME` line, then reads one
`step` command from stdin before each policy inference and state transition.
`quit` ends the session. This mode keeps no trajectory history in memory.

For a build-and-run integration check from the repository root:

```bash
PYTHON_EXECUTABLE="$(command -v python3)" native-runner/run_smoke.sh
```

The script defaults to the bundled DMM-7M GRPO-500 AOTI package. Set
`DMM_MODEL=/path/to/policy.pt2` to validate a different compatible package.

## Other archived models

Pass any compatible `.pt2` file with `--model`. The loader supports the two
contracts present in this repository:

- `obs-chat-v1`: observation and chat tensors;
- `explicit-random-v1`: observation, chat, initial exponential samples, and
  per-round Gumbel samples.

The sidecar next to a `.pt2` records its exact contract and compilation
profile. Keep the sidecar beside the package.

## Provenance

The MAPF graph/instance, observation tokenizer, model wrapper, PIBT, and RSE
path are extracted from the native DMM evaluation implementation used by the
MAPF-GPU experiments. See `SOURCE_MAP.md` for the component map,
`VALIDATION.md` for the verified smoke record, and `licenses/LICENSE-LAGAT`
for the upstream notice.
