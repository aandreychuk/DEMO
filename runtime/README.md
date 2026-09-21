# Local FastDMM runtime assets

`export_fastdmm_aoti.py` turns a trusted FastDMM training checkpoint into a
two-input AOTInductor package for the GPU in the current machine. The package
contract matches the native runner published with `dmm-mapf-checkpoints`:

- observations: `int64[1, N, 256]`;
- chat/neighbour IDs: `int64[1, N, 13]`;
- output: `float32[N, 5]` action probabilities;
- dynamic agent count: `2 <= N <= 2580` by default.

The environment needs PyTorch `2.13.0+cu126`, Ninja, and the matching CUDA
development headers. When CUDA is provided by Python wheels, install the
missing development part with:

```bash
uv pip install \
  nvidia-cuda-nvcc-cu12==12.6.85 \
  nvidia-cuda-cccl-cu12==12.6.77
```

The exporter writes `<package>.json` beside the package. The native C++ loader
uses `"input_contract": "obs-chat-v1"` in this sidecar to select the correct
two-input call path.

The vendored model definition comes from POGEMA-GPU commit
`60f6801af18273f3fba387258987b27c68e15b10` and retains its MIT license in
`vendor/fastdmm/LICENSE`. Its deterministic return omits one redundant
`.float()` call because Torch 2.13 otherwise records inconsistent autocast
metadata during AOTI lowering. The consensus tensor is already FP32 after the
first centered-log update, so this adaptation does not change values or dtype.

Example from the prepared WSL environment:

```bash
source /home/ubuntu/codex-mapf-export/.venv/bin/activate
python /mnt/d/GitHub/DEMO/runtime/export_fastdmm_aoti.py \
  --checkpoint /home/ubuntu/codex-mapf-export/artifacts/fastdmm-0.8m/fastdmm_grpo_stage2_step12800.pt \
  --output /mnt/d/GitHub/DEMO/artifacts/fastdmm-stage2-step12800-sm86.pt2 \
  --validate-agents 2,64,1000,2500
```

The generated package is machine-specific and intentionally ignored by Git.

## Native demo pipeline

The repository includes the standalone C++ runtime in `native-runner/`, a
deterministic lifelong warehouse scenario, and a loopback WebSocket bridge. From an
activated Linux/WSL environment with PyTorch 2.13:

```bash
python -m ensurepip --upgrade
python -m pip install -r runtime/requirements.txt
runtime/build_native.sh
runtime/run_demo.sh artifacts/fastdmm-stage2-step12800-bf16-sm86-dynamic-N2-2580.pt2
```

The bridge listens on `ws://127.0.0.1:18765`, runs the selected scenario in the
native process, and replays its collision-free trajectory at 10 Hz. Browser
requests for a previously computed agent count use the in-memory episode cache.
Runs and decision traces are written below ignored `runtime/runs/`.

Supported demo sizes are 25, 50, and 100 agents. The default native horizon is
600 steps; change it with `--max-steps` when invoking `run_demo.sh`.

`generate_lifelong_warehouse.py` creates the 44×32 map, 100 starts, a visual
layout manifest, and 4,000 randomly ordered tasks. A task visits its pallet,
one unloading cell, and the original pallet cell. The native runner updates the
goal and cost-to-go after every stage. Parked pallets are per-agent dynamic
obstacles only while that agent is carrying a load. All unloading bays except
the agent's assigned target are masked from both cost-to-go and PIBT. Guarded
service cells behind the bays and boundary end caps leave one aisle-side
entrance per bay. The browser renders a robotic arm in each guarded cell and
animates the cargo transfer when an agent advances from unloading to return.
After that transition, the native runtime holds the agent in its bay for five
additional simulation steps before allowing it to leave.
