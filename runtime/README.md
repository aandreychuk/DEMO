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
native process, and requests one new FastDMM + PIBT step for every simulation
tick. Pause stops issuing native step commands. Stop terminates the current
process, starts a fresh simulator, and holds its initial state. No trajectory is
precomputed or replayed by the browser bridge.

Supported demo sizes are 25, 50, and 100 agents. The bridge uses an unbounded
native horizon by default. Pass a positive `--max-steps` only for diagnostics.
The browser can also send `fail <agent>` through the bridge. The native runner
queues the failure for a single external recovery vehicle. Its shortest BFS
route ignores live-agent occupancy, giving the vehicle right of way; live agents
instead reserve both its current cell and the cell it just traversed through
their dynamic obstacle masks, cost-to-go recomputation, and PIBT constraints.
The vehicle enters the failed robot's cell, waits three ticks while its platform
lifts the stationary robot, and only then moves the robot toward repair. The
vehicle and failed robot remain outside relational observations and
`agent_chat_ids`.
After eight repair ticks, the robot resumes the same assignment and stage. A
carried pallet is left at the failure cell and becomes the repaired robot's
temporary recovery goal before its saved task goal is restored.

`generate_lifelong_warehouse.py` creates the 44×32 map, 100 starts, a visual
layout manifest, and 4,000 randomly ordered tasks. A task visits its pallet,
one unloading cell, and the original pallet cell. Pallets start with 12 items,
and the unloading arm removes one item on each visit. If the pallet becomes
empty, the native runner inserts a batch-loading goal on the opposite edge of
the warehouse before the return goal. After five stationary ticks its inventory
changes atomically from 0 to 12. The native runner updates the goal and cost-to-go
after every stage.

Parked pallets are per-agent dynamic obstacles only while that agent is carrying
a load. All unloading and loading bays except the agent's assigned target are
masked from both cost-to-go and PIBT. Guarded service cells behind the bays and
boundary end caps leave one aisle-side entrance per bay. The browser renders a
robotic arm in each unloading cell and animates the single-item transfer when an
agent advances to its next goal. Loading bays place all 12 items on an empty
pallet as one animated batch. Both service operations hold the agent in its bay
for five additional simulation steps. The renderer starts each operation only
after the robot has completed its entry and produced a stationary bay tick.

Pickup and return each hold the agent for two additional native steps. During
those holds the browser lifts or lowers the pallet using interpolated simulation
time, so the handling motion stays synchronized at every playback speed.
Completed task templates return to the randomized queue with a new assignment
ID, so task generation continues for the lifetime of the simulator.
