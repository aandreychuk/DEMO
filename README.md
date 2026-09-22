# MAPF Operations Deck

A browser-based 2.5D demonstration of the trained FastDMM policy. The default
runtime is fully self-contained: ONNX Runtime Web executes the 0.8M policy with
WebGPU, a compact WebAssembly core builds observations and performs BFS and
PIBT collision shielding, and a Web Worker owns the lifelong simulation. The
main thread receives the same compact binary frames used by the native runtime
and renders warehouses with up to 1000 agents using Babylon.js thin instances. When
WebGPU is unavailable, ONNX Runtime automatically falls back to single-threaded
WASM.

The visual cell pitch is larger than the physical robot and pallet footprint.
A pallet occupies about 75% of its cell, leaving visible clearance while two
loaded agents follow adjacent paths or interpolate through an L-shaped turn.

Each generated task starts with three goals: pick up a pallet, deliver it to an
unloading station, and return it to its original cell. Every pallet holds 12
separate goods. The unloading arm removes one item per delivery. When that item
was the last one, the simulator inserts a fourth goal at a loading bay on the
opposite side of the warehouse. The empty pallet waits there for five ticks,
receives a full batch of 12 goods in one operation, and only then returns to its
original cell. Empty robots can pass
under parked pallets. Loaded robots receive a pallet-aware cost-to-go map and
cannot enter cells occupied by parked pallets. Unloading cells are dead-end
bays: an agent may enter only its assigned bay and must leave through the same
aisle-side edge. A robotic arm behind every bay lifts one item from an arriving
pallet and places it into an open box on the outbound conveyor. Boxes travel in
a continuous line, hold 18 goods in a 3×3×2 stack, descend at the end of the
belt, and disappear below the floor. The arm reserves an exact empty box slot
when unloading begins and follows that moving target through the release. Its
3D reach targets the occupied
slot selected from the current inventory, giving all 12 pallet positions a
distinct pickup path. The gripper closes around the item before it moves. The
robot remains in the bay for five additional simulation steps during the
handoff so the transfer stays visually readable. The arm starts only after the
robot has completed its entry and produced its first stationary bay tick.
Picking up a loaded pallet and parking the empty pallet each take two additional
simulation steps; the rendered pallet rises and lowers smoothly during those
steps at every playback speed.

Click a robot or its carried pallet to open the right-side inspector. It shows
the assigned task, pallet inventory, all four possible route goals, the active
task stage, and per-agent distance, waiting, unloading, and completion counters. Global live
telemetry is grouped in the same panel. Selecting an agent also highlights its
physical pallet, assigned unloading bay, and the loading bay when the pallet is
about to become empty.

The inspector can break the selected robot. A separate recovery vehicle follows
a shortest BFS route to the failure and has right of way over live agents. It
drives into the failed robot's cell, lifts the stationary robot onto its low
platform over three ticks, carries it to the repair station, and returns to its
depot. The failed robot and recovery vehicle appear only as dynamic obstacles in
FastDMM cost-to-go and PIBT constraints; neither participates in relational
observations or `agent_chat_ids`. Repairs take eight ticks. The repaired robot
keeps its assignment and task stage. If it failed while loaded, its pallet stays
at the failure cell, and the robot returns to pick it up before resuming its
original goal.

## Run entirely in the browser

Building requires Node.js 22 or newer and pnpm. End users need only a current
browser; there is no Python, PyTorch, CUDA, or local bridge to install.

```powershell
pnpm install
pnpm dev -- --host 127.0.0.1 --port 4173
```

Open the local URL shown by Vite. The header reports FASTDMM // WEBGPU or
FASTDMM // WASM after the selected execution provider is ready. Runs with up
to 100 agents keep the original 100-slot ONNX graph. Selecting 101–1000 agents
in the layout editor loads a separate 1000-slot graph on demand and pads unused
slots. This increases inference time and GPU memory use at larger counts.

For a static self-hosted build:

```powershell
pnpm build
pnpm preview -- --host 127.0.0.1 --port 4173
```

The generated `dist/` directory contains only static files and can be served by
Nginx, Caddy, Cloudflare Pages, S3, or any equivalent static host.
The runtime uses one WASM thread, so it does not require COOP/COEP headers. HTTPS
is required for WebGPU on non-localhost deployments; the WASM provider remains
the compatibility fallback.

Every visible tick is calculated on demand. Pause suspends stepping; Stop
creates a fresh simulation and holds it at step zero. No trajectory is
precomputed or replayed.

The `LAYOUT` control opens a fixed-angle overhead pallet editor aligned with one
grid axis. WASD pans the view and the wheel zooms without changing its angle.
Click or drag over storage cells, including the two outer rows, to add and remove
pallets; undo, reset, and clear tools make larger changes practical. The editor
checks loaded-pallet access before enabling apply. Applying a valid layout keeps
it in the current browser session, regenerates 1000 deterministic starts and
4,000 tasks, then restarts the Worker simulation. The agent slider runs from
1 to 1000, independent of pallet density: unloaded robots may start under
pallets, and robots wait for a free pallet when all are assigned.

## Optional native AOTI runtime

The original CUDA/AOTI simulator remains available for performance comparison
and native validation. On Windows, run it in WSL, then start Vite with
`VITE_MAPF_RUNTIME=server` so the UI uses the loopback WebSocket transport:

```bash
source /home/ubuntu/codex-mapf-export/.venv/bin/activate
python -m pip install -r runtime/requirements.txt
runtime/build_native.sh
runtime/run_demo.sh \
  artifacts/fastdmm-stage2-step12800-bf16-sm86-dynamic-N2-2580.pt2 \
  --agents 100
```

```powershell
$env:VITE_MAPF_RUNTIME='server'
pnpm dev -- --host 127.0.0.1 --port 4173
```

## Verified 100-agent lifelong run

The included 44×33 warehouse has 713 pallet cells, filling about 60% of
the storage grid while preserving load-bearing aisles. It also has 31 unloading
cells along one full inner edge and 31 batch-loading cells shifted to one end of
the opposite edge. The two cells at the other end hold the tow depot and repair
station. A deterministic FastDMM 0.8M + PIBT run on the current machine completed
this 600-step horizon as follows:

| Metric | Result |
| --- | ---: |
| status | lifelong horizon complete |
| completed tasks | 233 |
| native runtime | 4.88 s |
| mean AOTI inference | 4.66 ms |
| vertex collisions | 0 |
| edge-swap collisions | 0 |
| obstacle collisions | 0 |

Generated trajectories stay in ignored `runtime/runs/`. The browser protocol is
documented in [docs/protocol.md](docs/protocol.md).

## Runtime components

- `runtime/native-runner/`: standalone C++ MovingAI/FastDMM/PIBT runtime;
- `runtime/bridge.py`: loopback WebSocket server and binary frame encoder;
- `runtime/generate_lifelong_warehouse.py`: compact warehouse, starts, layout,
  and deterministic random task queue;
- `runtime/export_fastdmm_aoti.py`: machine-specific AOTI exporter;
- `runtime/export_fastdmm_onnx.py`: fixed-slot browser ONNX exporter and parity check;
- `assembly/mapf-core.ts`: WebAssembly BFS, observation tokenizer, and PIBT shield;
- `src/browser-runtime.worker.ts`: ONNX inference and lifelong browser simulation;
- `src/main.ts`: Babylon.js renderer, interpolation, controls, and telemetry.

Generated checkpoints and `.pt2` packages stay in ignored `artifacts/`. The
Hugging Face token is never sent to Vite or the browser. The browser loads only
the checked-in, inference-only ONNX artifact.
