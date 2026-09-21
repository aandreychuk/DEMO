# MAPF Operations Deck

A browser-based 2.5D demonstration of the trained FastDMM policy. CUDA/AOTI
inference, observation construction, PIBT shielding, and collision checks run
locally in the native process. The browser receives compact state frames over a
loopback WebSocket and renders the 100-agent lifelong warehouse with Babylon.js
thin instances.

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
pallet and carries it into the service area. Its 3D reach targets the occupied
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

## Run the live demo

The checked-in native runtime targets Linux; on Windows, run it in WSL. Activate
the Python environment containing PyTorch `2.13.0+cu126`, then build and start
the local bridge:

```bash
source /home/ubuntu/codex-mapf-export/.venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -r runtime/requirements.txt
runtime/build_native.sh
runtime/run_demo.sh \
  artifacts/fastdmm-stage2-step12800-bf16-sm86-dynamic-N2-2580.pt2 \
  --agents 100
```

In a second terminal, start the browser client (Node.js 20.19 or newer):

```powershell
npm install
npm run dev -- --host 127.0.0.1 --port 4173
```

Open `http://127.0.0.1:4173/`. The agent selector starts a local simulator for
25, 50, or 100 agents. Every visible tick is computed on demand by the native
FastDMM + PIBT process. Pause suspends native stepping; Stop terminates the
current process, creates a fresh simulation, and holds it at step zero. Playback
speed, WASD movement, mouse orbit, and wheel zoom remain browser-side. If the bridge is unavailable, the page uses
clearly labelled synthetic motion while it retries the loopback connection.

## Verified 100-agent lifelong run

The included 44×32 warehouse has 160 pallet cells, 30 unloading cells along
one full inner edge, and 30 batch-loading cells along the opposite edge. A deterministic FastDMM 0.8M + PIBT run on
the current machine completed this
600-step horizon as follows:

| Metric | Result |
| --- | ---: |
| status | lifelong horizon complete |
| completed tasks | 602 |
| native runtime | 5.22 s |
| mean AOTI inference | 4.95 ms |
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
- `src/main.ts`: Babylon.js renderer, interpolation, controls, and telemetry.

Generated checkpoints and `.pt2` packages stay in ignored `artifacts/`. The
Hugging Face token is never sent to Vite or the browser.
