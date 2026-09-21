# MAPF Operations Deck

A browser-based 2.5D demonstration of the trained FastDMM policy. CUDA/AOTI
inference, observation construction, PIBT shielding, and collision checks run
locally in the native process. The browser receives compact state frames over a
loopback WebSocket and renders up to 2,500 agents with Babylon.js thin instances.

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
  --agents 1000
```

In a second terminal, start the browser client (Node.js 20.19 or newer):

```powershell
npm install
npm run dev -- --host 127.0.0.1 --port 4173
```

Open `http://127.0.0.1:4173/`. The agent selector starts or reuses a local run
for 64, 256, 1,000, or 2,500 agents. Pause, replay, playback speed, orbit, and
zoom controls remain browser-side. If the bridge is unavailable, the page uses
clearly labelled synthetic motion while it retries the loopback connection.

## Verified 1,000-agent run

The included deterministic warehouse scenario has 5,120 traversable cells. On
the current machine, FastDMM 0.8M with PIBT solved the 1,000-agent instance in
139 steps:

| Metric | Result |
| --- | ---: |
| status | solved |
| reached agents | 1,000 / 1,000 |
| native runtime | 3.91 s |
| mean AOTI inference | 22.63 ms |
| sum of costs | 63,396 |
| vertex collisions | 0 |
| edge-swap collisions | 0 |
| obstacle collisions | 0 |

Generated trajectories stay in ignored `runtime/runs/`. The browser protocol is
documented in [docs/protocol.md](docs/protocol.md).

## Runtime components

- `runtime/native-runner/`: standalone C++ MovingAI/FastDMM/PIBT runtime;
- `runtime/bridge.py`: loopback WebSocket server and binary frame encoder;
- `runtime/generate_warehouse.py`: shared 90×70 warehouse map and scenario;
- `runtime/export_fastdmm_aoti.py`: machine-specific AOTI exporter;
- `src/main.ts`: Babylon.js renderer, interpolation, controls, and telemetry.

Generated checkpoints and `.pt2` packages stay in ignored `artifacts/`. The
Hugging Face token is never sent to Vite or the browser.
