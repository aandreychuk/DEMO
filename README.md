# MAPF Operations Deck

A browser-based 2.5D visualizer for large multi-agent pathfinding runs. The
current prototype renders synthetic warehouse traffic so visual performance can
be evaluated before connecting the trained model.

## Run the visual prototype

Requirements: Node.js 20.19 or newer.

```powershell
npm install
npm run dev
```

Open the local address printed by Vite. The HUD can switch between 1,000, 2,500,
and 5,000 agents.

## Intended runtime architecture

- C++ loads the exported PyTorch model through AOTI and runs inference on CUDA.
- A loopback-only WebSocket bridge publishes compact state frames.
- Babylon.js renders thin instances and interpolates between discrete steps.
- The Hugging Face token is supplied only to the local runtime through
  `HF_TOKEN`; it must not be exposed through Vite or a browser message.

The wire format is documented in [docs/protocol.md](docs/protocol.md).
