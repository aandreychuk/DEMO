# Validation record

Validated on 2026-09-21 with:

- NVIDIA H100 80 GB;
- PyTorch `2.13.0+cu126`;
- GCC/G++ 11.4;
- CMake Release build with C++20;
- DMM-7M GRPO-500 AOTI package SHA-256
  `7814a770824dda213fb4b0a3e2936a275e44e68ebd6d7dad7051bb0d8118b280`.

The included three-agent smoke task completed with:

| Metric | Result |
| --- | ---: |
| status | solved |
| reached agents | 3 / 3 |
| episode steps | 3 |
| SoC | 8 |
| makespan | 3 |
| trajectory rows | 12 |
| vertex collisions | 0 |
| edge-swap collisions | 0 |

`run_smoke.sh` rebuilds the executable, runs this task, and independently
checks trajectory shape plus vertex and edge-swap collision freedom. This is a
native-path integration smoke test, not a claim of full benchmark parity.
