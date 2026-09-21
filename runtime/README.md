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
