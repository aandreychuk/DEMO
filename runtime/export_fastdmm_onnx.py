#!/usr/bin/env python3
"""Export the FastDMM browser policy as a fixed-size ONNX graph.

The browser simulator always allocates 100 policy slots. Smaller simulations
pad the unused slots with empty observations and never reference them from the
chat graph. A fixed shape keeps WebGPU graph compilation predictable and avoids
shipping one model per UI agent-count preset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from export_fastdmm_aoti import load_model, sample_inputs, sha256


class FastDMMOnnx(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, observations: Tensor, chat: Tensor) -> Tensor:
        consensus = self.model.deterministic_zero_act(observations, chat)
        return torch.softmax(consensus[0], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--opset", type=int, default=20)
    args = parser.parse_args()

    if args.agents < 2:
        raise ValueError("agents must be at least 2")
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    wrapper = FastDMMOnnx(model).eval()
    observations, chat = sample_inputs(args.agents, device)

    with torch.inference_mode():
        expected = wrapper(observations, chat).cpu().numpy()

    print(f"exporting fixed-N={args.agents} policy to {output_path}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (observations, chat),
            output_path,
            input_names=("observations", "chat"),
            output_names=("action_probabilities",),
            opset_version=args.opset,
            dynamo=True,
            external_data=False,
            # The PyTorch 2.13 ONNX optimizer currently folds
            # ``log(one_hot + 1e-8)`` into ``log(one_hot)`` in this graph,
            # producing -inf and NaN consensus values. Preserve the explicit
            # epsilon until that upstream optimization is safe.
            optimize=False,
        )

    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        ["action_probabilities"],
        {
            "observations": observations.cpu().numpy(),
            "chat": chat.cpu().numpy(),
        },
    )[0]
    max_abs = float(np.max(np.abs(actual - expected)))
    agreement = float(np.mean(actual.argmax(-1) == expected.argmax(-1)))
    sums = float(np.max(np.abs(actual.sum(-1) - 1.0)))
    if actual.shape != (args.agents, 5):
        raise RuntimeError(f"unexpected ONNX output shape: {actual.shape}")
    if agreement < 1.0 or max_abs > 1e-4 or sums > 1e-5:
        raise RuntimeError(
            "ONNX validation failed: "
            f"max_abs={max_abs}, agreement={agreement}, sum_error={sums}"
        )

    metadata = {
        "schema": "fastdmm-browser-policy/v1",
        "architecture": "FastDMM-0.8M",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint.get("val_step", checkpoint.get("iter_num")),
        "agent_slots": args.agents,
        "inputs": [
            {"name": "observations", "dtype": "int64", "shape": [1, args.agents, 256]},
            {"name": "chat", "dtype": "int64", "shape": [1, args.agents, 13]},
        ],
        "output": {
            "name": "action_probabilities",
            "dtype": "float32",
            "shape": [args.agents, 5],
        },
        "opset": args.opset,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "onnx_size_bytes": output_path.stat().st_size,
        "onnx_sha256": sha256(output_path),
        "validation": {
            "provider": "CPUExecutionProvider",
            "max_abs_error": max_abs,
            "action_agreement": agreement,
            "max_probability_sum_error": sums,
        },
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"validated: max_abs={max_abs:.3g}, action_agreement={agreement:.3f}; "
        f"size={output_path.stat().st_size / 1024 / 1024:.2f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
