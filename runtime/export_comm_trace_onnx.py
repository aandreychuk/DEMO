#!/usr/bin/env python3
"""Export the real per-round FastDMM consensus for the browser maze lab."""

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


class CommunicationTrace(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, observations: Tensor, chat: Tensor, neighbor_padding: Tensor
    ) -> Tensor:
        model = self.model
        batch, agents, _ = observations.shape
        key, value, padding, _ = model._encode(observations, neighbor_padding)
        dtype = key.dtype
        z = torch.zeros(batch * agents, 5, dtype=dtype, device=observations.device)
        h = model.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        rounds = []
        for _ in range(model.config.n_comm_rounds):
            logits, h = model._run_round(z, h, chat, key, value, padding)
            # Match deterministic_zero_act's optimized deployment path exactly.
            logits = torch.where(logits.eq(logits), logits, 0.0)
            vote = logits.argmax(dim=-1)
            target = model._centred_log_ohe(vote, dtype)
            z = z + model.config.dt * (target - z)
            rounds.append(torch.softmax(z.reshape(batch, agents, 5)[0], dim=-1))
        return torch.stack(rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=8)
    args = parser.parse_args()
    if args.agents < 2:
        raise ValueError("at least two agents are required")

    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    observations, chat = sample_inputs(args.agents, torch.device("cpu"))
    neighbor_padding = observations[:, :, 121:251].reshape(
        1, args.agents, 13, 10
    ).eq(66).all(-1)
    reference = torch.softmax(
        model.deterministic_zero_act(observations, chat)[0], dim=-1
    ).detach().numpy()
    for module in model.modules():
        if hasattr(module, "onnx_explicit_attention"):
            module.onnx_explicit_attention = True
    model.onnx_simplify_consensus = True
    wrapper = CommunicationTrace(model).eval()
    with torch.inference_mode():
        expected = wrapper(observations, chat, neighbor_padding).numpy()
        torch.onnx.export(
            wrapper,
            (observations, chat, neighbor_padding),
            output_path,
            input_names=("observations", "chat", "neighbor_padding"),
            output_names=("round_intents",),
            opset_version=20,
            dynamo=True,
            external_data=False,
            optimize=True,
        )

    onnx.checker.check_model(onnx.load(output_path))
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {
        "observations": observations.numpy(),
        "chat": chat.numpy(),
        "neighbor_padding": neighbor_padding.numpy(),
    })[0]
    if actual.shape != (model.config.n_comm_rounds, args.agents, 5):
        raise RuntimeError(f"unexpected trace shape: {actual.shape}")
    max_abs = float(np.max(np.abs(actual - expected)))
    final_max_abs = float(np.max(np.abs(actual[-1] - reference)))
    agreement = float(np.mean(actual[-1].argmax(-1) == reference.argmax(-1)))
    if max_abs > 1e-4 or final_max_abs > 1e-4 or agreement < 1.0:
        raise RuntimeError(
            f"trace mismatch: ONNX={max_abs}, final={final_max_abs}, agreement={agreement}"
        )
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps({
            "schema": "fastdmm-communication-trace/v1",
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_step": checkpoint.get("val_step", checkpoint.get("iter_num")),
            "agents": args.agents,
            "rounds": model.config.n_comm_rounds,
            "output": "round_intents[round, agent, action]",
            "max_abs_error": max_abs,
            "final_max_abs_error": final_max_abs,
            "final_action_agreement": agreement,
            "onnx_sha256": sha256(output_path),
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"validated {args.agents} agents, {model.config.n_comm_rounds} rounds; "
        f"max error {max_abs:.2g}; size {output_path.stat().st_size / 1024 / 1024:.2f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
