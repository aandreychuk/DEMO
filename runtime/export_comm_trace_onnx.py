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
from fast_dmm import ExplicitSDPASelfAttention, QueryCrossAttentionBlock


class CommunicationTrace(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, observations: Tensor, chat: Tensor
    ) -> tuple[Tensor, Tensor]:
        model = self.model
        batch, agents, _ = observations.shape
        key, value, padding, _ = model._encode(observations)
        dtype = key.dtype
        z = torch.zeros(batch * agents, 5, dtype=dtype, device=observations.device)
        h = model.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        rounds = []
        votes = []
        for _ in range(model.config.n_comm_rounds):
            logits, h = model._run_round(z, h, chat, key, value, padding)
            # Match deterministic_zero_act's optimized deployment path exactly.
            logits = torch.where(logits.eq(logits), logits, 0.0)
            vote = logits.argmax(dim=-1)
            votes.append(vote.reshape(batch, agents)[0].to(torch.float32))
            # This is algebraically the centred log of a smoothed one-hot.
            # Writing it directly keeps ONNX optimization from dropping the
            # 1e-8 before Log and turning four actions into -inf.
            target = (torch.nn.functional.one_hot(vote, 5).to(dtype) - 0.2) * 18.420680743952367
            z = z + model.config.dt * (target - z)
            rounds.append(z.reshape(batch, agents, 5)[0])
        return torch.stack(rounds), torch.stack(votes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=16)
    args = parser.parse_args()
    if args.agents < 1:
        raise ValueError("at least one model slot is required")

    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    observations, chat = sample_inputs(args.agents, torch.device("cpu"))
    reference = model.deterministic_zero_act(observations, chat)[0].detach().numpy()
    for module in model.modules():
        if isinstance(module, (ExplicitSDPASelfAttention, QueryCrossAttentionBlock)):
            module.onnx_explicit_attention = True
    wrapper = CommunicationTrace(model).eval()
    with torch.inference_mode():
        expected_z, expected_votes = wrapper(observations, chat)
        expected_z = expected_z.numpy()
        expected_votes = expected_votes.numpy()
        torch.onnx.export(
            wrapper,
            (observations, chat),
            output_path,
            input_names=("observations", "chat"),
            output_names=("round_z", "round_votes"),
            opset_version=20,
            dynamo=True,
            external_data=False,
            optimize=True,
        )

    onnx.checker.check_model(onnx.load(output_path))
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual_z, actual_votes = session.run(None, {
        "observations": observations.numpy(),
        "chat": chat.numpy(),
    })
    if actual_z.shape != (model.config.n_comm_rounds, args.agents, 5):
        raise RuntimeError(f"unexpected z shape: {actual_z.shape}")
    if actual_votes.shape != (model.config.n_comm_rounds, args.agents):
        raise RuntimeError(f"unexpected votes shape: {actual_votes.shape}")
    max_abs = float(np.max(np.abs(actual_z - expected_z)))
    final_max_abs = float(np.max(np.abs(actual_z[-1] - reference)))
    vote_agreement = float(np.mean(actual_votes == expected_votes))
    agreement = float(np.mean(actual_z[-1].argmax(-1) == reference.argmax(-1)))
    if max_abs > 1e-4 or final_max_abs > 1e-4 or agreement < 1.0 or vote_agreement < 1.0:
        raise RuntimeError(
            f"trace mismatch: ONNX={max_abs}, final={final_max_abs}, "
            f"actions={agreement}, votes={vote_agreement}"
        )
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps({
            "schema": "fastdmm-communication-z-and-votes/v3",
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_step": checkpoint.get("val_step", checkpoint.get("iter_num")),
            "agents": args.agents,
            "rounds": model.config.n_comm_rounds,
            "outputs": ["round_z[round, agent, action]", "round_votes[round, agent]"],
            "max_abs_error": max_abs,
            "final_max_abs_error": final_max_abs,
            "final_action_agreement": agreement,
            "round_vote_agreement": vote_agreement,
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
