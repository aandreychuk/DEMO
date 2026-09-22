#!/usr/bin/env python3
"""Convert the browser FastDMM ONNX graph to FP16 and verify its actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_inputs(agent_count: int) -> list[dict[str, np.ndarray]]:
    observations = np.full((1, agent_count, 256), 66, dtype=np.int64)
    observations[:, :, :121] = np.arange(121, dtype=np.int64) % 16
    chat = np.full((1, agent_count, 13), -1, dtype=np.int64)
    ids = np.arange(agent_count, dtype=np.int64)
    chat[0, :, 0] = (ids - 1) % agent_count
    chat[0, :, 1] = (ids + 1) % agent_count

    samples = [(observations, chat)]
    random = np.random.default_rng(20260922)
    for _ in range(8):
        samples.append(
            (
                random.integers(0, 67, size=(1, agent_count, 256), dtype=np.int64),
                random.integers(-1, agent_count, size=(1, agent_count, 13), dtype=np.int64),
            )
        )

    feeds = []
    for sample_observations, sample_chat in samples:
        neighbor_padding = np.all(
            sample_observations[:, :, 121:251].reshape(1, agent_count, 13, 10) == 66,
            axis=-1,
        )
        feeds.append(
            {
                "observations": sample_observations,
                "chat": sample_chat,
                "neighbor_padding": neighbor_padding,
            }
        )
    return feeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=100)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("FP16 output must not overwrite the FP32 source model")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = float16.convert_float_to_float16(
        onnx.load(input_path),
        min_positive_val=5.9604645e-8,
        max_finite_val=65504.0,
        keep_io_types=True,
    )
    onnx.checker.check_model(converted)
    onnx.save(converted, output_path)

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    reference_session = ort.InferenceSession(
        str(input_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    fp16_session = ort.InferenceSession(
        str(output_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )

    feeds_to_check = validation_inputs(args.agents)
    max_abs_error = 0.0
    absolute_error_sum = 0.0
    probability_count = 0
    max_sum_error = 0.0
    matching_actions = 0
    action_count = 0
    for feeds in feeds_to_check:
        reference = reference_session.run(["action_probabilities"], feeds)[0]
        actual = fp16_session.run(["action_probabilities"], feeds)[0]
        if actual.shape != (args.agents, 5) or actual.dtype != np.float32:
            raise RuntimeError(f"unexpected FP16 graph output: {actual.dtype} {actual.shape}")
        if not np.isfinite(actual).all():
            raise RuntimeError("FP16 graph produced non-finite probabilities")
        max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - reference))))
        absolute_error_sum += float(np.sum(np.abs(actual - reference)))
        probability_count += actual.size
        max_sum_error = max(max_sum_error, float(np.max(np.abs(actual.sum(-1) - 1.0))))
        matching_actions += int(np.count_nonzero(actual.argmax(-1) == reference.argmax(-1)))
        action_count += args.agents

    action_agreement = matching_actions / action_count
    mean_abs_error = absolute_error_sum / probability_count
    if action_agreement < 0.995 or mean_abs_error > 1e-2 or max_sum_error > 1e-3:
        raise RuntimeError(
            "FP16 validation failed: "
            f"mean_abs={mean_abs_error}, agreement={action_agreement}, sum_error={max_sum_error}"
        )

    source_metadata_path = input_path.with_suffix(input_path.suffix + ".json")
    metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "precision": "float16",
            "source_onnx": input_path.name,
            "source_onnx_sha256": sha256(input_path),
            "onnx_size_bytes": output_path.stat().st_size,
            "onnx_sha256": sha256(output_path),
            "validation": {
                "provider": "CPUExecutionProvider",
                "samples": len(feeds_to_check),
                "max_abs_error_vs_fp32": max_abs_error,
                "mean_abs_error_vs_fp32": mean_abs_error,
                "action_agreement_vs_fp32": action_agreement,
                "max_probability_sum_error": max_sum_error,
            },
        }
    )
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"validated FP16: mean_abs={mean_abs_error:.3g}, max_abs={max_abs_error:.3g}, "
        f"action_agreement={action_agreement:.4f}; "
        f"size={output_path.stat().st_size / 1024 / 1024:.2f} MiB"
    )


if __name__ == "__main__":
    main()
