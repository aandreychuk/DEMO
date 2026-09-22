#!/usr/bin/env python3
"""Resize the fixed 100-agent ONNX policy to 1000 graph slots.

The exported graph contains agent-count constants and pre-expanded copies of
the learned empty message. Both must be resized; changing input shapes alone
produces an invalid graph. The model weights and computation are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx


def resize(source: Path, target: Path, agents: int) -> None:
    model = onnx.load(source)
    old_agents = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    if old_agents != 100:
        raise ValueError(f"expected a 100-agent source graph, got {old_agents}")

    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_value == old_agents:
                dimension.dim_value = agents
            elif dimension.dim_value == old_agents + 1:
                dimension.dim_value = agents + 1

    for index, initializer in enumerate(model.graph.initializer):
        array = onnx.numpy_helper.to_array(initializer)
        if old_agents in initializer.dims:
            axis = list(initializer.dims).index(old_agents)
            array = np.repeat(array, agents // old_agents, axis=axis)
        elif array.dtype.kind in "iu" and array.size < 20:
            if np.any((array == old_agents) | (array == old_agents + 1)):
                array = array.copy()
                array[array == old_agents] = agents
                array[array == old_agents + 1] = agents + 1
        model.graph.initializer[index].CopyFrom(
            onnx.numpy_helper.from_array(array, initializer.name)
        )

    if "-webgpu" not in source.stem:
        onnx.checker.check_model(model)
    onnx.save(model, target)
    print(f"{target}: {target.stat().st_size / 1024 / 1024:.2f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=1000)
    args = parser.parse_args()
    if args.agents < 100 or args.agents % 100:
        raise ValueError("agent count must be a multiple of 100")
    root = Path(__file__).resolve().parents[1] / "public" / "runtime"
    for suffix in ("", "-fp16", "-webgpu-fp16"):
        stem = f"fastdmm-0.8m{suffix}"
        resize(root / f"{stem}.onnx", root / f"{stem}-{args.agents}.onnx", args.agents)


if __name__ == "__main__":
    main()
