#!/usr/bin/env python3
"""Export the archived FastDMM 0.8M policy as a local CUDA AOTI package."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
import time

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "fastdmm"))

from fast_dmm import FastDMM, FastDMMConfig  # noqa: E402


class _PrecisionWrapper(nn.Module):
    """Keep FP32 weights while using the checkpoint's BF16 inference recipe."""

    def __init__(self, model: FastDMM, precision: str) -> None:
        super().__init__()
        self.model = model
        self.precision = precision

    def forward(self, observations: Tensor, chat: Tensor) -> Tensor:
        if self.precision == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                consensus = self.model.deterministic_zero_act(observations, chat)
        else:
            consensus = self.model.deterministic_zero_act(observations, chat)
        # The native runner consumes one unbatched row per agent. Keep the
        # final normalization in FP32, matching the archived stage-2 package.
        return torch.softmax(consensus[0], dim=-1)


class FastDMMAOTI(nn.Module):
    def __init__(self, model: FastDMM, precision: str) -> None:
        super().__init__()
        self.inner = _PrecisionWrapper(model, precision)

    def forward(self, observations: Tensor, chat: Tensor) -> Tensor:
        return self.inner(observations, chat)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(checkpoint: dict) -> dict[str, Tensor]:
    state = checkpoint["model"]
    cleaned: dict[str, Tensor] = {}
    prefixes = ("_orig_mod.net.", "_orig_mod.", "net.")
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        cleaned[key] = value
    return cleaned


def sample_inputs(num_agents: int, device: torch.device) -> tuple[Tensor, Tensor]:
    observations = torch.full(
        (1, num_agents, 256), 66, dtype=torch.int64, device=device
    )
    # Exercise the encoder with deterministic, valid vocabulary IDs while
    # keeping all optional neighbour feature records padded.
    grid = torch.arange(121, device=device).remainder(16)
    observations[:, :, :121] = grid

    chat = torch.full(
        (1, num_agents, 13), -1, dtype=torch.int64, device=device
    )
    ids = torch.arange(num_agents, device=device)
    chat[0, :, 0] = (ids - 1).remainder(num_agents)
    chat[0, :, 1] = (ids + 1).remainder(num_agents)
    return observations.contiguous(), chat.contiguous()


def parse_agents(value: str) -> list[int]:
    agents = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not agents or any(item < 2 for item in agents):
        raise argparse.ArgumentTypeError("agent counts must be comma-separated integers >= 2")
    return agents


def precision_context(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def configure_cuda_toolkit() -> None:
    """Point AOTI at CUDA headers/libs shipped with the PyTorch environment."""
    site_packages = Path(torch.__file__).resolve().parent.parent
    runtime_root = site_packages / "nvidia" / "cuda_runtime"
    nvcc_root = site_packages / "nvidia" / "cuda_nvcc"
    cccl_root = site_packages / "nvidia" / "cuda_cccl"
    if "CUDA_HOME" not in os.environ:
        if not (runtime_root / "include" / "cuda.h").is_file():
            raise RuntimeError(
                "CUDA_HOME is unset and nvidia/cuda_runtime was not found in the environment"
            )
        os.environ["CUDA_HOME"] = str(runtime_root)

    # The runtime wheel has the public CUDA headers and libcudart; the matching
    # nvcc wheel supplies the `crt/` headers and libdevice used while building
    # the AOTI C++ wrapper and Triton kernels.
    nvcc_include = nvcc_root / "include"
    if not (nvcc_include / "crt" / "host_defines.h").is_file():
        raise RuntimeError(
            "nvidia-cuda-nvcc-cu12 is required (tested with version 12.6.85)"
        )
    cccl_include = cccl_root / "include"
    if not (cccl_include / "nv" / "target").is_file():
        raise RuntimeError(
            "nvidia-cuda-cccl-cu12 is required (tested with version 12.6.77)"
        )
    current_include_path = os.environ.get("CPLUS_INCLUDE_PATH", "")
    include_parts = [str(nvcc_include), str(cccl_include)]
    if current_include_path:
        include_parts.append(current_include_path)
    os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join(include_parts)

    # WSL exposes the host NVIDIA driver here, but its path is not always in
    # g++'s link search list even though PyTorch can load libcuda.so.1.
    wsl_driver = Path("/usr/lib/wsl/lib")
    if (wsl_driver / "libcuda.so").is_file():
        for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            current = os.environ.get(variable, "")
            parts = [str(wsl_driver)]
            if current:
                parts.append(current)
            os.environ[variable] = os.pathsep.join(parts)

    libdevice = nvcc_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if "TRITON_LIBDEVICE_PATH" not in os.environ and libdevice.is_file():
        os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FastDMM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_args" not in checkpoint or "model" not in checkpoint:
        raise ValueError("checkpoint must contain model_args and model")
    model = FastDMM(FastDMMConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.eval().to(device)
    return model, checkpoint


def validate_package(
    package_path: Path,
    eager: FastDMMAOTI,
    agent_counts: list[int],
    max_agents: int,
    device: torch.device,
) -> list[dict]:
    compiled = torch._inductor.aoti_load_package(
        str(package_path), run_single_threaded=True, device_index=device.index or 0
    )
    results: list[dict] = []
    for count in agent_counts:
        if count > max_agents:
            raise ValueError(f"validation count {count} exceeds max {max_agents}")
        observations, chat = sample_inputs(count, device)
        torch.cuda.synchronize(device)
        with torch.inference_mode(), precision_context(eager.inner.precision):
            expected = eager(observations, chat)
            # Exclude per-shape lazy initialization from the steady-state
            # policy latency stored in the package manifest.
            actual = compiled(observations, chat)
            torch.cuda.synchronize(device)
            timings = []
            for _ in range(5):
                started = time.perf_counter()
                actual = compiled(observations, chat)
                torch.cuda.synchronize(device)
                timings.append((time.perf_counter() - started) * 1000.0)
        elapsed_ms = median(timings)
        if actual.shape != (count, 5):
            raise RuntimeError(f"unexpected output shape for N={count}: {actual.shape}")
        if actual.dtype != torch.float32 or not torch.isfinite(actual).all():
            raise RuntimeError(f"invalid output for N={count}: {actual.dtype}")
        max_abs = float((actual - expected).abs().max())
        action_agreement = float(
            (actual.argmax(-1) == expected.argmax(-1)).float().mean()
        )
        probability_error = float((actual.sum(-1) - 1.0).abs().max())
        results.append(
            {
                "agents": count,
                "latency_ms": round(elapsed_ms, 3),
                "max_abs_error_vs_eager": max_abs,
                "action_agreement": action_agreement,
                "max_probability_sum_error": probability_error,
            }
        )
        print(
            f"validated N={count}: {elapsed_ms:.2f} ms, "
            f"max_abs={max_abs:.3g}, action_agreement={action_agreement:.3f}",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-num-agents", type=int, default=2580)
    parser.add_argument("--example-agents", type=int, default=8)
    parser.add_argument(
        "--validate-agents", type=parse_agents, default=parse_agents("2,8,64")
    )
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    if not torch.__version__.startswith("2.13."):
        raise RuntimeError(f"PyTorch 2.13 is required, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this AOTI export")
    configure_cuda_toolkit()
    if not 2 <= args.example_agents <= args.max_num_agents:
        raise ValueError("example-agents must be between 2 and max-num-agents")
    if args.max_num_agents < 2:
        raise ValueError("max-num-agents must be at least 2")

    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.suffix != ".pt2":
        raise ValueError("output must end in .pt2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    Path(str(output_path) + ".json").unlink(missing_ok=True)

    device = torch.device("cuda", torch.cuda.current_device())
    capability = list(torch.cuda.get_device_capability(device))
    gpu_name = torch.cuda.get_device_name(device)
    print(
        f"loading {checkpoint_path.name}; torch={torch.__version__}; "
        f"gpu={gpu_name}; capability=sm_{capability[0]}{capability[1]}",
        flush=True,
    )
    model, checkpoint = load_model(checkpoint_path, device)
    wrapper = FastDMMAOTI(model, args.precision).eval()
    observations, chat = sample_inputs(args.example_agents, device)
    agents = torch.export.Dim(
        "agents", min=2, max=args.max_num_agents
    )

    with torch.inference_mode(False), torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (observations, chat),
            dynamic_shapes=({1: agents}, {1: agents}),
            strict=True,
        )

    configs = {
        "shape_padding": False,
        "triton.cudagraphs": False,
    }
    started = time.perf_counter()
    print(f"compiling dynamic AOTI package to {output_path}", flush=True)
    torch._inductor.aoti_compile_and_package(
        exported, package_path=str(output_path), inductor_configs=configs
    )
    build_seconds = time.perf_counter() - started

    validations: list[dict] = []
    if not args.skip_validation:
        validations = validate_package(
            output_path,
            wrapper,
            args.validate_agents,
            args.max_num_agents,
            device,
        )

    checkpoint_hash = sha256(checkpoint_path)
    package_hash = sha256(output_path)
    metadata = {
        "schema_version": 1,
        "architecture": "FastDMM-0.8M",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint.get("val_step", checkpoint.get("iter_num")),
        "input_contract": "obs-chat-v1",
        "inputs": [
            {"name": "observations", "dtype": "int64", "shape": [1, "N", 256]},
            {"name": "chat", "dtype": "int64", "shape": [1, "N", 13]},
        ],
        "output": {
            "name": "action_probabilities",
            "dtype": "float32",
            "shape": ["N", 5],
        },
        "policy_mode": "deterministic-zero-argmax-rounds",
        "precision": args.precision,
        "agent_range": {"min": 2, "max": args.max_num_agents},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "compute_capability": capability,
        "platform": platform.platform(),
        "compiler": {
            "cc": os.environ.get("CC"),
            "cxx": os.environ.get("CXX"),
            "inductor_configs": configs,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "package_size_bytes": output_path.stat().st_size,
        "package_sha256": package_hash,
        "build_wall_seconds": round(build_seconds, 3),
        "validation": validations,
    }
    sidecar_path = Path(str(output_path) + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"package sha256: {package_hash}", flush=True)
    print(f"sidecar: {sidecar_path}", flush=True)


if __name__ == "__main__":
    main()
