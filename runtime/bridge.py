#!/usr/bin/env python3
"""Run the local AOTI MAPF solver and replay its states over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed


MAGIC = 0x4D415046
PROTOCOL = 2
MAX_AGENTS = 100


@dataclass(frozen=True)
class AgentState:
    x: int
    y: int
    loaded: bool
    task_stage: int
    task_id: int
    pallet_id: int
    station_x: int
    station_y: int
    completed_tasks: int


@dataclass
class Episode:
    frames: list[list[AgentState]]
    summary: dict[str, Any]
    planning_seconds: float


def load_trajectory(path: Path, agent_count: int) -> list[list[AgentState]]:
    frames: dict[int, list[AgentState | None]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            step = int(row["step"])
            agent = int(row["agent"])
            if agent >= agent_count:
                continue
            frame = frames.setdefault(step, [None] * agent_count)
            frame[agent] = AgentState(
                x=int(row["x"]),
                y=int(row["y"]),
                loaded=bool(int(row.get("loaded", "0"))),
                task_stage=int(row.get("task_stage", "-1")),
                task_id=int(row.get("task_id", "-1")),
                pallet_id=int(row.get("pallet_id", "-1")),
                station_x=int(row.get("station_x", "-1")),
                station_y=int(row.get("station_y", "-1")),
                completed_tasks=int(row.get("completed_tasks", "0")),
            )
    result: list[list[AgentState]] = []
    for step in sorted(frames):
        if any(position is None for position in frames[step]):
            raise RuntimeError(f"trajectory step {step} is incomplete")
        result.append([position for position in frames[step] if position is not None])
    if not result:
        raise RuntimeError("native runner returned an empty trajectory")
    return result


def encode_frame(
    step: int,
    states: list[AgentState],
    previous: list[AgentState] | None,
    final: bool = False,
) -> bytes:
    record_size = 24
    payload = bytearray(16 + record_size * len(states))
    completed_tasks = min(65535, states[0].completed_tasks if states else 0)
    struct.pack_into("<IHHII", payload, 0, MAGIC, PROTOCOL, completed_tasks, step, len(states))
    for agent, state in enumerate(states):
        waiting = previous is not None and (previous[agent].x, previous[agent].y) == (state.x, state.y)
        transitioned = previous is not None and (
            previous[agent].task_stage != state.task_stage
            or previous[agent].pallet_id != state.pallet_id
        )
        status = (1 if waiting else 0) | (2 if state.loaded else 0) | (4 if transitioned else 0)
        if final:
            status |= 8
        pallet_id = 65535 if state.pallet_id < 0 else min(65534, state.pallet_id)
        task_id = 0xFFFFFFFF if state.task_id < 0 else min(0xFFFFFFFE, state.task_id)
        struct.pack_into(
            "<IffBBHIhh", payload, 16 + agent * record_size,
            agent, float(state.x), float(state.y), status,
            max(0, state.task_stage), pallet_id, task_id,
            state.station_x, state.station_y,
        )
    return bytes(payload)


class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cache: dict[int, Episode] = {}
        self.run_lock = asyncio.Lock()

    async def solve(self, count: int) -> Episode:
        if count in self.cache:
            return self.cache[count]
        async with self.run_lock:
            if count in self.cache:
                return self.cache[count]
            run_dir = self.args.run_dir.resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            prefix = run_dir / f"warehouse-{count}"
            for suffix in (".summary.json", ".trajectory.tsv", ".decisions.tsv"):
                (Path(str(prefix) + suffix)).unlink(missing_ok=True)

            command = [
                str(self.args.runner.resolve()),
                "--map", str(self.args.map.resolve()),
                "--scen", str(self.args.scenario.resolve()),
                "--model", str(self.args.model.resolve()),
                "--num-agents", str(count),
                "--max-steps", str(self.args.max_steps),
                "--mode", "pibt",
                "--sampling", "deterministic",
                "--lifelong-tasks", str(self.args.tasks.resolve()),
                "--output-prefix", str(prefix),
            ]
            env = os.environ.copy()
            try:
                import torch

                torch_lib = str(Path(torch.__file__).parent / "lib")
                env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            except ImportError:
                pass
            env.setdefault("OMP_NUM_THREADS", str(self.args.threads))

            started = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                output = stdout.decode("utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"native runner failed with exit {process.returncode}:\n{output}")

            summary_path = Path(str(prefix) + ".summary.json")
            trajectory_path = Path(str(prefix) + ".trajectory.tsv")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            episode = Episode(
                frames=load_trajectory(trajectory_path, count),
                summary=summary,
                planning_seconds=time.perf_counter() - started,
            )
            self.cache[count] = episode
            return episode

    async def handler(self, socket: ServerConnection) -> None:
        try:
            await self._handle_connection(socket)
        except ConnectionClosed:
            pass

    async def _handle_connection(self, socket: ServerConnection) -> None:
        count = self.args.agents
        while True:
            await socket.send(json.dumps({"type": "status", "state": "planning", "agents": count}))
            try:
                episode = await self.solve(count)
            except Exception as error:
                await socket.send(json.dumps({"type": "error", "message": str(error)}))
                return

            summary = episode.summary
            calls = max(1, int(summary.get("policy_forward_calls", 1)))
            await socket.send(json.dumps({
                "type": "hello",
                "protocol": PROTOCOL,
                "map": {
                    "width": int(self.args.layout_data["width"]),
                    "height": int(self.args.layout_data["height"]),
                    "cellSize": 1,
                },
                "layout": self.args.layout_data,
                "lifelong": True,
                "looping": True,
                "agents": count,
                "tickRate": self.args.tick_rate,
                "frames": len(episode.frames),
                "planningSeconds": episode.planning_seconds,
                "summary": summary,
                "inferenceMs": float(summary.get("policy_forward_ms", 0)) / calls,
            }))

            index = 0
            paused = False
            speed = 1.0
            reload_count: int | None = None
            previous: list[AgentState] | None = None
            while reload_count is None:
                if not paused:
                    if index >= len(episode.frames):
                        await socket.send(json.dumps({"type": "status", "state": "loop"}))
                        index = 0
                        previous = None
                        continue
                    frame = episode.frames[index]
                    final = index == len(episode.frames) - 1 and bool(summary.get("solved"))
                    await socket.send(encode_frame(index, frame, previous, final))
                    previous = frame
                    index += 1
                    await asyncio.sleep(1.0 / max(1.0, self.args.tick_rate * speed))

                timeout = 0.001 if not paused else None
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message.get("type") != "control":
                    continue
                action = message.get("action")
                if action == "pause":
                    paused = True
                elif action == "stop":
                    index = 0
                    previous = None
                    paused = True
                    frame = episode.frames[index]
                    await socket.send(encode_frame(index, frame, previous, False))
                    previous = frame
                    index += 1
                    await socket.send(json.dumps({"type": "status", "state": "stopped"}))
                elif action == "run":
                    if index >= len(episode.frames):
                        index = 0
                        previous = None
                    paused = False
                elif action == "step" and paused:
                    if index >= len(episode.frames):
                        index = 0
                        previous = None
                    frame = episode.frames[index]
                    final = index == len(episode.frames) - 1 and bool(summary.get("solved"))
                    await socket.send(encode_frame(index, frame, previous, final))
                    previous = frame
                    index += 1
                elif action == "speed":
                    speed = min(3.0, max(0.25, float(message.get("value", 1))))
                elif action == "load":
                    requested = int(message.get("agents", count))
                    reload_count = min(MAX_AGENTS, max(2, requested))
            count = reload_count


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=root / "native-runner" / "build" / "dmm_native_runner")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=root / "scenarios" / "warehouse-lifelong-44x32.map")
    parser.add_argument("--scenario", type=Path, default=root / "scenarios" / "warehouse-lifelong-44x32.scen")
    parser.add_argument("--tasks", type=Path, default=root / "scenarios" / "warehouse-lifelong-44x32.tasks.tsv")
    parser.add_argument("--layout", type=Path, default=root / "scenarios" / "warehouse-lifelong-44x32.layout.json")
    parser.add_argument("--run-dir", type=Path, default=root / "runs")
    parser.add_argument("--agents", type=int, choices=(25, 50, 100), default=100)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--tick-rate", type=float, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    for path in (args.runner, args.model, args.map, args.scenario, args.tasks, args.layout):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")
    args.layout_data = json.loads(args.layout.read_text(encoding="utf-8"))
    bridge = Bridge(args)
    async with serve(bridge.handler, args.host, args.port, max_size=1 << 20):
        print(f"MAPF bridge listening on ws://{args.host}:{args.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
