#!/usr/bin/env python3
"""Drive the local AOTI MAPF simulator and stream live states over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed


MAGIC = 0x4D415046
PROTOCOL = 5
MAX_AGENTS = 100
LAYOUT_SEED = 800_000
TASK_COUNT = 4_000


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
    reload_x: int
    reload_y: int
    pallet_items: int
    reload_required: bool
    failed: bool
    recovery_state: int
    pallet_x: int
    pallet_y: int
    completed_tasks: int


@dataclass(frozen=True)
class TowState:
    x: int
    y: int
    state: int
    target_agent: int
    queued_rescues: int


@dataclass(frozen=True)
class NativeFrame:
    step: int
    states: list[AgentState]
    inference_ms: float
    tow: TowState


def parse_native_frame(line: str, expected_agents: int) -> NativeFrame | None:
    if not line.startswith("MAPF_FRAME\t"):
        return None
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 5:
        raise RuntimeError("native simulator emitted a malformed frame header")
    step = int(fields[1])
    completed_tasks = int(fields[2])
    inference_ms = float(fields[3])
    agent_count = int(fields[4])
    if agent_count != expected_agents or len(fields) != 6 + agent_count:
        raise RuntimeError(
            f"native simulator frame has {agent_count} agents, expected {expected_agents}"
        )
    states: list[AgentState] = []
    for record in fields[5 : 5 + agent_count]:
        values = [int(value) for value in record.split(",")]
        if len(values) != 16:
            raise RuntimeError("native simulator emitted a malformed agent record")
        states.append(
            AgentState(
                x=values[0],
                y=values[1],
                loaded=bool(values[2]),
                task_stage=values[3],
                task_id=values[4],
                pallet_id=values[5],
                station_x=values[6],
                station_y=values[7],
                reload_x=values[8],
                reload_y=values[9],
                pallet_items=values[10],
                reload_required=bool(values[11]),
                failed=bool(values[12]),
                recovery_state=values[13],
                pallet_x=values[14],
                pallet_y=values[15],
                completed_tasks=completed_tasks,
            )
        )
    tow_values = [int(value) for value in fields[-1].split(",")]
    if len(tow_values) != 5:
        raise RuntimeError("native simulator emitted a malformed tow record")
    tow = TowState(
        x=tow_values[0],
        y=tow_values[1],
        state=tow_values[2],
        target_agent=tow_values[3],
        queued_rescues=tow_values[4],
    )
    return NativeFrame(
        step=step,
        states=states,
        inference_ms=inference_ms,
        tow=tow,
    )


def encode_frame(frame: NativeFrame, previous: list[AgentState] | None) -> bytes:
    record_size = 36
    tow_record_size = 16
    payload = bytearray(16 + record_size * len(frame.states) + tow_record_size)
    completed_tasks = min(
        65535, frame.states[0].completed_tasks if frame.states else 0
    )
    struct.pack_into(
        "<IHHII",
        payload,
        0,
        MAGIC,
        PROTOCOL,
        completed_tasks,
        frame.step,
        len(frame.states),
    )
    for agent, state in enumerate(frame.states):
        waiting = previous is not None and (
            previous[agent].x,
            previous[agent].y,
        ) == (state.x, state.y)
        transitioned = previous is not None and (
            previous[agent].task_stage != state.task_stage
            or previous[agent].pallet_id != state.pallet_id
        )
        status = (
            (1 if waiting else 0)
            | (2 if state.loaded else 0)
            | (4 if transitioned else 0)
            | (8 if state.reload_required else 0)
            | (16 if state.failed else 0)
            | (32 if state.recovery_state == 4 else 0)
        )
        pallet_id = 65535 if state.pallet_id < 0 else min(65534, state.pallet_id)
        task_id = (
            0xFFFFFFFF if state.task_id < 0 else min(0xFFFFFFFE, state.task_id)
        )
        struct.pack_into(
            "<IffBBHIhhhhBBBBhh",
            payload,
            16 + agent * record_size,
            agent,
            float(state.x),
            float(state.y),
            status,
            max(0, state.task_stage),
            pallet_id,
            task_id,
            state.station_x,
            state.station_y,
            state.reload_x,
            state.reload_y,
            min(12, max(0, state.pallet_items)),
            12,
            state.recovery_state,
            0,
            state.pallet_x,
            state.pallet_y,
        )
    tow_offset = 16 + record_size * len(frame.states)
    tow_target = (
        65535
        if frame.tow.target_agent < 0
        else min(65534, frame.tow.target_agent)
    )
    struct.pack_into(
        "<ffBBHHH",
        payload,
        tow_offset,
        float(frame.tow.x),
        float(frame.tow.y),
        frame.tow.state,
        0,
        tow_target,
        min(65535, frame.tow.queued_rescues),
        0,
    )
    return bytes(payload)


class NativeSession:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        agent_count: int,
    ) -> None:
        self.process = process
        self.agent_count = agent_count
        self.diagnostics: deque[str] = deque(maxlen=40)

    async def read_frame(self) -> NativeFrame:
        if self.process.stdout is None:
            raise RuntimeError("native simulator stdout is unavailable")
        while True:
            raw = await self.process.stdout.readline()
            if not raw:
                return_code = await self.process.wait()
                details = "\n".join(self.diagnostics)
                raise RuntimeError(
                    f"native simulator exited with code {return_code}"
                    + (f":\n{details}" if details else "")
                )
            line = raw.decode("utf-8", errors="replace")
            frame = parse_native_frame(line, self.agent_count)
            if frame is not None:
                return frame
            self.diagnostics.append(line.rstrip())

    async def step(self) -> NativeFrame:
        if self.process.stdin is None or self.process.returncode is not None:
            raise RuntimeError("native simulator is not running")
        self.process.stdin.write(b"step\n")
        await self.process.stdin.drain()
        return await self.read_frame()

    async def fail_agent(self, agent: int) -> NativeFrame:
        if self.process.stdin is None or self.process.returncode is not None:
            raise RuntimeError("native simulator is not running")
        self.process.stdin.write(f"fail {agent}\n".encode("ascii"))
        await self.process.stdin.drain()
        return await self.read_frame()

    async def close(self) -> None:
        if self.process.returncode is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(b"quit\n")
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            await asyncio.wait_for(self.process.wait(), timeout=2)
            return
        except asyncio.TimeoutError:
            self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()


class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.start_lock = asyncio.Lock()

    def native_command(self, count: int) -> list[str]:
        return [
            str(self.args.runner.resolve()),
            "--map",
            str(self.args.map.resolve()),
            "--scen",
            str(self.args.scenario.resolve()),
            "--model",
            str(self.args.model.resolve()),
            "--num-agents",
            str(count),
            "--max-steps",
            str(self.args.max_steps),
            "--mode",
            "pibt",
            "--sampling",
            "deterministic",
            "--lifelong-tasks",
            str(self.args.tasks.resolve()),
            "--repair-x",
            str(self.args.layout_data["repairStation"]["x"]),
            "--repair-y",
            str(self.args.layout_data["repairStation"]["y"]),
            "--tow-depot-x",
            str(self.args.layout_data["towDepot"]["x"]),
            "--tow-depot-y",
            str(self.args.layout_data["towDepot"]["y"]),
            "--stream",
        ]

    def apply_layout(self, raw_pallets: object) -> int:
        if not isinstance(raw_pallets, list):
            raise ValueError("layout payload must contain a pallet list")
        layout = dict(self.args.layout_data)
        width = int(layout["width"])
        height = int(layout["height"])
        repair = (
            int(layout["repairStation"]["x"]),
            int(layout["repairStation"]["y"]),
        )
        tow_depot = (
            int(layout["towDepot"]["x"]),
            int(layout["towDepot"]["y"]),
        )
        recovery_approach = {
            (repair[0], repair[1] - 1),
            (tow_depot[0], tow_depot[1] - 1),
        }
        pallet_cells: set[tuple[int, int]] = set()
        for item in raw_pallets:
            if not isinstance(item, dict):
                raise ValueError("every pallet must have integer x and y coordinates")
            x = item.get("x")
            y = item.get("y")
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError("pallet coordinates must be integers")
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("pallet coordinates must be integers")
            if not (4 <= x <= width - 5 and 0 <= y < height):
                raise ValueError(f"pallet ({x}, {y}) is outside the editable storage grid")
            if (x, y) in recovery_approach | {repair, tow_depot}:
                raise ValueError("repair-station cells and their approaches must remain clear")
            if (x, y) in pallet_cells:
                raise ValueError(f"duplicate pallet cell ({x}, {y})")
            pallet_cells.add((x, y))
        if len(pallet_cells) < MAX_AGENTS:
            raise ValueError(f"at least {MAX_AGENTS} pallets are required")

        rows = self.args.map.read_text(encoding="ascii").splitlines()
        try:
            map_start = rows.index("map") + 1
        except ValueError as error:
            raise ValueError("map file has no grid section") from error
        grid = rows[map_start : map_start + height]
        if len(grid) != height or any(len(row) != width for row in grid):
            raise ValueError("map dimensions do not match the layout")
        passable = {
            (x, y)
            for y, row in enumerate(grid)
            for x, symbol in enumerate(row)
            if symbol != "@"
        }
        stations = {
            (int(station["x"]), int(station["y"]))
            for station in layout["stations"]
        }
        reload_stations = {
            (int(station["x"]), int(station["y"]))
            for station in layout["reloadStations"]
        }
        service_cells = stations | reload_stations | {repair, tow_depot}
        corridors = passable - pallet_cells - service_cells
        unload_approaches = {
            (x - 1, y)
            for x, y in stations
            if (x - 1, y) in corridors
        }
        reachable = set(unload_approaches)
        queue = deque(unload_approaches)
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in corridors and neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        if not any((x + 1, y) in reachable for x, y in reload_stations):
            raise ValueError("loading and unloading sides are disconnected")
        unreachable_pallets = [
            (x, y)
            for x, y in pallet_cells
            if not any(
                neighbor in reachable
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            )
        ]
        if unreachable_pallets:
            raise ValueError(
                f"{len(unreachable_pallets)} pallet cells have no loaded route to an aisle"
            )

        excluded = pallet_cells | service_cells | recovery_approach
        starts_pool = [
            (x, y)
            for y in range(1, height - 1)
            for x in range(1, width - 1)
            if (x, y) in passable and (x, y) not in excluded
        ]
        if len(starts_pool) < MAX_AGENTS:
            raise ValueError("layout leaves fewer than 100 valid robot start cells")

        pallets = sorted(pallet_cells)
        pallet_records = [
            {"id": index, "x": x, "y": y, "cargoType": (x * 31 + y * 17) % 3}
            for index, (x, y) in enumerate(pallets)
        ]
        layout["pallets"] = pallet_records

        rng = random.Random(LAYOUT_SEED)
        starts = rng.sample(starts_pool, MAX_AGENTS)
        goals = starts[1:] + starts[:1]
        scenario_lines = ["version 1"]
        for (sx, sy), (gx, gy) in zip(starts, goals):
            scenario_lines.append(
                f"0\t{self.args.map.name}\t{width}\t{height}\t{sx}\t{sy}\t{gx}\t{gy}\t0.0"
            )

        station_list = sorted(stations)
        reload_list = sorted(reload_stations)
        task_lines = [
            "task_id\tpallet_id\tpallet_x\tpallet_y\tstation_x\tstation_y"
            "\treload_x\treload_y"
        ]
        task_id = 0
        while task_id < TASK_COUNT:
            cycle = list(enumerate(pallets))
            rng.shuffle(cycle)
            for pallet_id, (px, py) in cycle:
                if task_id >= TASK_COUNT:
                    break
                sx, sy = rng.choice(station_list)
                rx, ry = rng.choice(reload_list)
                task_lines.append(
                    f"{task_id}\t{pallet_id}\t{px}\t{py}\t{sx}\t{sy}\t{rx}\t{ry}"
                )
                task_id += 1

        outputs = (
            (self.args.layout, json.dumps(layout, indent=2) + "\n", "utf-8"),
            (self.args.scenario, "\n".join(scenario_lines) + "\n", "ascii"),
            (self.args.tasks, "\n".join(task_lines) + "\n", "ascii"),
        )
        temporary: list[tuple[Path, Path]] = []
        try:
            for path, content, encoding in outputs:
                temp = path.with_name(path.name + ".tmp")
                temp.write_text(content, encoding=encoding, newline="\n")
                temporary.append((temp, path))
            for temp, path in temporary:
                temp.replace(path)
        finally:
            for temp, _ in temporary:
                temp.unlink(missing_ok=True)
        self.args.layout_data = layout
        return len(pallets)

    async def start_native(
        self, count: int
    ) -> tuple[NativeSession, NativeFrame, float]:
        async with self.start_lock:
            env = os.environ.copy()
            try:
                import torch

                torch_lib = str(Path(torch.__file__).parent / "lib")
                env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get(
                    "LD_LIBRARY_PATH", ""
                )
            except ImportError:
                pass
            env.setdefault("OMP_NUM_THREADS", str(self.args.threads))
            started = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                *self.native_command(count),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                limit=1 << 20,
            )
            session = NativeSession(process, count)
            try:
                initial = await session.read_frame()
            except Exception:
                await session.close()
                raise
            return session, initial, time.perf_counter() - started

    async def send_metrics(
        self, socket: ServerConnection, frame: NativeFrame
    ) -> None:
        await socket.send(
            json.dumps(
                {
                    "type": "metrics",
                    "inferenceMs": frame.inference_ms,
                }
            )
        )

    async def handler(self, socket: ServerConnection) -> None:
        try:
            await self._handle_connection(socket)
        except ConnectionClosed:
            pass

    async def _handle_connection(self, socket: ServerConnection) -> None:
        count = self.args.agents
        session: NativeSession | None = None
        try:
            while True:
                await socket.send(
                    json.dumps(
                        {"type": "status", "state": "planning", "agents": count}
                    )
                )
                try:
                    session, initial, planning_seconds = await self.start_native(count)
                except Exception as error:
                    await socket.send(
                        json.dumps({"type": "error", "message": str(error)})
                    )
                    return

                await socket.send(
                    json.dumps(
                        {
                            "type": "hello",
                            "protocol": PROTOCOL,
                            "map": {
                                "width": int(self.args.layout_data["width"]),
                                "height": int(self.args.layout_data["height"]),
                                "cellSize": 1,
                            },
                            "layout": self.args.layout_data,
                            "lifelong": True,
                            "simulator": True,
                            "streaming": True,
                            "agents": count,
                            "tickRate": self.args.tick_rate,
                            "planningSeconds": planning_seconds,
                            "summary": {"status": "running"},
                            "inferenceMs": initial.inference_ms,
                        }
                    )
                )
                await socket.send(encode_frame(initial, None))

                previous = initial.states
                paused = False
                speed = 1.0
                reload_count: int | None = None
                next_tick = asyncio.get_running_loop().time() + 1.0 / self.args.tick_rate

                while reload_count is None:
                    if paused:
                        raw = await socket.recv()
                    else:
                        timeout = max(
                            0.0, next_tick - asyncio.get_running_loop().time()
                        )
                        try:
                            raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            frame = await session.step()
                            await socket.send(encode_frame(frame, previous))
                            previous = frame.states
                            if frame.step == 1 or frame.step % 10 == 0:
                                await self.send_metrics(socket, frame)
                            next_tick = (
                                asyncio.get_running_loop().time()
                                + 1.0
                                / max(0.1, self.args.tick_rate * speed)
                            )
                            continue

                    if isinstance(raw, bytes):
                        continue
                    message = json.loads(raw)
                    if message.get("type") != "control":
                        continue
                    action = message.get("action")
                    if action == "pause":
                        paused = True
                    elif action == "run":
                        paused = False
                        next_tick = asyncio.get_running_loop().time()
                    elif action == "step" and paused:
                        frame = await session.step()
                        await socket.send(encode_frame(frame, previous))
                        previous = frame.states
                        await self.send_metrics(socket, frame)
                    elif action == "stop":
                        await session.close()
                        session, initial, _ = await self.start_native(count)
                        previous = initial.states
                        await socket.send(encode_frame(initial, None))
                        await self.send_metrics(socket, initial)
                        await socket.send(
                            json.dumps({"type": "status", "state": "stopped"})
                        )
                        paused = True
                    elif action == "speed":
                        speed = min(
                            3.0, max(0.25, float(message.get("value", 1)))
                        )
                    elif action == "fail":
                        agent = int(message.get("agent", -1))
                        if 0 <= agent < count:
                            frame = await session.fail_agent(agent)
                            await socket.send(encode_frame(frame, previous))
                            previous = frame.states
                    elif action == "load":
                        requested = int(message.get("agents", count))
                        reload_count = min(MAX_AGENTS, max(2, requested))
                    elif action == "layout":
                        try:
                            async with self.start_lock:
                                pallet_count = self.apply_layout(message.get("pallets"))
                            await socket.send(
                                json.dumps(
                                    {
                                        "type": "layout-applied",
                                        "pallets": pallet_count,
                                    }
                                )
                            )
                            reload_count = count
                        except (OSError, TypeError, ValueError) as error:
                            await socket.send(
                                json.dumps(
                                    {
                                        "type": "layout-error",
                                        "message": str(error),
                                    }
                                )
                            )

                await session.close()
                session = None
                count = reload_count
        finally:
            if session is not None:
                await session.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        type=Path,
        default=root / "native-runner" / "build" / "dmm_native_runner",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--map",
        type=Path,
        default=root / "scenarios" / "warehouse-lifelong-44x32.map",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=root / "scenarios" / "warehouse-lifelong-44x32.scen",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=root / "scenarios" / "warehouse-lifelong-44x32.tasks.tsv",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=root / "scenarios" / "warehouse-lifelong-44x32.layout.json",
    )
    parser.add_argument("--agents", type=int, choices=(25, 50, 100), default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="native step limit; 0 keeps the simulator running indefinitely",
    )
    parser.add_argument("--tick-rate", type=float, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    for path in (
        args.runner,
        args.model,
        args.map,
        args.scenario,
        args.tasks,
        args.layout,
    ):
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
