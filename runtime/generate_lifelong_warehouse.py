#!/usr/bin/env python3
"""Generate a compact warehouse and a deterministic lifelong task queue."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


WIDTH = 44
HEIGHT = 32
PALLET_X = range(5, WIDTH - 4, 8)
PALLET_Y = range(4, HEIGHT - 3, 7)
UNLOAD_X = WIDTH - 3
UNLOAD_Y = range(1, HEIGHT - 1)
UNLOAD_BACK_X = UNLOAD_X + 1
RELOAD_X = 2
RELOAD_Y = range(1, HEIGHT - 1)
RELOAD_BACK_X = RELOAD_X - 1
PALLET_CAPACITY = 12


def pallet_cells() -> list[tuple[int, int]]:
    return [
        (x, y)
        for gx in PALLET_X
        for gy in PALLET_Y
        for x in range(gx, gx + 2)
        for y in range(gy - 1, gy + 3)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "scenarios")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--tasks", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=800_000)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    map_name = "warehouse-lifelong-44x32.map"
    map_path = output / map_name
    scen_path = output / "warehouse-lifelong-44x32.scen"
    tasks_path = output / "warehouse-lifelong-44x32.tasks.tsv"
    layout_path = output / "warehouse-lifelong-44x32.layout.json"

    pallets = pallet_cells()
    stations = [(UNLOAD_X, y) for y in UNLOAD_Y]
    reload_stations = [(RELOAD_X, y) for y in RELOAD_Y]
    walls = {
        (x, y)
        for y in range(HEIGHT)
        for x in range(WIDTH)
        if x in (0, WIDTH - 1) or y in (0, HEIGHT - 1)
    }
    walls.update((UNLOAD_BACK_X, y) for y in UNLOAD_Y)
    walls.update((RELOAD_BACK_X, y) for y in RELOAD_Y)
    walls.update({
        (UNLOAD_X, min(UNLOAD_Y) - 1),
        (UNLOAD_X, max(UNLOAD_Y) + 1),
    })
    rows = [
        "".join("@" if (x, y) in walls else "." for x in range(WIDTH))
        for y in range(HEIGHT)
    ]
    excluded = set(pallets) | set(stations) | set(reload_stations)
    starts_pool = [
        (x, y)
        for y in range(1, HEIGHT - 1)
        for x in range(1, WIDTH - 1)
        if (x, y) not in excluded and (x, y) not in walls
    ]
    if args.agents > len(starts_pool) or args.agents > len(pallets):
        raise SystemExit("warehouse does not have enough starts or pallets")

    rng = random.Random(args.seed)
    starts = rng.sample(starts_pool, args.agents)
    goals = starts[1:] + starts[:1]

    map_path.write_text(
        f"type octile\nheight {HEIGHT}\nwidth {WIDTH}\nmap\n" + "\n".join(rows) + "\n",
        encoding="ascii",
    )
    with scen_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("version 1\n")
        for (sx, sy), (gx, gy) in zip(starts, goals):
            stream.write(f"0\t{map_name}\t{WIDTH}\t{HEIGHT}\t{sx}\t{sy}\t{gx}\t{gy}\t0.0\n")

    pallet_records = [
        {"id": index, "x": x, "y": y, "cargoType": (x * 31 + y * 17) % 3}
        for index, (x, y) in enumerate(pallets)
    ]
    layout_path.write_text(
        json.dumps(
            {
                "schema": "mapf-lifelong-layout/v2",
                "width": WIDTH,
                "height": HEIGHT,
                "palletCapacity": PALLET_CAPACITY,
                "pallets": pallet_records,
                "stations": [
                    {"id": index, "x": x, "y": y, "accessSide": "west"}
                    for index, (x, y) in enumerate(stations)
                ],
                "reloadStations": [
                    {"id": index, "x": x, "y": y, "accessSide": "east"}
                    for index, (x, y) in enumerate(reload_stations)
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    task_id = 0
    with tasks_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "task_id\tpallet_id\tpallet_x\tpallet_y\tstation_x\tstation_y"
            "\treload_x\treload_y\n"
        )
        while task_id < args.tasks:
            cycle = list(enumerate(pallets))
            rng.shuffle(cycle)
            for pallet_id, (px, py) in cycle:
                if task_id >= args.tasks:
                    break
                sx, sy = rng.choice(stations)
                rx, ry = rng.choice(reload_stations)
                stream.write(
                    f"{task_id}\t{pallet_id}\t{px}\t{py}\t{sx}\t{sy}"
                    f"\t{rx}\t{ry}\n"
                )
                task_id += 1

    print(f"wrote {map_path} ({WIDTH}x{HEIGHT})")
    print(f"wrote {scen_path} ({args.agents} agents)")
    print(
        f"wrote {layout_path} ({len(pallets)} pallets, {len(stations)} unload, "
        f"{len(reload_stations)} reload stations)"
    )
    print(f"wrote {tasks_path} ({args.tasks} tasks)")


if __name__ == "__main__":
    main()
