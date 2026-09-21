#!/usr/bin/env python3
"""Generate the warehouse map and a deterministic 2,500-agent scenario."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


WIDTH = 90
HEIGHT = 70
RACK_X = range(6, WIDTH - 5, 9)
RACK_Y = range(5, HEIGHT - 4, 8)


def is_blocked(x: int, y: int) -> bool:
    if x in (0, WIDTH - 1) or y in (0, HEIGHT - 1):
        return True
    return any(cx <= x <= cx + 1 and cy - 2 <= y <= cy + 3 for cx in RACK_X for cy in RACK_Y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "scenarios")
    parser.add_argument("--agents", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=800_000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    map_name = "warehouse-90x70.map"
    map_path = args.output / map_name
    scen_path = args.output / "warehouse-90x70.scen"

    rows = ["".join("@" if is_blocked(x, y) else "." for x in range(WIDTH)) for y in range(HEIGHT)]
    free = [(x, y) for y in range(HEIGHT) for x in range(WIDTH) if rows[y][x] == "."]
    if args.agents > len(free):
        raise SystemExit(f"requested {args.agents} agents, map only has {len(free)} free cells")

    rng = random.Random(args.seed)
    starts = rng.sample(free, args.agents)
    goals = rng.sample(free, args.agents)
    for _ in range(args.agents):
        if all(start != goal for start, goal in zip(starts, goals)):
            break
        goals = goals[1:] + goals[:1]

    map_path.write_text(
        f"type octile\nheight {HEIGHT}\nwidth {WIDTH}\nmap\n" + "\n".join(rows) + "\n",
        encoding="ascii",
    )
    with scen_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("version 1\n")
        for start, goal in zip(starts, goals):
            sx, sy = start
            gx, gy = goal
            stream.write(f"0\t{map_name}\t{WIDTH}\t{HEIGHT}\t{sx}\t{sy}\t{gx}\t{gy}\t0.0\n")

    print(f"wrote {map_path} ({len(free)} free cells)")
    print(f"wrote {scen_path} ({args.agents} agents)")


if __name__ == "__main__":
    main()
