# EPIBT warehouse

Open **EPIBT warehouse** from the demo navigation or load `epibt.html` directly.
The page runs the rotational planner in a dedicated browser worker. EPIBT
selects an operation and executes its first primitive each tick. Use
**Operation length** to switch between 3, 4 and 5 actions. The new planner
takes over at the same tick with the same robot poses, tasks and counters.
The next tick generates a plan at the new length. A robot can move one
cell forward, turn 90 degrees right or left in its cell, or wait. The
inspector shows the executed primitive and the full
operation selected in that tick. The dashed line shows the remaining moves of
that operation. EPIBT may revise them at the next tick.

The rotation model, operation enumeration, inheritance, reservations and
backtracking come from [Straple/EPIBT](https://github.com/Straple/EPIBT/tree/3808e832593c9fb46b3e119cbb36035cccd830e1)
at revision `3808e832593c9fb46b3e119cbb36035cccd830e1`, under the MIT license in
`vendor/LICENSE`. `web.cpp` initializes the upstream graph and heuristics and
provides the browser interface. The only change in the vendored source is a
serial implementation of `launch_threads` for WebAssembly.

The adapter enables the rotation model and operation inheritance with a
revisit limit of 10. Each selectable operation length is a separate build
of the same upstream planner. The adapter uses the upstream test system's
priority for multi-stop tasks: the sum of the orientation-aware distance to the current
target and the distance of the remaining leg, with smaller values planned
first. The display reports this priority for the last tick. The warehouse
worker generates pickup and delivery tasks from the bays on the map. The
telemetry divides completed deliveries by elapsed simulation ticks and
measures time spent in the planner call.

## Configuration files

Open **Edit map** to choose a preset or paint individual cells. The presets
include rack rows, a central junction, single-cell aisles, staggered gates,
switchback lanes with passing bays, and nested loops. Import and export work on the
editor draft. **Apply & restart** starts the draft at tick zero and pauses it.
The same map and robot count produce the same starts, initial orientations
and sequence of pickup and delivery pairs.

```json
{
  "schema": "epibt-warehouse/v2",
  "map": ["#####", "#...#", "#.SD#", "#...#", "#####"],
  "agentCount": 1
}
```

`#` marks a fixed obstacle. `.`, `S` and `D` are walkable, with `S` and `D`
rendered as pickup and delivery bays. A runnable map needs at least one of
each bay type, with all bays connected by aisles and room for the robot count.
The supported size is up to 48 × 40 cells and 200 robots. Export saves the
map and robot count. The editor also imports v1 files, reading the robot count
from their starting positions and using saved task points to recover missing
bay markers.

## Rebuild the browser runtime

Run `runtime/epibt/build.sh` with Emscripten 4.0.23 on the path. It produces
the `epibt`, `epibt-4` and `epibt-5` loader and WebAssembly pairs in
`public/runtime/` and the license notice. These files are checked in so
that the ordinary frontend and GitHub Pages builds can run with Node.js
and pnpm.
