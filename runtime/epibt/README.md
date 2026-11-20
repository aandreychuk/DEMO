# EPIBT warehouse

Open **EPIBT warehouse** from the demo navigation or load `epibt.html` directly.
The page runs the rotational planner in a dedicated browser worker. EPIBT
selects an operation of depth three and executes its first primitive each
tick. A robot can move one cell forward, turn 90 degrees right or left in its
cell, or wait. The inspector shows the executed primitive and the full
operation selected in that tick. The dashed line shows the remaining moves of
that operation. EPIBT may revise them at the next tick.

The rotation model, operation enumeration, inheritance, reservations and
backtracking come from [Straple/EPIBT](https://github.com/Straple/EPIBT/tree/3808e832593c9fb46b3e119cbb36035cccd830e1)
at revision `3808e832593c9fb46b3e119cbb36035cccd830e1`, under the MIT license in
`vendor/LICENSE`. `web.cpp` initializes the upstream graph and heuristics and
provides the browser interface. The only change in the vendored source is a
serial implementation of `launch_threads` for WebAssembly.

The adapter enables the rotation model and operation inheritance, with depth
3 and a revisit limit of 10. It uses the upstream test system's priority for
multi-stop tasks: the sum of the orientation-aware distance to the current
target and the distance of the remaining leg, with smaller values planned
first. The display reports this priority for the last tick. The warehouse worker cycles through
pickup and delivery tasks. The telemetry divides completed deliveries by
elapsed simulation ticks and measures time spent in the planner call.

## Configuration files

Use **Export JSON** to get the current starting configuration. Import starts
the selected configuration at tick zero and pauses it. Restart uses the same
starting configuration. Coordinates are zero-based, with `x` increasing to the
right and `y` increasing down. Headings are `E`, `S`, `W` or `N`.

```json
{
  "schema": "epibt-warehouse/v1",
  "map": ["#####", "#...#", "#.SD#", "#...#", "#####"],
  "agents": [{ "x": 2, "y": 2, "heading": "N" }],
  "tasks": [
    { "pickup": { "x": 2, "y": 2 }, "delivery": { "x": 3, "y": 2 } }
  ]
}
```

`#` marks a fixed obstacle. `.`, `S` and `D` are walkable, with `S` and `D`
rendered as pickup and delivery bays. Starts must be distinct, with all starts
and task points connected by walkable cells. The supported size is up to
48 × 40 cells, 200 robots and 4,096 task templates. Export includes the
map, starts, headings and tasks.

## Rebuild the browser runtime

Run `runtime/epibt/build.sh` with Emscripten 4.0.23 on the path. It produces
`public/runtime/epibt.mjs`, `public/runtime/epibt.wasm` and the license notice.
These files are checked in so that the ordinary frontend and GitHub Pages
builds can run with Node.js and pnpm.
