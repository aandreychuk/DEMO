# Source provenance

This directory is a compact extraction of the native DMM evaluation path used
by the MAPF-GPU experiments. It is packaged here so that inference does not
depend on an unpublished checkout or machine-specific paths.

| Packaged component | Source component | Purpose |
| --- | --- | --- |
| `src/main.cpp` | native runner adapter | CLI, execution loop, trajectory and metrics output |
| `src/dmm_standalone.cpp` | `lagat/dmm_standalone.cpp` | DMM tokenization, AOTI loading, and action logits |
| `src/lc_mapf_standalone.cpp` | `lagat/lc_mapf_standalone.cpp` | local observation construction |
| `src/mapf_gpt_standalone.cpp` | `lagat/mapf_gpt_standalone.cpp` | shared observation wrapper required by the policy interface |
| `src/policy.cpp` | `lagat/policy.cpp` | policy integration and optional RSE state |
| `src/pibt.cpp` | `lagat/pibt.cpp` | PIBT collision shielding |
| `src/graph.cpp`, `src/instance.cpp`, `src/dist_table.cpp` | `lagat/` graph/instance sources | MovingAI input and exact BFS goal distances |
| remaining `src/` and `include/` files | matching `lagat/` sources | collision tables, metrics, RNG, and utilities |

The bundled files retain the upstream LaGAT license in
`licenses/LICENSE-LAGAT`. Model artifacts have separate hashes and provenance
in the repository-level `manifest.json` and their adjacent sidecars.

This runner targets one explicit MAPF task. Dynamic multi-environment batching,
refill, and the vendor CUDA-BFS implementation remain part of POGEMA-GPU and
are deliberately not duplicated here.
