#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vendor="${root}/vendor/src"
output="${root}/../../public/runtime"

for depth in 3 4 5; do
  name=epibt
  if [[ "$depth" != 3 ]]; then name="epibt-$depth"; fi
  "${EMXX:-em++}" -std=c++20 -O3 -I "${vendor}" \
    -DENABLE_ROTATE_MODEL -DENABLE_EPIBT_IO -DENABLE_ASSERT \
    -DTHREADS_NUM=1 -DEPIBT_DEPTH="$depth" -DEPIBT_REVISIT=10 \
    "${root}/web.cpp" "${vendor}"/environment/*.cpp \
    "${vendor}"/planner/epibt/*.cpp "${vendor}"/utils/*.cpp \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=worker,node \
    -sALLOW_MEMORY_GROWTH=1 -sINVOKE_RUN=0 \
    -sEXPORTED_FUNCTIONS='["_epibt_init","_epibt_step","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["HEAPU8","HEAP32"]' \
    --no-entry -o "${output}/${name}.mjs"
  chmod 644 "${output}/${name}.wasm"
done
cp "${root}/vendor/LICENSE" "${output}/epibt-license.txt"
