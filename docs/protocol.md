# MAPF live protocol

The renderer and the algorithm process communicate over a local WebSocket
(`ws://127.0.0.1:18765` by default). The
browser is a passive consumer: planning, inference, collision checks, and metrics
remain in the native process.

## Bootstrap message

The first server message is JSON so integrations are easy to inspect:

```json
{
  "type": "hello",
  "protocol": 2,
  "map": { "width": 44, "height": 32, "cellSize": 1 },
  "agents": 100,
  "tickRate": 10,
  "frames": 601,
  "planningSeconds": 6.1,
  "inferenceMs": 5.5,
  "lifelong": true,
  "looping": true,
  "layout": { "pallets": [], "stations": [] },
  "summary": { "status": "lifelong_horizon", "completed_tasks": 630 }
}
```

The bridge emits JSON status messages while it is planning and at replay cycle
boundaries. Once the cached native horizon reaches its last frame, the bridge
starts it again and keeps streaming until the browser sends `stop`. Native
summary metrics are included in `hello`; model paths and environment details
are omitted.

## State frame

High-frequency state uses one little-endian binary message per simulation step:

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `uint32` | Magic `0x4d415046` (`MAPF`) |
| 4 | `uint16` | Protocol version (`2`) |
| 6 | `uint16` | Completed task count, saturated at 65,535 |
| 8 | `uint32` | Simulation step |
| 12 | `uint32` | Number of agents |
| 16 | repeated record | Agent records |

Each protocol 2 agent record is 24 bytes: `uint32 id`, `float32 x`, `float32 y`,
`uint8 status`, `uint8 task_stage`, `uint16 pallet_id`, `uint32 task_id`,
`int16 station_x`, and `int16 station_y`. Coordinates are grid coordinates.
Status is a bit field: bit 0 means waiting, bit 1 means loaded, bit 2 marks a
task-stage transition, and bit 3 marks the final replay frame. `task_stage` is
0 for pickup, 1 for unloading, and 2 for return. A pallet ID of 65,535 or task
ID of 4,294,967,295 means that no corresponding assignment exists.

The UI interpolates between state frames. It never feeds interpolated positions
back into MAPF logic.

## Browser commands

Commands are infrequent JSON messages:

```json
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "run" }
{ "type": "control", "action": "stop" }
{ "type": "control", "action": "step" }
{ "type": "control", "action": "speed", "value": 2.0 }
{ "type": "control", "action": "load", "agents": 100 }
```

`pause` preserves the current frame. `stop` resets the replay to frame zero and
holds it there; `run` resumes streaming from that frame.

The native process should listen on loopback only by default. Hugging Face tokens,
model paths, and AOTI runtime details are never sent to the browser.
