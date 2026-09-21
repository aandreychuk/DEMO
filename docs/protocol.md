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
  "protocol": 1,
  "map": { "width": 90, "height": 70, "cellSize": 1 },
  "agents": 1000,
  "tickRate": 20,
  "frames": 140,
  "planningSeconds": 4.1,
  "inferenceMs": 22.6,
  "summary": { "status": "solved", "makespan": 139 }
}
```

The bridge also emits JSON status messages while it is planning and when a
replay reaches its last frame. Native summary metrics are included in `hello`;
model paths and environment details are omitted.

## State frame

High-frequency state uses one little-endian binary message per simulation step:

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `uint32` | Magic `0x4d415046` (`MAPF`) |
| 4 | `uint16` | Protocol version (`1`) |
| 6 | `uint16` | Flags |
| 8 | `uint32` | Simulation step |
| 12 | `uint32` | Number of agents |
| 16 | repeated record | Agent records |

Each agent record is 16 bytes: `uint32 id`, `float32 x`, `float32 y`, `uint8
status`, and three reserved bytes. Coordinates are grid coordinates. Status is 0
for moving, 1 for waiting, 2 for replanned, and 3 for arrived.

The UI interpolates between state frames. It never feeds interpolated positions
back into MAPF logic.

## Browser commands

Commands are infrequent JSON messages:

```json
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "run" }
{ "type": "control", "action": "step" }
{ "type": "control", "action": "speed", "value": 2.0 }
{ "type": "control", "action": "load", "agents": 2500 }
```

The native process should listen on loopback only by default. Hugging Face tokens,
model paths, and AOTI runtime details are never sent to the browser.
