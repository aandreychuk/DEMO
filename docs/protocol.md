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
  "protocol": 3,
  "map": { "width": 44, "height": 32, "cellSize": 1 },
  "agents": 100,
  "tickRate": 10,
  "planningSeconds": 6.1,
  "inferenceMs": 0.0,
  "lifelong": true,
  "simulator": true,
  "streaming": true,
  "layout": { "pallets": [], "stations": [], "reloadStations": [] },
  "summary": { "status": "running" }
}
```

After the initial state, the bridge asks the native process to compute exactly
one FastDMM + PIBT transition per simulation tick and forwards the resulting
state immediately. It emits `metrics` messages as live inference timing becomes
available. Model paths and environment details are omitted.

## State frame

High-frequency state uses one little-endian binary message per simulation step:

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `uint32` | Magic `0x4d415046` (`MAPF`) |
| 4 | `uint16` | Protocol version (`3`) |
| 6 | `uint16` | Completed task count, saturated at 65,535 |
| 8 | `uint32` | Simulation step |
| 12 | `uint32` | Number of agents |
| 16 | repeated record | Agent records |

Each protocol 3 agent record is 32 bytes:

| Record offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `uint32` | Agent ID |
| 4 | `float32` | Grid X |
| 8 | `float32` | Grid Y |
| 12 | `uint8` | Status bits |
| 13 | `uint8` | Task stage |
| 14 | `uint16` | Pallet ID |
| 16 | `uint32` | Assignment ID |
| 20 | `int16` | Unloading station X |
| 22 | `int16` | Unloading station Y |
| 24 | `int16` | Reloading station X |
| 26 | `int16` | Reloading station Y |
| 28 | `uint8` | Items currently on the pallet |
| 29 | `uint8` | Pallet capacity (`12`) |
| 30 | `uint16` | Reserved |

Coordinates are grid coordinates. Status is a bit field: bit 0 means waiting,
bit 1 means that the robot carries a pallet, bit 2 marks a task-stage
transition, bit 3 means the current task requires a reload visit, and bit 4
marks a failed robot. A failed robot remains on its current cell and that cell
is treated as a static obstacle by every other agent.
`task_stage` is 0 for pickup, 1 for unloading, 2 for batch reloading, and 3 for
return. A pallet ID of 65,535 or task ID of 4,294,967,295 means that no
corresponding assignment exists. Protocol 1 and 2 frames remain readable by
the browser for compatibility.

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
{ "type": "control", "action": "fail", "agent": 21 }
```

`pause` preserves the native simulator state and stops requesting steps. `stop`
terminates that native process, starts a fresh simulation at frame zero, and
holds it there; `run` resumes live computation from that state.
`fail` irreversibly disables one agent for the current simulation. The native
planner pins it in place and rebuilds cost-to-go and observation obstacle maps
for the remaining agents. The failed agent is omitted from relational agent
records and `agent_chat_ids`, so it participates only as an obstacle. Starting
a fresh simulation clears all failures.

The native process should listen on loopback only by default. Hugging Face tokens,
model paths, and AOTI runtime details are never sent to the browser.
