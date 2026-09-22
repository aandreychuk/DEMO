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
  "protocol": 5,
  "map": { "width": 44, "height": 32, "cellSize": 1 },
  "agents": 100,
  "tickRate": 10,
  "planningSeconds": 6.1,
  "inferenceMs": 0.0,
  "lifelong": true,
  "simulator": true,
  "streaming": true,
  "layout": {
    "pallets": [],
    "stations": [],
    "reloadStations": [],
    "repairStation": { "x": 22, "y": 31 },
    "towDepot": { "x": 21, "y": 31 }
  },
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
| 4 | `uint16` | Protocol version (`5`) |
| 6 | `uint16` | Completed task count, saturated at 65,535 |
| 8 | `uint32` | Simulation step |
| 12 | `uint32` | Number of agents |
| 16 | repeated record | Agent records, followed by one recovery-vehicle record |

Each protocol 5 agent record is 36 bytes:

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
| 30 | `uint8` | Recovery state |
| 31 | `uint8` | Reserved |
| 32 | `int16` | Current physical pallet X |
| 34 | `int16` | Current physical pallet Y |

The 16-byte recovery-vehicle record follows all agent records:

| Record offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `float32` | Grid X |
| 4 | `float32` | Grid Y |
| 8 | `uint8` | Vehicle state |
| 9 | `uint8` | Reserved |
| 10 | `uint16` | Target agent, or 65,535 |
| 12 | `uint16` | Queued failures |
| 14 | `uint16` | Reserved |

Coordinates are grid coordinates. Status is a bit field: bit 0 means waiting,
bit 1 means that the robot carries a pallet, bit 2 marks a task-stage
transition, bit 3 means the current task requires a reload visit, bit 4 marks a
failed robot, and bit 5 marks a repaired robot recovering its dropped pallet.
Recovery state is 0 for normal operation, 1 for waiting for the vehicle, 2 for
transport, 3 for repair, and 4 for pallet recovery. Vehicle state is 0 for idle,
1 for driving to an agent, 2 for transport to repair, and 3 for returning to
the depot. Protocol 1 through 4 frames remain readable by the browser.
`task_stage` is 0 for pickup, 1 for unloading, 2 for batch reloading, and 3 for
return. A pallet ID of 65,535 or task ID of 4,294,967,295 means that no
corresponding assignment exists.

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
{ "type": "control", "action": "layout", "pallets": [{ "x": 4, "y": 1 }] }
```

`pause` preserves the native simulator state and stops requesting steps. `stop`
terminates that native process, starts a fresh simulation at frame zero, and
holds it there; `run` resumes live computation from that state.
`fail` queues one agent for recovery. The native process preserves its task and
stage, removes it from relational agent records and `agent_chat_ids`, and sends
the external recovery vehicle along a shortest BFS route. The vehicle is a
dynamic obstacle but never an algorithm-controlled agent. After transport and
eight repair ticks, the robot resumes its saved goal. A pallet carried during
failure remains at that cell and is recovered first. Starting a fresh simulation
clears the recovery queue and all failures.

`layout` validates the edited pallet coordinates, saves the layout, regenerates
100 deterministic start positions and 4,000 tasks, and restarts that browser's
native simulation. Pallets are accepted only inside the storage grid. Every
pallet must retain a loaded route to the unloading side, both service sides must
remain connected, and the repair-station approach stays reserved.

The native process should listen on loopback only by default. Hugging Face tokens,
model paths, and AOTI runtime details are never sent to the browser.
