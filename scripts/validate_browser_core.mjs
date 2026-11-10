import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const width = 44;
const height = 33;
const count = Number(process.argv[2] ?? 100);
const cells = width * height;
const wasm = await WebAssembly.instantiate(
  await readFile(resolve('public/runtime/mapf-core.wasm')),
  { env: { abort: () => { throw new Error('WASM core aborted'); } } },
);
const core = wasm.instance.exports;
core.configure(width, height, count);

for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    const blocked = x < 2 || x >= width - 2 || ((y === 0 || y === height - 1) && x >= width - 4);
    core.setWall(y * width + x, blocked ? 1 : 0);
  }
}
for (let y = 1; y < height - 1; y++) core.setStation(y * width + 41, 1);
for (let y = 0; y < height - 2; y++) core.setStation(y * width + 2, 1);
core.setStation((height - 1) * width + 2, 1);
core.setStation((height - 2) * width + 2, 1);

const positions = [];
const goals = [];
const freeCells = [];
for (let y = 1; y < height - 1; y++) {
  for (let x = 3; x < width - 3; x++) freeCells.push(y * width + x);
}
for (let i = 0; i < count; i++) positions.push(freeCells[Math.floor(i * freeCells.length / count)]);
for (let i = 0; i < count; i++) goals.push(positions[(i * 37 + 51) % count]);

function views() {
  const memory = core.memory.buffer;
  return {
    observations: new Int32Array(memory, core.observationsPointer(), count * 256),
    chat: new Int32Array(memory, core.chatPointer(), count * 13),
    probabilities: new Float32Array(memory, core.probabilitiesPointer(), count * 5),
    next: new Int32Array(memory, core.nextPositionsPointer(), count),
    actions: new Int32Array(memory, core.executedActionsPointer(), count),
  };
}

let ordinaryMoves = 0;
for (let tick = 0; tick < 10; tick++) {
  for (let i = 0; i < count; i++) core.setAgentState(i, positions[i], goals[i], 0, 0, 0);
  core.setTow(-1, -1, -1, -1, 0);
  core.buildInputs();
  let current = views();
  for (const token of current.observations) {
    if (token < 0 || token > 66) throw new Error(`observation token out of range: ${token}`);
  }
  for (const neighbor of current.chat) {
    if (neighbor < -1 || neighbor >= count) throw new Error(`chat id out of range: ${neighbor}`);
  }
  for (let agent = 0; agent < count; agent++) {
    const agentX = positions[agent] % width;
    const agentY = Math.floor(positions[agent] / width);
    for (let slot = 0; slot < 13; slot++) {
      const neighbor = current.chat[agent * 13 + slot];
      if (neighbor < 0) break;
      const neighborX = positions[neighbor] % width;
      const neighborY = Math.floor(positions[neighbor] / width);
      const goalX = goals[neighbor] % width;
      const goalY = Math.floor(goals[neighbor] / width);
      const base = agent * 256 + 121 + slot * 10;
      const expected = [
        neighborY - agentY + 20,
        neighborX - agentX + 20,
        Math.max(-20, Math.min(20, goalY - agentY)) + 20,
        Math.max(-20, Math.min(20, goalX - agentX)) + 20,
      ];
      for (let field = 0; field < expected.length; field++) {
        if (current.observations[base + field] !== expected[field]) {
          throw new Error(`row/column observation mismatch at agent ${agent}, slot ${slot}, field ${field}`);
        }
      }
    }
  }
  for (let i = 0; i < count; i++) {
    for (let action = 0; action < 5; action++) {
      // Stable but changing preferences exercise pushes and conflict resolution.
      current.probabilities[i * 5 + action] = ((i * 17 + action * 31 + tick * 7) % 101) / 101;
    }
  }
  core.plan();
  current = views();
  const occupied = new Set();
  for (let i = 0; i < count; i++) {
    const target = current.next[i];
    if (target < 0 || target >= cells) throw new Error(`invalid target for agent ${i}`);
    if (target !== positions[i]) ordinaryMoves++;
    if (occupied.has(target)) throw new Error(`vertex collision at tick ${tick}, cell ${target}`);
    occupied.add(target);
    const delta = Math.abs(target - positions[i]);
    if (delta !== 0 && delta !== 1 && delta !== width) throw new Error(`non-adjacent move for agent ${i}`);
    for (let j = i + 1; j < count; j++) {
      if (target === positions[j] && current.next[j] === positions[i] && target !== positions[i]) {
        throw new Error(`swap collision at tick ${tick}, agents ${i}/${j}`);
      }
    }
  }
  for (let i = 0; i < count; i++) positions[i] = current.next[i];
}
if (ordinaryMoves === 0) throw new Error('ordinary agents froze without a tow unit');

console.log('browser core validation: 10 collision-free ticks, token ranges valid');

// The tow unit has the highest priority, but the robot in its next cell must
// move as part of the same joint action. Pickup and carrying are the only
// intentional overlaps with the failed robot.
core.configure(5, 3, 2);
const at = (x, y) => y * 5 + x;
core.setStation(at(4, 1), 1);
let robot = at(2, 1);
const failedRobot = at(3, 1);
const planTow = (tow, goal, pickup, shared, carrying) => {
  core.setAgentState(0, robot, at(0, 1), 0, 0, 0);
  core.setAgentState(1, failedRobot, failedRobot, 0, 1, 1);
  core.setTow(tow, goal, pickup, shared, carrying);
  core.buildInputs();
  const probs = new Float32Array(core.memory.buffer, core.probabilitiesPointer(), 10);
  probs.fill(0);
  probs[0] = 1;
  probs[2] = 0.8;
  core.plan();
  const next = new Int32Array(core.memory.buffer, core.nextPositionsPointer(), 2);
  return { robot: next[0], failed: next[1], tow: core.getTowNext() };
};
let move = planTow(at(1, 1), failedRobot, 1, -1, 0);
if (move.tow !== at(2, 1) || move.robot === move.tow || move.failed !== failedRobot) {
  throw new Error(`tow did not displace the active robot: ${JSON.stringify(move)}`);
}
robot = move.robot;
move = planTow(at(2, 1), failedRobot, 1, -1, 0);
if (move.tow !== failedRobot || move.robot === move.tow) {
  throw new Error(`tow did not reach the failed robot: ${JSON.stringify(move)}`);
}
move = planTow(failedRobot, failedRobot, -1, 1, 1);
if (move.tow !== failedRobot || move.failed !== failedRobot) {
  throw new Error(`tow did not wait during loading: ${JSON.stringify(move)}`);
}
move = planTow(failedRobot, at(4, 1), -1, 1, 1);
if (move.tow !== at(4, 1) || move.robot === move.tow) {
  throw new Error(`tow did not carry the failed robot to repair: ${JSON.stringify(move)}`);
}
console.log('tow PIBT validation: displacement, pickup, loading, transport');

core.configure(12, 10, 50);
const crowdedPositions = Array.from({ length: 50 }, (_, i) => i + 1);
const crowdedGoals = crowdedPositions.map((_, i) => crowdedPositions[(i * 13 + 17) % 50]);
let crowdedTow = 0;
for (let tick = 0; tick < 40; tick++) {
  for (let i = 0; i < 50; i++) {
    core.setAgentState(i, crowdedPositions[i], crowdedGoals[i], 0, 0, 0);
  }
  core.setTow(crowdedTow, 119, -1, -1, 0);
  core.buildInputs();
  const scores = new Float32Array(core.memory.buffer, core.probabilitiesPointer(), 250);
  for (let i = 0; i < scores.length; i++) scores[i] = ((i * 31 + tick * 7) % 97) / 97;
  core.plan();
  const planned = new Int32Array(core.memory.buffer, core.nextPositionsPointer(), 50);
  const towPlanned = core.getTowNext();
  const occupied = new Set([towPlanned]);
  for (let i = 0; i < 50; i++) {
    if (occupied.has(planned[i])) throw new Error(`tow/crowd vertex collision at tick ${tick}`);
    occupied.add(planned[i]);
    if (planned[i] === crowdedTow && towPlanned === crowdedPositions[i]) {
      throw new Error(`tow/crowd swap collision at tick ${tick}`);
    }
  }
  for (let i = 0; i < 50; i++) crowdedPositions[i] = planned[i];
  crowdedTow = towPlanned;
}
if (crowdedTow === 0) throw new Error('tow never moved through the crowd');
console.log('tow PIBT validation: 40 collision-free crowded ticks');

// Both robots want the middle cell. Robot 1 must go first in manual mode,
// even though its own model prefers east and robot 0 has the lower ID.
core.configure(5, 3, 2);
const left = at(1, 1);
const middle = at(2, 1);
const right = at(3, 1);
function planManual(agent, direction, tow = -1, towGoal = -1) {
  core.setTow(tow, towGoal, -1, -1, 0);
  core.setAgentState(0, left, middle, 0, 0, 0);
  core.setAgentState(1, right, at(4, 1), 0, 0, 0);
  core.setManualControl(agent, direction);
  core.buildInputs();
  const scores = new Float32Array(core.memory.buffer, core.probabilitiesPointer(), 10);
  scores.fill(0);
  scores[4] = scores[9] = 1; // Both model policies prefer east.
  core.plan();
  return [...new Int32Array(core.memory.buffer, core.nextPositionsPointer(), 2)];
}
let planned = planManual(1, 3);
if (planned[1] !== middle || planned[0] === middle) throw new Error('manual priority or direction ignored');
core.setWall(middle, 1);
if (planManual(1, 3)[1] !== right) throw new Error('blocked manual direction did not wait');
core.setWall(middle, 0);
if (planManual(-1, 0)[1] !== at(4, 1)) throw new Error('model steering did not resume');
planned = planManual(1, 3, at(2, 0), middle);
if (core.getTowNext() !== middle || planned[1] !== right) throw new Error('tow lost its priority to manual control');
planned = planManual(1, 0, at(3, 0), right);
if (core.getTowNext() !== right || planned[1] !== at(4, 1)) throw new Error('manual robot did not yield to the tow chain');
console.log('manual PIBT validation: direction, wait, release, and tow priority');
