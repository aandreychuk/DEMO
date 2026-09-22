import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const width = 44;
const height = 33;
const count = 100;
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

for (let tick = 0; tick < 10; tick++) {
  for (let i = 0; i < count; i++) core.setAgentState(i, positions[i], goals[i], 0, 0, 0);
  core.setTow(-1, -1);
  core.buildInputs();
  let current = views();
  for (const token of current.observations) {
    if (token < 0 || token > 66) throw new Error(`observation token out of range: ${token}`);
  }
  for (const neighbor of current.chat) {
    if (neighbor < -1 || neighbor >= count) throw new Error(`chat id out of range: ${neighbor}`);
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

console.log('browser core validation: 10 collision-free ticks, token ranges valid');
