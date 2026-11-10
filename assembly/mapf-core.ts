// Dense browser-side MAPF primitives. AssemblyScript keeps the hot loops
// (cost-to-go BFS, observation tokenization and collision shielding) in WASM.

const MAX_AGENTS: i32 = 1000;
const OBS_TOKENS: i32 = 256;
const CHAT_SLOTS: i32 = 13;
const ACTIONS: i32 = 5;
const OBS_RADIUS: i32 = 5;
const LIMIT: i32 = 20;
const PAD_TOKEN: i32 = 66;

let width: i32 = 0;
let height: i32 = 0;
let cells: i32 = 0;
let agents: i32 = 0;

let walls = new Uint8Array(0);
let stations = new Uint8Array(0);
let pallets = new Uint8Array(0);
let positions = new Int32Array(MAX_AGENTS);
let goals = new Int32Array(MAX_AGENTS);
let loaded = new Uint8Array(MAX_AGENTS);
let failed = new Uint8Array(MAX_AGENTS);
let pinned = new Uint8Array(MAX_AGENTS);
let history = new Int8Array(MAX_AGENTS * 5);
let priorities = new Float32Array(MAX_AGENTS);

let observations = new Int32Array(MAX_AGENTS * OBS_TOKENS);
let chat = new Int32Array(MAX_AGENTS * CHAT_SLOTS);
let probabilities = new Float32Array(MAX_AGENTS * ACTIONS);
let nextPositions = new Int32Array(MAX_AGENTS + 1);
let executedActions = new Int32Array(MAX_AGENTS);

let distances = new Int16Array(0);
let queue = new Int32Array(0);
let agentAt = new Int32Array(0);
let failedAt = new Int32Array(0);
let occupiedNext = new Int32Array(0);
let preference = new Int32Array(MAX_AGENTS * ACTIONS);
let preferenceCount = new Int32Array(MAX_AGENTS);
let order = new Int32Array(MAX_AGENTS);
let towCell: i32 = -1;
let towGoal: i32 = -1;
let towPickupAgent: i32 = -1;
let towSharedAgent: i32 = -1;
let towCarrying: bool = false;
let towDistances = new Int16Array(0);
let manualAgent: i32 = -1;
let manualAction: i32 = 0;

export function configure(mapWidth: i32, mapHeight: i32, agentCount: i32): void {
  manualAgent = -1;
  width = mapWidth;
  height = mapHeight;
  cells = width * height;
  agents = min(MAX_AGENTS, max(1, agentCount));
  walls = new Uint8Array(cells);
  stations = new Uint8Array(cells);
  pallets = new Uint8Array(cells);
  distances = new Int16Array(MAX_AGENTS * cells);
  queue = new Int32Array(cells);
  agentAt = new Int32Array(cells);
  failedAt = new Int32Array(cells);
  occupiedNext = new Int32Array(cells);
  towDistances = new Int16Array(cells);
  positions.fill(-1);
  goals.fill(-1);
  loaded.fill(0);
  failed.fill(0);
  pinned.fill(0);
  history.fill(-1);
  priorities.fill(0.0);
  observations.fill(PAD_TOKEN);
  chat.fill(-1);
  probabilities.fill(0.0);
  nextPositions.fill(-1);
  executedActions.fill(0);
}

export function getAgentCount(): i32 { return agents; }
export function setWall(cell: i32, value: i32): void {
  if (cell >= 0 && cell < cells) walls[cell] = value != 0 ? 1 : 0;
}
export function setStation(cell: i32, value: i32): void {
  if (cell >= 0 && cell < cells) stations[cell] = value != 0 ? 1 : 0;
}
export function clearPallets(): void { pallets.fill(0); }
export function setPallet(cell: i32, value: i32): void {
  if (cell >= 0 && cell < cells) pallets[cell] = value != 0 ? 1 : 0;
}
export function setAgentState(index: i32, position: i32, goal: i32,
                              hasLoad: i32, isFailed: i32, isPinned: i32): void {
  if (index < 0 || index >= agents) return;
  positions[index] = position;
  goals[index] = goal;
  loaded[index] = hasLoad != 0 ? 1 : 0;
  failed[index] = isFailed != 0 ? 1 : 0;
  pinned[index] = isPinned != 0 ? 1 : 0;
}
export function setTow(current: i32, goal: i32, pickupAgent: i32,
                       sharedAgent: i32, carrying: i32): void {
  towCell = current;
  towGoal = goal;
  towPickupAgent = pickupAgent;
  towSharedAgent = sharedAgent;
  towCarrying = carrying != 0;
}
export function resetHistory(): void { history.fill(-1); }

export function setManualControl(agent: i32, action: i32): void {
  manualAgent = agent >= 0 && agent < agents ? agent : -1;
  manualAction = action >= 0 && action < ACTIONS ? action : 0;
}

function row(cell: i32): i32 { return cell / width; }
function col(cell: i32): i32 { return cell % width; }
function clampI(value: i32, low: i32, high: i32): i32 {
  return value < low ? low : (value > high ? high : value);
}

function neighbor(cell: i32, action: i32): i32 {
  const x = col(cell);
  const y = row(cell);
  if (action == 0) return cell;
  if (action == 1) return y > 0 ? cell - width : -1;
  if (action == 2) return y + 1 < height ? cell + width : -1;
  if (action == 3) return x > 0 ? cell - 1 : -1;
  if (action == 4) return x + 1 < width ? cell + 1 : -1;
  return -1;
}

function actionBetween(from: i32, to: i32): i32 {
  if (to == from - width) return 1;
  if (to == from + width) return 2;
  if (to == from - 1) return 3;
  if (to == from + 1) return 4;
  return 0;
}

function blockedFor(agent: i32, cell: i32): bool {
  if (cell < 0 || cell >= cells || walls[cell] != 0) return true;
  if (stations[cell] != 0 && cell != goals[agent] && cell != positions[agent]) return true;
  if (loaded[agent] != 0 && pallets[cell] != 0) return true;
  if (failedAt[cell] >= 0 && failedAt[cell] != agent) return true;
  return false;
}

function computeDistance(agent: i32): void {
  const offset = agent * cells;
  for (let index: i32 = 0; index < cells; ++index) distances[offset + index] = -1;
  const goal = goals[agent];
  if (goal < 0 || goal >= cells || blockedFor(agent, goal)) return;
  let head: i32 = 0;
  let tail: i32 = 0;
  queue[tail++] = goal;
  distances[offset + goal] = 0;
  while (head < tail) {
    const current = queue[head++];
    const nextDistance = distances[offset + current] + 1;
    for (let action: i32 = 1; action < ACTIONS; ++action) {
      const target = neighbor(current, action);
      if (target >= 0 && distances[offset + target] < 0 && !blockedFor(agent, target)) {
        distances[offset + target] = <i16>nextDistance;
        queue[tail++] = target;
      }
    }
  }
}

function costToken(value: i32): i32 {
  if (value == -80) return 41;
  if (value == -40) return 42;
  if (value == 40) return 43;
  return value + LIMIT;
}

function actionToken(action: i32): i32 {
  if (action < 0) return 44; // n = no prior action
  return 45 + min(4, action); // wait/up/down/left/right
}

function nextActionMask(agent: i32): i32 {
  const current = positions[agent];
  const offset = agent * cells;
  const currentDistance = current >= 0 ? distances[offset + current] : -1;
  if (currentDistance < 0) return 0;
  let mask: i32 = 0;
  for (let action: i32 = 1; action < ACTIONS; ++action) {
    const target = neighbor(current, action);
    if (target >= 0 && distances[offset + target] >= 0 && distances[offset + target] < currentDistance) {
      if (action == 1) mask |= 8;
      else if (action == 2) mask |= 4;
      else if (action == 3) mask |= 2;
      else mask |= 1;
    }
  }
  return mask;
}

function sortCandidates(ids: Int32Array, count: i32, center: i32): void {
  const cx = col(center);
  const cy = row(center);
  for (let i: i32 = 1; i < count; ++i) {
    const value = ids[i];
    const vx = col(positions[value]);
    const vy = row(positions[value]);
    const valueDistance = abs(vx - cx) + abs(vy - cy);
    let j = i - 1;
    while (j >= 0) {
      const other = ids[j];
      const ox = col(positions[other]);
      const oy = row(positions[other]);
      const otherDistance = abs(ox - cx) + abs(oy - cy);
      if (otherDistance < valueDistance ||
          (otherDistance == valueDistance && other < value)) break;
      ids[j + 1] = other;
      --j;
    }
    ids[j + 1] = value;
  }
}

export function buildInputs(): void {
  observations.fill(PAD_TOKEN);
  chat.fill(-1);
  agentAt.fill(-1);
  failedAt.fill(-1);
  for (let i: i32 = 0; i < agents; ++i) {
    if (positions[i] >= 0 && positions[i] < cells) {
      agentAt[positions[i]] = i;
      if (failed[i] != 0 && i != towSharedAgent) failedAt[positions[i]] = i;
    }
  }
  for (let i: i32 = 0; i < agents; ++i) computeDistance(i);
  const candidates = new Int32Array(MAX_AGENTS);
  for (let agent: i32 = 0; agent < agents; ++agent) {
    const distanceOffset = agent * cells;
    const base = agent * OBS_TOKENS;
    const center = positions[agent];
    const cy = row(center);
    const cx = col(center);
    const middle = distances[distanceOffset + center] >= 0 ? distances[distanceOffset + center] : 0;
    let cursor: i32 = 0;
    for (let dy: i32 = -OBS_RADIUS; dy <= OBS_RADIUS; ++dy) {
      for (let dx: i32 = -OBS_RADIUS; dx <= OBS_RADIUS; ++dx) {
        const x = cx + dx;
        const y = cy + dy;
        let encodedValue: i32 = -80;
        if (x >= 0 && x < width && y >= 0 && y < height) {
          const d = distances[distanceOffset + y * width + x];
          if (d >= 0) {
            const delta = d - middle;
            encodedValue = delta > LIMIT ? 40 : (delta < -LIMIT ? -40 : delta);
          }
        }
        observations[base + cursor++] = costToken(encodedValue);
      }
    }

    let candidateCount: i32 = 0;
    if (failed[agent] == 0) {
      for (let dy: i32 = -OBS_RADIUS; dy <= OBS_RADIUS; ++dy) {
        for (let dx: i32 = -OBS_RADIUS; dx <= OBS_RADIUS; ++dx) {
          const x = cx + dx;
          const y = cy + dy;
          if (x < 0 || x >= width || y < 0 || y >= height) continue;
          const candidate = agentAt[y * width + x];
          if (candidate >= 0 && failed[candidate] == 0) candidates[candidateCount++] = candidate;
        }
      }
      sortCandidates(candidates, candidateCount, center);
    }

    const visible = min(CHAT_SLOTS, candidateCount);
    for (let slot: i32 = 0; slot < visible; ++slot) {
      const other = candidates[slot];
      chat[agent * CHAT_SLOTS + slot] = other;
      const otherPosition = positions[other];
      const otherGoal = goals[other];
      // Match the training/native runtime contract: relative coordinates are
      // encoded as (row, column), not Cartesian (x, y).
      observations[base + cursor++] = row(otherPosition) - cy + LIMIT;
      observations[base + cursor++] = col(otherPosition) - cx + LIMIT;
      observations[base + cursor++] = clampI(row(otherGoal) - cy, -LIMIT, LIMIT) + LIMIT;
      observations[base + cursor++] = clampI(col(otherGoal) - cx, -LIMIT, LIMIT) + LIMIT;
      for (let h: i32 = 0; h < 5; ++h) {
        observations[base + cursor++] = actionToken(history[other * 5 + h]);
      }
      observations[base + cursor++] = 50 + nextActionMask(other);
    }
    while (cursor < OBS_TOKENS) observations[base + cursor++] = PAD_TOKEN;
  }
}

function forbidden(agent: i32, target: i32): bool {
  if (target < 0 || target >= cells || walls[target] != 0) return true;
  if (stations[target] != 0 && target != positions[agent] && target != goals[agent]) return true;
  if (loaded[agent] != 0 && pallets[target] != 0) return true;
  for (let other: i32 = 0; other < agents; ++other) {
    if (other != agent && other != towSharedAgent &&
        failed[other] != 0 && positions[other] == target) return true;
  }
  return false;
}

function buildPreferences(agent: i32): void {
  let count: i32 = 0;
  for (let action: i32 = 0; action < ACTIONS; ++action) {
    const target = neighbor(positions[agent], action);
    if (target < 0 || forbidden(agent, target)) continue;
    let insert = count;
    const score = probabilities[agent * ACTIONS + action];
    while (insert > 0) {
      const previousAction = preference[agent * ACTIONS + insert - 1];
      const previousScore = probabilities[agent * ACTIONS + previousAction];
      if (previousScore > score || (previousScore == score && previousAction < action)) break;
      preference[agent * ACTIONS + insert] = previousAction;
      --insert;
    }
    preference[agent * ACTIONS + insert] = action;
    ++count;
  }
  preferenceCount[agent] = count;
}

function assignPIBT(agent: i32, depth: i32): bool {
  if (nextPositions[agent] >= 0) return true;
  if (depth > agents + 1) return false;
  const origin = positions[agent];
  const manual = agent == manualAgent;
  // Only an earlier reservation (from the tow chain) can force manual steering aside.
  const count = manual && occupiedNext[origin] < 0 ? 0 : preferenceCount[agent];
  for (let rank: i32 = manual ? -2 : 0; rank < count; ++rank) {
    const action = rank == -2 ? manualAction : rank == -1 ? 0 : preference[agent * ACTIONS + rank];
    if (rank == -2 && (action == 0 || forbidden(agent, neighbor(origin, action)))) continue;
    if (manual && rank >= 0 && (action == manualAction || action == 0)) continue;
    const target = neighbor(origin, action);
    if (target < 0 || occupiedNext[target] >= 0) continue;
    const other = agentAt[target];
    if (other >= 0 && other != agent && nextPositions[other] == origin) continue;
    occupiedNext[target] = agent;
    nextPositions[agent] = target;
    if (other >= 0 && other != agent && nextPositions[other] < 0) {
      if (!assignPIBT(other, depth + 1)) {
        occupiedNext[target] = -1;
        nextPositions[agent] = -1;
        continue;
      }
    }
    return true;
  }
  if (occupiedNext[origin] < 0) {
    occupiedNext[origin] = agent;
    nextPositions[agent] = origin;
    return true;
  }
  return false;
}

function towPassable(target: i32): bool {
  return target >= 0 && target < cells && walls[target] == 0 &&
    (target == towCell || target == towGoal ||
     (stations[target] == 0 && pallets[target] == 0));
}

function buildTowDistances(): void {
  towDistances.fill(-1);
  if (towGoal < 0 || towGoal >= cells || !towPassable(towGoal)) return;
  let head: i32 = 0;
  let tail: i32 = 0;
  queue[tail++] = towGoal;
  towDistances[towGoal] = 0;
  while (head < tail) {
    const current = queue[head++];
    for (let action: i32 = 1; action < ACTIONS; ++action) {
      const target = neighbor(current, action);
      if (target < 0 || !towPassable(target) || towDistances[target] >= 0) continue;
      towDistances[target] = <i16>(towDistances[current] + 1);
      queue[tail++] = target;
    }
  }
}

function assignTow(): bool {
  if (towCell < 0 || towCell >= cells) return true;
  buildTowDistances();
  const candidates = new Int32Array(ACTIONS);
  let count: i32 = 0;
  for (let action: i32 = 1; action < ACTIONS && towCell != towGoal; ++action) {
    const target = neighbor(towCell, action);
    if (!towPassable(target)) continue;
    const distance = towDistances[target] < 0 ? 32767 : towDistances[target];
    let insert = count;
    while (insert > 0) {
      const prior = candidates[insert - 1];
      const priorDistance = towDistances[prior] < 0 ? 32767 : towDistances[prior];
      if (priorDistance <= distance) break;
      candidates[insert] = prior;
      --insert;
    }
    candidates[insert] = target;
    ++count;
  }
  candidates[count++] = towCell;
  for (let rank: i32 = 0; rank < count; ++rank) {
    const target = candidates[rank];
    const occupant = agentAt[target];
    const reserved = occupiedNext[target];
    const intentionalOverlap =
      (towPickupAgent >= 0 && target == towGoal &&
       occupant == towPickupAgent && reserved == towPickupAgent) ||
      (target == towCell && reserved == towSharedAgent && towSharedAgent >= 0);
    if (reserved >= 0 && !intentionalOverlap) continue;
    if (occupant >= 0 && occupant != agents && !intentionalOverlap &&
        nextPositions[occupant] == towCell) continue;
    const beforeNext = nextPositions.slice();
    const beforeOccupied = occupiedNext.slice();
    occupiedNext[target] = agents;
    nextPositions[agents] = target;
    if (occupant >= 0 && occupant != agents && !intentionalOverlap &&
        nextPositions[occupant] < 0 && !assignPIBT(occupant, 1)) {
      nextPositions.set(beforeNext);
      occupiedNext.set(beforeOccupied);
      continue;
    }
    return true;
  }
  return false;
}

function validJointPlan(): bool {
  const seen = new Int32Array(cells);
  seen.fill(-1);
  const hasTow = towCell >= 0 && towCell < cells;
  const towNext = nextPositions[agents];
  if (hasTow && (towNext < 0 || towNext >= cells)) return false;
  for (let i: i32 = 0; i < agents; ++i) {
    const target = i == towSharedAgent && towCarrying ? towNext : nextPositions[i];
    if (target < 0 || target >= cells || seen[target] >= 0) return false;
    seen[target] = i;
    if (i == towSharedAgent && towCarrying) continue;
    const other = agentAt[target];
    if (other >= 0 && other != i && nextPositions[other] == positions[i] &&
        target != positions[i]) return false;
  }
  if (!hasTow) return true;
  const overlap = seen[towNext];
  return overlap < 0 || overlap == towPickupAgent || overlap == towSharedAgent;
}

export function plan(): void {
  agentAt.fill(-1);
  occupiedNext.fill(-1);
  nextPositions.fill(-1);
  for (let i: i32 = 0; i < agents; ++i) {
    if (i != towSharedAgent) agentAt[positions[i]] = i;
    buildPreferences(i);
    order[i] = i;
  }
  if (towCell >= 0 && towCell < cells) agentAt[towCell] = agents;
  for (let i: i32 = 1; i < agents; ++i) {
    const value = order[i];
    let j = i - 1;
    while (j >= 0 && (priorities[order[j]] < priorities[value] ||
           (priorities[order[j]] == priorities[value] && order[j] > value))) {
      order[j + 1] = order[j];
      --j;
    }
    order[j + 1] = value;
  }
  for (let i: i32 = 0; i < agents; ++i) {
    if (failed[i] != 0 || pinned[i] != 0) {
      nextPositions[i] = positions[i];
      if (i != towSharedAgent || !towCarrying) occupiedNext[positions[i]] = i;
    }
  }
  let valid = assignTow();
  if (valid && manualAgent >= 0 && !assignPIBT(manualAgent, 0)) valid = false;
  for (let rank: i32 = 0; rank < agents; ++rank) {
    const agent = order[rank];
    if (nextPositions[agent] < 0 && !assignPIBT(agent, 0)) valid = false;
  }
  if (valid && !validJointPlan()) valid = false;
  if (!valid) {
    // A blocked inheritance chain cannot be repaired by forcing its agent to
    // wait: another agent may already have reserved that cell. Hold everyone.
    for (let i: i32 = 0; i < agents; ++i) nextPositions[i] = positions[i];
    nextPositions[agents] = towCell;
  }
  for (let i: i32 = 0; i < agents; ++i) {
    const action = actionBetween(positions[i], nextPositions[i]);
    executedActions[i] = action;
    for (let h: i32 = 0; h < 4; ++h) history[i * 5 + h] = history[i * 5 + h + 1];
    history[i * 5 + 4] = <i8>action;
    if (nextPositions[i] != goals[i]) priorities[i] += 1.0;
    else priorities[i] -= <f32>Math.floor(priorities[i]);
  }
}

export function observationsPointer(): usize { return observations.dataStart; }
export function chatPointer(): usize { return chat.dataStart; }
export function probabilitiesPointer(): usize { return probabilities.dataStart; }
export function nextPositionsPointer(): usize { return nextPositions.dataStart; }
export function getTowNext(): i32 { return nextPositions[agents]; }
export function executedActionsPointer(): usize { return executedActions.dataStart; }
