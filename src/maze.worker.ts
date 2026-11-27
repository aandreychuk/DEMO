import * as ort from 'onnxruntime-web';
import { applyMazeActions, bestMazeAction } from './maze-motion';
import { createMazeRandom, fillMazeVoteNoise } from './maze-random';
import { MAZE_HEIGHT, MAZE_MODEL_SLOTS, MAZE_WIDTH, defaultMazeConfig, validateMazeConfig, type MazeConfig } from './maze-scenario';

const ROUNDS = 4;
const ACTIONS = 5;
const OBS_TOKENS = 256;
const CHAT_SLOTS = 13;
const MODEL_SLOTS = MAZE_MODEL_SLOTS;

type MazeCore = {
  memory: WebAssembly.Memory;
  configure: (width: number, height: number, agents: number) => void;
  setWall: (cell: number, value: number) => void;
  setAgentState: (agent: number, position: number, goal: number, loaded: number, failed: number, pinned: number) => void;
  setTow: (current: number, goal: number, pickupAgent: number, sharedAgent: number, carrying: number) => void;
  buildInputs: () => void;
  recordAction: (agent: number, action: number) => void;
  plan: () => void;
  observationsPointer: () => number;
  chatPointer: () => number;
  probabilitiesPointer: () => number;
  nextPositionsPointer: () => number;
  executedActionsPointer: () => number;
};

type Command = { type: 'configure'; config: MazeConfig; seed: number; revision: number } | { type: 'advance'; revision: number };

const scope = self as unknown as {
  postMessage: (message: unknown) => void;
  onmessage: ((event: MessageEvent<Command>) => void) | null;
};

let core: MazeCore;
let session: ort.InferenceSession;
let config = defaultMazeConfig();
let positions = new Int32Array(0);
let goals = new Int32Array(0);
let walls = new Set<number>();
let plannedActions: number[] = [];
let oneShotNextPositions: Int32Array = new Int32Array(0);
let action = 0;
const backend = 'WASM';
let initialized = false;
let pending: Promise<void> = Promise.resolve();
let revision = 0;
let latestRevision = 0;
let voteRandom: (() => number) | null = null;
let goalRandom = createMazeRandom(0);

const observations = new BigInt64Array(MODEL_SLOTS * OBS_TOKENS);
const chat = new BigInt64Array(MODEL_SLOTS * CHAT_SLOTS);
const voteNoise = new Float32Array(ROUNDS * MODEL_SLOTS * ACTIONS);
const feeds = {
  observations: new ort.Tensor('int64', observations, [1, MODEL_SLOTS, OBS_TOKENS]),
  chat: new ort.Tensor('int64', chat, [1, MODEL_SLOTS, CHAT_SLOTS]),
  vote_noise: new ort.Tensor('float32', voteNoise, [ROUNDS, MODEL_SLOTS, ACTIONS]),
};

function post(message: object): void { scope.postMessage({ ...message, revision }); }

async function loadCore(): Promise<MazeCore> {
  const url = new URL(`${import.meta.env.BASE_URL}runtime/mapf-core.wasm?v=2`, self.location.origin);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Maze WASM core could not load (${response.status}).`);
  const result = await WebAssembly.instantiate(await response.arrayBuffer(), {
    env: { abort: (): never => { throw new Error('Maze WASM core aborted.'); } },
  });
  return result.instance.exports as unknown as MazeCore;
}

async function loadModel(): Promise<ort.InferenceSession> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  const url = new URL(`${import.meta.env.BASE_URL}runtime/fastdmm-0.8m-comm-trace.onnx?v=4`, self.location.origin);
  post({ type: 'status', text: 'Loading DMM-0.8M trace on WASM…' });
  return ort.InferenceSession.create(url.href, { executionProviders: ['wasm'] });
}

function resetScenario(): void {
  const error = validateMazeConfig(config);
  if (error) throw new Error(error);
  const agents = config.agents.length;
  walls = new Set(config.walls);
  core.configure(MAZE_WIDTH, MAZE_HEIGHT, agents);
  for (const cell of config.walls) core.setWall(cell, 1);
  positions = Int32Array.from(config.agents, (agent) => agent.start);
  goals = Int32Array.from(config.agents, (agent) => agent.goal!);
  if (config.lifelong) assignNewGoals();
  action = 0;
}

function allGoalsOccupied(): boolean {
  return positions.every((position, agent) => position === goals[agent]);
}

function syncCore(): void {
  core.setTow(-1, -1, -1, -1, 0);
  for (let i = 0; i < positions.length; i++) {
    core.setAgentState(i, positions[i], goals[i], 0, 0, 0);
  }
}

function publishSnapshot(trace: number[], votes: number[], chat: number[], inferenceMs: number, solved = false): void {
  post({
    type: 'snapshot', action, positions: Array.from(positions), goals: Array.from(goals),
    chat, trace, votes, plannedActions, solved, inferenceMs, backend,
  });
}

function chooseActions(trace: number[]): number[] {
  const final = trace.slice((ROUNDS - 1) * positions.length * ACTIONS);
  if (!config.lifelong) {
    const requested = Array.from(positions, (_, i) => bestMazeAction(final.slice(i * ACTIONS, (i + 1) * ACTIONS)));
    oneShotNextPositions = applyMazeActions(positions, requested, walls);
    return requested.map((move, i) => oneShotNextPositions[i] === positions[i] ? 0 : move);
  }
  const probabilities = new Float32Array(core.memory.buffer, core.probabilitiesPointer(), positions.length * ACTIONS);
  for (let agent = 0; agent < positions.length; agent++) {
    const offset = agent * ACTIONS;
    const maxZ = Math.max(...final.slice(offset, offset + ACTIONS));
    let sum = 0;
    for (let action = 0; action < ACTIONS; action++) {
      probabilities[offset + action] = Math.exp(final[offset + action] - maxZ);
      sum += probabilities[offset + action];
    }
    for (let action = 0; action < ACTIONS; action++) probabilities[offset + action] /= sum;
  }
  core.plan();
  return Array.from(new Int32Array(core.memory.buffer, core.executedActionsPointer(), positions.length));
}

async function infer(): Promise<void> {
  if (!config.lifelong && allGoalsOccupied()) {
    plannedActions = [];
    publishSnapshot([], [], Array(positions.length * CHAT_SLOTS).fill(-1), 0, true);
    return;
  }
  syncCore();
  core.buildInputs();
  const memory = core.memory.buffer;
  const obs32 = new Int32Array(memory, core.observationsPointer(), observations.length);
  const chat32 = new Int32Array(memory, core.chatPointer(), chat.length);
  for (let i = 0; i < observations.length; i++) observations[i] = BigInt(obs32[i]);
  for (let i = 0; i < chat.length; i++) chat[i] = BigInt(chat32[i]);
  if (voteRandom) fillMazeVoteNoise(voteNoise, voteRandom);
  const started = performance.now();
  const output = await session.run(feeds);
  let trace: number[];
  let votes: number[];
  try {
    const tensor = output.round_z;
    const voteTensor = output.round_votes;
    if (!tensor || tensor.dims.join(',') !== `${ROUNDS},${MODEL_SLOTS},${ACTIONS}`) {
      throw new Error('The model returned an unexpected communication trace.');
    }
    if (!voteTensor || voteTensor.dims.join(',') !== `${ROUNDS},${MODEL_SLOTS}`) {
      throw new Error('The model returned an unexpected vote trace.');
    }
    const all = tensor.data as Float32Array;
    const allVotes = voteTensor.data as Float32Array;
    trace = [];
    votes = [];
    for (let round = 0; round < ROUNDS; round++) {
      const start = round * MODEL_SLOTS * ACTIONS;
      trace.push(...all.subarray(start, start + positions.length * ACTIONS));
      votes.push(...allVotes.subarray(round * MODEL_SLOTS, round * MODEL_SLOTS + positions.length));
    }
    if (!trace.every(Number.isFinite)) throw new Error('The model returned invalid intent scores.');
    if (!votes.every((vote) => Number.isInteger(vote) && vote >= 0 && vote < ACTIONS)) {
      throw new Error('The model returned invalid communication votes.');
    }
  } finally {
    for (const tensor of Object.values(output)) tensor.dispose();
  }
  const messageSources = Array.from(chat32.subarray(0, positions.length * CHAT_SLOTS));
  const inferenceMs = performance.now() - started;
  plannedActions = chooseActions(trace);
  publishSnapshot(trace, votes, messageSources, inferenceMs);
}

function nextGoal(agent: number): number {
  const used = new Set(Array.from(goals).filter((_, index) => index !== agent));
  const seen = new Set([positions[agent]]);
  const queue = [positions[agent]];
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    const x = cell % MAZE_WIDTH;
    const y = Math.floor(cell / MAZE_WIDTH);
    for (const neighbor of [x > 0 ? cell - 1 : -1, x + 1 < MAZE_WIDTH ? cell + 1 : -1, y > 0 ? cell - MAZE_WIDTH : -1, y + 1 < MAZE_HEIGHT ? cell + MAZE_WIDTH : -1]) {
      if (neighbor < 0 || walls.has(neighbor) || seen.has(neighbor)) continue;
      seen.add(neighbor);
      queue.push(neighbor);
    }
  }
  const choices = queue.filter((cell) => cell !== positions[agent] && !used.has(cell));
  const fallback = queue.filter((cell) => cell !== positions[agent]);
  const candidates = choices.length ? choices : fallback;
  return candidates.length ? candidates[Math.floor(goalRandom() * candidates.length)] : positions[agent];
}

function assignNewGoals(): void {
  for (let i = 0; i < positions.length; i++) {
    if (positions[i] === goals[i]) goals[i] = nextGoal(i);
  }
}

async function handle(command: Command): Promise<void> {
  if (command.type === 'configure') {
    if (!initialized) {
      post({ type: 'status', text: 'Loading DMM-0.8M communication trace…' });
      [core, session] = await Promise.all([loadCore(), loadModel()]);
      initialized = true;
    }
    if (revision !== latestRevision) return;
    config = command.config;
    voteRandom = command.seed === 0 ? null : createMazeRandom(command.seed);
    goalRandom = createMazeRandom(command.seed ^ 0x9e3779b9);
    voteNoise.fill(0);
    resetScenario();
    post({ type: 'status', text: 'Computing the first communication trace…' });
    await infer();
    return;
  }
  if (config.lifelong) {
    positions.set(new Int32Array(core.memory.buffer, core.nextPositionsPointer(), positions.length));
    assignNewGoals();
  } else {
    if (allGoalsOccupied()) return;
    for (let i = 0; i < positions.length; i++) core.recordAction(i, plannedActions[i]);
    positions.set(oneShotNextPositions);
  }
  action++;
  await infer();
}

scope.onmessage = (event) => {
  const command = event.data;
  latestRevision = Math.max(latestRevision, command.revision);
  pending = pending.then(() => {
    if (command.revision !== latestRevision) return;
    revision = command.revision;
    return handle(command);
  }).catch((error: unknown) => {
    post({ type: 'error', text: error instanceof Error ? error.message : String(error) });
  });
};
