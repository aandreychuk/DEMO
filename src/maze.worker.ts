import * as ort from 'onnxruntime-web/webgpu';
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
  setTow: (current: number, previous: number) => void;
  buildInputs: () => void;
  plan: () => void;
  observationsPointer: () => number;
  chatPointer: () => number;
  probabilitiesPointer: () => number;
  nextPositionsPointer: () => number;
  executedActionsPointer: () => number;
};

const scope = self as unknown as {
  postMessage: (message: unknown) => void;
  onmessage: ((event: MessageEvent<{ type: 'start' | 'advance' | 'reset' | 'configure'; config?: MazeConfig }>) => void) | null;
};

let core: MazeCore;
let session: ort.InferenceSession;
let config = defaultMazeConfig();
let positions = new Int32Array(0);
let goals = new Int32Array(0);
let finished = new Uint8Array(0);
let action = 0;
let backend = 'WASM';
let lastExecuted = new Int32Array(0);
let initialized = false;
let pending: Promise<void> = Promise.resolve();

const observations = new BigInt64Array(MODEL_SLOTS * OBS_TOKENS);
const chat = new BigInt64Array(MODEL_SLOTS * CHAT_SLOTS);
const feeds = {
  observations: new ort.Tensor('int64', observations, [1, MODEL_SLOTS, OBS_TOKENS]),
  chat: new ort.Tensor('int64', chat, [1, MODEL_SLOTS, CHAT_SLOTS]),
};

function post(message: unknown): void { scope.postMessage(message); }

async function loadCore(): Promise<MazeCore> {
  const url = new URL(`${import.meta.env.BASE_URL}runtime/mapf-core.wasm`, self.location.origin);
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
  const url = new URL(`${import.meta.env.BASE_URL}runtime/fastdmm-0.8m-comm-trace.onnx`, self.location.origin);
  if ((navigator as Navigator & { gpu?: unknown }).gpu) {
    try {
      const webgpuSession = await ort.InferenceSession.create(url.href, {
        executionProviders: ['webgpu'],
      });
      backend = 'WebGPU';
      return webgpuSession;
    } catch (error) {
      console.warn('Maze trace WebGPU unavailable; using WASM.', error);
    }
  }
  backend = 'WASM';
  return ort.InferenceSession.create(url.href, { executionProviders: ['wasm'] });
}

function resetScenario(): void {
  const error = validateMazeConfig(config);
  if (error) throw new Error(error);
  const agents = config.agents.length;
  core.configure(MAZE_WIDTH, MAZE_HEIGHT, agents);
  for (const cell of config.walls) core.setWall(cell, 1);
  positions = Int32Array.from(config.agents, (agent) => agent.start);
  goals = Int32Array.from(config.agents, (agent) => agent.goal!);
  finished = new Uint8Array(agents);
  for (let i = 0; i < agents; i++) {
    if (positions[i] !== goals[i]) continue;
    if (config.lifelong) goals[i] = nextGoal(i);
    else finished[i] = 1;
  }
  lastExecuted = new Int32Array(agents);
  action = 0;
}

function syncCore(): void {
  core.setTow(-1, -1);
  for (let i = 0; i < positions.length; i++) {
    core.setAgentState(i, positions[i], goals[i], 0, 0, finished[i]);
  }
}

async function infer(): Promise<void> {
  syncCore();
  core.buildInputs();
  const memory = core.memory.buffer;
  const obs32 = new Int32Array(memory, core.observationsPointer(), observations.length);
  const chat32 = new Int32Array(memory, core.chatPointer(), chat.length);
  for (let i = 0; i < observations.length; i++) observations[i] = BigInt(obs32[i]);
  for (let i = 0; i < chat.length; i++) chat[i] = BigInt(chat32[i]);
  const started = performance.now();
  const output = await session.run(feeds);
  let trace: number[];
  try {
    const tensor = output.round_z;
    if (!tensor || tensor.dims.join(',') !== `${ROUNDS},${MODEL_SLOTS},${ACTIONS}`) {
      throw new Error('The model returned an unexpected communication trace.');
    }
    const all = tensor.data as Float32Array;
    trace = [];
    for (let round = 0; round < ROUNDS; round++) {
      const start = round * MODEL_SLOTS * ACTIONS;
      trace.push(...all.subarray(start, start + positions.length * ACTIONS));
    }
    if (!trace.every(Number.isFinite)) throw new Error('The model returned non-finite z values.');
    const final = trace.slice((ROUNDS - 1) * positions.length * ACTIONS);
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
  } finally {
    for (const tensor of Object.values(output)) tensor.dispose();
  }
  post({
    type: 'snapshot', action, positions: Array.from(positions), goals: Array.from(goals),
    chat: Array.from(chat32.subarray(0, positions.length * CHAT_SLOTS)), trace,
    previousActions: Array.from(lastExecuted), finished: Array.from(finished),
    solved: !config.lifelong && finished.every((value) => value !== 0),
    inferenceMs: performance.now() - started, backend,
  });
}

function nextGoal(agent: number): number {
  const blocked = new Set(config.walls);
  const used = new Set(Array.from(goals).filter((_, index) => index !== agent));
  const seen = new Set([positions[agent]]);
  const queue = [positions[agent]];
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    const x = cell % MAZE_WIDTH;
    const y = Math.floor(cell / MAZE_WIDTH);
    for (const neighbor of [x > 0 ? cell - 1 : -1, x + 1 < MAZE_WIDTH ? cell + 1 : -1, y > 0 ? cell - MAZE_WIDTH : -1, y + 1 < MAZE_HEIGHT ? cell + MAZE_WIDTH : -1]) {
      if (neighbor < 0 || blocked.has(neighbor) || seen.has(neighbor)) continue;
      seen.add(neighbor);
      queue.push(neighbor);
    }
  }
  const choices = queue.filter((cell) => cell !== positions[agent] && !used.has(cell));
  const fallback = queue.filter((cell) => cell !== positions[agent]);
  const candidates = choices.length ? choices : fallback;
  return candidates.length ? candidates[Math.floor(Math.random() * candidates.length)] : positions[agent];
}

async function handle(type: 'start' | 'advance' | 'reset' | 'configure', nextConfig?: MazeConfig): Promise<void> {
  if ((type === 'configure' || type === 'start') && nextConfig) {
    const error = validateMazeConfig(nextConfig);
    if (error) throw new Error(error);
    config = nextConfig;
  }
  if (!initialized) {
    post({ type: 'status', text: 'Loading DMM-0.8M communication trace…' });
    [core, session] = await Promise.all([loadCore(), loadModel()]);
    initialized = true;
    resetScenario();
    await infer();
    return;
  }
  if (type === 'reset' || type === 'configure') {
    resetScenario();
    await infer();
    return;
  }
  if (type !== 'advance') return;
  if (!config.lifelong && finished.every((value) => value !== 0)) return;
  core.plan();
  const next = new Int32Array(core.memory.buffer, core.nextPositionsPointer(), positions.length);
  const executed = new Int32Array(core.memory.buffer, core.executedActionsPointer(), positions.length);
  positions.set(next);
  lastExecuted.set(executed);
  for (let i = 0; i < positions.length; i++) {
    if (positions[i] !== goals[i]) continue;
    if (config.lifelong) goals[i] = nextGoal(i);
    else finished[i] = 1;
  }
  action++;
  await infer();
}

scope.onmessage = (event) => {
  pending = pending.then(() => handle(event.data.type, event.data.config)).catch((error: unknown) => {
    post({ type: 'error', text: error instanceof Error ? error.message : String(error) });
  });
};
