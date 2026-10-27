import * as ort from 'onnxruntime-web/webgpu';
import { MAZE_HEIGHT, MAZE_ROUTES, MAZE_ROWS, MAZE_WIDTH, mazeCell } from './maze-scenario';

const AGENTS = MAZE_ROUTES.length;
const ROUNDS = 4;
const ACTIONS = 5;
const OBS_TOKENS = 256;
const CHAT_SLOTS = 13;

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
  onmessage: ((event: MessageEvent<{ type: 'start' | 'advance' | 'reset' }>) => void) | null;
};

let core: MazeCore;
let session: ort.InferenceSession;
let positions = new Int32Array(AGENTS);
let goals = new Int32Array(AGENTS);
let action = 0;
let backend = 'WASM';
let lastExecuted = new Int32Array(AGENTS);
let initialized = false;
let pending: Promise<void> = Promise.resolve();

const observations = new BigInt64Array(AGENTS * OBS_TOKENS);
const chat = new BigInt64Array(AGENTS * CHAT_SLOTS);
const neighborPadding = new Uint8Array(AGENTS * CHAT_SLOTS);
const feeds = {
  observations: new ort.Tensor('int64', observations, [1, AGENTS, OBS_TOKENS]),
  chat: new ort.Tensor('int64', chat, [1, AGENTS, CHAT_SLOTS]),
  neighbor_padding: new ort.Tensor('bool', neighborPadding, [1, AGENTS, CHAT_SLOTS]),
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
  core.configure(MAZE_WIDTH, MAZE_HEIGHT, AGENTS);
  for (let y = 0; y < MAZE_HEIGHT; y++) {
    for (let x = 0; x < MAZE_WIDTH; x++) {
      if (MAZE_ROWS[y][x] === '#') core.setWall(mazeCell(x, y), 1);
    }
  }
  for (let i = 0; i < AGENTS; i++) {
    const [start, goal] = MAZE_ROUTES[i];
    if (MAZE_ROWS[start[1]][start[0]] !== '.' || MAZE_ROWS[goal[1]][goal[0]] !== '.') {
      throw new Error(`Maze route ${i + 1} is on a wall.`);
    }
    positions[i] = mazeCell(...start);
    goals[i] = mazeCell(...goal);
  }
  lastExecuted.fill(0);
  action = 0;
}

function syncCore(): void {
  core.setTow(-1, -1);
  for (let i = 0; i < AGENTS; i++) {
    core.setAgentState(i, positions[i], goals[i], 0, 0, 0);
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
  for (let agent = 0; agent < AGENTS; agent++) {
    for (let slot = 0; slot < CHAT_SLOTS; slot++) {
      let padded = 1;
      const start = agent * OBS_TOKENS + 121 + slot * 10;
      for (let token = 0; token < 10; token++) {
        if (obs32[start + token] !== 66) { padded = 0; break; }
      }
      neighborPadding[agent * CHAT_SLOTS + slot] = padded;
    }
  }
  const started = performance.now();
  const output = await session.run(feeds);
  let trace: number[];
  try {
    const tensor = output.round_intents;
    if (!tensor || tensor.dims.join(',') !== `${ROUNDS},${AGENTS},${ACTIONS}`) {
      throw new Error('The model returned an unexpected communication trace.');
    }
    trace = Array.from(tensor.data as Float32Array);
    const final = trace.slice((ROUNDS - 1) * AGENTS * ACTIONS);
    new Float32Array(core.memory.buffer, core.probabilitiesPointer(), AGENTS * ACTIONS).set(final);
  } finally {
    for (const tensor of Object.values(output)) tensor.dispose();
  }
  post({
    type: 'snapshot', action, positions: Array.from(positions), goals: Array.from(goals),
    chat: Array.from(chat32), trace, previousActions: Array.from(lastExecuted),
    inferenceMs: performance.now() - started, backend,
  });
}

async function handle(type: 'start' | 'advance' | 'reset'): Promise<void> {
  if (!initialized) {
    post({ type: 'status', text: 'Loading DMM-0.8M communication trace…' });
    [core, session] = await Promise.all([loadCore(), loadModel()]);
    initialized = true;
    resetScenario();
    await infer();
    return;
  }
  if (type === 'reset') {
    resetScenario();
    await infer();
    return;
  }
  if (type !== 'advance') return;
  core.plan();
  const next = new Int32Array(core.memory.buffer, core.nextPositionsPointer(), AGENTS);
  const executed = new Int32Array(core.memory.buffer, core.executedActionsPointer(), AGENTS);
  positions.set(next);
  lastExecuted.set(executed);
  for (let i = 0; i < AGENTS; i++) {
    if (positions[i] !== goals[i]) continue;
    const [home, destination] = MAZE_ROUTES[i];
    const homeCell = mazeCell(...home);
    goals[i] = goals[i] === homeCell ? mazeCell(...destination) : homeCell;
  }
  action++;
  await infer();
}

scope.onmessage = (event) => {
  pending = pending.then(() => handle(event.data.type)).catch((error: unknown) => {
    post({ type: 'error', text: error instanceof Error ? error.message : String(error) });
  });
};
