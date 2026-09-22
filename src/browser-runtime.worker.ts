import * as ort from 'onnxruntime-web/webgpu';
import scenarioText from '../runtime/scenarios/warehouse-lifelong-44x33.scen?raw';
import tasksText from '../runtime/scenarios/warehouse-lifelong-44x33.tasks.tsv?raw';
import initialLayout from '../runtime/scenarios/warehouse-lifelong-44x33.layout.json';

const MAX_AGENTS = 100;
const OBS_TOKENS = 256;
const CHAT_SLOTS = 13;
const ACTIONS = 5;
const PALLET_CAPACITY = 12;
const PALLET_DWELL = 2;
const UNLOAD_DWELL = 5;
const RELOAD_DWELL = 5;
const REPAIR_DWELL = 8;
const TOW_LOADING_DWELL = 3;
const TICK_RATE = 10;
const FRAME_MAGIC = 0x4d415046;
const PROTOCOL = 5;

type ServiceCell = { id?: number; x: number; y: number; accessSide?: string };
type PalletRecord = { id: number; x: number; y: number; cargoType: number };
type Layout = {
  schema: string;
  width: number;
  height: number;
  palletCapacity: number;
  pallets: PalletRecord[];
  stations: ServiceCell[];
  reloadStations: ServiceCell[];
  repairStation: ServiceCell;
  towDepot: ServiceCell;
};
type Task = {
  id: number;
  palletId: number;
  palletX: number;
  palletY: number;
  stationX: number;
  stationY: number;
  reloadX: number;
  reloadY: number;
};
type WasmCore = {
  memory: WebAssembly.Memory;
  configure: (width: number, height: number, agents: number) => void;
  setWall: (cell: number, value: number) => void;
  setStation: (cell: number, value: number) => void;
  clearPallets: () => void;
  setPallet: (cell: number, value: number) => void;
  setAgentState: (index: number, position: number, goal: number, loaded: number, failed: number, pinned: number) => void;
  setTow: (current: number, previous: number) => void;
  resetHistory: () => void;
  buildInputs: () => void;
  plan: () => void;
  observationsPointer: () => number;
  chatPointer: () => number;
  probabilitiesPointer: () => number;
  nextPositionsPointer: () => number;
  executedActionsPointer: () => number;
};
type ControlMessage = {
  type: 'control';
  action: string;
  agents?: number;
  agent?: number;
  value?: number;
  pallets?: Array<{ x: number; y: number }>;
};

type BrowserGpuAdapterInfo = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

type BrowserGpuDevice = {
  adapterInfo?: BrowserGpuAdapterInfo;
  destroy?: () => void;
};

type BrowserGpuAdapter = {
  info?: BrowserGpuAdapterInfo;
  isFallbackAdapter?: boolean;
  features?: { has: (feature: string) => boolean };
  requestDevice: (descriptor?: { requiredFeatures?: string[] }) => Promise<BrowserGpuDevice>;
};

type BrowserGpu = {
  requestAdapter: (options?: {
    powerPreference?: 'low-power' | 'high-performance';
    forceFallbackAdapter?: boolean;
  }) => Promise<BrowserGpuAdapter | null>;
};

const workerScope = self as unknown as {
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<ControlMessage>) => void) | null;
  setTimeout: (handler: () => void, timeout?: number) => number;
  clearTimeout: (id: number) => void;
};

let layout = structuredClone(initialLayout) as Layout;
const map = buildMap(layout.width, layout.height);
const scenarioStarts = parseScenario(scenarioText);
let taskTemplates = parseTasks(tasksText);
let core: WasmCore;
let session: ort.InferenceSession;
let booted = false;
let requestedAgentCount = MAX_AGENTS;
let backend = 'WASM';
let backendAdapter = '';
let agentCount = MAX_AGENTS;
let speed = 1;
let paused = false;
let timer: number | null = null;
let stepping = false;
let generation = 0;
let step = 0;
let completedTasks = 0;
let nextTaskId = 0;
let lastInferenceMs = 0;
const INFERENCE_WINDOW_SIZE = 32;
const inferenceWindow = new Float64Array(INFERENCE_WINDOW_SIZE);
let inferenceWindowIndex = 0;
let inferenceWindowCount = 0;
let inferenceWindowTotal = 0;

// The model shape is fixed at 100 agents. Reuse the CPU-side input tensors for
// every tick instead of allocating three typed arrays and three Tensor wrappers
// per inference. This also keeps long-running browser simulations out of the
// garbage collector's allocation path.
const observationInput = new BigInt64Array(MAX_AGENTS * OBS_TOKENS);
const chatInput = new BigInt64Array(MAX_AGENTS * CHAT_SLOTS);
const neighborPaddingInput = new Uint8Array(MAX_AGENTS * CHAT_SLOTS);
const inferenceFeeds = {
  observations: new ort.Tensor('int64', observationInput, [1, MAX_AGENTS, OBS_TOKENS]),
  chat: new ort.Tensor('int64', chatInput, [1, MAX_AGENTS, CHAT_SLOTS]),
  neighbor_padding: new ort.Tensor('bool', neighborPaddingInput, [1, MAX_AGENTS, CHAT_SLOTS]),
};

let positions = new Int32Array(MAX_AGENTS);
let goals = new Int32Array(MAX_AGENTS);
let taskIndex = new Int32Array(MAX_AGENTS);
let taskIds = new Int32Array(MAX_AGENTS);
let stages = new Uint8Array(MAX_AGENTS);
let loaded = new Uint8Array(MAX_AGENTS);
let dwell = new Int16Array(MAX_AGENTS);
let reloadStarted = new Uint8Array(MAX_AGENTS);
let returnStarted = new Uint8Array(MAX_AGENTS);
let requiresReload = new Uint8Array(MAX_AGENTS);
let failed = new Uint8Array(MAX_AGENTS);
let recovery = new Uint8Array(MAX_AGENTS);
let resumeGoals = new Int32Array(MAX_AGENTS);
let recoveringPallet = new Uint8Array(MAX_AGENTS);
let recoveryDwellStarted = new Uint8Array(MAX_AGENTS);
let repairDwell = new Int16Array(MAX_AGENTS);

let palletHome = new Int32Array(0);
let palletPositions = new Int32Array(0);
let palletItems = new Uint8Array(0);
let palletReserved = new Uint8Array(0);
let palletPresent = new Uint8Array(0);
let pendingTasks: number[] = [];
let pendingCursor = 0;

let towPosition = -1;
let towPrevious = -1;
let towState = 0;
let towTarget = -1;
let towLoadingDwell = 0;
let rescueQueue: number[] = [];

let previousFramePositions: Int32Array | null = null;
let previousFrameStages: Uint8Array | null = null;
let previousFramePalletIds: Int32Array | null = null;

function post(message: unknown, transfer?: Transferable[]): void {
  workerScope.postMessage(message, transfer);
}

function buildMap(width: number, height: number): { width: number; height: number; blocked: Uint8Array } {
  const blocked = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sideWall = x < 2 || x >= width - 2;
      const serviceEndCap = (y === 0 || y === height - 1) && x >= width - 4;
      blocked[y * width + x] = sideWall || serviceEndCap ? 1 : 0;
    }
  }
  return { width, height, blocked };
}

function parseScenario(source: string): number[] {
  return source.trim().split(/\r?\n/).slice(1).map((line) => {
    const fields = line.split('\t');
    return Number(fields[5]) * map.width + Number(fields[4]);
  });
}

function parseTasks(source: string): Task[] {
  return source.trim().split(/\r?\n/).slice(1).map((line) => {
    const v = line.split('\t').map(Number);
    return { id: v[0], palletId: v[1], palletX: v[2], palletY: v[3], stationX: v[4], stationY: v[5], reloadX: v[6], reloadY: v[7] };
  });
}

function cell(x: number, y: number): number { return y * map.width + x; }
function xOf(index: number): number { return index % map.width; }
function yOf(index: number): number { return Math.floor(index / map.width); }

function serviceCells(): number[] {
  return [
    ...layout.stations.map((entry) => cell(entry.x, entry.y)),
    ...layout.reloadStations.map((entry) => cell(entry.x, entry.y)),
    cell(layout.repairStation.x, layout.repairStation.y),
    cell(layout.towDepot.x, layout.towDepot.y),
  ];
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function shuffled<T>(values: T[], random: () => number): T[] {
  const result = values.slice();
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function rebuildScenarioForLayout(): void {
  const random = mulberry32(800_000);
  const services = new Set(serviceCells());
  const occupied = new Set(layout.pallets.map((pallet) => cell(pallet.x, pallet.y)));
  const pool: number[] = [];
  for (let y = 1; y < map.height - 1; y++) {
    for (let x = 1; x < map.width - 1; x++) {
      const index = cell(x, y);
      if (!map.blocked[index] && !services.has(index) && !occupied.has(index)) pool.push(index);
    }
  }
  const generatedStarts = shuffled(pool, random).slice(0, MAX_AGENTS);
  scenarioStarts.splice(0, scenarioStarts.length, ...generatedStarts);
  const stations = layout.stations;
  const reloads = layout.reloadStations;
  const tasks: Task[] = [];
  while (tasks.length < 4000) {
    const cycle = shuffled(layout.pallets, random);
    for (const pallet of cycle) {
      if (tasks.length >= 4000) break;
      const station = stations[Math.floor(random() * stations.length)];
      const reload = reloads[Math.floor(random() * reloads.length)];
      tasks.push({ id: tasks.length, palletId: pallet.id, palletX: pallet.x, palletY: pallet.y,
        stationX: station.x, stationY: station.y, reloadX: reload.x, reloadY: reload.y });
    }
  }
  taskTemplates = tasks;
}

async function loadWasm(): Promise<WasmCore> {
  const response = await fetch(new URL(`${import.meta.env.BASE_URL}runtime/mapf-core.wasm`, self.location.origin));
  if (!response.ok) throw new Error(`WASM core download failed: ${response.status}`);
  const imports = { env: { abort: (): never => { throw new Error('MAPF WASM core aborted'); } } };
  const result = await WebAssembly.instantiate(await response.arrayBuffer(), imports);
  return result.instance.exports as unknown as WasmCore;
}

async function loadPolicy(): Promise<ort.InferenceSession> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  const fp32ModelUrl = new URL(`${import.meta.env.BASE_URL}runtime/fastdmm-0.8m.onnx`, self.location.origin).href;
  const fp16ModelUrl = new URL(`${import.meta.env.BASE_URL}runtime/fastdmm-0.8m-fp16.onnx`, self.location.origin).href;
  const fusedFp32ModelUrl = new URL(`${import.meta.env.BASE_URL}runtime/fastdmm-0.8m-webgpu.onnx`, self.location.origin).href;
  const fusedFp16ModelUrl = new URL(`${import.meta.env.BASE_URL}runtime/fastdmm-0.8m-webgpu-fp16.onnx`, self.location.origin).href;
  const gpu = (navigator as Navigator & { gpu?: BrowserGpu }).gpu;
  if (gpu) {
    let device: BrowserGpuDevice | null = null;
    try {
      const adapter = await gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false,
      });
      if (!adapter) throw new Error('No WebGPU adapter was returned by the browser.');
      const supportsFp16 = adapter.features?.has('shader-f16') === true;
      const requiredFeatures: string[] = [];
      if (supportsFp16) requiredFeatures.push('shader-f16');
      if (adapter.features?.has('subgroups')) requiredFeatures.push('subgroups');
      device = await adapter.requestDevice({ requiredFeatures });

      // Supplying the exact device prevents ONNX Runtime from making a second,
      // opaque adapter choice. Browsers may otherwise honor powerPreference
      // differently and run the same build on different hardware.
      ort.env.webgpu.device = device;
      const sessionOptions = {
        executionProviders: [{
          name: 'webgpu' as const,
          device,
          preferredLayout: 'NHWC' as const,
          validationMode: 'wgpuOnly' as const,
          storageBufferCacheMode: 'simple' as const,
        }],
      };
      let precision = 'FP32';
      let webgpuSession: ort.InferenceSession;
      if (supportsFp16) {
        try {
          webgpuSession = await ort.InferenceSession.create(fusedFp16ModelUrl, sessionOptions);
          precision = 'FP16 FUSED';
        } catch (error) {
          console.warn('Fused FP16 WebGPU policy unavailable; retrying the portable FP16 graph.', error);
          try {
            webgpuSession = await ort.InferenceSession.create(fp16ModelUrl, sessionOptions);
            precision = 'FP16';
          } catch (fallbackError) {
            console.warn('FP16 WebGPU policy unavailable; retrying the FP32 graph.', fallbackError);
            webgpuSession = await ort.InferenceSession.create(fp32ModelUrl, sessionOptions);
          }
        }
      } else {
        try {
          webgpuSession = await ort.InferenceSession.create(fusedFp32ModelUrl, sessionOptions);
          precision = 'FP32 FUSED';
        } catch (error) {
          console.warn('Fused FP32 WebGPU policy unavailable; retrying the portable graph.', error);
          webgpuSession = await ort.InferenceSession.create(fp32ModelUrl, sessionOptions);
        }
      }
      backend = 'WEBGPU';
      const info = device.adapterInfo ?? adapter.info;
      backendAdapter = `${formatAdapter(info, adapter.isFallbackAdapter === true)} · ${precision}`;
      return webgpuSession;
    } catch (error) {
      device?.destroy?.();
      console.warn('WebGPU provider unavailable for this graph; using ONNX Runtime WASM.', error);
    }
  }
  backend = 'WASM';
  backendAdapter = '';
  return ort.InferenceSession.create(fp32ModelUrl, { executionProviders: ['wasm'] });
}

function formatAdapter(info: BrowserGpuAdapterInfo | undefined, fallback: boolean): string {
  const name = info?.description?.trim()
    || [info?.vendor, info?.architecture, info?.device].filter(Boolean).join(' ').trim()
    || 'GPU DETAILS HIDDEN';
  return fallback ? `${name} · SOFTWARE` : name;
}

function initializePallets(): void {
  const count = layout.pallets.reduce((max, pallet) => Math.max(max, pallet.id + 1), 0);
  palletHome = new Int32Array(count);
  palletPositions = new Int32Array(count);
  palletItems = new Uint8Array(count);
  palletReserved = new Uint8Array(count);
  palletPresent = new Uint8Array(count);
  palletHome.fill(-1);
  palletPositions.fill(-1);
  palletItems.fill(PALLET_CAPACITY);
  palletPresent.fill(1);
  for (const pallet of layout.pallets) {
    palletHome[pallet.id] = cell(pallet.x, pallet.y);
    palletPositions[pallet.id] = palletHome[pallet.id];
  }
}

function assignTask(agent: number): boolean {
  const candidates = pendingTasks.length;
  for (let attempt = 0; attempt < candidates; attempt++) {
    const index = pendingTasks[pendingCursor];
    pendingCursor = (pendingCursor + 1) % pendingTasks.length;
    const task = taskTemplates[index];
    if (palletReserved[task.palletId] || palletPositions[task.palletId] === positions[agent]) continue;
    palletReserved[task.palletId] = 1;
    taskIndex[agent] = index;
    taskIds[agent] = nextTaskId++;
    stages[agent] = 0;
    loaded[agent] = 0;
    dwell[agent] = 0;
    reloadStarted[agent] = 0;
    returnStarted[agent] = 0;
    requiresReload[agent] = 0;
    recovery[agent] = 0;
    resumeGoals[agent] = -1;
    recoveringPallet[agent] = 0;
    recoveryDwellStarted[agent] = 0;
    repairDwell[agent] = 0;
    goals[agent] = palletPositions[task.palletId];
    return true;
  }
  return false;
}

function resetSimulation(count: number): void {
  generation++;
  agentCount = Math.max(2, Math.min(MAX_AGENTS, count));
  positions = new Int32Array(MAX_AGENTS);
  goals = new Int32Array(MAX_AGENTS);
  taskIndex = new Int32Array(MAX_AGENTS); taskIndex.fill(-1);
  taskIds = new Int32Array(MAX_AGENTS); taskIds.fill(-1);
  stages = new Uint8Array(MAX_AGENTS);
  loaded = new Uint8Array(MAX_AGENTS);
  dwell = new Int16Array(MAX_AGENTS);
  reloadStarted = new Uint8Array(MAX_AGENTS);
  returnStarted = new Uint8Array(MAX_AGENTS);
  requiresReload = new Uint8Array(MAX_AGENTS);
  failed = new Uint8Array(MAX_AGENTS);
  recovery = new Uint8Array(MAX_AGENTS);
  resumeGoals = new Int32Array(MAX_AGENTS); resumeGoals.fill(-1);
  recoveringPallet = new Uint8Array(MAX_AGENTS);
  recoveryDwellStarted = new Uint8Array(MAX_AGENTS);
  repairDwell = new Int16Array(MAX_AGENTS);
  initializePallets();
  pendingTasks = taskTemplates.map((_, index) => index);
  pendingCursor = 0;
  nextTaskId = 0;
  for (let i = 0; i < agentCount; i++) positions[i] = scenarioStarts[i];
  for (let i = 0; i < agentCount; i++) {
    if (!assignTask(i)) throw new Error('Not enough distinct pallets for active agents.');
  }
  towPosition = cell(layout.towDepot.x, layout.towDepot.y);
  towPrevious = -1;
  towState = 0;
  towTarget = -1;
  towLoadingDwell = 0;
  rescueQueue = [];
  step = 0;
  completedTasks = 0;
  lastInferenceMs = 0;
  inferenceWindow.fill(0);
  inferenceWindowIndex = 0;
  inferenceWindowCount = 0;
  inferenceWindowTotal = 0;
  previousFramePositions = null;
  previousFrameStages = null;
  previousFramePalletIds = null;

  core.configure(map.width, map.height, agentCount);
  for (let index = 0; index < map.blocked.length; index++) core.setWall(index, map.blocked[index]);
  for (const service of serviceCells()) core.setStation(service, 1);
  core.resetHistory();
  syncCore();
}

function syncCore(): void {
  core.clearPallets();
  for (let id = 0; id < palletPresent.length; id++) {
    if (palletPresent[id] && palletPositions[id] >= 0) core.setPallet(palletPositions[id], 1);
  }
  for (let i = 0; i < agentCount; i++) {
    core.setAgentState(i, positions[i], goals[i], loaded[i], failed[i], failed[i] || dwell[i] > 0 ? 1 : 0);
  }
  core.setTow(towPosition, towPrevious);
}

function dispatchTow(): void {
  if (towState !== 0) return;
  while (rescueQueue.length) {
    const candidate = rescueQueue.shift()!;
    if (candidate >= 0 && candidate < agentCount && recovery[candidate] === 1) {
      towTarget = candidate;
      towState = 1;
      return;
    }
  }
}

function towNext(start: number, goalsList: number[]): number {
  const isGoal = new Uint8Array(map.blocked.length);
  for (const goal of goalsList) if (goal >= 0) isGoal[goal] = 1;
  if (isGoal[start]) return start;
  const blocked = map.blocked.slice();
  for (const service of serviceCells()) blocked[service] = 1;
  for (let id = 0; id < palletPresent.length; id++) {
    if (palletPresent[id] && palletPositions[id] >= 0) blocked[palletPositions[id]] = 1;
  }
  blocked[start] = 0;
  for (const goal of goalsList) if (goal >= 0) blocked[goal] = 0;
  const parent = new Int32Array(blocked.length); parent.fill(-1);
  const queue = new Int32Array(blocked.length);
  let head = 0; let tail = 0;
  queue[tail++] = start; parent[start] = start;
  let reached = -1;
  const offsets = [-map.width, map.width, -1, 1];
  while (head < tail && reached < 0) {
    const current = queue[head++];
    for (const offset of offsets) {
      const next = current + offset;
      if (next < 0 || next >= blocked.length) continue;
      if ((offset === -1 || offset === 1) && yOf(next) !== yOf(current)) continue;
      if (blocked[next] || parent[next] >= 0) continue;
      parent[next] = current;
      if (isGoal[next]) { reached = next; break; }
      queue[tail++] = next;
    }
  }
  if (reached < 0) return start;
  while (parent[reached] !== start) reached = parent[reached];
  return reached;
}

function advanceRecovery(): void {
  towPrevious = -1;
  for (let i = 0; i < agentCount; i++) {
    if (recovery[i] !== 3) continue;
    if (repairDwell[i] > 0) { repairDwell[i]--; continue; }
    failed[i] = 0;
    if (recoveringPallet[i] && taskIndex[i] >= 0) {
      const palletId = taskTemplates[taskIndex[i]].palletId;
      recovery[i] = 4;
      recoveryDwellStarted[i] = 0;
      goals[i] = palletPositions[palletId];
    } else {
      recovery[i] = 0;
      goals[i] = resumeGoals[i] >= 0 ? resumeGoals[i] : positions[i];
      resumeGoals[i] = -1;
    }
  }
  dispatchTow();
  if (towState === 1 && towTarget >= 0) {
    const pickup = positions[towTarget];
    const next = towNext(towPosition, [pickup]);
    if (next !== towPosition) { towPrevious = towPosition; towPosition = next; }
    if (towPosition === pickup) { towState = 4; towLoadingDwell = TOW_LOADING_DWELL; }
  } else if (towState === 4 && towTarget >= 0) {
    if (towLoadingDwell > 0) towLoadingDwell--;
    if (towLoadingDwell === 0) {
      positions[towTarget] = towPosition;
      goals[towTarget] = towPosition;
      recovery[towTarget] = 2;
      towState = 2;
    }
  } else if (towState === 2 && towTarget >= 0) {
    const repair = cell(layout.repairStation.x, layout.repairStation.y);
    const next = towNext(towPosition, [repair]);
    if (next !== towPosition) {
      towPrevious = towPosition; towPosition = next;
      positions[towTarget] = towPosition; goals[towTarget] = towPosition;
    }
    if (towPosition === repair) {
      positions[towTarget] = repair; goals[towTarget] = repair;
      recovery[towTarget] = 3; repairDwell[towTarget] = REPAIR_DWELL;
      towTarget = -1; towState = 3;
    }
  } else if (towState === 3) {
    const depot = cell(layout.towDepot.x, layout.towDepot.y);
    const next = towNext(towPosition, [depot]);
    if (next !== towPosition) { towPrevious = towPosition; towPosition = next; }
    if (towPosition === depot) { towState = 0; dispatchTow(); }
  }
}

async function inferAndPlan(expectedGeneration: number): Promise<boolean> {
  syncCore();
  core.buildInputs();
  const memory = core.memory.buffer;
  const obs32 = new Int32Array(memory, core.observationsPointer(), MAX_AGENTS * OBS_TOKENS);
  const chat32 = new Int32Array(memory, core.chatPointer(), MAX_AGENTS * CHAT_SLOTS);
  for (let i = 0; i < obs32.length; i++) observationInput[i] = BigInt(obs32[i]);
  for (let i = 0; i < chat32.length; i++) chatInput[i] = BigInt(chat32[i]);
  for (let agent = 0; agent < MAX_AGENTS; agent++) {
    for (let slot = 0; slot < CHAT_SLOTS; slot++) {
      let padded = 1;
      const start = agent * OBS_TOKENS + 121 + slot * 10;
      for (let feature = 0; feature < 10; feature++) {
        if (obs32[start + feature] !== 66) { padded = 0; break; }
      }
      neighborPaddingInput[agent * CHAT_SLOTS + slot] = padded;
    }
  }
  const started = performance.now();
  const output = await session.run(inferenceFeeds);
  try {
    if (generation !== expectedGeneration) return false;
    const elapsed = performance.now() - started;
    if (step >= 5) {
      if (inferenceWindowCount === INFERENCE_WINDOW_SIZE) {
        inferenceWindowTotal -= inferenceWindow[inferenceWindowIndex];
      } else {
        inferenceWindowCount++;
      }
      inferenceWindow[inferenceWindowIndex] = elapsed;
      inferenceWindowTotal += elapsed;
      inferenceWindowIndex = (inferenceWindowIndex + 1) % INFERENCE_WINDOW_SIZE;
      lastInferenceMs = inferenceWindowTotal / inferenceWindowCount;
    } else {
      lastInferenceMs = elapsed;
    }
    const values = output.action_probabilities.data as Float32Array;
    new Float32Array(core.memory.buffer, core.probabilitiesPointer(), MAX_AGENTS * ACTIONS).set(values);
  } finally {
    // ONNX Runtime keeps backend resources alive until output tensors are
    // disposed. Omitting this in an endless simulation gradually slows WebGPU.
    for (const tensor of Object.values(output)) tensor.dispose();
  }
  core.plan();
  const next = new Int32Array(core.memory.buffer, core.nextPositionsPointer(), MAX_AGENTS);
  for (let i = 0; i < agentCount; i++) positions[i] = next[i];
  return true;
}

function processTaskArrivals(): void {
  for (let i = 0; i < agentCount; i++) if (dwell[i] > 0) dwell[i]--;
  for (let i = 0; i < agentCount; i++) {
    if (failed[i] || taskIndex[i] < 0 || positions[i] !== goals[i]) continue;
    const task = taskTemplates[taskIndex[i]];
    const palletId = task.palletId;
    if (recovery[i] === 4) {
      if (!recoveryDwellStarted[i]) { recoveryDwellStarted[i] = 1; dwell[i] = PALLET_DWELL; continue; }
      if (dwell[i] > 0) continue;
      loaded[i] = 1; palletPresent[palletId] = 0; recovery[i] = 0;
      recoveringPallet[i] = 0; recoveryDwellStarted[i] = 0;
      goals[i] = resumeGoals[i] >= 0 ? resumeGoals[i] : palletHome[palletId];
      resumeGoals[i] = -1;
      continue;
    }
    if (stages[i] === 0) {
      loaded[i] = 1; palletPresent[palletId] = 0; stages[i] = 1;
      dwell[i] = PALLET_DWELL; goals[i] = cell(task.stationX, task.stationY);
    } else if (stages[i] === 1) {
      palletItems[palletId] = Math.max(0, palletItems[palletId] - 1);
      requiresReload[i] = palletItems[palletId] === 0 ? 1 : 0;
      dwell[i] = UNLOAD_DWELL; reloadStarted[i] = 0; returnStarted[i] = 0;
      if (requiresReload[i]) { stages[i] = 2; goals[i] = cell(task.reloadX, task.reloadY); }
      else { stages[i] = 3; goals[i] = palletHome[palletId]; }
    } else if (stages[i] === 2) {
      if (!reloadStarted[i]) { reloadStarted[i] = 1; dwell[i] = RELOAD_DWELL; continue; }
      if (dwell[i] > 0) continue;
      palletItems[palletId] = PALLET_CAPACITY; stages[i] = 3; reloadStarted[i] = 0;
      goals[i] = palletHome[palletId];
    } else {
      if (!returnStarted[i]) { returnStarted[i] = 1; dwell[i] = PALLET_DWELL; continue; }
      if (dwell[i] > 0) continue;
      loaded[i] = 0; palletPresent[palletId] = 1; palletPositions[palletId] = palletHome[palletId];
      palletReserved[palletId] = 0; taskIndex[i] = -1; taskIds[i] = -1;
      dwell[i] = 0; reloadStarted[i] = 0; returnStarted[i] = 0; requiresReload[i] = 0;
      completedTasks++;
      if (!assignTask(i)) goals[i] = positions[i];
    }
  }
}

function currentPalletId(agent: number): number {
  return taskIndex[agent] >= 0 ? taskTemplates[taskIndex[agent]].palletId : -1;
}

function encodeFrame(): ArrayBuffer {
  const recordSize = 36;
  const buffer = new ArrayBuffer(16 + recordSize * agentCount + 16);
  const view = new DataView(buffer);
  view.setUint32(0, FRAME_MAGIC, true);
  view.setUint16(4, PROTOCOL, true);
  view.setUint16(6, Math.min(65535, completedTasks), true);
  view.setUint32(8, step, true);
  view.setUint32(12, agentCount, true);
  const framePalletIds = new Int32Array(agentCount);
  for (let i = 0; i < agentCount; i++) {
    const task = taskIndex[i] >= 0 ? taskTemplates[taskIndex[i]] : null;
    const palletId = task ? task.palletId : -1;
    framePalletIds[i] = palletId;
    const waiting = previousFramePositions !== null && previousFramePositions[i] === positions[i];
    const transitioned = previousFrameStages !== null &&
      (previousFrameStages[i] !== stages[i] || previousFramePalletIds![i] !== palletId);
    const status = (waiting ? 1 : 0) | (loaded[i] ? 2 : 0) | (transitioned ? 4 : 0) |
      (requiresReload[i] ? 8 : 0) | (failed[i] ? 16 : 0) | (recovery[i] === 4 ? 32 : 0);
    const offset = 16 + i * recordSize;
    view.setUint32(offset, i, true);
    view.setFloat32(offset + 4, xOf(positions[i]), true);
    view.setFloat32(offset + 8, yOf(positions[i]), true);
    view.setUint8(offset + 12, status);
    view.setUint8(offset + 13, stages[i]);
    view.setUint16(offset + 14, palletId < 0 ? 0xffff : palletId, true);
    view.setUint32(offset + 16, taskIds[i] < 0 ? 0xffffffff : taskIds[i], true);
    view.setInt16(offset + 20, task?.stationX ?? -1, true);
    view.setInt16(offset + 22, task?.stationY ?? -1, true);
    view.setInt16(offset + 24, task?.reloadX ?? -1, true);
    view.setInt16(offset + 26, task?.reloadY ?? -1, true);
    view.setUint8(offset + 28, palletId >= 0 ? palletItems[palletId] : 0);
    view.setUint8(offset + 29, PALLET_CAPACITY);
    view.setUint8(offset + 30, recovery[i]);
    view.setUint8(offset + 31, 0);
    const physicalPallet = palletId >= 0 ? palletPositions[palletId] : -1;
    view.setInt16(offset + 32, physicalPallet >= 0 ? xOf(physicalPallet) : -1, true);
    view.setInt16(offset + 34, physicalPallet >= 0 ? yOf(physicalPallet) : -1, true);
  }
  const towOffset = 16 + recordSize * agentCount;
  view.setFloat32(towOffset, xOf(towPosition), true);
  view.setFloat32(towOffset + 4, yOf(towPosition), true);
  view.setUint8(towOffset + 8, towState);
  view.setUint8(towOffset + 9, 0);
  view.setUint16(towOffset + 10, towTarget < 0 ? 0xffff : towTarget, true);
  view.setUint16(towOffset + 12, rescueQueue.length, true);
  view.setUint16(towOffset + 14, 0, true);
  previousFramePositions = positions.slice(0, agentCount);
  previousFrameStages = stages.slice(0, agentCount);
  previousFramePalletIds = framePalletIds;
  return buffer;
}

function emitFrame(): void {
  const buffer = encodeFrame();
  post(buffer, [buffer]);
}

function hello(): void {
  post({
    type: 'hello', protocol: PROTOCOL, map: { width: map.width, height: map.height, cellSize: 1 },
    layout, lifelong: true, simulator: true, streaming: true, browserRuntime: true,
    backend, backendAdapter, agents: agentCount, tickRate: TICK_RATE, inferenceMs: lastInferenceMs,
    summary: { status: 'running' },
  });
}

async function tick(): Promise<void> {
  if (stepping || paused) return;
  stepping = true;
  const tickGeneration = generation;
  try {
    advanceRecovery();
    if (!await inferAndPlan(tickGeneration)) return;
    processTaskArrivals();
    step++;
    emitFrame();
    if (step === 1 || step % 10 === 0) post({ type: 'metrics', inferenceMs: lastInferenceMs, backend, backendAdapter });
  } catch (error) {
    paused = true;
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  } finally {
    stepping = false;
    schedule();
  }
}

function schedule(): void {
  if (timer !== null) workerScope.clearTimeout(timer);
  timer = null;
  if (!paused) timer = workerScope.setTimeout(() => void tick(), 1000 / (TICK_RATE * speed));
}

function failAgent(agent: number): void {
  if (agent < 0 || agent >= agentCount || recovery[agent] !== 0) return;
  generation++;
  failed[agent] = 1; recovery[agent] = 1; resumeGoals[agent] = goals[agent];
  recoveringPallet[agent] = loaded[agent]; recoveryDwellStarted[agent] = 0; repairDwell[agent] = 0;
  if (loaded[agent] && taskIndex[agent] >= 0) {
    const palletId = currentPalletId(agent);
    loaded[agent] = 0; palletPresent[palletId] = 1; palletPositions[palletId] = positions[agent];
  }
  dwell[agent] = 0; goals[agent] = positions[agent]; rescueQueue.push(agent); dispatchTow();
  emitFrame();
}

function applyLayout(raw: Array<{ x: number; y: number }>): void {
  const unique = new Map<string, { x: number; y: number }>();
  for (const entry of raw) unique.set(`${entry.x},${entry.y}`, entry);
  if (unique.size < MAX_AGENTS) throw new Error(`At least ${MAX_AGENTS} pallets are required.`);
  const pallets = [...unique.values()].sort((a, b) => a.x - b.x || a.y - b.y).map((entry, id) => ({
    id, x: entry.x, y: entry.y, cargoType: (entry.x * 31 + entry.y * 17) % 3,
  }));
  layout = { ...layout, pallets };
  rebuildScenarioForLayout();
  resetSimulation(agentCount);
}

workerScope.onmessage = (event): void => {
  const message = event.data;
  if (message.type !== 'control') return;
  if (!booted) {
    if (message.action === 'load') requestedAgentCount = Math.max(2, Math.min(MAX_AGENTS, Number(message.agents) || MAX_AGENTS));
    else if (message.action === 'speed') speed = Math.max(0.25, Math.min(3, Number(message.value) || 1));
    else if (message.action === 'pause' || message.action === 'stop') paused = true;
    else if (message.action === 'run') paused = false;
    return;
  }
  if (message.action === 'pause') { paused = true; schedule(); }
  else if (message.action === 'run') { paused = false; schedule(); }
  else if (message.action === 'speed') { speed = Math.max(0.25, Math.min(3, Number(message.value) || 1)); schedule(); }
  else if (message.action === 'step' && paused) { paused = false; void tick().finally(() => { paused = true; schedule(); }); }
  else if (message.action === 'stop') {
    paused = true; resetSimulation(agentCount); emitFrame();
    post({ type: 'metrics', inferenceMs: lastInferenceMs, backend, backendAdapter });
    post({ type: 'status', state: 'stopped' }); schedule();
  } else if (message.action === 'fail') failAgent(Number(message.agent));
  else if (message.action === 'load') {
    paused = true; post({ type: 'status', state: 'planning', agents: Number(message.agents) });
    resetSimulation(Number(message.agents)); paused = false; hello(); emitFrame(); schedule();
  } else if (message.action === 'layout' && Array.isArray(message.pallets)) {
    try {
      paused = true; post({ type: 'status', state: 'planning', agents: agentCount });
      applyLayout(message.pallets); post({ type: 'layout-applied', pallets: layout.pallets.length });
      paused = false; hello(); emitFrame(); schedule();
    } catch (error) {
      paused = false; post({ type: 'layout-error', message: error instanceof Error ? error.message : String(error) }); schedule();
    }
  }
};

async function boot(): Promise<void> {
  try {
    post({ type: 'status', state: 'planning', agents: agentCount });
    [core, session] = await Promise.all([loadWasm(), loadPolicy()]);
    resetSimulation(requestedAgentCount);
    booted = true;
    hello(); emitFrame(); schedule();
  } catch (error) {
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}

void boot();
