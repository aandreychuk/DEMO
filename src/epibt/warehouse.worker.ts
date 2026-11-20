import { parseWarehouse, prepareWarehouse, type WarehouseConfig } from './config';
import { nextPose, type Action, type Agent, type OperationLength, type WorkerCommand, type WorkerReply } from './model';

type Runtime = {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _epibt_init(width: number, height: number, cells: number): void;
  _epibt_step(count: number, input: number, output: number): void;
};

type Planner = { runtime: Runtime; length: OperationLength; input: number; output: number };

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null;
  postMessage(message: WorkerReply): void;
};

const INPUT_STRIDE = 4;
const OUTPUT_FIELDS = 3; // Cell, heading and priority, followed by the planned actions.
const MAX_BATCH_MS = 25;
let planner: Planner | null = null;
let config: WarehouseConfig;
let width = 0;
let pickups: number[] = [];
let deliveries: number[] = [];
let step = 0;
let completed = 0;
let nextTask = 0;
let agents: Array<Agent & { dropoff: number }> = [];

function assignTask(agent: Agent & { dropoff: number }): void {
  const pickupIndex = nextTask % pickups.length;
  const pickup = pickups[pickupIndex];
  const deliveryIndex = pickupIndex + Math.floor(nextTask / pickups.length);
  agent.task = ++nextTask;
  agent.dropoff = deliveries[deliveryIndex % deliveries.length];
  agent.stage = agent.cell === pickup ? 'Delivery' : 'Pickup';
  agent.target = agent.stage === 'Pickup' ? pickup : agent.dropoff;
}

async function loadPlanner(length: OperationLength): Promise<Planner> {
  const name = length === 3 ? 'epibt' : `epibt-${length}`;
  const moduleUrl = new URL(`${import.meta.env.BASE_URL}runtime/${name}.mjs`, self.location.origin);
  const createRuntime = (await import(/* @vite-ignore */ moduleUrl.href)).default as (options: {
    locateFile: (name: string) => string;
    print: (text: string) => void;
  }) => Promise<Runtime>;
  const runtime = await createRuntime({
    locateFile: name => new URL(name, moduleUrl).href,
    print: () => {},
  });
  const map = new TextEncoder().encode(config.map.join(''));
  const mapPointer = runtime._malloc(map.length);
  try {
    runtime.HEAPU8.set(map, mapPointer);
    runtime._epibt_init(width, config.map.length, mapPointer);
  } finally {
    runtime._free(mapPointer);
  }
  const input = runtime._malloc(config.agentCount * INPUT_STRIDE * Int32Array.BYTES_PER_ELEMENT);
  const output = runtime._malloc(config.agentCount * (OUTPUT_FIELDS + length) * Int32Array.BYTES_PER_ELEMENT);
  return { runtime, length, input, output };
}

function publish(operationLength: OperationLength, planningMs: number | null): void {
  scope.postMessage({ type: 'snapshot', step, completed, operationLength, planningMs, agents });
}

async function configure(request: WarehouseConfig, length: OperationLength): Promise<void> {
  config = parseWarehouse(request);
  const plan = prepareWarehouse(config);
  pickups = plan.pickups;
  deliveries = plan.deliveries;
  width = config.map[0].length;
  planner = await loadPlanner(length);
  agents = plan.starts.map((cell, index) => {
    const agent: Agent & { dropoff: number } = {
      cell, direction: index % 4, target: 0, task: 0, stage: 'Pickup',
      operation: [], priority: null, completed: 0, distance: 0, turns: 0, waits: 0, dropoff: 0,
    };
    assignTask(agent);
    return agent;
  });
  publish(length, null);
}

async function changeLength(length: OperationLength): Promise<void> {
  try {
    const next = await loadPlanner(length);
    if (planner) {
      planner.runtime._free(planner.input);
      planner.runtime._free(planner.output);
    }
    planner = next;
  } catch (error) {
    console.warn('Could not load the EPIBT planner:', error);
    scope.postMessage({ type: 'length-error' });
    return;
  }
  for (const agent of agents) agent.operation = [];
  publish(length, null);
}

function advance({ runtime, length, input, output }: Planner): number {
  const outputStride = OUTPUT_FIELDS + length;
  for (const [id, agent] of agents.entries()) {
    runtime.HEAP32.set([agent.cell, agent.direction, agent.target, agent.dropoff], input / 4 + id * INPUT_STRIDE);
  }

  const started = performance.now();
  runtime._epibt_step(agents.length, input, output);
  const planningMs = performance.now() - started;
  const destinations = new Set<number>();
  const origins = new Map(agents.map((agent, id) => [agent.cell, id]));

  const moves = agents.map((agent, id) => {
    const offset = output / 4 + id * outputStride;
    const [destination, direction, priority] = runtime.HEAP32.subarray(offset, offset + OUTPUT_FIELDS);
    const operation = Array.from(runtime.HEAP32.subarray(offset + OUTPUT_FIELDS, offset + outputStride)) as Action[];
    const tile = config.map[Math.floor(destination / width)]?.[destination % width];
    if (operation.some(action => action < 0 || action > 3) || !tile || tile === '#' || destinations.has(destination)) {
      throw new Error('EPIBT returned an invalid move.');
    }
    const expected = nextPose(agent, operation[0], width);
    const other = origins.get(destination);
    const swapped = other !== undefined && other !== id
      && runtime.HEAP32[output / 4 + other * outputStride] === agent.cell;
    const wrapped = operation[0] === 0 && Math.abs(destination % width - agent.cell % width) > 1;
    if (swapped || wrapped || expected.cell !== destination || expected.direction !== direction) {
      throw new Error('EPIBT returned an invalid move.');
    }
    destinations.add(destination);
    return { destination, direction, priority, operation };
  });

  for (const [id, agent] of agents.entries()) {
    const move = moves[id];
    agent.cell = move.destination;
    agent.direction = move.direction;
    agent.priority = move.priority;
    agent.operation = move.operation;
    agent.lastAction = move.operation[0];
    if (agent.lastAction === 0) agent.distance++;
    else if (agent.lastAction === 3) agent.waits++;
    else agent.turns++;
    if (agent.cell === agent.target) {
      if (agent.stage === 'Pickup') {
        agent.stage = 'Delivery';
        agent.target = agent.dropoff;
      } else {
        agent.completed++;
        completed++;
        assignTask(agent);
      }
    }
  }
  step++;
  return planningMs;
}

function advanceBatch(count: number): void {
  if (!planner) throw new Error('EPIBT is not ready.');
  const deadline = performance.now() + MAX_BATCH_MS;
  let planningMs = advance(planner);
  for (let index = 1; index < count && performance.now() < deadline; index++) {
    planningMs = advance(planner);
  }
  publish(planner.length, planningMs);
}

scope.onmessage = async ({ data }) => {
  try {
    if (data.type === 'configure') await configure(data.config, data.length);
    else if (data.type === 'length') await changeLength(data.length);
    else advanceBatch(data.count);
  } catch (error) {
    scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
