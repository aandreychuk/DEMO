import { HEADINGS, parseWarehouse, type Point, type WarehouseConfig } from './config';
import { nextPose, type Action, type Agent, type WorkerCommand, type WorkerReply } from './model';

type Runtime = {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _epibt_init(width: number, height: number, cells: number): void;
  _epibt_step(count: number, input: number, output: number): void;
};

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null;
  postMessage(message: WorkerReply): void;
};

const INPUT_STRIDE = 4;
const OUTPUT_STRIDE = 6;
let runtime: Runtime;
let config: WarehouseConfig;
let width = 0;
let step = 0;
let completed = 0;
let nextTask = 0;
let input = 0;
let output = 0;
let agents: Array<Agent & { dropoff: number }> = [];

const cell = (point: Point) => point.y * width + point.x;

function assignTask(agent: Agent & { dropoff: number }): void {
  const task = config.tasks[nextTask % config.tasks.length];
  const pickup = cell(task.pickup);
  agent.task = ++nextTask;
  agent.dropoff = cell(task.delivery);
  agent.stage = agent.cell === pickup ? 'Delivery' : 'Pickup';
  agent.target = agent.stage === 'Pickup' ? pickup : agent.dropoff;
}

async function configure(request: WarehouseConfig): Promise<void> {
  config = parseWarehouse(request);
  width = config.map[0].length;
  const moduleUrl = new URL(`${import.meta.env.BASE_URL}runtime/epibt.mjs`, self.location.origin);
  const createRuntime = (await import(/* @vite-ignore */ moduleUrl.href)).default as (options: {
    locateFile: (name: string) => string;
    print: (text: string) => void;
  }) => Promise<Runtime>;
  runtime = await createRuntime({
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

  agents = config.agents.map(start => {
    const agent: Agent & { dropoff: number } = {
      cell: cell(start), direction: HEADINGS.indexOf(start.heading), target: 0, task: 0, stage: 'Pickup',
      operation: [], priority: null, completed: 0, distance: 0, turns: 0, waits: 0, dropoff: 0,
    };
    assignTask(agent);
    return agent;
  });
  input = runtime._malloc(agents.length * INPUT_STRIDE * Int32Array.BYTES_PER_ELEMENT);
  output = runtime._malloc(agents.length * OUTPUT_STRIDE * Int32Array.BYTES_PER_ELEMENT);
  scope.postMessage({ type: 'snapshot', step, completed, planningMs: null, agents });
}

function advance(): void {
  if (!input || !output) throw new Error('EPIBT is not ready.');
  for (const [id, agent] of agents.entries()) {
    runtime.HEAP32.set([agent.cell, agent.direction, agent.target, agent.dropoff], input / 4 + id * INPUT_STRIDE);
  }

  const started = performance.now();
  runtime._epibt_step(agents.length, input, output);
  const planningMs = performance.now() - started;
  const destinations = new Set<number>();
  const origins = new Map(agents.map((agent, id) => [agent.cell, id]));

  const moves = agents.map((agent, id) => {
    const offset = output / 4 + id * OUTPUT_STRIDE;
    const [destination, direction, priority] = runtime.HEAP32.subarray(offset, offset + 3);
    const operation = Array.from(runtime.HEAP32.subarray(offset + 3, offset + OUTPUT_STRIDE)) as Action[];
    const tile = config.map[Math.floor(destination / width)]?.[destination % width];
    if (operation.some(action => action < 0 || action > 3) || !tile || tile === '#' || destinations.has(destination)) {
      throw new Error('EPIBT returned an invalid move.');
    }
    const expected = nextPose(agent, operation[0], width);
    const other = origins.get(destination);
    const swapped = other !== undefined && other !== id
      && runtime.HEAP32[output / 4 + other * OUTPUT_STRIDE] === agent.cell;
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
    if (move.operation[0] === 0) agent.distance++;
    else if (move.operation[0] === 3) agent.waits++;
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
  scope.postMessage({ type: 'snapshot', step: ++step, completed, planningMs, agents });
}

scope.onmessage = async ({ data }) => {
  try {
    if (data.type === 'configure') await configure(data.config);
    else advance();
  } catch (error) {
    scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
