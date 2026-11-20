export const WAREHOUSE_SCHEMA = 'epibt-warehouse/v1';
export const HEADINGS = ['E', 'S', 'W', 'N'] as const;
export const MAX_CONFIG_BYTES = 1_000_000;
const MAX_WIDTH = 48;
const MAX_HEIGHT = 40;
const MAX_AGENTS = 200;
const MAX_TASKS = 4096;

export type Heading = typeof HEADINGS[number];
export type Point = { x: number; y: number };
export type Start = Point & { heading: Heading };
export type Delivery = { pickup: Point; delivery: Point };

export type WarehouseConfig = {
  schema: typeof WAREHOUSE_SCHEMA;
  map: string[];
  agents: Start[];
  tasks: Delivery[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function parseWarehouse(value: unknown): WarehouseConfig {
  if (!isRecord(value) || value.schema !== WAREHOUSE_SCHEMA) {
    throw new Error('Expected an EPIBT warehouse JSON file (epibt-warehouse/v1).');
  }

  const { map } = value;
  if (!Array.isArray(map) || map.length < 2 || map.length > MAX_HEIGHT) {
    throw new Error('The map must have 2–40 rows.');
  }
  const width = typeof map[0] === 'string' ? map[0].length : 0;
  if (width < 2 || width > MAX_WIDTH || !map.every(row =>
    typeof row === 'string' && row.length === width && /^[.#SD]+$/.test(row)
  )) {
    throw new Error('The map must have 2–48 columns and use only #, ., S and D.');
  }
  const rows: string[] = map;

  function point(input: unknown): Point {
    if (!isRecord(input) || typeof input.x !== 'number' || typeof input.y !== 'number'
      || !Number.isInteger(input.x) || !Number.isInteger(input.y)
      || input.x < 0 || input.x >= width || input.y < 0 || input.y >= rows.length
      || rows[input.y][input.x] === '#') {
      throw new Error('A start or task point is outside the walkable map.');
    }
    return { x: input.x, y: input.y };
  }

  if (!Array.isArray(value.agents) || value.agents.length < 1 || value.agents.length > MAX_AGENTS) {
    throw new Error('The warehouse must have 1–200 robots.');
  }
  const agents = value.agents.map((input: unknown): Start => {
    const start = point(input);
    const heading = isRecord(input) ? HEADINGS.find(option => option === input.heading) : undefined;
    if (!heading) throw new Error('Robot headings must be E, S, W or N.');
    return { ...start, heading };
  });

  const cell = ({ x, y }: Point) => y * width + x;
  const starts = new Set(agents.map(cell));
  if (starts.size !== agents.length) throw new Error('Two robots share a start cell.');

  if (!Array.isArray(value.tasks) || value.tasks.length < 1 || value.tasks.length > MAX_TASKS) {
    throw new Error('The warehouse must have 1–4,096 delivery tasks.');
  }
  const tasks = value.tasks.map((input: unknown): Delivery => {
    if (!isRecord(input)) throw new Error('Every task must specify pickup and delivery points.');
    const pickup = point(input.pickup);
    const delivery = point(input.delivery);
    if (cell(pickup) === cell(delivery)) throw new Error('Pickup and delivery must use different cells.');
    return { pickup, delivery };
  });

  const visited = new Set([cell(agents[0])]);
  const queue: Point[] = [agents[0]];
  for (let head = 0; head < queue.length; head++) {
    const { x, y } = queue[head];
    for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) {
      const next = { x: x + dx, y: y + dy };
      const tile = rows[next.y]?.[next.x];
      const index = cell(next);
      if (!tile || tile === '#' || visited.has(index)) continue;
      visited.add(index);
      queue.push(next);
    }
  }
  if (agents.some(start => !visited.has(cell(start))) ||
      tasks.some(task => !visited.has(cell(task.pickup)) || !visited.has(cell(task.delivery)))) {
    throw new Error('All robots and task points must be connected by aisles.');
  }

  return { schema: WAREHOUSE_SCHEMA, map: rows, agents, tasks };
}

export function defaultWarehouse(): WarehouseConfig {
  const width = 44;
  const height = 27;
  const map: string[][] = Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) =>
      x === 0 || y === 0 || x === width - 1 || y === height - 1 ? '#' : '.'
    )
  );
  const pickups: Point[] = [];
  for (const y of [4, 11, 18]) {
    for (const x of [8, 14, 20, 26, 32, 38]) {
      for (let row = y; row < y + 5; row++) {
        map[row][x] = '#';
        map[row][x + 1] = '#';
      }
      for (const dx of [-1, 2]) {
        for (const dy of [1, 3]) {
          map[y + dy][x + dx] = 'S';
          pickups.push({ x: x + dx, y: y + dy });
        }
      }
    }
  }

  const deliveries = [3, 6, 9, 12, 15, 18, 21, 24].map(y => ({ x: 42, y }));
  for (const { x, y } of deliveries) map[y][x] = 'D';
  const starts: Point[] = [];
  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) if (map[y][x] === '.') starts.push({ x, y });
  }

  return {
    schema: WAREHOUSE_SCHEMA,
    map: map.map(row => row.join('')),
    agents: Array.from({ length: 48 }, (_, id) => ({
      ...starts[(id * 83) % starts.length], heading: HEADINGS[id % HEADINGS.length],
    })),
    tasks: pickups.map((pickup, i) => ({ pickup, delivery: deliveries[(i * 3 + Math.floor(i / 8)) % deliveries.length] })),
  };
}
