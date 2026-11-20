export const WAREHOUSE_SCHEMA = 'epibt-warehouse/v2';
export const MAX_CONFIG_BYTES = 1_000_000;
export const MAX_AGENTS = 200;
const MAX_WIDTH = 48;
const MAX_HEIGHT = 40;

export type Point = { x: number; y: number };
export type Tile = '.' | '#' | 'S' | 'D';
export type WarehouseConfig = {
  schema: typeof WAREHOUSE_SCHEMA;
  map: string[];
  agentCount: number;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function migrateMap(map: string[], tasks: unknown): string[] {
  if (!Array.isArray(tasks)) return map;
  const missing = ([['pickup', 'S'], ['delivery', 'D']] as const)
    .filter(([, tile]) => !map.some(row => row.includes(tile)));
  if (!missing.length) return map;

  const rows = [...map];
  function mark(task: unknown, key: string, tile: Tile): boolean {
    const point = isRecord(task) && task[key];
    if (!isRecord(point) || typeof point.x !== 'number' || typeof point.y !== 'number') return false;
    const { x, y } = point;
    if (rows[y]?.[x] !== '.') return false;
    rows[y] = rows[y].slice(0, x) + tile + rows[y].slice(x + 1);
    return true;
  }
  // Reserve one cell for each missing bay type before converting any other points.
  for (const [key, tile] of missing) tasks.some(task => mark(task, key, tile));
  for (const [key, tile] of missing) tasks.forEach(task => mark(task, key, tile));
  return rows;
}

export function parseWarehouse(value: unknown): WarehouseConfig {
  if (!isRecord(value) || (value.schema !== WAREHOUSE_SCHEMA && value.schema !== 'epibt-warehouse/v1')) {
    throw new Error('Expected an EPIBT warehouse JSON file.');
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

  const legacy = value.schema === 'epibt-warehouse/v1';
  const agentCount = legacy && Array.isArray(value.agents) ? value.agents.length : value.agentCount;
  if (typeof agentCount !== 'number' || !Number.isInteger(agentCount) || agentCount < 1 || agentCount > MAX_AGENTS) {
    throw new Error('The warehouse must have 1–200 agents.');
  }

  return { schema: WAREHOUSE_SCHEMA, map: legacy ? migrateMap(map, value.tasks) : [...map], agentCount };
}

export function prepareWarehouse(config: WarehouseConfig): {
  starts: number[];
  pickups: number[];
  deliveries: number[];
} {
  const width = config.map[0].length;
  const tiles = config.map.join('');
  const pickups: number[] = [];
  const deliveries: number[] = [];
  for (let cell = 0; cell < tiles.length; cell++) {
    if (tiles[cell] === 'S') pickups.push(cell);
    if (tiles[cell] === 'D') deliveries.push(cell);
  }
  if (!pickups.length || !deliveries.length) throw new Error('Place at least one pickup and one delivery bay.');

  const reachable = new Set([pickups[0]]);
  const queue = [pickups[0]];
  for (let index = 0; index < queue.length; index++) {
    const cell = queue[index];
    const x = cell % width;
    const neighbours = [x > 0 ? cell - 1 : -1, x < width - 1 ? cell + 1 : -1, cell - width, cell + width];
    for (const next of neighbours) {
      if (!tiles[next] || tiles[next] === '#' || reachable.has(next)) continue;
      reachable.add(next);
      queue.push(next);
    }
  }
  if ([...pickups, ...deliveries].some(cell => !reachable.has(cell))) {
    throw new Error('Connect all pickup and delivery bays by aisles.');
  }
  if (reachable.size < config.agentCount) throw new Error('There are too few reachable cells for this many agents.');

  const floor = Array.from(reachable).filter(cell => tiles[cell] === '.').sort((a, b) => a - b);
  const cells = floor.length >= config.agentCount ? floor : [...floor, ...pickups, ...deliveries];
  const starts = Array.from({ length: config.agentCount }, (_, index) =>
    cells[Math.floor(index * cells.length / config.agentCount)]
  );
  return { starts, pickups, deliveries };
}
