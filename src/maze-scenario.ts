export const MAZE_ROWS = [
  '###############',
  '#.....#.......#',
  '#.##..#..##...#',
  '#...#...#...#.#',
  '###.#.#.#.#.#.#',
  '#...#.#...#...#',
  '#.#.#.###.#.###',
  '#.#...#...#...#',
  '#...##..##..#.#',
  '#.............#',
  '###############',
] as const;

export const MAZE_WIDTH = MAZE_ROWS[0].length;
export const MAZE_HEIGHT = MAZE_ROWS.length;
export const MAZE_MODEL_SLOTS = 16;
export type MazeAgent = { start: number; goal: number | null };
export type MazeConfig = { walls: number[]; agents: MazeAgent[]; lifelong: boolean };
export const MAZE_ROUTES: ReadonlyArray<readonly [readonly [number, number], readonly [number, number]]> = [
  [[5, 3], [13, 9]],
  [[6, 3], [1, 9]],
  [[7, 3], [13, 1]],
  [[7, 2], [1, 1]],
  [[5, 5], [9, 9]],
  [[7, 5], [5, 1]],
  [[8, 5], [11, 5]],
  [[9, 5], [3, 5]],
] as const;

export const MAZE_COLORS = [
  '#49d9f8', '#ffa861', '#a8a0ff', '#62e4b9',
  '#ffd574', '#ff83b5', '#7eb8ff', '#d1ef8a',
  '#f49a7f', '#86ead9', '#c3a6ee', '#f6c36d',
  '#80c9f5', '#eaa5cb', '#b4df77', '#e9bca2',
] as const;

export function defaultMazeConfig(): MazeConfig {
  const walls: number[] = [];
  for (let y = 0; y < MAZE_HEIGHT; y++) {
    for (let x = 0; x < MAZE_WIDTH; x++) {
      if (MAZE_ROWS[y][x] === '#') walls.push(mazeCell(x, y));
    }
  }
  return {
    walls,
    agents: MAZE_ROUTES.map(([start, goal]) => ({ start: mazeCell(...start), goal: mazeCell(...goal) })),
    lifelong: true,
  };
}

export function cloneMazeConfig(config: MazeConfig): MazeConfig {
  return { walls: [...config.walls], agents: config.agents.map((agent) => ({ ...agent })), lifelong: config.lifelong };
}

export function validateMazeConfig(config: MazeConfig): string | null {
  const cells = MAZE_WIDTH * MAZE_HEIGHT;
  const validCell = (cell: number) => Number.isInteger(cell) && cell >= 0 && cell < cells;
  if (!config || !Array.isArray(config.walls) || !Array.isArray(config.agents) || typeof config.lifelong !== 'boolean') return 'Invalid maze configuration.';
  if (config.agents.length < 1 || config.agents.length > MAZE_MODEL_SLOTS) return `Choose 1–${MAZE_MODEL_SLOTS} agents.`;
  const walls = new Set(config.walls);
  if (walls.size !== config.walls.length || config.walls.some((cell) => !validCell(cell))) return 'Invalid wall positions.';
  for (let y = 0; y < MAZE_HEIGHT; y++) {
    for (let x = 0; x < MAZE_WIDTH; x++) {
      if ((x === 0 || y === 0 || x === MAZE_WIDTH - 1 || y === MAZE_HEIGHT - 1) && !walls.has(mazeCell(x, y))) return 'The outer border must stay closed.';
    }
  }
  const starts = new Set<number>();
  const goals = new Set<number>();
  for (const [index, agent] of config.agents.entries()) {
    if (!validCell(agent.start) || walls.has(agent.start)) return `Agent ${index + 1} starts on a wall.`;
    if (starts.has(agent.start)) return 'Two agents cannot share a start cell.';
    starts.add(agent.start);
    if (agent.goal === null) return `Choose a goal for agent ${index + 1}.`;
    if (!validCell(agent.goal) || walls.has(agent.goal)) return `Agent ${index + 1} has a goal on a wall.`;
    if (goals.has(agent.goal)) return 'Two agents cannot share a goal cell.';
    goals.add(agent.goal);
  }
  for (const [index, agent] of config.agents.entries()) {
    const seen = new Set([agent.start]);
    const queue = [agent.start];
    for (let head = 0; head < queue.length; head++) {
      const cell = queue[head];
      const x = mazeX(cell);
      const y = mazeY(cell);
      for (const neighbor of [x > 0 ? cell - 1 : -1, x + 1 < MAZE_WIDTH ? cell + 1 : -1, y > 0 ? cell - MAZE_WIDTH : -1, y + 1 < MAZE_HEIGHT ? cell + MAZE_WIDTH : -1]) {
        if (neighbor < 0 || walls.has(neighbor) || seen.has(neighbor)) continue;
        seen.add(neighbor);
        queue.push(neighbor);
      }
    }
    if (!seen.has(agent.goal!)) return `Agent ${index + 1} cannot reach its goal through the walls.`;
  }
  return null;
}

export function mazeCell(x: number, y: number): number {
  return y * MAZE_WIDTH + x;
}

export function mazeX(cell: number): number {
  return cell % MAZE_WIDTH;
}

export function mazeY(cell: number): number {
  return Math.floor(cell / MAZE_WIDTH);
}
