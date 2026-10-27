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
] as const;

export function mazeCell(x: number, y: number): number {
  return y * MAZE_WIDTH + x;
}

export function mazeX(cell: number): number {
  return cell % MAZE_WIDTH;
}

export function mazeY(cell: number): number {
  return Math.floor(cell / MAZE_WIDTH);
}
