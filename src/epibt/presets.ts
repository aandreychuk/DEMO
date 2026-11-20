import { WAREHOUSE_SCHEMA, type Tile, type WarehouseConfig } from './config';

function floor(width: number, height: number, interior: '.' | '#' = '.'): Tile[][] {
  return Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) =>
    x === 0 || y === 0 || x === width - 1 || y === height - 1 ? '#' : interior
  ));
}

function paint(map: Tile[][], left: number, top: number, right: number, bottom: number, tile: Tile): void {
  for (let y = top; y <= bottom; y++) map[y].fill(tile, left, right + 1);
}

function corridor(map: Tile[][], points: ReadonlyArray<readonly [number, number]>): void {
  for (let index = 1; index < points.length; index++) {
    const [x0, y0] = points[index - 1];
    const [x1, y1] = points[index];
    paint(map, Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1), '.');
  }
}

function warehouse(map: Tile[][], agentCount: number): WarehouseConfig {
  return { schema: WAREHOUSE_SCHEMA, map: map.map(row => row.join('')), agentCount };
}

function rackRows(): WarehouseConfig {
  const map = floor(44, 27);
  for (const y of [4, 11, 18]) {
    for (const x of [8, 14, 20, 26, 32, 38]) {
      for (let row = y; row < y + 5; row++) {
        map[row][x] = '#';
        map[row][x + 1] = '#';
      }
      for (const dx of [-1, 2]) {
        for (const dy of [1, 3]) map[y + dy][x + dx] = 'S';
      }
    }
  }
  for (const y of [3, 6, 9, 12, 15, 18, 21, 24]) map[y][42] = 'D';
  return warehouse(map, 48);
}

function centralJunction(): WarehouseConfig {
  const map = floor(31, 25);
  for (const x of [4, 19]) {
    for (const y of [4, 15]) {
      for (let dx = 0; dx <= 7; dx++) {
        map[y][x + dx] = '#';
        map[y + 4][x + dx] = '#';
      }
      for (const dx of [1, 3, 5, 7]) map[y + 2][x + dx] = 'S';
    }
  }
  for (const x of [4, 8, 12, 18, 22, 26]) {
    map[1][x] = 'D';
    map[23][x] = 'D';
  }
  return warehouse(map, 32);
}

function narrowAisles(): WarehouseConfig {
  const map = floor(33, 24);
  for (let x = 2; x <= 29; x += 3) {
    // Two columns of racks, one column of aisle, with single-row cross aisles.
    for (const top of [2, 9, 16]) paint(map, x, top, x + 1, top + 5, '#');
    for (const y of [4, 11, 18]) map[y][x - 1] = 'S';
  }
  for (const x of [4, 10, 16, 22, 28]) {
    map[1][x] = 'D';
    map[22][x] = 'D';
  }
  return warehouse(map, 40);
}

function staggeredGates(): WarehouseConfig {
  const map = floor(41, 25);
  const walls = [
    { x: 9, gates: [4, 16, 21] },
    { x: 19, gates: [7, 13, 19] },
    { x: 29, gates: [3, 11, 20] },
  ];
  for (const { x, gates } of walls) {
    paint(map, x, 1, x + 2, 23, '#');
    for (const y of gates) paint(map, x, y, x + 2, y, '.');
  }
  for (const [x, y] of [[3, 5], [3, 14], [14, 8], [14, 17], [24, 4], [24, 14], [34, 7], [34, 18]]) {
    paint(map, x, y, x + 1, y + 2, '#');
  }
  for (const y of [2, 6, 10, 14, 18, 22]) {
    map[y][1] = 'S';
    map[y][39] = 'D';
  }
  return warehouse(map, 36);
}

function switchbackLanes(): WarehouseConfig {
  const map = floor(40, 28, '#');
  corridor(map, [[2, 4], [10, 4], [10, 2], [27, 2], [27, 4], [37, 4]]);
  corridor(map, [[2, 12], [8, 12], [8, 8], [19, 8], [19, 11], [29, 11], [29, 9], [37, 9]]);
  corridor(map, [[2, 18], [13, 18], [13, 15], [23, 15], [23, 19], [32, 19], [32, 16], [37, 16]]);
  corridor(map, [[2, 25], [8, 25], [8, 22], [17, 22], [17, 25], [28, 25], [28, 23], [37, 23]]);
  for (const [x, top, bottom] of [[2, 4, 25], [37, 4, 23], [15, 2, 8], [33, 4, 9],
    [11, 8, 18], [20, 11, 25], [25, 11, 19], [16, 15, 22], [34, 16, 23]]) {
    paint(map, x, top, x, bottom, '.');
  }
  // Short passing bays next to the long, single-cell stretches.
  for (const [left, y] of [[5, 5], [18, 1], [29, 5], [13, 9], [22, 12],
    [5, 19], [17, 16], [27, 20], [4, 26], [22, 26], [30, 22]]) {
    paint(map, left, y, left + 2, y, '.');
  }
  for (const y of [6, 10, 15, 20, 24]) map[y][2] = 'S';
  for (const y of [5, 7, 10, 13, 18, 22]) map[y][37] = 'D';
  return warehouse(map, 18);
}

function nestedLoops(): WarehouseConfig {
  const map = floor(44, 31, '#');
  for (const [left, top, right, bottom] of [[2, 2, 41, 28], [8, 7, 35, 23], [14, 12, 29, 18]]) {
    corridor(map, [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]);
  }
  for (const [left, top, right, bottom] of [[12, 2, 12, 7], [28, 2, 28, 7], [2, 16, 8, 16],
    [35, 13, 41, 13], [31, 23, 31, 28], [18, 7, 18, 12], [24, 18, 24, 23], [8, 14, 14, 14], [29, 16, 35, 16]]) {
    paint(map, left, top, right, bottom, '.');
  }
  for (const [x, y] of [[5, 2], [21, 2], [37, 2], [41, 8], [41, 24], [26, 28], [9, 28], [2, 21]]) map[y][x] = 'S';
  for (const [x, y] of [[15, 12], [23, 12], [29, 14], [27, 18], [18, 18], [14, 16]]) map[y][x] = 'D';
  return warehouse(map, 18);
}

export const WAREHOUSE_PRESETS = [
  { id: 'rack-rows', name: 'Rack rows', config: rackRows() },
  { id: 'central-junction', name: 'Central junction', config: centralJunction() },
  { id: 'narrow-aisles', name: 'Narrow aisles', config: narrowAisles() },
  { id: 'staggered-gates', name: 'Staggered gates', config: staggeredGates() },
  { id: 'switchback-lanes', name: 'Switchback lanes', config: switchbackLanes() },
  { id: 'nested-loops', name: 'Nested loops', config: nestedLoops() },
];
