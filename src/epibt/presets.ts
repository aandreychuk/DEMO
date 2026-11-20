import { WAREHOUSE_SCHEMA, type Tile, type WarehouseConfig } from './config';

function floor(width: number, height: number): Tile[][] {
  return Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) =>
    x === 0 || y === 0 || x === width - 1 || y === height - 1 ? '#' : '.'
  ));
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
  const map = floor(36, 21);
  for (const x of [5, 10, 15, 20, 25, 30]) {
    for (let y = 3; y < 18; y++) {
      if (y === 10) continue;
      map[y][x] = '#';
      map[y][x + 1] = '#';
    }
    for (const y of [4, 7, 13, 16]) map[y][x - 1] = 'S';
  }
  for (const y of [3, 8, 12, 17]) {
    map[y][1] = 'D';
    map[y][34] = 'D';
  }
  return warehouse(map, 40);
}

export const WAREHOUSE_PRESETS = [
  { id: 'rack-rows', name: 'Rack rows', config: rackRows() },
  { id: 'central-junction', name: 'Central junction', config: centralJunction() },
  { id: 'narrow-aisles', name: 'Narrow aisles', config: narrowAisles() },
];
