import { MAZE_WIDTH } from './maze-scenario';

const ACTION_OFFSETS = [0, -MAZE_WIDTH, MAZE_WIDTH, -1, 1] as const;

export function bestMazeAction(scores: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  return best;
}

// Apply the requested moves together. Illegal moves stay put, with no fallback action.
export function applyMazeActions(positions: Int32Array, actions: readonly number[], walls: ReadonlySet<number>): Int32Array {
  const destinations = positions.map((position, i) => {
    const target = position + ACTION_OFFSETS[actions[i]];
    return walls.has(target) ? position : target;
  });
  const origins = new Map(Array.from(positions, (position, i) => [position, i]));
  const counts = new Map<number, number>();
  for (const target of destinations) counts.set(target, (counts.get(target) ?? 0) + 1);

  const next = destinations.map((target, i) => {
    const occupant = origins.get(target);
    const swaps = occupant !== undefined && occupant !== i && destinations[occupant] === positions[i];
    return swaps || counts.get(target)! > 1 ? positions[i] : target;
  });

  // An agent whose move was blocked also blocks movement into its current cell.
  let changed: boolean;
  do {
    changed = false;
    for (let i = 0; i < next.length; i++) {
      if (next[i] === positions[i]) continue;
      const occupant = origins.get(next[i]);
      if (occupant !== undefined && next[occupant] === positions[occupant]) {
        next[i] = positions[i];
        changed = true;
      }
    }
  } while (changed);
  return next;
}
