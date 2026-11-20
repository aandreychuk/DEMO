import type { WarehouseConfig } from './config';

export const ACTIONS = ['Forward', 'Turn right', 'Turn left', 'Wait'] as const;
export const ACTION_CODES = ['F', 'R', 'L', 'W'] as const;
export const OPERATION_LENGTHS = [3, 4, 5] as const;
const ROTATION = [0, 1, 3, 0] as const;

export type Action = 0 | 1 | 2 | 3;
export type OperationLength = typeof OPERATION_LENGTHS[number];
export type Pose = { cell: number; direction: number };

export function nextPose(pose: Pose, action: Action, width: number): Pose {
  const forwardCells = [1, width, -1, -width];
  return {
    cell: pose.cell + (action === 0 ? forwardCells[pose.direction] : 0),
    direction: (pose.direction + ROTATION[action]) % 4,
  };
}

export type Agent = Pose & {
  target: number;
  task: number;
  stage: 'Pickup' | 'Delivery';
  lastAction?: Action;
  operation: Action[];
  priority: number | null;
  completed: number;
  distance: number;
  turns: number;
  waits: number;
};

export type Snapshot = {
  type: 'snapshot';
  step: number;
  completed: number;
  operationLength: OperationLength;
  planningMs: number | null;
  agents: Agent[];
};

export type WorkerReply = Snapshot | { type: 'error'; message: string } | { type: 'length-error' };
export type WorkerCommand =
  | { type: 'configure'; config: WarehouseConfig; length: OperationLength }
  | { type: 'length'; length: OperationLength }
  | { type: 'step'; count: number };
