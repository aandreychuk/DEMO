export type MoveKey = 'w' | 'a' | 's' | 'd';

const STICK_DEAD_ZONE = 0.22;
const DRIVE_THRESHOLD = 0.5;

let readGamepads: (() => (Gamepad | null)[]) | undefined = navigator.getGamepads?.bind(navigator);

export function getActiveGamepad(): Gamepad | null {
  if (!readGamepads || document.hidden || !document.hasFocus()) return null;
  try {
    return readGamepads().find((pad) => pad?.connected && pad.mapping === 'standard') ?? null;
  } catch (error) {
    if (!(error instanceof DOMException) || error.name !== 'SecurityError') throw error;
    // A host can disable the Gamepad API in its embedded browser.
    readGamepads = undefined;
    return null;
  }
}

export function cameraStick(x: number, y: number): [number, number] {
  const magnitude = Math.hypot(x, y);
  if (magnitude <= STICK_DEAD_ZONE) return [0, 0];
  const strength = (Math.min(magnitude, 1) - STICK_DEAD_ZONE) / (1 - STICK_DEAD_ZONE);
  return [x / magnitude * strength, y / magnitude * strength];
}

export function gamepadMoveKey(pad: Gamepad, previous: MoveKey | null): MoveKey | null {
  const dpadX = Number(!!pad.buttons[15]?.pressed) - Number(!!pad.buttons[14]?.pressed);
  const dpadY = Number(!!pad.buttons[13]?.pressed) - Number(!!pad.buttons[12]?.pressed);
  const useDpad = dpadX !== 0 || dpadY !== 0;
  const x = useDpad ? dpadX : (pad.axes[0] ?? 0);
  const y = useDpad ? dpadY : (pad.axes[1] ?? 0);
  const horizontal = Math.abs(x) >= Math.abs(y);
  const key = horizontal ? (x < 0 ? 'a' : 'd') : (y < 0 ? 'w' : 's');
  const threshold = key === previous ? 0.35 : DRIVE_THRESHOLD;
  return Math.abs(horizontal ? x : y) >= threshold ? key : null;
}
