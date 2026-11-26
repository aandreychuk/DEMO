export type MoveKey = 'w' | 'a' | 's' | 'd';

const TOUCHPAD_BUTTON_INDEX = 17;

export const GamepadButton = {
  cross: 1 << 0,
  circle: 1 << 1,
  square: 1 << 2,
  triangle: 1 << 3,
  l1: 1 << 4,
  r1: 1 << 5,
  create: 1 << 8,
  options: 1 << 9,
  up: 1 << 12,
  down: 1 << 13,
  left: 1 << 14,
  right: 1 << 15,
  touchpad: 1 << TOUCHPAD_BUTTON_INDEX,
} as const;

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

export function hasGamepadTouchpad(pad: Gamepad): boolean {
  // Some controllers use this extra button for Share or Capture instead.
  return pad.buttons.length > TOUCHPAD_BUTTON_INDEX && /dualsense|dualshock|vendor:\s*054c\b/i.test(pad.id);
}

export function pressedGamepadButtons(pad: Gamepad): number {
  let pressed = 0;
  for (let index = 0; index <= 15; index++) {
    if (pad.buttons[index]?.pressed) pressed |= 1 << index;
  }
  if (pad.buttons[TOUCHPAD_BUTTON_INDEX]?.pressed && hasGamepadTouchpad(pad)) pressed |= GamepadButton.touchpad;
  return pressed;
}

export function cameraStick(x: number, y: number): [number, number] {
  const magnitude = Math.hypot(x, y);
  if (magnitude <= STICK_DEAD_ZONE) return [0, 0];
  const strength = (Math.min(magnitude, 1) - STICK_DEAD_ZONE) / (1 - STICK_DEAD_ZONE);
  return [x / magnitude * strength, y / magnitude * strength];
}

export function gamepadMoveKey(pad: Gamepad, previous: MoveKey | null, useDpadButtons: boolean): MoveKey | null {
  const dpadX = useDpadButtons ? Number(!!pad.buttons[15]?.pressed) - Number(!!pad.buttons[14]?.pressed) : 0;
  const dpadY = useDpadButtons ? Number(!!pad.buttons[13]?.pressed) - Number(!!pad.buttons[12]?.pressed) : 0;
  const useDpad = dpadX !== 0 || dpadY !== 0;
  const x = useDpad ? dpadX : (pad.axes[0] ?? 0);
  const y = useDpad ? dpadY : (pad.axes[1] ?? 0);
  const horizontal = Math.abs(x) >= Math.abs(y);
  const key = horizontal ? (x < 0 ? 'a' : 'd') : (y < 0 ? 'w' : 's');
  const threshold = key === previous ? 0.35 : DRIVE_THRESHOLD;
  return Math.abs(horizontal ? x : y) >= threshold ? key : null;
}

export function gamepadSpeedModifier(pad: Gamepad): boolean {
  return (pad.buttons[6]?.value ?? 0) >= 0.5 && (pad.buttons[7]?.value ?? 0) >= 0.5;
}
