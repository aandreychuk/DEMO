import { hasGamepadTouchpad } from './gamepad';
import { tr } from './i18n';
import diagramMarkup from './views/gamepad-diagram.html?raw';

type Mode = 'warehouse' | 'manual' | 'editor';
type Point = readonly [x: number, y: number];
type Callout = {
  side: 'left' | 'right';
  y: number;
  line: readonly Point[];
  control: string;
  warehouse: string;
  manual?: string;
  editor?: string;
  touchpad?: true;
};

const topCallouts: readonly Callout[] = [
  { side: 'left', y: 43, line: [[245, 75], [222, 75], [193, 48]], control: 'Create', warehouse: 'Stop simulation' },
  { side: 'left', y: 89, line: [[234, 116], [199, 116], [176, 94]], control: 'D-pad', warehouse: '↑ / ↓ · Speed', manual: 'Drive agent', editor: 'Agents ↑/↓ ±1 · ←/→ ±10' },
  { side: 'left', y: 141, line: [[255, 152], [212, 152], [187, 146]], control: 'Left stick', warehouse: 'Pan camera', manual: 'Drive agent', editor: 'Aim at cell' },
  { side: 'left', y: 195, line: [[274, 157], [274, 180], [236, 200]], control: 'L3', warehouse: 'Open layout editor', editor: 'Discard edits' },
  { side: 'right', y: 23, line: [[342, 63], [342, 28]], control: 'Touchpad', warehouse: 'Break agent', touchpad: true },
  { side: 'right', y: 55, line: [[395, 76], [431, 76], [451, 60]], control: 'Options', warehouse: 'Pause / resume', editor: 'Apply & restart' },
  { side: 'right', y: 85, line: [[390, 113], [384, 99], [384, 83], [435, 83], [452, 90]], control: '□', warehouse: 'Clear selection', editor: 'Undo edit' },
  { side: 'right', y: 115, line: [[430, 113], [449, 113], [460, 120]], control: '○', warehouse: 'Take control', manual: 'Release control', editor: 'Discard edits' },
  { side: 'right', y: 145, line: [[410, 133], [438, 136], [460, 150]], control: '×', warehouse: 'Select at reticle', manual: '', editor: 'Add or remove pallet' },
  { side: 'right', y: 175, line: [[384, 157], [417, 166], [439, 180]], control: 'Right stick', warehouse: 'Orbit camera' },
  { side: 'right', y: 213, line: [[366, 157], [366, 184], [403, 218]], control: 'R3', warehouse: 'Reset camera' },
];

const shoulderCallouts: readonly Callout[] = [
  { side: 'left', y: 38, line: [[218, 54], [197, 54], [184, 43]], control: 'L1', warehouse: 'Previous agent' },
  { side: 'right', y: 38, line: [[422, 54], [443, 54], [456, 43]], control: 'R1', warehouse: 'Next agent' },
  { side: 'left', y: 77, line: [[205, 77], [184, 82]], control: 'L2', warehouse: 'Zoom out', editor: 'Zoom out' },
  { side: 'right', y: 77, line: [[435, 77], [456, 82]], control: 'R2', warehouse: 'Zoom in', editor: 'Zoom in' },
];

function drawCallouts(callouts: readonly Callout[], mode: Mode, touchpad: boolean): string {
  return callouts.map((callout) => {
    const action = mode === 'manual' ? callout.manual ?? callout.warehouse : callout[mode];
    if (!action || (callout.touchpad && !touchpad)) return '';
    const { side, y, line, control } = callout;
    const left = side === 'left';
    const textX = left ? 14 : 480;
    const lineEnd = left ? 155 : 470;
    const points = [...line, [lineEnd, y + 5]].map((point) => point.join(',')).join(' ');
    return `<g>
      <polyline class="gamepad-leader" points="${points}" aria-hidden="true"/>
      <circle class="gamepad-anchor" cx="${line[0][0]}" cy="${line[0][1]}" r="2.5" aria-hidden="true"/>
      <text class="gamepad-control-label" x="${textX}" y="${y}">${tr(control)}</text>
      <text class="gamepad-action-label" x="${textX}" y="${y + 14}">${tr(action)}</text>
    </g>`;
  }).join('');
}

export function createGamepadDiagram(): (pad: Gamepad | null, editing: boolean, controlling: boolean) => void {
  const help = document.querySelector<HTMLDetailsElement>('#gamepad-help')!;
  help.insertAdjacentHTML('beforeend', diagramMarkup);
  const top = help.querySelector('[data-gamepad-top-callouts]')!;
  const shoulder = help.querySelector('[data-gamepad-shoulder-callouts]')!;
  const speedChord = help.querySelector('[data-gamepad-speed-chord]')!;
  const topTitle = help.querySelector('#gamepad-top-title')!;
  const shoulderTitle = help.querySelector('#gamepad-shoulder-title')!;
  let mode: Mode | null = null;
  let touchpad = false;

  function render(): void {
    if (!mode) return;
    topTitle.textContent = tr('Top view');
    shoulderTitle.textContent = tr('Trigger-side view');
    top.innerHTML = drawCallouts(topCallouts, mode, touchpad);
    shoulder.innerHTML = drawCallouts(shoulderCallouts, mode, touchpad);
    speedChord.toggleAttribute('hidden', mode !== 'manual');
    speedChord.querySelector('text')!.textContent = tr('Hold L2 + R2 and press ↑ / ↓ to change speed');
  }

  window.addEventListener('demo:languagechange', render);
  return (pad, editing, controlling) => {
    const nextMode = pad ? (editing ? 'editor' : controlling ? 'manual' : 'warehouse') : null;
    const nextTouchpad = pad !== null && hasGamepadTouchpad(pad);
    if (mode === nextMode && touchpad === nextTouchpad) return;
    mode = nextMode;
    touchpad = nextTouchpad;
    help.hidden = !mode;
    render();
  };
}
