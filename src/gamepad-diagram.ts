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
  { side: 'left', y: 30, line: [[238, 37], [225, 37], [208, 26]], control: 'Create', warehouse: 'Stop simulation' },
  { side: 'left', y: 56, line: [[233, 58], [217, 58], [204, 52]], control: 'D-pad', warehouse: '↑ / ↓ · Speed', manual: 'Drive agent', editor: 'Agents ↑↓ ±1 · ←→ ±10' },
  { side: 'left', y: 82, line: [[244, 77], [221, 77], [204, 78]], control: 'Left stick', warehouse: 'Pan camera', manual: 'Drive agent', editor: 'Aim at cell' },
  { side: 'left', y: 108, line: [[248, 88], [224, 100], [209, 104]], control: 'L3', warehouse: 'Show / hide help', editor: 'Show / hide help' },
  { side: 'right', y: 16, line: [[289, 31], [289, 12]], control: 'Touchpad', warehouse: 'Break agent', touchpad: true },
  { side: 'right', y: 34, line: [[316, 38], [335, 38], [347, 30]], control: 'Options', warehouse: 'Pause / resume', editor: 'Apply map' },
  { side: 'right', y: 52, line: [[324, 46], [336, 43], [351, 48]], control: '△', warehouse: 'Open layout editor', editor: 'Discard edits' },
  { side: 'right', y: 70, line: [[314, 57], [317, 52], [343, 52], [350, 56]], control: '□', warehouse: 'Clear selection', editor: 'Undo edit' },
  { side: 'right', y: 88, line: [[335, 57], [346, 60], [354, 72]], control: '○', warehouse: 'Take control', manual: 'Release control', editor: 'Discard edits' },
  { side: 'right', y: 106, line: [[324, 67], [339, 75], [354, 91]], control: '×', warehouse: 'Reset camera', editor: 'Add or remove pallet' },
  { side: 'right', y: 124, line: [[311, 80], [328, 90], [348, 112]], control: 'Right stick', warehouse: 'Orbit camera' },
];

const shoulderCallouts: readonly Callout[] = [
  { side: 'left', y: 20, line: [[224, 16], [201, 16]], control: 'L1', warehouse: 'Previous agent' },
  { side: 'right', y: 20, line: [[330, 16], [354, 16]], control: 'R1', warehouse: 'Next agent' },
  { side: 'left', y: 38, line: [[218, 28], [204, 34]], control: 'L2', warehouse: 'Zoom out', editor: 'Zoom out' },
  { side: 'right', y: 38, line: [[337, 28], [351, 34]], control: 'R2', warehouse: 'Zoom in', editor: 'Zoom in' },
];

function drawCallouts(callouts: readonly Callout[], mode: Mode, touchpad: boolean): string {
  return callouts.map((callout) => {
    const action = mode === 'manual' ? callout.manual ?? callout.warehouse : callout[mode];
    if (!action || (callout.touchpad && !touchpad)) return '';
    const { side, y, line, control } = callout;
    const left = side === 'left';
    const textX = left ? 12 : 376;
    const lineEnd = left ? 194 : 368;
    const points = [...line, [lineEnd, y - 4]].map((point) => point.join(',')).join(' ');
    return `<g>
      <polyline class="gamepad-leader" points="${points}" aria-hidden="true"/>
      <circle class="gamepad-anchor" cx="${line[0][0]}" cy="${line[0][1]}" r="1.9" aria-hidden="true"/>
      <text class="gamepad-action-label" x="${textX}" y="${y}"><tspan class="gamepad-control-label">${tr(control)}</tspan> · ${tr(action)}</text>
    </g>`;
  }).join('');
}

export function createGamepadDiagram(help: HTMLDetailsElement): (pad: Gamepad | null, editing: boolean, controlling: boolean) => void {
  help.insertAdjacentHTML('beforeend', diagramMarkup);
  const top = help.querySelector('[data-gamepad-top-callouts]')!;
  const shoulder = help.querySelector('[data-gamepad-shoulder-callouts]')!;
  const speedChord = help.querySelector('[data-gamepad-speed-chord]')!;
  const cycleNote = help.querySelector('[data-gamepad-cycle-note]')!;
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
    cycleNote.toggleAttribute('hidden', mode === 'editor');
    cycleNote.textContent = tr('Hold L1 / R1 to switch faster');
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
