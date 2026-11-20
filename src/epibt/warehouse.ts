import './warehouse.css';
import '../demo-header.css';
import { fmt, formatCount, formatFixed, formatNumber, initI18n, tr } from '../i18n';
import type { WarehouseConfig } from './config';
import { LayoutEditor } from './editor';
import { WarehouseMap } from './map';
import { ACTIONS, ACTION_CODES, OPERATION_LENGTHS, type Action, type OperationLength, type Snapshot, type WorkerCommand, type WorkerReply } from './model';
import { WAREHOUSE_PRESETS } from './presets';

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const canvas = element<HTMLCanvasElement>('epibt-map');
const mapView = new WarehouseMap(canvas);
const playButton = element<HTMLButtonElement>('epibt-play');
const stepButton = element<HTMLButtonElement>('epibt-next');
const speed = element<HTMLInputElement>('epibt-rate');
const editButton = element<HTMLButtonElement>('epibt-open-editor');
const previousRobot = element<HTMLButtonElement>('epibt-prev-robot');
const nextRobot = element<HTMLButtonElement>('epibt-next-robot');
const fleetScroll = element('epibt-fleet-scroll');
const lengthButtons = OPERATION_LENGTHS.map(length => ({
  length, button: element<HTMLButtonElement>(`epibt-length-${length}`),
}));
const error = element('epibt-error');
const operation = element('epibt-operation');
const planActionLabels = ['Forward', 'Right', 'Left', 'Wait'] as const;
const playbackRates = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  20, 30, 40, 50, 60, 70, 80, 90, 100,
  200, 300, 400, 500, 600, 700, 800, 900, 1000,
];
const MAX_PLAYBACK_UPDATES_PER_SECOND = 50;
const headingNames = ['East', 'South', 'West', 'North'] as const;
const headingGlyphs = ['→', '↓', '←', '↑'] as const;

let config = WAREHOUSE_PRESETS[0].config;
let worker: Worker | null = null;
let snapshot: Snapshot | null = null;
let previous: Snapshot | null = null;
let selected = 0;
let operationLength: OperationLength = 3;
let requestedLength: OperationLength | null = null;
let running = false;
let busy = true;
let timer: number | undefined;
let nextStepAt: number | undefined;
let animationFrame: number | undefined;
let transitionStarted = 0;
let errorText = '';
let resumeAfterEditor = false;
let rows: Array<{ row: HTMLTableRowElement; button: HTMLButtonElement; position: HTMLElement; heading: HTMLElement; action: HTMLElement }> = [];

initI18n();

function setText(id: string, text: string): void {
  const node = element(id);
  if (node.textContent !== text) node.textContent = text;
}

function coordinates(cell: number): string {
  const width = config.map[0].length;
  return `${cell % width}, ${Math.floor(cell / width)}`;
}

function actionLabel(action: Action | undefined): string {
  return action === undefined ? '—' : tr(ACTIONS[action]);
}

function playbackRate(): number {
  return playbackRates[speed.valueAsNumber];
}

function renderError(message: string): void {
  errorText = message;
  error.hidden = !message || !!editor.draft;
  error.textContent = tr(message);
}

function renderControls(): void {
  const ready = !!snapshot && !!worker;
  playButton.disabled = !ready;
  playButton.textContent = tr(running ? 'Pause' : 'Run');
  stepButton.disabled = !ready || busy || running || requestedLength !== null;
  const state = !worker ? 'EPIBT stopped' : !snapshot ? 'Loading EPIBT…'
    : requestedLength !== null ? 'Changing the planning horizon…' : running ? 'Running' : 'Paused';
  setText('epibt-state', tr(state));
  const rateLabel = formatCount(playbackRate(), 'stepsPerSecond');
  setText('epibt-rate-value', rateLabel);
  if (speed.getAttribute('aria-valuetext') !== rateLabel) speed.setAttribute('aria-valuetext', rateLabel);
  for (const { length, button } of lengthButtons) {
    button.disabled = !ready || requestedLength !== null;
    button.setAttribute('aria-pressed', String(length === operationLength));
  }
}

function renderInspector(): void {
  const agent = snapshot?.agents[selected];
  previousRobot.disabled = nextRobot.disabled = !agent || config.agentCount <= 1;
  setText('epibt-selected', fmt('Agent {id}', { id: String(selected + 1).padStart(3, '0') }));
  setText('epibt-stage', agent ? tr(agent.stage) : '—');
  setText('epibt-position', agent ? coordinates(agent.cell) : '—');
  setText('epibt-heading', agent ? `${headingGlyphs[agent.direction]} ${tr(headingNames[agent.direction])}` : '—');
  setText('epibt-task', agent ? formatNumber(agent.task) : '—');
  setText('epibt-goal', agent ? coordinates(agent.target) : '—');
  setText('epibt-action', actionLabel(agent?.lastAction));
  setText('epibt-priority', agent?.priority == null ? '—' : formatNumber(agent.priority));
  setText('epibt-distance', agent ? formatNumber(agent.distance) : '—');
  setText('epibt-turns', agent ? formatNumber(agent.turns) : '—');
  setText('epibt-waits', agent ? formatNumber(agent.waits) : '—');
  setText('epibt-agent-completed', agent ? formatNumber(agent.completed) : '—');

  const entries = agent?.operation ?? [];
  operation.replaceChildren(...Array.from({ length: operationLength }, (_, index) => {
    const action = entries[index];
    const item = document.createElement('li');
    item.textContent = action === undefined ? '—' : tr(planActionLabels[action]);
    if (action !== undefined) {
      const description = actionLabel(action);
      item.title = `${ACTION_CODES[action]} · ${description}`;
      item.setAttribute('aria-label', description);
    }
    return item;
  }));

  rows.forEach(({ row, button }, index) => {
    row.toggleAttribute('data-selected', index === selected);
    button.setAttribute('aria-pressed', String(index === selected));
  });
}

function renderFleet(): void {
  setText('epibt-agent-count', formatCount(config.agentCount, 'agents'));
  rows.forEach((row, index) => {
    const agent = snapshot?.agents[index];
    row.button.setAttribute('aria-label', fmt('Agent {id}', { id: row.button.textContent ?? '' }));
    row.position.textContent = agent ? coordinates(agent.cell) : '—';
    row.heading.textContent = agent ? headingGlyphs[agent.direction] : '—';
    row.heading.title = agent ? tr(headingNames[agent.direction]) : '';
    row.action.textContent = actionLabel(agent?.lastAction);
  });
}

function selectAgent(id: number): void {
  selected = id;
  renderInspector();
  draw();
  const row = rows[id].row.getBoundingClientRect();
  const top = fleetScroll.querySelector('th')!.getBoundingClientRect().bottom;
  const bottom = fleetScroll.getBoundingClientRect().top + fleetScroll.clientHeight;
  if (row.top < top) fleetScroll.scrollTop += row.top - top;
  else if (row.bottom > bottom) fleetScroll.scrollTop += row.bottom - bottom;
}

function rebuildFleet(): void {
  const root = element('epibt-agents');
  root.replaceChildren();
  rows = Array.from({ length: config.agentCount }, (_, id) => {
    const row = document.createElement('tr');
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = String(id + 1).padStart(3, '0');
    button.addEventListener('click', () => selectAgent(id));
    row.insertCell().append(button);
    const position = row.insertCell();
    const heading = row.insertCell();
    const action = row.insertCell();
    row.addEventListener('click', event => {
      if (event.target !== button) selectAgent(id);
    });
    root.append(row);
    return { row, button, position, heading, action };
  });
}

function renderSnapshot(): void {
  setText('epibt-step', snapshot ? formatNumber(snapshot.step) : '—');
  setText('epibt-completed', snapshot ? formatNumber(snapshot.completed) : '—');
  setText('epibt-throughput', snapshot?.step ? formatFixed(snapshot.completed / snapshot.step, 3) : '—');
  setText('epibt-time', snapshot?.planningMs == null ? '—' : formatFixed(snapshot.planningMs, 2));
  renderInspector();
  renderFleet();
  renderControls();
}

function draw(): void {
  if (animationFrame !== undefined) window.cancelAnimationFrame(animationFrame);
  if (editor.draft) {
    animationFrame = undefined;
    mapView.draw(editor.draft.map, [], [], 1, -1, editor.cursor);
    return;
  }
  const duration = Math.min(170, 950 / playbackRate());
  const blend = previous && snapshot ? Math.min(1, (performance.now() - transitionStarted) / duration) : 1;
  mapView.draw(config.map, snapshot?.agents ?? [], previous?.agents ?? [], blend, selected);
  animationFrame = blend < 1 ? window.requestAnimationFrame(draw) : undefined;
}

function schedule(): void {
  if (timer !== undefined) window.clearTimeout(timer);
  timer = undefined;
  if (!worker || !snapshot || busy) return;
  if (requestedLength !== null) {
    busy = true;
    worker.postMessage({ type: 'length', length: requestedLength } satisfies WorkerCommand);
  } else if (running) {
    const rate = playbackRate();
    const count = Math.ceil(rate / MAX_PLAYBACK_UPDATES_PER_SECOND);
    const interval = count * 1000 / rate;
    const now = performance.now();
    const deadline = nextStepAt ?? now + interval;
    timer = window.setTimeout(() => {
      const started = performance.now();
      nextStepAt = Math.max(deadline + interval, started + 1);
      requestSteps(count);
    }, Math.max(0, Math.ceil(deadline - now)));
  }
}

function requestSteps(count: number): void {
  if (!worker || !snapshot || busy) return;
  busy = true;
  renderControls();
  worker.postMessage({ type: 'step', count } satisfies WorkerCommand);
}

function fail(message: string): void {
  running = false;
  busy = false;
  requestedLength = null;
  worker?.terminate();
  worker = null;
  schedule();
  renderControls();
  renderError(message);
}

function start(next: WarehouseConfig, autoplay = false): void {
  worker?.terminate();
  operationLength = requestedLength ?? operationLength;
  requestedLength = null;
  config = next;
  snapshot = null;
  previous = null;
  nextStepAt = undefined;
  selected = Math.min(selected, config.agentCount - 1);
  running = autoplay;
  busy = true;
  schedule();
  renderError('');
  renderDimensions();
  rebuildFleet();
  const instance = new Worker(new URL('./warehouse.worker.ts', import.meta.url), { type: 'module' });
  worker = instance;
  instance.onmessage = ({ data }: MessageEvent<WorkerReply>) => {
    if (worker !== instance) return;
    if (data.type === 'error') {
      fail(data.message);
      return;
    }
    busy = false;
    if (data.type === 'length-error') {
      requestedLength = null;
      renderControls();
      renderError('Could not change the planning horizon.');
      schedule();
      return;
    }
    if (data.operationLength !== operationLength) {
      operationLength = data.operationLength;
      requestedLength = null;
    }
    previous = snapshot && data.step === snapshot.step + 1 ? snapshot : null;
    snapshot = data;
    transitionStarted = performance.now();
    renderSnapshot();
    draw();
    schedule();
  };
  instance.onerror = () => {
    if (worker === instance) fail('Could not start the EPIBT worker.');
  };
  instance.postMessage({ type: 'configure', config, length: operationLength } satisfies WorkerCommand);
  renderSnapshot();
  draw();
}

function renderDimensions(): void {
  const { map } = editor.draft ?? config;
  setText('epibt-dimensions', `${map[0].length} × ${map.length}`);
}

function labelMap(): void {
  canvas.setAttribute('aria-label', tr(editor.draft
    ? 'Editable warehouse map. Use arrow keys and Space to paint.'
    : 'Warehouse map with oriented agents'));
}

function showEditor(editing: boolean): void {
  canvas.tabIndex = editing ? 0 : -1;
  document.querySelector('.epibt-app')?.toggleAttribute('data-editing', editing);
  element('epibt-editor').hidden = !editing;
  for (const id of ['epibt-legend', 'epibt-insights', 'epibt-controls', 'epibt-length-control', 'epibt-open-editor']) {
    element(id).hidden = editing;
  }
  labelMap();
  renderError(errorText);
}

const editor = new LayoutEditor(canvas, mapView, () => {
  renderDimensions();
  draw();
}, next => {
  showEditor(false);
  if (next) {
    start(next, true);
  } else {
    running = resumeAfterEditor && !!worker;
    renderControls();
    renderDimensions();
    draw();
    schedule();
  }
  editButton.focus({ preventScroll: true });
});

editButton.addEventListener('click', () => {
  resumeAfterEditor = running;
  running = false;
  schedule();
  editor.open(config);
  showEditor(true);
  element('epibt-editor-title').focus({ preventScroll: true });
});

playButton.addEventListener('click', () => {
  running = !running;
  nextStepAt = undefined;
  renderControls();
  schedule();
});
stepButton.addEventListener('click', () => requestSteps(1));
element('epibt-reset').addEventListener('click', () => start(config));
previousRobot.addEventListener('click', () => selectAgent((selected + config.agentCount - 1) % config.agentCount));
nextRobot.addEventListener('click', () => selectAgent((selected + 1) % config.agentCount));
for (const { length, button } of lengthButtons) {
  button.addEventListener('click', () => {
    if (length === operationLength) return;
    requestedLength = length;
    renderError('');
    renderControls();
    schedule();
  });
}
speed.addEventListener('input', () => {
  nextStepAt = undefined;
  renderControls();
  schedule();
});
canvas.addEventListener('pointerdown', (event) => {
  if (editor.draft) return;
  const id = mapView.pick(event);
  if (id >= 0) selectAgent(id);
});
new ResizeObserver(draw).observe(canvas);

window.addEventListener('demo:languagechange', () => {
  renderSnapshot();
  renderError(errorText);
  labelMap();
});
start(config, true);
