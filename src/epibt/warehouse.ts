import './warehouse.css';
import '../qr-links.css';
import { fmt, formatCount, formatFixed, formatNumber, initI18n, tr } from '../i18n';
import { defaultWarehouse, MAX_CONFIG_BYTES, parseWarehouse, type WarehouseConfig } from './config';
import { WarehouseMap } from './map';
import { ACTIONS, ACTION_CODES, type Action, type Snapshot, type WorkerCommand, type WorkerReply } from './model';

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const canvas = element<HTMLCanvasElement>('epibt-map');
const mapView = new WarehouseMap(canvas);
const playButton = element<HTMLButtonElement>('epibt-play');
const stepButton = element<HTMLButtonElement>('epibt-next');
const speed = element<HTMLInputElement>('epibt-rate');
const fileInput = element<HTMLInputElement>('epibt-file');
const error = element('epibt-error');
const operation = element('epibt-operation');
const headingNames = ['East', 'South', 'West', 'North'] as const;
const headingGlyphs = ['→', '↓', '←', '↑'] as const;

let config = defaultWarehouse();
let worker: Worker | null = null;
let snapshot: Snapshot | null = null;
let previous: Snapshot | null = null;
let selected = 0;
let running = false;
let busy = true;
let timer: number | undefined;
let animationFrame: number | undefined;
let transitionStarted = 0;
let errorText = '';
let importVersion = 0;
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

function renderError(message: string): void {
  errorText = message;
  error.hidden = !message;
  error.textContent = tr(message);
}

function renderControls(): void {
  playButton.disabled = !snapshot || !worker;
  playButton.textContent = tr(running ? 'Pause' : 'Run');
  stepButton.disabled = !snapshot || !worker || busy || running;
  setText('epibt-state', tr(!worker ? 'EPIBT stopped' : !snapshot ? 'Loading EPIBT…' : running ? 'Running' : 'Paused'));
  setText('epibt-rate-value', fmt('{value} Hz', { value: speed.valueAsNumber }));
}

function renderInspector(): void {
  const agent = snapshot?.agents[selected];
  setText('epibt-selected', fmt('Robot {id}', { id: String(selected + 1).padStart(3, '0') }));
  setText('epibt-stage', agent ? tr(agent.stage) : '—');
  setText('epibt-position', agent ? coordinates(agent.cell) : '—');
  setText('epibt-heading', agent ? `${headingGlyphs[agent.direction]} ${tr(headingNames[agent.direction])}` : '—');
  setText('epibt-task', agent ? formatNumber(agent.task) : '—');
  setText('epibt-goal', agent ? coordinates(agent.target) : '—');
  setText('epibt-action', actionLabel(agent?.operation[0]));
  setText('epibt-priority', agent?.priority == null ? '—' : formatNumber(agent.priority));
  setText('epibt-distance', agent ? formatNumber(agent.distance) : '—');
  setText('epibt-turns', agent ? formatNumber(agent.turns) : '—');
  setText('epibt-waits', agent ? formatNumber(agent.waits) : '—');
  setText('epibt-agent-completed', agent ? formatNumber(agent.completed) : '—');

  const entries = agent?.operation ?? [];
  operation.replaceChildren(...Array.from({ length: 3 }, (_, index) => {
    const action = entries[index];
    const item = document.createElement('li');
    item.textContent = action === undefined ? '—' : `${ACTION_CODES[action]} · ${actionLabel(action)}`;
    return item;
  }));

  rows.forEach(({ row, button }, index) => {
    row.toggleAttribute('data-selected', index === selected);
    button.setAttribute('aria-pressed', String(index === selected));
  });
}

function renderFleet(): void {
  setText('epibt-agent-count', formatCount(config.agents.length, 'agents'));
  rows.forEach((row, index) => {
    const agent = snapshot?.agents[index];
    row.button.setAttribute('aria-label', fmt('Robot {id}', { id: row.button.textContent ?? '' }));
    row.position.textContent = agent ? coordinates(agent.cell) : '—';
    row.heading.textContent = agent ? headingGlyphs[agent.direction] : '—';
    row.heading.title = agent ? tr(headingNames[agent.direction]) : '';
    row.action.textContent = actionLabel(agent?.operation[0]);
  });
}

function selectAgent(id: number): void {
  selected = id;
  renderInspector();
  draw();
}

function rebuildFleet(): void {
  const root = element('epibt-agents');
  root.replaceChildren();
  rows = config.agents.map((_, id) => {
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
  const duration = Math.min(170, 950 / speed.valueAsNumber);
  const blend = previous && snapshot ? Math.min(1, (performance.now() - transitionStarted) / duration) : 1;
  mapView.draw(config, snapshot?.agents ?? [], previous?.agents ?? [], blend, selected);
  animationFrame = blend < 1 ? window.requestAnimationFrame(draw) : undefined;
}

function schedule(): void {
  if (timer !== undefined) window.clearTimeout(timer);
  timer = running && snapshot && !busy ? window.setTimeout(requestStep, 1000 / speed.valueAsNumber) : undefined;
}

function requestStep(): void {
  if (!worker || !snapshot || busy) return;
  busy = true;
  renderControls();
  worker.postMessage({ type: 'step' } satisfies WorkerCommand);
}

function fail(message: string): void {
  running = false;
  busy = false;
  worker?.terminate();
  worker = null;
  schedule();
  renderControls();
  renderError(message);
}

function start(next: WarehouseConfig, autoplay = false): void {
  worker?.terminate();
  config = next;
  snapshot = null;
  previous = null;
  selected = Math.min(selected, config.agents.length - 1);
  running = autoplay;
  busy = true;
  schedule();
  renderError('');
  setText('epibt-dimensions', `${config.map[0].length} × ${config.map.length}`);
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
    previous = snapshot;
    snapshot = data;
    transitionStarted = performance.now();
    renderSnapshot();
    draw();
    schedule();
  };
  instance.onerror = () => {
    if (worker === instance) fail('Could not start the EPIBT worker.');
  };
  instance.postMessage({ type: 'configure', config } satisfies WorkerCommand);
  renderSnapshot();
  draw();
}

playButton.addEventListener('click', () => {
  running = !running;
  renderControls();
  schedule();
});
stepButton.addEventListener('click', requestStep);
element('epibt-reset').addEventListener('click', () => start(config));
speed.addEventListener('input', () => {
  renderControls();
  schedule();
});
canvas.addEventListener('pointerdown', (event) => {
  const id = mapView.pick(event);
  if (id >= 0) selectAgent(id);
});
new ResizeObserver(draw).observe(canvas);

element('epibt-import').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  fileInput.value = '';
  if (!file) return;
  const version = ++importVersion;
  try {
    if (file.size > MAX_CONFIG_BYTES) throw new Error('The JSON file is larger than 1 MB.');
    const text = await file.text();
    if (version !== importVersion) return;
    start(parseWarehouse(JSON.parse(text)));
  } catch (problem) {
    if (version !== importVersion) return;
    renderError(problem instanceof SyntaxError ? 'The file does not contain valid JSON.'
      : problem instanceof Error ? problem.message : 'Could not read the JSON file.');
  }
});
element('epibt-export').addEventListener('click', () => {
  const file = new Blob([JSON.stringify(config, null, 2) + '\n'], { type: 'application/json' });
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'epibt-warehouse.json';
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
});

window.addEventListener('demo:languagechange', () => {
  renderSnapshot();
  renderError(errorText);
});
start(config, true);
