import './maze.css';
import { MAZE_COLORS, MAZE_HEIGHT, MAZE_ROUTES, MAZE_ROWS, MAZE_WIDTH, mazeX, mazeY } from './maze-scenario';

const AGENTS = MAZE_ROUTES.length;
const ROUNDS = 4;
const ACTIONS = [
  { name: 'Wait', color: '#9bb8ca' },
  { name: 'North', color: '#4fe1f6' },
  { name: 'South', color: '#ffa95a' },
  { name: 'West', color: '#ad9eff' },
  { name: 'East', color: '#66e5aa' },
] as const;
const BASE_ROUND_MS = 1300;

function actionIcon(action: number): string {
  if (action === 0) {
    return '<svg class="maze-action-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="4.5" fill="currentColor" /></svg>';
  }
  const rotation = [0, 0, 180, -90, 90][action];
  return `<svg class="maze-action-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 2 21 11h-5v11H8V11H3z" transform="rotate(${rotation} 12 12)" fill="currentColor" /></svg>`;
}

type Snapshot = {
  type: 'snapshot';
  action: number;
  positions: number[];
  goals: number[];
  chat: number[];
  trace: number[];
  previousActions: number[];
  inferenceMs: number;
  backend: string;
};
type WorkerMessage = Snapshot | { type: 'status'; text: string } | { type: 'error'; text: string };

function element<T extends Element = HTMLElement>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`Missing maze UI element: ${selector}`);
  return found;
}

const board = element('#maze-board');
const links = element<SVGSVGElement>('#maze-links');
const source = element('#maze-source');
const agentTitle = element('#maze-agent-title');
const agentSummary = element('#maze-agent-summary');
const roundStrip = element('#maze-round-strip');
const chart = element('#maze-chart');
const chartLegend = element('#maze-chart-legend');
const bars = element('#maze-bars');
const neighbors = element('#maze-neighbors');
const allAgents = element('#maze-all-agents');
const scrub = element<HTMLInputElement>('#maze-scrub');
const speedInput = element<HTMLInputElement>('#maze-speed');
const playButton = element<HTMLButtonElement>('#maze-play');
const nextButton = element<HTMLButtonElement>('#maze-next');
const previousButton = element<HTMLButtonElement>('#maze-prev');

let worker: Worker | null = null;
let frames: Snapshot[] = [];
let cursor = 0;
let selectedAgent = 0;
let playing = true;
let busy = true;
let speed = 1;
let timer: number | undefined;
const agentNodes: HTMLButtonElement[] = [];
const goalNodes: HTMLDivElement[] = [];

function atCell(node: HTMLElement, cell: number, inset: number): void {
  node.style.left = `${(mazeX(cell) + inset) / MAZE_WIDTH * 100}%`;
  node.style.top = `${(mazeY(cell) + inset) / MAZE_HEIGHT * 100}%`;
}

function buildBoard(): void {
  for (let y = 0; y < MAZE_HEIGHT; y++) {
    for (let x = 0; x < MAZE_WIDTH; x++) {
      const tile = document.createElement('div');
      tile.className = `maze-tile ${MAZE_ROWS[y][x] === '#' ? 'wall' : 'floor'}`;
      tile.style.left = `${x / MAZE_WIDTH * 100}%`;
      tile.style.top = `${y / MAZE_HEIGHT * 100}%`;
      board.insertBefore(tile, links);
    }
  }
  for (let i = 0; i < AGENTS; i++) {
    const goal = document.createElement('div');
    goal.className = 'maze-goal';
    goal.textContent = String(i + 1);
    goal.style.setProperty('--agent-color', MAZE_COLORS[i]);
    board.append(goal);
    goalNodes.push(goal);

    const agent = document.createElement('button');
    agent.type = 'button';
    agent.className = 'maze-agent';
    agent.setAttribute('aria-label', `Inspect agent ${String(i + 1).padStart(2, '0')}`);
    agent.style.setProperty('--agent-color', MAZE_COLORS[i]);
    agent.innerHTML = `<span class="arrow">${actionIcon(0)}</span><span class="agent-number">${i + 1}</span>`;
    agent.addEventListener('click', () => { selectedAgent = i; render(); });
    board.append(agent);
    agentNodes.push(agent);
  }
  chartLegend.innerHTML = ACTIONS.map((action) =>
    `<span><i style="background:${action.color}"></i>${action.name}</span>`
  ).join('');
}

function frameAt(step: number): { frame: Snapshot; round: number } {
  if (step === 0) return { frame: frames[0], round: 0 };
  return {
    frame: frames[Math.floor((step - 1) / ROUNDS)],
    round: ((step - 1) % ROUNDS) + 1,
  };
}

function probabilities(frame: Snapshot, round: number, agent: number): number[] {
  if (round === 0) return [0.2, 0.2, 0.2, 0.2, 0.2];
  const start = ((round - 1) * AGENTS + agent) * ACTIONS.length;
  return frame.trace.slice(start, start + ACTIONS.length);
}

function winner(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function setPlaying(value: boolean): void {
  playing = value;
  playButton.innerHTML = value ? 'Ⅱ <span>Pause</span>' : '▶ <span>Play</span>';
  schedule();
}

function schedule(): void {
  if (timer !== undefined) window.clearTimeout(timer);
  timer = undefined;
  if (!playing || busy || frames.length === 0) return;
  timer = window.setTimeout(() => { advanceRound(); }, BASE_ROUND_MS / speed);
}

function advanceRound(): void {
  if (busy || frames.length === 0) return;
  const lastRecorded = frames.length * ROUNDS;
  if (cursor < lastRecorded) {
    cursor++;
    render();
    schedule();
    return;
  }
  busy = true;
  nextButton.disabled = true;
  worker?.postMessage({ type: 'advance' });
}

function seek(step: number): void {
  if (frames.length === 0) return;
  setPlaying(false);
  cursor = Math.max(0, Math.min(frames.length * ROUNDS, step));
  render();
}

function renderLinks(frame: Snapshot, round: number): void {
  const pairs = new Set<string>();
  let content = '';
  for (let receiver = 0; receiver < AGENTS; receiver++) {
    for (let slot = 0; slot < 13; slot++) {
      const sender = frame.chat[receiver * 13 + slot];
      if (sender < 0 || sender >= AGENTS || sender === receiver) continue;
      const a = Math.min(receiver, sender);
      const b = Math.max(receiver, sender);
      const key = `${a}-${b}`;
      if (pairs.has(key)) continue;
      pairs.add(key);
      const selected = round > 0 && (a === selectedAgent || b === selectedAgent);
      content += `<line class="${selected ? 'selected' : ''}" x1="${mazeX(frame.positions[a]) + .5}" y1="${mazeY(frame.positions[a]) + .5}" x2="${mazeX(frame.positions[b]) + .5}" y2="${mazeY(frame.positions[b]) + .5}" />`;
    }
  }
  links.innerHTML = content;
}

function renderChart(frame: Snapshot, round: number): void {
  const x = (r: number) => 20 + r * 70;
  const y = (value: number) => 106 - Math.min(1, Math.max(0, value)) * 88;
  let svg = '<svg viewBox="0 0 320 120" role="img" aria-label="Selected agent intent probabilities over four communication rounds">';
  for (const level of [0, .5, 1]) svg += `<line class="grid" x1="16" y1="${y(level)}" x2="302" y2="${y(level)}" />`;
  svg += `<line class="marker" x1="${x(round)}" y1="14" x2="${x(round)}" y2="109" />`;
  for (let action = 0; action < ACTIONS.length; action++) {
    const points = Array.from({ length: ROUNDS + 1 }, (_, r) => `${x(r)},${y(probabilities(frame, r, selectedAgent)[action])}`).join(' ');
    svg += `<polyline class="trace" points="${points}" stroke="${ACTIONS[action].color}" />`;
    const current = probabilities(frame, round, selectedAgent)[action];
    svg += `<circle class="dot" cx="${x(round)}" cy="${y(current)}" r="3.5" fill="${ACTIONS[action].color}" />`;
  }
  chart.innerHTML = svg + '</svg>';
}

function renderInspector(frame: Snapshot, round: number): void {
  const current = probabilities(frame, round, selectedAgent);
  const intent = winner(current);
  const confidence = current[intent];
  agentTitle.textContent = `Agent ${String(selectedAgent + 1).padStart(2, '0')}`;
  agentTitle.style.color = MAZE_COLORS[selectedAgent];
  const goal = frame.goals[selectedAgent];
  const previousMove = ACTIONS[frame.previousActions[selectedAgent] ?? 0].name.toLowerCase();
  agentSummary.textContent = round === 0
    ? `Goal (${mazeX(goal)}, ${mazeY(goal)}). Communication has not started.`
    : `Goal (${mazeX(goal)}, ${mazeY(goal)}). Intent: ${ACTIONS[intent].name.toLowerCase()} at ${Math.round(confidence * 100)}%. Last executed move: ${previousMove}.`;

  roundStrip.innerHTML = Array.from({ length: ROUNDS }, (_, index) => {
    const r = index + 1;
    const values = probabilities(frame, r, selectedAgent);
    const action = winner(values);
    const confidence = Math.round(values[action] * 100);
    return `<button type="button" class="maze-round-chip ${r === round ? 'active' : ''} ${r > round ? 'future' : ''}" data-round="${r}" aria-label="Round ${r}: ${ACTIONS[action].name}, ${confidence}%"><small>Round ${r}</small><strong>${actionIcon(action)} ${confidence}%</strong></button>`;
  }).join('');
  for (const chip of roundStrip.querySelectorAll<HTMLButtonElement>('button')) {
    chip.addEventListener('click', () => seek(frame.action * ROUNDS + Number(chip.dataset.round)));
  }

  renderChart(frame, round);
  element('#maze-bars-title').textContent = round === 0 ? 'Before communication' : `Intent after round ${round}`;
  element('#maze-confidence').textContent = round === 0 ? 'Uniform starting state' : `${Math.round(confidence * 100)}% ${ACTIONS[intent].name.toLowerCase()}`;
  bars.innerHTML = ACTIONS.map((action, index) => `<div class="maze-bar ${index === intent && round > 0 ? 'active' : ''}"><span>${actionIcon(index)} ${action.name}</span><span class="track"><span class="fill" style="width:${(current[index] * 100).toFixed(1)}%;background:${action.color}"></span></span><output>${Math.round(current[index] * 100)}%</output></div>`).join('');

  const sources = new Set<number>();
  for (let slot = 0; slot < 13; slot++) {
    const id = frame.chat[selectedAgent * 13 + slot];
    if (id >= 0 && id < AGENTS && id !== selectedAgent) sources.add(id);
  }
  element('#maze-neighbor-count').textContent = `${sources.size} ${sources.size === 1 ? 'agent' : 'agents'}`;
  neighbors.innerHTML = sources.size
    ? [...sources].map((id) => `<span style="--agent-color:${MAZE_COLORS[id]}">Agent ${String(id + 1).padStart(2, '0')}</span>`).join('')
    : '<span>No agents in range</span>';
  allAgents.innerHTML = Array.from({ length: AGENTS }, (_, id) => {
    const values = probabilities(frame, round, id);
    const action = round === 0 ? 0 : winner(values);
    return `<button type="button" class="${id === selectedAgent ? 'selected' : ''}" data-agent="${id}" style="--agent-color:${MAZE_COLORS[id]}" aria-label="Agent ${String(id + 1).padStart(2, '0')}: ${round === 0 ? 'before communication' : ACTIONS[action].name}"><span>${String(id + 1).padStart(2, '0')}</span><b>${actionIcon(action)}</b></button>`;
  }).join('');
  for (const button of allAgents.querySelectorAll<HTMLButtonElement>('button')) {
    button.addEventListener('click', () => { selectedAgent = Number(button.dataset.agent); render(); });
  }
}

function render(): void {
  if (frames.length === 0) return;
  board.classList.add('ready');
  const { frame, round } = frameAt(cursor);
  element('#maze-action').textContent = String(frame.action + 1).padStart(3, '0');
  element('#maze-round').textContent = `${round} / ${ROUNDS}`;
  element('#maze-timeline-label').textContent = round === 0 ? 'Before communication' : `Communication round ${round} of ${ROUNDS}`;
  element('#maze-step-label').textContent = `Step ${cursor}`;
  scrub.max = String(frames.length * ROUNDS);
  scrub.value = String(cursor);
  previousButton.disabled = cursor === 0;
  nextButton.disabled = busy;

  let changes = 0;
  for (let i = 0; i < AGENTS; i++) {
    const current = probabilities(frame, round, i);
    const currentAction = winner(current);
    const priorAction = round > 1 ? winner(probabilities(frame, round - 1, i)) : -1;
    const changed = round > 1 && currentAction !== priorAction;
    if (changed) changes++;
    const agent = agentNodes[i];
    atCell(agent, frame.positions[i], .13);
    atCell(goalNodes[i], frame.goals[i], .21);
    agent.classList.toggle('selected', i === selectedAgent);
    agent.classList.remove('changed');
    if (changed) { void agent.offsetWidth; agent.classList.add('changed'); }
    agent.querySelector('.arrow')!.innerHTML = actionIcon(round === 0 ? 0 : currentAction);
    agent.title = `Agent ${i + 1}: ${round === 0 ? 'awaiting communication' : ACTIONS[currentAction].name.toLowerCase()} (${Math.round(current[currentAction] * 100)}%)`;
  }
  element('#maze-changed').textContent = round <= 1
    ? round === 0 ? 'Intents begin in the next round' : `${AGENTS} intents formed`
    : `${changes} changed intent${changes === 1 ? '' : 's'} this round`;
  renderLinks(frame, round);
  renderInspector(frame, round);
}

function bootWorker(): void {
  worker?.terminate();
  frames = [];
  cursor = 0;
  busy = true;
  board.classList.remove('ready');
  source.className = 'maze-source';
  source.querySelector('span')!.textContent = 'Loading model';
  nextButton.disabled = true;
  previousButton.disabled = true;
  const instance = new Worker(new URL('./maze.worker.ts', import.meta.url), { type: 'module' });
  worker = instance;
  instance.onmessage = (event: MessageEvent<WorkerMessage>) => {
    if (worker !== instance) return;
    const message = event.data;
    if (message.type === 'status') {
      source.querySelector('span')!.textContent = message.text;
    } else if (message.type === 'error') {
      busy = false;
      source.classList.add('error');
      source.querySelector('span')!.textContent = message.text;
      setPlaying(false);
      console.error(message.text);
    } else {
      const wasEmpty = frames.length === 0;
      frames.push(message);
      cursor = wasEmpty ? 0 : (frames.length - 1) * ROUNDS + 1;
      busy = false;
      source.classList.add('ready');
      source.querySelector('span')!.textContent = `DMM-0.8M on ${message.backend}`;
      render();
      schedule();
    }
  };
  instance.onerror = (error) => {
    source.classList.add('error');
    source.querySelector('span')!.textContent = 'Maze worker failed';
    setPlaying(false);
    console.error(error);
  };
  instance.postMessage({ type: 'start' });
}

buildBoard();
playButton.addEventListener('click', () => setPlaying(!playing));
nextButton.addEventListener('click', () => { setPlaying(false); advanceRound(); });
previousButton.addEventListener('click', () => seek(cursor - 1));
scrub.addEventListener('input', () => seek(Number(scrub.value)));
speedInput.addEventListener('input', () => {
  speed = Number(speedInput.value);
  const decimals = speed % 0.5 === 0 ? 1 : 2;
  element<HTMLOutputElement>('#maze-speed-value').value = `${speed.toFixed(decimals)}×`;
  schedule();
});
element<HTMLButtonElement>('#maze-reset').addEventListener('click', () => { setPlaying(true); bootWorker(); });
window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey || event.target instanceof HTMLInputElement) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); setPlaying(false); advanceRound(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); seek(cursor - 1); }
  if (event.key === ' ') { event.preventDefault(); setPlaying(!playing); }
});
bootWorker();
