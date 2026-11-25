import { onViewActivityChange } from './demo-view';
import { MAZE_COLORS, MAZE_HEIGHT, MAZE_MODEL_SLOTS, MAZE_WIDTH, cloneMazeConfig, defaultMazeConfig, mazePresetConfig, mazeX, mazeY, validateMazeConfig, type MazeConfig } from './maze-scenario';
import { bestMazeAction as winner } from './maze-motion';
import { fmt, formatCount, formatFixed, tr, translateMessage } from './i18n';

const ROUNDS = 4;
const ACTIONS = [
  { name: 'Wait', color: '#9bb8ca' },
  { name: 'Up', color: '#4fe1f6' },
  { name: 'Down', color: '#ffa95a' },
  { name: 'Left', color: '#ad9eff' },
  { name: 'Right', color: '#66e5aa' },
] as const;
const BASE_ROUND_MS = 1300;
const LIFELONG_NOTE = 'Votes update the intent scores. After the fourth round, the planner uses the scores to choose collision-free moves.';

function formatIntentScore(value: number): string {
  return `${value > 0 ? '+' : ''}${formatFixed(value, 2)}`;
}

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
  votes: number[];
  plannedActions: number[];
  finished: number[];
  solved: boolean;
  inferenceMs: number;
  backend: string;
};
type WorkerMessage = (Snapshot | { type: 'status'; text: string } | { type: 'error'; text: string }) & { revision: number };

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
const voteGrid = element('#maze-votes');
const votesSection = element<HTMLElement>('.maze-votes-section');
const decision = element<HTMLElement>('#maze-decision');
const chosenAction = element('#maze-chosen-action');
const neighbors = element('#maze-neighbors');
const allAgents = element('#maze-all-agents');
const scrub = element<HTMLInputElement>('#maze-scrub');
const speedInput = element<HTMLInputElement>('#maze-speed');
const playButton = element<HTMLButtonElement>('#maze-play');
const nextButton = element<HTMLButtonElement>('#maze-next');
const previousButton = element<HTMLButtonElement>('#maze-prev');
const editor = element<HTMLElement>('#maze-editor');
const analysis = element<HTMLElement>('.maze-analysis');
const editorAgents = element('#maze-editor-agents');
const editorMessage = element('#maze-editor-message');
const editorLifelong = element<HTMLInputElement>('#maze-lifelong');
const solvedBanner = element('#maze-solved');
const presetSelect = element<HTMLSelectElement>('#maze-preset');
const customPresetOption = element<HTMLOptionElement>('#maze-preset option[value="custom"]');
const modelNote = element('#maze-model-note');

let worker: Worker | null = null;
let revision = 0;
let config = defaultMazeConfig();
let draft: MazeConfig | null = null;
let editorTool: 'wall' | 'erase' | 'agent' | 'start' | 'goal' = 'wall';
let frames: Snapshot[] = [];
let cursor = 0;
let selectedAgent = 0;
let playing = true;
let mazeActive = false;
let busy = true;
let speed = 1;
let timer: number | undefined;
let lastChangedIntents = 0;
const agentNodes: HTMLButtonElement[] = [];
const goalNodes: HTMLDivElement[] = [];
const tileNodes: HTMLDivElement[] = [];
let sourceEnglish = 'Loading model';
let sourceValues: Record<string, string | number> = {};
let editorMessageEnglish = '';
let editorMessageValues: Record<string, string | number> = {};

function actionName(index: number): string { return tr(ACTIONS[index].name); }
function setSource(english: string, values: Record<string, string | number> = {}): void {
  sourceEnglish = english;
  sourceValues = values;
  const description = Object.keys(values).length ? fmt(english, values) : translateMessage(english);
  source.title = description;
  source.querySelector('span')!.textContent = description;
}
function setEditorMessage(english: string, values: Record<string, string | number> = {}): void {
  editorMessageEnglish = english;
  editorMessageValues = values;
  editorMessage.textContent = Object.keys(values).length ? fmt(english, values) : translateMessage(english);
}
function atCell(node: HTMLElement, cell: number, inset: number): void {
  node.style.left = `${(mazeX(cell) + inset) / MAZE_WIDTH * 100}%`;
  node.style.top = `${(mazeY(cell) + inset) / MAZE_HEIGHT * 100}%`;
}

function buildBoard(): void {
  for (let y = 0; y < MAZE_HEIGHT; y++) {
    for (let x = 0; x < MAZE_WIDTH; x++) {
      const tile = document.createElement('div');
      tile.className = 'maze-tile floor';
      tile.style.left = `${x / MAZE_WIDTH * 100}%`;
      tile.style.top = `${y / MAZE_HEIGHT * 100}%`;
      board.insertBefore(tile, links);
      tileNodes.push(tile);
    }
  }
  for (let i = 0; i < MAZE_MODEL_SLOTS; i++) {
    const goal = document.createElement('div');
    goal.className = 'maze-goal';
    goal.textContent = String(i + 1);
    goal.style.setProperty('--agent-color', MAZE_COLORS[i]);
    board.append(goal);
    goalNodes.push(goal);

    const agent = document.createElement('button');
    agent.type = 'button';
    agent.className = 'maze-agent';
    agent.setAttribute('aria-label', fmt('Inspect agent {id}', { id: String(i + 1).padStart(2, '0') }));
    agent.style.setProperty('--agent-color', MAZE_COLORS[i]);
    agent.innerHTML = `<span class="arrow">${actionIcon(0)}</span><span class="agent-number">${i + 1}</span>`;
    agent.addEventListener('click', () => { if (!draft) { selectedAgent = i; render(); } });
    board.append(agent);
    agentNodes.push(agent);
  }
}

function paintWalls(walls: readonly number[]): void {
  const occupied = new Set(walls);
  for (let cell = 0; cell < tileNodes.length; cell++) {
    tileNodes[cell].className = `maze-tile ${occupied.has(cell) ? 'wall' : 'floor'}`;
  }
}

function renderEditor(): void {
  if (!draft) return;
  board.classList.add('ready', 'editing');
  paintWalls(draft.walls);
  links.innerHTML = '';
  selectedAgent = Math.min(selectedAgent, draft.agents.length - 1);
  for (let i = 0; i < MAZE_MODEL_SLOTS; i++) {
    const agent = draft.agents[i];
    agentNodes[i].hidden = !agent;
    goalNodes[i].hidden = !agent || agent.goal === null;
    if (!agent) continue;
    atCell(agentNodes[i], agent.start, .13);
    agentNodes[i].querySelector('.arrow')!.innerHTML = actionIcon(0);
    agentNodes[i].title = '';
    agentNodes[i].classList.toggle('selected', i === selectedAgent);
    if (agent.goal !== null) atCell(goalNodes[i], agent.goal, .21);
  }
  editorLifelong.checked = draft.lifelong;
  element('#maze-editor-count').textContent = `${draft.agents.length} / ${MAZE_MODEL_SLOTS}`;
  editorAgents.innerHTML = draft.agents.map((agent, index) => `<button type="button" data-agent="${index}" class="${index === selectedAgent ? 'selected' : ''}" style="--agent-color:${MAZE_COLORS[index]}">${fmt('Agent {id}', { id: String(index + 1).padStart(2, '0') })}<span>(${mazeX(agent.start)}, ${mazeY(agent.start)}) → ${agent.goal === null ? tr('Set goal') : `(${mazeX(agent.goal)}, ${mazeY(agent.goal)})`}</span></button>`).join('');
  for (const button of editorAgents.querySelectorAll<HTMLButtonElement>('button')) {
    button.addEventListener('click', () => { selectedAgent = Number(button.dataset.agent); renderEditor(); });
  }
  for (const button of editor.querySelectorAll<HTMLButtonElement>('[data-tool]')) {
    button.classList.toggle('active', button.dataset.tool === editorTool);
  }
}

function openEditor(): void {
  setPlaying(false);
  solvedBanner.hidden = true;
  draft = cloneMazeConfig(config);
  presetSelect.disabled = true;
  editorTool = 'wall';
  setEditorMessage('');
  analysis.hidden = true;
  editor.hidden = false;
  renderEditor();
}

function closeEditor(): void {
  draft = null;
  presetSelect.disabled = false;
  editor.hidden = true;
  analysis.hidden = false;
  board.classList.remove('editing');
  paintWalls(config.walls);
  if (frames.length) render();
}

function editorCell(cell: number, dragging = false): void {
  if (!draft) return;
  const x = mazeX(cell);
  const y = mazeY(cell);
  if (x === 0 || y === 0 || x === MAZE_WIDTH - 1 || y === MAZE_HEIGHT - 1) {
    setEditorMessage('The outer border stays closed.');
    return;
  }
  const wall = draft.walls.includes(cell);
  const agentIndex = draft.agents.findIndex((agent) => agent.start === cell);
  const goalIndex = draft.agents.findIndex((agent) => agent.goal === cell);
  if (draft.agents.length === 0 && (editorTool === 'start' || editorTool === 'goal')) {
    setEditorMessage('Add an agent first.');
    return;
  }
  setEditorMessage('');
  if (editorTool === 'wall') {
    if (agentIndex >= 0 || goalIndex >= 0) setEditorMessage('Move the agent or goal before placing a wall here.');
    else if (!wall) draft.walls.push(cell);
  } else if (editorTool === 'erase') {
    if (wall) draft.walls = draft.walls.filter((item) => item !== cell);
    else if (agentIndex >= 0 && !dragging) {
      draft.agents.splice(agentIndex, 1);
      selectedAgent = Math.max(0, Math.min(selectedAgent, draft.agents.length - 1));
    } else if (goalIndex >= 0 && !dragging) draft.agents[goalIndex].goal = null;
  } else if (!dragging && editorTool === 'agent') {
    if (draft.agents.length >= MAZE_MODEL_SLOTS) setEditorMessage('The model supports up to {count} agents here.', { count: MAZE_MODEL_SLOTS });
    else if (wall || agentIndex >= 0) setEditorMessage('Choose a free floor cell for the new agent.');
    else {
      draft.agents.push({ start: cell, goal: null });
      selectedAgent = draft.agents.length - 1;
      editorTool = 'goal';
      setEditorMessage("Now set agent {id}'s goal.", { id: selectedAgent + 1 });
    }
  } else if (!dragging && editorTool === 'start') {
    if (wall || agentIndex >= 0) setEditorMessage('Choose a free floor cell for the start.');
    else draft.agents[selectedAgent].start = cell;
  } else if (!dragging && editorTool === 'goal') {
    if (wall || (goalIndex >= 0 && goalIndex !== selectedAgent)) setEditorMessage('Choose a floor cell without another goal.');
    else draft.agents[selectedAgent].goal = cell;
  }
  renderEditor();
}

function cellFromPointer(event: PointerEvent): number | null {
  const rect = board.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * MAZE_WIDTH);
  const y = Math.floor((event.clientY - rect.top) / rect.height * MAZE_HEIGHT);
  return x >= 0 && x < MAZE_WIDTH && y >= 0 && y < MAZE_HEIGHT ? y * MAZE_WIDTH + x : null;
}

function frameAt(step: number): { frame: Snapshot; round: number } {
  return { frame: frames[Math.floor(step / (ROUNDS + 1))], round: step % (ROUNDS + 1) };
}

function zValues(frame: Snapshot, round: number, agent: number): number[] {
  if (round === 0) return [0, 0, 0, 0, 0];
  const start = ((round - 1) * frame.positions.length + agent) * ACTIONS.length;
  return frame.trace.slice(start, start + ACTIONS.length);
}

function lastRecordedStep(): number {
  return (frames.length - 1) * (ROUNDS + 1) + (frames[frames.length - 1].solved ? 0 : ROUNDS);
}

function setPlaying(value: boolean): void {
  playing = value;
  playButton.innerHTML = value ? `Ⅱ <span>${tr('Pause')}</span>` : `▶ <span>${tr('Play')}</span>`;
  schedule();
}

function schedule(): void {
  if (timer !== undefined) window.clearTimeout(timer);
  timer = undefined;
  if (!mazeActive || !playing || busy || draft || frames.length === 0) return;
  if (frames[frames.length - 1].solved && cursor === lastRecordedStep()) return;
  timer = window.setTimeout(() => { advanceRound(); }, BASE_ROUND_MS / speed);
}

function advanceRound(): void {
  if (!mazeActive || busy || draft || frames.length === 0) return;
  const lastRecorded = lastRecordedStep();
  if (cursor < lastRecorded) {
    cursor++;
    if (cursor === lastRecorded && frames[frames.length - 1].solved) setPlaying(false);
    render();
    schedule();
    return;
  }
  if (frames[frames.length - 1]?.solved) return;
  busy = true;
  nextButton.disabled = true;
  worker?.postMessage({ type: 'advance', revision });
}

function seek(step: number): void {
  if (frames.length === 0) return;
  setPlaying(false);
  cursor = Math.max(0, Math.min(lastRecordedStep(), step));
  render();
}

function renderLinks(frame: Snapshot, round: number): void {
  const pairs = new Set<string>();
  let content = '';
  for (let receiver = 0; receiver < frame.positions.length; receiver++) {
    for (let slot = 0; slot < 13; slot++) {
      const sender = frame.chat[receiver * 13 + slot];
      if (sender < 0 || sender >= frame.positions.length || sender === receiver) continue;
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

function renderVotes(frame: Snapshot, round: number): void {
  const finalScores = round === ROUNDS ? zValues(frame, round, selectedAgent) : null;
  const agentVotes = Array.from({ length: ROUNDS }, (_, r) => frame.votes[r * frame.positions.length + selectedAgent]);
  const headers = agentVotes.map((_, r) => {
    const number = r + 1;
    const label = fmt('Round {round}', { round: number });
    return `<button class="maze-vote-heading" type="button" data-round="${number}" aria-label="${label}"${number === round ? ' aria-current="step"' : ''}>R${number}</button>`;
  }).join('');
  const rows = ACTIONS.map((action, index) => {
    const name = actionName(index);
    const votes = agentVotes.map((vote, r) => {
      const number = r + 1;
      const voted = vote === index;
      const label = voted ? `aria-label="${fmt('Round {round} vote: {action}', { round: number, action: name })}"` : 'aria-hidden="true"';
      return `<div class="maze-vote-cell ${voted ? 'voted' : ''} ${number === round ? 'current' : ''} ${number > round ? 'future' : ''}" style="--vote-color:${action.color}" ${label}>${voted ? actionIcon(index) : ''}</div>`;
    }).join('');
    const score = finalScores ? formatIntentScore(finalScores[index]) : '—';
    const scoreLabel = finalScores ? `aria-label="${fmt('Final intent score for {action}, {score}', { action: name, score })}"` : 'aria-hidden="true"';
    return `<div class="maze-vote-label" style="--vote-color:${action.color}">${actionIcon(index)} ${name}</div>${votes}<div class="maze-intent-score${finalScores ? '' : ' pending'}" ${scoreLabel}>${score}</div>`;
  }).join('');
  voteGrid.innerHTML = `<div></div>${headers}<div class="maze-vote-heading" title="${tr('Final intent score')}">${tr('Final score')}</div>${rows}`;

  const chosen = finalScores && !frame.finished[selectedAgent] ? frame.plannedActions[selectedAgent] : undefined;
  decision.hidden = chosen === undefined;
  chosenAction.innerHTML = chosen === undefined ? '' : `${actionIcon(chosen)} ${actionName(chosen)}`;
}

function renderInspector(frame: Snapshot, round: number): void {
  agentTitle.textContent = fmt('Agent {id}', { id: String(selectedAgent + 1).padStart(2, '0') });
  agentTitle.style.color = MAZE_COLORS[selectedAgent];
  const goal = frame.goals[selectedAgent];
  const goalPosition = { x: mazeX(goal), y: mazeY(goal) };
  agentSummary.textContent = frame.finished[selectedAgent]
    ? fmt('Goal ({x}, {y}) reached.', goalPosition)
    : fmt('Goal ({x}, {y}).', goalPosition);
  votesSection.hidden = frame.solved;
  if (!frame.solved) renderVotes(frame, round);

  const sources = new Set<number>();
  for (let slot = 0; slot < 13; slot++) {
    const id = frame.chat[selectedAgent * 13 + slot];
    if (id >= 0 && id < frame.positions.length && id !== selectedAgent) sources.add(id);
  }
  element('#maze-neighbor-count').textContent = formatCount(sources.size, 'agents');
  neighbors.innerHTML = sources.size
    ? [...sources].map((id) => `<span style="--agent-color:${MAZE_COLORS[id]}">${fmt('Agent {id}', { id: String(id + 1).padStart(2, '0') })}</span>`).join('')
    : `<span>${tr('No agents in range')}</span>`;
  allAgents.innerHTML = Array.from({ length: frame.positions.length }, (_, id) => {
    const values = zValues(frame, round, id);
    const action = round === 0 ? 0 : winner(values);
    return `<button type="button" class="${id === selectedAgent ? 'selected' : ''}" data-agent="${id}" style="--agent-color:${MAZE_COLORS[id]}" aria-label="${fmt('Agent {id}: {state}', { id: String(id + 1).padStart(2, '0'), state: round === 0 ? tr('before communication') : actionName(action) })}"><span>${String(id + 1).padStart(2, '0')}</span><b>${actionIcon(action)}</b></button>`;
  }).join('');
  for (const button of allAgents.querySelectorAll<HTMLButtonElement>('button')) {
    button.addEventListener('click', () => { selectedAgent = Number(button.dataset.agent); render(); });
  }
}

function renderRoundCopy(frame: Snapshot, round: number, changes: number): void {
  element('#maze-timeline-label').textContent = frame.solved ? tr('All goals reached')
    : round === 0 ? tr('Before communication') : fmt('Communication round {round} of {total}', { round, total: ROUNDS });
  element('#maze-step-label').textContent = fmt('Step {step}', { step: cursor });
  element('#maze-changed').textContent = frame.solved ? '' : round === 0
    ? tr('Intents begin in the next round')
    : round === 1
      ? formatCount(frame.positions.length, 'intentsFormed')
      : formatCount(changes, 'intentsChanged');
}

function render(): void {
  if (frames.length === 0 || draft) return;
  board.classList.add('ready');
  const { frame, round } = frameAt(cursor);
  selectedAgent = Math.min(selectedAgent, frame.positions.length - 1);
  paintWalls(config.walls);
  solvedBanner.hidden = !frame.solved;
  element('#maze-action').textContent = String(frame.action + 1).padStart(3, '0');
  element('#maze-round').textContent = `${round} / ${ROUNDS}`;
  scrub.max = String(lastRecordedStep());
  scrub.value = String(cursor);
  previousButton.disabled = cursor === 0;
  nextButton.disabled = busy || (cursor === Number(scrub.max) && frame.solved);
  playButton.disabled = frame.solved && cursor === Number(scrub.max);

  let changes = 0;
  for (let i = 0; i < MAZE_MODEL_SLOTS; i++) {
    const agent = agentNodes[i];
    const goalNode = goalNodes[i];
    agent.hidden = i >= frame.positions.length;
    goalNode.hidden = i >= frame.positions.length;
    if (i >= frame.positions.length) continue;
    const current = zValues(frame, round, i);
    const currentAction = winner(current);
    const priorAction = round > 1 ? winner(zValues(frame, round - 1, i)) : -1;
    const changed = round > 1 && currentAction !== priorAction;
    if (changed) changes++;
    atCell(agent, frame.positions[i], .13);
    atCell(goalNode, frame.goals[i], .21);
    agent.classList.toggle('selected', i === selectedAgent);
    agent.classList.toggle('finished', frame.finished[i] === 1);
    agent.classList.remove('changed');
    if (changed) { void agent.offsetWidth; agent.classList.add('changed'); }
    agent.querySelector('.arrow')!.innerHTML = actionIcon(frame.finished[i] === 1 || round === 0 ? 0 : currentAction);
    agent.title = frame.finished[i] === 1
      ? fmt('Agent {id}: goal reached', { id: i + 1 })
      : round === 0
        ? fmt('Agent {id}: awaiting communication', { id: i + 1 })
        : fmt('Agent {id}, intent {action} ({value})', {
          id: i + 1, action: actionName(currentAction).toLowerCase(), value: formatIntentScore(current[currentAction]),
        });
  }
  lastChangedIntents = changes;
  renderRoundCopy(frame, round, changes);
  renderLinks(frame, round);
  renderInspector(frame, round);
}

function renderModelNote(): void {
  modelNote.textContent = tr(config.lifelong ? LIFELONG_NOTE
    : 'One-shot without PIBT. Agents follow the model’s decisions and stop at their goals.');
}

function startScenario(): void {
  frames = [];
  cursor = 0;
  busy = true;
  board.classList.remove('ready');
  links.innerHTML = '';
  paintWalls(config.walls);
  solvedBanner.hidden = true;
  agentSummary.textContent = tr('Loading the selected maze scenario…');
  voteGrid.innerHTML = '';
  votesSection.hidden = false;
  decision.hidden = true;
  allAgents.innerHTML = '';
  renderModelNote();
  if (source.classList.contains('error') || !worker) setSource('Loading model');
  source.classList.remove('error');
  nextButton.disabled = true;
  previousButton.disabled = true;
  playButton.disabled = false;
  scrub.max = '0';
  scrub.value = '0';
  element('#maze-action').textContent = '001';
  element('#maze-round').textContent = `0 / ${ROUNDS}`;
  element('#maze-timeline-label').textContent = tr('Before communication');
  element('#maze-step-label').textContent = fmt('Step {step}', { step: 0 });
  element('#maze-changed').textContent = '';
  setPlaying(true);
  worker ??= createWorker();
  worker.postMessage({ type: 'configure', config, revision: ++revision });
}

function createWorker(): Worker {
  const instance = new Worker(new URL('./maze.worker.ts', import.meta.url), { type: 'module' });
  instance.onmessage = (event: MessageEvent<WorkerMessage>) => {
    if (worker !== instance) return;
    const message = event.data;
    if (message.revision !== revision) return;
    if (message.type === 'status') {
      setSource(message.text);
    } else if (message.type === 'error') {
      busy = false;
      source.classList.add('error');
      setSource(message.text);
      setPlaying(false);
      console.error(message.text);
    } else {
      frames.push(message);
      cursor = (frames.length - 1) * (ROUNDS + 1);
      busy = false;
      source.classList.add('ready');
      setSource('DMM-0.8M on {backend}', { backend: message.backend });
      if (message.solved) setPlaying(false);
      render();
      schedule();
    }
  };
  instance.onerror = (error) => {
    if (worker !== instance) return;
    instance.terminate();
    worker = null;
    source.classList.add('error');
    setSource('Maze worker failed');
    setPlaying(false);
    console.error(error);
  };
  return instance;
}

buildBoard();
voteGrid.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest<HTMLButtonElement>('button[data-round]') : null;
  if (!button) return;
  const round = Number(button.dataset.round);
  seek(frameAt(cursor).frame.action * (ROUNDS + 1) + round);
  voteGrid.querySelector<HTMLButtonElement>(`button[data-round="${round}"]`)?.focus({ preventScroll: true });
});
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
element<HTMLButtonElement>('#maze-reset').addEventListener('click', () => { if (!draft) startScenario(); });
presetSelect.addEventListener('change', () => {
  const next = mazePresetConfig(presetSelect.value);
  if (draft || !next) return;
  customPresetOption.hidden = true;
  config = next;
  selectedAgent = 0;
  startScenario();
});
const editButton = element<HTMLButtonElement>('#maze-edit');
editButton.addEventListener('click', () => { openEditor(); editButton.disabled = true; });
for (const button of editor.querySelectorAll<HTMLButtonElement>('[data-tool]')) {
  button.addEventListener('click', () => { editorTool = button.dataset.tool as typeof editorTool; renderEditor(); });
}
editorLifelong.addEventListener('change', () => { if (draft) { draft.lifelong = editorLifelong.checked; renderEditor(); } });
element<HTMLButtonElement>('#maze-editor-remove').addEventListener('click', () => {
  if (!draft || draft.agents.length === 0) return;
  draft.agents.splice(selectedAgent, 1);
  selectedAgent = Math.max(0, Math.min(selectedAgent, draft.agents.length - 1));
  renderEditor();
});
element<HTMLButtonElement>('#maze-editor-default').addEventListener('click', () => {
  draft = defaultMazeConfig();
  selectedAgent = 0;
  setEditorMessage('Default maze restored. Run the scenario to apply it.');
  renderEditor();
});
element<HTMLButtonElement>('#maze-editor-cancel').addEventListener('click', () => { closeEditor(); editButton.disabled = false; });
element<HTMLButtonElement>('#maze-editor-apply').addEventListener('click', () => {
  if (!draft) return;
  const error = validateMazeConfig(draft);
  if (error) { setEditorMessage(error); return; }
  config = cloneMazeConfig(draft);
  customPresetOption.hidden = false;
  presetSelect.value = 'custom';
  selectedAgent = Math.min(selectedAgent, config.agents.length - 1);
  closeEditor();
  editButton.disabled = false;
  startScenario();
});
let lastPaintedCell = -1;
board.addEventListener('pointerdown', (event) => {
  if (!draft || event.button !== 0) return;
  const cell = cellFromPointer(event);
  if (cell === null) return;
  event.preventDefault();
  lastPaintedCell = cell;
  board.setPointerCapture(event.pointerId);
  editorCell(cell);
});
board.addEventListener('pointermove', (event) => {
  if (!draft || !(event.buttons & 1) || (editorTool !== 'wall' && editorTool !== 'erase')) return;
  const cell = cellFromPointer(event);
  if (cell === null || cell === lastPaintedCell) return;
  lastPaintedCell = cell;
  editorCell(cell, true);
});
board.addEventListener('pointerup', () => { lastPaintedCell = -1; });
board.addEventListener('pointercancel', () => { lastPaintedCell = -1; });
window.addEventListener('demo:languagechange', () => {
  for (let i = 0; i < agentNodes.length; i++) {
    agentNodes[i].setAttribute('aria-label', fmt('Inspect agent {id}', { id: String(i + 1).padStart(2, '0') }));
  }
  playButton.innerHTML = playing ? `Ⅱ <span>${tr('Pause')}</span>` : `▶ <span>${tr('Play')}</span>`;
  setSource(sourceEnglish, sourceValues);
  setEditorMessage(editorMessageEnglish, editorMessageValues);
  renderModelNote();
  if (draft) {
    renderEditor();
    if (frames.length) {
      const { frame, round } = frameAt(cursor);
      renderRoundCopy(frame, round, lastChangedIntents);
    }
  }
  else if (frames.length) render();
  else agentSummary.textContent = tr('Loading the selected maze scenario…');
});
window.addEventListener('keydown', (event) => {
  if (!mazeActive) return;
  if (draft) {
    if (event.key === 'Escape') { closeEditor(); editButton.disabled = false; }
    return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey || (event.target instanceof Element && event.target.closest('a, button, input, select, summary'))) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); setPlaying(false); advanceRound(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); seek(cursor - 1); }
  if (event.key === ' ') { event.preventDefault(); setPlaying(!playing); }
});
onViewActivityChange('maze', (active) => {
  mazeActive = active;
  schedule();
});
startScenario();
