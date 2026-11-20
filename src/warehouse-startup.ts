import { fmt, tr } from './i18n';

const taskLabels = {
  connection: 'Local simulator',
  engine: 'Simulation engine',
  core: 'Simulation core',
  model: 'DMM-0.8M model',
  warehouse: 'Warehouse',
};

export type StartupTask = keyof typeof taskLabels;
type Values = Record<string, string | number>;
type TaskStatus = { state: 'pending' | 'active' | 'done'; detail: string; values?: Values };
export type StartupUpdate = {
  type: 'initialization';
  task: StartupTask;
  detail: string;
  done?: boolean;
  values?: Values;
};

const panel = document.querySelector<HTMLElement>('#warehouse-startup')!;
const title = document.querySelector<HTMLElement>('#warehouse-startup-title')!;
const count = document.querySelector<HTMLElement>('#warehouse-startup-count')!;
const progress = document.querySelector<HTMLProgressElement>('#warehouse-startup-progress')!;
const list = document.querySelector<HTMLUListElement>('#warehouse-startup-tasks')!;
const notice = document.querySelector<HTMLElement>('#warehouse-startup-notice')!;
const tasks = new Map<StartupTask, TaskStatus>();
let state: 'loading' | 'error' = 'loading';
let message = '';

function render(): void {
  const done = [...tasks.values()].filter((task) => task.state === 'done').length;
  panel.dataset.state = state;
  title.textContent = tr(state === 'error' ? 'Could not start the simulation' : 'Starting the simulation');
  count.textContent = fmt('{done} of {total} stages complete', { done, total: tasks.size });
  progress.max = tasks.size;
  progress.value = done;
  notice.hidden = !message;
  notice.textContent = tr(message);
  list.replaceChildren(...[...tasks].map(([task, status]) => {
    const row = document.createElement('li');
    row.dataset.state = status.state;
    const label = document.createElement('span');
    label.textContent = tr(taskLabels[task]);
    const detail = document.createElement('small');
    detail.textContent = fmt(status.detail, status.values ?? {});
    row.append(label, detail);
    return row;
  }));
}

export const warehouseStartup = {
  begin(steps: [StartupTask, ...StartupTask[]], detail: string): void {
    tasks.clear();
    for (const step of steps) tasks.set(step, { state: 'pending', detail: 'Waiting' });
    tasks.set(steps[0], { state: 'active', detail });
    state = 'loading';
    message = '';
    panel.hidden = false;
    render();
  },
  update(task: StartupTask, detail: string, done = false, values?: Values): void {
    if (panel.hidden || state !== 'loading' || !tasks.has(task)) return;
    tasks.set(task, { state: done ? 'done' : 'active', detail, values });
    render();
  },
  finish(): void {
    if (!panel.hidden) panel.hidden = true;
  },
  fail(reason: string): void {
    if (panel.hidden) return;
    state = 'error';
    message = reason;
    render();
  },
};

window.addEventListener('demo:languagechange', () => {
  if (!panel.hidden && tasks.size) render();
});
