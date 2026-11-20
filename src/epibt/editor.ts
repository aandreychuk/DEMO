import { tr } from '../i18n';
import { MAX_AGENTS, MAX_CONFIG_BYTES, parseWarehouse, prepareWarehouse, type Point, type Tile, type WarehouseConfig } from './config';
import type { WarehouseMap } from './map';
import { WAREHOUSE_PRESETS } from './presets';

function element<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

export class LayoutEditor {
  draft: WarehouseConfig | null = null;
  cursor: Point | null = null;

  private readonly presets = element<HTMLSelectElement>('epibt-preset');
  private readonly count = element<HTMLInputElement>('epibt-robot-count');
  private readonly file = element<HTMLInputElement>('epibt-file');
  private readonly error = element('epibt-file-error');
  private readonly validation = element('epibt-map-check');
  private readonly undoButton = element<HTMLButtonElement>('epibt-undo');
  private readonly applyButton = element<HTMLButtonElement>('epibt-apply');
  private readonly exportButton = element<HTMLButtonElement>('epibt-export');
  private readonly options = WAREHOUSE_PRESETS.map(preset => new Option(preset.name, preset.id));
  private history: WarehouseConfig[] = [];
  private errorText = '';
  private loadVersion = 0;
  private pointer: number | null = null;
  private previousCell: Point | null = null;
  private strokeChanged = false;

  constructor(
    canvas: HTMLCanvasElement,
    map: WarehouseMap,
    private readonly onChange: () => void,
    private readonly onClose: (config?: WarehouseConfig) => void,
  ) {
    this.presets.append(...this.options);
    this.count.max = String(MAX_AGENTS);
    this.presets.addEventListener('change', () => {
      const preset = WAREHOUSE_PRESETS.find(option => option.id === this.presets.value);
      if (preset && this.draft) this.replace(preset.config);
    });
    this.count.addEventListener('input', () => this.updateControls());
    this.count.addEventListener('change', () => {
      const config = this.readDraft();
      if (config && config.agentCount !== this.draft?.agentCount) {
        this.record();
        this.draft = config;
        this.updateControls();
      }
    });
    this.undoButton.addEventListener('click', () => this.undo());
    element('epibt-cancel').addEventListener('click', () => this.close());
    this.applyButton.addEventListener('click', () => {
      const config = this.readDraft();
      if (config && !this.mapProblem(config)) this.close(config);
    });
    element('epibt-import').addEventListener('click', () => this.file.click());
    this.file.addEventListener('change', () => void this.load());
    this.exportButton.addEventListener('click', () => this.download());
    window.addEventListener('demo:languagechange', () => this.localize());
    this.localize();

    canvas.addEventListener('pointerdown', event => {
      if (!this.draft || event.button !== 0 || this.pointer !== null) return;
      const cell = map.pickCell(event);
      if (!cell) return;
      this.pointer = event.pointerId;
      this.previousCell = cell;
      this.strokeChanged = false;
      this.cursor = cell;
      canvas.setPointerCapture(event.pointerId);
      this.paint(cell, cell, true);
    });
    canvas.addEventListener('pointermove', event => {
      if (!this.draft || (this.pointer !== null && event.pointerId !== this.pointer)) return;
      this.cursor = map.pickCell(event);
      if (this.pointer !== null && this.cursor) {
        this.paint(this.previousCell ?? this.cursor, this.cursor, true);
      } else {
        this.onChange();
      }
      this.previousCell = this.cursor;
    });
    canvas.addEventListener('pointerleave', () => {
      if (this.draft && this.pointer === null) {
        this.cursor = null;
        this.onChange();
      }
    });
    canvas.addEventListener('lostpointercapture', event => {
      if (event.pointerId === this.pointer) {
        this.pointer = null;
        this.previousCell = null;
      }
    });
    canvas.addEventListener('keydown', event => {
      if (!this.draft) return;
      const cell = this.cursor ?? { x: 0, y: 0 };
      const offsets: Record<string, Point> = {
        ArrowLeft: { x: -1, y: 0 }, ArrowRight: { x: 1, y: 0 },
        ArrowUp: { x: 0, y: -1 }, ArrowDown: { x: 0, y: 1 },
      };
      const offset = offsets[event.key];
      if (offset) {
        event.preventDefault();
        this.cursor = {
          x: Math.max(0, Math.min(this.draft.map[0].length - 1, cell.x + offset.x)),
          y: Math.max(0, Math.min(this.draft.map.length - 1, cell.y + offset.y)),
        };
        this.onChange();
      } else if (event.key === ' ') {
        event.preventDefault();
        this.cursor = cell;
        this.paint(cell, cell, false);
      } else if (event.key.toLowerCase() === 'z' && (event.ctrlKey || event.metaKey) && !event.shiftKey) {
        event.preventDefault();
        this.undo();
      }
    });
  }

  open(config: WarehouseConfig): void {
    this.history = [];
    this.show(config);
  }

  private close(config?: WarehouseConfig): void {
    this.loadVersion++;
    this.draft = null;
    this.cursor = null;
    this.pointer = null;
    this.previousCell = null;
    this.onClose(config);
  }

  private localize(): void {
    this.options.forEach((option, index) => {
      option.textContent = tr(WAREHOUSE_PRESETS[index].name);
    });
    this.showError(this.errorText);
    this.updateControls();
  }

  private show(config: WarehouseConfig): void {
    this.draft = config;
    this.cursor = null;
    this.count.value = String(config.agentCount);
    this.showError('');
    this.updateControls();
    this.onChange();
  }

  private replace(config: WarehouseConfig): void {
    this.record();
    this.show(config);
  }

  private record(): void {
    if (!this.draft) return;
    this.history.push(this.draft);
    if (this.history.length > 50) this.history.shift();
    this.loadVersion++;
  }

  private undo(): void {
    const previous = this.history.pop();
    if (!previous) return;
    this.loadVersion++;
    this.show(previous);
  }

  private paint(from: Point, to: Point, stroke: boolean): void {
    if (!this.draft) return;
    const brush = document.querySelector<HTMLInputElement>('input[name="epibt-brush"]:checked')!.value as Tile;
    const map = [...this.draft.map];
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const distance = Math.max(1, Math.abs(dx), Math.abs(dy));
    let changed = false;
    for (let step = 0; step <= distance; step++) {
      const x = Math.round(from.x + dx * step / distance);
      const y = Math.round(from.y + dy * step / distance);
      const row = map[y];
      if (row[x] === brush) continue;
      map[y] = row.slice(0, x) + brush + row.slice(x + 1);
      changed = true;
    }
    if (changed) {
      if (!stroke || !this.strokeChanged) this.record();
      this.strokeChanged = stroke;
      this.draft = { ...this.draft, map };
      this.updateControls();
    }
    this.onChange();
  }

  private readDraft(): WarehouseConfig | null {
    return this.draft && this.count.validity.valid
      ? { ...this.draft, agentCount: this.count.valueAsNumber }
      : null;
  }

  private mapProblem(config: WarehouseConfig | null): string {
    if (!config) return 'The warehouse must have 1–200 agents.';
    try {
      prepareWarehouse(config);
      return '';
    } catch (error) {
      return (error as Error).message;
    }
  }

  private updateControls(): void {
    if (!this.draft) return;
    const config = this.readDraft();
    const problem = this.mapProblem(config);
    const preset = config && WAREHOUSE_PRESETS.find(({ config: ready }) =>
      ready.agentCount === config.agentCount && ready.map.length === config.map.length &&
      ready.map.every((row, index) => row === config.map[index])
    );
    this.presets.value = preset?.id ?? '';
    this.undoButton.disabled = this.history.length === 0;
    this.applyButton.disabled = !!problem;
    this.exportButton.disabled = !config;
    this.validation.hidden = !problem;
    const message = tr(problem);
    if (this.validation.textContent !== message) this.validation.textContent = message;
  }

  private showError(message: string): void {
    this.errorText = message;
    this.error.hidden = !message;
    this.error.textContent = tr(message);
  }

  private async load(): Promise<void> {
    const file = this.file.files?.[0];
    this.file.value = '';
    if (!file || !this.draft) return;
    const version = ++this.loadVersion;
    try {
      if (file.size > MAX_CONFIG_BYTES) throw new Error('The JSON file is larger than 1 MB.');
      const text = await file.text();
      if (version === this.loadVersion) this.replace(parseWarehouse(JSON.parse(text)));
    } catch (error) {
      if (version !== this.loadVersion) return;
      this.showError(error instanceof SyntaxError ? 'The file does not contain valid JSON.'
        : error instanceof Error ? error.message : 'Could not read the JSON file.');
    }
  }

  private download(): void {
    const config = this.readDraft();
    if (!config) return;
    const file = new Blob([JSON.stringify(config, null, 2) + '\n'], { type: 'application/json' });
    const url = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'epibt-warehouse.json';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
