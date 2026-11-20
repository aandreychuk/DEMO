import type { Point } from './config';
import { nextPose, type Agent } from './model';

const PICKUP = '#bb8b51';
const DELIVERY = '#63cba3';
const ROBOT = '#55c6e7';
const CARGO = '#ffc878';

export class WarehouseMap {
  private readonly context: CanvasRenderingContext2D;
  private agents: Array<{ x: number; y: number }> = [];
  private cellSize = 1;
  private grid = { left: 0, top: 0, columns: 0, rows: 0 };

  constructor(private readonly canvas: HTMLCanvasElement) {
    const context = canvas.getContext('2d');
    if (!context) throw new Error('The warehouse canvas is unavailable.');
    this.context = context;
  }

  pick(event: PointerEvent): number {
    const bounds = this.canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    let index = -1;
    let distance = this.cellSize * 0.55;
    this.agents.forEach((agent, id) => {
      const offset = Math.hypot(x - agent.x, y - agent.y);
      if (offset < distance) {
        distance = offset;
        index = id;
      }
    });
    return index;
  }

  pickCell(event: PointerEvent): Point | null {
    const bounds = this.canvas.getBoundingClientRect();
    const { left, top, columns, rows } = this.grid;
    const x = Math.floor((event.clientX - bounds.left - left) / this.cellSize);
    const y = Math.floor((event.clientY - bounds.top - top) / this.cellSize);
    return x >= 0 && x < columns && y >= 0 && y < rows ? { x, y } : null;
  }

  draw(map: string[], agents: Agent[], previous: Agent[], blend: number, selected: number, cursor?: Point | null): void {
    const { context, canvas } = this;
    const columns = map[0].length;
    const rows = map.length;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio, 2);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const size = Math.max(1, Math.min((width - 42) / columns, (height - 38) / rows));
    this.cellSize = size;
    const left = (width - size * columns) / 2;
    const top = (height - size * rows) / 2;
    this.grid = { left, top, columns, rows };
    const centerX = (cell: number) => left + (cell % columns + 0.5) * size;
    const centerY = (cell: number) => top + (Math.floor(cell / columns) + 0.5) * size;

    context.fillStyle = '#101f2b';
    context.fillRect(left, top, columns * size, rows * size);
    context.strokeStyle = '#1b3848';
    context.lineWidth = 0.75;
    context.beginPath();
    for (let x = 0; x <= columns; x++) {
      context.moveTo(left + x * size, top);
      context.lineTo(left + x * size, top + rows * size);
    }
    for (let y = 0; y <= rows; y++) {
      context.moveTo(left, top + y * size);
      context.lineTo(left + columns * size, top + y * size);
    }
    context.stroke();

    context.font = '500 9px Inter, system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = '#86aab9';
    for (let x = 0; x < columns; x += 5) context.fillText(String(x), centerX(x), top - 8);
    for (let y = 0; y < rows; y += 5) context.fillText(String(y), left - 11, centerY(y * columns));

    map.forEach((row, y) => {
      for (let x = 0; x < columns; x++) {
        const tile = row[x];
        const px = left + x * size;
        const py = top + y * size;
        if (tile === '#') {
          context.fillStyle = '#294352';
          context.fillRect(px + 0.5, py + 0.5, size - 1, size - 1);
          context.fillStyle = '#395565';
          context.fillRect(px + size * 0.12, py + size * 0.11, size * 0.76, size * 0.12);
        } else if (tile === 'S' || tile === 'D') {
          const color = tile === 'S' ? PICKUP : DELIVERY;
          context.fillStyle = tile === 'S' ? '#392f27' : '#143b36';
          context.fillRect(px + 1, py + 1, size - 2, size - 2);
          context.strokeStyle = color;
          context.strokeRect(px + size * 0.25, py + size * 0.25, size * 0.5, size * 0.5);
        }
      }
    });

    if (cursor) {
      context.strokeStyle = '#a8f0fb';
      context.lineWidth = 1.5;
      context.strokeRect(left + cursor.x * size + 1, top + cursor.y * size + 1, size - 2, size - 2);
    }

    const focus = agents[selected];
    if (focus) {
      const gx = centerX(focus.target);
      const gy = centerY(focus.target);
      context.strokeStyle = '#f2f7bc';
      context.lineWidth = 1.5;
      context.strokeRect(gx - size * 0.38, gy - size * 0.38, size * 0.76, size * 0.76);
      let pose = focus;
      context.strokeStyle = '#90e9f7';
      context.setLineDash([3, 3]);
      context.beginPath();
      context.moveTo(centerX(pose.cell), centerY(pose.cell));
      for (const action of focus.operation.slice(1)) {
        pose = { ...pose, ...nextPose(pose, action, columns) };
        context.lineTo(centerX(pose.cell), centerY(pose.cell));
      }
      context.stroke();
      context.setLineDash([]);
    }

    this.agents = agents.map((agent, id) => {
      const from = previous[id] ?? agent;
      const x = centerX(from.cell) + (centerX(agent.cell) - centerX(from.cell)) * blend;
      const y = centerY(from.cell) + (centerY(agent.cell) - centerY(from.cell)) * blend;
      const change = (agent.direction - from.direction + 6) % 4 - 2;
      context.save();
      context.translate(x, y);
      context.scale(size, size);
      if (id === selected) {
        context.beginPath();
        context.arc(0, 0, 0.46, 0, Math.PI * 2);
        context.strokeStyle = '#d3fbff';
        context.lineWidth = 0.06;
        context.stroke();
      }
      context.rotate((from.direction + change * blend) * (Math.PI / 2));
      context.beginPath();
      context.roundRect(-0.34, -0.29, 0.68, 0.58, 0.12);
      context.fillStyle = agent.stage === 'Delivery' ? CARGO : ROBOT;
      context.fill();
      context.beginPath();
      context.moveTo(0.23, 0);
      context.lineTo(-0.09, -0.16);
      context.lineTo(-0.09, 0.16);
      context.closePath();
      context.fillStyle = '#0b2634';
      context.fill();
      context.restore();
      return { x, y };
    });
  }
}
