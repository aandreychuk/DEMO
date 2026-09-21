import '@babylonjs/core/Engines/Extensions/engine.query';
import '@babylonjs/core/Meshes/thinInstanceMesh';
import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';
import { Engine } from '@babylonjs/core/Engines/engine';
import { GlowLayer } from '@babylonjs/core/Layers/glowLayer';
import { DirectionalLight } from '@babylonjs/core/Lights/directionalLight';
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight';
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial';
import { Color3, Color4 } from '@babylonjs/core/Maths/math.color';
import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import { Scene } from '@babylonjs/core/scene';
import './style.css';

const MAP_WIDTH = 90;
const MAP_DEPTH = 70;
const CELL_SIZE = 0.82;
const MAX_AGENTS = 2500;
const FALLBACK_STEP_SECONDS = 0.72;
const FRAME_MAGIC = 0x4d415046;
const WS_URL = import.meta.env.VITE_MAPF_WS_URL ?? 'ws://127.0.0.1:18765';

type LiveFrame = {
  positions: Float32Array;
  statuses: Uint8Array;
  step: number;
  receivedAt: number;
};

const canvas = document.querySelector<HTMLCanvasElement>('#scene')!;
const engine = new Engine(canvas, true, { preserveDrawingBuffer: false, stencil: false }, true);
const scene = new Scene(engine);
scene.clearColor = new Color4(0.018, 0.035, 0.055, 1);
scene.ambientColor = new Color3(0.09, 0.15, 0.2);

const camera = new ArcRotateCamera('camera', -Math.PI / 4, 1.02, 58, new Vector3(0, 0, 0), scene);
camera.attachControl(canvas, true);
camera.lowerRadiusLimit = 18;
camera.upperRadiusLimit = 95;
camera.lowerBetaLimit = 0.28;
camera.upperBetaLimit = 1.38;
camera.wheelPrecision = 28;
camera.panningSensibility = 72;
camera.inertia = 0.82;
camera.fov = 0.68;

new HemisphericLight('ambient', new Vector3(0, 1, 0), scene).intensity = 0.74;
const keyLight = new DirectionalLight('key', new Vector3(-0.45, -1, 0.32), scene);
keyLight.position = new Vector3(20, 38, -18);
keyLight.intensity = 1.9;

const floor = MeshBuilder.CreateGround('operations-floor', {
  width: MAP_WIDTH * CELL_SIZE + 7,
  height: MAP_DEPTH * CELL_SIZE + 7,
}, scene);
const floorMaterial = new StandardMaterial('floor-material', scene);
floorMaterial.diffuseColor = Color3.FromHexString('#071521');
floorMaterial.specularColor = Color3.FromHexString('#1c4658');
floorMaterial.specularPower = 48;
floor.material = floorMaterial;
floor.position.y = -0.05;

createGrid();
createWarehouseRacks();
createBoundaryLights();

const robotMesh = createRobotMesh();
const matrices = new Float32Array(MAX_AGENTS * 16);
const colors = new Float32Array(MAX_AGENTS * 4);
const cells = buildFreeCells();
let agentCount = 1000;
let paused = false;
let speed = 1;
let simTime = 0;
let socket: WebSocket | null = null;
let live = false;
let liveTickRate = 20;
let livePrevious: LiveFrame | null = null;
let liveCurrent: LiveFrame | null = null;
let lastInferenceMs = 0;

for (let i = 0; i < MAX_AGENTS; i++) setAgentColor(i, 0);
robotMesh.thinInstanceSetBuffer('matrix', matrices, 16, false);
robotMesh.thinInstanceSetBuffer('color', colors, 4, false);
robotMesh.thinInstanceCount = agentCount;

function createRobotMesh(): Mesh {
  const body = MeshBuilder.CreateCylinder('agent-template', {
    height: 0.34,
    diameterTop: 0.46,
    diameterBottom: 0.58,
    tessellation: 8,
  }, scene);
  body.position.y = 0.28;
  const dome = MeshBuilder.CreateCylinder('agent-dome', {
    height: 0.14,
    diameterTop: 0.24,
    diameterBottom: 0.42,
    tessellation: 8,
  }, scene);
  dome.position.y = 0.51;
  const nose = MeshBuilder.CreateBox('agent-heading', { width: 0.15, height: 0.08, depth: 0.18 }, scene);
  nose.position.set(0, 0.35, 0.28);
  const merged = Mesh.MergeMeshes([body, dome, nose], true, true, undefined, false, true)!;
  merged.name = 'agents';
  const material = new StandardMaterial('agent-material', scene);
  material.diffuseColor = Color3.White();
  material.emissiveColor = new Color3(0.045, 0.19, 0.27);
  material.specularColor = new Color3(0.75, 0.92, 1);
  material.specularPower = 72;
  merged.material = material;
  merged.alwaysSelectAsActiveMesh = true;
  return merged;
}

function createGrid(): void {
  const points: Vector3[][] = [];
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  for (let x = 0; x <= MAP_WIDTH; x += 2) {
    const px = x * CELL_SIZE - halfW;
    points.push([new Vector3(px, 0.004, -halfD), new Vector3(px, 0.004, halfD)]);
  }
  for (let z = 0; z <= MAP_DEPTH; z += 2) {
    const pz = z * CELL_SIZE - halfD;
    points.push([new Vector3(-halfW, 0.004, pz), new Vector3(halfW, 0.004, pz)]);
  }
  const grid = MeshBuilder.CreateLineSystem('navigation-grid', { lines: points }, scene);
  grid.color = Color3.FromHexString('#14506b');
  grid.alpha = 0.38;
}

function createWarehouseRacks(): void {
  const rack = MeshBuilder.CreateBox('rack-template', { width: 1.6, height: 1.65, depth: 4.8 }, scene);
  const rackMaterial = new StandardMaterial('rack-material', scene);
  rackMaterial.diffuseColor = Color3.FromHexString('#18303d');
  rackMaterial.emissiveColor = Color3.FromHexString('#07151e');
  rackMaterial.specularColor = Color3.FromHexString('#3c6b7c');
  rack.material = rackMaterial;
  const transforms: number[] = [];
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  for (let gx = 6; gx < MAP_WIDTH - 5; gx += 9) {
    for (let gz = 5; gz < MAP_DEPTH - 4; gz += 8) {
      Matrix.Translation((gx + 0.5) * CELL_SIZE - halfW, 0.8, (gz + 0.5) * CELL_SIZE - halfD)
        .copyToArray(transforms, transforms.length);
    }
  }
  rack.thinInstanceSetBuffer('matrix', new Float32Array(transforms), 16, true);
  const cap = MeshBuilder.CreateBox('rack-cap-template', { width: 1.68, height: 0.07, depth: 4.88 }, scene);
  const capMaterial = new StandardMaterial('rack-cap-material', scene);
  capMaterial.emissiveColor = Color3.FromHexString('#24708d');
  capMaterial.diffuseColor = Color3.FromHexString('#2c7995');
  cap.material = capMaterial;
  const capTransforms = new Float32Array(transforms);
  for (let i = 0; i < capTransforms.length; i += 16) capTransforms[i + 13] = 1.68;
  cap.thinInstanceSetBuffer('matrix', capTransforms, 16, true);
}

function createBoundaryLights(): void {
  const marker = MeshBuilder.CreateCylinder('boundary-marker', { height: 0.08, diameter: 0.22, tessellation: 10 }, scene);
  const material = new StandardMaterial('marker-material', scene);
  material.emissiveColor = Color3.FromHexString('#1edcff');
  material.disableLighting = true;
  marker.material = material;
  const transforms: number[] = [];
  const halfW = MAP_WIDTH * CELL_SIZE / 2 + 1.8;
  const halfD = MAP_DEPTH * CELL_SIZE / 2 + 1.8;
  for (let i = 0; i <= 32; i++) {
    const x = -halfW + (i / 32) * halfW * 2;
    Matrix.Translation(x, 0.08, -halfD).copyToArray(transforms, transforms.length);
    Matrix.Translation(x, 0.08, halfD).copyToArray(transforms, transforms.length);
  }
  marker.thinInstanceSetBuffer('matrix', new Float32Array(transforms), 16, true);
  const glow = new GlowLayer('signal-glow', scene, { blurKernelSize: 18 });
  glow.intensity = 0.65;
  glow.addIncludedOnlyMesh(marker);
}

function isRackCell(x: number, z: number): boolean {
  for (let gx = 6; gx < MAP_WIDTH - 5; gx += 9) {
    for (let gz = 5; gz < MAP_DEPTH - 4; gz += 8) {
      if (x >= gx && x <= gx + 1 && z >= gz - 2 && z <= gz + 3) return true;
    }
  }
  return false;
}

function buildFreeCells(): Array<{ x: number; z: number; direction: number }> {
  const result: Array<{ x: number; z: number; direction: number }> = [];
  for (let x = 1; x < MAP_WIDTH - 1; x++) {
    for (let z = 1; z < MAP_DEPTH - 1; z++) {
      if (!isRackCell(x, z)) result.push({ x, z, direction: x % 2 === 0 ? 1 : -1 });
    }
  }
  return result;
}

function writeMatrix(index: number, gridX: number, gridZ: number, angle: number): void {
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const offset = index * 16;
  matrices[offset] = cos;
  matrices[offset + 1] = 0;
  matrices[offset + 2] = -sin;
  matrices[offset + 3] = 0;
  matrices[offset + 4] = 0;
  matrices[offset + 5] = 1;
  matrices[offset + 6] = 0;
  matrices[offset + 7] = 0;
  matrices[offset + 8] = sin;
  matrices[offset + 9] = 0;
  matrices[offset + 10] = cos;
  matrices[offset + 11] = 0;
  matrices[offset + 12] = (gridX + 0.5) * CELL_SIZE - halfW;
  matrices[offset + 13] = 0.02;
  matrices[offset + 14] = (gridZ + 0.5) * CELL_SIZE - halfD;
  matrices[offset + 15] = 1;
}

function updateFallback(time: number): void {
  const step = time / FALLBACK_STEP_SECONDS;
  const wholeStep = Math.floor(step);
  const phase = smoothstep(step - wholeStep);
  for (let i = 0; i < agentCount; i++) {
    const cell = cells[(i * 47) % cells.length];
    const range = MAP_DEPTH - 2;
    const z = 1 + mod(cell.z - 1 + cell.direction * (wholeStep + phase), range);
    writeMatrix(i, cell.x, z, cell.direction > 0 ? 0 : Math.PI);
  }
  robotMesh.thinInstanceBufferUpdated('matrix');
}

function updateLive(now: number): void {
  if (!liveCurrent) return;
  const from = livePrevious ?? liveCurrent;
  const alpha = Math.min(1, (now - liveCurrent.receivedAt) / (1000 / liveTickRate));
  for (let i = 0; i < agentCount; i++) {
    const x0 = from.positions[i * 2];
    const z0 = from.positions[i * 2 + 1];
    const x1 = liveCurrent.positions[i * 2];
    const z1 = liveCurrent.positions[i * 2 + 1];
    const x = x0 + (x1 - x0) * alpha;
    const z = z0 + (z1 - z0) * alpha;
    const dx = x1 - x0;
    const dz = z1 - z0;
    const angle = dx === 0 && dz === 0 ? 0 : Math.atan2(dx, dz);
    writeMatrix(i, x, z, angle);
  }
  robotMesh.thinInstanceBufferUpdated('matrix');
}

function setAgentColor(index: number, status: number): void {
  const color = status === 2
    ? [0.65, 0.38, 1, 1]
    : status === 3
      ? [0.35, 0.95, 0.68, 1]
      : status === 1
        ? [1, 0.62, 0.2, 1]
        : [0.12, 0.78 + (index % 7) * 0.025, 1, 1];
  colors.set(color, index * 4);
}

function parseFrame(buffer: ArrayBuffer): void {
  const view = new DataView(buffer);
  if (view.byteLength < 16 || view.getUint32(0, true) !== FRAME_MAGIC || view.getUint16(4, true) !== 1) return;
  const step = view.getUint32(8, true);
  const count = Math.min(MAX_AGENTS, view.getUint32(12, true));
  if (view.byteLength < 16 + count * 16) return;
  const positions = new Float32Array(count * 2);
  const statuses = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const offset = 16 + i * 16;
    const id = view.getUint32(offset, true);
    if (id >= count) continue;
    positions[id * 2] = view.getFloat32(offset + 4, true);
    positions[id * 2 + 1] = view.getFloat32(offset + 8, true);
    statuses[id] = view.getUint8(offset + 12);
    setAgentColor(id, statuses[id]);
  }
  if (count !== agentCount) setAgentCount(count);
  livePrevious = liveCurrent;
  liveCurrent = { positions, statuses, step, receivedAt: performance.now() };
  robotMesh.thinInstanceBufferUpdated('color');
}

function setConnection(state: string, label: string): void {
  const connection = document.querySelector<HTMLElement>('.connection')!;
  connection.dataset.state = state;
  document.querySelector('#source-label')!.textContent = label;
}

function setAgentCount(count: number): void {
  agentCount = Math.min(MAX_AGENTS, count);
  robotMesh.thinInstanceCount = agentCount;
  agentsValue.textContent = agentCount.toLocaleString('en-US');
  const select = document.querySelector<HTMLSelectElement>('#agent-count')!;
  if ([...select.options].some((option) => Number(option.value) === agentCount)) select.value = String(agentCount);
}

function sendControl(action: string, extra: Record<string, number> = {}): void {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'control', action, ...extra }));
}

function connect(): void {
  setConnection('connecting', 'CONNECTING // LOCAL');
  const ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';
  socket = ws;
  ws.addEventListener('open', () => setConnection('planning', 'FASTDMM // PLANNING'));
  ws.addEventListener('message', (event) => {
    if (event.data instanceof ArrayBuffer) {
      live = true;
      parseFrame(event.data);
      setConnection('live', 'FASTDMM // LOCAL');
      return;
    }
    const message = JSON.parse(String(event.data));
    if (message.type === 'hello') {
      live = true;
      paused = false;
      syncPauseButton();
      liveTickRate = Number(message.tickRate) || 20;
      lastInferenceMs = Number(message.inferenceMs) || 0;
      setAgentCount(Number(message.agents));
      setConnection('live', 'FASTDMM // LOCAL');
    } else if (message.type === 'status' && message.state === 'planning') {
      livePrevious = null;
      liveCurrent = null;
      setConnection('planning', `PLANNING // ${Number(message.agents).toLocaleString('en-US')}`);
    } else if (message.type === 'status' && message.state === 'complete') {
      paused = true;
      syncPauseButton();
      setConnection('complete', 'REPLAY // COMPLETE');
    } else if (message.type === 'error') {
      setConnection('error', 'RUNTIME // ERROR');
      console.error(message.message);
    }
  });
  ws.addEventListener('close', () => {
    if (socket !== ws) return;
    live = false;
    livePrevious = null;
    liveCurrent = null;
    setConnection('fallback', 'DEMO // RECONNECTING');
    window.setTimeout(connect, 2000);
  });
  ws.addEventListener('error', () => ws.close());
}

function smoothstep(t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped * clamped * (3 - 2 * clamped);
}

function mod(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

const fpsValue = document.querySelector('#fps-value')!;
const stepValue = document.querySelector('#step-value')!;
const agentsValue = document.querySelector('#agents-value')!;
const throughputValue = document.querySelector('#throughput-value')!;
const latencyValue = document.querySelector('#latency-value')!;
let telemetryElapsed = 0;

engine.runRenderLoop(() => {
  const dt = Math.min(engine.getDeltaTime() / 1000, 0.05);
  if (!paused && !live) simTime += dt * speed;
  if (live) updateLive(performance.now());
  else updateFallback(simTime);
  scene.render();
  telemetryElapsed += dt;
  if (telemetryElapsed > 0.25) {
    telemetryElapsed = 0;
    fpsValue.textContent = String(Math.round(engine.getFps()));
    const step = liveCurrent?.step ?? Math.floor(simTime / FALLBACK_STEP_SECONDS);
    stepValue.textContent = String(step).padStart(4, '0');
    throughputValue.textContent = Math.round(agentCount * (live ? liveTickRate : 1 / FALLBACK_STEP_SECONDS) * speed).toLocaleString('en-US');
    latencyValue.textContent = live ? `${lastInferenceMs.toFixed(1)} ms` : 'DEMO';
  }
});

const pauseButton = document.querySelector<HTMLButtonElement>('#pause-button')!;
function syncPauseButton(): void {
  document.querySelector('#pause-icon')!.textContent = paused ? '▶' : 'Ⅱ';
  document.querySelector('#pause-label')!.textContent = paused ? 'RUN' : 'PAUSE';
}
pauseButton.addEventListener('click', () => {
  paused = !paused;
  syncPauseButton();
  sendControl(paused ? 'pause' : 'run');
});

document.querySelector<HTMLSelectElement>('#agent-count')!.addEventListener('change', (event) => {
  const count = Number((event.currentTarget as HTMLSelectElement).value);
  if (socket?.readyState === WebSocket.OPEN) {
    paused = false;
    syncPauseButton();
    setConnection('planning', `PLANNING // ${count.toLocaleString('en-US')}`);
    sendControl('load', { agents: count });
  } else {
    setAgentCount(count);
  }
});

document.querySelector<HTMLInputElement>('#speed')!.addEventListener('input', (event) => {
  speed = Number((event.currentTarget as HTMLInputElement).value);
  document.querySelector('#speed-value')!.textContent = `${speed.toFixed(2).replace(/0$/, '')}×`;
  sendControl('speed', { value: speed });
});

document.querySelector('#reset-camera')!.addEventListener('click', () => {
  camera.alpha = -Math.PI / 4;
  camera.beta = 1.02;
  camera.radius = 58;
  camera.target.set(0, 0, 0);
});

window.addEventListener('resize', () => engine.resize());
connect();
