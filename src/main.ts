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
createWarehousePallets();
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
  const base = MeshBuilder.CreateCylinder('agent-base', {
    height: 0.12,
    diameter: 0.64,
    tessellation: 16,
  }, scene);
  base.position.y = 0.1;
  const bumper = MeshBuilder.CreateTorus('agent-bumper', {
    diameter: 0.58,
    thickness: 0.055,
    tessellation: 20,
  }, scene);
  bumper.position.y = 0.16;
  const body = MeshBuilder.CreateCylinder('agent-body', {
    height: 0.3,
    diameterTop: 0.47,
    diameterBottom: 0.56,
    tessellation: 16,
  }, scene);
  body.position.y = 0.3;
  const shoulder = MeshBuilder.CreateTorus('agent-shoulder', {
    diameter: 0.44,
    thickness: 0.045,
    tessellation: 20,
  }, scene);
  shoulder.position.y = 0.44;
  const dome = MeshBuilder.CreateSphere('agent-dome', { diameter: 0.34, segments: 10 }, scene);
  dome.scaling.y = 0.48;
  dome.position.y = 0.49;
  const lidar = MeshBuilder.CreateCylinder('agent-lidar', {
    height: 0.1,
    diameter: 0.16,
    tessellation: 12,
  }, scene);
  lidar.position.y = 0.59;
  const sensorPods: Mesh[] = [];
  for (let i = 0; i < 4; i++) {
    const angle = i * Math.PI / 2 + Math.PI / 4;
    const pod = MeshBuilder.CreateSphere(`agent-sensor-${i}`, { diameter: 0.085, segments: 6 }, scene);
    pod.position.set(Math.sin(angle) * 0.2, 0.45, Math.cos(angle) * 0.2);
    sensorPods.push(pod);
  }
  const merged = Mesh.MergeMeshes(
    [base, bumper, body, shoulder, dome, lidar, ...sensorPods],
    true,
    true,
    undefined,
    false,
    true,
  )!;
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
  for (let x = 0; x <= MAP_WIDTH; x++) {
    const px = x * CELL_SIZE - halfW;
    points.push([new Vector3(px, 0.004, -halfD), new Vector3(px, 0.004, halfD)]);
  }
  for (let z = 0; z <= MAP_DEPTH; z++) {
    const pz = z * CELL_SIZE - halfD;
    points.push([new Vector3(-halfW, 0.004, pz), new Vector3(halfW, 0.004, pz)]);
  }
  const grid = MeshBuilder.CreateLineSystem('navigation-grid', { lines: points }, scene);
  grid.color = Color3.FromHexString('#14506b');
  grid.alpha = 0.24;
}

function createWarehousePallets(): void {
  const footprint = CELL_SIZE * 0.88;
  const half = footprint / 2;
  const legOffset = half - 0.09;
  const frameParts: Mesh[] = [];
  for (const x of [-legOffset, legOffset]) {
    for (const z of [-legOffset, legOffset]) {
      const leg = MeshBuilder.CreateBox('pallet-leg', { width: 0.1, height: 0.82, depth: 0.1 }, scene);
      leg.position.set(x, 0.43, z);
      frameParts.push(leg);
      const foot = MeshBuilder.CreateBox('pallet-foot', { width: 0.17, height: 0.08, depth: 0.17 }, scene);
      foot.position.set(x, 0.04, z);
      frameParts.push(foot);
    }
  }
  for (const z of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox('pallet-rail-x', { width: footprint, height: 0.1, depth: 0.1 }, scene);
    rail.position.set(0, 0.84, z);
    frameParts.push(rail);
  }
  for (const x of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox('pallet-rail-z', { width: 0.1, height: 0.1, depth: footprint }, scene);
    rail.position.set(x, 0.84, 0);
    frameParts.push(rail);
  }
  const frame = Mesh.MergeMeshes(frameParts, true, true, undefined, false, true)!;
  frame.name = 'pallet-frames';
  const frameMaterial = new StandardMaterial('pallet-frame-material', scene);
  frameMaterial.diffuseColor = Color3.FromHexString('#17313e');
  frameMaterial.emissiveColor = Color3.FromHexString('#07151e');
  frameMaterial.specularColor = Color3.FromHexString('#315d70');
  frame.material = frameMaterial;

  const deckParts: Mesh[] = [];
  for (const z of [-0.27, -0.09, 0.09, 0.27]) {
    const slat = MeshBuilder.CreateBox('pallet-slat', {
      width: footprint,
      height: 0.09,
      depth: 0.12,
    }, scene);
    slat.position.set(0, 0.94, z);
    deckParts.push(slat);
  }
  const deck = Mesh.MergeMeshes(deckParts, true, true, undefined, false, true)!;
  deck.name = 'pallet-decks';
  const deckMaterial = new StandardMaterial('pallet-deck-material', scene);
  deckMaterial.diffuseColor = Color3.FromHexString('#2d8098');
  deckMaterial.emissiveColor = Color3.FromHexString('#0b3442');
  deckMaterial.specularColor = Color3.FromHexString('#7bdcf2');
  deckMaterial.specularPower = 64;
  deck.material = deckMaterial;

  const crateParts: Mesh[] = [];
  const crateBody = MeshBuilder.CreateBox('cargo-crate-body', { width: 0.56, height: 0.48, depth: 0.56 }, scene);
  crateBody.position.y = 1.25;
  crateParts.push(crateBody);
  for (const x of [-0.255, 0.255]) {
    for (const z of [-0.255, 0.255]) {
      const brace = MeshBuilder.CreateBox('cargo-crate-brace', { width: 0.045, height: 0.52, depth: 0.045 }, scene);
      brace.position.set(x, 1.25, z);
      crateParts.push(brace);
    }
  }
  for (const z of [-0.23, 0.23]) {
    const batten = MeshBuilder.CreateBox('cargo-crate-batten', { width: 0.6, height: 0.045, depth: 0.055 }, scene);
    batten.position.set(0, 1.51, z);
    crateParts.push(batten);
  }
  const crate = Mesh.MergeMeshes(crateParts, true, true, undefined, false, true)!;
  crate.name = 'cargo-crates';
  const crateMaterial = new StandardMaterial('cargo-crate-material', scene);
  crateMaterial.diffuseColor = Color3.FromHexString('#a96832');
  crateMaterial.emissiveColor = Color3.FromHexString('#261307');
  crateMaterial.specularColor = Color3.FromHexString('#d7a064');
  crateMaterial.specularPower = 28;
  crate.material = crateMaterial;

  const cartonParts: Mesh[] = [];
  for (const x of [-0.165, 0.165]) {
    const carton = MeshBuilder.CreateBox('cargo-carton-lower', { width: 0.3, height: 0.32, depth: 0.52 }, scene);
    carton.position.set(x, 1.15, 0);
    cartonParts.push(carton);
  }
  const upperCarton = MeshBuilder.CreateBox('cargo-carton-upper', { width: 0.5, height: 0.28, depth: 0.42 }, scene);
  upperCarton.position.set(0, 1.46, 0);
  cartonParts.push(upperCarton);
  const cartons = Mesh.MergeMeshes(cartonParts, true, true, undefined, false, true)!;
  cartons.name = 'cargo-cartons';
  const cartonMaterial = new StandardMaterial('cargo-carton-material', scene);
  cartonMaterial.diffuseColor = Color3.FromHexString('#b98b5e');
  cartonMaterial.emissiveColor = Color3.FromHexString('#291b10');
  cartonMaterial.specularColor = Color3.FromHexString('#d7bd97');
  cartonMaterial.specularPower = 20;
  cartons.material = cartonMaterial;

  const drumParts: Mesh[] = [];
  for (const x of [-0.15, 0.15]) {
    for (const z of [-0.15, 0.15]) {
      const drum = MeshBuilder.CreateCylinder('cargo-drum', {
        height: 0.48,
        diameter: 0.23,
        tessellation: 12,
      }, scene);
      drum.position.set(x, 1.24, z);
      drumParts.push(drum);
      for (const y of [1.04, 1.44]) {
        const band = MeshBuilder.CreateTorus('cargo-drum-band', {
          diameter: 0.2,
          thickness: 0.025,
          tessellation: 12,
        }, scene);
        band.position.set(x, y, z);
        drumParts.push(band);
      }
    }
  }
  const drums = Mesh.MergeMeshes(drumParts, true, true, undefined, false, true)!;
  drums.name = 'cargo-drums';
  const drumMaterial = new StandardMaterial('cargo-drum-material', scene);
  drumMaterial.diffuseColor = Color3.FromHexString('#296d78');
  drumMaterial.emissiveColor = Color3.FromHexString('#09252d');
  drumMaterial.specularColor = Color3.FromHexString('#70c8d8');
  drumMaterial.specularPower = 52;
  drums.material = drumMaterial;

  const palletTransforms: number[] = [];
  const crateTransforms: number[] = [];
  const cartonTransforms: number[] = [];
  const drumTransforms: number[] = [];
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  for (let gx = 6; gx < MAP_WIDTH - 5; gx += 9) {
    for (let gz = 5; gz < MAP_DEPTH - 4; gz += 8) {
      for (let x = gx; x <= gx + 1; x++) {
        for (let z = gz - 2; z <= gz + 3; z++) {
          const transform = Matrix.Translation(
            (x + 0.5) * CELL_SIZE - halfW,
            0,
            (z + 0.5) * CELL_SIZE - halfD,
          );
          transform.copyToArray(palletTransforms, palletTransforms.length);
          const cargoTransforms = (x * 31 + z * 17) % 3 === 0
            ? crateTransforms
            : (x * 31 + z * 17) % 3 === 1
              ? cartonTransforms
              : drumTransforms;
          transform.copyToArray(cargoTransforms, cargoTransforms.length);
        }
      }
    }
  }
  const palletMatrices = new Float32Array(palletTransforms);
  frame.thinInstanceSetBuffer('matrix', palletMatrices, 16, true);
  deck.thinInstanceSetBuffer('matrix', palletMatrices, 16, true);
  crate.thinInstanceSetBuffer('matrix', new Float32Array(crateTransforms), 16, true);
  cartons.thinInstanceSetBuffer('matrix', new Float32Array(cartonTransforms), 16, true);
  drums.thinInstanceSetBuffer('matrix', new Float32Array(drumTransforms), 16, true);
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

function writeMatrix(index: number, gridX: number, gridZ: number): void {
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  const offset = index * 16;
  matrices[offset] = 1;
  matrices[offset + 1] = 0;
  matrices[offset + 2] = 0;
  matrices[offset + 3] = 0;
  matrices[offset + 4] = 0;
  matrices[offset + 5] = 1;
  matrices[offset + 6] = 0;
  matrices[offset + 7] = 0;
  matrices[offset + 8] = 0;
  matrices[offset + 9] = 0;
  matrices[offset + 10] = 1;
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
    writeMatrix(i, cell.x, z);
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
    writeMatrix(i, x, z);
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
