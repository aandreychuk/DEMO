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
const MAX_AGENTS = 5000;
const STEP_SECONDS = 0.72;

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

for (let i = 0; i < MAX_AGENTS; i++) {
  const palette = i % 29 === 0
    ? [0.65, 0.38, 1, 1]
    : i % 17 === 0
      ? [1, 0.62, 0.2, 1]
      : [0.12, 0.78 + (i % 7) * 0.025, 1, 1];
  colors.set(palette, i * 4);
}

robotMesh.thinInstanceSetBuffer('matrix', matrices, 16, false);
robotMesh.thinInstanceSetBuffer('color', colors, 4, true);
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
  const lineMaterial = new StandardMaterial('grid-lines', scene);
  lineMaterial.emissiveColor = Color3.FromHexString('#12364a');
  lineMaterial.alpha = 0.5;
  lineMaterial.disableLighting = true;

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
      Matrix.Translation(
        gx * CELL_SIZE - halfW,
        0.8,
        gz * CELL_SIZE - halfD,
      ).copyToArray(transforms, transforms.length);
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

function buildFreeCells(): Array<{ x: number; z: number; direction: number }> {
  const result: Array<{ x: number; z: number; direction: number }> = [];
  for (let x = 1; x < MAP_WIDTH - 1; x++) {
    const rackColumn = x >= 5 && (x - 6) % 9 <= 1;
    if (rackColumn) continue;
    for (let z = 1; z < MAP_DEPTH - 1; z++) {
      result.push({ x, z, direction: x % 2 === 0 ? 1 : -1 });
    }
  }
  return result;
}

function updateAgents(time: number): void {
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  const step = time / STEP_SECONDS;
  const wholeStep = Math.floor(step);
  const phase = smoothstep(step - wholeStep);

  for (let i = 0; i < agentCount; i++) {
    const cell = cells[(i * 47) % cells.length];
    const range = MAP_DEPTH - 2;
    const offset = cell.direction * (wholeStep + phase);
    const z = 1 + mod(cell.z - 1 + offset, range);
    const worldX = cell.x * CELL_SIZE - halfW;
    const worldZ = z * CELL_SIZE - halfD;
    const angle = cell.direction > 0 ? 0 : Math.PI;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const matrixOffset = i * 16;
    matrices[matrixOffset] = cos;
    matrices[matrixOffset + 1] = 0;
    matrices[matrixOffset + 2] = -sin;
    matrices[matrixOffset + 3] = 0;
    matrices[matrixOffset + 4] = 0;
    matrices[matrixOffset + 5] = 1;
    matrices[matrixOffset + 6] = 0;
    matrices[matrixOffset + 7] = 0;
    matrices[matrixOffset + 8] = sin;
    matrices[matrixOffset + 9] = 0;
    matrices[matrixOffset + 10] = cos;
    matrices[matrixOffset + 11] = 0;
    matrices[matrixOffset + 12] = worldX;
    matrices[matrixOffset + 13] = 0.02;
    matrices[matrixOffset + 14] = worldZ;
    matrices[matrixOffset + 15] = 1;
  }
  robotMesh.thinInstanceBufferUpdated('matrix');
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
  if (!paused) simTime += dt * speed;
  updateAgents(simTime);
  scene.render();

  telemetryElapsed += dt;
  if (telemetryElapsed > 0.25) {
    telemetryElapsed = 0;
    const fps = Math.round(engine.getFps());
    fpsValue.textContent = String(fps);
    stepValue.textContent = String(Math.floor(simTime / STEP_SECONDS)).padStart(4, '0');
    throughputValue.textContent = Math.round(agentCount / STEP_SECONDS * speed).toLocaleString('en-US');
    latencyValue.textContent = `${(3.1 + agentCount / 2400).toFixed(1)} ms`;
  }
});

const pauseButton = document.querySelector<HTMLButtonElement>('#pause-button')!;
pauseButton.addEventListener('click', () => {
  paused = !paused;
  document.querySelector('#pause-icon')!.textContent = paused ? '▶' : 'Ⅱ';
  document.querySelector('#pause-label')!.textContent = paused ? 'RUN' : 'PAUSE';
});

document.querySelector<HTMLSelectElement>('#agent-count')!.addEventListener('change', (event) => {
  agentCount = Number((event.currentTarget as HTMLSelectElement).value);
  robotMesh.thinInstanceCount = agentCount;
  agentsValue.textContent = agentCount.toLocaleString('en-US');
});

document.querySelector<HTMLInputElement>('#speed')!.addEventListener('input', (event) => {
  speed = Number((event.currentTarget as HTMLInputElement).value);
  document.querySelector('#speed-value')!.textContent = `${speed.toFixed(2).replace(/0$/, '')}×`;
});

document.querySelector('#reset-camera')!.addEventListener('click', () => {
  camera.alpha = -Math.PI / 4;
  camera.beta = 1.02;
  camera.radius = 58;
  camera.target.set(0, 0, 0);
});

window.addEventListener('resize', () => engine.resize());
