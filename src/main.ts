import '@babylonjs/core/Engines/Extensions/engine.query';
import '@babylonjs/core/Meshes/thinInstanceMesh';
import '@babylonjs/core/Culling/ray';
import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';
import { Engine } from '@babylonjs/core/Engines/engine';
import { PointerEventTypes } from '@babylonjs/core/Events/pointerEvents';
import { GlowLayer } from '@babylonjs/core/Layers/glowLayer';
import { DirectionalLight } from '@babylonjs/core/Lights/directionalLight';
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight';
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial';
import { Color3, Color4 } from '@babylonjs/core/Maths/math.color';
import { Matrix, Quaternion, Vector3 } from '@babylonjs/core/Maths/math.vector';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import { Scene } from '@babylonjs/core/scene';
import './style.css';

const MAP_WIDTH = 44;
const MAP_DEPTH = 32;
const CELL_SIZE = 0.9;
const MAX_AGENTS = 100;
const UNLOAD_X = MAP_WIDTH - 3;
const UNLOAD_MIN_Z = 1;
const UNLOAD_MAX_Z = MAP_DEPTH - 2;
const UNLOAD_STATION_COUNT = UNLOAD_MAX_Z - UNLOAD_MIN_Z + 1;
const FALLBACK_STEP_SECONDS = 0.72;
const FRAME_MAGIC = 0x4d415046;
const WS_URL = import.meta.env.VITE_MAPF_WS_URL ?? 'ws://127.0.0.1:18765';

type LiveFrame = {
  positions: Float32Array;
  statuses: Uint8Array;
  stages: Uint8Array;
  palletIds: Uint16Array;
  taskIds: Uint32Array;
  stationPositions: Int16Array;
  completedTasks: number;
  step: number;
  receivedAt: number;
};

type PalletCell = { id: number; x: number; z: number; cargoType: number };
type CargoTransfer = { gridX: number; gridZ: number; worldY: number; cargoType: number };
type AgentStats = {
  distance: number;
  moveTicks: number;
  waitTicks: number;
  handoffs: number;
  completedTasks: number;
  taskStartedAt: number;
  lastTaskId: number;
};

const canvas = document.querySelector<HTMLCanvasElement>('#scene')!;
const engine = new Engine(canvas, true, { preserveDrawingBuffer: false, stencil: false }, true);
const scene = new Scene(engine);
scene.clearColor = new Color4(0.018, 0.035, 0.055, 1);
scene.ambientColor = new Color3(0.09, 0.15, 0.2);

const camera = new ArcRotateCamera('camera', -Math.PI / 4, 1.02, 32, new Vector3(0, 0, 0), scene);
camera.attachControl(canvas, true);
camera.lowerRadiusLimit = 12;
camera.upperRadiusLimit = 58;
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
const palletCells = buildPalletCells();
const palletScene = createWarehousePallets(palletCells);
const unloadingScene = createUnloadingZone();
createBoundaryLights();

const robotMesh = createRobotMesh();
const robotLoad = createCarriedPalletMeshes();
const agentPicker = createAgentPicker();
const selectionHalo = createSelectionHalo();
const taskMarkers = createTaskMarkers();
const matrices = new Float32Array(MAX_AGENTS * 16);
const loadMatrices = new Float32Array(MAX_AGENTS * 16);
const cargoMatrices = [
  new Float32Array((MAX_AGENTS + UNLOAD_STATION_COUNT) * 16),
  new Float32Array((MAX_AGENTS + UNLOAD_STATION_COUNT) * 16),
  new Float32Array((MAX_AGENTS + UNLOAD_STATION_COUNT) * 16),
];
const loadAgentIds = new Int16Array(MAX_AGENTS);
const cargoAgentIds = [
  new Int16Array(MAX_AGENTS + UNLOAD_STATION_COUNT),
  new Int16Array(MAX_AGENTS + UNLOAD_STATION_COUNT),
  new Int16Array(MAX_AGENTS + UNLOAD_STATION_COUNT),
];
const colors = new Float32Array(MAX_AGENTS * 4);
const cells = buildFreeCells();
let agentCount = 100;
let paused = false;
let speed = 1;
let simTime = 0;
let socket: WebSocket | null = null;
let live = false;
let liveTickRate = 10;
let livePrevious: LiveFrame | null = null;
let liveCurrent: LiveFrame | null = null;
let lastInferenceMs = 0;
let hiddenPalletKey = '';
let selectedAgentId = -1;
const agentStats: AgentStats[] = Array.from({ length: MAX_AGENTS }, () => createAgentStats());
const cameraKeys = new Set<string>();

for (let i = 0; i < MAX_AGENTS; i++) setAgentColor(i, 0);
robotMesh.thinInstanceSetBuffer('matrix', matrices, 16, false);
robotMesh.thinInstanceSetBuffer('color', colors, 4, false);
robotMesh.thinInstanceCount = agentCount;
agentPicker.thinInstanceSetBuffer('matrix', matrices, 16, false);
agentPicker.thinInstanceCount = agentCount;
robotLoad.frame.thinInstanceSetBuffer('matrix', loadMatrices, 16, false);
robotLoad.deck.thinInstanceSetBuffer('matrix', loadMatrices, 16, false);
for (let type = 0; type < robotLoad.cargo.length; type++) {
  robotLoad.cargo[type].thinInstanceSetBuffer('matrix', cargoMatrices[type], 16, false);
  robotLoad.cargo[type].thinInstanceCount = 0;
}
robotLoad.frame.thinInstanceCount = 0;
robotLoad.deck.thinInstanceCount = 0;
robotMesh.thinInstanceEnablePicking = true;
agentPicker.thinInstanceEnablePicking = true;
robotLoad.frame.thinInstanceEnablePicking = true;
robotLoad.deck.thinInstanceEnablePicking = true;
for (const cargo of robotLoad.cargo) cargo.thinInstanceEnablePicking = true;

scene.onPointerObservable.add((pointerInfo) => {
  const pickInfo = pointerInfo.pickInfo;
  if (pointerInfo.type !== PointerEventTypes.POINTERDOWN || !pickInfo) return;
  const instance = pickInfo.thinInstanceIndex;
  let agent = -1;
  if (instance >= 0 && (pickInfo.pickedMesh === robotMesh || pickInfo.pickedMesh === agentPicker)) {
    agent = instance;
  } else if (instance >= 0 && (pickInfo.pickedMesh === robotLoad.frame || pickInfo.pickedMesh === robotLoad.deck)) {
    agent = loadAgentIds[instance] ?? -1;
  } else if (instance >= 0 && pickInfo.pickedMesh) {
    const cargoType = robotLoad.cargo.indexOf(pickInfo.pickedMesh as Mesh);
    if (cargoType >= 0) agent = cargoAgentIds[cargoType][instance] ?? -1;
  }
  if (agent < 0 && pickInfo.pickedPoint) {
    let nearestDistance = (CELL_SIZE * 0.72) ** 2;
    for (let id = 0; id < agentCount; id++) {
      const dx = matrices[id * 16 + 12] - pickInfo.pickedPoint.x;
      const dz = matrices[id * 16 + 14] - pickInfo.pickedPoint.z;
      const distance = dx * dx + dz * dz;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        agent = id;
      }
    }
  }
  if (agent >= 0 && agent < agentCount) selectAgent(agent);
}, PointerEventTypes.POINTERDOWN);

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

function createSelectionHalo(): Mesh {
  const halo = MeshBuilder.CreateTorus('selected-agent-halo', {
    diameter: 0.82,
    thickness: 0.035,
    tessellation: 36,
  }, scene);
  const material = new StandardMaterial('selected-agent-halo-material', scene);
  material.diffuseColor = Color3.FromHexString('#fff176');
  material.emissiveColor = Color3.FromHexString('#e3b92f');
  material.disableLighting = true;
  halo.material = material;
  halo.position.y = 0.045;
  halo.isPickable = false;
  halo.isVisible = false;
  halo.alwaysSelectAsActiveMesh = true;
  return halo;
}

function createTaskMarkers(): { pallet: Mesh; station: Mesh } {
  const pallet = MeshBuilder.CreateTorus('selected-task-pallet', {
    diameter: CELL_SIZE * 1.08,
    thickness: 0.055,
    tessellation: 40,
  }, scene);
  const palletMaterial = new StandardMaterial('selected-task-pallet-material', scene);
  palletMaterial.diffuseColor = Color3.FromHexString('#ffc247');
  palletMaterial.emissiveColor = Color3.FromHexString('#ff9f1c');
  palletMaterial.disableLighting = true;
  pallet.material = palletMaterial;
  pallet.isPickable = false;
  pallet.isVisible = false;
  pallet.alwaysSelectAsActiveMesh = true;

  const station = MeshBuilder.CreateBox('selected-task-station', {
    width: CELL_SIZE * 0.88,
    height: 0.045,
    depth: CELL_SIZE * 0.88,
  }, scene);
  const stationMaterial = new StandardMaterial('selected-task-station-material', scene);
  stationMaterial.diffuseColor = Color3.FromHexString('#b572ff');
  stationMaterial.emissiveColor = Color3.FromHexString('#7e35df');
  stationMaterial.alpha = 0.58;
  stationMaterial.disableLighting = true;
  station.material = stationMaterial;
  station.isPickable = false;
  station.isVisible = false;
  station.alwaysSelectAsActiveMesh = true;

  const markerGlow = new GlowLayer('task-marker-glow', scene, { blurKernelSize: 22 });
  markerGlow.intensity = 0.72;
  markerGlow.addIncludedOnlyMesh(pallet);
  markerGlow.addIncludedOnlyMesh(station);
  return { pallet, station };
}

function hideTaskMarkers(): void {
  taskMarkers.pallet.isVisible = false;
  taskMarkers.station.isVisible = false;
}

function updateTaskMarkers(now: number): void {
  const frame = liveCurrent;
  const id = selectedAgentId;
  if (!frame || id < 0 || id >= frame.statuses.length) {
    hideTaskMarkers();
    return;
  }
  const palletId = frame.palletIds[id];
  const pallet = palletId < palletCells.length ? palletCells[palletId] : null;
  const stationX = frame.stationPositions[id * 2];
  const stationZ = frame.stationPositions[id * 2 + 1];
  if (!pallet || stationX < 0 || stationZ < 0) {
    hideTaskMarkers();
    return;
  }

  if ((frame.statuses[id] & 2) !== 0) {
    taskMarkers.pallet.position.set(matrices[id * 16 + 12], 1.02, matrices[id * 16 + 14]);
  } else {
    taskMarkers.pallet.position.copyFrom(worldAt(pallet.x, pallet.z, 1.02));
  }
  taskMarkers.station.position.copyFrom(worldAt(stationX, stationZ, 0.065));
  const palletPulse = 1 + Math.sin(now * 0.009) * 0.07;
  const stationPulse = 1 + Math.sin(now * 0.007 + Math.PI / 2) * 0.055;
  taskMarkers.pallet.scaling.set(palletPulse, palletPulse, palletPulse);
  taskMarkers.station.scaling.set(stationPulse, 1, stationPulse);
  taskMarkers.pallet.isVisible = true;
  taskMarkers.station.isVisible = true;
}

function createAgentPicker(): Mesh {
  const picker = MeshBuilder.CreateCylinder('agent-pick-target', {
    height: 1.65,
    diameter: CELL_SIZE * 0.9,
    tessellation: 12,
  }, scene);
  const material = new StandardMaterial('agent-pick-target-material', scene);
  material.alpha = 0;
  material.disableLighting = true;
  picker.material = material;
  picker.position.y = 0.78;
  picker.isPickable = true;
  picker.alwaysSelectAsActiveMesh = true;
  return picker;
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

function buildPalletCells(): PalletCell[] {
  const result: PalletCell[] = [];
  let id = 0;
  for (let gx = 5; gx < MAP_WIDTH - 4; gx += 8) {
    for (let gz = 4; gz < MAP_DEPTH - 3; gz += 7) {
      for (let x = gx; x < gx + 2; x++) {
        for (let z = gz - 1; z < gz + 3; z++) {
          result.push({ id: id++, x, z, cargoType: (x * 31 + z * 17) % 3 });
        }
      }
    }
  }
  return result;
}

function createWarehousePallets(pallets: PalletCell[]): {
  update: (hidden: Set<number>) => void;
} {
  const footprint = CELL_SIZE * 0.88;
  const half = footprint / 2;
  const legOffset = half - 0.09;
  const frameParts: Mesh[] = [];
  for (const x of [-legOffset, legOffset]) {
    for (const z of [-legOffset, legOffset]) {
      const leg = MeshBuilder.CreateBox('pallet-leg', { width: 0.1, height: 0.7, depth: 0.1 }, scene);
      leg.position.set(x, 0.38, z);
      frameParts.push(leg);
      const foot = MeshBuilder.CreateBox('pallet-foot', { width: 0.17, height: 0.08, depth: 0.17 }, scene);
      foot.position.set(x, 0.04, z);
      frameParts.push(foot);
    }
  }
  for (const z of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox('pallet-rail-x', { width: footprint, height: 0.1, depth: 0.1 }, scene);
    rail.position.set(0, 0.78, z);
    frameParts.push(rail);
  }
  for (const x of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox('pallet-rail-z', { width: 0.1, height: 0.1, depth: footprint }, scene);
    rail.position.set(x, 0.78, 0);
    frameParts.push(rail);
  }
  const frame = Mesh.MergeMeshes(frameParts, true, true, undefined, false, true)!;
  const frameMaterial = new StandardMaterial('pallet-frame-material', scene);
  frameMaterial.diffuseColor = Color3.FromHexString('#17313e');
  frameMaterial.emissiveColor = Color3.FromHexString('#07151e');
  frameMaterial.specularColor = Color3.FromHexString('#315d70');
  frame.material = frameMaterial;

  const deckParts: Mesh[] = [];
  for (const z of [-0.29, -0.1, 0.1, 0.29]) {
    const slat = MeshBuilder.CreateBox('pallet-slat', { width: footprint, height: 0.09, depth: 0.13 }, scene);
    slat.position.set(0, 0.88, z);
    deckParts.push(slat);
  }
  const deck = Mesh.MergeMeshes(deckParts, true, true, undefined, false, true)!;
  const deckMaterial = new StandardMaterial('pallet-deck-material', scene);
  deckMaterial.diffuseColor = Color3.FromHexString('#2d8098');
  deckMaterial.emissiveColor = Color3.FromHexString('#0b3442');
  deckMaterial.specularColor = Color3.FromHexString('#7bdcf2');
  deckMaterial.specularPower = 64;
  deck.material = deckMaterial;

  const crateParts: Mesh[] = [];
  const crateBody = MeshBuilder.CreateBox('cargo-crate-body', { width: 0.59, height: 0.48, depth: 0.59 }, scene);
  crateBody.position.y = 1.19;
  crateParts.push(crateBody);
  for (const x of [-0.275, 0.275]) {
    for (const z of [-0.275, 0.275]) {
      const brace = MeshBuilder.CreateBox('cargo-crate-brace', { width: 0.045, height: 0.52, depth: 0.045 }, scene);
      brace.position.set(x, 1.19, z);
      crateParts.push(brace);
    }
  }
  const crate = Mesh.MergeMeshes(crateParts, true, true, undefined, false, true)!;
  const crateMaterial = new StandardMaterial('cargo-crate-material', scene);
  crateMaterial.diffuseColor = Color3.FromHexString('#a96832');
  crateMaterial.emissiveColor = Color3.FromHexString('#261307');
  crate.material = crateMaterial;

  const cartonParts: Mesh[] = [];
  for (const x of [-0.17, 0.17]) {
    const carton = MeshBuilder.CreateBox('cargo-carton-lower', { width: 0.31, height: 0.32, depth: 0.56 }, scene);
    carton.position.set(x, 1.09, 0);
    cartonParts.push(carton);
  }
  const upperCarton = MeshBuilder.CreateBox('cargo-carton-upper', { width: 0.52, height: 0.28, depth: 0.44 }, scene);
  upperCarton.position.set(0, 1.4, 0);
  cartonParts.push(upperCarton);
  const cartons = Mesh.MergeMeshes(cartonParts, true, true, undefined, false, true)!;
  const cartonMaterial = new StandardMaterial('cargo-carton-material', scene);
  cartonMaterial.diffuseColor = Color3.FromHexString('#b98b5e');
  cartonMaterial.emissiveColor = Color3.FromHexString('#291b10');
  cartons.material = cartonMaterial;

  const drumParts: Mesh[] = [];
  for (const x of [-0.16, 0.16]) {
    for (const z of [-0.16, 0.16]) {
      const drum = MeshBuilder.CreateCylinder('cargo-drum', { height: 0.48, diameter: 0.24, tessellation: 12 }, scene);
      drum.position.set(x, 1.18, z);
      drumParts.push(drum);
    }
  }
  const drums = Mesh.MergeMeshes(drumParts, true, true, undefined, false, true)!;
  const drumMaterial = new StandardMaterial('cargo-drum-material', scene);
  drumMaterial.diffuseColor = Color3.FromHexString('#296d78');
  drumMaterial.emissiveColor = Color3.FromHexString('#09252d');
  drums.material = drumMaterial;

  const allMeshes = [frame, deck, crate, cartons, drums];
  for (const mesh of allMeshes) mesh.alwaysSelectAsActiveMesh = true;

  const update = (hidden: Set<number>) => {
    const palletTransforms: number[] = [];
    const cargoTransforms: number[][] = [[], [], []];
    for (const pallet of pallets) {
      if (hidden.has(pallet.id)) continue;
      const transform = gridTransform(pallet.x, pallet.z);
      transform.copyToArray(palletTransforms, palletTransforms.length);
      transform.copyToArray(cargoTransforms[pallet.cargoType], cargoTransforms[pallet.cargoType].length);
    }
    const setInstances = (mesh: Mesh, values: number[]) => {
      mesh.thinInstanceSetBuffer('matrix', new Float32Array(values), 16, false);
      mesh.thinInstanceCount = values.length / 16;
    };
    setInstances(frame, palletTransforms);
    setInstances(deck, palletTransforms);
    setInstances(crate, cargoTransforms[0]);
    setInstances(cartons, cargoTransforms[1]);
    setInstances(drums, cargoTransforms[2]);
  };
  update(new Set());
  return { update };
}

function createCarriedPalletMeshes(): {
  frame: Mesh;
  deck: Mesh;
  cargo: [Mesh, Mesh, Mesh];
} {
  const footprint = CELL_SIZE * 0.88;
  const half = footprint / 2;
  const legOffset = half - 0.09;
  const frameParts: Mesh[] = [];
  for (const x of [-legOffset, legOffset]) {
    for (const z of [-legOffset, legOffset]) {
      const leg = MeshBuilder.CreateBox('carried-pallet-leg', { width: 0.1, height: 0.7, depth: 0.1 }, scene);
      leg.position.set(x, 0.38, z);
      frameParts.push(leg);
      const foot = MeshBuilder.CreateBox('carried-pallet-foot', { width: 0.17, height: 0.08, depth: 0.17 }, scene);
      foot.position.set(x, 0.04, z);
      frameParts.push(foot);
    }
  }
  for (const z of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox('carried-pallet-rail-x', { width: footprint, height: 0.1, depth: 0.1 }, scene);
    rail.position.set(0, 0.78, z);
    frameParts.push(rail);
  }
  for (const x of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox('carried-pallet-rail-z', { width: 0.1, height: 0.1, depth: footprint }, scene);
    rail.position.set(x, 0.78, 0);
    frameParts.push(rail);
  }
  const frame = Mesh.MergeMeshes(frameParts, true, true, undefined, false, true)!;
  const frameMaterial = new StandardMaterial('carried-pallet-frame-material', scene);
  frameMaterial.diffuseColor = Color3.FromHexString('#17313e');
  frameMaterial.emissiveColor = Color3.FromHexString('#07151e');
  frameMaterial.specularColor = Color3.FromHexString('#315d70');
  frame.material = frameMaterial;

  const deckParts: Mesh[] = [];
  for (const z of [-0.29, -0.1, 0.1, 0.29]) {
    const slat = MeshBuilder.CreateBox('carried-pallet-slat', { width: footprint, height: 0.09, depth: 0.13 }, scene);
    slat.position.set(0, 0.88, z);
    deckParts.push(slat);
  }
  const deck = Mesh.MergeMeshes(deckParts, true, true, undefined, false, true)!;
  const deckMaterial = new StandardMaterial('carried-pallet-deck-material', scene);
  deckMaterial.diffuseColor = Color3.FromHexString('#2d8098');
  deckMaterial.emissiveColor = Color3.FromHexString('#0b3442');
  deckMaterial.specularColor = Color3.FromHexString('#7bdcf2');
  deckMaterial.specularPower = 64;
  deck.material = deckMaterial;

  const crateParts: Mesh[] = [];
  const crateBody = MeshBuilder.CreateBox('carried-cargo-crate-body', { width: 0.59, height: 0.48, depth: 0.59 }, scene);
  crateBody.position.y = 1.19;
  crateParts.push(crateBody);
  for (const x of [-0.275, 0.275]) {
    for (const z of [-0.275, 0.275]) {
      const brace = MeshBuilder.CreateBox('carried-cargo-crate-brace', { width: 0.045, height: 0.52, depth: 0.045 }, scene);
      brace.position.set(x, 1.19, z);
      crateParts.push(brace);
    }
  }
  const crate = Mesh.MergeMeshes(crateParts, true, true, undefined, false, true)!;
  const crateMaterial = new StandardMaterial('carried-cargo-crate-material', scene);
  crateMaterial.diffuseColor = Color3.FromHexString('#a96832');
  crateMaterial.emissiveColor = Color3.FromHexString('#261307');
  crate.material = crateMaterial;

  const cartonParts: Mesh[] = [];
  for (const x of [-0.17, 0.17]) {
    const carton = MeshBuilder.CreateBox('carried-cargo-carton-lower', { width: 0.31, height: 0.32, depth: 0.56 }, scene);
    carton.position.set(x, 1.09, 0);
    cartonParts.push(carton);
  }
  const upperCarton = MeshBuilder.CreateBox('carried-cargo-carton-upper', { width: 0.52, height: 0.28, depth: 0.44 }, scene);
  upperCarton.position.set(0, 1.4, 0);
  cartonParts.push(upperCarton);
  const cartons = Mesh.MergeMeshes(cartonParts, true, true, undefined, false, true)!;
  const cartonMaterial = new StandardMaterial('carried-cargo-carton-material', scene);
  cartonMaterial.diffuseColor = Color3.FromHexString('#b98b5e');
  cartonMaterial.emissiveColor = Color3.FromHexString('#291b10');
  cartons.material = cartonMaterial;

  const drumParts: Mesh[] = [];
  for (const x of [-0.16, 0.16]) {
    for (const z of [-0.16, 0.16]) {
      const drum = MeshBuilder.CreateCylinder('carried-cargo-drum', { height: 0.48, diameter: 0.24, tessellation: 12 }, scene);
      drum.position.set(x, 1.18, z);
      drumParts.push(drum);
    }
  }
  const drums = Mesh.MergeMeshes(drumParts, true, true, undefined, false, true)!;
  const drumMaterial = new StandardMaterial('carried-cargo-drum-material', scene);
  drumMaterial.diffuseColor = Color3.FromHexString('#296d78');
  drumMaterial.emissiveColor = Color3.FromHexString('#09252d');
  drums.material = drumMaterial;

  const cargo: [Mesh, Mesh, Mesh] = [crate, cartons, drums];
  for (const mesh of [frame, deck, ...cargo]) mesh.alwaysSelectAsActiveMesh = true;
  return { frame, deck, cargo };
}

function createUnloadingZone(): {
  trigger: (stationZ: number, cargoType: number, startedAt: number) => void;
  update: (now: number) => CargoTransfer[];
  reset: () => void;
} {
  const pad = MeshBuilder.CreateBox('unloading-pad', {
    width: CELL_SIZE * 0.82,
    height: 0.035,
    depth: CELL_SIZE * 0.82,
  }, scene);
  const padMaterial = new StandardMaterial('unloading-pad-material', scene);
  padMaterial.diffuseColor = Color3.FromHexString('#244d59');
  padMaterial.emissiveColor = Color3.FromHexString('#0e5365');
  pad.material = padMaterial;
  const padTransforms: number[] = [];
  for (let z = UNLOAD_MIN_Z; z <= UNLOAD_MAX_Z; z++) {
    const transform = gridTransform(UNLOAD_X, z);
    transform.setTranslation(transform.getTranslation().add(new Vector3(0, 0.02, 0)));
    transform.copyToArray(padTransforms, padTransforms.length);
  }
  pad.thinInstanceSetBuffer('matrix', new Float32Array(padTransforms), 16, true);

  const serviceStrip = MeshBuilder.CreateBox('unloading-service-strip', {
    width: CELL_SIZE * 0.86,
    height: 0.035,
    depth: UNLOAD_STATION_COUNT * CELL_SIZE,
  }, scene);
  const serviceMaterial = new StandardMaterial('unloading-service-material', scene);
  serviceMaterial.diffuseColor = Color3.FromHexString('#17242b');
  serviceMaterial.emissiveColor = Color3.FromHexString('#121a1f');
  serviceMaterial.specularColor = Color3.FromHexString('#526b73');
  serviceStrip.material = serviceMaterial;
  serviceStrip.position.copyFrom(worldAt(MAP_WIDTH - 2, (UNLOAD_MIN_Z + UNLOAD_MAX_Z) / 2, -0.01));

  const endstop = MeshBuilder.CreateBox('unloading-endstop', {
    width: CELL_SIZE * 0.9,
    height: 0.2,
    depth: 0.14,
  }, scene);
  const safetyMaterial = new StandardMaterial('unloading-safety-material', scene);
  safetyMaterial.diffuseColor = Color3.FromHexString('#be7626');
  safetyMaterial.emissiveColor = Color3.FromHexString('#44260a');
  endstop.material = safetyMaterial;
  const endstopTransforms: number[] = [];
  for (const z of [0, MAP_DEPTH - 1]) {
    const transform = gridTransform(UNLOAD_X, z);
    transform.setTranslation(transform.getTranslation().add(new Vector3(0, 0.1, 0)));
    transform.copyToArray(endstopTransforms, endstopTransforms.length);
  }
  endstop.thinInstanceSetBuffer('matrix', new Float32Array(endstopTransforms), 16, true);

  const beacon = MeshBuilder.CreateCylinder('unloading-beacon', { height: 0.09, diameter: 0.18, tessellation: 12 }, scene);
  const beaconMaterial = new StandardMaterial('unloading-beacon-material', scene);
  beaconMaterial.emissiveColor = Color3.FromHexString('#ffbd3e');
  beaconMaterial.disableLighting = true;
  beacon.material = beaconMaterial;
  const beaconTransforms: number[] = [];
  for (const z of [UNLOAD_MIN_Z, UNLOAD_MAX_Z]) {
    const transform = gridTransform(MAP_WIDTH - 2, z);
    transform.setTranslation(transform.getTranslation().add(new Vector3(0, 0.16, 0)));
    transform.copyToArray(beaconTransforms, beaconTransforms.length);
  }
  beacon.thinInstanceSetBuffer('matrix', new Float32Array(beaconTransforms), 16, true);

  const armMaterial = new StandardMaterial('robotic-arm-material', scene);
  armMaterial.diffuseColor = Color3.FromHexString('#d9822b');
  armMaterial.emissiveColor = Color3.FromHexString('#3d1c05');
  armMaterial.specularColor = Color3.FromHexString('#ffd08b');
  armMaterial.specularPower = 72;
  const jointMaterial = new StandardMaterial('robotic-arm-joint-material', scene);
  jointMaterial.diffuseColor = Color3.FromHexString('#263944');
  jointMaterial.emissiveColor = Color3.FromHexString('#0b171d');
  jointMaterial.specularColor = Color3.FromHexString('#99c0ca');
  const gripperMaterial = new StandardMaterial('robotic-arm-gripper-material', scene);
  gripperMaterial.diffuseColor = Color3.FromHexString('#63d8e8');
  gripperMaterial.emissiveColor = Color3.FromHexString('#123e48');

  const base = MeshBuilder.CreateCylinder('robotic-arm-base', { height: 0.16, diameter: 0.58, tessellation: 20 }, scene);
  base.material = jointMaterial;
  const pedestal = MeshBuilder.CreateCylinder('robotic-arm-pedestal', {
    height: 0.58,
    diameterTop: 0.29,
    diameterBottom: 0.38,
    tessellation: 16,
  }, scene);
  pedestal.material = armMaterial;
  const shoulder = MeshBuilder.CreateSphere('robotic-arm-shoulder', { diameter: 0.31, segments: 12 }, scene);
  shoulder.material = jointMaterial;
  const upper = MeshBuilder.CreateBox('robotic-arm-upper', { width: 1, height: 0.16, depth: 0.18 }, scene);
  upper.material = armMaterial;
  const forearm = MeshBuilder.CreateBox('robotic-arm-forearm', { width: 1, height: 0.13, depth: 0.15 }, scene);
  forearm.material = armMaterial;
  const movingJoint = MeshBuilder.CreateSphere('robotic-arm-moving-joint', { diameter: 0.24, segments: 10 }, scene);
  movingJoint.material = jointMaterial;

  const gripperParts: Mesh[] = [];
  const gripperBar = MeshBuilder.CreateBox('robotic-arm-gripper-bar', { width: 0.16, height: 0.09, depth: 0.46 }, scene);
  gripperParts.push(gripperBar);
  for (const z of [-0.18, 0.18]) {
    const finger = MeshBuilder.CreateBox('robotic-arm-gripper-finger', { width: 0.08, height: 0.3, depth: 0.07 }, scene);
    finger.position.set(-0.01, -0.14, z);
    gripperParts.push(finger);
  }
  const gripper = Mesh.MergeMeshes(gripperParts, true, true, undefined, false, true)!;
  gripper.name = 'robotic-arm-gripper';
  gripper.material = gripperMaterial;

  const baseTransforms: number[] = [];
  const pedestalTransforms: number[] = [];
  const shoulderTransforms: number[] = [];
  for (let z = UNLOAD_MIN_Z; z <= UNLOAD_MAX_Z; z++) {
    const armPosition = worldAt(MAP_WIDTH - 2, z, 0);
    Matrix.Translation(armPosition.x, 0.08, armPosition.z).copyToArray(baseTransforms, baseTransforms.length);
    Matrix.Translation(armPosition.x, 0.43, armPosition.z).copyToArray(pedestalTransforms, pedestalTransforms.length);
    Matrix.Translation(armPosition.x, 0.78, armPosition.z).copyToArray(shoulderTransforms, shoulderTransforms.length);
  }
  base.thinInstanceSetBuffer('matrix', new Float32Array(baseTransforms), 16, true);
  pedestal.thinInstanceSetBuffer('matrix', new Float32Array(pedestalTransforms), 16, true);
  shoulder.thinInstanceSetBuffer('matrix', new Float32Array(shoulderTransforms), 16, true);

  const upperMatrices = new Float32Array(UNLOAD_STATION_COUNT * 16);
  const forearmMatrices = new Float32Array(UNLOAD_STATION_COUNT * 16);
  const jointMatrices = new Float32Array(UNLOAD_STATION_COUNT * 2 * 16);
  const gripperMatrices = new Float32Array(UNLOAD_STATION_COUNT * 16);
  upper.thinInstanceSetBuffer('matrix', upperMatrices, 16, false);
  forearm.thinInstanceSetBuffer('matrix', forearmMatrices, 16, false);
  movingJoint.thinInstanceSetBuffer('matrix', jointMatrices, 16, false);
  gripper.thinInstanceSetBuffer('matrix', gripperMatrices, 16, false);
  upper.thinInstanceCount = UNLOAD_STATION_COUNT;
  forearm.thinInstanceCount = UNLOAD_STATION_COUNT;
  movingJoint.thinInstanceCount = UNLOAD_STATION_COUNT * 2;
  gripper.thinInstanceCount = UNLOAD_STATION_COUNT;
  for (const mesh of [base, pedestal, shoulder, upper, forearm, movingJoint, gripper]) {
    mesh.alwaysSelectAsActiveMesh = true;
  }

  type ArmEvent = { startedAt: number; cargoType: number };
  type Point = { x: number; y: number };
  type Pose = { elbow: Point; grip: Point };
  const events = new Map<number, ArmEvent>();
  const durationMs = 2000;

  const mix = (a: number, b: number, t: number): number => a + (b - a) * smoothstep(t);
  const mixPoint = (a: Point, b: Point, t: number): Point => ({ x: mix(a.x, b.x, t), y: mix(a.y, b.y, t) });
  const mixPose = (a: Pose, b: Pose, t: number): Pose => ({
    elbow: mixPoint(a.elbow, b.elbow, t),
    grip: mixPoint(a.grip, b.grip, t),
  });
  const segmentMatrix = (from: Point, to: Point, z: number): Matrix => {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    return Matrix.Compose(
      new Vector3(Math.hypot(dx, dy), 1, 1),
      Quaternion.RotationAxis(Vector3.Forward(), Math.atan2(dy, dx)),
      new Vector3((from.x + to.x) / 2, (from.y + to.y) / 2, z),
    );
  };

  const update = (now: number): CargoTransfer[] => {
    const transfers: CargoTransfer[] = [];
    for (let index = 0; index < UNLOAD_STATION_COUNT; index++) {
      const stationZ = UNLOAD_MIN_Z + index;
      const stationWorld = worldAt(UNLOAD_X, stationZ, 0);
      const armWorld = worldAt(MAP_WIDTH - 2, stationZ, 0);
      const shoulderPoint = { x: armWorld.x, y: 0.78 };
      const parked: Pose = {
        elbow: { x: armWorld.x - 0.05, y: 1.38 },
        grip: { x: armWorld.x + 0.11, y: 1.88 },
      };
      const reached: Pose = {
        elbow: { x: armWorld.x - 0.43, y: 1.4 },
        grip: { x: stationWorld.x, y: 1.68 },
      };
      const lifted: Pose = {
        elbow: { x: armWorld.x - 0.19, y: 1.5 },
        grip: { x: armWorld.x - 0.35, y: 2.0 },
      };
      const dropped: Pose = {
        elbow: { x: armWorld.x + 0.12, y: 1.42 },
        grip: { x: armWorld.x + 0.31, y: 1.67 },
      };
      const event = events.get(stationZ);
      const progress = event ? Math.max(0, (now - event.startedAt) / durationMs) : 0;
      let pose = parked;
      if (event) {
        if (progress < 0.32) pose = mixPose(parked, reached, progress / 0.32);
        else if (progress < 0.42) pose = reached;
        else if (progress < 0.66) pose = mixPose(reached, lifted, (progress - 0.42) / 0.24);
        else if (progress < 0.84) pose = mixPose(lifted, dropped, (progress - 0.66) / 0.18);
        else if (progress < 0.91) pose = dropped;
        else pose = mixPose(dropped, parked, (progress - 0.91) / 0.09);

        if (progress < 0.9) {
          let cargoGridX = UNLOAD_X;
          let cargoY = 0.1;
          if (progress >= 0.38 && progress < 0.68) {
            const phase = smoothstep((progress - 0.38) / 0.3);
            cargoGridX = mix(UNLOAD_X, UNLOAD_X + 0.61, phase);
            cargoY = mix(0.1, 0.53, phase);
          } else if (progress >= 0.68) {
            const phase = smoothstep((progress - 0.68) / 0.22);
            cargoGridX = mix(UNLOAD_X + 0.61, UNLOAD_X + 1.3, phase);
            cargoY = mix(0.53, 0.22, phase);
          }
          transfers.push({ gridX: cargoGridX, gridZ: stationZ, worldY: cargoY, cargoType: event.cargoType });
        }
        if (progress >= 1) events.delete(stationZ);
      }

      segmentMatrix(shoulderPoint, pose.elbow, armWorld.z).copyToArray(upperMatrices, index * 16);
      segmentMatrix(pose.elbow, pose.grip, armWorld.z).copyToArray(forearmMatrices, index * 16);
      Matrix.Translation(pose.elbow.x, pose.elbow.y, armWorld.z).copyToArray(jointMatrices, index * 32);
      Matrix.Translation(pose.grip.x, pose.grip.y, armWorld.z).copyToArray(jointMatrices, index * 32 + 16);
      Matrix.Translation(pose.grip.x, pose.grip.y - 0.03, armWorld.z).copyToArray(gripperMatrices, index * 16);
    }
    upper.thinInstanceBufferUpdated('matrix');
    forearm.thinInstanceBufferUpdated('matrix');
    movingJoint.thinInstanceBufferUpdated('matrix');
    gripper.thinInstanceBufferUpdated('matrix');
    return transfers;
  };

  update(0);
  return {
    trigger: (stationZ, cargoType, startedAt) => {
      if (stationZ < UNLOAD_MIN_Z || stationZ > UNLOAD_MAX_Z) return;
      events.set(stationZ, { startedAt, cargoType });
    },
    update,
    reset: () => events.clear(),
  };
}

function worldAt(gridX: number, gridZ: number, worldY: number): Vector3 {
  return new Vector3(
    (gridX + 0.5) * CELL_SIZE - MAP_WIDTH * CELL_SIZE / 2,
    worldY,
    (gridZ + 0.5) * CELL_SIZE - MAP_DEPTH * CELL_SIZE / 2,
  );
}

function gridTransform(x: number, z: number): Matrix {
  return Matrix.Translation(
    (x + 0.5) * CELL_SIZE - MAP_WIDTH * CELL_SIZE / 2,
    0,
    (z + 0.5) * CELL_SIZE - MAP_DEPTH * CELL_SIZE / 2,
  );
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
    for (let z = 1; z < MAP_DEPTH - 1; z++) {
      if (z >= UNLOAD_MIN_Z && z <= UNLOAD_MAX_Z && x >= UNLOAD_X) continue;
      result.push({ x, z, direction: x % 2 === 0 ? 1 : -1 });
    }
  }
  return result;
}

function writeTransform(
  target: Float32Array,
  index: number,
  gridX: number,
  gridZ: number,
  worldY = 0.02,
): void {
  const halfW = MAP_WIDTH * CELL_SIZE / 2;
  const halfD = MAP_DEPTH * CELL_SIZE / 2;
  const offset = index * 16;
  target[offset] = 1;
  target[offset + 1] = 0;
  target[offset + 2] = 0;
  target[offset + 3] = 0;
  target[offset + 4] = 0;
  target[offset + 5] = 1;
  target[offset + 6] = 0;
  target[offset + 7] = 0;
  target[offset + 8] = 0;
  target[offset + 9] = 0;
  target[offset + 10] = 1;
  target[offset + 11] = 0;
  target[offset + 12] = (gridX + 0.5) * CELL_SIZE - halfW;
  target[offset + 13] = worldY;
  target[offset + 14] = (gridZ + 0.5) * CELL_SIZE - halfD;
  target[offset + 15] = 1;
}

function updateFallback(time: number): void {
  unloadingScene.update(performance.now());
  selectionHalo.isVisible = false;
  hideTaskMarkers();
  const step = time / FALLBACK_STEP_SECONDS;
  const wholeStep = Math.floor(step);
  const phase = smoothstep(step - wholeStep);
  for (let i = 0; i < agentCount; i++) {
    const cell = cells[(i * 47) % cells.length];
    const range = MAP_DEPTH - 2;
    const z = 1 + mod(cell.z - 1 + cell.direction * (wholeStep + phase), range);
    writeTransform(matrices, i, cell.x, z);
  }
  robotMesh.thinInstanceBufferUpdated('matrix');
  agentPicker.thinInstanceBufferUpdated('matrix');
  robotLoad.frame.thinInstanceCount = 0;
  robotLoad.deck.thinInstanceCount = 0;
  for (const cargo of robotLoad.cargo) cargo.thinInstanceCount = 0;
}

function updateLive(now: number): void {
  if (!liveCurrent) return;
  const transfers = unloadingScene.update(now);
  const from = livePrevious ?? liveCurrent;
  const frameDurationMs = 1000 / Math.max(0.1, liveTickRate * speed);
  const alpha = Math.min(1, (now - liveCurrent.receivedAt) / frameDurationMs);
  let loadedCount = 0;
  const cargoCounts = [0, 0, 0];
  loadAgentIds.fill(-1);
  for (const ids of cargoAgentIds) ids.fill(-1);
  for (let i = 0; i < agentCount; i++) {
    const x0 = from.positions[i * 2];
    const z0 = from.positions[i * 2 + 1];
    const x1 = liveCurrent.positions[i * 2];
    const z1 = liveCurrent.positions[i * 2 + 1];
    const x = x0 + (x1 - x0) * alpha;
    const z = z0 + (z1 - z0) * alpha;
    writeTransform(matrices, i, x, z);
    if (i === selectedAgentId) {
      const position = worldAt(x, z, 0.045);
      selectionHalo.position.copyFrom(position);
      const pulse = 1 + Math.sin(now * 0.008) * 0.055;
      selectionHalo.scaling.set(pulse, pulse, pulse);
      selectionHalo.isVisible = true;
    }
    if ((liveCurrent.statuses[i] & 2) !== 0) {
      const loadIndex = loadedCount++;
      loadAgentIds[loadIndex] = i;
      writeTransform(loadMatrices, loadIndex, x, z, 0.1);
      const palletId = liveCurrent.palletIds[i];
      if (liveCurrent.stages[i] === 1 && palletId < palletCells.length) {
        const cargoType = palletCells[palletId].cargoType;
        const cargoIndex = cargoCounts[cargoType]++;
        cargoAgentIds[cargoType][cargoIndex] = i;
        writeTransform(cargoMatrices[cargoType], cargoIndex, x, z, 0.1);
      }
    }
  }
  for (const transfer of transfers) {
    const cargoIndex = cargoCounts[transfer.cargoType]++;
    cargoAgentIds[transfer.cargoType][cargoIndex] = -1;
    writeTransform(
      cargoMatrices[transfer.cargoType],
      cargoIndex,
      transfer.gridX,
      transfer.gridZ,
      transfer.worldY,
    );
  }
  robotMesh.thinInstanceBufferUpdated('matrix');
  agentPicker.thinInstanceBufferUpdated('matrix');
  robotLoad.frame.thinInstanceCount = loadedCount;
  robotLoad.deck.thinInstanceCount = loadedCount;
  if (loadedCount) {
    robotLoad.frame.thinInstanceBufferUpdated('matrix');
    robotLoad.deck.thinInstanceBufferUpdated('matrix');
  }
  for (let type = 0; type < robotLoad.cargo.length; type++) {
    robotLoad.cargo[type].thinInstanceCount = cargoCounts[type];
    if (cargoCounts[type]) robotLoad.cargo[type].thinInstanceBufferUpdated('matrix');
  }
}

function setAgentColor(index: number, status: number): void {
  const loaded = (status & 2) !== 0;
  const waiting = (status & 1) !== 0;
  const transitioned = (status & 4) !== 0;
  const color = index === selectedAgentId
    ? [1, 0.93, 0.32, 1]
    : transitioned
    ? [0.5, 1, 0.68, 1]
    : loaded
      ? (waiting ? [0.82, 0.44, 0.16, 1] : [1, 0.66, 0.2, 1])
      : waiting
        ? [0.5, 0.56, 0.62, 1]
        : [0.12, 0.78 + (index % 7) * 0.025, 1, 1];
  colors.set(color, index * 4);
}

function parseFrame(buffer: ArrayBuffer): void {
  const view = new DataView(buffer);
  if (view.byteLength < 16 || view.getUint32(0, true) !== FRAME_MAGIC) return;
  const protocol = view.getUint16(4, true);
  if (protocol !== 1 && protocol !== 2) return;
  const recordSize = protocol === 2 ? 24 : 16;
  const step = view.getUint32(8, true);
  const priorFrame = liveCurrent && step > liveCurrent.step ? liveCurrent : null;
  if (liveCurrent && step < liveCurrent.step) {
    unloadingScene.reset();
    resetAgentStats();
  }
  const completedTasks = view.getUint16(6, true);
  const count = Math.min(MAX_AGENTS, view.getUint32(12, true));
  if (view.byteLength < 16 + count * recordSize) return;
  const positions = new Float32Array(count * 2);
  const statuses = new Uint8Array(count);
  const stages = new Uint8Array(count);
  const palletIds = new Uint16Array(count);
  const taskIds = new Uint32Array(count);
  const stationPositions = new Int16Array(count * 2);
  taskIds.fill(0xffffffff);
  stationPositions.fill(-1);
  const hiddenPallets = new Set<number>();
  for (let i = 0; i < count; i++) {
    const offset = 16 + i * recordSize;
    const id = view.getUint32(offset, true);
    if (id >= count) continue;
    positions[id * 2] = view.getFloat32(offset + 4, true);
    positions[id * 2 + 1] = view.getFloat32(offset + 8, true);
    statuses[id] = view.getUint8(offset + 12);
    stages[id] = view.getUint8(offset + 13);
    palletIds[id] = view.getUint16(offset + 14, true);
    if (protocol === 2) {
      taskIds[id] = view.getUint32(offset + 16, true);
      stationPositions[id * 2] = view.getInt16(offset + 20, true);
      stationPositions[id * 2 + 1] = view.getInt16(offset + 22, true);
    }
    if ((statuses[id] & 2) !== 0 && palletIds[id] !== 65535) hiddenPallets.add(palletIds[id]);
    setAgentColor(id, statuses[id]);
  }
  const nextHiddenKey = [...hiddenPallets].sort((a, b) => a - b).join(',');
  if (nextHiddenKey !== hiddenPalletKey) {
    palletScene.update(hiddenPallets);
    hiddenPalletKey = nextHiddenKey;
  }
  if (count !== agentCount) setAgentCount(count);
  const receivedAt = performance.now();
  if (priorFrame) {
    for (let id = 0; id < count && id < priorFrame.stages.length; id++) {
      const stats = agentStats[id];
      const dx = Math.abs(positions[id * 2] - priorFrame.positions[id * 2]);
      const dz = Math.abs(positions[id * 2 + 1] - priorFrame.positions[id * 2 + 1]);
      const distance = dx + dz;
      if (distance > 0) {
        stats.distance += distance;
        stats.moveTicks += 1;
      } else {
        stats.waitTicks += 1;
      }
      if (priorFrame.stages[id] === 1 && stages[id] === 2) stats.handoffs += 1;
      if (taskIds[id] !== priorFrame.taskIds[id]) {
        if (priorFrame.taskIds[id] !== 0xffffffff) stats.completedTasks += 1;
        stats.taskStartedAt = step;
        stats.lastTaskId = taskIds[id];
      }
      if (priorFrame.stages[id] !== 1 || stages[id] !== 2) continue;
      const palletId = palletIds[id];
      const x = Math.round(positions[id * 2]);
      const stationZ = Math.round(positions[id * 2 + 1]);
      if (x === UNLOAD_X && palletId < palletCells.length) {
        unloadingScene.trigger(stationZ, palletCells[palletId].cargoType, receivedAt);
      }
    }
  } else {
    for (let id = 0; id < count; id++) {
      if (agentStats[id].lastTaskId === 0xffffffff) {
        agentStats[id].lastTaskId = taskIds[id];
        agentStats[id].taskStartedAt = step;
      }
    }
  }
  livePrevious = priorFrame;
  liveCurrent = {
    positions,
    statuses,
    stages,
    palletIds,
    taskIds,
    stationPositions,
    completedTasks,
    step,
    receivedAt,
  };
  robotMesh.thinInstanceBufferUpdated('color');
  renderAgentPanel();
}

function createAgentStats(): AgentStats {
  return {
    distance: 0,
    moveTicks: 0,
    waitTicks: 0,
    handoffs: 0,
    completedTasks: 0,
    taskStartedAt: 0,
    lastTaskId: 0xffffffff,
  };
}

function resetAgentStats(): void {
  for (let i = 0; i < agentStats.length; i++) agentStats[i] = createAgentStats();
  renderAgentPanel();
}

function selectAgent(agent: number): void {
  const previous = selectedAgentId;
  selectedAgentId = agent;
  if (previous >= 0) setAgentColor(previous, liveCurrent?.statuses[previous] ?? 0);
  setAgentColor(agent, liveCurrent?.statuses[agent] ?? 0);
  robotMesh.thinInstanceBufferUpdated('color');
  selectionHalo.isVisible = true;
  renderAgentPanel();
}

function clearAgentSelection(): void {
  const previous = selectedAgentId;
  selectedAgentId = -1;
  selectionHalo.isVisible = false;
  hideTaskMarkers();
  if (previous >= 0) {
    setAgentColor(previous, liveCurrent?.statuses[previous] ?? 0);
    robotMesh.thinInstanceBufferUpdated('color');
  }
  renderAgentPanel();
}

function renderAgentPanel(): void {
  const empty = document.querySelector<HTMLElement>('#agent-empty')!;
  const details = document.querySelector<HTMLElement>('#agent-details')!;
  const clearButton = document.querySelector<HTMLButtonElement>('#clear-selection')!;
  const frame = liveCurrent;
  if (!frame || selectedAgentId < 0 || selectedAgentId >= frame.stages.length) {
    empty.hidden = false;
    details.hidden = true;
    clearButton.hidden = true;
    return;
  }

  empty.hidden = true;
  details.hidden = false;
  clearButton.hidden = false;
  const id = selectedAgentId;
  const stage = Math.min(2, frame.stages[id]);
  const status = frame.statuses[id];
  const palletId = frame.palletIds[id];
  const taskId = frame.taskIds[id];
  const pallet = palletId < palletCells.length ? palletCells[palletId] : null;
  const stationX = frame.stationPositions[id * 2];
  const stationZ = frame.stationPositions[id * 2 + 1];
  const stats = agentStats[id];
  const waiting = (status & 1) !== 0;
  const loaded = (status & 2) !== 0;
  const transitioned = (status & 4) !== 0;
  const atUnloadingBay = stage === 2 && Math.round(frame.positions[id * 2]) === UNLOAD_X;
  const stateLabel = transitioned
    ? 'HANDOFF'
    : atUnloadingBay && waiting
      ? 'UNLOADING'
      : waiting
        ? 'WAITING'
        : loaded
          ? 'LOADED'
          : 'EMPTY';
  const statePill = document.querySelector<HTMLElement>('#selected-agent-state')!;
  statePill.textContent = stateLabel;
  statePill.className = `state-pill${loaded ? ' loaded' : ''}${waiting ? ' waiting' : ''}`;

  document.querySelector('#selected-agent')!.textContent = `AGENT ${String(id).padStart(3, '0')}`;
  document.querySelector('#agent-position')!.textContent =
    `${Math.round(frame.positions[id * 2])}, ${Math.round(frame.positions[id * 2 + 1])}`;
  document.querySelector('#agent-task')!.textContent = taskId === 0xffffffff ? '—' : `#${taskId}`;
  const cargoNames = ['CRATE', 'CARTONS', 'DRUMS'];
  document.querySelector('#agent-pallet')!.textContent = pallet
    ? `P${String(palletId).padStart(3, '0')} · ${cargoNames[pallet.cargoType]}`
    : '—';
  document.querySelector('#agent-task-age')!.textContent = `${Math.max(0, frame.step - stats.taskStartedAt)} TICKS`;

  const formatCoordinate = (x: number, z: number): string => x >= 0 && z >= 0 ? `${x}, ${z}` : '—';
  document.querySelector('#pickup-coordinate')!.textContent = pallet ? formatCoordinate(pallet.x, pallet.z) : '—';
  document.querySelector('#unload-coordinate')!.textContent = formatCoordinate(stationX, stationZ);
  document.querySelector('#return-coordinate')!.textContent = pallet ? formatCoordinate(pallet.x, pallet.z) : '—';

  const stageElements = [
    document.querySelector<HTMLElement>('#stage-pickup')!,
    document.querySelector<HTMLElement>('#stage-unload')!,
    document.querySelector<HTMLElement>('#stage-return')!,
  ];
  const routeElements = [
    document.querySelector<HTMLElement>('#route-pickup')!,
    document.querySelector<HTMLElement>('#route-unload')!,
    document.querySelector<HTMLElement>('#route-return')!,
  ];
  for (let index = 0; index < 3; index++) {
    const state = index < stage ? 'done' : index === stage ? 'active' : '';
    stageElements[index].className = state;
    routeElements[index].className = state;
  }

  document.querySelector('#agent-distance')!.textContent = stats.distance.toFixed(0);
  document.querySelector('#agent-waits')!.textContent = String(stats.waitTicks);
  document.querySelector('#agent-handoffs')!.textContent = String(stats.handoffs);
  document.querySelector('#agent-completed')!.textContent = String(stats.completedTasks);
}

function setConnection(state: string, label: string): void {
  const connection = document.querySelector<HTMLElement>('.connection')!;
  connection.dataset.state = state;
  document.querySelector('#source-label')!.textContent = label;
}

function setAgentCount(count: number): void {
  agentCount = Math.min(MAX_AGENTS, count);
  if (selectedAgentId >= agentCount) clearAgentSelection();
  robotMesh.thinInstanceCount = agentCount;
  agentPicker.thinInstanceCount = agentCount;
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
  ws.addEventListener('open', () => setConnection('planning', 'LIFELONG // PLANNING'));
  ws.addEventListener('message', (event) => {
    if (event.data instanceof ArrayBuffer) {
      live = true;
      parseFrame(event.data);
      setConnection('live', 'FASTDMM // SIMULATOR');
      return;
    }
    const message = JSON.parse(String(event.data));
    if (message.type === 'hello') {
      live = true;
      paused = false;
      syncPauseButton();
      liveTickRate = Number(message.tickRate) || 10;
      lastInferenceMs = Number(message.inferenceMs) || 0;
      setAgentCount(Number(message.agents));
      sendControl('speed', { value: speed });
      setConnection('live', 'FASTDMM // SIMULATOR');
    } else if (message.type === 'status' && message.state === 'planning') {
      livePrevious = null;
      liveCurrent = null;
      unloadingScene.reset();
      resetAgentStats();
      clearAgentSelection();
      setConnection('planning', `PLANNING // ${Number(message.agents).toLocaleString('en-US')}`);
    } else if (message.type === 'status' && message.state === 'stopped') {
      paused = true;
      syncPauseButton();
      setConnection('stopped', 'SIMULATION // STOPPED');
    } else if (message.type === 'metrics') {
      lastInferenceMs = Number(message.inferenceMs) || 0;
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
    unloadingScene.reset();
    resetAgentStats();
    clearAgentSelection();
    palletScene.update(new Set());
    hiddenPalletKey = '';
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

function updateCameraMovement(dt: number): void {
  if (cameraKeys.size === 0) return;
  const forward = camera.target.subtract(camera.position);
  forward.y = 0;
  if (forward.lengthSquared() < 1e-6) return;
  forward.normalize();
  const right = Vector3.Cross(Vector3.Up(), forward).normalize();
  const direction = Vector3.Zero();
  if (cameraKeys.has('w')) direction.addInPlace(forward);
  if (cameraKeys.has('s')) direction.subtractInPlace(forward);
  if (cameraKeys.has('d')) direction.addInPlace(right);
  if (cameraKeys.has('a')) direction.subtractInPlace(right);
  if (direction.lengthSquared() < 1e-6) return;
  direction.normalize();
  const movementSpeed = Math.max(2.5, camera.radius * 0.18);
  camera.target.addInPlace(direction.scale(movementSpeed * dt));
  const margin = CELL_SIZE * 3;
  const maxX = MAP_WIDTH * CELL_SIZE / 2 + margin;
  const maxZ = MAP_DEPTH * CELL_SIZE / 2 + margin;
  camera.target.x = Math.max(-maxX, Math.min(maxX, camera.target.x));
  camera.target.z = Math.max(-maxZ, Math.min(maxZ, camera.target.z));
}

const fpsValue = document.querySelector('#fps-value')!;
const stepValue = document.querySelector('#step-value')!;
const agentsValue = document.querySelector('#agents-value')!;
const throughputValue = document.querySelector('#throughput-value')!;
const latencyValue = document.querySelector('#latency-value')!;
const loadedValue = document.querySelector('#loaded-value')!;
const tasksValue = document.querySelector('#tasks-value')!;
let telemetryElapsed = 0;

engine.runRenderLoop(() => {
  const dt = Math.min(engine.getDeltaTime() / 1000, 0.05);
  updateCameraMovement(dt);
  if (!paused && !live) simTime += dt * speed;
  if (live) updateLive(performance.now());
  else updateFallback(simTime);
  updateTaskMarkers(performance.now());
  scene.render();
  telemetryElapsed += dt;
  if (telemetryElapsed > 0.25) {
    telemetryElapsed = 0;
    fpsValue.textContent = String(Math.round(engine.getFps()));
    const step = liveCurrent?.step ?? Math.floor(simTime / FALLBACK_STEP_SECONDS);
    stepValue.textContent = String(step).padStart(4, '0');
    throughputValue.textContent = paused
      ? '0'
      : Math.round(agentCount * (live ? liveTickRate : 1 / FALLBACK_STEP_SECONDS) * speed).toLocaleString('en-US');
    latencyValue.textContent = live ? `${lastInferenceMs.toFixed(1)} ms` : 'DEMO';
    loadedValue.textContent = String(liveCurrent ? liveCurrent.statuses.reduce((sum, status) => sum + ((status & 2) !== 0 ? 1 : 0), 0) : 0);
    tasksValue.textContent = String(liveCurrent?.completedTasks ?? 0);
  }
});

const pauseButton = document.querySelector<HTMLButtonElement>('#pause-button')!;
const stopButton = document.querySelector<HTMLButtonElement>('#stop-button')!;
document.querySelector('#clear-selection')!.addEventListener('click', clearAgentSelection);
function syncPauseButton(): void {
  document.querySelector('#pause-icon')!.textContent = paused ? '▶' : 'Ⅱ';
  document.querySelector('#pause-label')!.textContent = paused ? 'RUN' : 'PAUSE';
}
pauseButton.addEventListener('click', () => {
  paused = !paused;
  syncPauseButton();
  sendControl(paused ? 'pause' : 'run');
});

stopButton.addEventListener('click', () => {
  paused = true;
  simTime = 0;
  syncPauseButton();
  unloadingScene.reset();
  resetAgentStats();
  sendControl('stop');
  setConnection('stopped', 'SIMULATION // STOPPED');
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
  camera.radius = 32;
  camera.target.set(0, 0, 0);
});

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement | null;
  if (target?.matches('input:not([type="range"]), select, textarea, [contenteditable="true"]')) return;
  const key = event.key.toLowerCase();
  if (!['w', 'a', 's', 'd'].includes(key)) return;
  cameraKeys.add(key);
  event.preventDefault();
});

window.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  if (!['w', 'a', 's', 'd'].includes(key)) return;
  cameraKeys.delete(key);
  event.preventDefault();
});

window.addEventListener('blur', () => cameraKeys.clear());

window.addEventListener('resize', () => engine.resize());
connect();
