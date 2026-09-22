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
import { TransformNode } from '@babylonjs/core/Meshes/transformNode';
import './style.css';

const MAP_WIDTH = 44;
const MAP_DEPTH = 32;
const CELL_SIZE = 1.056;
const PALLET_FOOTPRINT = 0.792;
const MAX_AGENTS = 100;
const UNLOAD_X = MAP_WIDTH - 3;
const UNLOAD_MIN_Z = 1;
const UNLOAD_MAX_Z = MAP_DEPTH - 2;
const UNLOAD_STATION_COUNT = UNLOAD_MAX_Z - UNLOAD_MIN_Z + 1;
const RELOAD_X = 2;
const RELOAD_MIN_Z = 1;
const RELOAD_MAX_Z = MAP_DEPTH - 2;
const RELOAD_STATION_COUNT = RELOAD_MAX_Z - RELOAD_MIN_Z + 1;
const REPAIR_X = 22;
const REPAIR_Z = MAP_DEPTH - 2;
const TOW_DEPOT_X = REPAIR_X - 1;
const TOW_DEPOT_Z = REPAIR_Z;
const PALLET_CAPACITY = 12;
const PALLET_DWELL_TICKS = 2;
const TOW_LOADING_TICKS = 3;
const UNLOAD_DWELL_TICKS = 5;
const RELOAD_DWELL_TICKS = 5;
const GOODS_BASE_Y = 1.04;
const GOODS_SLOT_X = [-0.27, -0.09, 0.09, 0.27];
const GOODS_SLOT_Z = [-0.22, 0, 0.22];
const CONVEYOR_X = MAP_WIDTH - 1;
const CONVEYOR_BOX_CAPACITY = 18;
const CONVEYOR_BOX_COUNT = 28;
const CONVEYOR_START_Z = 0.25;
const CONVEYOR_END_Z = MAP_DEPTH - 1.25;
const CONVEYOR_HORIZONTAL_TICKS = 305;
const CONVEYOR_DESCENT_TICKS = 10;
const CONVEYOR_CYCLE_TICKS = CONVEYOR_HORIZONTAL_TICKS + CONVEYOR_DESCENT_TICKS;
const CONVEYOR_BOX_BASE_Y = 0.18;
const MAX_RENDERED_GOODS = MAX_AGENTS * PALLET_CAPACITY
  + CONVEYOR_BOX_COUNT * CONVEYOR_BOX_CAPACITY
  + UNLOAD_STATION_COUNT
  + RELOAD_STATION_COUNT;
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
  reloadPositions: Int16Array;
  palletItems: Uint8Array;
  recoveryStates: Uint8Array;
  palletPositions: Int16Array;
  tow: TowFrame;
  completedTasks: number;
  step: number;
  receivedAt: number;
};

type TowFrame = {
  x: number;
  z: number;
  state: number;
  targetAgent: number;
  queuedRescues: number;
};

type PalletCell = { id: number; x: number; z: number; cargoType: number };
type GoodsTransfer = { gridX: number; gridZ: number; worldY: number; cargoType: number };
type ConveyorPlacement = { boxId: number; generation: number; slot: number };
type ConveyorController = {
  reserve: (stationZ: number, cargoType: number, startedStep: number) => ConveyorPlacement;
  slotTarget: (placement: ConveyorPlacement, simulationStep: number) => Vector3;
  update: (simulationStep: number) => GoodsTransfer[];
  reset: () => void;
};
type PendingHandoff = {
  taskId: number;
  palletId: number;
  stationZ: number;
  itemsBefore: number;
};
type PendingReload = {
  taskId: number;
  stationZ: number;
};
type PalletMotion = {
  kind: 'pickup' | 'drop';
  taskId: number;
  palletId: number;
  startedStep: number;
};
type PendingPalletMotion = Omit<PalletMotion, 'startedStep'>;
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

const camera = new ArcRotateCamera('camera', -Math.PI / 4, 1.02, 37.5, new Vector3(0, 0, 0), scene);
camera.attachControl(canvas, true);
camera.lowerRadiusLimit = 12;
camera.upperRadiusLimit = 68;
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
const palletInventory = new Uint8Array(palletCells.length);
palletInventory.fill(PALLET_CAPACITY);
const palletScene = createWarehousePallets(palletCells);
const conveyorScene = createConveyor();
const unloadingScene = createUnloadingZone(conveyorScene);
const reloadingScene = createReloadingZone();
createRepairStation();
const towTruckScene = createTowTruck();
createBoundaryLights();

const robotMesh = createRobotMesh();
const robotLoad = createCarriedPalletMeshes();
const agentPicker = createAgentPicker();
const selectionHalo = createSelectionHalo();
const taskMarkers = createTaskMarkers();
const matrices = new Float32Array(MAX_AGENTS * 16);
const loadMatrices = new Float32Array(MAX_AGENTS * 16);
const cargoMatrices = [
  new Float32Array(MAX_RENDERED_GOODS * 16),
  new Float32Array(MAX_RENDERED_GOODS * 16),
  new Float32Array(MAX_RENDERED_GOODS * 16),
];
const loadAgentIds = new Int16Array(MAX_AGENTS);
const cargoAgentIds = [
  new Int16Array(MAX_RENDERED_GOODS),
  new Int16Array(MAX_RENDERED_GOODS),
  new Int16Array(MAX_RENDERED_GOODS),
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
let towLoadingStartedStep = 0;
let lastInferenceMs = 0;
let hiddenPalletKey = '';
let palletInventoryRevision = 0;
let selectedAgentId = -1;
const pendingHandoffs: Array<PendingHandoff | null> = Array.from(
  { length: MAX_AGENTS },
  () => null,
);
const pendingReloads: Array<PendingReload | null> = Array.from(
  { length: MAX_AGENTS },
  () => null,
);
const pendingPalletMotions: Array<PendingPalletMotion | null> = Array.from(
  { length: MAX_AGENTS },
  () => null,
);
const palletMotions: Array<PalletMotion | null> = Array.from(
  { length: MAX_AGENTS },
  () => null,
);
const agentStats: AgentStats[] = Array.from({ length: MAX_AGENTS }, () => createAgentStats());
const cameraKeys = new Set<string>();
const cameraKeyByCode: Readonly<Record<string, string>> = {
  KeyW: 'w',
  KeyA: 'a',
  KeyS: 's',
  KeyD: 'd',
};

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

function createTaskMarkers(): { pallet: Mesh; station: Mesh; reload: Mesh } {
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

  const reload = MeshBuilder.CreateBox('selected-task-reload', {
    width: CELL_SIZE * 0.88,
    height: 0.045,
    depth: CELL_SIZE * 0.88,
  }, scene);
  const reloadMaterial = new StandardMaterial('selected-task-reload-material', scene);
  reloadMaterial.diffuseColor = Color3.FromHexString('#5bffa4');
  reloadMaterial.emissiveColor = Color3.FromHexString('#20b86b');
  reloadMaterial.alpha = 0.58;
  reloadMaterial.disableLighting = true;
  reload.material = reloadMaterial;
  reload.isPickable = false;
  reload.isVisible = false;
  reload.alwaysSelectAsActiveMesh = true;

  const markerGlow = new GlowLayer('task-marker-glow', scene, { blurKernelSize: 22 });
  markerGlow.intensity = 0.72;
  markerGlow.addIncludedOnlyMesh(pallet);
  markerGlow.addIncludedOnlyMesh(station);
  markerGlow.addIncludedOnlyMesh(reload);
  return { pallet, station, reload };
}

function hideTaskMarkers(): void {
  taskMarkers.pallet.isVisible = false;
  taskMarkers.station.isVisible = false;
  taskMarkers.reload.isVisible = false;
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
  const palletX = frame.palletPositions[id * 2];
  const palletZ = frame.palletPositions[id * 2 + 1];
  const stationX = frame.stationPositions[id * 2];
  const stationZ = frame.stationPositions[id * 2 + 1];
  const reloadX = frame.reloadPositions[id * 2];
  const reloadZ = frame.reloadPositions[id * 2 + 1];
  if (!pallet || stationX < 0 || stationZ < 0) {
    hideTaskMarkers();
    return;
  }

  if ((frame.statuses[id] & 2) !== 0) {
    taskMarkers.pallet.position.set(matrices[id * 16 + 12], 1.02, matrices[id * 16 + 14]);
  } else {
    taskMarkers.pallet.position.copyFrom(worldAt(
      palletX >= 0 ? palletX : pallet.x,
      palletZ >= 0 ? palletZ : pallet.z,
      1.02,
    ));
  }
  taskMarkers.station.position.copyFrom(worldAt(stationX, stationZ, 0.065));
  if (reloadX >= 0 && reloadZ >= 0) {
    taskMarkers.reload.position.copyFrom(worldAt(reloadX, reloadZ, 0.065));
  }
  const palletPulse = 1 + Math.sin(now * 0.009) * 0.07;
  const stationPulse = 1 + Math.sin(now * 0.007 + Math.PI / 2) * 0.055;
  taskMarkers.pallet.scaling.set(palletPulse, palletPulse, palletPulse);
  taskMarkers.station.scaling.set(stationPulse, 1, stationPulse);
  taskMarkers.reload.scaling.set(stationPulse, 1, stationPulse);
  taskMarkers.pallet.isVisible = true;
  taskMarkers.station.isVisible = true;
  taskMarkers.reload.isVisible = reloadX >= 0 && reloadZ >= 0
    && (((frame.statuses[id] & 8) !== 0) || frame.palletItems[id] <= 1);
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

function goodsSlotOffset(slot: number): { x: number; z: number } {
  return {
    x: GOODS_SLOT_X[slot % GOODS_SLOT_X.length],
    z: GOODS_SLOT_Z[Math.floor(slot / GOODS_SLOT_X.length)],
  };
}

function goodsTransform(
  gridX: number,
  gridZ: number,
  palletLift: number,
  slot: number,
  extraHeight = 0,
): Matrix {
  const center = worldAt(gridX, gridZ, GOODS_BASE_Y + palletLift + extraHeight);
  const offset = goodsSlotOffset(slot);
  return Matrix.Translation(center.x + offset.x, center.y, center.z + offset.z);
}

function createPalletVisual(prefix: string): { frame: Mesh; deck: Mesh } {
  const footprint = PALLET_FOOTPRINT;
  const half = footprint / 2;
  const legOffset = half - 0.09;
  const frameParts: Mesh[] = [];
  for (const x of [-legOffset, legOffset]) {
    for (const z of [-legOffset, legOffset]) {
      const leg = MeshBuilder.CreateBox(prefix + '-pallet-leg', {
        width: 0.1,
        height: 0.7,
        depth: 0.1,
      }, scene);
      leg.position.set(x, 0.38, z);
      frameParts.push(leg);
      const foot = MeshBuilder.CreateBox(prefix + '-pallet-foot', {
        width: 0.17,
        height: 0.08,
        depth: 0.17,
      }, scene);
      foot.position.set(x, 0.04, z);
      frameParts.push(foot);
    }
  }
  for (const z of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox(prefix + '-pallet-rail-x', {
      width: footprint,
      height: 0.1,
      depth: 0.1,
    }, scene);
    rail.position.set(0, 0.78, z);
    frameParts.push(rail);
  }
  for (const x of [-half + 0.055, half - 0.055]) {
    const rail = MeshBuilder.CreateBox(prefix + '-pallet-rail-z', {
      width: 0.1,
      height: 0.1,
      depth: footprint,
    }, scene);
    rail.position.set(x, 0.78, 0);
    frameParts.push(rail);
  }
  const frame = Mesh.MergeMeshes(frameParts, true, true, undefined, false, true)!;
  const frameMaterial = new StandardMaterial(prefix + '-pallet-frame-material', scene);
  frameMaterial.diffuseColor = Color3.FromHexString('#17313e');
  frameMaterial.emissiveColor = Color3.FromHexString('#07151e');
  frameMaterial.specularColor = Color3.FromHexString('#315d70');
  frame.material = frameMaterial;

  const deckParts: Mesh[] = [];
  for (const z of [-0.29, -0.1, 0.1, 0.29]) {
    const slat = MeshBuilder.CreateBox(prefix + '-pallet-slat', {
      width: footprint,
      height: 0.09,
      depth: 0.13,
    }, scene);
    slat.position.set(0, 0.88, z);
    deckParts.push(slat);
  }
  const deck = Mesh.MergeMeshes(deckParts, true, true, undefined, false, true)!;
  const deckMaterial = new StandardMaterial(prefix + '-pallet-deck-material', scene);
  deckMaterial.diffuseColor = Color3.FromHexString('#2d8098');
  deckMaterial.emissiveColor = Color3.FromHexString('#0b3442');
  deckMaterial.specularColor = Color3.FromHexString('#7bdcf2');
  deckMaterial.specularPower = 64;
  deck.material = deckMaterial;
  return { frame, deck };
}

function createGoodsMeshes(prefix: string): [Mesh, Mesh, Mesh] {
  const crateParts: Mesh[] = [];
  const crateBody = MeshBuilder.CreateBox(prefix + '-goods-crate', {
    width: 0.145,
    height: 0.18,
    depth: 0.16,
  }, scene);
  crateParts.push(crateBody);
  for (const z of [-0.068, 0.068]) {
    const brace = MeshBuilder.CreateBox(prefix + '-goods-crate-brace', {
      width: 0.16,
      height: 0.025,
      depth: 0.018,
    }, scene);
    brace.position.set(0, 0.04, z);
    crateParts.push(brace);
  }
  const crate = Mesh.MergeMeshes(crateParts, true, true, undefined, false, true)!;
  const crateMaterial = new StandardMaterial(prefix + '-goods-crate-material', scene);
  crateMaterial.diffuseColor = Color3.FromHexString('#d98136');
  crateMaterial.emissiveColor = Color3.FromHexString('#321707');
  crate.material = crateMaterial;

  const cartonParts: Mesh[] = [];
  const cartonBody = MeshBuilder.CreateBox(prefix + '-goods-carton', {
    width: 0.15,
    height: 0.17,
    depth: 0.165,
  }, scene);
  cartonParts.push(cartonBody);
  const tape = MeshBuilder.CreateBox(prefix + '-goods-carton-tape', {
    width: 0.025,
    height: 0.176,
    depth: 0.171,
  }, scene);
  cartonParts.push(tape);
  const carton = Mesh.MergeMeshes(cartonParts, true, true, undefined, false, true)!;
  const cartonMaterial = new StandardMaterial(prefix + '-goods-carton-material', scene);
  cartonMaterial.diffuseColor = Color3.FromHexString('#e4b477');
  cartonMaterial.emissiveColor = Color3.FromHexString('#39220e');
  carton.material = cartonMaterial;

  const drumParts: Mesh[] = [];
  const drumBody = MeshBuilder.CreateCylinder(prefix + '-goods-drum', {
    height: 0.18,
    diameter: 0.14,
    tessellation: 12,
  }, scene);
  drumParts.push(drumBody);
  for (const y of [-0.075, 0.075]) {
    const ring = MeshBuilder.CreateTorus(prefix + '-goods-drum-ring', {
      diameter: 0.142,
      thickness: 0.012,
      tessellation: 12,
    }, scene);
    ring.position.y = y;
    drumParts.push(ring);
  }
  const drum = Mesh.MergeMeshes(drumParts, true, true, undefined, false, true)!;
  const drumMaterial = new StandardMaterial(prefix + '-goods-drum-material', scene);
  drumMaterial.diffuseColor = Color3.FromHexString('#36a0ad');
  drumMaterial.emissiveColor = Color3.FromHexString('#092b32');
  drumMaterial.specularColor = Color3.FromHexString('#8ee9ef');
  drum.material = drumMaterial;

  const goods: [Mesh, Mesh, Mesh] = [crate, carton, drum];
  for (const mesh of goods) mesh.alwaysSelectAsActiveMesh = true;
  return goods;
}

function createWarehousePallets(pallets: PalletCell[]): {
  update: (
    hidden: Set<number>,
    inventory: Uint8Array,
    overrides?: Map<number, { x: number; z: number }>,
  ) => void;
} {
  const { frame, deck } = createPalletVisual('warehouse');
  const cargo = createGoodsMeshes('warehouse');
  for (const mesh of [frame, deck, ...cargo]) mesh.alwaysSelectAsActiveMesh = true;

  const update = (
    hidden: Set<number>,
    inventory: Uint8Array,
    overrides = new Map<number, { x: number; z: number }>(),
  ) => {
    const palletTransforms: number[] = [];
    const goodsTransforms: number[][] = [[], [], []];
    for (const pallet of pallets) {
      if (hidden.has(pallet.id)) continue;
      const position = overrides.get(pallet.id) ?? pallet;
      const transform = gridTransform(position.x, position.z);
      transform.copyToArray(palletTransforms, palletTransforms.length);
      const quantity = Math.min(PALLET_CAPACITY, inventory[pallet.id] ?? PALLET_CAPACITY);
      for (let slot = 0; slot < quantity; slot++) {
        goodsTransform(position.x, position.z, 0, slot)
          .copyToArray(goodsTransforms[pallet.cargoType], goodsTransforms[pallet.cargoType].length);
      }
    }
    const setInstances = (mesh: Mesh, values: number[]) => {
      mesh.thinInstanceSetBuffer('matrix', new Float32Array(values), 16, false);
      mesh.thinInstanceCount = values.length / 16;
    };
    setInstances(frame, palletTransforms);
    setInstances(deck, palletTransforms);
    for (let type = 0; type < cargo.length; type++) {
      setInstances(cargo[type], goodsTransforms[type]);
    }
  };
  update(new Set(), palletInventory);
  return { update };
}

function createCarriedPalletMeshes(): {
  frame: Mesh;
  deck: Mesh;
  cargo: [Mesh, Mesh, Mesh];
} {
  const { frame, deck } = createPalletVisual('carried');
  const cargo = createGoodsMeshes('carried');
  for (const mesh of [frame, deck, ...cargo]) mesh.alwaysSelectAsActiveMesh = true;
  return { frame, deck, cargo };
}

function createConveyor(): ConveyorController {
  type ConveyorItem = { cargoType: number; visibleAtStep: number };
  type ConveyorPose = {
    generation: number;
    phaseTick: number;
    gridZ: number;
    worldY: number;
  };

  const beltLength = (CONVEYOR_END_Z - CONVEYOR_START_Z) * CELL_SIZE + 0.9;
  const beltCenterZ = (CONVEYOR_START_Z + CONVEYOR_END_Z) / 2;
  const beltCenter = worldAt(CONVEYOR_X, beltCenterZ, 0.09);
  const belt = MeshBuilder.CreateBox('outbound-conveyor-belt', {
    width: 0.9,
    height: 0.16,
    depth: beltLength,
  }, scene);
  const beltMaterial = new StandardMaterial('outbound-conveyor-belt-material', scene);
  beltMaterial.diffuseColor = Color3.FromHexString('#18272d');
  beltMaterial.emissiveColor = Color3.FromHexString('#071217');
  beltMaterial.specularColor = Color3.FromHexString('#66828a');
  beltMaterial.specularPower = 56;
  belt.material = beltMaterial;
  belt.position.copyFrom(beltCenter);

  const rail = MeshBuilder.CreateBox('outbound-conveyor-rail', {
    width: 0.055,
    height: 0.24,
    depth: beltLength,
  }, scene);
  const railMaterial = new StandardMaterial('outbound-conveyor-rail-material', scene);
  railMaterial.diffuseColor = Color3.FromHexString('#51636a');
  railMaterial.emissiveColor = Color3.FromHexString('#17242a');
  railMaterial.specularColor = Color3.FromHexString('#a8c0c6');
  rail.material = railMaterial;
  const railTransforms: number[] = [];
  for (const offsetX of [-0.49, 0.49]) {
    Matrix.Translation(beltCenter.x + offsetX, 0.18, beltCenter.z)
      .copyToArray(railTransforms, railTransforms.length);
  }
  rail.thinInstanceSetBuffer('matrix', new Float32Array(railTransforms), 16, true);

  const roller = MeshBuilder.CreateCylinder('outbound-conveyor-roller', {
    height: 0.83,
    diameter: 0.11,
    tessellation: 12,
  }, scene);
  roller.material = railMaterial;
  const rollerTransforms: number[] = [];
  const rollerRotation = Quaternion.RotationAxis(Vector3.Forward(), Math.PI / 2);
  for (let z = CONVEYOR_START_Z; z <= CONVEYOR_END_Z; z += 0.8) {
    const rollerPosition = worldAt(CONVEYOR_X, z, 0.19);
    Matrix.Compose(Vector3.One(), rollerRotation, rollerPosition)
      .copyToArray(rollerTransforms, rollerTransforms.length);
  }
  roller.thinInstanceSetBuffer('matrix', new Float32Array(rollerTransforms), 16, true);

  const shaft = MeshBuilder.CreateBox('outbound-conveyor-drop-shaft', {
    width: 1.05,
    height: 1.25,
    depth: 1.05,
  }, scene);
  const shaftMaterial = new StandardMaterial('outbound-conveyor-drop-shaft-material', scene);
  shaftMaterial.diffuseColor = Color3.FromHexString('#071116');
  shaftMaterial.emissiveColor = Color3.FromHexString('#020709');
  shaft.material = shaftMaterial;
  shaft.position.copyFrom(worldAt(CONVEYOR_X, CONVEYOR_END_Z, -0.62));

  const boxParts: Mesh[] = [];
  const boxFloor = MeshBuilder.CreateBox('conveyor-box-floor', {
    width: 0.84,
    height: 0.06,
    depth: 0.84,
  }, scene);
  boxFloor.position.y = 0.03;
  boxParts.push(boxFloor);
  for (const x of [-0.405, 0.405]) {
    const wall = MeshBuilder.CreateBox('conveyor-box-wall-x', {
      width: 0.055,
      height: 0.5,
      depth: 0.84,
    }, scene);
    wall.position.set(x, 0.28, 0);
    boxParts.push(wall);
  }
  for (const z of [-0.405, 0.405]) {
    const wall = MeshBuilder.CreateBox('conveyor-box-wall-z', {
      width: 0.75,
      height: 0.5,
      depth: 0.055,
    }, scene);
    wall.position.set(0, 0.28, z);
    boxParts.push(wall);
  }
  const box = Mesh.MergeMeshes(boxParts, true, true, undefined, false, true)!;
  box.name = 'conveyor-box';
  const boxMaterial = new StandardMaterial('conveyor-box-material', scene);
  boxMaterial.diffuseColor = Color3.FromHexString('#bd7b3d');
  boxMaterial.emissiveColor = Color3.FromHexString('#351b0a');
  boxMaterial.specularColor = Color3.FromHexString('#d8a36d');
  box.material = boxMaterial;
  box.alwaysSelectAsActiveMesh = true;
  belt.alwaysSelectAsActiveMesh = true;
  rail.alwaysSelectAsActiveMesh = true;
  roller.alwaysSelectAsActiveMesh = true;

  const boxMatrices = new Float32Array(CONVEYOR_BOX_COUNT * 16);
  box.thinInstanceSetBuffer('matrix', boxMatrices, 16, false);
  box.thinInstanceCount = CONVEYOR_BOX_COUNT;
  const contents = Array.from(
    { length: CONVEYOR_BOX_COUNT },
    () => new Map<number, Array<ConveyorItem | null>>(),
  );

  const poseAt = (boxId: number, simulationStep: number): ConveyorPose => {
    const phaseOffset = boxId * CONVEYOR_CYCLE_TICKS / CONVEYOR_BOX_COUNT;
    const cyclePosition = simulationStep + phaseOffset;
    const generation = Math.floor(cyclePosition / CONVEYOR_CYCLE_TICKS);
    const phaseTick = mod(cyclePosition, CONVEYOR_CYCLE_TICKS);
    if (phaseTick < CONVEYOR_HORIZONTAL_TICKS) {
      return {
        generation,
        phaseTick,
        gridZ: CONVEYOR_START_Z
          + (CONVEYOR_END_Z - CONVEYOR_START_Z) * phaseTick / CONVEYOR_HORIZONTAL_TICKS,
        worldY: CONVEYOR_BOX_BASE_Y,
      };
    }
    const descent = smoothstep(
      (phaseTick - CONVEYOR_HORIZONTAL_TICKS) / CONVEYOR_DESCENT_TICKS,
    );
    return {
      generation,
      phaseTick,
      gridZ: CONVEYOR_END_Z,
      worldY: CONVEYOR_BOX_BASE_Y - descent * 1.5,
    };
  };

  const contentsFor = (boxId: number, generation: number): Array<ConveyorItem | null> => {
    const generations = contents[boxId];
    let items = generations.get(generation);
    if (!items) {
      items = Array.from({ length: CONVEYOR_BOX_CAPACITY }, () => null);
      generations.set(generation, items);
    }
    return items;
  };

  const itemOffset = (slot: number): Vector3 => {
    const column = slot % 3;
    const row = Math.floor(slot / 3) % 3;
    const layer = Math.floor(slot / 9);
    return new Vector3(
      (column - 1) * 0.225,
      0.17 + layer * 0.2,
      (row - 1) * 0.225,
    );
  };

  const slotTarget = (placement: ConveyorPlacement, simulationStep: number): Vector3 => {
    const pose = poseAt(placement.boxId, simulationStep);
    const center = worldAt(CONVEYOR_X, pose.gridZ, pose.worldY);
    return center.add(itemOffset(placement.slot));
  };

  const reserve = (
    stationZ: number,
    cargoType: number,
    startedStep: number,
  ): ConveyorPlacement => {
    const depositStep = startedStep + UNLOAD_DWELL_TICKS * 0.9;
    let best: { boxId: number; generation: number; slot: number; distance: number } | null = null;
    for (let boxId = 0; boxId < CONVEYOR_BOX_COUNT; boxId++) {
      const startPose = poseAt(boxId, startedStep);
      const depositPose = poseAt(boxId, depositStep);
      if (startPose.generation !== depositPose.generation
          || depositPose.phaseTick >= CONVEYOR_HORIZONTAL_TICKS) continue;
      const items = contentsFor(boxId, depositPose.generation);
      const slot = items.findIndex((item) => item === null);
      if (slot < 0) continue;
      const distance = Math.abs(depositPose.gridZ - stationZ);
      if (!best || distance < best.distance) {
        best = { boxId, generation: depositPose.generation, slot, distance };
      }
    }
    if (!best) {
      const boxId = 0;
      const generation = poseAt(boxId, depositStep).generation;
      const items = contentsFor(boxId, generation);
      const slot = Math.max(0, items.findIndex((item) => item === null));
      best = { boxId, generation, slot, distance: 0 };
    }
    contentsFor(best.boxId, best.generation)[best.slot] = {
      cargoType,
      visibleAtStep: depositStep,
    };
    return { boxId: best.boxId, generation: best.generation, slot: best.slot };
  };

  const update = (simulationStep: number): GoodsTransfer[] => {
    const transfers: GoodsTransfer[] = [];
    for (let boxId = 0; boxId < CONVEYOR_BOX_COUNT; boxId++) {
      const pose = poseAt(boxId, simulationStep);
      const center = worldAt(CONVEYOR_X, pose.gridZ, pose.worldY);
      Matrix.Translation(center.x, center.y, center.z)
        .copyToArray(boxMatrices, boxId * 16);
      const generations = contents[boxId];
      for (const generation of generations.keys()) {
        if (generation < pose.generation) generations.delete(generation);
      }
      const items = generations.get(pose.generation);
      if (!items) continue;
      for (let slot = 0; slot < items.length; slot++) {
        const item = items[slot];
        if (!item || simulationStep < item.visibleAtStep) continue;
        const target = center.add(itemOffset(slot));
        transfers.push({
          gridX: (target.x + MAP_WIDTH * CELL_SIZE / 2) / CELL_SIZE - 0.5,
          gridZ: (target.z + MAP_DEPTH * CELL_SIZE / 2) / CELL_SIZE - 0.5,
          worldY: target.y,
          cargoType: item.cargoType,
        });
      }
    }
    box.thinInstanceBufferUpdated('matrix');
    return transfers;
  };

  update(0);
  return {
    reserve,
    slotTarget,
    update,
    reset: () => {
      for (const generations of contents) generations.clear();
    },
  };
}

function createUnloadingZone(conveyor: ConveyorController): {
  trigger: (stationZ: number, cargoType: number, slot: number, startedStep: number) => void;
  update: (simulationStep: number) => GoodsTransfer[];
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

  type ArmEvent = {
    startedStep: number;
    cargoType: number;
    slot: number;
    placement: ConveyorPlacement;
  };
  type Point = { x: number; y: number; z: number };
  type Pose = { elbow: Point; grip: Point };
  const events = new Map<number, ArmEvent>();
  const mix = (a: number, b: number, t: number): number => a + (b - a) * smoothstep(t);
  const mixPoint = (a: Point, b: Point, t: number): Point => ({
    x: mix(a.x, b.x, t),
    y: mix(a.y, b.y, t),
    z: mix(a.z, b.z, t),
  });
  const mixPose = (a: Pose, b: Pose, t: number): Pose => ({
    elbow: mixPoint(a.elbow, b.elbow, t),
    grip: mixPoint(a.grip, b.grip, t),
  });
  const segmentMatrix = (from: Point, to: Point): Matrix => {
    const direction = new Vector3(to.x - from.x, to.y - from.y, to.z - from.z);
    const length = direction.length();
    const normalized = direction.scale(1 / Math.max(length, 1e-6));
    const localAxis = new Vector3(1, 0, 0);
    const dot = Math.max(-1, Math.min(1, Vector3.Dot(localAxis, normalized)));
    const rotationAxis = Vector3.Cross(localAxis, normalized);
    const rotation = rotationAxis.lengthSquared() > 1e-8
      ? Quaternion.RotationAxis(rotationAxis.normalize(), Math.acos(dot))
      : dot < 0
        ? Quaternion.RotationAxis(Vector3.Up(), Math.PI)
        : Quaternion.Identity();
    return Matrix.Compose(
      new Vector3(length, 1, 1),
      rotation,
      new Vector3((from.x + to.x) / 2, (from.y + to.y) / 2, (from.z + to.z) / 2),
    );
  };

  const update = (simulationStep: number): GoodsTransfer[] => {
    const transfers: GoodsTransfer[] = [];
    for (let index = 0; index < UNLOAD_STATION_COUNT; index++) {
      const stationZ = UNLOAD_MIN_Z + index;
      const stationWorld = worldAt(UNLOAD_X, stationZ, 0);
      const armWorld = worldAt(MAP_WIDTH - 2, stationZ, 0);
      const event = events.get(stationZ);
      const slotOffset = event ? goodsSlotOffset(event.slot) : { x: 0, z: 0 };
      const conveyorTarget = event
        ? conveyor.slotTarget(event.placement, simulationStep)
        : worldAt(CONVEYOR_X, stationZ, CONVEYOR_BOX_BASE_Y + 0.17);
      const shoulderPoint = { x: armWorld.x, y: 0.78, z: armWorld.z };
      const parked: Pose = {
        elbow: { x: armWorld.x - 0.05, y: 1.38, z: armWorld.z },
        grip: { x: armWorld.x + 0.11, y: 1.88, z: armWorld.z },
      };
      const reached: Pose = {
        elbow: {
          x: armWorld.x - 0.43 + slotOffset.x * 0.2,
          y: 1.4,
          z: armWorld.z + slotOffset.z * 0.55,
        },
        grip: {
          x: stationWorld.x + slotOffset.x,
          y: GOODS_BASE_Y + 0.26,
          z: stationWorld.z + slotOffset.z,
        },
      };
      const lifted: Pose = {
        elbow: { x: armWorld.x - 0.19, y: 1.5, z: armWorld.z + slotOffset.z * 0.25 },
        grip: { x: armWorld.x - 0.35, y: 2.0, z: armWorld.z + slotOffset.z * 0.4 },
      };
      const dropped: Pose = {
        elbow: {
          x: armWorld.x + 0.48,
          y: 1.28,
          z: armWorld.z + (conveyorTarget.z - armWorld.z) * 0.45,
        },
        grip: {
          x: conveyorTarget.x,
          y: conveyorTarget.y + 0.16,
          z: conveyorTarget.z,
        },
      };
      const progress = event
        ? Math.max(0, (simulationStep - event.startedStep) / UNLOAD_DWELL_TICKS)
        : 0;
      let pose = parked;
      if (event) {
        if (progress < 0.32) pose = mixPose(parked, reached, progress / 0.32);
        else if (progress < 0.42) pose = reached;
        else if (progress < 0.66) pose = mixPose(reached, lifted, (progress - 0.42) / 0.24);
        else if (progress < 0.84) pose = mixPose(lifted, dropped, (progress - 0.66) / 0.18);
        else if (progress < 0.91) pose = dropped;
        else pose = mixPose(dropped, parked, (progress - 0.91) / 0.09);

        if (progress < 0.9) {
          let cargoGridX = UNLOAD_X + slotOffset.x / CELL_SIZE;
          let cargoGridZ = stationZ + slotOffset.z / CELL_SIZE;
          let cargoY = GOODS_BASE_Y + 0.1;
          if (progress >= 0.38) {
            cargoGridX = (pose.grip.x + MAP_WIDTH * CELL_SIZE / 2) / CELL_SIZE - 0.5;
            cargoGridZ = (pose.grip.z + MAP_DEPTH * CELL_SIZE / 2) / CELL_SIZE - 0.5;
            cargoY = pose.grip.y - 0.16;
          }
          transfers.push({ gridX: cargoGridX, gridZ: cargoGridZ, worldY: cargoY, cargoType: event.cargoType });
        }
        if (progress >= 1) events.delete(stationZ);
      }

      segmentMatrix(shoulderPoint, pose.elbow).copyToArray(upperMatrices, index * 16);
      segmentMatrix(pose.elbow, pose.grip).copyToArray(forearmMatrices, index * 16);
      Matrix.Translation(pose.elbow.x, pose.elbow.y, pose.elbow.z).copyToArray(jointMatrices, index * 32);
      Matrix.Translation(pose.grip.x, pose.grip.y, pose.grip.z).copyToArray(jointMatrices, index * 32 + 16);
      const closeProgress = event
        ? Math.min(1, Math.max(0, (progress - 0.28) / 0.1))
        : 0;
      const openProgress = event
        ? Math.min(1, Math.max(0, (progress - 0.84) / 0.06))
        : 0;
      const gripScale = 1 - 0.32 * closeProgress * (1 - openProgress);
      Matrix.Compose(
        new Vector3(1, 1, gripScale),
        Quaternion.Identity(),
        new Vector3(pose.grip.x, pose.grip.y - 0.03, pose.grip.z),
      ).copyToArray(gripperMatrices, index * 16);
    }
    upper.thinInstanceBufferUpdated('matrix');
    forearm.thinInstanceBufferUpdated('matrix');
    movingJoint.thinInstanceBufferUpdated('matrix');
    gripper.thinInstanceBufferUpdated('matrix');
    return transfers;
  };

  update(0);
  return {
    trigger: (stationZ, cargoType, slot, startedStep) => {
      if (stationZ < UNLOAD_MIN_Z || stationZ > UNLOAD_MAX_Z) return;
      events.set(stationZ, {
        startedStep,
        cargoType,
        slot,
        placement: conveyor.reserve(stationZ, cargoType, startedStep),
      });
    },
    update,
    reset: () => events.clear(),
  };
}

function createReloadingZone(): {
  trigger: (stationZ: number, cargoType: number, startedStep: number) => void;
  update: (simulationStep: number) => GoodsTransfer[];
  reset: () => void;
} {
  const pad = MeshBuilder.CreateBox('reloading-pad', {
    width: CELL_SIZE * 0.82,
    height: 0.035,
    depth: CELL_SIZE * 0.82,
  }, scene);
  const padMaterial = new StandardMaterial('reloading-pad-material', scene);
  padMaterial.diffuseColor = Color3.FromHexString('#205447');
  padMaterial.emissiveColor = Color3.FromHexString('#0f624b');
  pad.material = padMaterial;
  const padTransforms: number[] = [];
  for (let z = RELOAD_MIN_Z; z <= RELOAD_MAX_Z; z++) {
    const transform = gridTransform(RELOAD_X, z);
    transform.setTranslation(transform.getTranslation().add(new Vector3(0, 0.02, 0)));
    transform.copyToArray(padTransforms, padTransforms.length);
  }
  pad.thinInstanceSetBuffer('matrix', new Float32Array(padTransforms), 16, true);

  const serviceStrip = MeshBuilder.CreateBox('reloading-service-strip', {
    width: CELL_SIZE * 0.86,
    height: 0.035,
    depth: RELOAD_STATION_COUNT * CELL_SIZE,
  }, scene);
  const serviceMaterial = new StandardMaterial('reloading-service-material', scene);
  serviceMaterial.diffuseColor = Color3.FromHexString('#142d28');
  serviceMaterial.emissiveColor = Color3.FromHexString('#0b211b');
  serviceStrip.material = serviceMaterial;
  serviceStrip.position.copyFrom(worldAt(1, (RELOAD_MIN_Z + RELOAD_MAX_Z) / 2, -0.01));

  const machineMaterial = new StandardMaterial('reloading-machine-material', scene);
  machineMaterial.diffuseColor = Color3.FromHexString('#39b985');
  machineMaterial.emissiveColor = Color3.FromHexString('#0b3e2c');
  machineMaterial.specularColor = Color3.FromHexString('#9affd3');
  const darkMaterial = new StandardMaterial('reloading-machine-dark-material', scene);
  darkMaterial.diffuseColor = Color3.FromHexString('#213a3b');
  darkMaterial.emissiveColor = Color3.FromHexString('#0a1819');

  const pillar = MeshBuilder.CreateBox('reloading-pillar', {
    width: 0.12,
    height: 1.65,
    depth: 0.12,
  }, scene);
  pillar.material = machineMaterial;
  const magazine = MeshBuilder.CreateBox('reloading-magazine', {
    width: 0.56,
    height: 0.34,
    depth: 0.48,
  }, scene);
  magazine.material = darkMaterial;
  const head = MeshBuilder.CreateCylinder('reloading-head', {
    height: 0.3,
    diameterTop: 0.2,
    diameterBottom: 0.3,
    tessellation: 12,
  }, scene);
  head.material = machineMaterial;

  const pillarTransforms: number[] = [];
  const magazineTransforms: number[] = [];
  for (let z = RELOAD_MIN_Z; z <= RELOAD_MAX_Z; z++) {
    const service = worldAt(1, z, 0);
    for (const dz of [-0.3, 0.3]) {
      Matrix.Translation(service.x, 0.83, service.z + dz)
        .copyToArray(pillarTransforms, pillarTransforms.length);
    }
    Matrix.Translation(service.x, 1.76, service.z)
      .copyToArray(magazineTransforms, magazineTransforms.length);
  }
  pillar.thinInstanceSetBuffer('matrix', new Float32Array(pillarTransforms), 16, true);
  magazine.thinInstanceSetBuffer('matrix', new Float32Array(magazineTransforms), 16, true);

  const headMatrices = new Float32Array(RELOAD_STATION_COUNT * 16);
  head.thinInstanceSetBuffer('matrix', headMatrices, 16, false);
  head.thinInstanceCount = RELOAD_STATION_COUNT;
  for (const mesh of [pad, serviceStrip, pillar, magazine, head]) {
    mesh.alwaysSelectAsActiveMesh = true;
  }

  const events = new Map<number, { startedStep: number; cargoType: number }>();
  const update = (simulationStep: number): GoodsTransfer[] => {
    const transfers: GoodsTransfer[] = [];
    for (let index = 0; index < RELOAD_STATION_COUNT; index++) {
      const stationZ = RELOAD_MIN_Z + index;
      const event = events.get(stationZ);
      const progress = event === undefined
        ? 0
        : Math.max(0, (simulationStep - event.startedStep) / RELOAD_DWELL_TICKS);
      const cycle = progress > 0 && progress < 1 ? progress : 0;
      const station = worldAt(RELOAD_X, stationZ, 0);
      const service = worldAt(1, stationZ, 0);
      const reach = Math.sin(cycle * Math.PI);
      const x = service.x + (station.x - service.x) * reach * 0.76;
      const y = 1.55 - reach * 0.31;
      Matrix.Translation(x, y, service.z).copyToArray(headMatrices, index * 16);
      if (event && progress < 1) {
        const transferProgress = smoothstep(Math.max(0, Math.min(1, (progress - 0.08) / 0.76)));
        for (let slot = 0; slot < PALLET_CAPACITY; slot++) {
          const slotOffset = goodsSlotOffset(slot);
          transfers.push({
            gridX: 1 + slotOffset.x / CELL_SIZE + transferProgress,
            gridZ: stationZ + slotOffset.z / CELL_SIZE,
            worldY: 1.55 + (GOODS_BASE_Y + 0.1 - 1.55) * transferProgress,
            cargoType: event.cargoType,
          });
        }
      }
      if (progress >= 1) events.delete(stationZ);
    }
    head.thinInstanceBufferUpdated('matrix');
    return transfers;
  };

  update(0);
  return {
    trigger: (stationZ, cargoType, startedStep) => {
      if (stationZ < RELOAD_MIN_Z || stationZ > RELOAD_MAX_Z) return;
      events.set(stationZ, { startedStep, cargoType });
    },
    update,
    reset: () => events.clear(),
  };
}

function createRepairStation(): void {
  const padMaterial = new StandardMaterial('repair-pad-material', scene);
  padMaterial.diffuseColor = Color3.FromHexString('#173a43');
  padMaterial.emissiveColor = Color3.FromHexString('#0b4b55');
  padMaterial.specularColor = Color3.FromHexString('#67e8f1');

  const repairPad = MeshBuilder.CreateBox('repair-pad', {
    width: CELL_SIZE * 0.9,
    height: 0.055,
    depth: CELL_SIZE * 0.9,
  }, scene);
  repairPad.position.copyFrom(worldAt(REPAIR_X, REPAIR_Z, 0.012));
  repairPad.material = padMaterial;

  const depotPad = repairPad.clone('tow-depot-pad')!;
  depotPad.position.copyFrom(worldAt(TOW_DEPOT_X, TOW_DEPOT_Z, 0.012));

  const structureMaterial = new StandardMaterial('repair-structure-material', scene);
  structureMaterial.diffuseColor = Color3.FromHexString('#263a46');
  structureMaterial.emissiveColor = Color3.FromHexString('#0b1820');
  structureMaterial.specularColor = Color3.FromHexString('#91b7c4');
  const accentMaterial = new StandardMaterial('repair-accent-material', scene);
  accentMaterial.diffuseColor = Color3.FromHexString('#55f0d2');
  accentMaterial.emissiveColor = Color3.FromHexString('#18a995');
  accentMaterial.disableLighting = true;

  const center = worldAt((REPAIR_X + TOW_DEPOT_X) / 2, MAP_DEPTH - 0.35, 0);
  const back = MeshBuilder.CreateBox('repair-shop-back', {
    width: CELL_SIZE * 2.2,
    height: 1.45,
    depth: 0.28,
  }, scene);
  back.position.set(center.x, 0.72, center.z);
  back.material = structureMaterial;
  const roof = MeshBuilder.CreateBox('repair-shop-roof', {
    width: CELL_SIZE * 2.35,
    height: 0.16,
    depth: CELL_SIZE * 1.35,
  }, scene);
  roof.position.set(center.x, 1.48, center.z - CELL_SIZE * 0.4);
  roof.material = structureMaterial;
  for (const x of [TOW_DEPOT_X - 0.42, REPAIR_X + 0.42]) {
    const post = MeshBuilder.CreateBox('repair-shop-post', {
      width: 0.14,
      height: 1.45,
      depth: 0.14,
    }, scene);
    const position = worldAt(x, REPAIR_Z + 0.44, 0.72);
    post.position.copyFrom(position);
    post.material = structureMaterial;
  }
  const sign = MeshBuilder.CreateBox('repair-shop-sign', {
    width: CELL_SIZE * 1.25,
    height: 0.18,
    depth: 0.08,
  }, scene);
  sign.position.set(center.x, 1.18, center.z - 0.18);
  sign.material = accentMaterial;
  const beacon = MeshBuilder.CreateCylinder('repair-shop-beacon', {
    height: 0.14,
    diameter: 0.22,
    tessellation: 14,
  }, scene);
  beacon.position.set(center.x, 1.65, center.z - 0.25);
  beacon.material = accentMaterial;
  const glow = new GlowLayer('repair-shop-glow', scene, { blurKernelSize: 18 });
  glow.intensity = 0.55;
  glow.addIncludedOnlyMesh(sign);
  glow.addIncludedOnlyMesh(beacon);
}

function createTowTruck(): {
  update: (x: number, z: number, state: number, lift: number, now: number) => void;
} {
  const root = new TransformNode('tow-truck-root', scene);
  const bodyMaterial = new StandardMaterial('tow-truck-body-material', scene);
  bodyMaterial.diffuseColor = Color3.FromHexString('#e8a028');
  bodyMaterial.emissiveColor = Color3.FromHexString('#4b2604');
  bodyMaterial.specularColor = Color3.FromHexString('#ffd989');
  const darkMaterial = new StandardMaterial('tow-truck-dark-material', scene);
  darkMaterial.diffuseColor = Color3.FromHexString('#1b2a32');
  darkMaterial.emissiveColor = Color3.FromHexString('#071017');
  const lightMaterial = new StandardMaterial('tow-truck-light-material', scene);
  lightMaterial.diffuseColor = Color3.FromHexString('#70f6ff');
  lightMaterial.emissiveColor = Color3.FromHexString('#20c9dd');
  lightMaterial.disableLighting = true;

  const chassis = MeshBuilder.CreateBox('tow-truck-chassis', {
    width: 0.7,
    height: 0.12,
    depth: 0.8,
  }, scene);
  chassis.position.y = 0.16;
  chassis.material = darkMaterial;
  chassis.parent = root;
  const bed = MeshBuilder.CreateBox('tow-truck-flatbed', {
    width: 0.64,
    height: 0.08,
    depth: 0.66,
  }, scene);
  bed.position.set(0, 0.25, -0.03);
  bed.material = bodyMaterial;
  bed.parent = root;
  const nose = MeshBuilder.CreateBox('tow-truck-nose', {
    width: 0.62,
    height: 0.18,
    depth: 0.18,
  }, scene);
  nose.position.set(0, 0.25, 0.39);
  nose.material = bodyMaterial;
  nose.parent = root;
  const window = MeshBuilder.CreateBox('tow-truck-window', {
    width: 0.48,
    height: 0.07,
    depth: 0.025,
  }, scene);
  window.position.set(0, 0.3, 0.49);
  window.material = lightMaterial;
  window.parent = root;
  const liftRails = [-0.22, 0.22].map((x) => {
    const rail = MeshBuilder.CreateBox('tow-truck-lift-rail', {
      width: 0.08,
      height: 0.045,
      depth: 0.62,
    }, scene);
    rail.position.set(x, 0.305, -0.04);
    rail.material = darkMaterial;
    rail.parent = root;
    return rail;
  });
  for (const x of [-0.37, 0.37]) {
    for (const z of [-0.28, 0.28]) {
      const wheel = MeshBuilder.CreateCylinder('tow-truck-wheel', {
        height: 0.12,
        diameter: 0.21,
        tessellation: 14,
      }, scene);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, 0.14, z);
      wheel.material = darkMaterial;
      wheel.parent = root;
    }
  }
  const beacon = MeshBuilder.CreateCylinder('tow-truck-beacon', {
    height: 0.11,
    diameter: 0.14,
    tessellation: 12,
  }, scene);
  beacon.position.set(0, 0.42, 0.36);
  beacon.material = lightMaterial;
  beacon.parent = root;
  const glow = new GlowLayer('tow-truck-glow', scene, { blurKernelSize: 14 });
  glow.intensity = 0.7;
  glow.addIncludedOnlyMesh(window);
  glow.addIncludedOnlyMesh(beacon);
  root.position.copyFrom(worldAt(TOW_DEPOT_X, TOW_DEPOT_Z, 0.015));

  let previousX = TOW_DEPOT_X;
  let previousZ = TOW_DEPOT_Z;
  return {
    update: (x, z, state, lift, now) => {
      const position = worldAt(x, z, state === 0 ? 0.015 : 0.025);
      root.position.copyFrom(position);
      const dx = x - previousX;
      const dz = z - previousZ;
      if (Math.abs(dx) + Math.abs(dz) > 0.01) root.rotation.y = Math.atan2(dx, dz);
      previousX = x;
      previousZ = z;
      const pulse = state === 0 ? 0.45 : 0.75 + Math.sin(now * 0.018) * 0.25;
      lightMaterial.emissiveColor.set(0.12 * pulse, 0.75 * pulse, pulse);
      const platformLift = smoothstep(Math.max(0, Math.min(1, lift)));
      bed.position.y = 0.25 + platformLift * 0.14;
      for (const rail of liftRails) rail.position.y = 0.305 + platformLift * 0.14;
    },
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
  const step = time / FALLBACK_STEP_SECONDS;
  unloadingScene.update(step);
  reloadingScene.update(step);
  conveyorScene.update(step);
  selectionHalo.isVisible = false;
  hideTaskMarkers();
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

function resetServiceAnimations(): void {
  pendingHandoffs.fill(null);
  pendingReloads.fill(null);
  pendingPalletMotions.fill(null);
  palletMotions.fill(null);
  unloadingScene.reset();
  reloadingScene.reset();
  conveyorScene.reset();
  towLoadingStartedStep = 0;
}

function resetPalletInventory(): void {
  palletInventory.fill(PALLET_CAPACITY);
  palletInventoryRevision += 1;
  palletScene.update(new Set(), palletInventory);
  hiddenPalletKey = '';
}

function updateLive(now: number): void {
  if (!liveCurrent) return;
  const from = livePrevious ?? liveCurrent;
  const frameDurationMs = 1000 / Math.max(0.1, liveTickRate * speed);
  const alpha = Math.min(1, (now - liveCurrent.receivedAt) / frameDurationMs);
  const simulationStep = from.step + (liveCurrent.step - from.step) * alpha;
  const towLoadingActive = from.tow.state === 4 || liveCurrent.tow.state === 4;
  const towLoadingProgress = towLoadingActive
    ? Math.max(0, Math.min(1, (simulationStep - towLoadingStartedStep) / TOW_LOADING_TICKS))
    : 0;
  const transfers = [
    ...unloadingScene.update(simulationStep),
    ...reloadingScene.update(simulationStep),
    ...conveyorScene.update(simulationStep),
  ];
  const hiddenPallets = new Set<number>();
  const palletOverrides = new Map<number, { x: number; z: number }>();
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
    const towLoadingTarget =
      (from.tow.state === 4 && from.tow.targetAgent === i)
      || (liveCurrent.tow.state === 4 && liveCurrent.tow.targetAgent === i);
    const fromTransported = from.recoveryStates[i] === 2 ? 1 : 0;
    const toTransported = liveCurrent.recoveryStates[i] === 2 ? 1 : 0;
    const transportLift = towLoadingTarget
      ? smoothstep(towLoadingProgress)
      : fromTransported + (toTransported - fromTransported) * alpha;
    writeTransform(matrices, i, x, z, 0.02 + transportLift * 0.55);
    if (i === selectedAgentId) {
      const position = worldAt(x, z, 0.045);
      selectionHalo.position.copyFrom(position);
      const pulse = 1 + Math.sin(now * 0.008) * 0.055;
      selectionHalo.scaling.set(pulse, pulse, pulse);
      selectionHalo.isVisible = true;
    }
    const palletId = liveCurrent.palletIds[i];
    const palletX = liveCurrent.palletPositions[i * 2];
    const palletZ = liveCurrent.palletPositions[i * 2 + 1];
    if ((liveCurrent.statuses[i] & 2) === 0
        && palletId < palletCells.length && palletX >= 0 && palletZ >= 0) {
      palletOverrides.set(palletId, { x: palletX, z: palletZ });
    }
    const enteringPickup = from.taskIds[i] === liveCurrent.taskIds[i]
      && ((from.stages[i] === 0 && liveCurrent.stages[i] === 1)
        || (from.recoveryStates[i] === 4 && liveCurrent.recoveryStates[i] === 0))
      && (from.statuses[i] & 2) === 0;
    const motion = palletMotions[i];
    const motionProgress = motion
      ? Math.max(0, Math.min(1, (simulationStep - motion.startedStep) / PALLET_DWELL_TICKS))
      : 1;
    const activeDrop = motion?.kind === 'drop' && motionProgress < 1;
    const showLoadedPallet = (liveCurrent.statuses[i] & 2) !== 0 && !enteringPickup;
    if (showLoadedPallet || activeDrop) {
      const loadIndex = loadedCount++;
      loadAgentIds[loadIndex] = i;
      const visualPalletId = activeDrop ? motion.palletId : palletId;
      const liftProgress = motion?.kind === 'pickup'
        ? smoothstep(motionProgress)
        : motion?.kind === 'drop'
          ? 1 - smoothstep(motionProgress)
          : 1;
      writeTransform(loadMatrices, loadIndex, x, z, 0.1 * liftProgress);
      if (visualPalletId < palletCells.length) hiddenPallets.add(visualPalletId);
      if (visualPalletId < palletCells.length) {
        const cargoType = palletCells[visualPalletId].cargoType;
        const taskUnchanged = from.taskIds[i] === liveCurrent.taskIds[i];
        const pendingHandoff = pendingHandoffs[i];
        const fromQuantity = taskUnchanged
          ? Math.min(PALLET_CAPACITY, from.palletItems[i])
          : palletInventory[visualPalletId];
        const currentQuantity = activeDrop
          ? palletInventory[visualPalletId]
          : Math.min(PALLET_CAPACITY, liveCurrent.palletItems[i]);
        const isReloadingBatchTransition = taskUnchanged
          && currentQuantity > fromQuantity
          && Math.round(x0) === RELOAD_X
          && Math.round(z0) === liveCurrent.reloadPositions[i * 2 + 1];
        const baseQuantity = pendingHandoff?.taskId === liveCurrent.taskIds[i]
          ? pendingHandoff.itemsBefore
          : isReloadingBatchTransition && alpha < 1
            ? fromQuantity
            : currentQuantity;
        for (let slot = 0; slot < baseQuantity; slot++) {
          const cargoIndex = cargoCounts[cargoType]++;
          cargoAgentIds[cargoType][cargoIndex] = i;
          goodsTransform(x, z, 0.1 * liftProgress, slot)
            .copyToArray(cargoMatrices[cargoType], cargoIndex * 16);
        }
      }
    }
  }
  const overrideKey = [...palletOverrides.entries()]
    .sort(([a], [b]) => a - b)
    .map(([id, position]) => `${id}:${position.x}:${position.z}`)
    .join(',');
  const nextHiddenKey = `${palletInventoryRevision}|${[...hiddenPallets].sort((a, b) => a - b).join(',')}|${overrideKey}`;
  if (nextHiddenKey !== hiddenPalletKey) {
    palletScene.update(hiddenPallets, palletInventory, palletOverrides);
    hiddenPalletKey = nextHiddenKey;
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
  const towX = from.tow.x + (liveCurrent.tow.x - from.tow.x) * alpha;
  const towZ = from.tow.z + (liveCurrent.tow.z - from.tow.z) * alpha;
  const fromTowLift = from.tow.state === 2 ? 1 : 0;
  const toTowLift = liveCurrent.tow.state === 2 ? 1 : 0;
  const towLift = towLoadingActive
    ? towLoadingProgress
    : fromTowLift + (toTowLift - fromTowLift) * alpha;
  towTruckScene.update(
    towX,
    towZ,
    liveCurrent.tow.state,
    towLift,
    now,
  );
}

function setAgentColor(index: number, status: number, recoveryState = 0): void {
  const loaded = (status & 2) !== 0;
  const waiting = (status & 1) !== 0;
  const transitioned = (status & 4) !== 0;
  const failed = (status & 16) !== 0;
  const color = failed
    ? [1, 0.12, 0.07, 1]
    : recoveryState === 4
    ? [0.35, 1, 0.62, 1]
    : index === selectedAgentId
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
  if (protocol < 1 || protocol > 5) return;
  const recordSize = protocol >= 4 ? 36 : protocol === 3 ? 32 : protocol === 2 ? 24 : 16;
  const step = view.getUint32(8, true);
  const priorFrame = liveCurrent && step > liveCurrent.step ? liveCurrent : null;
  if (liveCurrent && step < liveCurrent.step) {
    resetServiceAnimations();
    resetAgentStats();
  }
  const completedTasks = view.getUint16(6, true);
  const count = Math.min(MAX_AGENTS, view.getUint32(12, true));
  const towRecordSize = protocol >= 4 ? 16 : 0;
  if (view.byteLength < 16 + count * recordSize + towRecordSize) return;
  const positions = new Float32Array(count * 2);
  const statuses = new Uint8Array(count);
  const stages = new Uint8Array(count);
  const palletIds = new Uint16Array(count);
  const taskIds = new Uint32Array(count);
  const stationPositions = new Int16Array(count * 2);
  const reloadPositions = new Int16Array(count * 2);
  const palletItems = new Uint8Array(count);
  const recoveryStates = new Uint8Array(count);
  const palletPositions = new Int16Array(count * 2);
  taskIds.fill(0xffffffff);
  stationPositions.fill(-1);
  reloadPositions.fill(-1);
  palletItems.fill(PALLET_CAPACITY);
  palletPositions.fill(-1);
  for (let i = 0; i < count; i++) {
    const offset = 16 + i * recordSize;
    const id = view.getUint32(offset, true);
    if (id >= count) continue;
    positions[id * 2] = view.getFloat32(offset + 4, true);
    positions[id * 2 + 1] = view.getFloat32(offset + 8, true);
    statuses[id] = view.getUint8(offset + 12);
    const encodedStage = view.getUint8(offset + 13);
    stages[id] = protocol < 3 && encodedStage === 2 ? 3 : encodedStage;
    palletIds[id] = view.getUint16(offset + 14, true);
    if (protocol >= 2) {
      taskIds[id] = view.getUint32(offset + 16, true);
      stationPositions[id * 2] = view.getInt16(offset + 20, true);
      stationPositions[id * 2 + 1] = view.getInt16(offset + 22, true);
    }
    if (protocol >= 3) {
      reloadPositions[id * 2] = view.getInt16(offset + 24, true);
      reloadPositions[id * 2 + 1] = view.getInt16(offset + 26, true);
      palletItems[id] = Math.min(PALLET_CAPACITY, view.getUint8(offset + 28));
      const palletId = palletIds[id];
      if (palletId < palletInventory.length && palletInventory[palletId] !== palletItems[id]) {
        palletInventory[palletId] = palletItems[id];
        palletInventoryRevision += 1;
      }
    }
    if (protocol >= 4) {
      recoveryStates[id] = view.getUint8(offset + 30);
      palletPositions[id * 2] = view.getInt16(offset + 32, true);
      palletPositions[id * 2 + 1] = view.getInt16(offset + 34, true);
    }
    const enteringUnloadingBay = stages[id] >= 2
      && Math.round(positions[id * 2]) === UNLOAD_X
      && (statuses[id] & 4) !== 0
      && (statuses[id] & 1) === 0;
    setAgentColor(
      id,
      enteringUnloadingBay ? statuses[id] & ~4 : statuses[id],
      recoveryStates[id],
    );
  }
  let tow: TowFrame = {
    x: TOW_DEPOT_X,
    z: TOW_DEPOT_Z,
    state: 0,
    targetAgent: 0xffff,
    queuedRescues: 0,
  };
  if (protocol >= 4) {
    const towOffset = 16 + count * recordSize;
    tow = {
      x: view.getFloat32(towOffset, true),
      z: view.getFloat32(towOffset + 4, true),
      state: view.getUint8(towOffset + 8),
      targetAgent: view.getUint16(towOffset + 10, true),
      queuedRescues: view.getUint16(towOffset + 12, true),
    };
  }
  if (tow.state === 4
      && (priorFrame?.tow.state !== 4 || priorFrame.tow.targetAgent !== tow.targetAgent)) {
    towLoadingStartedStep = step;
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
      const enteredPostUnloadStage = priorFrame.stages[id] === 1 && stages[id] >= 2;
      const enteredPickupStage = priorFrame.stages[id] === 0 && stages[id] === 1
        && priorFrame.taskIds[id] === taskIds[id];
      const recoveredPallet = priorFrame.recoveryStates[id] === 4
        && recoveryStates[id] === 0
        && (statuses[id] & 2) !== 0;
      if (taskIds[id] !== priorFrame.taskIds[id]) {
        if (priorFrame.taskIds[id] !== 0xffffffff) stats.completedTasks += 1;
        stats.taskStartedAt = step;
        stats.lastTaskId = taskIds[id];
      }
      const palletId = palletIds[id];
      const x = Math.round(positions[id * 2]);
      const stationZ = Math.round(positions[id * 2 + 1]);
      if ((enteredPickupStage || recoveredPallet) && palletId < palletCells.length) {
        pendingPalletMotions[id] = { kind: 'pickup', taskId: taskIds[id], palletId };
      }
      const pallet = palletId < palletCells.length ? palletCells[palletId] : null;
      const arrivedAtReturn = stages[id] === 3
        && pallet !== null
        && x === pallet.x
        && stationZ === pallet.z
        && (priorFrame.positions[id * 2] !== positions[id * 2]
          || priorFrame.positions[id * 2 + 1] !== positions[id * 2 + 1]);
      if (arrivedAtReturn) {
        pendingPalletMotions[id] = { kind: 'drop', taskId: taskIds[id], palletId };
      }
      if (enteredPostUnloadStage && x === UNLOAD_X && palletId < palletCells.length) {
        pendingHandoffs[id] = {
          taskId: taskIds[id],
          palletId,
          stationZ,
          itemsBefore: Math.min(PALLET_CAPACITY, priorFrame.palletItems[id]),
        };
      }
      const pending = pendingHandoffs[id];
      const waitingAtStation = (statuses[id] & 1) !== 0
        && stages[id] >= 2
        && x === UNLOAD_X
        && priorFrame.positions[id * 2] === positions[id * 2]
        && priorFrame.positions[id * 2 + 1] === positions[id * 2 + 1];
      if (pending && pending.taskId === taskIds[id] && waitingAtStation) {
        unloadingScene.trigger(
          pending.stationZ,
          palletCells[pending.palletId].cargoType,
          Math.max(0, pending.itemsBefore - 1),
          priorFrame.step,
        );
        stats.handoffs += 1;
        setAgentColor(id, statuses[id] | 4);
        pendingHandoffs[id] = null;
      } else if (pending && (pending.taskId !== taskIds[id] || stages[id] < 2 || x !== UNLOAD_X)) {
        pendingHandoffs[id] = null;
      }
      const reloadX = reloadPositions[id * 2];
      const reloadZ = reloadPositions[id * 2 + 1];
      const arrivedAtReload = stages[id] === 2
        && x === reloadX
        && stationZ === reloadZ
        && (priorFrame.positions[id * 2] !== positions[id * 2]
          || priorFrame.positions[id * 2 + 1] !== positions[id * 2 + 1]);
      if (arrivedAtReload) {
        pendingReloads[id] = { taskId: taskIds[id], stationZ: reloadZ };
      }
      const pendingReload = pendingReloads[id];
      const waitingAtReload = (statuses[id] & 1) !== 0
        && stages[id] === 2
        && x === reloadX
        && stationZ === reloadZ
        && priorFrame.positions[id * 2] === positions[id * 2]
        && priorFrame.positions[id * 2 + 1] === positions[id * 2 + 1];
      if (pendingReload && pendingReload.taskId === taskIds[id] && waitingAtReload) {
        reloadingScene.trigger(
          pendingReload.stationZ,
          palletCells[palletId].cargoType,
          priorFrame.step,
        );
        pendingReloads[id] = null;
      } else if (pendingReload
          && (pendingReload.taskId !== taskIds[id] || stages[id] !== 2)) {
        pendingReloads[id] = null;
      }
      const pendingPallet = pendingPalletMotions[id];
      const waitingAtPallet = (statuses[id] & 1) !== 0
        && pallet !== null
        && x === (palletPositions[id * 2] >= 0 ? palletPositions[id * 2] : pallet.x)
        && stationZ === (palletPositions[id * 2 + 1] >= 0
          ? palletPositions[id * 2 + 1]
          : pallet.z);
      if (pendingPallet && pendingPallet.taskId === taskIds[id] && waitingAtPallet) {
        palletMotions[id] = { ...pendingPallet, startedStep: priorFrame.step };
        pendingPalletMotions[id] = null;
      } else if (pendingPallet && pendingPallet.taskId !== taskIds[id]) {
        pendingPalletMotions[id] = null;
      }
      const palletMotion = palletMotions[id];
      if (palletMotion && step - palletMotion.startedStep > PALLET_DWELL_TICKS) {
        palletMotions[id] = null;
      } else if (palletMotion?.kind === 'pickup'
          && taskIds[id] !== palletMotion.taskId) {
        palletMotions[id] = null;
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
    reloadPositions,
    palletItems,
    recoveryStates,
    palletPositions,
    tow,
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
  if (previous >= 0) setAgentColor(
    previous,
    liveCurrent?.statuses[previous] ?? 0,
    liveCurrent?.recoveryStates[previous] ?? 0,
  );
  setAgentColor(
    agent,
    liveCurrent?.statuses[agent] ?? 0,
    liveCurrent?.recoveryStates[agent] ?? 0,
  );
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
    setAgentColor(
      previous,
      liveCurrent?.statuses[previous] ?? 0,
      liveCurrent?.recoveryStates[previous] ?? 0,
    );
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
  const nativeStage = Math.min(3, frame.stages[id]);
  const status = frame.statuses[id];
  const palletId = frame.palletIds[id];
  const taskId = frame.taskIds[id];
  const pallet = palletId < palletCells.length ? palletCells[palletId] : null;
  const stationX = frame.stationPositions[id * 2];
  const stationZ = frame.stationPositions[id * 2 + 1];
  const reloadX = frame.reloadPositions[id * 2];
  const reloadZ = frame.reloadPositions[id * 2 + 1];
  const stats = agentStats[id];
  const waiting = (status & 1) !== 0;
  const loaded = (status & 2) !== 0;
  const transitioned = (status & 4) !== 0;
  const failed = (status & 16) !== 0;
  const recoveryState = frame.recoveryStates[id];
  const agentX = Math.round(frame.positions[id * 2]);
  const agentZ = Math.round(frame.positions[id * 2 + 1]);
  const atUnloadingBay = nativeStage >= 2 && agentX === UNLOAD_X && agentZ === stationZ;
  const atReloadingBay = nativeStage >= 2 && agentX === reloadX && agentZ === reloadZ;
  const stage = atUnloadingBay ? 1 : atReloadingBay ? 2 : nativeStage;
  const palletMotion = palletMotions[id];
  const loadingOntoTow = recoveryState === 1
    && frame.tow.state === 4
    && frame.tow.targetAgent === id;
  const stateLabel = loadingOntoTow
    ? 'LOADING ONTO TOW'
    : recoveryState === 1
    ? 'WAITING FOR TOW'
    : recoveryState === 2
      ? 'EVACUATING'
      : recoveryState === 3
        ? 'UNDER REPAIR'
        : recoveryState === 4
          ? 'RECOVERING PALLET'
    : palletMotion?.kind === 'pickup' && waiting
    ? 'LOADING PALLET'
    : palletMotion?.kind === 'drop' && waiting
      ? 'PARKING PALLET'
      : atReloadingBay && waiting
        ? 'BATCH LOADING'
      : transitioned && waiting && atUnloadingBay
        ? 'HANDOFF'
    : atUnloadingBay && waiting
      ? 'UNLOADING ITEM'
      : waiting
        ? 'WAITING'
        : loaded
          ? 'LOADED'
          : 'EMPTY';
  const statePill = document.querySelector<HTMLElement>('#selected-agent-state')!;
  statePill.textContent = stateLabel;
  statePill.className = `state-pill${failed ? ' failed' : recoveryState === 4 ? ' recovery' : loaded ? ' loaded' : ''}${waiting && !failed ? ' waiting' : ''}`;
  const failButton = document.querySelector<HTMLButtonElement>('#fail-agent-button')!;
  failButton.disabled = recoveryState !== 0 || socket?.readyState !== WebSocket.OPEN;
  document.querySelector('#fail-agent-label')!.textContent = recoveryState !== 0
    ? 'RECOVERY IN PROGRESS'
    : 'BREAK AGENT';

  document.querySelector('#selected-agent')!.textContent = `AGENT ${String(id).padStart(3, '0')}`;
  document.querySelector('#agent-position')!.textContent =
    `${Math.round(frame.positions[id * 2])}, ${Math.round(frame.positions[id * 2 + 1])}`;
  document.querySelector('#agent-task')!.textContent = taskId === 0xffffffff ? '—' : `#${taskId}`;
  const cargoNames = ['CRATE', 'CARTONS', 'DRUMS'];
  const physicalPalletX = frame.palletPositions[id * 2];
  const physicalPalletZ = frame.palletPositions[id * 2 + 1];
  const droppedLabel = recoveryState !== 0 && !loaded
      && physicalPalletX >= 0 && physicalPalletZ >= 0
      && pallet !== null
      && (physicalPalletX !== pallet.x || physicalPalletZ !== pallet.z)
    ? ` · @ ${physicalPalletX},${physicalPalletZ}`
    : '';
  document.querySelector('#agent-pallet')!.textContent = pallet
    ? `P${String(palletId).padStart(3, '0')} · ${cargoNames[pallet.cargoType]} · ${frame.palletItems[id]}/${PALLET_CAPACITY}${droppedLabel}`
    : '—';
  document.querySelector('#agent-task-age')!.textContent = `${Math.max(0, frame.step - stats.taskStartedAt)} TICKS`;

  const formatCoordinate = (x: number, z: number): string => x >= 0 && z >= 0 ? `${x}, ${z}` : '—';
  document.querySelector('#pickup-coordinate')!.textContent = pallet ? formatCoordinate(pallet.x, pallet.z) : '—';
  document.querySelector('#unload-coordinate')!.textContent = formatCoordinate(stationX, stationZ);
  document.querySelector('#reload-coordinate')!.textContent = formatCoordinate(reloadX, reloadZ);
  document.querySelector('#return-coordinate')!.textContent = pallet ? formatCoordinate(pallet.x, pallet.z) : '—';

  const stageElements = [
    document.querySelector<HTMLElement>('#stage-pickup')!,
    document.querySelector<HTMLElement>('#stage-unload')!,
    document.querySelector<HTMLElement>('#stage-reload')!,
    document.querySelector<HTMLElement>('#stage-return')!,
  ];
  const routeElements = [
    document.querySelector<HTMLElement>('#route-pickup')!,
    document.querySelector<HTMLElement>('#route-unload')!,
    document.querySelector<HTMLElement>('#route-reload')!,
    document.querySelector<HTMLElement>('#route-return')!,
  ];
  const reloadRequired = (status & 8) !== 0 || (nativeStage <= 1 && frame.palletItems[id] <= 1);
  for (let index = 0; index < 4; index++) {
    const state = index === 2 && !reloadRequired
      ? (stage > 1 ? 'skipped' : '')
      : index < stage
        ? 'done'
        : index === stage
          ? 'active'
          : '';
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
      resetServiceAnimations();
      resetPalletInventory();
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
    resetServiceAnimations();
    resetPalletInventory();
    resetAgentStats();
    clearAgentSelection();
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
const failedValue = document.querySelector('#failed-value')!;
const recoveryValue = document.querySelector('#recovery-value')!;
const towStatusValue = document.querySelector('#tow-status-value')!;
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
    failedValue.textContent = String(liveCurrent ? liveCurrent.statuses.reduce((sum, status) => sum + ((status & 16) !== 0 ? 1 : 0), 0) : 0);
    recoveryValue.textContent = String(liveCurrent
      ? liveCurrent.recoveryStates.reduce((sum, state) => sum + (state !== 0 ? 1 : 0), 0)
      : 0);
    if (liveCurrent) {
      const towLabels = ['IDLE', 'TO AGENT', 'TRANSPORT', 'RETURNING', 'LOADING'];
      const target = liveCurrent.tow.targetAgent === 0xffff
        ? ''
        : ` · A${String(liveCurrent.tow.targetAgent).padStart(3, '0')}`;
      const queued = liveCurrent.tow.queuedRescues > 0
        ? ` · Q${liveCurrent.tow.queuedRescues}`
        : '';
      towStatusValue.textContent = `${towLabels[liveCurrent.tow.state] ?? 'ACTIVE'}${target}${queued}`;
    } else {
      towStatusValue.textContent = 'IDLE';
    }
    tasksValue.textContent = String(liveCurrent?.completedTasks ?? 0);
  }
});

const pauseButton = document.querySelector<HTMLButtonElement>('#pause-button')!;
const stopButton = document.querySelector<HTMLButtonElement>('#stop-button')!;
const failAgentButton = document.querySelector<HTMLButtonElement>('#fail-agent-button')!;
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
  resetServiceAnimations();
  resetAgentStats();
  sendControl('stop');
  setConnection('stopped', 'SIMULATION // STOPPED');
});

failAgentButton.addEventListener('click', () => {
  if (selectedAgentId < 0 || !liveCurrent) return;
  if (liveCurrent.recoveryStates[selectedAgentId] !== 0) return;
  failAgentButton.disabled = true;
  document.querySelector('#fail-agent-label')!.textContent = 'BREAKING…';
  sendControl('fail', { agent: selectedAgentId });
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
  camera.radius = 37.5;
  camera.target.set(0, 0, 0);
});

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement | null;
  if (target?.matches('input:not([type="range"]), select, textarea, [contenteditable="true"]')) return;
  const key = cameraKeyByCode[event.code];
  if (!key) return;
  cameraKeys.add(key);
  event.preventDefault();
});

window.addEventListener('keyup', (event) => {
  const key = cameraKeyByCode[event.code];
  if (!key) return;
  cameraKeys.delete(key);
  event.preventDefault();
});

window.addEventListener('blur', () => cameraKeys.clear());

window.addEventListener('resize', () => engine.resize());
connect();
