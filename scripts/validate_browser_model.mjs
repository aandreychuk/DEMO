import * as ort from 'onnxruntime-web';
import { resolve } from 'node:path';

const count = Number(process.argv[2] ?? 100);
if (count !== 100 && count !== 1000) throw new Error('Test model size must be 100 or 1000.');
const modelName = count > 100 ? 'fastdmm-0.8m-1000.onnx' : 'fastdmm-0.8m.onnx';
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(resolve('public/runtime/' + modelName), {
  executionProviders: ['wasm'],
});
const observations = new BigInt64Array(count * 256);
const chat = new BigInt64Array(count * 13);
const neighborPadding = new Uint8Array(count * 13);
observations.fill(66n);
chat.fill(-1n);
neighborPadding.fill(1);
const result = await session.run({
  observations: new ort.Tensor('int64', observations, [1, count > 100 ? 1000 : 100, 256]),
  chat: new ort.Tensor('int64', chat, [1, count > 100 ? 1000 : 100, 13]),
  neighbor_padding: new ort.Tensor('bool', neighborPadding, [1, count > 100 ? 1000 : 100, 13]),
});
const output = result.action_probabilities;
if (output.dims.join(',') !== (count > 100 ? '1000,5' : '100,5')) throw new Error(`unexpected output shape: ${output.dims}`);
const values = output.data;
for (let agent = 0; agent < count; agent++) {
  let sum = 0;
  for (let action = 0; action < 5; action++) {
    const value = values[agent * 5 + action];
    if (!Number.isFinite(value)) throw new Error(`non-finite output for agent ${agent}`);
    sum += value;
  }
  if (Math.abs(sum - 1) > 1e-4) throw new Error(`probabilities do not sum to one for agent ${agent}`);
}
console.log('browser model validation: ONNX Runtime Web/WASM output is finite for ' + count + ' agents');
