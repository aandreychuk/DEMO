import * as ort from 'onnxruntime-web';
import { resolve } from 'node:path';

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(resolve('public/runtime/fastdmm-0.8m.onnx'), {
  executionProviders: ['wasm'],
});
const observations = new BigInt64Array(100 * 256);
const chat = new BigInt64Array(100 * 13);
const neighborPadding = new Uint8Array(100 * 13);
observations.fill(66n);
chat.fill(-1n);
neighborPadding.fill(1);
const result = await session.run({
  observations: new ort.Tensor('int64', observations, [1, 100, 256]),
  chat: new ort.Tensor('int64', chat, [1, 100, 13]),
  neighbor_padding: new ort.Tensor('bool', neighborPadding, [1, 100, 13]),
});
const output = result.action_probabilities;
if (output.dims.join(',') !== '100,5') throw new Error(`unexpected output shape: ${output.dims}`);
const values = output.data;
for (let agent = 0; agent < 100; agent++) {
  let sum = 0;
  for (let action = 0; action < 5; action++) {
    const value = values[agent * 5 + action];
    if (!Number.isFinite(value)) throw new Error(`non-finite output for agent ${agent}`);
    sum += value;
  }
  if (Math.abs(sum - 1) > 1e-4) throw new Error(`probabilities do not sum to one for agent ${agent}`);
}
console.log('browser model validation: ONNX Runtime Web/WASM output is finite [100,5]');
