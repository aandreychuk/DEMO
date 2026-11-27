export function createMazeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let bits = Math.imul(state ^ (state >>> 15), state | 1);
    bits ^= bits + Math.imul(bits ^ (bits >>> 7), bits | 61);
    return (((bits ^ (bits >>> 14)) >>> 0) + 0.5) / 0x100000000;
  };
}

export function fillMazeVoteNoise(values: Float32Array, random: () => number): void {
  // Adding independent Gumbel noise before argmax samples from the model's logits.
  for (let i = 0; i < values.length; i++) values[i] = -Math.log(-Math.log(random()));
}
