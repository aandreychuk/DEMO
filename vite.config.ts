import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        warehouse: resolve(import.meta.dirname, 'index.html'),
        epibt: resolve(import.meta.dirname, 'epibt.html'),
        maze: resolve(import.meta.dirname, 'maze.html'),
      },
    },
  },
});
