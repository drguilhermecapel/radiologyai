/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// O backend real (``radiologyai serve``) atende em /api; no desenvolvimento o
// Vite repassa as chamadas. Porta configurável para os testes de ponta a ponta.
const backend = process.env.RADIOLOGYAI_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: backend, changeOrigin: false } },
  },
  preview: {
    proxy: { "/api": { target: backend, changeOrigin: false } },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
  },
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
