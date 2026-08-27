import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  // The footron wall runs footron-web-shell (Electron 14 = Chromium 93), which
  // cannot parse ES2022 syntax like class static blocks — target it explicitly.
  build: { outDir: "web", emptyOutDir: true, target: "chrome93" },
  test: {
    environment: "jsdom",
    globals: false,
  },
});
