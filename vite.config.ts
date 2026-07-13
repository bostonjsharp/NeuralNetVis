import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "web", emptyOutDir: true },
  test: {
    environment: "jsdom",
    globals: false,
  },
});
