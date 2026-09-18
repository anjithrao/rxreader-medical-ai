import {defineConfig} from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({command}) => ({
  base: command === "build" ? "/static/" : "/",
  plugins: [react()],
  server: {
    proxy: {
      "/process": "http://localhost:5000",
      "/translate_references": "http://localhost:5000",
      "/nearby_pharmacies": "http://localhost:5000",
      "/route_to_pharmacy": "http://localhost:5000",
      "/chat": "http://localhost:5000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          motion: ["framer-motion"],
          three: ["three"],
          map: ["leaflet"],
        },
      },
    },
  },
}));
