import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The model API (model/main.py) has no CORS middleware, so the dev server
// proxies it. Run it on port 8001 (8000 is the backend API):
//   cd ../model && uvicorn main:app --port 8001
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/model-api": {
        target: "http://localhost:8001",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/model-api/, ""),
      },
    },
  },
});
