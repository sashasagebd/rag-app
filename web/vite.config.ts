// Dev server on http://localhost:5173. /api requests go to the FastAPI backend on :8000,
// so the browser sees one origin and the session cookie just works.
// `npm run build` writes web/dist/, which FastAPI serves in production.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
