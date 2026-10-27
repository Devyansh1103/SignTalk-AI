import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// The AI models live in ../frontend/public/models (written by scripts/export_onnx.py),
// so both frontends always serve the same model without a second copy.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  publicDir: '../frontend/public',
  server: { port: 5174 },
  preview: { port: 4174 },
})
