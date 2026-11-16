import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// The built site must also open from a folder (file://), where browsers refuse module
// scripts. So the build is one classic script, and the tags lose their module attributes.
function classicScript(): Plugin {
  return {
    name: "classic-script",
    enforce: "post",
    transformIndexHtml(html) {
      return html
        .replace(/<script type="module" crossorigin src="([^"]+)"><\/script>/g, "")
        .replace(/<link rel="stylesheet" crossorigin href=/g, '<link rel="stylesheet" href=')
        .replace("</body>", '  <script defer src="./assets/app.js"></script>\n  </body>');
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [react(), classicScript()],
  build: {
    target: "es2020",
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        format: "iife",
        entryFileNames: "assets/app.js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: { host: "127.0.0.1", port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
