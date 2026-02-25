import { defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());

  // In dev, proxy to gateway; in prod, gateway serves the built files
  const gatewayTarget = env.VITE_GATEWAY_URL || "http://localhost:30000";
  const proxyConf: Record<string, any> = {};
  for (const path of ["/v1", "/ws", "/health", "/models", "/list", "/transfers", "/download", "/upload", "/share", "/proxy", "/api"]) {
    proxyConf[path] = {
      target: gatewayTarget,
      changeOrigin: true,
      ws: path === "/ws" || path === "/api",
    };
  }

  return {
    server: {
      host: "0.0.0.0",
      port: 5174,
      proxy: proxyConf,
    },
    plugins: [
      topLevelAwait({
        promiseExportName: "__tla",
        promiseImportName: (i: number) => `__tla_${i}`,
      }),
    ],
    build: {
      outDir: "dist",
      sourcemap: false,
    },
  };
});
