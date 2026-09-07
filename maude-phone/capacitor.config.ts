import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.maude.phone',
  appName: 'MAUDE',
  webDir: 'dist',
  server: {
    // MagicDNS stays valid if the Tailscale IP changes. HTTP :30080 is the
    // Safari/PWA path (no cert prompt). Native app still loads HTTPS :30000.
    url: 'https://server.tail00a82a.ts.net:30000',
    cleartext: true,
    allowNavigation: [
      'server.tail00a82a.ts.net',
      'server.tail00a82a.ts.net:30000',
      'server.tail00a82a.ts.net:30080',
      'server.tail00a82a.ts.net:8998',
      '100.66.49.48',
      '100.66.49.48:30000',
      '100.66.49.48:30080',
      '100.66.49.48:8998',
    ],
  },
  android: {
    allowMixedContent: true,
    captureInput: true,
    webContentsDebuggingEnabled: true,
  },
  ios: {
    contentInset: 'always',
    allowsLinkPreview: false,
    webContentsDebuggingEnabled: true,
    preferredContentMode: 'mobile',
  },
  plugins: {
    CapacitorHttp: {
      enabled: false,  // Disabled: native HTTP breaks SSE streaming (ReadableStream)
    },
  },
};

export default config;
