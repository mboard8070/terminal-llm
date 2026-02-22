import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.maude.phone',
  appName: 'MAUDE',
  webDir: 'dist',
  server: {
    // Point to the Spark gateway — the app loads from the server, not bundled
    url: 'https://100.107.132.16:30000',
    cleartext: true,
    allowNavigation: ['100.107.132.16', '100.107.132.16:30000', '100.107.132.16:30080', '100.107.132.16:8998'],
  },
  android: {
    allowMixedContent: true,
    captureInput: true,
    webContentsDebuggingEnabled: true,
  },
  plugins: {
    CapacitorHttp: {
      enabled: true,
    },
  },
};

export default config;
