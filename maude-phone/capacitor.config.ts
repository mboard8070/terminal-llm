import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.maude.phone',
  appName: 'MAUDE',
  webDir: 'dist',
  server: {
    // Point to the Spark gateway HTTP mirror — avoids self-signed cert issues in WKWebView
    url: 'http://100.107.132.16:30080',
    cleartext: true,
    allowNavigation: ['100.107.132.16', '100.107.132.16:30000', '100.107.132.16:30080', '100.107.132.16:8998'],
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
