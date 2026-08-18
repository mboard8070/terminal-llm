import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.maude.phone',
  appName: 'MAUDE',
  webDir: 'dist',
  server: {
    // Point to the Aveak Windows gateway on the primary HTTPS port.
    url: 'https://desktop-aveak19:30000',
    cleartext: true,
    allowNavigation: [
      'desktop-aveak19',
      'desktop-aveak19:30000',
      'desktop-aveak19:30080',
      'desktop-aveak19:8998',
      '100.86.227.15',
      '100.86.227.15:30000',
      '100.86.227.15:30080',
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
