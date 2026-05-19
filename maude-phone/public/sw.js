// MAUDE Service Worker — enables PWA "Add to Home Screen"
const CACHE_NAME = 'maude-v8';

// Always resolve to a real Response so respondWith() never sees null/undefined.
const offlineResponse = () =>
  new Response('Offline', { status: 503, headers: { 'Content-Type': 'text/plain' } });

// App shell files to pre-cache on install
const APP_SHELL = [
  '/',
  '/manifest.json',
  '/assets/maude-icon.svg',
  '/assets/maude-icon-180.png',
  '/assets/maude-icon-512.png',
];

// Paths that should always go to network (API, WebSocket, streaming)
const NETWORK_ONLY_PATTERNS = [
  '/v1/',
  '/ws/',
  '/api/',
  '/proxy/',
  '/app/v1/',
  '/app/ws/',
  '/app/api/',
  '/app/proxy/',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return Promise.allSettled(
        APP_SHELL.map((url) => cache.add(url).catch(() => {}))
      );
    })
  );
  // Activate immediately — don't wait for old tabs
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      )
    ).then(() => self.clients.claim())
    .then(() => {
      // Notify all open clients to reload
      self.clients.matchAll({ type: 'window' }).then((clients) => {
        clients.forEach((client) => client.postMessage({ type: 'SW_UPDATED' }));
      });
    })
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Network-only for API calls, WebSocket upgrades, streaming endpoints
  if (NETWORK_ONLY_PATTERNS.some((p) => url.pathname.startsWith(p))) {
    return;
  }

  // Network-first for navigation AND all JS/CSS assets
  // This ensures new builds are always fetched from network
  if (
    event.request.mode === 'navigate' ||
    url.pathname.startsWith('/assets/') ||
    url.pathname.endsWith('.js') ||
    url.pathname.endsWith('.css')
  ) {
    event.respondWith(
      fetch(event.request, { cache: 'no-store' })
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(event.request);
          if (cached) return cached;
          const shell = await caches.match('/');
          return shell || offlineResponse();
        })
    );
    return;
  }

  // Cache-first for static assets (images, fonts, wasm)
  if (
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.svg') ||
    url.pathname.endsWith('.woff2') ||
    url.pathname.endsWith('.wasm')
  ) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request)
          .then((response) => {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
            return response;
          })
          .catch(() => offlineResponse());
      })
    );
    return;
  }

  // Default: network-first
  event.respondWith(
    fetch(event.request, { cache: 'no-store' })
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(async () => {
        const cached = await caches.match(event.request);
        return cached || offlineResponse();
      })
  );
});
