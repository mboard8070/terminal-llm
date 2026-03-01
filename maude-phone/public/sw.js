// MAUDE Service Worker — enables PWA "Add to Home Screen" on iOS Safari
const CACHE_NAME = 'maude-v2';

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
      // Best-effort pre-cache — don't fail install if some assets 404
      return Promise.allSettled(
        APP_SHELL.map((url) => cache.add(url).catch(() => {}))
      );
    })
  );
  // Activate immediately, don't wait for old tabs to close
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      )
    )
  );
  // Take control of all open tabs immediately
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Network-only for API calls, WebSocket upgrades, streaming endpoints
  if (NETWORK_ONLY_PATTERNS.some((p) => url.pathname.startsWith(p))) {
    return; // Let the browser handle it normally
  }

  // Network-first for HTML navigation requests (always get fresh app)
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request) || caches.match('/'))
    );
    return;
  }

  // Cache-first for static assets (JS, CSS, images, fonts, WASM)
  if (
    url.pathname.startsWith('/assets/') ||
    url.pathname.endsWith('.js') ||
    url.pathname.endsWith('.css') ||
    url.pathname.endsWith('.wasm') ||
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.svg') ||
    url.pathname.endsWith('.woff2')
  ) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        });
      })
    );
    return;
  }

  // Default: network-first for everything else
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
