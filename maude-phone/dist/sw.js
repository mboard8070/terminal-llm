// MAUDE service worker removal shim.
// The phone app is a live gateway client; stale PWA caching breaks chat requests.
const KILL_CACHE = 'maude-sw-kill-v1';

async function cleanup() {
  try {
    const names = await caches.keys();
    await Promise.all(names.map((name) => caches.delete(name)));
  } catch (_) {}
  try {
    await self.registration.unregister();
  } catch (_) {}
  try {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const client of clients) {
      client.postMessage({ type: 'SW_REMOVED' });
    }
  } catch (_) {}
}

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(cleanup());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(cleanup());
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'REMOVE_SW') event.waitUntil(cleanup());
});

// Do not intercept anything. Every request must hit the live gateway.
self.addEventListener('fetch', () => {});
