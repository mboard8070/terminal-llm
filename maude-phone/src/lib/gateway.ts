const FALLBACK_GATEWAY_URL = "https://desktop-aveak19:30000";

function isLocalAppOrigin(): boolean {
  const loc = window.location;
  return (
    loc.protocol === "file:" ||
    loc.protocol === "capacitor:" ||
    loc.protocol === "ionic:" ||
    loc.hostname === "localhost" ||
    loc.hostname === "127.0.0.1" ||
    loc.hostname === ""
  );
}

export function getGatewayUrl(): string {
  const loc = window.location;
  if (!isLocalAppOrigin() && (loc.protocol === "http:" || loc.protocol === "https:")) {
    return `${loc.protocol}//${loc.host}`;
  }
  return FALLBACK_GATEWAY_URL;
}

export function getGatewayWsUrl(path: string): string {
  const base = new URL(getGatewayUrl());
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return `${base.protocol}//${base.host}${path}`;
}


export async function clearGatewayClientState(): Promise<void> {
  try {
    if ("serviceWorker" in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((reg) => reg.unregister()));
    }
  } catch { /* best effort */ }
  try {
    if ("caches" in window) {
      const names = await caches.keys();
      await Promise.all(names.map((name) => caches.delete(name)));
    }
  } catch { /* best effort */ }
}

export function isGatewayLoadError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const msg = err.message.toLowerCase();
  return err.name === "TypeError" && (msg.includes("load failed") || msg.includes("failed to fetch"));
}

export async function fetchGateway(pathOrUrl: string, init: RequestInit = {}, timeoutMs = 10000): Promise<Response> {
  const url = pathOrUrl.startsWith("http") ? pathOrUrl : `${getGatewayUrl()}${pathOrUrl}`;
  const run = async (): Promise<Response> => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, { ...init, cache: "no-store", signal: init.signal ?? controller.signal });
    } finally {
      window.clearTimeout(timeout);
    }
  };

  try {
    return await run();
  } catch (err) {
    if (!isGatewayLoadError(err)) throw err;
    await clearGatewayClientState();
    return run();
  }
}
