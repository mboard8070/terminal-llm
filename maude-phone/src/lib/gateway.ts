const FALLBACK_GATEWAY_URL = "https://100.107.132.16:30000";

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
