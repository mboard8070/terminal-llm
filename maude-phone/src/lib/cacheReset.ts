export async function resetAppCache(): Promise<string[]> {
  const lines: string[] = [];

  if ("serviceWorker" in navigator) {
    const regs = await navigator.serviceWorker.getRegistrations();
    await Promise.all(regs.map((reg) => reg.unregister()));
    lines.push(`Unregistered ${regs.length} service worker${regs.length === 1 ? "" : "s"}`);
  } else {
    lines.push("Service workers unavailable");
  }

  if ("caches" in window) {
    const names = await caches.keys();
    await Promise.all(names.map((name) => caches.delete(name)));
    lines.push(`Deleted ${names.length} cache${names.length === 1 ? "" : "s"}`);
  } else {
    lines.push("Cache API unavailable");
  }

  localStorage.clear();
  sessionStorage.clear();
  lines.push("Cleared local app state");

  return lines;
}

export async function resetAppCacheAndReload(): Promise<void> {
  await resetAppCache();
  window.location.replace(`/?fresh=${Date.now()}`);
}
