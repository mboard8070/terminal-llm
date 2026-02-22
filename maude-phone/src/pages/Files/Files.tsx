import { FC, useState, useEffect, useCallback, useRef } from "react";

interface FileEntry { name: string; size: number; is_dir: boolean; modified: number; }

function getGatewayUrl(): string { return `${window.location.protocol}//${window.location.host}`; }
function formatSize(b: number): string { if (b < 1024) return b + " B"; if (b < 1048576) return (b / 1024).toFixed(1) + " KB"; return (b / 1048576).toFixed(1) + " MB"; }
function formatDate(ts: number): string { return new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }

export const Files: FC = () => {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [currentPath, setCurrentPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"shared" | "transfers">("shared");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadFiles = useCallback(async (path?: string) => {
    setLoading(true); setError("");
    try {
      const url = tab === "transfers" ? `${getGatewayUrl()}/transfers` : path ? `${getGatewayUrl()}/list?path=${encodeURIComponent(path)}` : `${getGatewayUrl()}/list`;
      const resp = await fetch(url);
      const data = await resp.json();
      if (data.error) setError(data.error); else { setFiles(data.files || []); setCurrentPath(data.path || ""); }
    } catch (err: unknown) { setError(err instanceof Error ? err.message : "Failed"); }
    setLoading(false);
  }, [tab]);

  useEffect(() => { loadFiles(); }, [loadFiles]);

  const handleDownload = (name: string) => { window.open(`${getGatewayUrl()}/${tab === "transfers" ? "download-transfer" : "download"}/${encodeURIComponent(name)}`); };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    setLoading(true);
    try { const resp = await fetch(`${getGatewayUrl()}/upload/${encodeURIComponent(file.name)}`, { method: "POST", body: file }); const data = await resp.json(); if (data.error) setError(data.error); else loadFiles(); }
    catch (err: unknown) { setError(err instanceof Error ? err.message : "Upload failed"); }
    setLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="flex h-full flex-col bg-maude-bg">
      <div className="flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-3">
        <div className="flex items-center gap-2"><span className="text-lg">{"\u25A4"}</span><h1 className="text-sm font-semibold text-maude-text">Files</h1></div>
        <div className="flex items-center gap-2">
          <button onClick={() => fileInputRef.current?.click()} className="rounded-lg fire-bg px-3 py-1 text-xs font-medium text-white">Upload</button>
          <button onClick={() => loadFiles()} className="rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-muted">{"\u21BB"}</button>
          <input ref={fileInputRef} type="file" onChange={handleUpload} className="hidden" />
        </div>
      </div>

      <div className="flex shrink-0 border-b border-maude-border bg-maude-surface">
        {(["shared", "transfers"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)} className={`flex-1 py-2 text-xs font-medium capitalize ${tab === t ? "border-b-2 border-maude-accent text-maude-accent" : "text-maude-muted"}`}>{t}</button>
        ))}
      </div>

      {currentPath && (
        <div className="flex items-center gap-2 border-b border-maude-border bg-maude-surface/50 px-4 py-2">
          <button onClick={() => { const p = currentPath.split("/").slice(0, -1).join("/"); loadFiles(p || undefined); }} className="text-xs text-maude-accent">{"\u2190"} Up</button>
          <span className="truncate text-xs text-maude-muted">{currentPath}</span>
        </div>
      )}

      {error && <div className="bg-red-900/30 px-4 py-2 text-xs text-red-400">{error}</div>}

      <div className="no-scrollbar flex-1 overflow-y-auto">
        {loading && <div className="flex h-32 items-center justify-center"><div className="h-6 w-6 animate-spin rounded-full border-2 border-maude-accent border-t-transparent" /></div>}
        {!loading && files.length === 0 && <div className="flex h-32 items-center justify-center"><p className="text-sm text-maude-muted">No files found.</p></div>}
        {!loading && files.map((f) => (
          <button key={f.name} onClick={() => f.is_dir ? loadFiles(currentPath ? `${currentPath}/${f.name}` : f.name) : handleDownload(f.name)}
            className="flex w-full items-center gap-3 border-b border-maude-border/50 px-4 py-3 text-left hover:bg-maude-surface">
            <span className="text-lg">{f.is_dir ? "\uD83D\uDCC1" : "\uD83D\uDCC4"}</span>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm text-maude-text">{f.name}</div>
              <div className="mt-0.5 text-[10px] text-maude-muted">{f.is_dir ? "Directory" : formatSize(f.size)} · {formatDate(f.modified)}</div>
            </div>
            {!f.is_dir && <span className="text-xs text-maude-muted">{"\u2193"}</span>}
          </button>
        ))}
      </div>
    </div>
  );
};
