import { FC, useState, useRef, useCallback } from "react";
import { getGatewayUrl } from "../../lib/gateway";


const BOOKMARKS = [
  { label: "Google", url: "https://www.google.com" },
  { label: "GitHub", url: "https://github.com" },
  { label: "Reddit", url: "https://www.reddit.com" },
  { label: "HN", url: "https://news.ycombinator.com" },
];

export const Browser: FC = () => {
  const [url, setUrl] = useState("");
  const [displayUrl, setDisplayUrl] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [mode, setMode] = useState<"proxy" | "iframe">("proxy");
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const loadUrl = useCallback(async (targetUrl: string) => {
    if (!targetUrl.trim()) return;
    let fullUrl = targetUrl.trim();
    if (!fullUrl.startsWith("http://") && !fullUrl.startsWith("https://")) fullUrl = "https://" + fullUrl;
    setDisplayUrl(fullUrl);
    setError("");

    if (mode === "iframe") {
      setUrl(fullUrl);
      setHistory((prev) => [...prev.slice(0, historyIndex + 1), fullUrl]);
      setHistoryIndex((prev) => prev + 1);
      return;
    }

    setLoading(true);
    try {
      const resp = await fetch(`${getGatewayUrl()}/proxy?url=${encodeURIComponent(fullUrl)}`);
      if (!resp.ok) { setError(`Failed: ${resp.status}`); setLoading(false); return; }
      const ct = resp.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const data = await resp.json();
        if (data.redirect) { setLoading(false); loadUrl(data.redirect); return; }
        setError(data.error || "Unknown error");
      } else {
        setContent(await resp.text());
      }
      setHistory((prev) => [...prev.slice(0, historyIndex + 1), fullUrl]);
      setHistoryIndex((prev) => prev + 1);
    } catch (err: unknown) { setError(err instanceof Error ? err.message : "Failed"); }
    setLoading(false);
  }, [mode, historyIndex]);

  const goBack = () => { if (historyIndex > 0) { setHistoryIndex(historyIndex - 1); loadUrl(history[historyIndex - 1]); } };
  const goForward = () => { if (historyIndex < history.length - 1) { setHistoryIndex(historyIndex + 1); loadUrl(history[historyIndex + 1]); } };

  return (
    <div className="flex h-full flex-col bg-maude-bg">
      <div className="flex shrink-0 flex-col border-b border-maude-border bg-maude-surface">
        <form onSubmit={(e) => { e.preventDefault(); loadUrl(displayUrl); }} className="flex items-center gap-2 px-3 py-2">
          <div className="flex gap-1">
            <button type="button" onClick={goBack} disabled={historyIndex <= 0} className="rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30">{"\u25C0"}</button>
            <button type="button" onClick={goForward} disabled={historyIndex >= history.length - 1} className="rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30">{"\u25B6"}</button>
            <button type="button" onClick={() => loadUrl(displayUrl)} className="rounded px-2 py-1 text-sm text-maude-muted">{"\u21BB"}</button>
          </div>
          <input type="text" value={displayUrl} onChange={(e) => setDisplayUrl(e.target.value)} placeholder="Enter URL..."
            className="flex-1 rounded-lg bg-maude-bg px-3 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent" />
          <button type="button" onClick={() => setMode(mode === "proxy" ? "iframe" : "proxy")} className="rounded-lg bg-maude-bg px-2 py-1 text-[10px] text-maude-muted">
            {mode === "proxy" ? "PROXY" : "IFRAME"}
          </button>
        </form>
        <div className="flex gap-1 overflow-x-auto px-3 pb-2 no-scrollbar">
          {BOOKMARKS.map((bm) => (
            <button key={bm.url} onClick={() => { setDisplayUrl(bm.url); loadUrl(bm.url); }}
              className="shrink-0 rounded-full bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text">{bm.label}</button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        {loading && <div className="flex h-full items-center justify-center"><div className="h-6 w-6 animate-spin rounded-full border-2 border-maude-accent border-t-transparent" /></div>}
        {error && <div className="flex h-full items-center justify-center p-8 text-center"><p className="text-red-400">{error}</p></div>}
        {!loading && !error && mode === "proxy" && content && <iframe srcDoc={content} className="h-full w-full border-0 bg-white" sandbox="allow-scripts allow-same-origin allow-forms" title="Browser" />}
        {!loading && !error && mode === "iframe" && url && <iframe ref={iframeRef} src={url} className="h-full w-full border-0 bg-white" sandbox="allow-scripts allow-same-origin allow-forms allow-popups" title="Browser" />}
        {!loading && !error && !content && !url && <div className="flex h-full flex-col items-center justify-center gap-4 text-center"><span className="text-4xl">{"\u25CE"}</span><p className="text-sm text-maude-muted">Enter a URL to browse the web.</p></div>}
      </div>
    </div>
  );
};
