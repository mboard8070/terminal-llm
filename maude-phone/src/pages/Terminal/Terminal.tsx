import { FC, useEffect, useRef, useState } from "react";

export const Terminal: FC = () => {
  const termRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const terminalRef = useRef<any>(null);
  const fitRef = useRef<any>(null);
  const [status, setStatus] = useState<"connecting" | "connected" | "disconnected">("disconnected");

  useEffect(() => {
    let term: any;
    let cleanupResizeObserver: (() => void) | undefined;

    const init = async () => {
      const { Terminal: XTerm } = await import("@xterm/xterm");
      const { FitAddon } = await import("@xterm/addon-fit");
      const { WebLinksAddon } = await import("@xterm/addon-web-links");

      if (!document.querySelector('link[href*="xterm"]')) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = "https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/css/xterm.min.css";
        document.head.appendChild(link);
      }

      term = new XTerm({
        cursorBlink: true, fontSize: 16,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        theme: {
          background: "#0d1117", foreground: "#e6edf3", cursor: "#ff4500", cursorAccent: "#0d1117",
          selectionBackground: "#30363d", black: "#0d1117", red: "#ff7b72", green: "#7ee787",
          yellow: "#ffa657", blue: "#79c0ff", magenta: "#d2a8ff", cyan: "#a5d6ff", white: "#e6edf3",
          brightBlack: "#484f58", brightRed: "#ffa198", brightGreen: "#56d364", brightYellow: "#e3b341",
          brightBlue: "#a5d6ff", brightMagenta: "#d2a8ff", brightCyan: "#b1bac4", brightWhite: "#f0f6fc",
        },
        allowTransparency: true, scrollback: 5000,
      });

      const fitAddon = new FitAddon();
      term.loadAddon(fitAddon);
      term.loadAddon(new WebLinksAddon());
      terminalRef.current = term;
      fitRef.current = fitAddon;

      if (termRef.current) { term.open(termRef.current); fitAddon.fit(); }

      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/terminal`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        setStatus("connected");
        const dims = fitAddon.proposeDimensions();
        if (dims) ws.send(JSON.stringify({ type: "resize", cols: dims.cols, rows: dims.rows }));
      };
      ws.onmessage = (e) => { term.write(e.data instanceof ArrayBuffer ? new Uint8Array(e.data) : e.data); };
      ws.onclose = () => { setStatus("disconnected"); term.write("\r\n\x1b[33m[Connection closed]\x1b[0m\r\n"); };
      ws.onerror = () => { setStatus("disconnected"); };

      term.onData((data: string) => { if (ws.readyState === WebSocket.OPEN) ws.send(data); });

      const handleResize = () => {
        fitAddon.fit();
        const dims = fitAddon.proposeDimensions();
        if (dims && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "resize", cols: dims.cols, rows: dims.rows }));
      };
      const resizeObserver = new ResizeObserver(handleResize);
      if (termRef.current) resizeObserver.observe(termRef.current);
      cleanupResizeObserver = () => resizeObserver.disconnect();
    };

    init();

    return () => {
      cleanupResizeObserver?.();
      wsRef.current?.close();
      terminalRef.current?.dispose();
    };
  }, []);

  return (
    <div className="flex h-full flex-col bg-[#0d1117]">
      <div className="flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm text-maude-text">&gt;_ Terminal</span>
          <span className={`h-2 w-2 rounded-full ${status === "connected" ? "bg-green-400" : status === "connecting" ? "bg-yellow-400" : "bg-red-400"}`} />
          <span className="text-xs text-maude-muted">{status}</span>
        </div>
        {status === "disconnected" && (
          <button onClick={() => window.location.reload()} className="rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text">Reconnect</button>
        )}
      </div>

      {/* Special keys */}
      <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-maude-border bg-maude-surface px-2 py-1">
        {[
          { label: "Esc", key: "\x1b" }, { label: "Tab", key: "\t" },
          { label: "Ctrl+C", key: "\x03" }, { label: "Ctrl+D", key: "\x04" },
          { label: "Ctrl+Z", key: "\x1a" }, { label: "Ctrl+L", key: "\x0c" },
          { label: "\u2191", key: "\x1b[A" }, { label: "\u2193", key: "\x1b[B" },
          { label: "\u2190", key: "\x1b[D" }, { label: "\u2192", key: "\x1b[C" },
        ].map((btn) => (
          <button key={btn.label} onClick={() => { wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send(btn.key); terminalRef.current?.focus(); }}
            className="shrink-0 rounded bg-maude-bg px-2 py-1 text-[11px] font-mono text-maude-muted active:bg-maude-accent active:text-white">{btn.label}</button>
        ))}
      </div>

      <div ref={termRef} className="flex-1 overflow-hidden px-1 py-1" />
    </div>
  );
};
