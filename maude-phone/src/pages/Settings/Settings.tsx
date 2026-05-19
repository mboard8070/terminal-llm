import { FC, useState, useEffect } from "react";
import { getGatewayUrl } from "../../lib/gateway";

interface ServiceStatus { status: string; port: number; }
interface HealthStatus {
  status: string;
  services?: {
    llama_server?: ServiceStatus;
    voice_server?: ServiceStatus;
  };
  gateway_port?: number;
}
interface ModelInfo { id: string; provider: string; available: boolean; }


const THEMES = [
  { id: "dark", label: "MAUDE Dark", desc: "Default dark theme" },
  { id: "professional", label: "Professional", desc: "Clean corporate dark" },
  { id: "modern", label: "Modern Terminal", desc: "Clean slate & indigo" },
  { id: "retro-green", label: "80s Green CRT", desc: "Phosphor green terminal" },
  { id: "retro-amber", label: "80s Amber CRT", desc: "Amber phosphor terminal" },
];

function applyTheme(id: string) {
  document.documentElement.setAttribute("data-theme", id);
  localStorage.setItem("maude-theme", id);
}

export const Settings: FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [defaultModel, setDefaultModel] = useState(() => {
    const stored = localStorage.getItem("maude-default-model");
    return !stored || stored === "mistral-large-latest" ? "nemotron-super" : stored;
  });
  const [defaultVoice, setDefaultVoice] = useState(() => localStorage.getItem("maude-default-voice") || "NATF2.pt");
  const [theme, setTheme] = useState(() => localStorage.getItem("maude-theme") || "dark");

  // Gateway reachable = Spark is connected (regardless of whether sub-services are up)
  const sparkConnected = health !== null;
  const gatewayPort = health?.gateway_port ?? (new URL(getGatewayUrl()).port || "30080");
  const llmService = health?.services?.llama_server;
  const voiceService = health?.services?.voice_server;

  useEffect(() => {
    fetch(`${getGatewayUrl()}/health`).then((r) => r.json()).then(setHealth).catch(() => setHealth(null));
    fetch(`${getGatewayUrl()}/models`).then((r) => r.json()).then((d) => setModels(d.models || [])).catch(() => setModels([]));
  }, []);

  const saveModel = (m: string) => { setDefaultModel(m); localStorage.setItem("maude-default-model", m); };
  const saveVoice = (v: string) => { setDefaultVoice(v); localStorage.setItem("maude-default-voice", v); };

  const svcLabel = (svc?: ServiceStatus) => {
    if (!svc) return { text: "\u2014", color: "text-maude-muted" };
    if (svc.status === "up" || svc.status === "ok") return { text: `${svc.port} (${svc.status})`, color: "text-green-400" };
    return { text: `${svc.port} (down)`, color: "text-red-400" };
  };

  const llm = svcLabel(llmService);
  const ppx = svcLabel(voiceService);

  return (
    <div className="no-scrollbar h-full overflow-y-auto bg-maude-bg">
      <div className="border-b border-maude-border bg-maude-surface px-4 py-3">
        <h1 className="text-lg font-semibold text-maude-text">Settings</h1>
      </div>

      <div className="space-y-6 p-4">
        {/* Connection */}
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted">Connection</h2>
          <div className="space-y-2 rounded-xl bg-maude-surface p-4">
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Spark Status</span>
              <span className={`flex items-center gap-1.5 text-sm ${sparkConnected ? "text-green-400" : "text-red-400"}`}>
                <span className={`h-2 w-2 rounded-full ${sparkConnected ? "bg-green-400" : "bg-red-400"}`} />
                {sparkConnected ? "Connected" : "Offline"}
              </span>
            </div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Gateway</span><span className={`font-mono text-sm ${sparkConnected ? "text-green-400" : "text-maude-muted"}`}>{sparkConnected ? `${gatewayPort} (up)` : "\u2014"}</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">LLM</span><span className={`font-mono text-sm ${llm.color}`}>{llm.text}</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Voice Server</span><span className={`font-mono text-sm ${ppx.color}`}>{ppx.text}</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Tailscale</span><span className="text-sm text-green-400">Active</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Host</span><span className="font-mono text-sm text-maude-muted">{getGatewayUrl().replace(/^https?:\/\//, "")}</span></div>
          </div>
        </section>

        {/* Theme */}
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted">Theme</h2>
          <div className="space-y-1 rounded-xl bg-maude-surface p-2">
            {THEMES.map((t) => (
              <button key={t.id} onClick={() => { setTheme(t.id); applyTheme(t.id); }}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${t.id === theme ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`}>
                <span>{t.label}</span><span className="text-xs text-maude-muted">{t.desc}</span>
              </button>
            ))}
          </div>
        </section>

        {/* Model */}
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted">Default Model</h2>
          <div className="space-y-1 rounded-xl bg-maude-surface p-2">
            {models.map((m) => (
              <button key={m.id} onClick={() => saveModel(m.id)}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${m.id === defaultModel ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`}>
                <div className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${m.available ? "bg-green-400" : "bg-red-400"}`} />{m.id}</div>
                <span className="text-xs text-maude-muted">{m.provider}</span>
              </button>
            ))}
            {models.length === 0 && <p className="px-3 py-2 text-sm text-maude-muted">Loading models...</p>}
          </div>
        </section>

        {/* Voice */}
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted">Voice</h2>
          <div className="rounded-xl bg-maude-surface p-4">
            <select value={defaultVoice} onChange={(e) => saveVoice(e.target.value)}
              className="w-full rounded-lg bg-maude-bg px-3 py-2.5 text-sm text-maude-text outline-none focus:ring-1 focus:ring-maude-accent">
              {["NATF0.pt","NATF1.pt","NATF2.pt","NATF3.pt","NATM0.pt","NATM1.pt","NATM2.pt","NATM3.pt"].map((v) => (
                <option key={v} value={v}>{v.replace(".pt","")}{v === "NATF2.pt" ? " (MAUDE)" : ""}{v === "NATM1.pt" ? " (Male)" : ""}</option>
              ))}
            </select>
          </div>
        </section>

        {/* Network */}
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted">Network</h2>
          <div className="rounded-xl bg-maude-surface p-4"><p className="text-sm text-maude-muted">Network settings are managed via Tailscale and your device's system settings.</p></div>
        </section>

        {/* About */}
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted">About</h2>
          <div className="space-y-2 rounded-xl bg-maude-surface p-4">
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Version</span><span className="text-sm text-maude-muted">1.8.3</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Engine</span><span className="text-sm text-maude-muted">Mistral + Codestral + Claude</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Voice</span><span className="text-sm text-maude-muted">MAUDE Voice ({(localStorage.getItem("maude-default-voice") || "NATF2.pt").replace(".pt", "")})</span></div>
            <div className="flex items-center justify-between"><span className="text-sm text-maude-text">Hub</span><span className="text-sm font-mono">DGX Spark</span></div>
            <div className="pt-2 text-center text-xs text-maude-muted"><span className="fire-gradient font-bold">MAUDE</span> — Multi-Agent Unified Dispatch Engine</div>
          </div>
        </section>
      </div>
    </div>
  );
};
