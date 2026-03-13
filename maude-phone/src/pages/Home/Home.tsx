import { FC, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

interface AppIcon {
  path: string;
  label: string;
  icon: string;
  description: string;
}

const APPS: AppIcon[] = [
  { path: "/maude", label: "MAUDE", icon: "\u25C6", description: "AI Chat" },
  { path: "/maude/voice", label: "Voice", icon: "\uD83C\uDF99\uFE0F", description: "PersonaPlex" },
  { path: "/terminal", label: "Terminal", icon: ">_", description: "SSH Shell" },
  { path: "/browser", label: "Browser", icon: "\u25CE", description: "Web" },
  { path: "/messages", label: "Messages", icon: "\u2709", description: "Telegram" },
  { path: "/files", label: "Files", icon: "\u25A4", description: "File Manager" },
  { path: "/collab", label: "Collab", icon: "\u29BF", description: "Mesh Status" },
  { path: "/command-center", label: "System", icon: "\u25A3", description: "Command Center" },
  { path: "/settings", label: "Settings", icon: "\u2699", description: "Configure" },
];

function getGatewayUrl(): string {
  const loc = window.location;
  return `${loc.protocol}//${loc.host}`;
}

export const Home: FC = () => {
  const navigate = useNavigate();
  const [time, setTime] = useState(new Date());
  const [health, setHealth] = useState<{ status: string } | null>(null);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const check = () => {
      fetch(`${getGatewayUrl()}/health`)
        .then((r) => r.json())
        .then(setHealth)
        .catch(() => setHealth(null));
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full flex-col px-4 pt-6">
      {/* Header */}
      <div className="mb-2 text-center">
        <h1 className="fire-gradient text-4xl font-black tracking-tight">MAUDE</h1>
        <p className="mt-1 text-xs text-maude-muted">Multi-Agent Unified Dispatch Engine</p>
      </div>

      {/* Clock */}
      <div className="mb-4 text-center">
        <div className="text-5xl font-light tabular-nums text-maude-text">
          {time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </div>
        <div className="mt-1 text-sm text-maude-muted">
          {time.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" })}
        </div>
      </div>

      {/* Status */}
      <div className="mb-4 flex items-center justify-center gap-3 text-xs">
        <span className={`flex items-center gap-1 ${health?.status ? "text-green-400" : "text-red-400"}`}>
          <span className={`inline-block h-2 w-2 rounded-full ${health?.status ? "bg-green-400" : "bg-red-400"}`} />
          Spark {health?.status ? "Connected" : "Offline"}
        </span>
        <span className="text-maude-muted">|</span>
        <span className="text-maude-muted">Tailscale Active</span>
      </div>

      {/* App Grid */}
      <div className="grid flex-1 grid-cols-3 gap-3 content-start">
        {APPS.map((app) => (
          <button
            key={app.path}
            onClick={() => navigate(app.path)}
            className="flex flex-col items-center justify-center rounded-2xl bg-maude-surface p-4 transition-all active:scale-95 hover:bg-maude-card"
          >
            <span className="mb-2 text-3xl">{app.icon}</span>
            <span className="text-sm font-medium text-maude-text">{app.label}</span>
            <span className="mt-0.5 text-[10px] text-maude-muted">{app.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
