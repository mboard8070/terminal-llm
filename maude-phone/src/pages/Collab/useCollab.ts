import { useState, useEffect, useCallback } from "react";

function apiBase(): string {
  const loc = window.location;
  return `${loc.protocol}//${loc.host}`;
}

export interface PresenceEntry {
  client_id: string;
  client_type: string;
  hostname: string;
  status: string;
  activity: string;
  active_conversation: string;
  active_project: string;
  last_seen: number;
}

export interface ActivityEvent {
  id: string;
  ts: number;
  hostname: string;
  client_id: string;
  type: string;
  summary: string;
  data: Record<string, unknown>;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
  hostname: string;
  conversations: string[];
  files: string[];
  tags: string[];
}

export interface Task {
  id: string;
  source: string;
  target: string;
  capability: string;
  status: string;
  prompt: string;
  result: string | null;
  project_id: string;
  created_at: number;
  updated_at: number;
}

export interface CollabStatus {
  hostname: string;
  ts: number;
  presence: PresenceEntry[];
  activity: ActivityEvent[];
  projects: Project[];
  tasks: Task[];
}

export function useCollab(pollInterval = 10000) {
  const [status, setStatus] = useState<CollabStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/collab/status`);
      if (r.ok) {
        setStatus(await r.json());
      }
    } catch {
      // Gateway might not be available
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, pollInterval);
    return () => clearInterval(interval);
  }, [refresh, pollInterval]);

  const createProject = useCallback(
    async (name: string, description = "", tags: string[] = []) => {
      const r = await fetch(`${apiBase()}/api/collab/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, tags }),
      });
      if (r.ok) {
        refresh();
        return await r.json();
      }
    },
    [refresh],
  );

  const dispatchTask = useCallback(
    async (prompt: string, target = "", capability = "SHELL") => {
      const r = await fetch(`${apiBase()}/api/collab/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, target, capability }),
      });
      if (r.ok) {
        refresh();
        return await r.json();
      }
    },
    [refresh],
  );

  return { status, loading, refresh, createProject, dispatchTask };
}

// Detect device platform
function detectDevice(): { clientType: string; label: string } {
  const ua = navigator.userAgent;
  if (/iPad/.test(ua)) return { clientType: "ipad", label: "iPad" };
  if (/iPhone/.test(ua)) return { clientType: "iphone", label: "iPhone" };
  if (/Android/.test(ua) && /Mobile/.test(ua)) return { clientType: "android", label: "Android" };
  if (/Android/.test(ua)) return { clientType: "android-tablet", label: "Android Tablet" };
  if (/Macintosh/.test(ua)) return { clientType: "mac", label: "Mac" };
  if (/Windows/.test(ua)) return { clientType: "windows", label: "Windows" };
  return { clientType: "phone", label: "Phone" };
}

// App-level presence heartbeat (call once at app init)
let heartbeatStarted = false;

export function startPresenceHeartbeat() {
  if (heartbeatStarted) return;
  heartbeatStarted = true;

  const device = detectDevice();
  const clientId = `${device.clientType}-${Math.random().toString(36).slice(2, 8)}`;

  const beat = () => {
    fetch(`${apiBase()}/api/collab/presence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_id: clientId,
        client_type: device.clientType,
        activity: document.visibilityState === "visible" ? "browsing app" : "background",
      }),
    }).catch(() => {});
  };

  beat();
  setInterval(beat, 30000);
}
