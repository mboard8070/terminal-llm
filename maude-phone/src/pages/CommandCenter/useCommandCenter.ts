import { useState, useEffect, useCallback } from "react";
import { fetchGateway, getGatewayUrl } from "../../lib/gateway";


export interface SystemStats {
  cpu_percent: number;
  ram: { used_gb: number; total_gb: number; percent: number };
  disk: { used_gb: number; total_gb: number; percent: number };
  gpu?: {
    name: string;
    temperature_c: number | string;
    utilization_percent: number | string;
    memory_used_mb?: number | string;
    memory_total_mb?: number | string;
    power_w?: string;
  };
}

export interface GpuProcess {
  pid: string;
  name: string;
  memory_mb: number;
}

export interface GpuProcesses {
  total_mb: number;
  used_mb: number;
  free_mb: number;
  processes: GpuProcess[];
}

export interface Session {
  session_id: string;
  channel: string;
  started_at: string;
  last_message_at: string;
  message_count: number;
}

export interface Activity {
  channel: string;
  role: string;
  content: string;
  timestamp: string | null;
}

export interface SchedulerTask {
  id: string;
  name: string;
  cron: string;
  prompt: string;
  enabled: boolean;
  last_run: string;
  next_run: string;
  run_count: number;
  last_result: string;
}

export interface SchedulerStatus {
  stats: { total: number; active: number; total_runs: number };
  tasks: SchedulerTask[];
}

export interface MissionLog {
  time: string;
  kind: string;
  message: string;
}

export interface Mission {
  id: string;
  title: string;
  objective: string;
  status: string;
  cadence: string;
  progress: { done: number; total: number };
  next_action: string;
  blockers: string[];
  artifacts: string[];
  schedule: { task_id?: string; cron?: string; enabled?: boolean };
  updated_at: string;
  recent_logs: MissionLog[];
}

export interface MissionStatus {
  stats: { total: number; active: number; blocked: number; scheduled: number };
  missions: Mission[];
}

export interface NodeInfo {
  name: string;
  type: string;
  status: string;
  services?: Record<string, boolean>;
  os?: string;
  ip?: string;
  platform?: string;
  version?: string;
  last_seen?: string;
}

export interface GatewayStatus {
  ok: boolean;
  url: string;
  error?: string;
  checked_at?: number;
}

async function fetchApi<T>(endpoint: string): Promise<T | null> {
  const r = await fetchGateway(`/api/command-center/${endpoint}`, {}, 7000);
  if (!r.ok) throw new Error(`${endpoint}: HTTP ${r.status}`);
  return await r.json();
}

async function checkGateway(): Promise<GatewayStatus> {
  const url = getGatewayUrl();
  try {
    const r = await fetchGateway("/api/ping", {}, 5000);
    if (!r.ok) return { ok: false, url, error: `HTTP ${r.status}`, checked_at: Date.now() };
    return { ok: true, url, checked_at: Date.now() };
  } catch (err) {
    const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    return { ok: false, url, error: message, checked_at: Date.now() };
  }
}

export function useCommandCenter(refreshInterval = 10000) {
  const [system, setSystem] = useState<SystemStats | null>(null);
  const [gpuProcesses, setGpuProcesses] = useState<GpuProcesses | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [scheduler, setScheduler] = useState<SchedulerStatus | null>(null);
  const [missions, setMissions] = useState<MissionStatus | null>(null);
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [gatewayStatus, setGatewayStatus] = useState<GatewayStatus>({ ok: false, url: getGatewayUrl() });

  const refresh = useCallback(async () => {
    const gateway = await checkGateway();
    setGatewayStatus(gateway);

    const [sys, gpu, sess, act, sched, missionData, nodeData] = await Promise.all([
      fetchApi<SystemStats>("system").catch(() => null),
      fetchApi<GpuProcesses>("gpu-processes").catch(() => null),
      fetchApi<{ sessions: Session[] }>("sessions?limit=10").catch(() => null),
      fetchApi<{ activities: Activity[] }>("activity?limit=15").catch(() => null),
      fetchApi<SchedulerStatus>("scheduler").catch(() => null),
      fetchApi<MissionStatus>("missions?limit=20").catch(() => null),
      fetchApi<{ nodes: NodeInfo[] }>("nodes").catch(() => null),
    ]);

    setSystem(sys);
    setGpuProcesses(gpu && Array.isArray(gpu.processes) ? gpu : null);
    setSessions(sess?.sessions || []);
    setActivity(act?.activities || []);
    setScheduler(sched);
    setMissions(missionData);
    setNodes(nodeData?.nodes || []);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, refreshInterval);
    return () => clearInterval(interval);
  }, [refresh, refreshInterval]);

  return { system, gpuProcesses, sessions, activity, scheduler, missions, nodes, gatewayStatus, loading, refresh };
}
