import { useState, useEffect, useCallback } from "react";

function getGatewayUrl(): string {
  return `${window.location.protocol}//${window.location.host}`;
}

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

async function fetchApi<T>(endpoint: string): Promise<T | null> {
  try {
    const r = await fetch(`${getGatewayUrl()}/api/command-center/${endpoint}`);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export function useCommandCenter(refreshInterval = 10000) {
  const [system, setSystem] = useState<SystemStats | null>(null);
  const [gpuProcesses, setGpuProcesses] = useState<GpuProcesses | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [scheduler, setScheduler] = useState<SchedulerStatus | null>(null);
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const [sys, gpu, sess, act, sched, nodeData] = await Promise.all([
      fetchApi<SystemStats>("system"),
      fetchApi<GpuProcesses>("gpu-processes"),
      fetchApi<{ sessions: Session[] }>("sessions?limit=10"),
      fetchApi<{ activities: Activity[] }>("activity?limit=15"),
      fetchApi<SchedulerStatus>("scheduler"),
      fetchApi<{ nodes: NodeInfo[] }>("nodes"),
    ]);

    setSystem(sys);
    setGpuProcesses(gpu);
    setSessions(sess?.sessions || []);
    setActivity(act?.activities || []);
    setScheduler(sched);
    setNodes(nodeData?.nodes || []);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, refreshInterval);
    return () => clearInterval(interval);
  }, [refresh, refreshInterval]);

  return { system, gpuProcesses, sessions, activity, scheduler, nodes, loading, refresh };
}
