import { FC, useState } from "react";
import {
  useCommandCenter,
  SystemStats,
  GpuProcesses,
  Session,
  Activity,
  SchedulerStatus,
  Mission,
  NodeInfo,
} from "./useCommandCenter";

// ── Stat Card ────────────────────────────────────────────────

const StatCard: FC<{ label: string; value: string | number; sub?: string; color?: string }> = ({
  label, value, sub, color = "text-maude-accent",
}) => (
  <div className="rounded-xl bg-maude-surface p-3">
    <p className="text-[10px] uppercase tracking-wider text-maude-muted">{label}</p>
    <p className={`text-xl font-bold ${color}`}>{value}</p>
    {sub && <p className="text-[10px] text-maude-muted">{sub}</p>}
  </div>
);

// ── GPU Bar ──────────────────────────────────────────────────

const GpuBar: FC<{ processes: GpuProcesses }> = ({ processes }) => {
  const pct = processes.total_mb > 0 ? (processes.used_mb / processes.total_mb) * 100 : 0;
  return (
    <div className="rounded-xl bg-maude-surface p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-medium text-maude-text">GPU Memory</p>
        <p className="text-xs text-maude-muted">
          {(processes.used_mb / 1024).toFixed(1)}GB / {(processes.total_mb / 1024).toFixed(0)}GB
        </p>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-maude-bg">
        <div className="h-full rounded-full bg-maude-accent transition-all" style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      {processes.processes.length > 0 && (
        <div className="mt-2 space-y-1">
          {processes.processes.map((p) => (
            <div key={p.pid} className="flex items-center justify-between text-[11px]">
              <span className="truncate text-maude-text">{p.name}</span>
              <span className="text-maude-muted">{(p.memory_mb / 1024).toFixed(1)}GB</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Node Card ────────────────────────────────────────────────

const NodeCard: FC<{ node: NodeInfo }> = ({ node }) => (
  <div className="flex items-center gap-3 rounded-xl bg-maude-surface p-3">
    <span className={`inline-block h-2.5 w-2.5 rounded-full ${node.status === "online" ? "bg-green-400" : node.status === "offline" ? "bg-red-400" : "bg-yellow-400"}`} />
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-maude-text">{node.name}</span>
        <span className="text-[10px] text-maude-muted">{node.type}</span>
      </div>
      {node.services && (
        <div className="mt-1 flex flex-wrap gap-1.5">
          {Object.entries(node.services).map(([svc, up]) => (
            <span key={svc} className={`rounded px-1.5 py-0.5 text-[9px] ${up ? "bg-green-400/10 text-green-400" : "bg-red-400/10 text-red-400"}`}>
              {svc}
            </span>
          ))}
        </div>
      )}
      {node.ip && <p className="mt-0.5 text-[10px] text-maude-muted">{node.os || node.platform || ""} {node.ip}</p>}
    </div>
  </div>
);

// ── Scheduler Card ───────────────────────────────────────────

const TaskCard: FC<{ task: SchedulerStatus["tasks"][0] }> = ({ task }) => (
  <div className="rounded-xl bg-maude-surface p-3">
    <div className="flex items-center gap-2">
      <span className={`inline-block h-2 w-2 rounded-full ${task.enabled ? "bg-green-400" : "bg-gray-500"}`} />
      <span className="text-sm font-medium text-maude-text">{task.name}</span>
      <span className="ml-auto font-mono text-[10px] text-maude-muted">{task.cron}</span>
    </div>
    <p className="mt-1 truncate text-xs text-maude-muted">{task.prompt}</p>
    <div className="mt-1 flex gap-3 text-[10px] text-maude-muted">
      <span>{task.run_count} runs</span>
      {task.last_run && <span>Last: {new Date(task.last_run).toLocaleDateString()}</span>}
    </div>
  </div>
);

// ── Mission Card ─────────────────────────────────────────────

const MissionCard: FC<{ mission: Mission }> = ({ mission }) => {
  const total = mission.progress.total || 0;
  const done = mission.progress.done || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const statusColor =
    mission.status === "blocked"
      ? "bg-red-400"
      : mission.status === "complete"
        ? "bg-green-400"
        : mission.status === "paused"
          ? "bg-yellow-400"
          : "bg-maude-accent";

  return (
    <div className="rounded-xl bg-maude-surface p-3">
      <div className="flex items-start gap-2">
        <span className={`mt-1 inline-block h-2.5 w-2.5 shrink-0 rounded-full ${statusColor}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium text-maude-text">{mission.title}</span>
            <span className="ml-auto shrink-0 text-[10px] uppercase text-maude-muted">{mission.status}</span>
          </div>
          <p className="mt-1 line-clamp-2 text-xs text-maude-muted">{mission.objective}</p>
        </div>
      </div>

      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between text-[10px] text-maude-muted">
          <span>{done}/{total} steps</span>
          {mission.cadence && <span className="truncate pl-2">{mission.cadence}</span>}
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-maude-bg">
          <div className="h-full rounded-full bg-maude-accent transition-all" style={{ width: `${Math.min(pct, 100)}%` }} />
        </div>
      </div>

      {mission.next_action && (
        <div className="mt-3 rounded-lg bg-maude-bg px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-wider text-maude-muted">Next</p>
          <p className="line-clamp-2 text-xs text-maude-text">{mission.next_action}</p>
        </div>
      )}

      {!!mission.blockers?.length && (
        <div className="mt-2 space-y-1">
          {mission.blockers.slice(0, 2).map((blocker) => (
            <p key={blocker} className="rounded bg-red-400/10 px-2 py-1 text-[11px] text-red-300">
              {blocker}
            </p>
          ))}
        </div>
      )}

      <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
        {mission.schedule?.task_id && (
          <span className="rounded bg-green-400/10 px-1.5 py-0.5 text-green-400">
            {mission.schedule.cron || "scheduled"}
          </span>
        )}
        {!!mission.artifacts?.length && (
          <span className="rounded bg-maude-bg px-1.5 py-0.5 text-maude-muted">
            {mission.artifacts.length} artifacts
          </span>
        )}
        {mission.updated_at && (
          <span className="rounded bg-maude-bg px-1.5 py-0.5 text-maude-muted">
            {new Date(mission.updated_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
};

// ── Activity Item ────────────────────────────────────────────

const ActivityItem: FC<{ item: Activity }> = ({ item }) => (
  <div className="flex items-start gap-2 py-2">
    <span className={`mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full ${item.role === "user" ? "bg-green-400" : "bg-maude-accent"}`} />
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] font-medium uppercase text-maude-muted">{item.channel}</span>
        <span className="text-[10px] text-maude-muted">{item.role}</span>
      </div>
      <p className="truncate text-xs text-maude-text">{item.content}</p>
    </div>
  </div>
);

// ── Session Item ─────────────────────────────────────────────

const SessionItem: FC<{ session: Session }> = ({ session }) => (
  <div className="flex items-center justify-between rounded-xl bg-maude-surface p-3">
    <div>
      <span className="text-sm font-medium text-maude-text">{session.session_id.slice(0, 8)}</span>
      <span className="ml-2 text-[10px] text-maude-muted">{session.channel}</span>
    </div>
    <div className="text-right">
      <p className="text-xs text-maude-muted">{session.message_count} msgs</p>
      <p className="text-[10px] text-maude-muted">{new Date(session.last_message_at).toLocaleDateString()}</p>
    </div>
  </div>
);

// ── Main Page ────────────────────────────────────────────────

type TabKey = "overview" | "missions" | "nodes" | "activity" | "scheduler";

export const CommandCenter: FC = () => {
  const { system, gpuProcesses, sessions, activity, scheduler, missions, nodes, gatewayStatus, loading, refresh } = useCommandCenter();
  const [tab, setTab] = useState<TabKey>("overview");

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-maude-muted">
        Loading command center...
      </div>
    );
  }

  const tabs: { key: TabKey; label: string }[] = [
    { key: "overview", label: "Overview" },
    { key: "missions", label: "Missions" },
    { key: "nodes", label: "Nodes" },
    { key: "activity", label: "Activity" },
    { key: "scheduler", label: "Tasks" },
  ];

  const gpuTemp = typeof system?.gpu?.temperature_c === "number" ? system.gpu.temperature_c : 0;
  const tempColor = gpuTemp > 80 ? "text-red-400" : gpuTemp > 60 ? "text-yellow-400" : "text-green-400";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 pt-4 pb-2">
        <h1 className="text-lg font-bold text-maude-text">Command Center</h1>
        <button
          onClick={refresh}
          className="ml-auto rounded-lg bg-maude-surface px-2 py-1 text-xs text-maude-muted active:bg-maude-card"
        >
          Refresh
        </button>
      </div>

      {!gatewayStatus.ok && (
        <div className="mx-4 mb-3 rounded-lg border border-red-400/30 bg-red-400/10 p-3">
          <p className="text-sm font-medium text-red-300">Gateway not reachable</p>
          <p className="mt-1 break-all font-mono text-[11px] text-red-200">{gatewayStatus.url}</p>
          {gatewayStatus.error && <p className="mt-1 text-xs text-red-200">{gatewayStatus.error}</p>}
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 px-4 pb-3">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              tab === t.key ? "bg-maude-accent text-white" : "bg-maude-surface text-maude-muted"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">

        {tab === "overview" && (
          <div className="space-y-3">
            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-2">
              <StatCard label="CPU" value={`${system?.cpu_percent ?? 0}%`} sub={`${system?.ram?.used_gb ?? 0}/${system?.ram?.total_gb ?? 0}GB RAM`} />
              <StatCard label="GPU Temp" value={`${gpuTemp}\u00B0C`} sub={system?.gpu?.name || "N/A"} color={tempColor} />
              <StatCard label="Disk" value={`${system?.disk?.percent ?? 0}%`} sub={`${system?.disk?.used_gb ?? 0}/${system?.disk?.total_gb ?? 0}GB`} />
              <StatCard label="Missions" value={missions?.stats?.active ?? 0} sub={`${scheduler?.stats?.active ?? 0} scheduled tasks`} />
            </div>

            {/* GPU memory */}
            {gpuProcesses && <GpuBar processes={gpuProcesses} />}

            {/* Recent sessions */}
            {sessions.length > 0 && (
              <>
                <p className="pt-1 text-xs font-semibold uppercase tracking-wider text-maude-muted">Recent Sessions</p>
                <div className="space-y-1.5">
                  {sessions.slice(0, 5).map((s) => (
                    <SessionItem key={s.session_id + s.channel} session={s} />
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {tab === "missions" && (
          <div className="space-y-2">
            {missions?.stats && (
              <div className="grid grid-cols-4 gap-2">
                <StatCard label="Total" value={missions.stats.total} />
                <StatCard label="Active" value={missions.stats.active} color="text-green-400" />
                <StatCard label="Blocked" value={missions.stats.blocked} color="text-red-400" />
                <StatCard label="Sched" value={missions.stats.scheduled} />
              </div>
            )}
            {!missions?.missions?.length ? (
              <p className="py-8 text-center text-sm text-maude-muted">No missions</p>
            ) : (
              missions.missions.map((mission) => <MissionCard key={mission.id} mission={mission} />)
            )}
          </div>
        )}

        {tab === "nodes" && (
          <div className="space-y-2">
            {nodes.length === 0 ? (
              <p className="py-8 text-center text-sm text-maude-muted">No nodes detected</p>
            ) : (
              nodes.map((n, i) => <NodeCard key={n.name + i} node={n} />)
            )}
          </div>
        )}

        {tab === "activity" && (
          <div className="divide-y divide-maude-border">
            {activity.length === 0 ? (
              <p className="py-8 text-center text-sm text-maude-muted">No recent activity</p>
            ) : (
              activity.map((a, i) => <ActivityItem key={i} item={a} />)
            )}
          </div>
        )}

        {tab === "scheduler" && (
          <div className="space-y-2">
            {scheduler?.stats && (
              <div className="grid grid-cols-3 gap-2">
                <StatCard label="Total" value={scheduler.stats.total} />
                <StatCard label="Active" value={scheduler.stats.active} color="text-green-400" />
                <StatCard label="Runs" value={scheduler.stats.total_runs} />
              </div>
            )}
            {!scheduler?.tasks?.length ? (
              <p className="py-8 text-center text-sm text-maude-muted">No scheduled tasks</p>
            ) : (
              scheduler.tasks.map((t) => <TaskCard key={t.id} task={t} />)
            )}
          </div>
        )}
      </div>
    </div>
  );
};
