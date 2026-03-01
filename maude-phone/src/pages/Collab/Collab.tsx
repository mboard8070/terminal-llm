import { FC, useState } from "react";
import {
  useCollab,
  PresenceEntry,
  ActivityEvent,
  Project,
  Task,
} from "./useCollab";

function timeAgo(ts: number, now: number): string {
  const diff = Math.max(0, Math.floor(now - ts));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-yellow-500",
  running: "bg-blue-500",
  completed: "bg-green-500",
  failed: "bg-red-500",
};

const CLIENT_ICONS: Record<string, string> = {
  gateway: "\u{2B21}",
  tui: ">_",
  cli: "$",
  macos: "\u{1F4BB}",
  mac: "\u{1F4BB}",
  iphone: "\u{1F4F1}",
  ipad: "\u{1F4F2}",
  android: "\u{1F4F1}",
  "android-tablet": "\u{1F4F2}",
  phone: "\u{1F4F1}",
  windows: "\u{1F5A5}",
  unknown: "\u{25CF}",
};

// ── Presence Card ─────────────────────────────────────────────

const PresenceCard: FC<{ entry: PresenceEntry; now: number }> = ({ entry, now }) => (
  <div className="flex items-center gap-3 rounded-xl bg-maude-surface p-3">
    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-maude-card text-lg">
      {CLIENT_ICONS[entry.client_type] || CLIENT_ICONS.unknown}
    </div>
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-2">
        <span className="font-medium text-maude-text">{entry.hostname}</span>
        <span className="text-[10px] text-maude-muted">{entry.client_type}</span>
        <span className="ml-auto inline-block h-2 w-2 rounded-full bg-green-400" />
      </div>
      <p className="truncate text-xs text-maude-muted">
        {entry.activity || "idle"} &middot; {timeAgo(entry.last_seen, now)}
      </p>
    </div>
  </div>
);

// ── Activity Event ────────────────────────────────────────────

const EVENT_ICONS: Record<string, string> = {
  chat: "\u{1F4AC}",
  task_dispatched: "\u{1F680}",
  project_created: "\u{1F4C1}",
  custom: "\u{2022}",
};

const ActivityItem: FC<{ event: ActivityEvent; now: number }> = ({ event, now }) => (
  <div className="flex items-start gap-2 py-1.5">
    <span className="mt-0.5 text-sm">{EVENT_ICONS[event.type] || EVENT_ICONS.custom}</span>
    <div className="min-w-0 flex-1">
      <p className="text-sm text-maude-text">{event.summary}</p>
      <p className="text-[10px] text-maude-muted">
        {event.hostname} &middot; {timeAgo(event.ts, now)}
      </p>
    </div>
  </div>
);

// ── Project Card ──────────────────────────────────────────────

const ProjectCard: FC<{ project: Project }> = ({ project }) => (
  <div className="rounded-xl bg-maude-surface p-3">
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium text-maude-text">{project.name}</span>
      {project.tags.map((tag) => (
        <span key={tag} className="rounded bg-maude-card px-1.5 py-0.5 text-[10px] text-maude-muted">
          {tag}
        </span>
      ))}
    </div>
    {project.description && (
      <p className="mt-1 text-xs text-maude-muted">{project.description}</p>
    )}
    <div className="mt-2 flex gap-3 text-[10px] text-maude-muted">
      <span>{project.conversations.length} conversations</span>
      <span>{project.files.length} files</span>
      <span>{project.hostname}</span>
    </div>
  </div>
);

// ── Task Card ─────────────────────────────────────────────────

const TaskCard: FC<{ task: Task; now: number }> = ({ task, now }) => (
  <div className="rounded-xl bg-maude-surface p-3">
    <div className="flex items-center gap-2">
      <span className={`inline-block h-2 w-2 rounded-full ${STATUS_COLORS[task.status] || "bg-gray-500"}`} />
      <span className="text-[10px] font-medium uppercase text-maude-muted">{task.status}</span>
      <span className="ml-auto text-[10px] text-maude-muted">{timeAgo(task.created_at, now)}</span>
    </div>
    <p className="mt-1 truncate text-sm text-maude-text">{task.prompt}</p>
    <div className="mt-1 flex gap-2 text-[10px] text-maude-muted">
      <span>{task.source} → {task.target || "local"}</span>
      <span>{task.capability}</span>
    </div>
    {task.result && (
      <pre className="mt-2 max-h-20 overflow-auto rounded bg-maude-card p-2 text-[10px] text-maude-text">
        {task.result.slice(0, 300)}
      </pre>
    )}
  </div>
);

// ── Main Collab Page ──────────────────────────────────────────

type TabKey = "presence" | "activity" | "projects" | "tasks";

export const Collab: FC = () => {
  const { status, loading } = useCollab();
  const [activeTab, setActiveTab] = useState<TabKey>("presence");

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-maude-muted">
        Loading collaboration status...
      </div>
    );
  }

  if (!status) {
    return (
      <div className="flex h-full items-center justify-center text-maude-muted">
        Unable to connect to gateway
      </div>
    );
  }

  const now = status.ts;
  const tabs: { key: TabKey; label: string; count: number }[] = [
    { key: "presence", label: "Online", count: status.presence.length },
    { key: "activity", label: "Activity", count: status.activity.length },
    { key: "projects", label: "Projects", count: status.projects.length },
    { key: "tasks", label: "Tasks", count: status.tasks.length },
  ];

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 pt-4 pb-2">
        <h1 className="text-lg font-bold text-maude-text">Collaboration</h1>
        <span className="ml-auto flex items-center gap-1 text-xs text-maude-muted">
          <span className="inline-block h-2 w-2 rounded-full bg-green-400" />
          {status.hostname}
        </span>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 px-4 pb-3">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              activeTab === tab.key
                ? "bg-maude-accent text-white"
                : "bg-maude-surface text-maude-muted"
            }`}
          >
            {tab.label}
            {tab.count > 0 && (
              <span className="ml-1 opacity-70">{tab.count}</span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {activeTab === "presence" && (
          <div className="flex flex-col gap-2">
            {status.presence.length === 0 ? (
              <p className="py-8 text-center text-sm text-maude-muted">No devices online</p>
            ) : (
              status.presence.map((p) => (
                <PresenceCard key={p.client_id} entry={p} now={now} />
              ))
            )}
          </div>
        )}

        {activeTab === "activity" && (
          <div className="flex flex-col divide-y divide-maude-border">
            {status.activity.length === 0 ? (
              <p className="py-8 text-center text-sm text-maude-muted">No recent activity</p>
            ) : (
              status.activity.map((evt) => (
                <ActivityItem key={evt.id} event={evt} now={now} />
              ))
            )}
          </div>
        )}

        {activeTab === "projects" && (
          <div className="flex flex-col gap-2">
            {status.projects.length === 0 ? (
              <p className="py-8 text-center text-sm text-maude-muted">No projects yet</p>
            ) : (
              status.projects.map((proj) => (
                <ProjectCard key={proj.id} project={proj} />
              ))
            )}
          </div>
        )}

        {activeTab === "tasks" && (
          <div className="flex flex-col gap-2">
            {status.tasks.length === 0 ? (
              <p className="py-8 text-center text-sm text-maude-muted">No tasks dispatched</p>
            ) : (
              status.tasks.map((task) => (
                <TaskCard key={task.id} task={task} now={now} />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};
