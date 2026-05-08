import { FC, useEffect, useRef, useState } from "react";
import { ChatMessage, TraceInfo, ToolStep } from "../hooks/useChat";

function useTypewriter(content: string, active: boolean): string {
  const [pos, setPos] = useState(0);
  const rafRef = useRef<number>(0);
  const wasActive = useRef(false);

  if (active) wasActive.current = true;

  useEffect(() => {
    // Never animated — render full content instantly (history messages)
    if (!active && !wasActive.current) { setPos(content.length); return; }

    // Animate: either currently streaming, or finishing reveal after stream ended
    const len = content.length;
    let last = 0;
    const tick = (now: number) => {
      if (now - last >= 16) {
        last = now;
        setPos((p) => {
          if (p >= len) return p;
          return p + Math.max(2, Math.floor((len - p) / 30));
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [content, active]);

  return content.slice(0, pos);
}

interface Props {
  message: ChatMessage;
  animate?: boolean;
}

function renderMarkdown(text: string): string {
  // Resolve relative /download/ URLs to absolute gateway URLs
  const baseUrl = `${window.location.protocol}//${window.location.host}`;
  let html = text
    // Images: ![alt](url) — resolve relative URLs and add error handling + max-height
    .replace(
      /!\[([^\]]*)\]\(([^)]+)\)/g,
      (_match: string, alt: string, url: string) => {
        const resolvedUrl = url.startsWith("/") ? `${baseUrl}${url}` : url;
        return `<img src="${resolvedUrl}" alt="${alt}" style="max-width:100%; max-height:50vh; border-radius:8px; margin:8px 0; object-fit:contain;" loading="lazy" onerror="this.style.display='none'" />`;
      },
    )
    // Links: [text](url)
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener" class="text-blue-400 underline">$1</a>',
    )
    .replace(
      /```(\w*)\n([\s\S]*?)```/g,
      '<pre class="my-2 rounded-lg bg-[#0d1117] p-3 text-sm overflow-x-auto"><code class="text-green-300">$2</code></pre>',
    )
    .replace(/`([^`]+)`/g, '<code class="rounded bg-[#0d1117] px-1.5 py-0.5 text-sm text-orange-300">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/^- (.+)$/gm, '<li class="ml-4 list-disc">$1</li>')
    .replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>')
    .replace(/\n/g, "<br/>");
  return html;
}

const TOOL_VERBS: Record<string, string> = {
  web_search: "searched the web",
  web_browse: "browsed a page",
  run_command: "ran a command",
  read_file: "read a file",
  write_file: "wrote a file",
  edit_file: "edited a file",
  list_directory: "listed a directory",
  gmail_list: "checked email",
  gmail_read: "read an email",
  gmail_send: "sent an email",
  calendar_list_events: "checked calendar",
  calendar_create_event: "created an event",
  drive_list: "browsed Drive",
  drive_search: "searched Drive",
  drive_create_doc: "created a doc",
  contacts_list: "looked up contacts",
  contacts_search: "searched contacts",
  youtube_search: "searched YouTube",
  web_image_search: "searched for images",
  generate_image: "generated an image",
  share_file: "shared a file",
  view_image: "viewed an image",
  dispatch_task: "dispatched a task",
  run_agent: "spawned an agent",
  run_agents: "spawned agents",
  execute_plan: "ran plan mode",
  change_directory: "changed directory",
  get_working_directory: "checked directory",
};

function buildToolSummary(steps: ToolStep[]): string {
  // Count each tool, then describe with friendly verbs
  const counts = new Map<string, number>();
  for (const s of steps.filter((step) => !step.kind || step.kind === "tool")) {
    counts.set(s.name, (counts.get(s.name) || 0) + 1);
  }
  const parts: string[] = [];
  for (const [name, count] of counts) {
    const verb = TOOL_VERBS[name] || name.replace(/_/g, " ");
    if (count > 1) {
      // Pluralize: "read a file" → "read 3 files", "ran a command" → "ran 2 commands"
      const plural = verb.replace(/(?:a |an )(\w+)$/, `${count} $1s`);
      parts.push(plural === verb ? `${verb} x${count}` : plural);
    } else {
      parts.push(verb);
    }
  }
  if (parts.length <= 2) return parts.join(" and ");
  return parts.slice(0, -1).join(", ") + ", and " + parts[parts.length - 1];
}

const TOOL_ICONS: Record<string, string> = {
  model_route: "\u21C4",
  parallel_start: "\u2225",
  context_trim: "\u25F1",
  web_search: "\uD83D\uDD0D",
  web_browse: "\uD83C\uDF10",
  run_command: "\u26A1",
  read_file: "\uD83D\uDCC4",
  write_file: "\u270F\uFE0F",
  list_directory: "\uD83D\uDCC2",
  gmail_list: "\uD83D\uDCE7",
  gmail_read: "\uD83D\uDCE7",
  gmail_send: "\uD83D\uDCE8",
  calendar_list_events: "\uD83D\uDCC5",
  calendar_create_event: "\uD83D\uDCC5",
  drive_list: "\uD83D\uDCBE",
  drive_search: "\uD83D\uDCBE",
  drive_create_doc: "\uD83D\uDCC4",
  contacts_list: "\uD83D\uDC64",
  contacts_search: "\uD83D\uDC64",
  youtube_search: "\u25B6\uFE0F",
  web_image_search: "\uD83D\uDDBC\uFE0F",
  generate_image: "\uD83C\uDFA8",
  share_file: "\uD83D\uDCE4",
  view_image: "\uD83D\uDC41\uFE0F",
};

const ToolActivity: FC<{ steps: ToolStep[]; streaming: boolean; contentStarted: boolean }> = ({ steps, streaming, contentStarted }) => {
  if (!steps.length) return null;

  const anyRunning = steps.some((s) => s.status === "running");
  const finalToolSummary = buildToolSummary(steps);

  return (
    <div className="mb-2 space-y-1">
      {steps.map((step, i) => {
        const icon = TOOL_ICONS[step.name] || "\u2699\uFE0F";
        const isRunning = step.status === "running";
        const isError = step.status === "error";
        const borderColor = isRunning ? "border-cyan-400/40" : isError ? "border-red-400/40" : "border-cyan-500/20";

        return (
          <div
            key={`${step.name}-${i}`}
            className={`border-l-2 ${borderColor} pl-2.5 py-0.5 transition-all duration-300`}
            style={{ animation: streaming ? "fadeSlideIn 0.3s ease-out" : "none" }}
          >
            <div className="flex items-center gap-1.5">
              {isRunning && streaming ? (
                <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-cyan-300/30 border-t-cyan-300" />
              ) : (
                <span className="text-[11px]">{icon}</span>
              )}
              <span className="text-[11px] font-semibold text-cyan-300">{step.task || step.name}</span>
              {isRunning && (
                <span className="animate-pulse text-[10px] font-medium text-cyan-300">still working</span>
              )}
              {step.elapsed !== undefined && (
                <span className="ml-auto font-mono text-[10px] text-maude-muted">{step.elapsed.toFixed(1)}s</span>
              )}
            </div>
            {step.task && (!step.kind || step.kind === "tool") && (
              <div className="truncate font-mono text-[10px] leading-tight text-maude-muted">{step.name}</div>
            )}
            {step.args && (
              <div className="truncate font-mono text-[10px] leading-tight text-maude-muted">{step.args}</div>
            )}
            {step.result && (
              <div className={`truncate font-mono text-[10px] leading-tight ${isError ? "text-red-400" : "text-green-400/80"}`}>
                {isError ? "\u2717 " : "\u2713 "}{step.result}
              </div>
            )}
          </div>
        );
      })}
      {streaming && !anyRunning && !contentStarted && (
        <div className="flex items-center gap-1.5 border-l-2 border-cyan-400/20 py-1 pl-2.5"
             style={{ animation: "fadeSlideIn 0.3s ease-out" }}>
          <span className="inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50" style={{ animationDelay: "0ms" }} />
          <span className="inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50" style={{ animationDelay: "150ms" }} />
          <span className="inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50" style={{ animationDelay: "300ms" }} />
          <span className="animate-pulse text-[10px] text-cyan-400/50">thinking</span>
        </div>
      )}
      {!streaming && finalToolSummary && (
        <div className="mt-1 border-l-2 border-green-400/30 py-0.5 pl-2.5">
          <span className="text-[10px] text-green-400/70">
            {"\u2713 "}
            {finalToolSummary}
            {(() => {
              const total = steps.reduce((sum, s) => sum + (s.elapsed || 0), 0);
              return total > 0 ? ` \u2014 ${total.toFixed(1)}s` : "";
            })()}
          </span>
        </div>
      )}
    </div>
  );
};

const TraceBadge: FC<{ trace: TraceInfo }> = ({ trace }) => {
  const totalInput = trace.promptTokens + trace.cacheReadTokens + trace.cacheCreateTokens;
  if (!totalInput && !trace.tools.length && !trace.route) return null;

  const cachePct = totalInput > 0
    ? Math.round((trace.cacheReadTokens / totalInput) * 100)
    : 0;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-maude-muted">
      {trace.route && (
        <span className="rounded bg-maude-bg px-1.5 py-0.5 text-cyan-300">
          {trace.route.requestedModel && trace.route.requestedModel !== trace.route.resolvedModel
            ? `${trace.route.requestedModel} -> ${trace.route.resolvedModel}`
            : trace.route.resolvedModel || trace.route.requestedModel}
        </span>
      )}
      {trace.tools.length > 0 && (
        <span className="rounded bg-maude-bg px-1.5 py-0.5">
          {trace.tools.length} tool{trace.tools.length > 1 ? "s" : ""}
        </span>
      )}
      {totalInput + trace.completionTokens > 0 && (
        <span className="rounded bg-maude-bg px-1.5 py-0.5">
          {totalInput + trace.completionTokens} tok
        </span>
      )}
      {cachePct > 0 && (
        <span className="rounded bg-maude-bg px-1.5 py-0.5 text-green-400">
          {cachePct}% cached
        </span>
      )}
      {trace.elapsed > 0 && (
        <span className="rounded bg-maude-bg px-1.5 py-0.5">
          {trace.elapsed.toFixed(1)}s
        </span>
      )}
    </div>
  );
};

const MODEL_LABELS: Record<string, string> = {
  "claude-opus-4-20250514": "Claude Opus",
  "claude-sonnet-4-20250514": "Claude Sonnet",
  "mistral-large-latest": "Mistral Large",
  "codestral-latest": "Codestral",
  "devstral-2512": "Devstral",
  "devstral-small-latest": "Devstral Small",
  "devstral-medium-latest": "Devstral Medium",
  "nemotron": "Nemotron",
  "nemotron-super": "Nemotron Super",
  "nemotron-a3b": "Nemotron A3B",
  "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron Super",
  "nvidia/nemotron-3-nano-30b-a3b": "Nemotron A3B",
  "nemotron-nano": "Nemotron A3B",
  "a3b": "Nemotron A3B",
  "gemma-4-31b": "Gemma 4",
  "llava": "LLaVA",
};

export const MessageBubble: FC<Props> = ({ message, animate }) => {
  const isUser = message.role === "user";
  const displayedContent = useTypewriter(message.content, !!animate);
  const hasToolSteps = !isUser && message.toolSteps && message.toolSteps.length > 0;
  const showDots = !message.content && !isUser && !hasToolSteps;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${isUser ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`}>
        {message.model && !isUser && (
          <div className="mb-1 text-[10px] font-medium tracking-wider text-maude-muted">{MODEL_LABELS[message.model] || message.model}</div>
        )}
        {/* Render attached images (supports both legacy imageUrl and new imageUrls) */}
        {(() => {
          const urls = message.imageUrls || (message.imageUrl ? [message.imageUrl] : []);
          if (!urls.length) return null;
          const base = `${window.location.protocol}//${window.location.host}`;
          return (
            <div className={`mb-2 flex gap-2 ${urls.length > 1 ? "overflow-x-auto" : ""}`}>
              {urls.map((url, i) => (
                <img
                  key={url}
                  src={`${base}${url}`}
                  alt={`Attached photo ${i + 1}`}
                  className={`rounded-lg ${urls.length > 1 ? "h-32 w-32 shrink-0 object-cover" : "max-w-full"}`}
                  loading="lazy"
                />
              ))}
            </div>
          );
        })()}
        {hasToolSteps && <ToolActivity steps={message.toolSteps!} streaming={!!animate} contentStarted={!!message.content} />}
        {displayedContent && (
          <div className="break-words text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMarkdown(displayedContent) }} />
        )}
        {!isUser && message.trace && <TraceBadge trace={message.trace} />}
        {showDots && (
          <div className="flex gap-1">
            <span className="h-2 w-2 animate-bounce rounded-full bg-maude-muted" style={{ animationDelay: "0ms" }} />
            <span className="h-2 w-2 animate-bounce rounded-full bg-maude-muted" style={{ animationDelay: "150ms" }} />
            <span className="h-2 w-2 animate-bounce rounded-full bg-maude-muted" style={{ animationDelay: "300ms" }} />
          </div>
        )}
      </div>
    </div>
  );
};
