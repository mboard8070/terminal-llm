import { FC, useEffect, useRef, useState } from "react";
import { ChatMessage, TraceInfo } from "../hooks/useChat";

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
  let html = text
    // Images: ![alt](url)
    .replace(
      /!\[([^\]]*)\]\(([^)]+)\)/g,
      '<img src="$2" alt="$1" style="max-width:100%; border-radius:8px; margin:8px 0;" loading="lazy" />',
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

const TraceBadge: FC<{ trace: TraceInfo }> = ({ trace }) => {
  const totalInput = trace.promptTokens + trace.cacheReadTokens + trace.cacheCreateTokens;
  if (!totalInput && !trace.tools.length) return null;

  const cachePct = totalInput > 0
    ? Math.round((trace.cacheReadTokens / totalInput) * 100)
    : 0;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-maude-muted">
      {trace.tools.length > 0 && (
        <span className="rounded bg-maude-bg px-1.5 py-0.5">
          {trace.tools.length} tool{trace.tools.length > 1 ? "s" : ""}
        </span>
      )}
      <span className="rounded bg-maude-bg px-1.5 py-0.5">
        {totalInput + trace.completionTokens} tok
      </span>
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

export const MessageBubble: FC<Props> = ({ message, animate }) => {
  const isUser = message.role === "user";
  const displayedContent = useTypewriter(message.content, !!animate);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${isUser ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`}>
        {message.model && !isUser && (
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-maude-muted">{message.model}</div>
        )}
        {message.imageUrl && (
          <img
            src={`${window.location.protocol}//${window.location.host}${message.imageUrl}`}
            alt="Attached photo"
            className="mb-2 max-w-full rounded-lg"
            loading="lazy"
          />
        )}
        <div className="break-words text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMarkdown(displayedContent) }} />
        {!isUser && message.trace && <TraceBadge trace={message.trace} />}
        {!message.content && !isUser && (
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
