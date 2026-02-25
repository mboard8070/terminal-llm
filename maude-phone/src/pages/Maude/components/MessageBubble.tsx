import { FC } from "react";
import { ChatMessage } from "../hooks/useChat";

interface Props {
  message: ChatMessage;
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

export const MessageBubble: FC<Props> = ({ message }) => {
  const isUser = message.role === "user";

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
        <div className="break-words text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} />
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
