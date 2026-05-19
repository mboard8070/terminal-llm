import { FC, useState, useEffect, useRef } from "react";
import { getGatewayUrl } from "../../lib/gateway";

interface Message {
  id: number;
  from: string;
  text: string;
  date: number;
  outgoing: boolean;
}


export const Messages: FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    setMessages([{
      id: 1, from: "MAUDE", text: "Telegram integration ready. Messages from the Telegram bot will appear here.", date: Date.now() / 1000, outgoing: false,
    }]);
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const text = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { id: Date.now(), from: "You", text, date: Date.now() / 1000, outgoing: true }]);
    setLoading(true);
    try {
      const resp = await fetch(`${getGatewayUrl()}/v1/chat/completions`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "nemotron-super",
          messages: [{ role: "system", content: "You are MAUDE. Respond briefly and helpfully, like a text message." }, { role: "user", content: text }],
          max_tokens: 500, stream: false,
        }),
      });
      if (resp.ok) {
        const data = await resp.json();
        const reply = data.choices?.[0]?.message?.content || "...";
        setMessages((prev) => [...prev, { id: Date.now() + 1, from: "MAUDE", text: reply, date: Date.now() / 1000, outgoing: false }]);
      }
    } catch { /* ignore */ }
    setLoading(false);
  };

  return (
    <div className="flex h-full flex-col bg-maude-bg">
      <div className="flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{"\u2709"}</span>
          <h1 className="text-sm font-semibold text-maude-text">Messages</h1>
        </div>
        <span className="rounded-full bg-maude-bg px-2 py-0.5 text-[10px] text-maude-muted">Telegram</span>
      </div>

      <div ref={scrollRef} className="no-scrollbar flex-1 overflow-y-auto px-4 py-4">
        {messages.map((msg) => (
          <div key={msg.id} className={`mb-3 flex ${msg.outgoing ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] rounded-2xl px-4 py-2.5 ${msg.outgoing ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`}>
              {!msg.outgoing && <div className="mb-0.5 text-[10px] font-medium text-maude-accent">{msg.from}</div>}
              <p className="text-sm">{msg.text}</p>
              <div className="mt-1 text-[10px] opacity-50">{new Date(msg.date * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start"><div className="rounded-2xl bg-maude-surface px-4 py-3"><div className="flex gap-1">
            <span className="h-2 w-2 animate-bounce rounded-full bg-maude-muted" style={{ animationDelay: "0ms" }} />
            <span className="h-2 w-2 animate-bounce rounded-full bg-maude-muted" style={{ animationDelay: "150ms" }} />
            <span className="h-2 w-2 animate-bounce rounded-full bg-maude-muted" style={{ animationDelay: "300ms" }} />
          </div></div></div>
        )}
      </div>

      <div className="flex items-center gap-2 border-t border-maude-border bg-maude-surface p-3">
        <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") sendMessage(); }}
          placeholder="Message..." className="min-h-[44px] flex-1 rounded-xl bg-maude-bg px-4 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent" />
        <button onClick={sendMessage} disabled={!input.trim() || loading} className="flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30">{"\u2191"}</button>
      </div>
    </div>
  );
};
