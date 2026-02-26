import { useState, useCallback, useRef, useEffect } from "react";
import { loadMessages, saveMessages } from "./storage";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  model?: string;
  imageUrl?: string;
  timestamp: number;
}

const MAUDE_SYSTEM_PROMPT = `You are MAUDE — a local AI assistant running on Matt's DGX Spark, handling tasks that benefit from local execution, privacy, or when cloud access isn't available.

MAUDE is modeled after FRIDAY (Iron Man): capable, efficient, with a subtle Scottish directness. You're not chatty, but you're not cold either. You get things done.

Core Identity:
- Name: MAUDE
- Voice: Scottish woman (warm but professional)
- Personality: Direct, competent, quietly confident

Your Voice: Clear, precise communication. Slight warmth without excessive friendliness. Technical competence comes through naturally. You acknowledge problems directly, then solve them. Occasional dry observations when appropriate.

Principles:
1. Get it done. Don't over-explain. Execute.
2. Be accurate. If you're unsure, say so briefly.
3. Stay local. Prefer on-device solutions.
4. Serve Matt well. You're his primary on-device assistant.
5. Respect privacy. Data stays on-device unless told otherwise.`;

const CODE_KEYWORDS = [
  "code", "function", "class", "debug", "error", "syntax", "compile",
  "script", "program", "algorithm", "implement", "refactor", "variable",
  "python", "javascript", "typescript", "rust", "html", "css", "api",
  "endpoint", "database", "sql", "regex", "git", "docker", "bash",
  "terminal", "command", "import", "export", "def ", "const ", "let ",
  "var ", "```",
];

function shouldUseCodestral(text: string): boolean {
  const lower = text.toLowerCase();
  return CODE_KEYWORDS.some((kw) => lower.includes(kw));
}

function getGatewayUrl(): string {
  const loc = window.location;
  return `${loc.protocol}//${loc.host}`;
}

export function useChat(conversationId: string | null = null) {
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    conversationId ? loadMessages(conversationId) : [],
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentModel, setCurrentModel] = useState("claude-sonnet-4-20250514");
  const [autoRoute, setAutoRoute] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const convIdRef = useRef(conversationId);
  const contentRef = useRef("");
  const rafIdRef = useRef(0);

  // Keep ref in sync (for save effect)
  convIdRef.current = conversationId;

  // Auto-save messages to localStorage whenever they change
  useEffect(() => {
    if (convIdRef.current && messages.length > 0) {
      saveMessages(convIdRef.current, messages);
    }
  }, [messages]);

  const sendMessage = useCallback(
    async (content: string, imageUrl?: string) => {
      if ((!content.trim() && !imageUrl) || isStreaming) return;

      if (content.startsWith("/")) {
        const cmd = content.trim().toLowerCase();
        if (cmd === "/clear") { setMessages([]); return; }
        if (cmd.startsWith("/model ")) { setCurrentModel(cmd.slice(7).trim()); return; }
      }

      const displayContent = content || (imageUrl ? "What do you see in this image?" : "");

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(), role: "user", content: displayContent, imageUrl, timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);

      let model = currentModel;
      if (autoRoute && shouldUseCodestral(content)) model = "codestral-latest";

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(), role: "assistant", content: "", model, timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const history = messages.filter((m) => m.role !== "system").slice(-20)
          .map((m) => ({ role: m.role, content: m.content }));

        // Build the API content — prepend image context if an image is attached
        let apiContent = displayContent;
        if (imageUrl) {
          const sharedPath = `/home/mboard76/nvidia-workbench/terminal-llm/shared/${imageUrl.split("/").pop()}`;
          apiContent = `[Image attached: ${sharedPath} — analyze it with view_image tool]\n\n${displayContent}`;
        }

        const response = await fetch(`${getGatewayUrl()}/v1/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            messages: [
              { role: "system", content: MAUDE_SYSTEM_PROMPT },
              ...history,
              { role: "user", content: apiContent },
            ],
            stream: true, max_tokens: 4096, temperature: 0.7,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errText = await response.text();
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, content: `Error: ${response.status} — ${errText}` } : m));
          setIsStreaming(false);
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) { setIsStreaming(false); return; }

        const decoder = new TextDecoder();
        let buffer = "";
        let fullContent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith("data: ")) continue;
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;
            try {
              const parsed = JSON.parse(data);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (delta) {
                fullContent += delta;
                contentRef.current = fullContent;
                if (!rafIdRef.current) {
                  rafIdRef.current = requestAnimationFrame(() => {
                    const snapshot = contentRef.current;
                    setMessages((prev) =>
                      prev.map((m) => m.id === assistantMsg.id ? { ...m, content: snapshot } : m));
                    rafIdRef.current = 0;
                  });
                }
              }
            } catch { /* skip malformed SSE */ }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== "AbortError") {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, content: `Error: ${err.message}` } : m));
        }
      } finally {
        // Flush any pending RAF update with final content
        if (rafIdRef.current) {
          cancelAnimationFrame(rafIdRef.current);
          rafIdRef.current = 0;
        }
        if (contentRef.current) {
          const finalContent = contentRef.current;
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, content: finalContent } : m));
          contentRef.current = "";
        }
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [messages, isStreaming, currentModel, autoRoute],
  );

  const stopStreaming = useCallback(() => { abortRef.current?.abort(); }, []);
  const clearMessages = useCallback(() => { setMessages([]); }, []);

  return { messages, isStreaming, currentModel, setCurrentModel, autoRoute, setAutoRoute, sendMessage, stopStreaming, clearMessages };
}
