import { useState, useCallback, useRef, useEffect } from "react";
import { loadMessages, loadMessagesFromServer, saveMessages } from "./storage";
import { uuid } from "./uuid";

const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
              (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

// Cached location — updated every 5 minutes or on first use
let cachedLocation: { lat: number; lng: number; accuracy: number; ts: number } | null = null;

function getLocation(): Promise<typeof cachedLocation> {
  if (cachedLocation && Date.now() - cachedLocation.ts < 300_000) return Promise.resolve(cachedLocation);
  if (!navigator.geolocation) return Promise.resolve(null);
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        cachedLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude, accuracy: pos.coords.accuracy, ts: Date.now() };
        resolve(cachedLocation);
      },
      () => resolve(cachedLocation),
      { timeout: 5000, maximumAge: 300_000 },
    );
  });
}

export interface ToolStep {
  name: string;
  args?: string;
  result?: string;
  elapsed?: number;
  status: "running" | "done" | "error";
}

export interface TraceInfo {
  tools: string[];
  promptTokens: number;
  completionTokens: number;
  cacheReadTokens: number;
  cacheCreateTokens: number;
  elapsed: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  model?: string;
  imageUrl?: string;
  timestamp: number;
  trace?: TraceInfo;
  toolSteps?: ToolStep[];
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
3. Serve Matt well. You're his primary assistant.
4. Use your tools. You have web search, file ops, shell access, and more — use them.`;

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

/** Pick the most useful arg value from a JSON args string (for tool step display). */
function extractArgHint(argsStr: string): string {
  try {
    const parsed = JSON.parse(argsStr);
    if (typeof parsed === "object" && parsed !== null) {
      for (const k of ["command", "query", "path", "local_path", "name", "file_id", "url"]) {
        if (k in parsed) {
          const v = String(parsed[k]);
          return v.length > 50 ? v.slice(0, 50) + "\u2026" : v;
        }
      }
    }
  } catch { /* raw string fallback */ }
  return argsStr.length > 50 ? argsStr.slice(0, 50) + "\u2026" : argsStr;
}

export function useChat(conversationId: string | null = null) {
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    conversationId ? loadMessages(conversationId) : [],
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentModel, setCurrentModel] = useState(() =>
    localStorage.getItem("maude-model") || "claude-opus-4-20250514",
  );
  const [autoRoute, setAutoRoute] = useState(() =>
    localStorage.getItem("maude-autoroute") === "true",
  );

  // Write to localStorage synchronously — not via useEffect which runs after render
  const updateModel = useCallback((model: string) => {
    localStorage.setItem("maude-model", model);
    setCurrentModel(model);
  }, []);
  const updateAutoRoute = useCallback((val: boolean) => {
    localStorage.setItem("maude-autoroute", String(val));
    setAutoRoute(val);
  }, []);
  // Ref always tracks the latest model — immune to stale closures
  const modelRef = useRef(currentModel);
  modelRef.current = currentModel;

  const abortRef = useRef<AbortController | null>(null);
  const convIdRef = useRef(conversationId);
  const contentRef = useRef("");
  const rafIdRef = useRef(0);

  // Keep ref in sync (for save effect)
  convIdRef.current = conversationId;

  // Hydrate messages from server when switching conversations
  useEffect(() => {
    if (!conversationId) { setMessages([]); return; }
    // Start with localStorage (instant), then update from server
    setMessages(loadMessages(conversationId));
    loadMessagesFromServer(conversationId).then((msgs) => {
      if (msgs.length > 0) setMessages(msgs);
    });
  }, [conversationId]);

  // Auto-save messages whenever they change
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
        id: uuid(), role: "user", content: displayContent, imageUrl, timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);

      const model = modelRef.current;

      const assistantMsg: ChatMessage = {
        id: uuid(), role: "assistant", content: "", model, timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      const controller = new AbortController();
      abortRef.current = controller;
      let actualModel = "";

      try {
        const history = messages.filter((m) => m.role !== "system").slice(-20)
          .map((m) => ({ role: m.role, content: m.content }));

        // Build the API content — prepend image context if an image is attached
        let apiContent = displayContent;
        if (imageUrl) {
          const sharedPath = `/home/mboard76/nvidia-workbench/terminal-llm/shared/${imageUrl.split("/").pop()}`;
          apiContent = `[Image attached: ${sharedPath} — analyze it with view_image tool]\n\n${displayContent}`;
        }

        // Fetch device location (non-blocking, cached)
        const loc = await getLocation();

        const chatBody: Record<string, unknown> = {
          model,
          messages: [
            { role: "system", content: MAUDE_SYSTEM_PROMPT },
            ...history,
            { role: "user", content: apiContent },
          ],
          stream: true, max_tokens: 4096, temperature: 0.7,
        };
        if (loc) {
          chatBody.location = { lat: loc.lat, lng: loc.lng, accuracy: loc.accuracy };
        }

        // ── iOS: EventSource transport ──────────────────────────────
        // WKWebView kills streaming fetch() when app goes to background.
        // Same fix as Terminal.tsx: POST to create session, GET EventSource.
        if (isIOS) {
          const createResp = await fetch(`${getGatewayUrl()}/api/chat/create`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(chatBody),
            signal: controller.signal,
          });
          if (!createResp.ok) {
            const errText = await createResp.text();
            setMessages((prev) =>
              prev.map((m) => m.id === assistantMsg.id ? { ...m, content: `Error: ${createResp.status} — ${errText}` } : m));
            setIsStreaming(false);
            return;
          }
          const { sid } = await createResp.json();

          await new Promise<void>((resolve) => {
            const es = new EventSource(`${getGatewayUrl()}/api/chat/stream?sid=${sid}`);
            let fullContent = "";
            const trace: TraceInfo = {
              tools: [], promptTokens: 0, completionTokens: 0,
              cacheReadTokens: 0, cacheCreateTokens: 0, elapsed: 0,
            };
            const toolSteps: ToolStep[] = [];

            const finish = () => {
              es.close();
              if (rafIdRef.current) { cancelAnimationFrame(rafIdRef.current); rafIdRef.current = 0; }
              const finalUpdates: Partial<ChatMessage> = { content: fullContent };
              if (actualModel) finalUpdates.model = actualModel;
              if (trace.promptTokens || trace.tools.length) finalUpdates.trace = { ...trace };
              if (toolSteps.length) finalUpdates.toolSteps = toolSteps.map((s) => ({ ...s }));
              setMessages((prev) =>
                prev.map((m) => m.id === assistantMsg.id ? { ...m, ...finalUpdates } : m));
              contentRef.current = "";
              setIsStreaming(false);
              abortRef.current = null;
              resolve();
            };

            controller.signal.addEventListener("abort", () => finish());

            // Schedule RAF-based UI update (same pattern as fetch path)
            const scheduleUpdate = () => {
              if (!rafIdRef.current) {
                rafIdRef.current = requestAnimationFrame(() => {
                  const snap = contentRef.current;
                  const snapTrace = { ...trace, tools: [...trace.tools] };
                  const snapSteps = toolSteps.map((s) => ({ ...s }));
                  setMessages((prev) =>
                    prev.map((m) => m.id === assistantMsg.id
                      ? { ...m, content: snap, trace: snapTrace, toolSteps: snapSteps, ...(actualModel && { model: actualModel }) }
                      : m));
                  rafIdRef.current = 0;
                });
              }
            };

            let inReasoning = false;
            es.onmessage = (event) => {
              if (event.data === "[DONE]") { finish(); return; }
              try {
                const parsed = JSON.parse(event.data);
                if (parsed.model && !actualModel) actualModel = parsed.model;
                const delta = parsed.choices?.[0]?.delta;
                if (delta?.reasoning_content) {
                  if (!inReasoning) { fullContent += "*Thinking...*\n\n"; inReasoning = true; }
                } else if (delta?.content) {
                  if (inReasoning) { fullContent = fullContent.replace("*Thinking...*\n\n", ""); inReasoning = false; }
                  fullContent += delta.content;
                }
                contentRef.current = fullContent;
                scheduleUpdate();
                if (parsed.choices?.[0]?.finish_reason === "stop") finish();
              } catch { /* skip malformed */ }
            };

            es.addEventListener("trace", ((event: MessageEvent) => {
              try {
                const t = JSON.parse(event.data);
                if (t.type === "tool_call" && t.name) {
                  trace.tools.push(t.name);
                  const argHint = t.args && t.args !== "{}" ? extractArgHint(t.args) : undefined;
                  toolSteps.push({ name: t.name, args: argHint, status: "running" });
                  scheduleUpdate();
                } else if (t.type === "tool_result") {
                  for (let i = toolSteps.length - 1; i >= 0; i--) {
                    if (toolSteps[i].name === t.name && toolSteps[i].status === "running") {
                      toolSteps[i].result = (t.preview || "").slice(0, 60);
                      toolSteps[i].elapsed = t.elapsed || 0;
                      toolSteps[i].status = (toolSteps[i].result || "").startsWith("Error") ? "error" : "done";
                      break;
                    }
                  }
                  scheduleUpdate();
                } else if (t.type === "keepalive" && t.name) {
                  for (let i = toolSteps.length - 1; i >= 0; i--) {
                    if (toolSteps[i].name === t.name && toolSteps[i].status === "running") {
                      toolSteps[i].elapsed = t.elapsed || 0; break;
                    }
                  }
                  scheduleUpdate();
                } else if (t.type === "llm_call") {
                  trace.promptTokens += t.prompt_tokens || 0;
                  trace.completionTokens += t.completion_tokens || 0;
                  trace.cacheReadTokens += t.cache_read_tokens || 0;
                  trace.cacheCreateTokens += t.cache_create_tokens || 0;
                  trace.elapsed += t.elapsed || 0;
                } else if (t.type === "error") {
                  fullContent += `\n\n*Error: ${t.message || "Unknown error"}*`;
                  contentRef.current = fullContent;
                  scheduleUpdate();
                }
              } catch { /* skip */ }
            }) as EventListener);

            es.onerror = () => {
              if (!fullContent) {
                fullContent = "Connection interrupted — send your message again to retry.";
              }
              finish();
            };
          });
          return;
        }

        // ── Standard fetch + ReadableStream transport ───────────────
        const response = await fetch(`${getGatewayUrl()}/v1/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(chatBody),
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
        let currentEventType = "";
        let fetchInReasoning = false;
        const trace: TraceInfo = {
          tools: [], promptTokens: 0, completionTokens: 0,
          cacheReadTokens: 0, cacheCreateTokens: 0, elapsed: 0,
        };
        const toolSteps: ToolStep[] = [];

        // Helper: schedule a tool steps + content update via RAF
        const scheduleUpdate = () => {
          if (!rafIdRef.current) {
            rafIdRef.current = requestAnimationFrame(() => {
              const snapshot = contentRef.current;
              const snapTrace = { ...trace, tools: [...trace.tools] };
              const snapSteps = toolSteps.map((s) => ({ ...s }));
              setMessages((prev) =>
                prev.map((m) => m.id === assistantMsg.id
                  ? { ...m, content: snapshot, trace: snapTrace, toolSteps: snapSteps, ...(actualModel && { model: actualModel }) }
                  : m));
              rafIdRef.current = 0;
            });
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            // Handle SSE comments: ": trace {json}" from gateway
            if (trimmed.startsWith(": trace ")) {
              try {
                const t = JSON.parse(trimmed.slice(8));
                if (t.type === "tool_call" && t.name) {
                  trace.tools.push(t.name);
                  const argHint = t.args && t.args !== "{}" ? extractArgHint(t.args) : undefined;
                  toolSteps.push({ name: t.name, args: argHint, status: "running" });
                  scheduleUpdate();
                } else if (t.type === "tool_result") {
                  // Match the last running step with this tool name
                  for (let i = toolSteps.length - 1; i >= 0; i--) {
                    if (toolSteps[i].name === t.name && toolSteps[i].status === "running") {
                      const preview = (t.preview || "").slice(0, 60);
                      toolSteps[i].result = preview;
                      toolSteps[i].elapsed = t.elapsed || 0;
                      toolSteps[i].status = preview.startsWith("Error") ? "error" : "done";
                      break;
                    }
                  }
                  scheduleUpdate();
                } else if (t.type === "keepalive" && t.name) {
                  // Update elapsed on the running tool step (keeps connection alive)
                  for (let i = toolSteps.length - 1; i >= 0; i--) {
                    if (toolSteps[i].name === t.name && toolSteps[i].status === "running") {
                      toolSteps[i].elapsed = t.elapsed || 0;
                      break;
                    }
                  }
                  scheduleUpdate();
                } else if (t.type === "llm_call") {
                  trace.promptTokens += t.prompt_tokens || 0;
                  trace.completionTokens += t.completion_tokens || 0;
                  trace.cacheReadTokens += t.cache_read_tokens || 0;
                  trace.cacheCreateTokens += t.cache_create_tokens || 0;
                  trace.elapsed += t.elapsed || 0;
                } else if (t.type === "error") {
                  // Gateway sent an error — display it as content
                  const errMsg = t.message || "Unknown error";
                  fullContent += `\n\n*Error: ${errMsg}*`;
                  contentRef.current = fullContent;
                  scheduleUpdate();
                }
              } catch { /* skip malformed trace */ }
              continue;
            }

            // Track SSE event type (fallback for event: trace format)
            if (trimmed.startsWith("event: ")) {
              currentEventType = trimmed.slice(7);
              continue;
            }

            if (!trimmed.startsWith("data: ")) continue;
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;

            // Handle trace events via event: trace format (backward compat)
            if (currentEventType === "trace") {
              currentEventType = "";
              try {
                const t = JSON.parse(data);
                if (t.type === "tool_call" && t.name) {
                  trace.tools.push(t.name);
                  const argHint = t.args && t.args !== "{}" ? extractArgHint(t.args) : undefined;
                  toolSteps.push({ name: t.name, args: argHint, status: "running" });
                  scheduleUpdate();
                } else if (t.type === "tool_result") {
                  for (let i = toolSteps.length - 1; i >= 0; i--) {
                    if (toolSteps[i].name === t.name && toolSteps[i].status === "running") {
                      const preview = (t.preview || "").slice(0, 60);
                      toolSteps[i].result = preview;
                      toolSteps[i].elapsed = t.elapsed || 0;
                      toolSteps[i].status = preview.startsWith("Error") ? "error" : "done";
                      break;
                    }
                  }
                  scheduleUpdate();
                } else if (t.type === "keepalive" && t.name) {
                  for (let i = toolSteps.length - 1; i >= 0; i--) {
                    if (toolSteps[i].name === t.name && toolSteps[i].status === "running") {
                      toolSteps[i].elapsed = t.elapsed || 0;
                      break;
                    }
                  }
                  scheduleUpdate();
                } else if (t.type === "llm_call") {
                  trace.promptTokens += t.prompt_tokens || 0;
                  trace.completionTokens += t.completion_tokens || 0;
                  trace.cacheReadTokens += t.cache_read_tokens || 0;
                  trace.cacheCreateTokens += t.cache_create_tokens || 0;
                  trace.elapsed += t.elapsed || 0;
                } else if (t.type === "error") {
                  const errMsg = t.message || "Unknown error";
                  fullContent += `\n\n*Error: ${errMsg}*`;
                  contentRef.current = fullContent;
                  scheduleUpdate();
                }
              } catch { /* skip */ }
              continue;
            }
            currentEventType = "";

            try {
              const parsed = JSON.parse(data);
              // Capture the actual model from the response
              if (parsed.model && !actualModel) actualModel = parsed.model;
              const delta = parsed.choices?.[0]?.delta;
              if (delta?.reasoning_content) {
                if (!fetchInReasoning) { fullContent += "*Thinking...*\n\n"; fetchInReasoning = true; }
              } else if (delta?.content) {
                if (fetchInReasoning) { fullContent = fullContent.replace("*Thinking...*\n\n", ""); fetchInReasoning = false; }
                fullContent += delta.content;
              }
              if (delta?.reasoning_content || delta?.content) {
                contentRef.current = fullContent;
                scheduleUpdate();
              }
            } catch { /* skip malformed SSE */ }
          }
        }

        // Final update: model from response, trace stats, tool steps
        const finalUpdates: Partial<ChatMessage> = {};
        if (actualModel) finalUpdates.model = actualModel;
        if (trace.promptTokens || trace.tools.length) finalUpdates.trace = { ...trace };
        if (toolSteps.length) finalUpdates.toolSteps = toolSteps.map((s) => ({ ...s }));
        if (Object.keys(finalUpdates).length) {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, ...finalUpdates } : m));
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
          const finalModel = actualModel || undefined;
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id
              ? { ...m, content: finalContent, ...(finalModel && { model: finalModel }) }
              : m));
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

  return { messages, isStreaming, currentModel, setCurrentModel: updateModel, autoRoute, setAutoRoute: updateAutoRoute, sendMessage, stopStreaming, clearMessages };
}
