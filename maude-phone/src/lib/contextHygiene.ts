/**
 * Phone-side context hygiene (MAUDE cost-reduction steps 1–3).
 *
 * Tool loops and full schema selection run on the gateway. The phone still
 * owns the conversation history it *sends*, so we:
 *   - keep recent turns verbatim
 *   - roll older turns into a single summary system note
 *   - hard-cap message bodies (mobile bandwidth / iOS reliability)
 *   - attach a stable session_id for sticky tool-domain activation
 *
 * Mirrors knobs from maude_core/context_hygiene.py defaults.
 */

export const CTX_KEEP_RECENT_TURNS = 12;
export const CTX_MAX_MSG_CHARS = 4000;
export const CTX_SUMMARY_ENTRY_CHARS = 140;
export const CTX_SUMMARY_MAX_ENTRIES = 12;

export type HistoryMessage = {
  role: string;
  content: string;
};

function truncate(text: string, limit: number): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 20))}\n... (truncated)`;
}

/** Cap a single message body for transport reliability. */
export function compactHistoryContent(
  content: string,
  maxChars: number = CTX_MAX_MSG_CHARS,
): string {
  if (!content || content.length <= maxChars) return content;
  // Keep head + tail so earlier instructions and final answer both survive
  const head = Math.floor(maxChars * 0.45);
  const tail = Math.floor(maxChars * 0.45);
  return `${content.slice(0, head)}\n\n... [older content trimmed for mobile reliability] ...\n\n${content.slice(-tail)}`;
}

function summarizeDropped(
  dropped: HistoryMessage[],
  maxEntries: number = CTX_SUMMARY_MAX_ENTRIES,
): string {
  const lines = [
    `[Earlier conversation summarized — ${dropped.length} messages omitted]`,
  ];
  let entries = 0;
  for (const msg of dropped) {
    const role = String(msg.role || "user").toUpperCase();
    if (role === "SYSTEM") continue;
    let text = (msg.content || "").replace(/\n/g, " ").trim();
    if (!text) continue;
    if (text.length > CTX_SUMMARY_ENTRY_CHARS) {
      text = `${text.slice(0, CTX_SUMMARY_ENTRY_CHARS - 3)}...`;
    }
    lines.push(`- ${role}: ${text}`);
    entries += 1;
    if (entries >= maxEntries) break;
  }
  return lines.join("\n");
}

/**
 * Sliding window + rolling summary for phone → gateway history.
 * Preserves system messages; compresses older user/assistant turns.
 */
export function prepareHistoryForGateway(
  messages: HistoryMessage[],
  opts?: {
    keepRecent?: number;
    maxMsgChars?: number;
  },
): { history: HistoryMessage[]; meta: { removed: number; summarized: boolean } } {
  const keepRecent = Math.max(2, opts?.keepRecent ?? CTX_KEEP_RECENT_TURNS);
  const maxChars = opts?.maxMsgChars ?? CTX_MAX_MSG_CHARS;

  const systemMsgs = messages.filter(
    (m) => m.role === "system" && !String(m.content || "").startsWith("[Earlier conversation summarized"),
  );
  const priorSummaries = messages.filter(
    (m) => m.role === "system" && String(m.content || "").startsWith("[Earlier conversation summarized"),
  );
  const nonSystem = messages.filter((m) => m.role !== "system");

  let removed = 0;
  let dropped: HistoryMessage[] = [];
  let recent = nonSystem;
  if (nonSystem.length > keepRecent) {
    dropped = nonSystem.slice(0, -keepRecent);
    recent = nonSystem.slice(-keepRecent);
    removed = dropped.length;
  }

  const prepared: HistoryMessage[] = systemMsgs.map((m) => ({
    role: m.role,
    content: compactHistoryContent(m.content, maxChars),
  }));

  if (dropped.length > 0) {
    prepared.push({
      role: "system",
      content: summarizeDropped(dropped),
    });
  } else if (priorSummaries.length > 0) {
    prepared.push(priorSummaries[priorSummaries.length - 1]);
  }

  for (const m of recent) {
    prepared.push({
      role: m.role,
      content: compactHistoryContent(m.content, maxChars),
    });
  }

  return {
    history: prepared,
    meta: { removed, summarized: dropped.length > 0 },
  };
}

/** Stable session key for sticky tool-domain activation on the gateway. */
export function phoneSessionId(conversationId: string | null | undefined): string {
  if (conversationId && conversationId.trim()) {
    return `phone:${conversationId.trim()}`;
  }
  // Per-tab fallback so sticky domains don't leak across unrelated chats
  try {
    const key = "maude-phone-session-id";
    let sid = localStorage.getItem(key);
    if (!sid) {
      sid = `phone-tab:${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem(key, sid);
    }
    return sid;
  } catch {
    return "phone:default";
  }
}
