import { FC } from "react";
import type { Conversation } from "../hooks/useConversations";

interface Props {
  open: boolean;
  onClose: () => void;
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNewChat: () => void;
}

function groupByDate(convs: Conversation[]) {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfYesterday = startOfToday - 86_400_000;
  const startOf7Days = startOfToday - 7 * 86_400_000;

  const groups: { label: string; items: Conversation[] }[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Previous 7 Days", items: [] },
    { label: "Older", items: [] },
  ];

  for (const c of convs) {
    if (c.updatedAt >= startOfToday) groups[0].items.push(c);
    else if (c.updatedAt >= startOfYesterday) groups[1].items.push(c);
    else if (c.updatedAt >= startOf7Days) groups[2].items.push(c);
    else groups[3].items.push(c);
  }

  return groups.filter((g) => g.items.length > 0);
}

export const ConversationDrawer: FC<Props> = ({
  open,
  onClose,
  conversations,
  activeId,
  onSelect,
  onDelete,
  onNewChat,
}) => {
  const groups = groupByDate(conversations);

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 z-40 bg-black/50 transition-opacity duration-200 ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={onClose}
      />

      {/* Drawer panel */}
      <div
        className={`fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-maude-border bg-maude-surface transition-transform duration-200 ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        {/* Drawer header */}
        <div className="flex items-center justify-between border-b border-maude-border px-4 py-3">
          <h2 className="text-sm font-semibold text-maude-text">Conversations</h2>
          <button
            onClick={() => {
              onNewChat();
              onClose();
            }}
            className="rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-accent hover:text-maude-text"
          >
            + New Chat
          </button>
        </div>

        {/* Conversation list */}
        <div className="no-scrollbar flex-1 overflow-y-auto p-2">
          {groups.length === 0 && (
            <p className="px-2 py-8 text-center text-xs text-maude-muted">No conversations yet</p>
          )}

          {groups.map((group) => (
            <div key={group.label} className="mb-3">
              <p className="mb-1 px-2 text-[10px] font-semibold uppercase tracking-wider text-maude-muted">
                {group.label}
              </p>
              {group.items.map((conv) => (
                <div
                  key={conv.id}
                  className={`group flex items-center rounded-lg px-2 py-2 text-sm transition-colors ${
                    conv.id === activeId
                      ? "bg-maude-bg text-maude-accent"
                      : "text-maude-text hover:bg-maude-bg"
                  }`}
                >
                  <button
                    className="min-w-0 flex-1 truncate text-left"
                    onClick={() => {
                      onSelect(conv.id);
                      onClose();
                    }}
                  >
                    {conv.title}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(conv.id);
                    }}
                    className="ml-1 shrink-0 rounded px-1 text-maude-muted opacity-0 transition-opacity hover:text-red-400 group-hover:opacity-100"
                    aria-label="Delete conversation"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </>
  );
};
