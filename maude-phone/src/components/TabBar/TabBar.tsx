import { FC } from "react";
import { useLocation, useNavigate } from "react-router-dom";

interface TabItem {
  path: string;
  label: string;
  icon: string;
  match: string[];
}

const TABS: TabItem[] = [
  { path: "/", label: "Home", icon: "\u2B21", match: ["/"] },
  { path: "/maude", label: "Chat", icon: "\u25C6", match: ["/maude"] },
  { path: "/maude/voice", label: "Voice", icon: "\uD83C\uDF99\uFE0F", match: ["/maude/voice"] },
  { path: "/terminal", label: "Term", icon: ">_", match: ["/terminal"] },
  { path: "/files", label: "Files", icon: "\u25A4", match: ["/files"] },
  { path: "/settings", label: "Set", icon: "\u2699", match: ["/settings"] },
];

export const TabBar: FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <nav className="safe-bottom flex shrink-0 items-center justify-around border-t border-maude-border bg-maude-surface px-1 pb-1 pt-1">
      {TABS.map((tab) => {
        const active = tab.match.includes(location.pathname);
        return (
          <button
            key={tab.path}
            onClick={() => navigate(tab.path)}
            className={`flex min-h-[44px] min-w-[44px] flex-col items-center justify-center rounded-lg px-2 py-1 text-xs transition-colors ${
              active
                ? "text-maude-accent"
                : "text-maude-muted hover:text-maude-text"
            }`}
          >
            <span className="text-base leading-none">{tab.icon}</span>
            <span className="mt-0.5">{tab.label}</span>
          </button>
        );
      })}
    </nav>
  );
};
