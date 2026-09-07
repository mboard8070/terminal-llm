import { FC, useState } from "react";
import { SELECTABLE_MODELS, normalizeModelId } from "../../../lib/models";

interface Props {
  currentModel: string;
  onSelect: (model: string) => void;
  autoRoute: boolean;
  onToggleAutoRoute: (val: boolean) => void;
}

export const ModelSelector: FC<Props> = ({ currentModel, onSelect, autoRoute, onToggleAutoRoute }) => {
  const [open, setOpen] = useState(false);
  const normalizedCurrentModel = normalizeModelId(currentModel);
  const current = SELECTABLE_MODELS.find((m) => m.id === normalizedCurrentModel) || SELECTABLE_MODELS[0];

  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)} className="flex items-center gap-1.5 rounded-lg bg-maude-bg px-3 py-1.5 text-xs text-maude-muted transition-colors hover:text-maude-text">
        <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
        {current.label}
        {autoRoute && <span className="text-[10px] text-maude-accent">AUTO</span>}
      </button>
      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 w-56 rounded-xl border border-maude-border bg-maude-surface p-2 shadow-xl">
          {SELECTABLE_MODELS.map((m) => (
            <button key={m.id} onClick={() => { onSelect(m.id); setOpen(false); }}
              className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${m.id === normalizedCurrentModel ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`}>
              <span>{m.label}</span>
              <span className="text-xs text-maude-muted">{m.desc}</span>
            </button>
          ))}
          <div className="mt-2 border-t border-maude-border pt-2">
            <button onClick={() => onToggleAutoRoute(!autoRoute)}
              className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm text-maude-text hover:bg-maude-bg">
              <span>Auto-route code</span>
              <span className={`text-xs ${autoRoute ? "text-green-400" : "text-maude-muted"}`}>{autoRoute ? "ON" : "OFF"}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
