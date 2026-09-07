export const SELECTABLE_MODELS = [
  { id: "nemotron-super", label: "Nemotron Super", desc: "OpenRouter 120B", provider: "openrouter" },
  { id: "mistral-large-latest", label: "Mistral Large", desc: "General", provider: "mistral" },
  { id: "muse-spark", label: "Muse Spark", desc: "Meta 1.3", provider: "meta" },
  { id: "grok", label: "Grok", desc: "X Premium CLI", provider: "grok-cli" },
  { id: "codex", label: "Codex", desc: "CLI", provider: "codex-cli" },
] as const;

export const SELECTABLE_MODEL_IDS = new Set<string>(SELECTABLE_MODELS.map((m) => m.id));

export const MODEL_ALIASES: Record<string, string> = {
  "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
  "codex-cli": "codex",
  grok4: "grok",
  "grok-4.5": "grok",
  "grok-4.6": "grok",
  mistral: "mistral-large-latest",
  muse: "muse-spark",
  spark: "muse-spark",
  "muse-spark-1.3": "muse-spark",
};

const REMOVED_MODELS = new Set([
  "claude-opus-4-20250514",
  "claude-sonnet-4-20250514",
  "claude",
  "sonnet",
  "codestral-latest",
  "codestral",
  "devstral-2512",
  "devstral",
  "devstral-small-latest",
  "devstral-small",
  "devstral-medium-latest",
  "devstral-medium",
  "nemotron",
  "local",
  "nemotron-a3b",
  "a3b",
  "nemotron-nano",
  "nvidia/nemotron-3-nano-30b-a3b",
  "gemma-4-31b",
  "gemma4",
  "gemma",
  "llava",
  "vision",
]);

export function normalizeModelId(model: string | null | undefined): string {
  const value = (model || "").trim();
  if (!value || REMOVED_MODELS.has(value)) return "nemotron-super";
  const aliased = MODEL_ALIASES[value] || value;
  if (REMOVED_MODELS.has(aliased) || !SELECTABLE_MODEL_IDS.has(aliased)) return "nemotron-super";
  return aliased;
}
