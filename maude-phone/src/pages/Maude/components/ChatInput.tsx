import { FC, useState, useRef, useEffect } from "react";

function getGatewayUrl(): string {
  const loc = window.location;
  return `${loc.protocol}//${loc.host}`;
}

interface Props {
  onSend: (text: string, imageUrls?: string[]) => void;
  isStreaming: boolean;
  onStop: () => void;
}

export const ChatInput: FC<Props> = ({ onSend, isStreaming, onStop }) => {
  const [text, setText] = useState("");
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const attachInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = () => {
    if (pendingImages.length > 0 || text.trim()) {
      onSend(text.trim(), pendingImages.length > 0 ? pendingImages : undefined);
      setText("");
      setPendingImages([]);
      if (inputRef.current) inputRef.current.style.height = "44px";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const handleInput = () => {
    if (inputRef.current) {
      inputRef.current.style.height = "44px";
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + "px";
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      const newUrls: string[] = [];
      for (const file of Array.from(files)) {
        const filename = `camera_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.jpg`;
        const resp = await fetch(`${getGatewayUrl()}/share/${encodeURIComponent(filename)}`, {
          method: "POST",
          body: file,
        });
        if (resp.ok) {
          newUrls.push(`/download/${filename}`);
        }
      }
      if (newUrls.length > 0) {
        setPendingImages((prev) => [...prev, ...newUrls]);
      }
    } catch {
      /* upload failed silently */
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (attachInputRef.current) attachInputRef.current.value = "";
    }
  };

  const removeImage = (index: number) => {
    setPendingImages((prev) => prev.filter((_, i) => i !== index));
  };

  const canSend = pendingImages.length > 0 || text.trim();

  return (
    <div className="border-t border-maude-border bg-maude-surface p-3">
      {/* Thumbnail previews */}
      {pendingImages.length > 0 && (
        <div className="mb-2 flex gap-2 overflow-x-auto">
          {pendingImages.map((img, i) => (
            <div key={img} className="relative shrink-0">
              <img
                src={`${getGatewayUrl()}${img}`}
                alt={`Pending upload ${i + 1}`}
                className="h-20 w-20 rounded-lg object-cover"
              />
              <button
                onClick={() => removeImage(i)}
                className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-red-600 text-xs text-white"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        {/* Camera button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-maude-bg text-lg text-maude-muted hover:text-maude-text disabled:opacity-30"
        >
          {uploading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-maude-accent border-t-transparent" />
          ) : (
            "\uD83D\uDCF7"
          )}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileSelect}
          className="hidden"
        />

        {/* Attach from gallery button */}
        <button
          onClick={() => attachInputRef.current?.click()}
          disabled={uploading}
          className="flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-maude-bg text-lg text-maude-muted hover:text-maude-text disabled:opacity-30"
        >
          {"\uD83D\uDCCE"}
        </button>
        <input
          ref={attachInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileSelect}
          className="hidden"
        />

        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Message MAUDE..."
          rows={1}
          className="min-h-[44px] max-h-[120px] flex-1 resize-none rounded-xl bg-maude-bg px-4 py-3 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
        />

        {isStreaming ? (
          <button onClick={onStop} className="flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-red-600 text-white">
            &#9632;
          </button>
        ) : (
          <button onClick={handleSubmit} disabled={!canSend} className="flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30">
            &#8593;
          </button>
        )}
      </div>
    </div>
  );
};
