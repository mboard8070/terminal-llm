import { FC, useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { encodeMessage, decodeMessage } from "../../protocol/encoder";
import type { SocketStatus } from "../../protocol/types";

// MAUDE system prompt for voice
const MAUDE_TEXT_PROMPT =
  "You are MAUDE, a capable AI assistant with a warm Scottish accent. " +
  "You are direct, competent, and quietly confident. " +
  "Keep responses concise and natural for voice conversation. " +
  "You run locally on Matt’s DGX Spark workstation.";

const DEFAULT_VOICE = "NATF2.pt";

function getGatewayUrl(): string {
  return `${window.location.protocol}//${window.location.host}`;
}

function getVoiceUrl(imageContext?: string): string {
  // Route through the gateway WSS proxy (/api/chat) so iOS can reuse
  // the already-trusted TLS session instead of needing a second
  // self-signed cert trust for port 8998.
  const host = window.location.host; // includes port (e.g. 100.x.x.x:30000)
  const base = `wss://${host}`;

  let prompt = MAUDE_TEXT_PROMPT;
  if (imageContext) {
    prompt += "\n\n--- Image Context ---\n" + imageContext;
  }

  const params = new URLSearchParams({
    text_prompt: prompt,
  });

  return `${base}/api/chat?${params}`;
}

// ─────────────────────────────────────────────────────────────────
// AudioWorklet ring-buffer playback. The hardware audio clock pulls
// samples at a constant rate from a shared ring buffer. The main
// thread pushes PCM into the buffer. This avoids all the timing
// issues of AudioBufferSourceNode scheduling on Android WebView.
// ─────────────────────────────────────────────────────────────────

const RING_WORKLET_CODE = `
class RingPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufSize = Math.round(sampleRate * 4);
    this.buf = new Float32Array(this.bufSize);
    this.writePos = 0;
    this.readPos = 0;
    this.started = false;
    this.preBuffer = Math.round(sampleRate * 0.5); // 500ms initial buffer
    this.underruns = 0;
    this.lastSample = 0;
    this.reportCounter = 0;

    this.port.onmessage = (e) => {
      if (e.data.type === 'audio') {
        const pcm = e.data.pcm;
        for (let i = 0; i < pcm.length; i++) {
          this.buf[(this.writePos + i) % this.bufSize] = pcm[i];
        }
        this.writePos = (this.writePos + pcm.length) % this.bufSize;
      } else if (e.data.type === 'reset') {
        this.writePos = 0;
        this.readPos = 0;
        this.buf.fill(0);
        this.started = false;
        this.underruns = 0;
        this.lastSample = 0;
      }
    };
  }

  available() {
    let a = this.writePos - this.readPos;
    if (a < 0) a += this.bufSize;
    return a;
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    const avail = this.available();

    // Wait for initial buffer
    if (!this.started) {
      out.fill(0);
      if (avail >= this.preBuffer) {
        this.started = true;
      }
      return true;
    }

    // Play available samples, hold last for any gap
    const toRead = Math.min(out.length, avail);
    for (let i = 0; i < toRead; i++) {
      this.lastSample = this.buf[this.readPos];
      out[i] = this.lastSample;
      this.readPos = (this.readPos + 1) % this.bufSize;
    }
    if (toRead < out.length) {
      this.underruns++;
      for (let i = toRead; i < out.length; i++) out[i] = this.lastSample;
    }

    // Report every ~500ms
    this.reportCounter++;
    if (this.reportCounter >= 187) {
      this.reportCounter = 0;
      this.port.postMessage({
        type: 'state', avail: avail, underruns: this.underruns
      });
    }
    return true;
  }
}
registerProcessor('ring-player', RingPlayerProcessor);
`;

interface PlaybackNode {
  feedAudio: (frame: Float32Array) => void;
  reset: () => void;
  connect: (dest: AudioNode) => void;
  disconnect: () => void;
}

async function createWorkletPlaybackNode(
  ctx: AudioContext,
  onStateChange?: (state: string, detail: Record<string, number>) => void,
): Promise<PlaybackNode> {
  const blob = new Blob([RING_WORKLET_CODE], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  const workletNode = new AudioWorkletNode(ctx, "ring-player", {
    outputChannelCount: [1],
  });

  workletNode.port.onmessage = (e) => {
    if (e.data?.type === "state" && onStateChange) {
      onStateChange(e.data.state, e.data);
    }
  };

  const gainNode = ctx.createGain();
  gainNode.gain.value = 6.0;
  workletNode.connect(gainNode);

  return {
    feedAudio(frame: Float32Array) {
      workletNode.port.postMessage({ type: "audio", pcm: frame }, [frame.buffer]);
    },
    reset() {
      workletNode.port.postMessage({ type: "reset" });
    },
    connect(dest: AudioNode) {
      gainNode.connect(dest);
    },
    disconnect() {
      try { gainNode.disconnect(); } catch {}
      try { workletNode.disconnect(); } catch {}
    },
  };
}

// Waveform visualizer
const Waveform: FC<{ analyser: AnalyserNode | null; active: boolean; color: string }> = ({ analyser, active, color }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    if (!analyser || !active || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const bufLen = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufLen);

    const draw = () => {
      animRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.beginPath();
      const sliceWidth = canvas.width / bufLen;
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [analyser, active, color]);

  return <canvas ref={canvasRef} width={300} height={60} className="w-full rounded-lg" />;
};

export const Voice: FC = () => {
  const navigate = useNavigate();
  const [status, setStatus] = useState<SocketStatus>("disconnected");
  const [transcript, setTranscript] = useState("");
  const [toolActive, setToolActive] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState("");
  const [audioDebug, setAudioDebug] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageAnalysis, setImageAnalysis] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploading, setUploading] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);
  const imageContextRef = useRef<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackRef = useRef<PlaybackNode | null>(null);
  const recorderRef = useRef<any>(null);
  const serverAnalyserRef = useRef<AnalyserNode | null>(null);
  const micAnalyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const durationRef = useRef(0);
  const timerRef = useRef<number>(0);
  const audioFramesRef = useRef(0);
  const decodedFramesRef = useRef(0);
  const workletUnderrunsRef = useRef(0);
  const workletBufRef = useRef(0);
  const debugIntervalRef = useRef<number>(0);

  const connect = useCallback(async () => {
    setError("");
    setTranscript("");
    setDuration(0);
    durationRef.current = 0;

    try {
      // 1. Set up AudioContext — force sample rate and unlock on mobile
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({ sampleRate: 48000 });
      }
      const ctx = audioContextRef.current;
      await ctx.resume();

      // Android WebView audio unlock: play a tiny silent buffer
      const silentBuf = ctx.createBuffer(1, 1, ctx.sampleRate);
      const silentSrc = ctx.createBufferSource();
      silentSrc.buffer = silentBuf;
      silentSrc.connect(ctx.destination);
      silentSrc.start();

      audioFramesRef.current = 0;
      decodedFramesRef.current = 0;
      setAudioDebug(`ctx: ${ctx.state} ${ctx.sampleRate}Hz`);

      // 2. Create AudioWorklet ring-buffer playback node
      if (!playbackRef.current) {
        playbackRef.current = await createWorkletPlaybackNode(ctx, (_state, detail) => {
          if (detail.underruns != null) workletUnderrunsRef.current = detail.underruns;
          if (detail.avail != null) workletBufRef.current = detail.avail;
        });
        playbackRef.current.connect(ctx.destination);
      }
      playbackRef.current.reset();
      workletUnderrunsRef.current = 0;

      // Server audio analyser
      const sAnalyser = ctx.createAnalyser();
      playbackRef.current.connect(sAnalyser);
      serverAnalyserRef.current = sAnalyser;

      // 3. Request microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      streamRef.current = stream;

      // Mic analyser
      const mAnalyser = ctx.createAnalyser();
      const micSource = ctx.createMediaStreamSource(stream);
      micSource.connect(mAnalyser);
      micAnalyserRef.current = mAnalyser;

      // 5. Connect WebSocket
      const url = getVoiceUrl(imageContextRef.current ?? undefined);
      console.log("Connecting to voice server:", url);
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      socketRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        console.log("voice server WS open, waiting for handshake");
      };

      ws.onmessage = (e: MessageEvent) => {
        try {
          const data = new Uint8Array(e.data);

          const kind = data[0];

          if (kind === 0x00) {
            // Handshake
            console.log("voice server handshake received");
            setStatus("connected");
            startRecording(ws, stream, ctx);
            timerRef.current = window.setInterval(() => {
              durationRef.current += 1;
              setDuration(durationRef.current);
              const ctxState = audioContextRef.current?.state ?? "?";
              const bufMs = Math.round(workletBufRef.current / 48);
              setAudioDebug(`dec:${decodedFramesRef.current} buf:${bufMs}ms ur:${workletUnderrunsRef.current}`);
            }, 1000);
          } else if (kind === 0x02) {
            // Text
            const text = new TextDecoder().decode(data.slice(1));
            if (text.includes("[Searching...]")) {
              setToolActive(true);
            } else if (text.includes("[Tool result:]") || text.includes("[Error:")) {
              setToolActive(false);
            }
            setTranscript((prev) => prev + text);
          } else if (kind === 0x03) {
            // Raw PCM (float32 @ 24kHz)
            decodedFramesRef.current++;
            const rawBytes = data.slice(1);
            const pcm24 = new Float32Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength / 4);
            // Resample 24kHz → 48kHz (linear interpolation)
            const pcm48 = new Float32Array(pcm24.length * 2);
            for (let i = 0; i < pcm48.length; i++) {
              const srcIdx = i * 0.5;
              const lo = srcIdx | 0;
              const hi = Math.min(lo + 1, pcm24.length - 1);
              const frac = srcIdx - lo;
              pcm48[i] = pcm24[lo] + (pcm24[hi] - pcm24[lo]) * frac;
            }
            playbackRef.current?.feedAudio(pcm48);
          }
        } catch (err) {
          console.error("Message decode error:", err);
        }
      };

      ws.onclose = (e) => {
        console.log("voice server WS closed:", e.code, e.reason);
        setStatus("disconnected");
        stopRecording();
        clearInterval(timerRef.current);
      };

      ws.onerror = (e) => {
        console.error("voice server WS error:", e);
        setError("WebSocket connection failed. Is voice server running?");
        setStatus("disconnected");
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Connection failed";
      console.error("Voice connect error:", msg);
      setError(msg);
      setStatus("disconnected");
    }
  }, []);

  const startRecording = useCallback(
    async (ws: WebSocket, stream: MediaStream, ctx: AudioContext) => {
      try {
        const Recorder = (await import("opus-recorder")).default;
        const encoderPath = (
          await import("opus-recorder/dist/encoderWorker.min.js?url")
        ).default;

        const sourceNode = ctx.createMediaStreamSource(stream);

        const recorder = new Recorder({
          encoderPath,
          bufferLength: Math.round((960 * ctx.sampleRate) / 24000),
          encoderFrameSize: 20,
          encoderSampleRate: 24000,
          maxFramesPerPage: 2,
          numberOfChannels: 1,
          recordingGain: 1,
          resampleQuality: 3,
          encoderComplexity: 3,
          encoderApplication: 2049,
          streamPages: true,
          sourceNode,
        });

        recorder.ondataavailable = (data: Uint8Array) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(encodeMessage({ type: "audio", data }));
          }
        };

        recorder.onstart = () => {
          console.log("Opus recorder started");
        };

        recorder.start();
        recorderRef.current = recorder;
      } catch (err) {
        console.error("Recorder start error:", err);
        setError("Failed to start microphone recording");
      }
    },
    [],
  );

  const stopRecording = useCallback(() => {
    if (recorderRef.current) {
      try { recorderRef.current.stop(); } catch {}
      recorderRef.current = null;
    }
    // Stop mic tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const disconnect = useCallback(() => {
    stopRecording();
    clearInterval(timerRef.current);
    clearInterval(debugIntervalRef.current);
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setStatus("disconnected");
  }, [stopRecording]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const filename = `voice_camera_${Date.now()}.jpg`;
    const gatewayUrl = getGatewayUrl();

    // Show thumbnail immediately
    const objectUrl = URL.createObjectURL(file);
    setImageUrl(objectUrl);
    setImageAnalysis(null);
    setUploading(true);

    try {
      // 1. Upload to shared/
      const uploadResp = await fetch(`${gatewayUrl}/share/${filename}`, {
        method: "POST",
        body: file,
      });
      if (!uploadResp.ok) throw new Error("Upload failed");
      setUploading(false);

      // 2. Analyze with LLaVA
      setAnalyzing(true);
      const analyzeResp = await fetch(`${gatewayUrl}/api/analyze-image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename,
          question: "Describe this image in detail. What do you see?",
        }),
      });
      if (!analyzeResp.ok) throw new Error("Analysis failed");
      const result = await analyzeResp.json();
      const analysis = result.analysis || "No analysis returned.";

      setImageAnalysis(analysis);
      setAnalyzing(false);
      imageContextRef.current = `The user shared an image (${filename}). Analysis: ${analysis}`;

      // 3. Reconnect with image context if currently connected
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        disconnect();
        await new Promise((resolve) => setTimeout(resolve, 300));
        connect();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Image processing failed";
      setError(msg);
      setUploading(false);
      setAnalyzing(false);
    }
  }, [connect, disconnect]);

  const clearImageContext = useCallback(async () => {
    imageContextRef.current = null;
    setImageUrl(null);
    setImageAnalysis(null);
    // Reconnect without context if connected
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      disconnect();
      await new Promise((resolve) => setTimeout(resolve, 300));
      connect();
    }
  }, [connect, disconnect]);

  useEffect(() => {
    return () => { disconnect(); };
  }, []);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const isConnected = status === "connected";
  const isConnecting = status === "connecting";

  return (
    <div className="flex h-full flex-col bg-maude-bg">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2">
        <div className="flex items-center gap-2">
          <h1 className="fire-gradient text-lg font-bold">MAUDE</h1>
          <span className="rounded-full bg-maude-bg px-2 py-0.5 text-[10px] uppercase tracking-wider text-maude-accent">
            Voice
          </span>
        </div>
        <button
          onClick={() => navigate("/maude")}
          className="rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text"
        >
          Text Mode
        </button>
      </div>

      {/* Main content */}
      <div className="flex flex-1 flex-col items-center justify-center gap-6 overflow-y-auto px-6 pb-4">
        {/* Status indicator */}
        <div className="flex flex-col items-center gap-2">
          <div
            className={`h-32 w-32 rounded-full border-4 ${
              isConnected
                ? "animate-pulse border-maude-accent shadow-[0_0_30px_rgba(255,69,0,0.3)]"
                : isConnecting
                ? "animate-spin border-maude-muted"
                : "border-maude-border"
            } flex items-center justify-center`}
          >
            <span className="text-4xl">
              {isConnected ? "\uD83C\uDF99\uFE0F" : isConnecting ? "\u23F3" : "\uD83C\uDF99\uFE0F"}
            </span>
          </div>

          <span className="text-sm text-maude-muted">
            {isConnected
              ? `Connected \u2022 ${formatTime(duration)}`
              : isConnecting
              ? "Connecting to MAUDE Voice..."
              : "Tap to start voice chat"}
          </span>
        </div>

        {/* Waveforms */}
        {isConnected && (
          <div className="w-full max-w-xs space-y-3">
            <div>
              <span className="mb-1 block text-[10px] uppercase tracking-wider text-maude-muted">MAUDE</span>
              <div className="rounded-lg bg-maude-surface p-2">
                <Waveform analyser={serverAnalyserRef.current} active={isConnected} color="#ff4500" />
              </div>
            </div>
            <div>
              <span className="mb-1 block text-[10px] uppercase tracking-wider text-maude-muted">You</span>
              <div className="rounded-lg bg-maude-surface p-2">
                <Waveform analyser={micAnalyserRef.current} active={isConnected} color="#888" />
              </div>
            </div>
          </div>
        )}

        {/* Camera / Gallery buttons */}
        {isConnected && (
          <div className="flex gap-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={analyzing || uploading}
              className="flex items-center gap-1.5 rounded-xl bg-maude-surface px-4 py-2 text-sm text-maude-text transition-all hover:bg-maude-border disabled:opacity-40"
            >
              <span>{"\uD83D\uDCF7"}</span> Camera
            </button>
            <button
              onClick={() => galleryInputRef.current?.click()}
              disabled={analyzing || uploading}
              className="flex items-center gap-1.5 rounded-xl bg-maude-surface px-4 py-2 text-sm text-maude-text transition-all hover:bg-maude-border disabled:opacity-40"
            >
              <span>{"\uD83D\uDDBC\uFE0F"}</span> Gallery
            </button>
          </div>
        )}

        {/* Hidden file inputs */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileSelect}
          className="hidden"
        />
        <input
          ref={galleryInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        {/* Image analysis panel */}
        {imageUrl && (
          <div className="w-full max-w-xs rounded-xl bg-maude-surface p-3">
            <span className="mb-2 block text-[10px] uppercase tracking-wider text-maude-muted">Image Context</span>
            <img src={imageUrl} alt="Captured" className="mb-2 h-24 w-full rounded-lg object-cover" />
            {uploading && (
              <p className="text-xs text-maude-muted">Uploading...</p>
            )}
            {analyzing && (
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-maude-accent border-t-transparent" />
                <span className="text-xs text-maude-muted">Analyzing with LLaVA...</span>
              </div>
            )}
            {imageAnalysis && (
              <p className="text-xs leading-relaxed text-maude-text">{imageAnalysis}</p>
            )}
            {imageAnalysis && (
              <button
                onClick={clearImageContext}
                className="mt-2 text-[10px] text-maude-muted underline hover:text-maude-text"
              >
                Clear image context
              </button>
            )}
          </div>
        )}

        {/* Tool-active indicator */}
        {toolActive && (
          <div className="flex items-center gap-2 rounded-xl bg-maude-accent/10 px-4 py-2">
            <div className="h-3 w-3 animate-spin rounded-full border-2 border-maude-accent border-t-transparent" />
            <span className="text-xs font-medium text-maude-accent">Searching...</span>
          </div>
        )}

        {/* Transcript */}
        {transcript && (
          <div className="w-full max-w-xs rounded-xl bg-maude-surface p-3">
            <span className="mb-1 block text-[10px] uppercase tracking-wider text-maude-muted">Transcript</span>
            <div className="max-h-48 overflow-y-auto text-sm text-maude-text">
              {transcript.split("\n").map((line, i) => {
                if (line.includes("[Searching...]")) {
                  return <p key={i} className="my-1 text-xs italic text-maude-accent">{line}</p>;
                }
                if (line.includes("[Tool result:]")) {
                  return <p key={i} className="mt-2 mb-1 text-[10px] font-bold uppercase tracking-wider text-maude-accent">{line}</p>;
                }
                if (line.includes("[Error:")) {
                  return <p key={i} className="my-1 text-xs text-red-400">{line}</p>;
                }
                return <span key={i}>{line}{i < transcript.split("\n").length - 1 ? "\n" : ""}</span>;
              })}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="w-full max-w-xs rounded-xl bg-red-900/30 p-3">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Connect/Disconnect button */}
        <button
          onClick={isConnected || isConnecting ? disconnect : connect}
          className={`min-w-[200px] rounded-2xl px-8 py-4 text-base font-semibold text-white transition-all ${
            isConnected
              ? "bg-red-600 hover:bg-red-700"
              : isConnecting
              ? "bg-maude-muted"
              : "fire-bg hover:opacity-90"
          }`}
          disabled={isConnecting}
        >
          {isConnected ? "End Call" : isConnecting ? "Connecting..." : "Start Voice Chat"}
        </button>

        {/* Voice info */}
        <div className="text-center text-[10px] text-maude-muted">
          Voice: {(localStorage.getItem("maude-default-voice") || DEFAULT_VOICE).replace(".pt", "")}
          {" \u2022 "}MAUDE Voice
        </div>

        {/* Audio debug info */}
        {audioDebug && (
          <div className="text-center font-mono text-[10px] text-maude-muted opacity-60">
            {audioDebug}
          </div>
        )}
      </div>
    </div>
  );
};
