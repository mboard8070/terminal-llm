import { FC, useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { encodeMessage, decodeMessage } from "../../protocol/encoder";
import type { SocketStatus } from "../../protocol/types";

// MAUDE system prompt for PersonaPlex voice
const MAUDE_TEXT_PROMPT =
  "You are MAUDE, a capable AI assistant with a warm Scottish accent. " +
  "You are direct, competent, and quietly confident — like MAUDE’s design. "
  "Keep responses concise and natural for voice conversation. " +
  "You run locally on Matt's DGX Spark workstation.";

const DEFAULT_VOICE = "NATF2.pt";

// PersonaPlex native WebSocket port — connect directly, bypassing the gateway
// relay which adds jitter from raw TCP proxying + TLS record overhead.
const PERSONAPLEX_PORT = 8998;

function getGatewayUrl(): string {
  return `${window.location.protocol}//${window.location.host}`;
}

function getPersonaPlexUrl(imageContext?: string): string {
  const host = window.location.hostname;
  const base = `wss://${host}:${PERSONAPLEX_PORT}`;

  let prompt = MAUDE_TEXT_PROMPT;
  if (imageContext) {
    prompt += "\n\n--- Image Context ---\n" + imageContext;
  }

  const voice = localStorage.getItem("maude-default-voice") || DEFAULT_VOICE;
  const params = new URLSearchParams({
    text_temperature: "0.7",
    text_topk: "25",
    audio_temperature: "0.8",
    audio_topk: "250",
    pad_mult: "0",
    text_seed: String(Math.round(Math.random() * 1000000)),
    audio_seed: String(Math.round(Math.random() * 1000000)),
    repetition_penalty_context: "64",
    repetition_penalty: "1.0",
    text_prompt: prompt,
    voice_prompt: voice,
  });

  return `${base}/api/chat?${params}`;
}

// ─────────────────────────────────────────────────────────────────
// Consolidated-chunk scheduled playback for Android WebView.
// Merges small decoded PCM frames into larger ~100ms chunks before
// scheduling them as AudioBufferSourceNodes. This eliminates clicks
// from too many tiny buffers and smooths out network/decode jitter.
// ─────────────────────────────────────────────────────────────────

interface PlaybackNode {
  feedAudio: (frame: Float32Array) => void;
  reset: () => void;
  connect: (dest: AudioNode) => void;
  disconnect: () => void;
}

function createScheduledPlaybackNode(ctx: AudioContext): PlaybackNode {
  let nextPlayTime = 0;
  let gainNode: GainNode | null = null;

  // Consolidation: accumulate decoded frames into ~100ms chunks
  const CHUNK_MS = 100;
  const CHUNK_SAMPLES = Math.round(ctx.sampleRate * CHUNK_MS / 1000);
  let accumBuf = new Float32Array(CHUNK_SAMPLES);
  let accumPos = 0;

  // Flush timer: if no new audio arrives within 60ms, flush whatever
  // we have. This prevents audio getting stuck in the accumulator
  // when PersonaPlex sends in bursts with gaps between them.
  let flushTimer: ReturnType<typeof setTimeout> | null = null;
  const FLUSH_DELAY_MS = 60;

  // Initial buffering: wait for 800ms before starting playback
  const INITIAL_BUFFER_SEC = 0.8;
  let pendingSamples = 0;
  let pendingChunks: Float32Array[] = [];
  let started = false;

  // Crossfade to eliminate clicks at chunk boundaries
  const XFADE = 64;
  let prevTail = new Float32Array(XFADE);

  function scheduleChunk(pcm: Float32Array) {
    if (!gainNode || pcm.length === 0) return;

    // Crossfade: blend previous tail with this chunk's head
    const xlen = Math.min(XFADE, pcm.length);
    for (let i = 0; i < xlen; i++) {
      const t = i / XFADE;
      pcm[i] = pcm[i] * t + prevTail[i] * (1 - t);
    }

    // Soft-clip
    for (let i = 0; i < pcm.length; i++) {
      const s = pcm[i];
      if (s > 1.0) pcm[i] = 1.0;
      else if (s < -1.0) pcm[i] = -1.0;
    }

    // Save tail for next crossfade
    const tailStart = Math.max(0, pcm.length - XFADE);
    const tail = pcm.slice(tailStart);
    prevTail = new Float32Array(XFADE);
    prevTail.set(tail, 0);

    const buf = ctx.createBuffer(1, pcm.length, ctx.sampleRate);
    buf.getChannelData(0).set(pcm);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(gainNode);
    src.start(nextPlayTime);
    nextPlayTime += pcm.length / ctx.sampleRate;
  }

  function flushAccum() {
    if (flushTimer !== null) { clearTimeout(flushTimer); flushTimer = null; }
    if (accumPos === 0) return;
    const chunk = accumBuf.slice(0, accumPos);
    accumPos = 0;
    if (!started) {
      pendingChunks.push(chunk);
      pendingSamples += chunk.length;
      if (pendingSamples >= ctx.sampleRate * INITIAL_BUFFER_SEC) {
        started = true;
        nextPlayTime = ctx.currentTime + 0.1;
        for (const c of pendingChunks) scheduleChunk(c);
        pendingChunks = [];
        pendingSamples = 0;
      }
    } else {
      // If we've fallen behind (underrun), jump ahead
      if (nextPlayTime < ctx.currentTime) {
        nextPlayTime = ctx.currentTime + 0.1;
      }
      // If too far ahead (>3s), skip forward to limit latency
      if (nextPlayTime > ctx.currentTime + 3.0) {
        nextPlayTime = ctx.currentTime + 0.5;
      }
      scheduleChunk(chunk);
    }
  }

  function armFlushTimer() {
    if (flushTimer !== null) clearTimeout(flushTimer);
    flushTimer = setTimeout(flushAccum, FLUSH_DELAY_MS);
  }

  return {
    feedAudio(frame: Float32Array) {
      let offset = 0;
      while (offset < frame.length) {
        const space = CHUNK_SAMPLES - accumPos;
        const take = Math.min(space, frame.length - offset);
        accumBuf.set(frame.subarray(offset, offset + take), accumPos);
        accumPos += take;
        offset += take;
        if (accumPos >= CHUNK_SAMPLES) {
          flushAccum();
        }
      }
      // If there's leftover in the accumulator, arm the flush timer
      // so it doesn't sit there waiting for more data that may not come soon
      if (accumPos > 0) {
        armFlushTimer();
      }
    },
    reset() {
      if (flushTimer !== null) { clearTimeout(flushTimer); flushTimer = null; }
      nextPlayTime = 0;
      started = false;
      accumPos = 0;
      pendingChunks = [];
      pendingSamples = 0;
      prevTail = new Float32Array(XFADE);
    },
    connect(dest: AudioNode) {
      if (!gainNode) {
        gainNode = ctx.createGain();
        gainNode.gain.value = 1.5;
      }
      gainNode.connect(dest);
    },
    disconnect() {
      if (flushTimer !== null) { clearTimeout(flushTimer); flushTimer = null; }
      if (gainNode) {
        try { gainNode.disconnect(); } catch {}
      }
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
  const decoderWorkerRef = useRef<Worker | null>(null);
  const serverAnalyserRef = useRef<AnalyserNode | null>(null);
  const micAnalyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const durationRef = useRef(0);
  const timerRef = useRef<number>(0);
  const audioFramesRef = useRef(0);
  const decodedFramesRef = useRef(0);
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

      // 2. Create scheduled playback node (AudioBufferSourceNode scheduling)
      if (!playbackRef.current) {
        playbackRef.current = createScheduledPlaybackNode(ctx);
        playbackRef.current.connect(ctx.destination);
      }
      playbackRef.current.reset();

      // Server audio analyser
      const sAnalyser = ctx.createAnalyser();
      playbackRef.current.connect(sAnalyser);
      serverAnalyserRef.current = sAnalyser;

      // 3. Set up Opus decoder worker
      if (decoderWorkerRef.current) {
        decoderWorkerRef.current.terminate();
      }
      const decoderUrl = new URL("/assets/decoderWorker.min.js", window.location.origin).href;
      const worker = new Worker(decoderUrl);
      decoderWorkerRef.current = worker;

      worker.onerror = (e) => {
        console.error("Decoder worker error:", e);
        setError("Audio decoder failed to load");
        setAudioDebug((prev) => prev + " | WORKER ERR");
      };

      worker.postMessage({
        command: "init",
        bufferLength: Math.round(960 * ctx.sampleRate / 24000),
        decoderSampleRate: 24000,
        outputBufferSampleRate: ctx.sampleRate,
        resampleQuality: 0,
      });

      // Wait for decoder to initialize
      await new Promise<void>((resolve) => setTimeout(resolve, 800));

      worker.onmessage = (e: MessageEvent) => {
        if (e.data?.[0]) {
          decodedFramesRef.current++;
          const decoded: Float32Array = e.data[0];
          playbackRef.current?.feedAudio(decoded);
        }
      };

      // 4. Request microphone
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
      const url = getPersonaPlexUrl(imageContextRef.current ?? undefined);
      console.log("Connecting to PersonaPlex:", url);
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      socketRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        console.log("PersonaPlex WS open, waiting for handshake");
      };

      ws.onmessage = (e: MessageEvent) => {
        try {
          const data = new Uint8Array(e.data);
          const msg = decodeMessage(data);

          if (msg.type === "handshake") {
            console.log("PersonaPlex handshake received");
            setStatus("connected");
            startRecording(ws, stream, ctx);

            timerRef.current = window.setInterval(() => {
              durationRef.current += 1;
              setDuration(durationRef.current);
              const ctxState = audioContextRef.current?.state ?? "?";
              setAudioDebug(`ctx:${ctxState} rx:${audioFramesRef.current} dec:${decodedFramesRef.current}`);
            }, 1000);
          } else if (msg.type === "audio") {
            audioFramesRef.current++;
            decoderWorkerRef.current?.postMessage(
              { command: "decode", pages: msg.data },
              [msg.data.buffer],
            );
          } else if (msg.type === "text") {
            setTranscript((prev) => prev + msg.data);
          } else if (msg.type === "error") {
            console.error("PersonaPlex error:", msg.data);
            setError(msg.data);
          }
        } catch (err) {
          console.error("Message decode error:", err);
        }
      };

      ws.onclose = (e) => {
        console.log("PersonaPlex WS closed:", e.code, e.reason);
        setStatus("disconnected");
        stopRecording();
        clearInterval(timerRef.current);
      };

      ws.onerror = (e) => {
        console.error("PersonaPlex WS error:", e);
        setError("WebSocket connection failed. Is PersonaPlex running?");
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
          encoderComplexity: 0,
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
    if (decoderWorkerRef.current) {
      decoderWorkerRef.current.terminate();
      decoderWorkerRef.current = null;
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
      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-6">
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
              ? "Connecting to PersonaPlex..."
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

        {/* Transcript */}
        {transcript && (
          <div className="w-full max-w-xs rounded-xl bg-maude-surface p-3">
            <span className="mb-1 block text-[10px] uppercase tracking-wider text-maude-muted">Transcript</span>
            <p className="text-sm text-maude-text">{transcript}</p>
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
          {" \u2022 "}PersonaPlex
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
