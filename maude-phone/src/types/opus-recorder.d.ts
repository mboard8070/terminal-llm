declare module "opus-recorder" {
  interface RecorderOptions {
    encoderPath?: string;
    bufferLength?: number;
    encoderFrameSize?: number;
    encoderSampleRate?: number;
    maxFramesPerPage?: number;
    numberOfChannels?: number;
    recordingGain?: number;
    resampleQuality?: number;
    encoderComplexity?: number;
    encoderApplication?: number;
    streamPages?: boolean;
    mediaTrackConstraints?: MediaStreamConstraints;
    sourceNode?: MediaStreamAudioSourceNode;
  }

  class Recorder {
    constructor(options?: RecorderOptions);
    ondataavailable: (data: Uint8Array) => void;
    onstart: () => void;
    onstop: () => void;
    encodedSamplePosition: number;
    start(stream?: MediaStream): void;
    stop(): void;
    pause(): void;
    resume(): void;
    close(): void;
  }

  export default Recorder;
}

declare module "opus-recorder/dist/encoderWorker.min.js?url" {
  const url: string;
  export default url;
}
