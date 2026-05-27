import { g as C, c as G } from "./index-DcCvk7hW.js";
function O(h, y) {
  for (var c = 0; c < y.length; c++) {
    const a = y[c];
    if (typeof a != "string" && !Array.isArray(a)) {
      for (const o in a) if (o !== "default" && !(o in h)) {
        const t = Object.getOwnPropertyDescriptor(a, o);
        t && Object.defineProperty(h, o, t.get ? t : { enumerable: true, get: () => a[o] });
      }
    }
  }
  return Object.freeze(Object.defineProperty(h, Symbol.toStringTag, { value: "Module" }));
}
var N = { exports: {} };
(function(h, y) {
  (function(c, a) {
    h.exports = a();
  })(typeof self < "u" ? self : G, function() {
    return function(c) {
      var a = {};
      function o(t) {
        if (a[t]) return a[t].exports;
        var s = a[t] = { i: t, l: false, exports: {} };
        return c[t].call(s.exports, s, s.exports, o), s.l = true, s.exports;
      }
      return o.m = c, o.c = a, o.d = function(t, s, u) {
        o.o(t, s) || Object.defineProperty(t, s, { enumerable: true, get: u });
      }, o.r = function(t) {
        typeof Symbol < "u" && Symbol.toStringTag && Object.defineProperty(t, Symbol.toStringTag, { value: "Module" }), Object.defineProperty(t, "__esModule", { value: true });
      }, o.t = function(t, s) {
        if (1 & s && (t = o(t)), 8 & s || 4 & s && typeof t == "object" && t && t.__esModule) return t;
        var u = /* @__PURE__ */ Object.create(null);
        if (o.r(u), Object.defineProperty(u, "default", { enumerable: true, value: t }), 2 & s && typeof t != "string") for (var n in t) o.d(u, n, (function(e) {
          return t[e];
        }).bind(null, n));
        return u;
      }, o.n = function(t) {
        var s = t && t.__esModule ? function() {
          return t.default;
        } : function() {
          return t;
        };
        return o.d(s, "a", s), s;
      }, o.o = function(t, s) {
        return Object.prototype.hasOwnProperty.call(t, s);
      }, o.p = "", o(o.s = 0);
    }([function(c, a, o) {
      (function(t) {
        function s(e, r) {
          if (e == null) return {};
          var i, d, f = function(p, g) {
            if (p == null) return {};
            var m, v, b = {}, x = Object.keys(p);
            for (v = 0; v < x.length; v++) m = x[v], g.indexOf(m) >= 0 || (b[m] = p[m]);
            return b;
          }(e, r);
          if (Object.getOwnPropertySymbols) {
            var l = Object.getOwnPropertySymbols(e);
            for (d = 0; d < l.length; d++) i = l[d], r.indexOf(i) >= 0 || Object.prototype.propertyIsEnumerable.call(e, i) && (f[i] = e[i]);
          }
          return f;
        }
        var u = t.AudioContext || t.webkitAudioContext, n = function e() {
          var r = this, i = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
          if (!e.isRecordingSupported()) throw new Error("Recording is not supported in this browser");
          this.state = "inactive", this.config = Object.assign({ bufferLength: 4096, encoderApplication: 2049, encoderFrameSize: 20, encoderPath: "encoderWorker.min.js", encoderSampleRate: 48e3, maxFramesPerPage: 40, mediaTrackConstraints: true, monitorGain: 0, numberOfChannels: 1, recordingGain: 1, resampleQuality: 3, streamPages: false, wavBitDepth: 16, sourceNode: { context: null } }, i), this.encodedSamplePosition = 0, this.initAudioContext(), this.initialize = this.initWorklet().then(function() {
            return r.initEncoder();
          });
        };
        n.isRecordingSupported = function() {
          var e = t.navigator && t.navigator.mediaDevices && t.navigator.mediaDevices.getUserMedia;
          return u && e && t.WebAssembly;
        }, n.version = "8.0.5", n.prototype.clearStream = function() {
          this.stream && (this.stream.getTracks ? this.stream.getTracks().forEach(function(e) {
            return e.stop();
          }) : this.stream.stop());
        }, n.prototype.close = function() {
          return this.monitorGainNode.disconnect(), this.recordingGainNode.disconnect(), this.sourceNode && this.sourceNode.disconnect(), this.clearStream(), this.encoder && (this.encoderNode.disconnect(), this.encoder.postMessage({ command: "close" })), this.config.sourceNode.context ? Promise.resolve() : this.audioContext.close();
        }, n.prototype.encodeBuffers = function(e) {
          if (this.state === "recording") {
            for (var r = [], i = 0; i < e.numberOfChannels; i++) r[i] = e.getChannelData(i);
            this.encoder.postMessage({ command: "encode", buffers: r });
          }
        }, n.prototype.initAudioContext = function() {
          this.audioContext = this.config.sourceNode.context ? this.config.sourceNode.context : new u(), this.monitorGainNode = this.audioContext.createGain(), this.setMonitorGain(this.config.monitorGain), this.recordingGainNode = this.audioContext.createGain(), this.setRecordingGain(this.config.recordingGain);
        }, n.prototype.initEncoder = function() {
          var e = this;
          this.audioContext.audioWorklet ? (this.encoderNode = new AudioWorkletNode(this.audioContext, "encoder-worklet", { numberOfOutputs: 0 }), this.encoder = this.encoderNode.port) : (console.log("audioWorklet support not detected. Falling back to scriptProcessor"), this.encodeBuffers = function() {
            return delete e.encodeBuffers;
          }, this.encoderNode = this.audioContext.createScriptProcessor(this.config.bufferLength, this.config.numberOfChannels, this.config.numberOfChannels), this.encoderNode.onaudioprocess = function(r) {
            var i = r.inputBuffer;
            return e.encodeBuffers(i);
          }, this.encoderNode.connect(this.audioContext.destination), this.encoder = new t.Worker(this.config.encoderPath));
        }, n.prototype.initSourceNode = function() {
          var e = this;
          return this.config.sourceNode.context ? (this.sourceNode = this.config.sourceNode, Promise.resolve()) : t.navigator.mediaDevices.getUserMedia({ audio: this.config.mediaTrackConstraints }).then(function(r) {
            e.stream = r, e.sourceNode = e.audioContext.createMediaStreamSource(r);
          });
        }, n.prototype.initWorker = function() {
          var e = this, r = (this.config.streamPages ? this.streamPage : this.storePage).bind(this);
          return this.recordedPages = [], this.totalLength = 0, new Promise(function(i) {
            e.encoder.addEventListener("message", function l(p) {
              var g = p.data;
              switch (g.message) {
                case "ready":
                  i();
                  break;
                case "page":
                  e.encodedSamplePosition = g.samplePosition, r(g.page);
                  break;
                case "done":
                  e.encoder.removeEventListener("message", l), e.finish();
              }
            }), e.encoder.start && e.encoder.start();
            var d = e.config, f = (d.sourceNode, s(d, ["sourceNode"]));
            e.encoder.postMessage(Object.assign({ command: "init", originalSampleRate: e.audioContext.sampleRate, wavSampleRate: e.audioContext.sampleRate }, f));
          });
        }, n.prototype.initWorklet = function() {
          return this.audioContext.audioWorklet ? this.audioContext.audioWorklet.addModule(this.config.encoderPath) : Promise.resolve();
        }, n.prototype.pause = function(e) {
          var r = this;
          if (this.state === "recording") return this.state = "paused", this.recordingGainNode.disconnect(), e && this.config.streamPages ? new Promise(function(i) {
            r.encoder.addEventListener("message", function d(f) {
              f.data.message === "flushed" && (r.encoder.removeEventListener("message", d), r.onpause(), i());
            }), r.encoder.start && r.encoder.start(), r.encoder.postMessage({ command: "flush" });
          }) : (this.onpause(), Promise.resolve());
        }, n.prototype.resume = function() {
          this.state === "paused" && (this.state = "recording", this.recordingGainNode.connect(this.encoderNode), this.onresume());
        }, n.prototype.setRecordingGain = function(e) {
          this.config.recordingGain = e, this.recordingGainNode && this.audioContext && this.recordingGainNode.gain.setTargetAtTime(e, this.audioContext.currentTime, 0.01);
        }, n.prototype.setMonitorGain = function(e) {
          this.config.monitorGain = e, this.monitorGainNode && this.audioContext && this.monitorGainNode.gain.setTargetAtTime(e, this.audioContext.currentTime, 0.01);
        }, n.prototype.start = function() {
          var e = this;
          return this.state === "inactive" ? (this.state = "loading", this.encodedSamplePosition = 0, this.audioContext.resume().then(function() {
            return e.initialize;
          }).then(function() {
            return Promise.all([e.initSourceNode(), e.initWorker()]);
          }).then(function() {
            e.state = "recording", e.encoder.postMessage({ command: "getHeaderPages" }), e.sourceNode.connect(e.monitorGainNode), e.sourceNode.connect(e.recordingGainNode), e.monitorGainNode.connect(e.audioContext.destination), e.recordingGainNode.connect(e.encoderNode), e.onstart();
          }).catch(function(r) {
            throw e.state = "inactive", r;
          })) : Promise.resolve();
        }, n.prototype.stop = function() {
          var e = this;
          return this.state === "paused" || this.state === "recording" ? (this.state = "inactive", this.recordingGainNode.connect(this.encoderNode), this.monitorGainNode.disconnect(), this.clearStream(), new Promise(function(r) {
            e.encoder.addEventListener("message", function i(d) {
              d.data.message === "done" && (e.encoder.removeEventListener("message", i), r());
            }), e.encoder.start && e.encoder.start(), e.encoder.postMessage({ command: "done" });
          })) : Promise.resolve();
        }, n.prototype.storePage = function(e) {
          this.recordedPages.push(e), this.totalLength += e.length;
        }, n.prototype.streamPage = function(e) {
          this.ondataavailable(e);
        }, n.prototype.finish = function() {
          if (!this.config.streamPages) {
            var e = new Uint8Array(this.totalLength);
            this.recordedPages.reduce(function(r, i) {
              return e.set(i, r), r + i.length;
            }, 0), this.ondataavailable(e);
          }
          this.onstop();
        }, n.prototype.ondataavailable = function() {
        }, n.prototype.onpause = function() {
        }, n.prototype.onresume = function() {
        }, n.prototype.onstart = function() {
        }, n.prototype.onstop = function() {
        }, c.exports = n;
      }).call(this, o(1));
    }, function(c, a) {
      var o;
      o = /* @__PURE__ */ function() {
        return this;
      }();
      try {
        o = o || new Function("return this")();
      } catch {
        typeof window == "object" && (o = window);
      }
      c.exports = o;
    }]);
  });
})(N);
var P = N.exports;
const S = C(P), k = O({ __proto__: null, default: S }, [P]);
export {
  k as r
};
