let ay, Qc;
let __tla = (async () => {
  function Hc(e, t) {
    for (var n = 0; n < t.length; n++) {
      const r = t[n];
      if (typeof r != "string" && !Array.isArray(r)) {
        for (const l in r) if (l !== "default" && !(l in e)) {
          const a = Object.getOwnPropertyDescriptor(r, l);
          a && Object.defineProperty(e, l, a.get ? a : {
            enumerable: true,
            get: () => r[l]
          });
        }
      }
    }
    return Object.freeze(Object.defineProperty(e, Symbol.toStringTag, {
      value: "Module"
    }));
  }
  (function() {
    const t = document.createElement("link").relList;
    if (t && t.supports && t.supports("modulepreload")) return;
    for (const l of document.querySelectorAll('link[rel="modulepreload"]')) r(l);
    new MutationObserver((l) => {
      for (const a of l) if (a.type === "childList") for (const o of a.addedNodes) o.tagName === "LINK" && o.rel === "modulepreload" && r(o);
    }).observe(document, {
      childList: true,
      subtree: true
    });
    function n(l) {
      const a = {};
      return l.integrity && (a.integrity = l.integrity), l.referrerPolicy && (a.referrerPolicy = l.referrerPolicy), l.crossOrigin === "use-credentials" ? a.credentials = "include" : l.crossOrigin === "anonymous" ? a.credentials = "omit" : a.credentials = "same-origin", a;
    }
    function r(l) {
      if (l.ep) return;
      l.ep = true;
      const a = n(l);
      fetch(l.href, a);
    }
  })();
  ay = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
  Qc = function(e) {
    return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
  };
  var Kc = {
    exports: {}
  }, Ya = {}, Gc = {
    exports: {}
  }, re = {};
  var Pl = Symbol.for("react.element"), Om = Symbol.for("react.portal"), Im = Symbol.for("react.fragment"), zm = Symbol.for("react.strict_mode"), Am = Symbol.for("react.profiler"), Um = Symbol.for("react.provider"), $m = Symbol.for("react.context"), Fm = Symbol.for("react.forward_ref"), Bm = Symbol.for("react.suspense"), Wm = Symbol.for("react.memo"), Vm = Symbol.for("react.lazy"), au = Symbol.iterator;
  function Hm(e) {
    return e === null || typeof e != "object" ? null : (e = au && e[au] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var Yc = {
    isMounted: function() {
      return false;
    },
    enqueueForceUpdate: function() {
    },
    enqueueReplaceState: function() {
    },
    enqueueSetState: function() {
    }
  }, Jc = Object.assign, Xc = {};
  function Rr(e, t, n) {
    this.props = e, this.context = t, this.refs = Xc, this.updater = n || Yc;
  }
  Rr.prototype.isReactComponent = {};
  Rr.prototype.setState = function(e, t) {
    if (typeof e != "object" && typeof e != "function" && e != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
    this.updater.enqueueSetState(this, e, t, "setState");
  };
  Rr.prototype.forceUpdate = function(e) {
    this.updater.enqueueForceUpdate(this, e, "forceUpdate");
  };
  function Zc() {
  }
  Zc.prototype = Rr.prototype;
  function Ys(e, t, n) {
    this.props = e, this.context = t, this.refs = Xc, this.updater = n || Yc;
  }
  var Js = Ys.prototype = new Zc();
  Js.constructor = Ys;
  Jc(Js, Rr.prototype);
  Js.isPureReactComponent = true;
  var ou = Array.isArray, qc = Object.prototype.hasOwnProperty, Xs = {
    current: null
  }, ed = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function td(e, t, n) {
    var r, l = {}, a = null, o = null;
    if (t != null) for (r in t.ref !== void 0 && (o = t.ref), t.key !== void 0 && (a = "" + t.key), t) qc.call(t, r) && !ed.hasOwnProperty(r) && (l[r] = t[r]);
    var i = arguments.length - 2;
    if (i === 1) l.children = n;
    else if (1 < i) {
      for (var s = Array(i), c = 0; c < i; c++) s[c] = arguments[c + 2];
      l.children = s;
    }
    if (e && e.defaultProps) for (r in i = e.defaultProps, i) l[r] === void 0 && (l[r] = i[r]);
    return {
      $$typeof: Pl,
      type: e,
      key: a,
      ref: o,
      props: l,
      _owner: Xs.current
    };
  }
  function Qm(e, t) {
    return {
      $$typeof: Pl,
      type: e.type,
      key: t,
      ref: e.ref,
      props: e.props,
      _owner: e._owner
    };
  }
  function Zs(e) {
    return typeof e == "object" && e !== null && e.$$typeof === Pl;
  }
  function Km(e) {
    var t = {
      "=": "=0",
      ":": "=2"
    };
    return "$" + e.replace(/[=:]/g, function(n) {
      return t[n];
    });
  }
  var su = /\/+/g;
  function yo(e, t) {
    return typeof e == "object" && e !== null && e.key != null ? Km("" + e.key) : t.toString(36);
  }
  function sa(e, t, n, r, l) {
    var a = typeof e;
    (a === "undefined" || a === "boolean") && (e = null);
    var o = false;
    if (e === null) o = true;
    else switch (a) {
      case "string":
      case "number":
        o = true;
        break;
      case "object":
        switch (e.$$typeof) {
          case Pl:
          case Om:
            o = true;
        }
    }
    if (o) return o = e, l = l(o), e = r === "" ? "." + yo(o, 0) : r, ou(l) ? (n = "", e != null && (n = e.replace(su, "$&/") + "/"), sa(l, t, n, "", function(c) {
      return c;
    })) : l != null && (Zs(l) && (l = Qm(l, n + (!l.key || o && o.key === l.key ? "" : ("" + l.key).replace(su, "$&/") + "/") + e)), t.push(l)), 1;
    if (o = 0, r = r === "" ? "." : r + ":", ou(e)) for (var i = 0; i < e.length; i++) {
      a = e[i];
      var s = r + yo(a, i);
      o += sa(a, t, n, s, l);
    }
    else if (s = Hm(e), typeof s == "function") for (e = s.call(e), i = 0; !(a = e.next()).done; ) a = a.value, s = r + yo(a, i++), o += sa(a, t, n, s, l);
    else if (a === "object") throw t = String(e), Error("Objects are not valid as a React child (found: " + (t === "[object Object]" ? "object with keys {" + Object.keys(e).join(", ") + "}" : t) + "). If you meant to render a collection of children, use an array instead.");
    return o;
  }
  function Bl(e, t, n) {
    if (e == null) return e;
    var r = [], l = 0;
    return sa(e, r, "", "", function(a) {
      return t.call(n, a, l++);
    }), r;
  }
  function Gm(e) {
    if (e._status === -1) {
      var t = e._result;
      t = t(), t.then(function(n) {
        (e._status === 0 || e._status === -1) && (e._status = 1, e._result = n);
      }, function(n) {
        (e._status === 0 || e._status === -1) && (e._status = 2, e._result = n);
      }), e._status === -1 && (e._status = 0, e._result = t);
    }
    if (e._status === 1) return e._result.default;
    throw e._result;
  }
  var lt = {
    current: null
  }, ia = {
    transition: null
  }, Ym = {
    ReactCurrentDispatcher: lt,
    ReactCurrentBatchConfig: ia,
    ReactCurrentOwner: Xs
  };
  function nd() {
    throw Error("act(...) is not supported in production builds of React.");
  }
  re.Children = {
    map: Bl,
    forEach: function(e, t, n) {
      Bl(e, function() {
        t.apply(this, arguments);
      }, n);
    },
    count: function(e) {
      var t = 0;
      return Bl(e, function() {
        t++;
      }), t;
    },
    toArray: function(e) {
      return Bl(e, function(t) {
        return t;
      }) || [];
    },
    only: function(e) {
      if (!Zs(e)) throw Error("React.Children.only expected to receive a single React element child.");
      return e;
    }
  };
  re.Component = Rr;
  re.Fragment = Im;
  re.Profiler = Am;
  re.PureComponent = Ys;
  re.StrictMode = zm;
  re.Suspense = Bm;
  re.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Ym;
  re.act = nd;
  re.cloneElement = function(e, t, n) {
    if (e == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + e + ".");
    var r = Jc({}, e.props), l = e.key, a = e.ref, o = e._owner;
    if (t != null) {
      if (t.ref !== void 0 && (a = t.ref, o = Xs.current), t.key !== void 0 && (l = "" + t.key), e.type && e.type.defaultProps) var i = e.type.defaultProps;
      for (s in t) qc.call(t, s) && !ed.hasOwnProperty(s) && (r[s] = t[s] === void 0 && i !== void 0 ? i[s] : t[s]);
    }
    var s = arguments.length - 2;
    if (s === 1) r.children = n;
    else if (1 < s) {
      i = Array(s);
      for (var c = 0; c < s; c++) i[c] = arguments[c + 2];
      r.children = i;
    }
    return {
      $$typeof: Pl,
      type: e.type,
      key: l,
      ref: a,
      props: r,
      _owner: o
    };
  };
  re.createContext = function(e) {
    return e = {
      $$typeof: $m,
      _currentValue: e,
      _currentValue2: e,
      _threadCount: 0,
      Provider: null,
      Consumer: null,
      _defaultValue: null,
      _globalName: null
    }, e.Provider = {
      $$typeof: Um,
      _context: e
    }, e.Consumer = e;
  };
  re.createElement = td;
  re.createFactory = function(e) {
    var t = td.bind(null, e);
    return t.type = e, t;
  };
  re.createRef = function() {
    return {
      current: null
    };
  };
  re.forwardRef = function(e) {
    return {
      $$typeof: Fm,
      render: e
    };
  };
  re.isValidElement = Zs;
  re.lazy = function(e) {
    return {
      $$typeof: Vm,
      _payload: {
        _status: -1,
        _result: e
      },
      _init: Gm
    };
  };
  re.memo = function(e, t) {
    return {
      $$typeof: Wm,
      type: e,
      compare: t === void 0 ? null : t
    };
  };
  re.startTransition = function(e) {
    var t = ia.transition;
    ia.transition = {};
    try {
      e();
    } finally {
      ia.transition = t;
    }
  };
  re.unstable_act = nd;
  re.useCallback = function(e, t) {
    return lt.current.useCallback(e, t);
  };
  re.useContext = function(e) {
    return lt.current.useContext(e);
  };
  re.useDebugValue = function() {
  };
  re.useDeferredValue = function(e) {
    return lt.current.useDeferredValue(e);
  };
  re.useEffect = function(e, t) {
    return lt.current.useEffect(e, t);
  };
  re.useId = function() {
    return lt.current.useId();
  };
  re.useImperativeHandle = function(e, t, n) {
    return lt.current.useImperativeHandle(e, t, n);
  };
  re.useInsertionEffect = function(e, t) {
    return lt.current.useInsertionEffect(e, t);
  };
  re.useLayoutEffect = function(e, t) {
    return lt.current.useLayoutEffect(e, t);
  };
  re.useMemo = function(e, t) {
    return lt.current.useMemo(e, t);
  };
  re.useReducer = function(e, t, n) {
    return lt.current.useReducer(e, t, n);
  };
  re.useRef = function(e) {
    return lt.current.useRef(e);
  };
  re.useState = function(e) {
    return lt.current.useState(e);
  };
  re.useSyncExternalStore = function(e, t, n) {
    return lt.current.useSyncExternalStore(e, t, n);
  };
  re.useTransition = function() {
    return lt.current.useTransition();
  };
  re.version = "18.3.1";
  Gc.exports = re;
  var v = Gc.exports;
  const Jm = Qc(v), Xm = Hc({
    __proto__: null,
    default: Jm
  }, [
    v
  ]);
  var Zm = v, qm = Symbol.for("react.element"), ep = Symbol.for("react.fragment"), tp = Object.prototype.hasOwnProperty, np = Zm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, rp = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function rd(e, t, n) {
    var r, l = {}, a = null, o = null;
    n !== void 0 && (a = "" + n), t.key !== void 0 && (a = "" + t.key), t.ref !== void 0 && (o = t.ref);
    for (r in t) tp.call(t, r) && !rp.hasOwnProperty(r) && (l[r] = t[r]);
    if (e && e.defaultProps) for (r in t = e.defaultProps, t) l[r] === void 0 && (l[r] = t[r]);
    return {
      $$typeof: qm,
      type: e,
      key: a,
      ref: o,
      props: l,
      _owner: np.current
    };
  }
  Ya.Fragment = ep;
  Ya.jsx = rd;
  Ya.jsxs = rd;
  Kc.exports = Ya;
  var u = Kc.exports, Xo = {}, ld = {
    exports: {}
  }, yt = {}, ad = {
    exports: {}
  }, od = {};
  (function(e) {
    function t(M, H) {
      var W = M.length;
      M.push(H);
      e: for (; 0 < W; ) {
        var le = W - 1 >>> 1, te = M[le];
        if (0 < l(te, H)) M[le] = H, M[W] = te, W = le;
        else break e;
      }
    }
    function n(M) {
      return M.length === 0 ? null : M[0];
    }
    function r(M) {
      if (M.length === 0) return null;
      var H = M[0], W = M.pop();
      if (W !== H) {
        M[0] = W;
        e: for (var le = 0, te = M.length, fe = te >>> 1; le < fe; ) {
          var Z = 2 * (le + 1) - 1, ue = M[Z], Se = Z + 1, ne = M[Se];
          if (0 > l(ue, W)) Se < te && 0 > l(ne, ue) ? (M[le] = ne, M[Se] = W, le = Se) : (M[le] = ue, M[Z] = W, le = Z);
          else if (Se < te && 0 > l(ne, W)) M[le] = ne, M[Se] = W, le = Se;
          else break e;
        }
      }
      return H;
    }
    function l(M, H) {
      var W = M.sortIndex - H.sortIndex;
      return W !== 0 ? W : M.id - H.id;
    }
    if (typeof performance == "object" && typeof performance.now == "function") {
      var a = performance;
      e.unstable_now = function() {
        return a.now();
      };
    } else {
      var o = Date, i = o.now();
      e.unstable_now = function() {
        return o.now() - i;
      };
    }
    var s = [], c = [], m = 1, d = null, g = 3, x = false, w = false, S = false, _ = typeof setTimeout == "function" ? setTimeout : null, p = typeof clearTimeout == "function" ? clearTimeout : null, f = typeof setImmediate < "u" ? setImmediate : null;
    typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
    function h(M) {
      for (var H = n(c); H !== null; ) {
        if (H.callback === null) r(c);
        else if (H.startTime <= M) r(c), H.sortIndex = H.expirationTime, t(s, H);
        else break;
        H = n(c);
      }
    }
    function j(M) {
      if (S = false, h(M), !w) if (n(s) !== null) w = true, Ve(C);
      else {
        var H = n(c);
        H !== null && ft(j, H.startTime - M);
      }
    }
    function C(M, H) {
      w = false, S && (S = false, p(P), P = -1), x = true;
      var W = g;
      try {
        for (h(H), d = n(s); d !== null && (!(d.expirationTime > H) || M && !G()); ) {
          var le = d.callback;
          if (typeof le == "function") {
            d.callback = null, g = d.priorityLevel;
            var te = le(d.expirationTime <= H);
            H = e.unstable_now(), typeof te == "function" ? d.callback = te : d === n(s) && r(s), h(H);
          } else r(s);
          d = n(s);
        }
        if (d !== null) var fe = true;
        else {
          var Z = n(c);
          Z !== null && ft(j, Z.startTime - H), fe = false;
        }
        return fe;
      } finally {
        d = null, g = W, x = false;
      }
    }
    var R = false, k = null, P = -1, $ = 5, L = -1;
    function G() {
      return !(e.unstable_now() - L < $);
    }
    function X() {
      if (k !== null) {
        var M = e.unstable_now();
        L = M;
        var H = true;
        try {
          H = k(true, M);
        } finally {
          H ? de() : (R = false, k = null);
        }
      } else R = false;
    }
    var de;
    if (typeof f == "function") de = function() {
      f(X);
    };
    else if (typeof MessageChannel < "u") {
      var se = new MessageChannel(), Ce = se.port2;
      se.port1.onmessage = X, de = function() {
        Ce.postMessage(null);
      };
    } else de = function() {
      _(X, 0);
    };
    function Ve(M) {
      k = M, R || (R = true, de());
    }
    function ft(M, H) {
      P = _(function() {
        M(e.unstable_now());
      }, H);
    }
    e.unstable_IdlePriority = 5, e.unstable_ImmediatePriority = 1, e.unstable_LowPriority = 4, e.unstable_NormalPriority = 3, e.unstable_Profiling = null, e.unstable_UserBlockingPriority = 2, e.unstable_cancelCallback = function(M) {
      M.callback = null;
    }, e.unstable_continueExecution = function() {
      w || x || (w = true, Ve(C));
    }, e.unstable_forceFrameRate = function(M) {
      0 > M || 125 < M ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : $ = 0 < M ? Math.floor(1e3 / M) : 5;
    }, e.unstable_getCurrentPriorityLevel = function() {
      return g;
    }, e.unstable_getFirstCallbackNode = function() {
      return n(s);
    }, e.unstable_next = function(M) {
      switch (g) {
        case 1:
        case 2:
        case 3:
          var H = 3;
          break;
        default:
          H = g;
      }
      var W = g;
      g = H;
      try {
        return M();
      } finally {
        g = W;
      }
    }, e.unstable_pauseExecution = function() {
    }, e.unstable_requestPaint = function() {
    }, e.unstable_runWithPriority = function(M, H) {
      switch (M) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          break;
        default:
          M = 3;
      }
      var W = g;
      g = M;
      try {
        return H();
      } finally {
        g = W;
      }
    }, e.unstable_scheduleCallback = function(M, H, W) {
      var le = e.unstable_now();
      switch (typeof W == "object" && W !== null ? (W = W.delay, W = typeof W == "number" && 0 < W ? le + W : le) : W = le, M) {
        case 1:
          var te = -1;
          break;
        case 2:
          te = 250;
          break;
        case 5:
          te = 1073741823;
          break;
        case 4:
          te = 1e4;
          break;
        default:
          te = 5e3;
      }
      return te = W + te, M = {
        id: m++,
        callback: H,
        priorityLevel: M,
        startTime: W,
        expirationTime: te,
        sortIndex: -1
      }, W > le ? (M.sortIndex = W, t(c, M), n(s) === null && M === n(c) && (S ? (p(P), P = -1) : S = true, ft(j, W - le))) : (M.sortIndex = te, t(s, M), w || x || (w = true, Ve(C))), M;
    }, e.unstable_shouldYield = G, e.unstable_wrapCallback = function(M) {
      var H = g;
      return function() {
        var W = g;
        g = H;
        try {
          return M.apply(this, arguments);
        } finally {
          g = W;
        }
      };
    };
  })(od);
  ad.exports = od;
  var lp = ad.exports;
  var ap = v, xt = lp;
  function b(e) {
    for (var t = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, n = 1; n < arguments.length; n++) t += "&args[]=" + encodeURIComponent(arguments[n]);
    return "Minified React error #" + e + "; visit " + t + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  var sd = /* @__PURE__ */ new Set(), ul = {};
  function Kn(e, t) {
    kr(e, t), kr(e + "Capture", t);
  }
  function kr(e, t) {
    for (ul[e] = t, e = 0; e < t.length; e++) sd.add(t[e]);
  }
  var Xt = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), Zo = Object.prototype.hasOwnProperty, op = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, iu = {}, uu = {};
  function sp(e) {
    return Zo.call(uu, e) ? true : Zo.call(iu, e) ? false : op.test(e) ? uu[e] = true : (iu[e] = true, false);
  }
  function ip(e, t, n, r) {
    if (n !== null && n.type === 0) return false;
    switch (typeof t) {
      case "function":
      case "symbol":
        return true;
      case "boolean":
        return r ? false : n !== null ? !n.acceptsBooleans : (e = e.toLowerCase().slice(0, 5), e !== "data-" && e !== "aria-");
      default:
        return false;
    }
  }
  function up(e, t, n, r) {
    if (t === null || typeof t > "u" || ip(e, t, n, r)) return true;
    if (r) return false;
    if (n !== null) switch (n.type) {
      case 3:
        return !t;
      case 4:
        return t === false;
      case 5:
        return isNaN(t);
      case 6:
        return isNaN(t) || 1 > t;
    }
    return false;
  }
  function at(e, t, n, r, l, a, o) {
    this.acceptsBooleans = t === 2 || t === 3 || t === 4, this.attributeName = r, this.attributeNamespace = l, this.mustUseProperty = n, this.propertyName = e, this.type = t, this.sanitizeURL = a, this.removeEmptyString = o;
  }
  var Ye = {};
  "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e) {
    Ye[e] = new at(e, 0, false, e, null, false, false);
  });
  [
    [
      "acceptCharset",
      "accept-charset"
    ],
    [
      "className",
      "class"
    ],
    [
      "htmlFor",
      "for"
    ],
    [
      "httpEquiv",
      "http-equiv"
    ]
  ].forEach(function(e) {
    var t = e[0];
    Ye[t] = new at(t, 1, false, e[1], null, false, false);
  });
  [
    "contentEditable",
    "draggable",
    "spellCheck",
    "value"
  ].forEach(function(e) {
    Ye[e] = new at(e, 2, false, e.toLowerCase(), null, false, false);
  });
  [
    "autoReverse",
    "externalResourcesRequired",
    "focusable",
    "preserveAlpha"
  ].forEach(function(e) {
    Ye[e] = new at(e, 2, false, e, null, false, false);
  });
  "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e) {
    Ye[e] = new at(e, 3, false, e.toLowerCase(), null, false, false);
  });
  [
    "checked",
    "multiple",
    "muted",
    "selected"
  ].forEach(function(e) {
    Ye[e] = new at(e, 3, true, e, null, false, false);
  });
  [
    "capture",
    "download"
  ].forEach(function(e) {
    Ye[e] = new at(e, 4, false, e, null, false, false);
  });
  [
    "cols",
    "rows",
    "size",
    "span"
  ].forEach(function(e) {
    Ye[e] = new at(e, 6, false, e, null, false, false);
  });
  [
    "rowSpan",
    "start"
  ].forEach(function(e) {
    Ye[e] = new at(e, 5, false, e.toLowerCase(), null, false, false);
  });
  var qs = /[\-:]([a-z])/g;
  function ei(e) {
    return e[1].toUpperCase();
  }
  "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e) {
    var t = e.replace(qs, ei);
    Ye[t] = new at(t, 1, false, e, null, false, false);
  });
  "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e) {
    var t = e.replace(qs, ei);
    Ye[t] = new at(t, 1, false, e, "http://www.w3.org/1999/xlink", false, false);
  });
  [
    "xml:base",
    "xml:lang",
    "xml:space"
  ].forEach(function(e) {
    var t = e.replace(qs, ei);
    Ye[t] = new at(t, 1, false, e, "http://www.w3.org/XML/1998/namespace", false, false);
  });
  [
    "tabIndex",
    "crossOrigin"
  ].forEach(function(e) {
    Ye[e] = new at(e, 1, false, e.toLowerCase(), null, false, false);
  });
  Ye.xlinkHref = new at("xlinkHref", 1, false, "xlink:href", "http://www.w3.org/1999/xlink", true, false);
  [
    "src",
    "href",
    "action",
    "formAction"
  ].forEach(function(e) {
    Ye[e] = new at(e, 1, false, e.toLowerCase(), null, true, true);
  });
  function ti(e, t, n, r) {
    var l = Ye.hasOwnProperty(t) ? Ye[t] : null;
    (l !== null ? l.type !== 0 : r || !(2 < t.length) || t[0] !== "o" && t[0] !== "O" || t[1] !== "n" && t[1] !== "N") && (up(t, n, l, r) && (n = null), r || l === null ? sp(t) && (n === null ? e.removeAttribute(t) : e.setAttribute(t, "" + n)) : l.mustUseProperty ? e[l.propertyName] = n === null ? l.type === 3 ? false : "" : n : (t = l.attributeName, r = l.attributeNamespace, n === null ? e.removeAttribute(t) : (l = l.type, n = l === 3 || l === 4 && n === true ? "" : "" + n, r ? e.setAttributeNS(r, t, n) : e.setAttribute(t, n))));
  }
  var tn = ap.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, Wl = Symbol.for("react.element"), lr = Symbol.for("react.portal"), ar = Symbol.for("react.fragment"), ni = Symbol.for("react.strict_mode"), qo = Symbol.for("react.profiler"), id = Symbol.for("react.provider"), ud = Symbol.for("react.context"), ri = Symbol.for("react.forward_ref"), es = Symbol.for("react.suspense"), ts = Symbol.for("react.suspense_list"), li = Symbol.for("react.memo"), an = Symbol.for("react.lazy"), cd = Symbol.for("react.offscreen"), cu = Symbol.iterator;
  function Ir(e) {
    return e === null || typeof e != "object" ? null : (e = cu && e[cu] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var Re = Object.assign, wo;
  function Gr(e) {
    if (wo === void 0) try {
      throw Error();
    } catch (n) {
      var t = n.stack.trim().match(/\n( *(at )?)/);
      wo = t && t[1] || "";
    }
    return `
` + wo + e;
  }
  var So = false;
  function ko(e, t) {
    if (!e || So) return "";
    So = true;
    var n = Error.prepareStackTrace;
    Error.prepareStackTrace = void 0;
    try {
      if (t) if (t = function() {
        throw Error();
      }, Object.defineProperty(t.prototype, "props", {
        set: function() {
          throw Error();
        }
      }), typeof Reflect == "object" && Reflect.construct) {
        try {
          Reflect.construct(t, []);
        } catch (c) {
          var r = c;
        }
        Reflect.construct(e, [], t);
      } else {
        try {
          t.call();
        } catch (c) {
          r = c;
        }
        e.call(t.prototype);
      }
      else {
        try {
          throw Error();
        } catch (c) {
          r = c;
        }
        e();
      }
    } catch (c) {
      if (c && r && typeof c.stack == "string") {
        for (var l = c.stack.split(`
`), a = r.stack.split(`
`), o = l.length - 1, i = a.length - 1; 1 <= o && 0 <= i && l[o] !== a[i]; ) i--;
        for (; 1 <= o && 0 <= i; o--, i--) if (l[o] !== a[i]) {
          if (o !== 1 || i !== 1) do
            if (o--, i--, 0 > i || l[o] !== a[i]) {
              var s = `
` + l[o].replace(" at new ", " at ");
              return e.displayName && s.includes("<anonymous>") && (s = s.replace("<anonymous>", e.displayName)), s;
            }
          while (1 <= o && 0 <= i);
          break;
        }
      }
    } finally {
      So = false, Error.prepareStackTrace = n;
    }
    return (e = e ? e.displayName || e.name : "") ? Gr(e) : "";
  }
  function cp(e) {
    switch (e.tag) {
      case 5:
        return Gr(e.type);
      case 16:
        return Gr("Lazy");
      case 13:
        return Gr("Suspense");
      case 19:
        return Gr("SuspenseList");
      case 0:
      case 2:
      case 15:
        return e = ko(e.type, false), e;
      case 11:
        return e = ko(e.type.render, false), e;
      case 1:
        return e = ko(e.type, true), e;
      default:
        return "";
    }
  }
  function ns(e) {
    if (e == null) return null;
    if (typeof e == "function") return e.displayName || e.name || null;
    if (typeof e == "string") return e;
    switch (e) {
      case ar:
        return "Fragment";
      case lr:
        return "Portal";
      case qo:
        return "Profiler";
      case ni:
        return "StrictMode";
      case es:
        return "Suspense";
      case ts:
        return "SuspenseList";
    }
    if (typeof e == "object") switch (e.$$typeof) {
      case ud:
        return (e.displayName || "Context") + ".Consumer";
      case id:
        return (e._context.displayName || "Context") + ".Provider";
      case ri:
        var t = e.render;
        return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
      case li:
        return t = e.displayName || null, t !== null ? t : ns(e.type) || "Memo";
      case an:
        t = e._payload, e = e._init;
        try {
          return ns(e(t));
        } catch {
        }
    }
    return null;
  }
  function dp(e) {
    var t = e.type;
    switch (e.tag) {
      case 24:
        return "Cache";
      case 9:
        return (t.displayName || "Context") + ".Consumer";
      case 10:
        return (t._context.displayName || "Context") + ".Provider";
      case 18:
        return "DehydratedFragment";
      case 11:
        return e = t.render, e = e.displayName || e.name || "", t.displayName || (e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef");
      case 7:
        return "Fragment";
      case 5:
        return t;
      case 4:
        return "Portal";
      case 3:
        return "Root";
      case 6:
        return "Text";
      case 16:
        return ns(t);
      case 8:
        return t === ni ? "StrictMode" : "Mode";
      case 22:
        return "Offscreen";
      case 12:
        return "Profiler";
      case 21:
        return "Scope";
      case 13:
        return "Suspense";
      case 19:
        return "SuspenseList";
      case 25:
        return "TracingMarker";
      case 1:
      case 0:
      case 17:
      case 2:
      case 14:
      case 15:
        if (typeof t == "function") return t.displayName || t.name || null;
        if (typeof t == "string") return t;
    }
    return null;
  }
  function kn(e) {
    switch (typeof e) {
      case "boolean":
      case "number":
      case "string":
      case "undefined":
        return e;
      case "object":
        return e;
      default:
        return "";
    }
  }
  function dd(e) {
    var t = e.type;
    return (e = e.nodeName) && e.toLowerCase() === "input" && (t === "checkbox" || t === "radio");
  }
  function fp(e) {
    var t = dd(e) ? "checked" : "value", n = Object.getOwnPropertyDescriptor(e.constructor.prototype, t), r = "" + e[t];
    if (!e.hasOwnProperty(t) && typeof n < "u" && typeof n.get == "function" && typeof n.set == "function") {
      var l = n.get, a = n.set;
      return Object.defineProperty(e, t, {
        configurable: true,
        get: function() {
          return l.call(this);
        },
        set: function(o) {
          r = "" + o, a.call(this, o);
        }
      }), Object.defineProperty(e, t, {
        enumerable: n.enumerable
      }), {
        getValue: function() {
          return r;
        },
        setValue: function(o) {
          r = "" + o;
        },
        stopTracking: function() {
          e._valueTracker = null, delete e[t];
        }
      };
    }
  }
  function Vl(e) {
    e._valueTracker || (e._valueTracker = fp(e));
  }
  function fd(e) {
    if (!e) return false;
    var t = e._valueTracker;
    if (!t) return true;
    var n = t.getValue(), r = "";
    return e && (r = dd(e) ? e.checked ? "true" : "false" : e.value), e = r, e !== n ? (t.setValue(e), true) : false;
  }
  function Sa(e) {
    if (e = e || (typeof document < "u" ? document : void 0), typeof e > "u") return null;
    try {
      return e.activeElement || e.body;
    } catch {
      return e.body;
    }
  }
  function rs(e, t) {
    var n = t.checked;
    return Re({}, t, {
      defaultChecked: void 0,
      defaultValue: void 0,
      value: void 0,
      checked: n ?? e._wrapperState.initialChecked
    });
  }
  function du(e, t) {
    var n = t.defaultValue == null ? "" : t.defaultValue, r = t.checked != null ? t.checked : t.defaultChecked;
    n = kn(t.value != null ? t.value : n), e._wrapperState = {
      initialChecked: r,
      initialValue: n,
      controlled: t.type === "checkbox" || t.type === "radio" ? t.checked != null : t.value != null
    };
  }
  function md(e, t) {
    t = t.checked, t != null && ti(e, "checked", t, false);
  }
  function ls(e, t) {
    md(e, t);
    var n = kn(t.value), r = t.type;
    if (n != null) r === "number" ? (n === 0 && e.value === "" || e.value != n) && (e.value = "" + n) : e.value !== "" + n && (e.value = "" + n);
    else if (r === "submit" || r === "reset") {
      e.removeAttribute("value");
      return;
    }
    t.hasOwnProperty("value") ? as(e, t.type, n) : t.hasOwnProperty("defaultValue") && as(e, t.type, kn(t.defaultValue)), t.checked == null && t.defaultChecked != null && (e.defaultChecked = !!t.defaultChecked);
  }
  function fu(e, t, n) {
    if (t.hasOwnProperty("value") || t.hasOwnProperty("defaultValue")) {
      var r = t.type;
      if (!(r !== "submit" && r !== "reset" || t.value !== void 0 && t.value !== null)) return;
      t = "" + e._wrapperState.initialValue, n || t === e.value || (e.value = t), e.defaultValue = t;
    }
    n = e.name, n !== "" && (e.name = ""), e.defaultChecked = !!e._wrapperState.initialChecked, n !== "" && (e.name = n);
  }
  function as(e, t, n) {
    (t !== "number" || Sa(e.ownerDocument) !== e) && (n == null ? e.defaultValue = "" + e._wrapperState.initialValue : e.defaultValue !== "" + n && (e.defaultValue = "" + n));
  }
  var Yr = Array.isArray;
  function gr(e, t, n, r) {
    if (e = e.options, t) {
      t = {};
      for (var l = 0; l < n.length; l++) t["$" + n[l]] = true;
      for (n = 0; n < e.length; n++) l = t.hasOwnProperty("$" + e[n].value), e[n].selected !== l && (e[n].selected = l), l && r && (e[n].defaultSelected = true);
    } else {
      for (n = "" + kn(n), t = null, l = 0; l < e.length; l++) {
        if (e[l].value === n) {
          e[l].selected = true, r && (e[l].defaultSelected = true);
          return;
        }
        t !== null || e[l].disabled || (t = e[l]);
      }
      t !== null && (t.selected = true);
    }
  }
  function os(e, t) {
    if (t.dangerouslySetInnerHTML != null) throw Error(b(91));
    return Re({}, t, {
      value: void 0,
      defaultValue: void 0,
      children: "" + e._wrapperState.initialValue
    });
  }
  function mu(e, t) {
    var n = t.value;
    if (n == null) {
      if (n = t.children, t = t.defaultValue, n != null) {
        if (t != null) throw Error(b(92));
        if (Yr(n)) {
          if (1 < n.length) throw Error(b(93));
          n = n[0];
        }
        t = n;
      }
      t == null && (t = ""), n = t;
    }
    e._wrapperState = {
      initialValue: kn(n)
    };
  }
  function pd(e, t) {
    var n = kn(t.value), r = kn(t.defaultValue);
    n != null && (n = "" + n, n !== e.value && (e.value = n), t.defaultValue == null && e.defaultValue !== n && (e.defaultValue = n)), r != null && (e.defaultValue = "" + r);
  }
  function pu(e) {
    var t = e.textContent;
    t === e._wrapperState.initialValue && t !== "" && t !== null && (e.value = t);
  }
  function hd(e) {
    switch (e) {
      case "svg":
        return "http://www.w3.org/2000/svg";
      case "math":
        return "http://www.w3.org/1998/Math/MathML";
      default:
        return "http://www.w3.org/1999/xhtml";
    }
  }
  function ss(e, t) {
    return e == null || e === "http://www.w3.org/1999/xhtml" ? hd(t) : e === "http://www.w3.org/2000/svg" && t === "foreignObject" ? "http://www.w3.org/1999/xhtml" : e;
  }
  var Hl, gd = function(e) {
    return typeof MSApp < "u" && MSApp.execUnsafeLocalFunction ? function(t, n, r, l) {
      MSApp.execUnsafeLocalFunction(function() {
        return e(t, n, r, l);
      });
    } : e;
  }(function(e, t) {
    if (e.namespaceURI !== "http://www.w3.org/2000/svg" || "innerHTML" in e) e.innerHTML = t;
    else {
      for (Hl = Hl || document.createElement("div"), Hl.innerHTML = "<svg>" + t.valueOf().toString() + "</svg>", t = Hl.firstChild; e.firstChild; ) e.removeChild(e.firstChild);
      for (; t.firstChild; ) e.appendChild(t.firstChild);
    }
  });
  function cl(e, t) {
    if (t) {
      var n = e.firstChild;
      if (n && n === e.lastChild && n.nodeType === 3) {
        n.nodeValue = t;
        return;
      }
    }
    e.textContent = t;
  }
  var qr = {
    animationIterationCount: true,
    aspectRatio: true,
    borderImageOutset: true,
    borderImageSlice: true,
    borderImageWidth: true,
    boxFlex: true,
    boxFlexGroup: true,
    boxOrdinalGroup: true,
    columnCount: true,
    columns: true,
    flex: true,
    flexGrow: true,
    flexPositive: true,
    flexShrink: true,
    flexNegative: true,
    flexOrder: true,
    gridArea: true,
    gridRow: true,
    gridRowEnd: true,
    gridRowSpan: true,
    gridRowStart: true,
    gridColumn: true,
    gridColumnEnd: true,
    gridColumnSpan: true,
    gridColumnStart: true,
    fontWeight: true,
    lineClamp: true,
    lineHeight: true,
    opacity: true,
    order: true,
    orphans: true,
    tabSize: true,
    widows: true,
    zIndex: true,
    zoom: true,
    fillOpacity: true,
    floodOpacity: true,
    stopOpacity: true,
    strokeDasharray: true,
    strokeDashoffset: true,
    strokeMiterlimit: true,
    strokeOpacity: true,
    strokeWidth: true
  }, mp = [
    "Webkit",
    "ms",
    "Moz",
    "O"
  ];
  Object.keys(qr).forEach(function(e) {
    mp.forEach(function(t) {
      t = t + e.charAt(0).toUpperCase() + e.substring(1), qr[t] = qr[e];
    });
  });
  function vd(e, t, n) {
    return t == null || typeof t == "boolean" || t === "" ? "" : n || typeof t != "number" || t === 0 || qr.hasOwnProperty(e) && qr[e] ? ("" + t).trim() : t + "px";
  }
  function xd(e, t) {
    e = e.style;
    for (var n in t) if (t.hasOwnProperty(n)) {
      var r = n.indexOf("--") === 0, l = vd(n, t[n], r);
      n === "float" && (n = "cssFloat"), r ? e.setProperty(n, l) : e[n] = l;
    }
  }
  var pp = Re({
    menuitem: true
  }, {
    area: true,
    base: true,
    br: true,
    col: true,
    embed: true,
    hr: true,
    img: true,
    input: true,
    keygen: true,
    link: true,
    meta: true,
    param: true,
    source: true,
    track: true,
    wbr: true
  });
  function is(e, t) {
    if (t) {
      if (pp[e] && (t.children != null || t.dangerouslySetInnerHTML != null)) throw Error(b(137, e));
      if (t.dangerouslySetInnerHTML != null) {
        if (t.children != null) throw Error(b(60));
        if (typeof t.dangerouslySetInnerHTML != "object" || !("__html" in t.dangerouslySetInnerHTML)) throw Error(b(61));
      }
      if (t.style != null && typeof t.style != "object") throw Error(b(62));
    }
  }
  function us(e, t) {
    if (e.indexOf("-") === -1) return typeof t.is == "string";
    switch (e) {
      case "annotation-xml":
      case "color-profile":
      case "font-face":
      case "font-face-src":
      case "font-face-uri":
      case "font-face-format":
      case "font-face-name":
      case "missing-glyph":
        return false;
      default:
        return true;
    }
  }
  var cs = null;
  function ai(e) {
    return e = e.target || e.srcElement || window, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
  }
  var ds = null, vr = null, xr = null;
  function hu(e) {
    if (e = bl(e)) {
      if (typeof ds != "function") throw Error(b(280));
      var t = e.stateNode;
      t && (t = eo(t), ds(e.stateNode, e.type, t));
    }
  }
  function yd(e) {
    vr ? xr ? xr.push(e) : xr = [
      e
    ] : vr = e;
  }
  function wd() {
    if (vr) {
      var e = vr, t = xr;
      if (xr = vr = null, hu(e), t) for (e = 0; e < t.length; e++) hu(t[e]);
    }
  }
  function Sd(e, t) {
    return e(t);
  }
  function kd() {
  }
  var No = false;
  function Nd(e, t, n) {
    if (No) return e(t, n);
    No = true;
    try {
      return Sd(e, t, n);
    } finally {
      No = false, (vr !== null || xr !== null) && (kd(), wd());
    }
  }
  function dl(e, t) {
    var n = e.stateNode;
    if (n === null) return null;
    var r = eo(n);
    if (r === null) return null;
    n = r[t];
    e: switch (t) {
      case "onClick":
      case "onClickCapture":
      case "onDoubleClick":
      case "onDoubleClickCapture":
      case "onMouseDown":
      case "onMouseDownCapture":
      case "onMouseMove":
      case "onMouseMoveCapture":
      case "onMouseUp":
      case "onMouseUpCapture":
      case "onMouseEnter":
        (r = !r.disabled) || (e = e.type, r = !(e === "button" || e === "input" || e === "select" || e === "textarea")), e = !r;
        break e;
      default:
        e = false;
    }
    if (e) return null;
    if (n && typeof n != "function") throw Error(b(231, t, typeof n));
    return n;
  }
  var fs = false;
  if (Xt) try {
    var zr = {};
    Object.defineProperty(zr, "passive", {
      get: function() {
        fs = true;
      }
    }), window.addEventListener("test", zr, zr), window.removeEventListener("test", zr, zr);
  } catch {
    fs = false;
  }
  function hp(e, t, n, r, l, a, o, i, s) {
    var c = Array.prototype.slice.call(arguments, 3);
    try {
      t.apply(n, c);
    } catch (m) {
      this.onError(m);
    }
  }
  var el = false, ka = null, Na = false, ms = null, gp = {
    onError: function(e) {
      el = true, ka = e;
    }
  };
  function vp(e, t, n, r, l, a, o, i, s) {
    el = false, ka = null, hp.apply(gp, arguments);
  }
  function xp(e, t, n, r, l, a, o, i, s) {
    if (vp.apply(this, arguments), el) {
      if (el) {
        var c = ka;
        el = false, ka = null;
      } else throw Error(b(198));
      Na || (Na = true, ms = c);
    }
  }
  function Gn(e) {
    var t = e, n = e;
    if (e.alternate) for (; t.return; ) t = t.return;
    else {
      e = t;
      do
        t = e, t.flags & 4098 && (n = t.return), e = t.return;
      while (e);
    }
    return t.tag === 3 ? n : null;
  }
  function jd(e) {
    if (e.tag === 13) {
      var t = e.memoizedState;
      if (t === null && (e = e.alternate, e !== null && (t = e.memoizedState)), t !== null) return t.dehydrated;
    }
    return null;
  }
  function gu(e) {
    if (Gn(e) !== e) throw Error(b(188));
  }
  function yp(e) {
    var t = e.alternate;
    if (!t) {
      if (t = Gn(e), t === null) throw Error(b(188));
      return t !== e ? null : e;
    }
    for (var n = e, r = t; ; ) {
      var l = n.return;
      if (l === null) break;
      var a = l.alternate;
      if (a === null) {
        if (r = l.return, r !== null) {
          n = r;
          continue;
        }
        break;
      }
      if (l.child === a.child) {
        for (a = l.child; a; ) {
          if (a === n) return gu(l), e;
          if (a === r) return gu(l), t;
          a = a.sibling;
        }
        throw Error(b(188));
      }
      if (n.return !== r.return) n = l, r = a;
      else {
        for (var o = false, i = l.child; i; ) {
          if (i === n) {
            o = true, n = l, r = a;
            break;
          }
          if (i === r) {
            o = true, r = l, n = a;
            break;
          }
          i = i.sibling;
        }
        if (!o) {
          for (i = a.child; i; ) {
            if (i === n) {
              o = true, n = a, r = l;
              break;
            }
            if (i === r) {
              o = true, r = a, n = l;
              break;
            }
            i = i.sibling;
          }
          if (!o) throw Error(b(189));
        }
      }
      if (n.alternate !== r) throw Error(b(190));
    }
    if (n.tag !== 3) throw Error(b(188));
    return n.stateNode.current === n ? e : t;
  }
  function Ed(e) {
    return e = yp(e), e !== null ? Cd(e) : null;
  }
  function Cd(e) {
    if (e.tag === 5 || e.tag === 6) return e;
    for (e = e.child; e !== null; ) {
      var t = Cd(e);
      if (t !== null) return t;
      e = e.sibling;
    }
    return null;
  }
  var _d = xt.unstable_scheduleCallback, vu = xt.unstable_cancelCallback, wp = xt.unstable_shouldYield, Sp = xt.unstable_requestPaint, Ie = xt.unstable_now, kp = xt.unstable_getCurrentPriorityLevel, oi = xt.unstable_ImmediatePriority, Pd = xt.unstable_UserBlockingPriority, ja = xt.unstable_NormalPriority, Np = xt.unstable_LowPriority, Rd = xt.unstable_IdlePriority, Ja = null, Ft = null;
  function jp(e) {
    if (Ft && typeof Ft.onCommitFiberRoot == "function") try {
      Ft.onCommitFiberRoot(Ja, e, void 0, (e.current.flags & 128) === 128);
    } catch {
    }
  }
  var Lt = Math.clz32 ? Math.clz32 : _p, Ep = Math.log, Cp = Math.LN2;
  function _p(e) {
    return e >>>= 0, e === 0 ? 32 : 31 - (Ep(e) / Cp | 0) | 0;
  }
  var Ql = 64, Kl = 4194304;
  function Jr(e) {
    switch (e & -e) {
      case 1:
        return 1;
      case 2:
        return 2;
      case 4:
        return 4;
      case 8:
        return 8;
      case 16:
        return 16;
      case 32:
        return 32;
      case 64:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return e & 4194240;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
      case 67108864:
        return e & 130023424;
      case 134217728:
        return 134217728;
      case 268435456:
        return 268435456;
      case 536870912:
        return 536870912;
      case 1073741824:
        return 1073741824;
      default:
        return e;
    }
  }
  function Ea(e, t) {
    var n = e.pendingLanes;
    if (n === 0) return 0;
    var r = 0, l = e.suspendedLanes, a = e.pingedLanes, o = n & 268435455;
    if (o !== 0) {
      var i = o & ~l;
      i !== 0 ? r = Jr(i) : (a &= o, a !== 0 && (r = Jr(a)));
    } else o = n & ~l, o !== 0 ? r = Jr(o) : a !== 0 && (r = Jr(a));
    if (r === 0) return 0;
    if (t !== 0 && t !== r && !(t & l) && (l = r & -r, a = t & -t, l >= a || l === 16 && (a & 4194240) !== 0)) return t;
    if (r & 4 && (r |= n & 16), t = e.entangledLanes, t !== 0) for (e = e.entanglements, t &= r; 0 < t; ) n = 31 - Lt(t), l = 1 << n, r |= e[n], t &= ~l;
    return r;
  }
  function Pp(e, t) {
    switch (e) {
      case 1:
      case 2:
      case 4:
        return t + 250;
      case 8:
      case 16:
      case 32:
      case 64:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return t + 5e3;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
      case 67108864:
        return -1;
      case 134217728:
      case 268435456:
      case 536870912:
      case 1073741824:
        return -1;
      default:
        return -1;
    }
  }
  function Rp(e, t) {
    for (var n = e.suspendedLanes, r = e.pingedLanes, l = e.expirationTimes, a = e.pendingLanes; 0 < a; ) {
      var o = 31 - Lt(a), i = 1 << o, s = l[o];
      s === -1 ? (!(i & n) || i & r) && (l[o] = Pp(i, t)) : s <= t && (e.expiredLanes |= i), a &= ~i;
    }
  }
  function ps(e) {
    return e = e.pendingLanes & -1073741825, e !== 0 ? e : e & 1073741824 ? 1073741824 : 0;
  }
  function Td() {
    var e = Ql;
    return Ql <<= 1, !(Ql & 4194240) && (Ql = 64), e;
  }
  function jo(e) {
    for (var t = [], n = 0; 31 > n; n++) t.push(e);
    return t;
  }
  function Rl(e, t, n) {
    e.pendingLanes |= t, t !== 536870912 && (e.suspendedLanes = 0, e.pingedLanes = 0), e = e.eventTimes, t = 31 - Lt(t), e[t] = n;
  }
  function Tp(e, t) {
    var n = e.pendingLanes & ~t;
    e.pendingLanes = t, e.suspendedLanes = 0, e.pingedLanes = 0, e.expiredLanes &= t, e.mutableReadLanes &= t, e.entangledLanes &= t, t = e.entanglements;
    var r = e.eventTimes;
    for (e = e.expirationTimes; 0 < n; ) {
      var l = 31 - Lt(n), a = 1 << l;
      t[l] = 0, r[l] = -1, e[l] = -1, n &= ~a;
    }
  }
  function si(e, t) {
    var n = e.entangledLanes |= t;
    for (e = e.entanglements; n; ) {
      var r = 31 - Lt(n), l = 1 << r;
      l & t | e[r] & t && (e[r] |= t), n &= ~l;
    }
  }
  var he = 0;
  function bd(e) {
    return e &= -e, 1 < e ? 4 < e ? e & 268435455 ? 16 : 536870912 : 4 : 1;
  }
  var Md, ii, Dd, Ld, Od, hs = false, Gl = [], mn = null, pn = null, hn = null, fl = /* @__PURE__ */ new Map(), ml = /* @__PURE__ */ new Map(), sn = [], bp = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
  function xu(e, t) {
    switch (e) {
      case "focusin":
      case "focusout":
        mn = null;
        break;
      case "dragenter":
      case "dragleave":
        pn = null;
        break;
      case "mouseover":
      case "mouseout":
        hn = null;
        break;
      case "pointerover":
      case "pointerout":
        fl.delete(t.pointerId);
        break;
      case "gotpointercapture":
      case "lostpointercapture":
        ml.delete(t.pointerId);
    }
  }
  function Ar(e, t, n, r, l, a) {
    return e === null || e.nativeEvent !== a ? (e = {
      blockedOn: t,
      domEventName: n,
      eventSystemFlags: r,
      nativeEvent: a,
      targetContainers: [
        l
      ]
    }, t !== null && (t = bl(t), t !== null && ii(t)), e) : (e.eventSystemFlags |= r, t = e.targetContainers, l !== null && t.indexOf(l) === -1 && t.push(l), e);
  }
  function Mp(e, t, n, r, l) {
    switch (t) {
      case "focusin":
        return mn = Ar(mn, e, t, n, r, l), true;
      case "dragenter":
        return pn = Ar(pn, e, t, n, r, l), true;
      case "mouseover":
        return hn = Ar(hn, e, t, n, r, l), true;
      case "pointerover":
        var a = l.pointerId;
        return fl.set(a, Ar(fl.get(a) || null, e, t, n, r, l)), true;
      case "gotpointercapture":
        return a = l.pointerId, ml.set(a, Ar(ml.get(a) || null, e, t, n, r, l)), true;
    }
    return false;
  }
  function Id(e) {
    var t = On(e.target);
    if (t !== null) {
      var n = Gn(t);
      if (n !== null) {
        if (t = n.tag, t === 13) {
          if (t = jd(n), t !== null) {
            e.blockedOn = t, Od(e.priority, function() {
              Dd(n);
            });
            return;
          }
        } else if (t === 3 && n.stateNode.current.memoizedState.isDehydrated) {
          e.blockedOn = n.tag === 3 ? n.stateNode.containerInfo : null;
          return;
        }
      }
    }
    e.blockedOn = null;
  }
  function ua(e) {
    if (e.blockedOn !== null) return false;
    for (var t = e.targetContainers; 0 < t.length; ) {
      var n = gs(e.domEventName, e.eventSystemFlags, t[0], e.nativeEvent);
      if (n === null) {
        n = e.nativeEvent;
        var r = new n.constructor(n.type, n);
        cs = r, n.target.dispatchEvent(r), cs = null;
      } else return t = bl(n), t !== null && ii(t), e.blockedOn = n, false;
      t.shift();
    }
    return true;
  }
  function yu(e, t, n) {
    ua(e) && n.delete(t);
  }
  function Dp() {
    hs = false, mn !== null && ua(mn) && (mn = null), pn !== null && ua(pn) && (pn = null), hn !== null && ua(hn) && (hn = null), fl.forEach(yu), ml.forEach(yu);
  }
  function Ur(e, t) {
    e.blockedOn === t && (e.blockedOn = null, hs || (hs = true, xt.unstable_scheduleCallback(xt.unstable_NormalPriority, Dp)));
  }
  function pl(e) {
    function t(l) {
      return Ur(l, e);
    }
    if (0 < Gl.length) {
      Ur(Gl[0], e);
      for (var n = 1; n < Gl.length; n++) {
        var r = Gl[n];
        r.blockedOn === e && (r.blockedOn = null);
      }
    }
    for (mn !== null && Ur(mn, e), pn !== null && Ur(pn, e), hn !== null && Ur(hn, e), fl.forEach(t), ml.forEach(t), n = 0; n < sn.length; n++) r = sn[n], r.blockedOn === e && (r.blockedOn = null);
    for (; 0 < sn.length && (n = sn[0], n.blockedOn === null); ) Id(n), n.blockedOn === null && sn.shift();
  }
  var yr = tn.ReactCurrentBatchConfig, Ca = true;
  function Lp(e, t, n, r) {
    var l = he, a = yr.transition;
    yr.transition = null;
    try {
      he = 1, ui(e, t, n, r);
    } finally {
      he = l, yr.transition = a;
    }
  }
  function Op(e, t, n, r) {
    var l = he, a = yr.transition;
    yr.transition = null;
    try {
      he = 4, ui(e, t, n, r);
    } finally {
      he = l, yr.transition = a;
    }
  }
  function ui(e, t, n, r) {
    if (Ca) {
      var l = gs(e, t, n, r);
      if (l === null) Lo(e, t, r, _a, n), xu(e, r);
      else if (Mp(l, e, t, n, r)) r.stopPropagation();
      else if (xu(e, r), t & 4 && -1 < bp.indexOf(e)) {
        for (; l !== null; ) {
          var a = bl(l);
          if (a !== null && Md(a), a = gs(e, t, n, r), a === null && Lo(e, t, r, _a, n), a === l) break;
          l = a;
        }
        l !== null && r.stopPropagation();
      } else Lo(e, t, r, null, n);
    }
  }
  var _a = null;
  function gs(e, t, n, r) {
    if (_a = null, e = ai(r), e = On(e), e !== null) if (t = Gn(e), t === null) e = null;
    else if (n = t.tag, n === 13) {
      if (e = jd(t), e !== null) return e;
      e = null;
    } else if (n === 3) {
      if (t.stateNode.current.memoizedState.isDehydrated) return t.tag === 3 ? t.stateNode.containerInfo : null;
      e = null;
    } else t !== e && (e = null);
    return _a = e, null;
  }
  function zd(e) {
    switch (e) {
      case "cancel":
      case "click":
      case "close":
      case "contextmenu":
      case "copy":
      case "cut":
      case "auxclick":
      case "dblclick":
      case "dragend":
      case "dragstart":
      case "drop":
      case "focusin":
      case "focusout":
      case "input":
      case "invalid":
      case "keydown":
      case "keypress":
      case "keyup":
      case "mousedown":
      case "mouseup":
      case "paste":
      case "pause":
      case "play":
      case "pointercancel":
      case "pointerdown":
      case "pointerup":
      case "ratechange":
      case "reset":
      case "resize":
      case "seeked":
      case "submit":
      case "touchcancel":
      case "touchend":
      case "touchstart":
      case "volumechange":
      case "change":
      case "selectionchange":
      case "textInput":
      case "compositionstart":
      case "compositionend":
      case "compositionupdate":
      case "beforeblur":
      case "afterblur":
      case "beforeinput":
      case "blur":
      case "fullscreenchange":
      case "focus":
      case "hashchange":
      case "popstate":
      case "select":
      case "selectstart":
        return 1;
      case "drag":
      case "dragenter":
      case "dragexit":
      case "dragleave":
      case "dragover":
      case "mousemove":
      case "mouseout":
      case "mouseover":
      case "pointermove":
      case "pointerout":
      case "pointerover":
      case "scroll":
      case "toggle":
      case "touchmove":
      case "wheel":
      case "mouseenter":
      case "mouseleave":
      case "pointerenter":
      case "pointerleave":
        return 4;
      case "message":
        switch (kp()) {
          case oi:
            return 1;
          case Pd:
            return 4;
          case ja:
          case Np:
            return 16;
          case Rd:
            return 536870912;
          default:
            return 16;
        }
      default:
        return 16;
    }
  }
  var cn = null, ci = null, ca = null;
  function Ad() {
    if (ca) return ca;
    var e, t = ci, n = t.length, r, l = "value" in cn ? cn.value : cn.textContent, a = l.length;
    for (e = 0; e < n && t[e] === l[e]; e++) ;
    var o = n - e;
    for (r = 1; r <= o && t[n - r] === l[a - r]; r++) ;
    return ca = l.slice(e, 1 < r ? 1 - r : void 0);
  }
  function da(e) {
    var t = e.keyCode;
    return "charCode" in e ? (e = e.charCode, e === 0 && t === 13 && (e = 13)) : e = t, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
  }
  function Yl() {
    return true;
  }
  function wu() {
    return false;
  }
  function wt(e) {
    function t(n, r, l, a, o) {
      this._reactName = n, this._targetInst = l, this.type = r, this.nativeEvent = a, this.target = o, this.currentTarget = null;
      for (var i in e) e.hasOwnProperty(i) && (n = e[i], this[i] = n ? n(a) : a[i]);
      return this.isDefaultPrevented = (a.defaultPrevented != null ? a.defaultPrevented : a.returnValue === false) ? Yl : wu, this.isPropagationStopped = wu, this;
    }
    return Re(t.prototype, {
      preventDefault: function() {
        this.defaultPrevented = true;
        var n = this.nativeEvent;
        n && (n.preventDefault ? n.preventDefault() : typeof n.returnValue != "unknown" && (n.returnValue = false), this.isDefaultPrevented = Yl);
      },
      stopPropagation: function() {
        var n = this.nativeEvent;
        n && (n.stopPropagation ? n.stopPropagation() : typeof n.cancelBubble != "unknown" && (n.cancelBubble = true), this.isPropagationStopped = Yl);
      },
      persist: function() {
      },
      isPersistent: Yl
    }), t;
  }
  var Tr = {
    eventPhase: 0,
    bubbles: 0,
    cancelable: 0,
    timeStamp: function(e) {
      return e.timeStamp || Date.now();
    },
    defaultPrevented: 0,
    isTrusted: 0
  }, di = wt(Tr), Tl = Re({}, Tr, {
    view: 0,
    detail: 0
  }), Ip = wt(Tl), Eo, Co, $r, Xa = Re({}, Tl, {
    screenX: 0,
    screenY: 0,
    clientX: 0,
    clientY: 0,
    pageX: 0,
    pageY: 0,
    ctrlKey: 0,
    shiftKey: 0,
    altKey: 0,
    metaKey: 0,
    getModifierState: fi,
    button: 0,
    buttons: 0,
    relatedTarget: function(e) {
      return e.relatedTarget === void 0 ? e.fromElement === e.srcElement ? e.toElement : e.fromElement : e.relatedTarget;
    },
    movementX: function(e) {
      return "movementX" in e ? e.movementX : (e !== $r && ($r && e.type === "mousemove" ? (Eo = e.screenX - $r.screenX, Co = e.screenY - $r.screenY) : Co = Eo = 0, $r = e), Eo);
    },
    movementY: function(e) {
      return "movementY" in e ? e.movementY : Co;
    }
  }), Su = wt(Xa), zp = Re({}, Xa, {
    dataTransfer: 0
  }), Ap = wt(zp), Up = Re({}, Tl, {
    relatedTarget: 0
  }), _o = wt(Up), $p = Re({}, Tr, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Fp = wt($p), Bp = Re({}, Tr, {
    clipboardData: function(e) {
      return "clipboardData" in e ? e.clipboardData : window.clipboardData;
    }
  }), Wp = wt(Bp), Vp = Re({}, Tr, {
    data: 0
  }), ku = wt(Vp), Hp = {
    Esc: "Escape",
    Spacebar: " ",
    Left: "ArrowLeft",
    Up: "ArrowUp",
    Right: "ArrowRight",
    Down: "ArrowDown",
    Del: "Delete",
    Win: "OS",
    Menu: "ContextMenu",
    Apps: "ContextMenu",
    Scroll: "ScrollLock",
    MozPrintableKey: "Unidentified"
  }, Qp = {
    8: "Backspace",
    9: "Tab",
    12: "Clear",
    13: "Enter",
    16: "Shift",
    17: "Control",
    18: "Alt",
    19: "Pause",
    20: "CapsLock",
    27: "Escape",
    32: " ",
    33: "PageUp",
    34: "PageDown",
    35: "End",
    36: "Home",
    37: "ArrowLeft",
    38: "ArrowUp",
    39: "ArrowRight",
    40: "ArrowDown",
    45: "Insert",
    46: "Delete",
    112: "F1",
    113: "F2",
    114: "F3",
    115: "F4",
    116: "F5",
    117: "F6",
    118: "F7",
    119: "F8",
    120: "F9",
    121: "F10",
    122: "F11",
    123: "F12",
    144: "NumLock",
    145: "ScrollLock",
    224: "Meta"
  }, Kp = {
    Alt: "altKey",
    Control: "ctrlKey",
    Meta: "metaKey",
    Shift: "shiftKey"
  };
  function Gp(e) {
    var t = this.nativeEvent;
    return t.getModifierState ? t.getModifierState(e) : (e = Kp[e]) ? !!t[e] : false;
  }
  function fi() {
    return Gp;
  }
  var Yp = Re({}, Tl, {
    key: function(e) {
      if (e.key) {
        var t = Hp[e.key] || e.key;
        if (t !== "Unidentified") return t;
      }
      return e.type === "keypress" ? (e = da(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? Qp[e.keyCode] || "Unidentified" : "";
    },
    code: 0,
    location: 0,
    ctrlKey: 0,
    shiftKey: 0,
    altKey: 0,
    metaKey: 0,
    repeat: 0,
    locale: 0,
    getModifierState: fi,
    charCode: function(e) {
      return e.type === "keypress" ? da(e) : 0;
    },
    keyCode: function(e) {
      return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
    },
    which: function(e) {
      return e.type === "keypress" ? da(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
    }
  }), Jp = wt(Yp), Xp = Re({}, Xa, {
    pointerId: 0,
    width: 0,
    height: 0,
    pressure: 0,
    tangentialPressure: 0,
    tiltX: 0,
    tiltY: 0,
    twist: 0,
    pointerType: 0,
    isPrimary: 0
  }), Nu = wt(Xp), Zp = Re({}, Tl, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: fi
  }), qp = wt(Zp), eh = Re({}, Tr, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), th = wt(eh), nh = Re({}, Xa, {
    deltaX: function(e) {
      return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
    },
    deltaY: function(e) {
      return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), rh = wt(nh), lh = [
    9,
    13,
    27,
    32
  ], mi = Xt && "CompositionEvent" in window, tl = null;
  Xt && "documentMode" in document && (tl = document.documentMode);
  var ah = Xt && "TextEvent" in window && !tl, Ud = Xt && (!mi || tl && 8 < tl && 11 >= tl), ju = " ", Eu = false;
  function $d(e, t) {
    switch (e) {
      case "keyup":
        return lh.indexOf(t.keyCode) !== -1;
      case "keydown":
        return t.keyCode !== 229;
      case "keypress":
      case "mousedown":
      case "focusout":
        return true;
      default:
        return false;
    }
  }
  function Fd(e) {
    return e = e.detail, typeof e == "object" && "data" in e ? e.data : null;
  }
  var or = false;
  function oh(e, t) {
    switch (e) {
      case "compositionend":
        return Fd(t);
      case "keypress":
        return t.which !== 32 ? null : (Eu = true, ju);
      case "textInput":
        return e = t.data, e === ju && Eu ? null : e;
      default:
        return null;
    }
  }
  function sh(e, t) {
    if (or) return e === "compositionend" || !mi && $d(e, t) ? (e = Ad(), ca = ci = cn = null, or = false, e) : null;
    switch (e) {
      case "paste":
        return null;
      case "keypress":
        if (!(t.ctrlKey || t.altKey || t.metaKey) || t.ctrlKey && t.altKey) {
          if (t.char && 1 < t.char.length) return t.char;
          if (t.which) return String.fromCharCode(t.which);
        }
        return null;
      case "compositionend":
        return Ud && t.locale !== "ko" ? null : t.data;
      default:
        return null;
    }
  }
  var ih = {
    color: true,
    date: true,
    datetime: true,
    "datetime-local": true,
    email: true,
    month: true,
    number: true,
    password: true,
    range: true,
    search: true,
    tel: true,
    text: true,
    time: true,
    url: true,
    week: true
  };
  function Cu(e) {
    var t = e && e.nodeName && e.nodeName.toLowerCase();
    return t === "input" ? !!ih[e.type] : t === "textarea";
  }
  function Bd(e, t, n, r) {
    yd(r), t = Pa(t, "onChange"), 0 < t.length && (n = new di("onChange", "change", null, n, r), e.push({
      event: n,
      listeners: t
    }));
  }
  var nl = null, hl = null;
  function uh(e) {
    qd(e, 0);
  }
  function Za(e) {
    var t = ur(e);
    if (fd(t)) return e;
  }
  function ch(e, t) {
    if (e === "change") return t;
  }
  var Wd = false;
  if (Xt) {
    var Po;
    if (Xt) {
      var Ro = "oninput" in document;
      if (!Ro) {
        var _u = document.createElement("div");
        _u.setAttribute("oninput", "return;"), Ro = typeof _u.oninput == "function";
      }
      Po = Ro;
    } else Po = false;
    Wd = Po && (!document.documentMode || 9 < document.documentMode);
  }
  function Pu() {
    nl && (nl.detachEvent("onpropertychange", Vd), hl = nl = null);
  }
  function Vd(e) {
    if (e.propertyName === "value" && Za(hl)) {
      var t = [];
      Bd(t, hl, e, ai(e)), Nd(uh, t);
    }
  }
  function dh(e, t, n) {
    e === "focusin" ? (Pu(), nl = t, hl = n, nl.attachEvent("onpropertychange", Vd)) : e === "focusout" && Pu();
  }
  function fh(e) {
    if (e === "selectionchange" || e === "keyup" || e === "keydown") return Za(hl);
  }
  function mh(e, t) {
    if (e === "click") return Za(t);
  }
  function ph(e, t) {
    if (e === "input" || e === "change") return Za(t);
  }
  function hh(e, t) {
    return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
  }
  var It = typeof Object.is == "function" ? Object.is : hh;
  function gl(e, t) {
    if (It(e, t)) return true;
    if (typeof e != "object" || e === null || typeof t != "object" || t === null) return false;
    var n = Object.keys(e), r = Object.keys(t);
    if (n.length !== r.length) return false;
    for (r = 0; r < n.length; r++) {
      var l = n[r];
      if (!Zo.call(t, l) || !It(e[l], t[l])) return false;
    }
    return true;
  }
  function Ru(e) {
    for (; e && e.firstChild; ) e = e.firstChild;
    return e;
  }
  function Tu(e, t) {
    var n = Ru(e);
    e = 0;
    for (var r; n; ) {
      if (n.nodeType === 3) {
        if (r = e + n.textContent.length, e <= t && r >= t) return {
          node: n,
          offset: t - e
        };
        e = r;
      }
      e: {
        for (; n; ) {
          if (n.nextSibling) {
            n = n.nextSibling;
            break e;
          }
          n = n.parentNode;
        }
        n = void 0;
      }
      n = Ru(n);
    }
  }
  function Hd(e, t) {
    return e && t ? e === t ? true : e && e.nodeType === 3 ? false : t && t.nodeType === 3 ? Hd(e, t.parentNode) : "contains" in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : false : false;
  }
  function Qd() {
    for (var e = window, t = Sa(); t instanceof e.HTMLIFrameElement; ) {
      try {
        var n = typeof t.contentWindow.location.href == "string";
      } catch {
        n = false;
      }
      if (n) e = t.contentWindow;
      else break;
      t = Sa(e.document);
    }
    return t;
  }
  function pi(e) {
    var t = e && e.nodeName && e.nodeName.toLowerCase();
    return t && (t === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || t === "textarea" || e.contentEditable === "true");
  }
  function gh(e) {
    var t = Qd(), n = e.focusedElem, r = e.selectionRange;
    if (t !== n && n && n.ownerDocument && Hd(n.ownerDocument.documentElement, n)) {
      if (r !== null && pi(n)) {
        if (t = r.start, e = r.end, e === void 0 && (e = t), "selectionStart" in n) n.selectionStart = t, n.selectionEnd = Math.min(e, n.value.length);
        else if (e = (t = n.ownerDocument || document) && t.defaultView || window, e.getSelection) {
          e = e.getSelection();
          var l = n.textContent.length, a = Math.min(r.start, l);
          r = r.end === void 0 ? a : Math.min(r.end, l), !e.extend && a > r && (l = r, r = a, a = l), l = Tu(n, a);
          var o = Tu(n, r);
          l && o && (e.rangeCount !== 1 || e.anchorNode !== l.node || e.anchorOffset !== l.offset || e.focusNode !== o.node || e.focusOffset !== o.offset) && (t = t.createRange(), t.setStart(l.node, l.offset), e.removeAllRanges(), a > r ? (e.addRange(t), e.extend(o.node, o.offset)) : (t.setEnd(o.node, o.offset), e.addRange(t)));
        }
      }
      for (t = [], e = n; e = e.parentNode; ) e.nodeType === 1 && t.push({
        element: e,
        left: e.scrollLeft,
        top: e.scrollTop
      });
      for (typeof n.focus == "function" && n.focus(), n = 0; n < t.length; n++) e = t[n], e.element.scrollLeft = e.left, e.element.scrollTop = e.top;
    }
  }
  var vh = Xt && "documentMode" in document && 11 >= document.documentMode, sr = null, vs = null, rl = null, xs = false;
  function bu(e, t, n) {
    var r = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
    xs || sr == null || sr !== Sa(r) || (r = sr, "selectionStart" in r && pi(r) ? r = {
      start: r.selectionStart,
      end: r.selectionEnd
    } : (r = (r.ownerDocument && r.ownerDocument.defaultView || window).getSelection(), r = {
      anchorNode: r.anchorNode,
      anchorOffset: r.anchorOffset,
      focusNode: r.focusNode,
      focusOffset: r.focusOffset
    }), rl && gl(rl, r) || (rl = r, r = Pa(vs, "onSelect"), 0 < r.length && (t = new di("onSelect", "select", null, t, n), e.push({
      event: t,
      listeners: r
    }), t.target = sr)));
  }
  function Jl(e, t) {
    var n = {};
    return n[e.toLowerCase()] = t.toLowerCase(), n["Webkit" + e] = "webkit" + t, n["Moz" + e] = "moz" + t, n;
  }
  var ir = {
    animationend: Jl("Animation", "AnimationEnd"),
    animationiteration: Jl("Animation", "AnimationIteration"),
    animationstart: Jl("Animation", "AnimationStart"),
    transitionend: Jl("Transition", "TransitionEnd")
  }, To = {}, Kd = {};
  Xt && (Kd = document.createElement("div").style, "AnimationEvent" in window || (delete ir.animationend.animation, delete ir.animationiteration.animation, delete ir.animationstart.animation), "TransitionEvent" in window || delete ir.transitionend.transition);
  function qa(e) {
    if (To[e]) return To[e];
    if (!ir[e]) return e;
    var t = ir[e], n;
    for (n in t) if (t.hasOwnProperty(n) && n in Kd) return To[e] = t[n];
    return e;
  }
  var Gd = qa("animationend"), Yd = qa("animationiteration"), Jd = qa("animationstart"), Xd = qa("transitionend"), Zd = /* @__PURE__ */ new Map(), Mu = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
  function jn(e, t) {
    Zd.set(e, t), Kn(t, [
      e
    ]);
  }
  for (var bo = 0; bo < Mu.length; bo++) {
    var Mo = Mu[bo], xh = Mo.toLowerCase(), yh = Mo[0].toUpperCase() + Mo.slice(1);
    jn(xh, "on" + yh);
  }
  jn(Gd, "onAnimationEnd");
  jn(Yd, "onAnimationIteration");
  jn(Jd, "onAnimationStart");
  jn("dblclick", "onDoubleClick");
  jn("focusin", "onFocus");
  jn("focusout", "onBlur");
  jn(Xd, "onTransitionEnd");
  kr("onMouseEnter", [
    "mouseout",
    "mouseover"
  ]);
  kr("onMouseLeave", [
    "mouseout",
    "mouseover"
  ]);
  kr("onPointerEnter", [
    "pointerout",
    "pointerover"
  ]);
  kr("onPointerLeave", [
    "pointerout",
    "pointerover"
  ]);
  Kn("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
  Kn("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
  Kn("onBeforeInput", [
    "compositionend",
    "keypress",
    "textInput",
    "paste"
  ]);
  Kn("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
  Kn("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
  Kn("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
  var Xr = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), wh = new Set("cancel close invalid load scroll toggle".split(" ").concat(Xr));
  function Du(e, t, n) {
    var r = e.type || "unknown-event";
    e.currentTarget = n, xp(r, t, void 0, e), e.currentTarget = null;
  }
  function qd(e, t) {
    t = (t & 4) !== 0;
    for (var n = 0; n < e.length; n++) {
      var r = e[n], l = r.event;
      r = r.listeners;
      e: {
        var a = void 0;
        if (t) for (var o = r.length - 1; 0 <= o; o--) {
          var i = r[o], s = i.instance, c = i.currentTarget;
          if (i = i.listener, s !== a && l.isPropagationStopped()) break e;
          Du(l, i, c), a = s;
        }
        else for (o = 0; o < r.length; o++) {
          if (i = r[o], s = i.instance, c = i.currentTarget, i = i.listener, s !== a && l.isPropagationStopped()) break e;
          Du(l, i, c), a = s;
        }
      }
    }
    if (Na) throw e = ms, Na = false, ms = null, e;
  }
  function ye(e, t) {
    var n = t[Ns];
    n === void 0 && (n = t[Ns] = /* @__PURE__ */ new Set());
    var r = e + "__bubble";
    n.has(r) || (ef(t, e, 2, false), n.add(r));
  }
  function Do(e, t, n) {
    var r = 0;
    t && (r |= 4), ef(n, e, r, t);
  }
  var Xl = "_reactListening" + Math.random().toString(36).slice(2);
  function vl(e) {
    if (!e[Xl]) {
      e[Xl] = true, sd.forEach(function(n) {
        n !== "selectionchange" && (wh.has(n) || Do(n, false, e), Do(n, true, e));
      });
      var t = e.nodeType === 9 ? e : e.ownerDocument;
      t === null || t[Xl] || (t[Xl] = true, Do("selectionchange", false, t));
    }
  }
  function ef(e, t, n, r) {
    switch (zd(t)) {
      case 1:
        var l = Lp;
        break;
      case 4:
        l = Op;
        break;
      default:
        l = ui;
    }
    n = l.bind(null, t, n, e), l = void 0, !fs || t !== "touchstart" && t !== "touchmove" && t !== "wheel" || (l = true), r ? l !== void 0 ? e.addEventListener(t, n, {
      capture: true,
      passive: l
    }) : e.addEventListener(t, n, true) : l !== void 0 ? e.addEventListener(t, n, {
      passive: l
    }) : e.addEventListener(t, n, false);
  }
  function Lo(e, t, n, r, l) {
    var a = r;
    if (!(t & 1) && !(t & 2) && r !== null) e: for (; ; ) {
      if (r === null) return;
      var o = r.tag;
      if (o === 3 || o === 4) {
        var i = r.stateNode.containerInfo;
        if (i === l || i.nodeType === 8 && i.parentNode === l) break;
        if (o === 4) for (o = r.return; o !== null; ) {
          var s = o.tag;
          if ((s === 3 || s === 4) && (s = o.stateNode.containerInfo, s === l || s.nodeType === 8 && s.parentNode === l)) return;
          o = o.return;
        }
        for (; i !== null; ) {
          if (o = On(i), o === null) return;
          if (s = o.tag, s === 5 || s === 6) {
            r = a = o;
            continue e;
          }
          i = i.parentNode;
        }
      }
      r = r.return;
    }
    Nd(function() {
      var c = a, m = ai(n), d = [];
      e: {
        var g = Zd.get(e);
        if (g !== void 0) {
          var x = di, w = e;
          switch (e) {
            case "keypress":
              if (da(n) === 0) break e;
            case "keydown":
            case "keyup":
              x = Jp;
              break;
            case "focusin":
              w = "focus", x = _o;
              break;
            case "focusout":
              w = "blur", x = _o;
              break;
            case "beforeblur":
            case "afterblur":
              x = _o;
              break;
            case "click":
              if (n.button === 2) break e;
            case "auxclick":
            case "dblclick":
            case "mousedown":
            case "mousemove":
            case "mouseup":
            case "mouseout":
            case "mouseover":
            case "contextmenu":
              x = Su;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              x = Ap;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              x = qp;
              break;
            case Gd:
            case Yd:
            case Jd:
              x = Fp;
              break;
            case Xd:
              x = th;
              break;
            case "scroll":
              x = Ip;
              break;
            case "wheel":
              x = rh;
              break;
            case "copy":
            case "cut":
            case "paste":
              x = Wp;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              x = Nu;
          }
          var S = (t & 4) !== 0, _ = !S && e === "scroll", p = S ? g !== null ? g + "Capture" : null : g;
          S = [];
          for (var f = c, h; f !== null; ) {
            h = f;
            var j = h.stateNode;
            if (h.tag === 5 && j !== null && (h = j, p !== null && (j = dl(f, p), j != null && S.push(xl(f, j, h)))), _) break;
            f = f.return;
          }
          0 < S.length && (g = new x(g, w, null, n, m), d.push({
            event: g,
            listeners: S
          }));
        }
      }
      if (!(t & 7)) {
        e: {
          if (g = e === "mouseover" || e === "pointerover", x = e === "mouseout" || e === "pointerout", g && n !== cs && (w = n.relatedTarget || n.fromElement) && (On(w) || w[Zt])) break e;
          if ((x || g) && (g = m.window === m ? m : (g = m.ownerDocument) ? g.defaultView || g.parentWindow : window, x ? (w = n.relatedTarget || n.toElement, x = c, w = w ? On(w) : null, w !== null && (_ = Gn(w), w !== _ || w.tag !== 5 && w.tag !== 6) && (w = null)) : (x = null, w = c), x !== w)) {
            if (S = Su, j = "onMouseLeave", p = "onMouseEnter", f = "mouse", (e === "pointerout" || e === "pointerover") && (S = Nu, j = "onPointerLeave", p = "onPointerEnter", f = "pointer"), _ = x == null ? g : ur(x), h = w == null ? g : ur(w), g = new S(j, f + "leave", x, n, m), g.target = _, g.relatedTarget = h, j = null, On(m) === c && (S = new S(p, f + "enter", w, n, m), S.target = h, S.relatedTarget = _, j = S), _ = j, x && w) t: {
              for (S = x, p = w, f = 0, h = S; h; h = er(h)) f++;
              for (h = 0, j = p; j; j = er(j)) h++;
              for (; 0 < f - h; ) S = er(S), f--;
              for (; 0 < h - f; ) p = er(p), h--;
              for (; f--; ) {
                if (S === p || p !== null && S === p.alternate) break t;
                S = er(S), p = er(p);
              }
              S = null;
            }
            else S = null;
            x !== null && Lu(d, g, x, S, false), w !== null && _ !== null && Lu(d, _, w, S, true);
          }
        }
        e: {
          if (g = c ? ur(c) : window, x = g.nodeName && g.nodeName.toLowerCase(), x === "select" || x === "input" && g.type === "file") var C = ch;
          else if (Cu(g)) if (Wd) C = ph;
          else {
            C = fh;
            var R = dh;
          }
          else (x = g.nodeName) && x.toLowerCase() === "input" && (g.type === "checkbox" || g.type === "radio") && (C = mh);
          if (C && (C = C(e, c))) {
            Bd(d, C, n, m);
            break e;
          }
          R && R(e, g, c), e === "focusout" && (R = g._wrapperState) && R.controlled && g.type === "number" && as(g, "number", g.value);
        }
        switch (R = c ? ur(c) : window, e) {
          case "focusin":
            (Cu(R) || R.contentEditable === "true") && (sr = R, vs = c, rl = null);
            break;
          case "focusout":
            rl = vs = sr = null;
            break;
          case "mousedown":
            xs = true;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            xs = false, bu(d, n, m);
            break;
          case "selectionchange":
            if (vh) break;
          case "keydown":
          case "keyup":
            bu(d, n, m);
        }
        var k;
        if (mi) e: {
          switch (e) {
            case "compositionstart":
              var P = "onCompositionStart";
              break e;
            case "compositionend":
              P = "onCompositionEnd";
              break e;
            case "compositionupdate":
              P = "onCompositionUpdate";
              break e;
          }
          P = void 0;
        }
        else or ? $d(e, n) && (P = "onCompositionEnd") : e === "keydown" && n.keyCode === 229 && (P = "onCompositionStart");
        P && (Ud && n.locale !== "ko" && (or || P !== "onCompositionStart" ? P === "onCompositionEnd" && or && (k = Ad()) : (cn = m, ci = "value" in cn ? cn.value : cn.textContent, or = true)), R = Pa(c, P), 0 < R.length && (P = new ku(P, e, null, n, m), d.push({
          event: P,
          listeners: R
        }), k ? P.data = k : (k = Fd(n), k !== null && (P.data = k)))), (k = ah ? oh(e, n) : sh(e, n)) && (c = Pa(c, "onBeforeInput"), 0 < c.length && (m = new ku("onBeforeInput", "beforeinput", null, n, m), d.push({
          event: m,
          listeners: c
        }), m.data = k));
      }
      qd(d, t);
    });
  }
  function xl(e, t, n) {
    return {
      instance: e,
      listener: t,
      currentTarget: n
    };
  }
  function Pa(e, t) {
    for (var n = t + "Capture", r = []; e !== null; ) {
      var l = e, a = l.stateNode;
      l.tag === 5 && a !== null && (l = a, a = dl(e, n), a != null && r.unshift(xl(e, a, l)), a = dl(e, t), a != null && r.push(xl(e, a, l))), e = e.return;
    }
    return r;
  }
  function er(e) {
    if (e === null) return null;
    do
      e = e.return;
    while (e && e.tag !== 5);
    return e || null;
  }
  function Lu(e, t, n, r, l) {
    for (var a = t._reactName, o = []; n !== null && n !== r; ) {
      var i = n, s = i.alternate, c = i.stateNode;
      if (s !== null && s === r) break;
      i.tag === 5 && c !== null && (i = c, l ? (s = dl(n, a), s != null && o.unshift(xl(n, s, i))) : l || (s = dl(n, a), s != null && o.push(xl(n, s, i)))), n = n.return;
    }
    o.length !== 0 && e.push({
      event: t,
      listeners: o
    });
  }
  var Sh = /\r\n?/g, kh = /\u0000|\uFFFD/g;
  function Ou(e) {
    return (typeof e == "string" ? e : "" + e).replace(Sh, `
`).replace(kh, "");
  }
  function Zl(e, t, n) {
    if (t = Ou(t), Ou(e) !== t && n) throw Error(b(425));
  }
  function Ra() {
  }
  var ys = null, ws = null;
  function Ss(e, t) {
    return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
  }
  var ks = typeof setTimeout == "function" ? setTimeout : void 0, Nh = typeof clearTimeout == "function" ? clearTimeout : void 0, Iu = typeof Promise == "function" ? Promise : void 0, jh = typeof queueMicrotask == "function" ? queueMicrotask : typeof Iu < "u" ? function(e) {
    return Iu.resolve(null).then(e).catch(Eh);
  } : ks;
  function Eh(e) {
    setTimeout(function() {
      throw e;
    });
  }
  function Oo(e, t) {
    var n = t, r = 0;
    do {
      var l = n.nextSibling;
      if (e.removeChild(n), l && l.nodeType === 8) if (n = l.data, n === "/$") {
        if (r === 0) {
          e.removeChild(l), pl(t);
          return;
        }
        r--;
      } else n !== "$" && n !== "$?" && n !== "$!" || r++;
      n = l;
    } while (n);
    pl(t);
  }
  function gn(e) {
    for (; e != null; e = e.nextSibling) {
      var t = e.nodeType;
      if (t === 1 || t === 3) break;
      if (t === 8) {
        if (t = e.data, t === "$" || t === "$!" || t === "$?") break;
        if (t === "/$") return null;
      }
    }
    return e;
  }
  function zu(e) {
    e = e.previousSibling;
    for (var t = 0; e; ) {
      if (e.nodeType === 8) {
        var n = e.data;
        if (n === "$" || n === "$!" || n === "$?") {
          if (t === 0) return e;
          t--;
        } else n === "/$" && t++;
      }
      e = e.previousSibling;
    }
    return null;
  }
  var br = Math.random().toString(36).slice(2), $t = "__reactFiber$" + br, yl = "__reactProps$" + br, Zt = "__reactContainer$" + br, Ns = "__reactEvents$" + br, Ch = "__reactListeners$" + br, _h = "__reactHandles$" + br;
  function On(e) {
    var t = e[$t];
    if (t) return t;
    for (var n = e.parentNode; n; ) {
      if (t = n[Zt] || n[$t]) {
        if (n = t.alternate, t.child !== null || n !== null && n.child !== null) for (e = zu(e); e !== null; ) {
          if (n = e[$t]) return n;
          e = zu(e);
        }
        return t;
      }
      e = n, n = e.parentNode;
    }
    return null;
  }
  function bl(e) {
    return e = e[$t] || e[Zt], !e || e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3 ? null : e;
  }
  function ur(e) {
    if (e.tag === 5 || e.tag === 6) return e.stateNode;
    throw Error(b(33));
  }
  function eo(e) {
    return e[yl] || null;
  }
  var js = [], cr = -1;
  function En(e) {
    return {
      current: e
    };
  }
  function we(e) {
    0 > cr || (e.current = js[cr], js[cr] = null, cr--);
  }
  function xe(e, t) {
    cr++, js[cr] = e.current, e.current = t;
  }
  var Nn = {}, et = En(Nn), ut = En(false), Fn = Nn;
  function Nr(e, t) {
    var n = e.type.contextTypes;
    if (!n) return Nn;
    var r = e.stateNode;
    if (r && r.__reactInternalMemoizedUnmaskedChildContext === t) return r.__reactInternalMemoizedMaskedChildContext;
    var l = {}, a;
    for (a in n) l[a] = t[a];
    return r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = t, e.__reactInternalMemoizedMaskedChildContext = l), l;
  }
  function ct(e) {
    return e = e.childContextTypes, e != null;
  }
  function Ta() {
    we(ut), we(et);
  }
  function Au(e, t, n) {
    if (et.current !== Nn) throw Error(b(168));
    xe(et, t), xe(ut, n);
  }
  function tf(e, t, n) {
    var r = e.stateNode;
    if (t = t.childContextTypes, typeof r.getChildContext != "function") return n;
    r = r.getChildContext();
    for (var l in r) if (!(l in t)) throw Error(b(108, dp(e) || "Unknown", l));
    return Re({}, n, r);
  }
  function ba(e) {
    return e = (e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext || Nn, Fn = et.current, xe(et, e), xe(ut, ut.current), true;
  }
  function Uu(e, t, n) {
    var r = e.stateNode;
    if (!r) throw Error(b(169));
    n ? (e = tf(e, t, Fn), r.__reactInternalMemoizedMergedChildContext = e, we(ut), we(et), xe(et, e)) : we(ut), xe(ut, n);
  }
  var Kt = null, to = false, Io = false;
  function nf(e) {
    Kt === null ? Kt = [
      e
    ] : Kt.push(e);
  }
  function Ph(e) {
    to = true, nf(e);
  }
  function Cn() {
    if (!Io && Kt !== null) {
      Io = true;
      var e = 0, t = he;
      try {
        var n = Kt;
        for (he = 1; e < n.length; e++) {
          var r = n[e];
          do
            r = r(true);
          while (r !== null);
        }
        Kt = null, to = false;
      } catch (l) {
        throw Kt !== null && (Kt = Kt.slice(e + 1)), _d(oi, Cn), l;
      } finally {
        he = t, Io = false;
      }
    }
    return null;
  }
  var dr = [], fr = 0, Ma = null, Da = 0, St = [], kt = 0, Bn = null, Gt = 1, Yt = "";
  function bn(e, t) {
    dr[fr++] = Da, dr[fr++] = Ma, Ma = e, Da = t;
  }
  function rf(e, t, n) {
    St[kt++] = Gt, St[kt++] = Yt, St[kt++] = Bn, Bn = e;
    var r = Gt;
    e = Yt;
    var l = 32 - Lt(r) - 1;
    r &= ~(1 << l), n += 1;
    var a = 32 - Lt(t) + l;
    if (30 < a) {
      var o = l - l % 5;
      a = (r & (1 << o) - 1).toString(32), r >>= o, l -= o, Gt = 1 << 32 - Lt(t) + l | n << l | r, Yt = a + e;
    } else Gt = 1 << a | n << l | r, Yt = e;
  }
  function hi(e) {
    e.return !== null && (bn(e, 1), rf(e, 1, 0));
  }
  function gi(e) {
    for (; e === Ma; ) Ma = dr[--fr], dr[fr] = null, Da = dr[--fr], dr[fr] = null;
    for (; e === Bn; ) Bn = St[--kt], St[kt] = null, Yt = St[--kt], St[kt] = null, Gt = St[--kt], St[kt] = null;
  }
  var vt = null, gt = null, Ee = false, Dt = null;
  function lf(e, t) {
    var n = Nt(5, null, null, 0);
    n.elementType = "DELETED", n.stateNode = t, n.return = e, t = e.deletions, t === null ? (e.deletions = [
      n
    ], e.flags |= 16) : t.push(n);
  }
  function $u(e, t) {
    switch (e.tag) {
      case 5:
        var n = e.type;
        return t = t.nodeType !== 1 || n.toLowerCase() !== t.nodeName.toLowerCase() ? null : t, t !== null ? (e.stateNode = t, vt = e, gt = gn(t.firstChild), true) : false;
      case 6:
        return t = e.pendingProps === "" || t.nodeType !== 3 ? null : t, t !== null ? (e.stateNode = t, vt = e, gt = null, true) : false;
      case 13:
        return t = t.nodeType !== 8 ? null : t, t !== null ? (n = Bn !== null ? {
          id: Gt,
          overflow: Yt
        } : null, e.memoizedState = {
          dehydrated: t,
          treeContext: n,
          retryLane: 1073741824
        }, n = Nt(18, null, null, 0), n.stateNode = t, n.return = e, e.child = n, vt = e, gt = null, true) : false;
      default:
        return false;
    }
  }
  function Es(e) {
    return (e.mode & 1) !== 0 && (e.flags & 128) === 0;
  }
  function Cs(e) {
    if (Ee) {
      var t = gt;
      if (t) {
        var n = t;
        if (!$u(e, t)) {
          if (Es(e)) throw Error(b(418));
          t = gn(n.nextSibling);
          var r = vt;
          t && $u(e, t) ? lf(r, n) : (e.flags = e.flags & -4097 | 2, Ee = false, vt = e);
        }
      } else {
        if (Es(e)) throw Error(b(418));
        e.flags = e.flags & -4097 | 2, Ee = false, vt = e;
      }
    }
  }
  function Fu(e) {
    for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13; ) e = e.return;
    vt = e;
  }
  function ql(e) {
    if (e !== vt) return false;
    if (!Ee) return Fu(e), Ee = true, false;
    var t;
    if ((t = e.tag !== 3) && !(t = e.tag !== 5) && (t = e.type, t = t !== "head" && t !== "body" && !Ss(e.type, e.memoizedProps)), t && (t = gt)) {
      if (Es(e)) throw af(), Error(b(418));
      for (; t; ) lf(e, t), t = gn(t.nextSibling);
    }
    if (Fu(e), e.tag === 13) {
      if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e) throw Error(b(317));
      e: {
        for (e = e.nextSibling, t = 0; e; ) {
          if (e.nodeType === 8) {
            var n = e.data;
            if (n === "/$") {
              if (t === 0) {
                gt = gn(e.nextSibling);
                break e;
              }
              t--;
            } else n !== "$" && n !== "$!" && n !== "$?" || t++;
          }
          e = e.nextSibling;
        }
        gt = null;
      }
    } else gt = vt ? gn(e.stateNode.nextSibling) : null;
    return true;
  }
  function af() {
    for (var e = gt; e; ) e = gn(e.nextSibling);
  }
  function jr() {
    gt = vt = null, Ee = false;
  }
  function vi(e) {
    Dt === null ? Dt = [
      e
    ] : Dt.push(e);
  }
  var Rh = tn.ReactCurrentBatchConfig;
  function Fr(e, t, n) {
    if (e = n.ref, e !== null && typeof e != "function" && typeof e != "object") {
      if (n._owner) {
        if (n = n._owner, n) {
          if (n.tag !== 1) throw Error(b(309));
          var r = n.stateNode;
        }
        if (!r) throw Error(b(147, e));
        var l = r, a = "" + e;
        return t !== null && t.ref !== null && typeof t.ref == "function" && t.ref._stringRef === a ? t.ref : (t = function(o) {
          var i = l.refs;
          o === null ? delete i[a] : i[a] = o;
        }, t._stringRef = a, t);
      }
      if (typeof e != "string") throw Error(b(284));
      if (!n._owner) throw Error(b(290, e));
    }
    return e;
  }
  function ea(e, t) {
    throw e = Object.prototype.toString.call(t), Error(b(31, e === "[object Object]" ? "object with keys {" + Object.keys(t).join(", ") + "}" : e));
  }
  function Bu(e) {
    var t = e._init;
    return t(e._payload);
  }
  function of(e) {
    function t(p, f) {
      if (e) {
        var h = p.deletions;
        h === null ? (p.deletions = [
          f
        ], p.flags |= 16) : h.push(f);
      }
    }
    function n(p, f) {
      if (!e) return null;
      for (; f !== null; ) t(p, f), f = f.sibling;
      return null;
    }
    function r(p, f) {
      for (p = /* @__PURE__ */ new Map(); f !== null; ) f.key !== null ? p.set(f.key, f) : p.set(f.index, f), f = f.sibling;
      return p;
    }
    function l(p, f) {
      return p = wn(p, f), p.index = 0, p.sibling = null, p;
    }
    function a(p, f, h) {
      return p.index = h, e ? (h = p.alternate, h !== null ? (h = h.index, h < f ? (p.flags |= 2, f) : h) : (p.flags |= 2, f)) : (p.flags |= 1048576, f);
    }
    function o(p) {
      return e && p.alternate === null && (p.flags |= 2), p;
    }
    function i(p, f, h, j) {
      return f === null || f.tag !== 6 ? (f = Wo(h, p.mode, j), f.return = p, f) : (f = l(f, h), f.return = p, f);
    }
    function s(p, f, h, j) {
      var C = h.type;
      return C === ar ? m(p, f, h.props.children, j, h.key) : f !== null && (f.elementType === C || typeof C == "object" && C !== null && C.$$typeof === an && Bu(C) === f.type) ? (j = l(f, h.props), j.ref = Fr(p, f, h), j.return = p, j) : (j = xa(h.type, h.key, h.props, null, p.mode, j), j.ref = Fr(p, f, h), j.return = p, j);
    }
    function c(p, f, h, j) {
      return f === null || f.tag !== 4 || f.stateNode.containerInfo !== h.containerInfo || f.stateNode.implementation !== h.implementation ? (f = Vo(h, p.mode, j), f.return = p, f) : (f = l(f, h.children || []), f.return = p, f);
    }
    function m(p, f, h, j, C) {
      return f === null || f.tag !== 7 ? (f = $n(h, p.mode, j, C), f.return = p, f) : (f = l(f, h), f.return = p, f);
    }
    function d(p, f, h) {
      if (typeof f == "string" && f !== "" || typeof f == "number") return f = Wo("" + f, p.mode, h), f.return = p, f;
      if (typeof f == "object" && f !== null) {
        switch (f.$$typeof) {
          case Wl:
            return h = xa(f.type, f.key, f.props, null, p.mode, h), h.ref = Fr(p, null, f), h.return = p, h;
          case lr:
            return f = Vo(f, p.mode, h), f.return = p, f;
          case an:
            var j = f._init;
            return d(p, j(f._payload), h);
        }
        if (Yr(f) || Ir(f)) return f = $n(f, p.mode, h, null), f.return = p, f;
        ea(p, f);
      }
      return null;
    }
    function g(p, f, h, j) {
      var C = f !== null ? f.key : null;
      if (typeof h == "string" && h !== "" || typeof h == "number") return C !== null ? null : i(p, f, "" + h, j);
      if (typeof h == "object" && h !== null) {
        switch (h.$$typeof) {
          case Wl:
            return h.key === C ? s(p, f, h, j) : null;
          case lr:
            return h.key === C ? c(p, f, h, j) : null;
          case an:
            return C = h._init, g(p, f, C(h._payload), j);
        }
        if (Yr(h) || Ir(h)) return C !== null ? null : m(p, f, h, j, null);
        ea(p, h);
      }
      return null;
    }
    function x(p, f, h, j, C) {
      if (typeof j == "string" && j !== "" || typeof j == "number") return p = p.get(h) || null, i(f, p, "" + j, C);
      if (typeof j == "object" && j !== null) {
        switch (j.$$typeof) {
          case Wl:
            return p = p.get(j.key === null ? h : j.key) || null, s(f, p, j, C);
          case lr:
            return p = p.get(j.key === null ? h : j.key) || null, c(f, p, j, C);
          case an:
            var R = j._init;
            return x(p, f, h, R(j._payload), C);
        }
        if (Yr(j) || Ir(j)) return p = p.get(h) || null, m(f, p, j, C, null);
        ea(f, j);
      }
      return null;
    }
    function w(p, f, h, j) {
      for (var C = null, R = null, k = f, P = f = 0, $ = null; k !== null && P < h.length; P++) {
        k.index > P ? ($ = k, k = null) : $ = k.sibling;
        var L = g(p, k, h[P], j);
        if (L === null) {
          k === null && (k = $);
          break;
        }
        e && k && L.alternate === null && t(p, k), f = a(L, f, P), R === null ? C = L : R.sibling = L, R = L, k = $;
      }
      if (P === h.length) return n(p, k), Ee && bn(p, P), C;
      if (k === null) {
        for (; P < h.length; P++) k = d(p, h[P], j), k !== null && (f = a(k, f, P), R === null ? C = k : R.sibling = k, R = k);
        return Ee && bn(p, P), C;
      }
      for (k = r(p, k); P < h.length; P++) $ = x(k, p, P, h[P], j), $ !== null && (e && $.alternate !== null && k.delete($.key === null ? P : $.key), f = a($, f, P), R === null ? C = $ : R.sibling = $, R = $);
      return e && k.forEach(function(G) {
        return t(p, G);
      }), Ee && bn(p, P), C;
    }
    function S(p, f, h, j) {
      var C = Ir(h);
      if (typeof C != "function") throw Error(b(150));
      if (h = C.call(h), h == null) throw Error(b(151));
      for (var R = C = null, k = f, P = f = 0, $ = null, L = h.next(); k !== null && !L.done; P++, L = h.next()) {
        k.index > P ? ($ = k, k = null) : $ = k.sibling;
        var G = g(p, k, L.value, j);
        if (G === null) {
          k === null && (k = $);
          break;
        }
        e && k && G.alternate === null && t(p, k), f = a(G, f, P), R === null ? C = G : R.sibling = G, R = G, k = $;
      }
      if (L.done) return n(p, k), Ee && bn(p, P), C;
      if (k === null) {
        for (; !L.done; P++, L = h.next()) L = d(p, L.value, j), L !== null && (f = a(L, f, P), R === null ? C = L : R.sibling = L, R = L);
        return Ee && bn(p, P), C;
      }
      for (k = r(p, k); !L.done; P++, L = h.next()) L = x(k, p, P, L.value, j), L !== null && (e && L.alternate !== null && k.delete(L.key === null ? P : L.key), f = a(L, f, P), R === null ? C = L : R.sibling = L, R = L);
      return e && k.forEach(function(X) {
        return t(p, X);
      }), Ee && bn(p, P), C;
    }
    function _(p, f, h, j) {
      if (typeof h == "object" && h !== null && h.type === ar && h.key === null && (h = h.props.children), typeof h == "object" && h !== null) {
        switch (h.$$typeof) {
          case Wl:
            e: {
              for (var C = h.key, R = f; R !== null; ) {
                if (R.key === C) {
                  if (C = h.type, C === ar) {
                    if (R.tag === 7) {
                      n(p, R.sibling), f = l(R, h.props.children), f.return = p, p = f;
                      break e;
                    }
                  } else if (R.elementType === C || typeof C == "object" && C !== null && C.$$typeof === an && Bu(C) === R.type) {
                    n(p, R.sibling), f = l(R, h.props), f.ref = Fr(p, R, h), f.return = p, p = f;
                    break e;
                  }
                  n(p, R);
                  break;
                } else t(p, R);
                R = R.sibling;
              }
              h.type === ar ? (f = $n(h.props.children, p.mode, j, h.key), f.return = p, p = f) : (j = xa(h.type, h.key, h.props, null, p.mode, j), j.ref = Fr(p, f, h), j.return = p, p = j);
            }
            return o(p);
          case lr:
            e: {
              for (R = h.key; f !== null; ) {
                if (f.key === R) if (f.tag === 4 && f.stateNode.containerInfo === h.containerInfo && f.stateNode.implementation === h.implementation) {
                  n(p, f.sibling), f = l(f, h.children || []), f.return = p, p = f;
                  break e;
                } else {
                  n(p, f);
                  break;
                }
                else t(p, f);
                f = f.sibling;
              }
              f = Vo(h, p.mode, j), f.return = p, p = f;
            }
            return o(p);
          case an:
            return R = h._init, _(p, f, R(h._payload), j);
        }
        if (Yr(h)) return w(p, f, h, j);
        if (Ir(h)) return S(p, f, h, j);
        ea(p, h);
      }
      return typeof h == "string" && h !== "" || typeof h == "number" ? (h = "" + h, f !== null && f.tag === 6 ? (n(p, f.sibling), f = l(f, h), f.return = p, p = f) : (n(p, f), f = Wo(h, p.mode, j), f.return = p, p = f), o(p)) : n(p, f);
    }
    return _;
  }
  var Er = of(true), sf = of(false), La = En(null), Oa = null, mr = null, xi = null;
  function yi() {
    xi = mr = Oa = null;
  }
  function wi(e) {
    var t = La.current;
    we(La), e._currentValue = t;
  }
  function _s(e, t, n) {
    for (; e !== null; ) {
      var r = e.alternate;
      if ((e.childLanes & t) !== t ? (e.childLanes |= t, r !== null && (r.childLanes |= t)) : r !== null && (r.childLanes & t) !== t && (r.childLanes |= t), e === n) break;
      e = e.return;
    }
  }
  function wr(e, t) {
    Oa = e, xi = mr = null, e = e.dependencies, e !== null && e.firstContext !== null && (e.lanes & t && (it = true), e.firstContext = null);
  }
  function Et(e) {
    var t = e._currentValue;
    if (xi !== e) if (e = {
      context: e,
      memoizedValue: t,
      next: null
    }, mr === null) {
      if (Oa === null) throw Error(b(308));
      mr = e, Oa.dependencies = {
        lanes: 0,
        firstContext: e
      };
    } else mr = mr.next = e;
    return t;
  }
  var In = null;
  function Si(e) {
    In === null ? In = [
      e
    ] : In.push(e);
  }
  function uf(e, t, n, r) {
    var l = t.interleaved;
    return l === null ? (n.next = n, Si(t)) : (n.next = l.next, l.next = n), t.interleaved = n, qt(e, r);
  }
  function qt(e, t) {
    e.lanes |= t;
    var n = e.alternate;
    for (n !== null && (n.lanes |= t), n = e, e = e.return; e !== null; ) e.childLanes |= t, n = e.alternate, n !== null && (n.childLanes |= t), n = e, e = e.return;
    return n.tag === 3 ? n.stateNode : null;
  }
  var on = false;
  function ki(e) {
    e.updateQueue = {
      baseState: e.memoizedState,
      firstBaseUpdate: null,
      lastBaseUpdate: null,
      shared: {
        pending: null,
        interleaved: null,
        lanes: 0
      },
      effects: null
    };
  }
  function cf(e, t) {
    e = e.updateQueue, t.updateQueue === e && (t.updateQueue = {
      baseState: e.baseState,
      firstBaseUpdate: e.firstBaseUpdate,
      lastBaseUpdate: e.lastBaseUpdate,
      shared: e.shared,
      effects: e.effects
    });
  }
  function Jt(e, t) {
    return {
      eventTime: e,
      lane: t,
      tag: 0,
      payload: null,
      callback: null,
      next: null
    };
  }
  function vn(e, t, n) {
    var r = e.updateQueue;
    if (r === null) return null;
    if (r = r.shared, ie & 2) {
      var l = r.pending;
      return l === null ? t.next = t : (t.next = l.next, l.next = t), r.pending = t, qt(e, n);
    }
    return l = r.interleaved, l === null ? (t.next = t, Si(r)) : (t.next = l.next, l.next = t), r.interleaved = t, qt(e, n);
  }
  function fa(e, t, n) {
    if (t = t.updateQueue, t !== null && (t = t.shared, (n & 4194240) !== 0)) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, si(e, n);
    }
  }
  function Wu(e, t) {
    var n = e.updateQueue, r = e.alternate;
    if (r !== null && (r = r.updateQueue, n === r)) {
      var l = null, a = null;
      if (n = n.firstBaseUpdate, n !== null) {
        do {
          var o = {
            eventTime: n.eventTime,
            lane: n.lane,
            tag: n.tag,
            payload: n.payload,
            callback: n.callback,
            next: null
          };
          a === null ? l = a = o : a = a.next = o, n = n.next;
        } while (n !== null);
        a === null ? l = a = t : a = a.next = t;
      } else l = a = t;
      n = {
        baseState: r.baseState,
        firstBaseUpdate: l,
        lastBaseUpdate: a,
        shared: r.shared,
        effects: r.effects
      }, e.updateQueue = n;
      return;
    }
    e = n.lastBaseUpdate, e === null ? n.firstBaseUpdate = t : e.next = t, n.lastBaseUpdate = t;
  }
  function Ia(e, t, n, r) {
    var l = e.updateQueue;
    on = false;
    var a = l.firstBaseUpdate, o = l.lastBaseUpdate, i = l.shared.pending;
    if (i !== null) {
      l.shared.pending = null;
      var s = i, c = s.next;
      s.next = null, o === null ? a = c : o.next = c, o = s;
      var m = e.alternate;
      m !== null && (m = m.updateQueue, i = m.lastBaseUpdate, i !== o && (i === null ? m.firstBaseUpdate = c : i.next = c, m.lastBaseUpdate = s));
    }
    if (a !== null) {
      var d = l.baseState;
      o = 0, m = c = s = null, i = a;
      do {
        var g = i.lane, x = i.eventTime;
        if ((r & g) === g) {
          m !== null && (m = m.next = {
            eventTime: x,
            lane: 0,
            tag: i.tag,
            payload: i.payload,
            callback: i.callback,
            next: null
          });
          e: {
            var w = e, S = i;
            switch (g = t, x = n, S.tag) {
              case 1:
                if (w = S.payload, typeof w == "function") {
                  d = w.call(x, d, g);
                  break e;
                }
                d = w;
                break e;
              case 3:
                w.flags = w.flags & -65537 | 128;
              case 0:
                if (w = S.payload, g = typeof w == "function" ? w.call(x, d, g) : w, g == null) break e;
                d = Re({}, d, g);
                break e;
              case 2:
                on = true;
            }
          }
          i.callback !== null && i.lane !== 0 && (e.flags |= 64, g = l.effects, g === null ? l.effects = [
            i
          ] : g.push(i));
        } else x = {
          eventTime: x,
          lane: g,
          tag: i.tag,
          payload: i.payload,
          callback: i.callback,
          next: null
        }, m === null ? (c = m = x, s = d) : m = m.next = x, o |= g;
        if (i = i.next, i === null) {
          if (i = l.shared.pending, i === null) break;
          g = i, i = g.next, g.next = null, l.lastBaseUpdate = g, l.shared.pending = null;
        }
      } while (true);
      if (m === null && (s = d), l.baseState = s, l.firstBaseUpdate = c, l.lastBaseUpdate = m, t = l.shared.interleaved, t !== null) {
        l = t;
        do
          o |= l.lane, l = l.next;
        while (l !== t);
      } else a === null && (l.shared.lanes = 0);
      Vn |= o, e.lanes = o, e.memoizedState = d;
    }
  }
  function Vu(e, t, n) {
    if (e = t.effects, t.effects = null, e !== null) for (t = 0; t < e.length; t++) {
      var r = e[t], l = r.callback;
      if (l !== null) {
        if (r.callback = null, r = n, typeof l != "function") throw Error(b(191, l));
        l.call(r);
      }
    }
  }
  var Ml = {}, Bt = En(Ml), wl = En(Ml), Sl = En(Ml);
  function zn(e) {
    if (e === Ml) throw Error(b(174));
    return e;
  }
  function Ni(e, t) {
    switch (xe(Sl, t), xe(wl, e), xe(Bt, Ml), e = t.nodeType, e) {
      case 9:
      case 11:
        t = (t = t.documentElement) ? t.namespaceURI : ss(null, "");
        break;
      default:
        e = e === 8 ? t.parentNode : t, t = e.namespaceURI || null, e = e.tagName, t = ss(t, e);
    }
    we(Bt), xe(Bt, t);
  }
  function Cr() {
    we(Bt), we(wl), we(Sl);
  }
  function df(e) {
    zn(Sl.current);
    var t = zn(Bt.current), n = ss(t, e.type);
    t !== n && (xe(wl, e), xe(Bt, n));
  }
  function ji(e) {
    wl.current === e && (we(Bt), we(wl));
  }
  var _e = En(0);
  function za(e) {
    for (var t = e; t !== null; ) {
      if (t.tag === 13) {
        var n = t.memoizedState;
        if (n !== null && (n = n.dehydrated, n === null || n.data === "$?" || n.data === "$!")) return t;
      } else if (t.tag === 19 && t.memoizedProps.revealOrder !== void 0) {
        if (t.flags & 128) return t;
      } else if (t.child !== null) {
        t.child.return = t, t = t.child;
        continue;
      }
      if (t === e) break;
      for (; t.sibling === null; ) {
        if (t.return === null || t.return === e) return null;
        t = t.return;
      }
      t.sibling.return = t.return, t = t.sibling;
    }
    return null;
  }
  var zo = [];
  function Ei() {
    for (var e = 0; e < zo.length; e++) zo[e]._workInProgressVersionPrimary = null;
    zo.length = 0;
  }
  var ma = tn.ReactCurrentDispatcher, Ao = tn.ReactCurrentBatchConfig, Wn = 0, Pe = null, $e = null, Be = null, Aa = false, ll = false, kl = 0, Th = 0;
  function Xe() {
    throw Error(b(321));
  }
  function Ci(e, t) {
    if (t === null) return false;
    for (var n = 0; n < t.length && n < e.length; n++) if (!It(e[n], t[n])) return false;
    return true;
  }
  function _i(e, t, n, r, l, a) {
    if (Wn = a, Pe = t, t.memoizedState = null, t.updateQueue = null, t.lanes = 0, ma.current = e === null || e.memoizedState === null ? Lh : Oh, e = n(r, l), ll) {
      a = 0;
      do {
        if (ll = false, kl = 0, 25 <= a) throw Error(b(301));
        a += 1, Be = $e = null, t.updateQueue = null, ma.current = Ih, e = n(r, l);
      } while (ll);
    }
    if (ma.current = Ua, t = $e !== null && $e.next !== null, Wn = 0, Be = $e = Pe = null, Aa = false, t) throw Error(b(300));
    return e;
  }
  function Pi() {
    var e = kl !== 0;
    return kl = 0, e;
  }
  function Ut() {
    var e = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return Be === null ? Pe.memoizedState = Be = e : Be = Be.next = e, Be;
  }
  function Ct() {
    if ($e === null) {
      var e = Pe.alternate;
      e = e !== null ? e.memoizedState : null;
    } else e = $e.next;
    var t = Be === null ? Pe.memoizedState : Be.next;
    if (t !== null) Be = t, $e = e;
    else {
      if (e === null) throw Error(b(310));
      $e = e, e = {
        memoizedState: $e.memoizedState,
        baseState: $e.baseState,
        baseQueue: $e.baseQueue,
        queue: $e.queue,
        next: null
      }, Be === null ? Pe.memoizedState = Be = e : Be = Be.next = e;
    }
    return Be;
  }
  function Nl(e, t) {
    return typeof t == "function" ? t(e) : t;
  }
  function Uo(e) {
    var t = Ct(), n = t.queue;
    if (n === null) throw Error(b(311));
    n.lastRenderedReducer = e;
    var r = $e, l = r.baseQueue, a = n.pending;
    if (a !== null) {
      if (l !== null) {
        var o = l.next;
        l.next = a.next, a.next = o;
      }
      r.baseQueue = l = a, n.pending = null;
    }
    if (l !== null) {
      a = l.next, r = r.baseState;
      var i = o = null, s = null, c = a;
      do {
        var m = c.lane;
        if ((Wn & m) === m) s !== null && (s = s.next = {
          lane: 0,
          action: c.action,
          hasEagerState: c.hasEagerState,
          eagerState: c.eagerState,
          next: null
        }), r = c.hasEagerState ? c.eagerState : e(r, c.action);
        else {
          var d = {
            lane: m,
            action: c.action,
            hasEagerState: c.hasEagerState,
            eagerState: c.eagerState,
            next: null
          };
          s === null ? (i = s = d, o = r) : s = s.next = d, Pe.lanes |= m, Vn |= m;
        }
        c = c.next;
      } while (c !== null && c !== a);
      s === null ? o = r : s.next = i, It(r, t.memoizedState) || (it = true), t.memoizedState = r, t.baseState = o, t.baseQueue = s, n.lastRenderedState = r;
    }
    if (e = n.interleaved, e !== null) {
      l = e;
      do
        a = l.lane, Pe.lanes |= a, Vn |= a, l = l.next;
      while (l !== e);
    } else l === null && (n.lanes = 0);
    return [
      t.memoizedState,
      n.dispatch
    ];
  }
  function $o(e) {
    var t = Ct(), n = t.queue;
    if (n === null) throw Error(b(311));
    n.lastRenderedReducer = e;
    var r = n.dispatch, l = n.pending, a = t.memoizedState;
    if (l !== null) {
      n.pending = null;
      var o = l = l.next;
      do
        a = e(a, o.action), o = o.next;
      while (o !== l);
      It(a, t.memoizedState) || (it = true), t.memoizedState = a, t.baseQueue === null && (t.baseState = a), n.lastRenderedState = a;
    }
    return [
      a,
      r
    ];
  }
  function ff() {
  }
  function mf(e, t) {
    var n = Pe, r = Ct(), l = t(), a = !It(r.memoizedState, l);
    if (a && (r.memoizedState = l, it = true), r = r.queue, Ri(gf.bind(null, n, r, e), [
      e
    ]), r.getSnapshot !== t || a || Be !== null && Be.memoizedState.tag & 1) {
      if (n.flags |= 2048, jl(9, hf.bind(null, n, r, l, t), void 0, null), We === null) throw Error(b(349));
      Wn & 30 || pf(n, t, l);
    }
    return l;
  }
  function pf(e, t, n) {
    e.flags |= 16384, e = {
      getSnapshot: t,
      value: n
    }, t = Pe.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, Pe.updateQueue = t, t.stores = [
      e
    ]) : (n = t.stores, n === null ? t.stores = [
      e
    ] : n.push(e));
  }
  function hf(e, t, n, r) {
    t.value = n, t.getSnapshot = r, vf(t) && xf(e);
  }
  function gf(e, t, n) {
    return n(function() {
      vf(t) && xf(e);
    });
  }
  function vf(e) {
    var t = e.getSnapshot;
    e = e.value;
    try {
      var n = t();
      return !It(e, n);
    } catch {
      return true;
    }
  }
  function xf(e) {
    var t = qt(e, 1);
    t !== null && Ot(t, e, 1, -1);
  }
  function Hu(e) {
    var t = Ut();
    return typeof e == "function" && (e = e()), t.memoizedState = t.baseState = e, e = {
      pending: null,
      interleaved: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Nl,
      lastRenderedState: e
    }, t.queue = e, e = e.dispatch = Dh.bind(null, Pe, e), [
      t.memoizedState,
      e
    ];
  }
  function jl(e, t, n, r) {
    return e = {
      tag: e,
      create: t,
      destroy: n,
      deps: r,
      next: null
    }, t = Pe.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, Pe.updateQueue = t, t.lastEffect = e.next = e) : (n = t.lastEffect, n === null ? t.lastEffect = e.next = e : (r = n.next, n.next = e, e.next = r, t.lastEffect = e)), e;
  }
  function yf() {
    return Ct().memoizedState;
  }
  function pa(e, t, n, r) {
    var l = Ut();
    Pe.flags |= e, l.memoizedState = jl(1 | t, n, void 0, r === void 0 ? null : r);
  }
  function no(e, t, n, r) {
    var l = Ct();
    r = r === void 0 ? null : r;
    var a = void 0;
    if ($e !== null) {
      var o = $e.memoizedState;
      if (a = o.destroy, r !== null && Ci(r, o.deps)) {
        l.memoizedState = jl(t, n, a, r);
        return;
      }
    }
    Pe.flags |= e, l.memoizedState = jl(1 | t, n, a, r);
  }
  function Qu(e, t) {
    return pa(8390656, 8, e, t);
  }
  function Ri(e, t) {
    return no(2048, 8, e, t);
  }
  function wf(e, t) {
    return no(4, 2, e, t);
  }
  function Sf(e, t) {
    return no(4, 4, e, t);
  }
  function kf(e, t) {
    if (typeof t == "function") return e = e(), t(e), function() {
      t(null);
    };
    if (t != null) return e = e(), t.current = e, function() {
      t.current = null;
    };
  }
  function Nf(e, t, n) {
    return n = n != null ? n.concat([
      e
    ]) : null, no(4, 4, kf.bind(null, t, e), n);
  }
  function Ti() {
  }
  function jf(e, t) {
    var n = Ct();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Ci(t, r[1]) ? r[0] : (n.memoizedState = [
      e,
      t
    ], e);
  }
  function Ef(e, t) {
    var n = Ct();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Ci(t, r[1]) ? r[0] : (e = e(), n.memoizedState = [
      e,
      t
    ], e);
  }
  function Cf(e, t, n) {
    return Wn & 21 ? (It(n, t) || (n = Td(), Pe.lanes |= n, Vn |= n, e.baseState = true), t) : (e.baseState && (e.baseState = false, it = true), e.memoizedState = n);
  }
  function bh(e, t) {
    var n = he;
    he = n !== 0 && 4 > n ? n : 4, e(true);
    var r = Ao.transition;
    Ao.transition = {};
    try {
      e(false), t();
    } finally {
      he = n, Ao.transition = r;
    }
  }
  function _f() {
    return Ct().memoizedState;
  }
  function Mh(e, t, n) {
    var r = yn(e);
    if (n = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    }, Pf(e)) Rf(t, n);
    else if (n = uf(e, t, n, r), n !== null) {
      var l = rt();
      Ot(n, e, r, l), Tf(n, t, r);
    }
  }
  function Dh(e, t, n) {
    var r = yn(e), l = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    };
    if (Pf(e)) Rf(t, l);
    else {
      var a = e.alternate;
      if (e.lanes === 0 && (a === null || a.lanes === 0) && (a = t.lastRenderedReducer, a !== null)) try {
        var o = t.lastRenderedState, i = a(o, n);
        if (l.hasEagerState = true, l.eagerState = i, It(i, o)) {
          var s = t.interleaved;
          s === null ? (l.next = l, Si(t)) : (l.next = s.next, s.next = l), t.interleaved = l;
          return;
        }
      } catch {
      } finally {
      }
      n = uf(e, t, l, r), n !== null && (l = rt(), Ot(n, e, r, l), Tf(n, t, r));
    }
  }
  function Pf(e) {
    var t = e.alternate;
    return e === Pe || t !== null && t === Pe;
  }
  function Rf(e, t) {
    ll = Aa = true;
    var n = e.pending;
    n === null ? t.next = t : (t.next = n.next, n.next = t), e.pending = t;
  }
  function Tf(e, t, n) {
    if (n & 4194240) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, si(e, n);
    }
  }
  var Ua = {
    readContext: Et,
    useCallback: Xe,
    useContext: Xe,
    useEffect: Xe,
    useImperativeHandle: Xe,
    useInsertionEffect: Xe,
    useLayoutEffect: Xe,
    useMemo: Xe,
    useReducer: Xe,
    useRef: Xe,
    useState: Xe,
    useDebugValue: Xe,
    useDeferredValue: Xe,
    useTransition: Xe,
    useMutableSource: Xe,
    useSyncExternalStore: Xe,
    useId: Xe,
    unstable_isNewReconciler: false
  }, Lh = {
    readContext: Et,
    useCallback: function(e, t) {
      return Ut().memoizedState = [
        e,
        t === void 0 ? null : t
      ], e;
    },
    useContext: Et,
    useEffect: Qu,
    useImperativeHandle: function(e, t, n) {
      return n = n != null ? n.concat([
        e
      ]) : null, pa(4194308, 4, kf.bind(null, t, e), n);
    },
    useLayoutEffect: function(e, t) {
      return pa(4194308, 4, e, t);
    },
    useInsertionEffect: function(e, t) {
      return pa(4, 2, e, t);
    },
    useMemo: function(e, t) {
      var n = Ut();
      return t = t === void 0 ? null : t, e = e(), n.memoizedState = [
        e,
        t
      ], e;
    },
    useReducer: function(e, t, n) {
      var r = Ut();
      return t = n !== void 0 ? n(t) : t, r.memoizedState = r.baseState = t, e = {
        pending: null,
        interleaved: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: e,
        lastRenderedState: t
      }, r.queue = e, e = e.dispatch = Mh.bind(null, Pe, e), [
        r.memoizedState,
        e
      ];
    },
    useRef: function(e) {
      var t = Ut();
      return e = {
        current: e
      }, t.memoizedState = e;
    },
    useState: Hu,
    useDebugValue: Ti,
    useDeferredValue: function(e) {
      return Ut().memoizedState = e;
    },
    useTransition: function() {
      var e = Hu(false), t = e[0];
      return e = bh.bind(null, e[1]), Ut().memoizedState = e, [
        t,
        e
      ];
    },
    useMutableSource: function() {
    },
    useSyncExternalStore: function(e, t, n) {
      var r = Pe, l = Ut();
      if (Ee) {
        if (n === void 0) throw Error(b(407));
        n = n();
      } else {
        if (n = t(), We === null) throw Error(b(349));
        Wn & 30 || pf(r, t, n);
      }
      l.memoizedState = n;
      var a = {
        value: n,
        getSnapshot: t
      };
      return l.queue = a, Qu(gf.bind(null, r, a, e), [
        e
      ]), r.flags |= 2048, jl(9, hf.bind(null, r, a, n, t), void 0, null), n;
    },
    useId: function() {
      var e = Ut(), t = We.identifierPrefix;
      if (Ee) {
        var n = Yt, r = Gt;
        n = (r & ~(1 << 32 - Lt(r) - 1)).toString(32) + n, t = ":" + t + "R" + n, n = kl++, 0 < n && (t += "H" + n.toString(32)), t += ":";
      } else n = Th++, t = ":" + t + "r" + n.toString(32) + ":";
      return e.memoizedState = t;
    },
    unstable_isNewReconciler: false
  }, Oh = {
    readContext: Et,
    useCallback: jf,
    useContext: Et,
    useEffect: Ri,
    useImperativeHandle: Nf,
    useInsertionEffect: wf,
    useLayoutEffect: Sf,
    useMemo: Ef,
    useReducer: Uo,
    useRef: yf,
    useState: function() {
      return Uo(Nl);
    },
    useDebugValue: Ti,
    useDeferredValue: function(e) {
      var t = Ct();
      return Cf(t, $e.memoizedState, e);
    },
    useTransition: function() {
      var e = Uo(Nl)[0], t = Ct().memoizedState;
      return [
        e,
        t
      ];
    },
    useMutableSource: ff,
    useSyncExternalStore: mf,
    useId: _f,
    unstable_isNewReconciler: false
  }, Ih = {
    readContext: Et,
    useCallback: jf,
    useContext: Et,
    useEffect: Ri,
    useImperativeHandle: Nf,
    useInsertionEffect: wf,
    useLayoutEffect: Sf,
    useMemo: Ef,
    useReducer: $o,
    useRef: yf,
    useState: function() {
      return $o(Nl);
    },
    useDebugValue: Ti,
    useDeferredValue: function(e) {
      var t = Ct();
      return $e === null ? t.memoizedState = e : Cf(t, $e.memoizedState, e);
    },
    useTransition: function() {
      var e = $o(Nl)[0], t = Ct().memoizedState;
      return [
        e,
        t
      ];
    },
    useMutableSource: ff,
    useSyncExternalStore: mf,
    useId: _f,
    unstable_isNewReconciler: false
  };
  function Tt(e, t) {
    if (e && e.defaultProps) {
      t = Re({}, t), e = e.defaultProps;
      for (var n in e) t[n] === void 0 && (t[n] = e[n]);
      return t;
    }
    return t;
  }
  function Ps(e, t, n, r) {
    t = e.memoizedState, n = n(r, t), n = n == null ? t : Re({}, t, n), e.memoizedState = n, e.lanes === 0 && (e.updateQueue.baseState = n);
  }
  var ro = {
    isMounted: function(e) {
      return (e = e._reactInternals) ? Gn(e) === e : false;
    },
    enqueueSetState: function(e, t, n) {
      e = e._reactInternals;
      var r = rt(), l = yn(e), a = Jt(r, l);
      a.payload = t, n != null && (a.callback = n), t = vn(e, a, l), t !== null && (Ot(t, e, l, r), fa(t, e, l));
    },
    enqueueReplaceState: function(e, t, n) {
      e = e._reactInternals;
      var r = rt(), l = yn(e), a = Jt(r, l);
      a.tag = 1, a.payload = t, n != null && (a.callback = n), t = vn(e, a, l), t !== null && (Ot(t, e, l, r), fa(t, e, l));
    },
    enqueueForceUpdate: function(e, t) {
      e = e._reactInternals;
      var n = rt(), r = yn(e), l = Jt(n, r);
      l.tag = 2, t != null && (l.callback = t), t = vn(e, l, r), t !== null && (Ot(t, e, r, n), fa(t, e, r));
    }
  };
  function Ku(e, t, n, r, l, a, o) {
    return e = e.stateNode, typeof e.shouldComponentUpdate == "function" ? e.shouldComponentUpdate(r, a, o) : t.prototype && t.prototype.isPureReactComponent ? !gl(n, r) || !gl(l, a) : true;
  }
  function bf(e, t, n) {
    var r = false, l = Nn, a = t.contextType;
    return typeof a == "object" && a !== null ? a = Et(a) : (l = ct(t) ? Fn : et.current, r = t.contextTypes, a = (r = r != null) ? Nr(e, l) : Nn), t = new t(n, a), e.memoizedState = t.state !== null && t.state !== void 0 ? t.state : null, t.updater = ro, e.stateNode = t, t._reactInternals = e, r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = l, e.__reactInternalMemoizedMaskedChildContext = a), t;
  }
  function Gu(e, t, n, r) {
    e = t.state, typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(n, r), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(n, r), t.state !== e && ro.enqueueReplaceState(t, t.state, null);
  }
  function Rs(e, t, n, r) {
    var l = e.stateNode;
    l.props = n, l.state = e.memoizedState, l.refs = {}, ki(e);
    var a = t.contextType;
    typeof a == "object" && a !== null ? l.context = Et(a) : (a = ct(t) ? Fn : et.current, l.context = Nr(e, a)), l.state = e.memoizedState, a = t.getDerivedStateFromProps, typeof a == "function" && (Ps(e, t, a, n), l.state = e.memoizedState), typeof t.getDerivedStateFromProps == "function" || typeof l.getSnapshotBeforeUpdate == "function" || typeof l.UNSAFE_componentWillMount != "function" && typeof l.componentWillMount != "function" || (t = l.state, typeof l.componentWillMount == "function" && l.componentWillMount(), typeof l.UNSAFE_componentWillMount == "function" && l.UNSAFE_componentWillMount(), t !== l.state && ro.enqueueReplaceState(l, l.state, null), Ia(e, n, l, r), l.state = e.memoizedState), typeof l.componentDidMount == "function" && (e.flags |= 4194308);
  }
  function _r(e, t) {
    try {
      var n = "", r = t;
      do
        n += cp(r), r = r.return;
      while (r);
      var l = n;
    } catch (a) {
      l = `
Error generating stack: ` + a.message + `
` + a.stack;
    }
    return {
      value: e,
      source: t,
      stack: l,
      digest: null
    };
  }
  function Fo(e, t, n) {
    return {
      value: e,
      source: null,
      stack: n ?? null,
      digest: t ?? null
    };
  }
  function Ts(e, t) {
    try {
      console.error(t.value);
    } catch (n) {
      setTimeout(function() {
        throw n;
      });
    }
  }
  var zh = typeof WeakMap == "function" ? WeakMap : Map;
  function Mf(e, t, n) {
    n = Jt(-1, n), n.tag = 3, n.payload = {
      element: null
    };
    var r = t.value;
    return n.callback = function() {
      Fa || (Fa = true, $s = r), Ts(e, t);
    }, n;
  }
  function Df(e, t, n) {
    n = Jt(-1, n), n.tag = 3;
    var r = e.type.getDerivedStateFromError;
    if (typeof r == "function") {
      var l = t.value;
      n.payload = function() {
        return r(l);
      }, n.callback = function() {
        Ts(e, t);
      };
    }
    var a = e.stateNode;
    return a !== null && typeof a.componentDidCatch == "function" && (n.callback = function() {
      Ts(e, t), typeof r != "function" && (xn === null ? xn = /* @__PURE__ */ new Set([
        this
      ]) : xn.add(this));
      var o = t.stack;
      this.componentDidCatch(t.value, {
        componentStack: o !== null ? o : ""
      });
    }), n;
  }
  function Yu(e, t, n) {
    var r = e.pingCache;
    if (r === null) {
      r = e.pingCache = new zh();
      var l = /* @__PURE__ */ new Set();
      r.set(t, l);
    } else l = r.get(t), l === void 0 && (l = /* @__PURE__ */ new Set(), r.set(t, l));
    l.has(n) || (l.add(n), e = Xh.bind(null, e, t, n), t.then(e, e));
  }
  function Ju(e) {
    do {
      var t;
      if ((t = e.tag === 13) && (t = e.memoizedState, t = t !== null ? t.dehydrated !== null : true), t) return e;
      e = e.return;
    } while (e !== null);
    return null;
  }
  function Xu(e, t, n, r, l) {
    return e.mode & 1 ? (e.flags |= 65536, e.lanes = l, e) : (e === t ? e.flags |= 65536 : (e.flags |= 128, n.flags |= 131072, n.flags &= -52805, n.tag === 1 && (n.alternate === null ? n.tag = 17 : (t = Jt(-1, 1), t.tag = 2, vn(n, t, 1))), n.lanes |= 1), e);
  }
  var Ah = tn.ReactCurrentOwner, it = false;
  function nt(e, t, n, r) {
    t.child = e === null ? sf(t, null, n, r) : Er(t, e.child, n, r);
  }
  function Zu(e, t, n, r, l) {
    n = n.render;
    var a = t.ref;
    return wr(t, l), r = _i(e, t, n, r, a, l), n = Pi(), e !== null && !it ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, en(e, t, l)) : (Ee && n && hi(t), t.flags |= 1, nt(e, t, r, l), t.child);
  }
  function qu(e, t, n, r, l) {
    if (e === null) {
      var a = n.type;
      return typeof a == "function" && !Ai(a) && a.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0 ? (t.tag = 15, t.type = a, Lf(e, t, a, r, l)) : (e = xa(n.type, null, r, t, t.mode, l), e.ref = t.ref, e.return = t, t.child = e);
    }
    if (a = e.child, !(e.lanes & l)) {
      var o = a.memoizedProps;
      if (n = n.compare, n = n !== null ? n : gl, n(o, r) && e.ref === t.ref) return en(e, t, l);
    }
    return t.flags |= 1, e = wn(a, r), e.ref = t.ref, e.return = t, t.child = e;
  }
  function Lf(e, t, n, r, l) {
    if (e !== null) {
      var a = e.memoizedProps;
      if (gl(a, r) && e.ref === t.ref) if (it = false, t.pendingProps = r = a, (e.lanes & l) !== 0) e.flags & 131072 && (it = true);
      else return t.lanes = e.lanes, en(e, t, l);
    }
    return bs(e, t, n, r, l);
  }
  function Of(e, t, n) {
    var r = t.pendingProps, l = r.children, a = e !== null ? e.memoizedState : null;
    if (r.mode === "hidden") if (!(t.mode & 1)) t.memoizedState = {
      baseLanes: 0,
      cachePool: null,
      transitions: null
    }, xe(hr, pt), pt |= n;
    else {
      if (!(n & 1073741824)) return e = a !== null ? a.baseLanes | n : n, t.lanes = t.childLanes = 1073741824, t.memoizedState = {
        baseLanes: e,
        cachePool: null,
        transitions: null
      }, t.updateQueue = null, xe(hr, pt), pt |= e, null;
      t.memoizedState = {
        baseLanes: 0,
        cachePool: null,
        transitions: null
      }, r = a !== null ? a.baseLanes : n, xe(hr, pt), pt |= r;
    }
    else a !== null ? (r = a.baseLanes | n, t.memoizedState = null) : r = n, xe(hr, pt), pt |= r;
    return nt(e, t, l, n), t.child;
  }
  function If(e, t) {
    var n = t.ref;
    (e === null && n !== null || e !== null && e.ref !== n) && (t.flags |= 512, t.flags |= 2097152);
  }
  function bs(e, t, n, r, l) {
    var a = ct(n) ? Fn : et.current;
    return a = Nr(t, a), wr(t, l), n = _i(e, t, n, r, a, l), r = Pi(), e !== null && !it ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, en(e, t, l)) : (Ee && r && hi(t), t.flags |= 1, nt(e, t, n, l), t.child);
  }
  function ec(e, t, n, r, l) {
    if (ct(n)) {
      var a = true;
      ba(t);
    } else a = false;
    if (wr(t, l), t.stateNode === null) ha(e, t), bf(t, n, r), Rs(t, n, r, l), r = true;
    else if (e === null) {
      var o = t.stateNode, i = t.memoizedProps;
      o.props = i;
      var s = o.context, c = n.contextType;
      typeof c == "object" && c !== null ? c = Et(c) : (c = ct(n) ? Fn : et.current, c = Nr(t, c));
      var m = n.getDerivedStateFromProps, d = typeof m == "function" || typeof o.getSnapshotBeforeUpdate == "function";
      d || typeof o.UNSAFE_componentWillReceiveProps != "function" && typeof o.componentWillReceiveProps != "function" || (i !== r || s !== c) && Gu(t, o, r, c), on = false;
      var g = t.memoizedState;
      o.state = g, Ia(t, r, o, l), s = t.memoizedState, i !== r || g !== s || ut.current || on ? (typeof m == "function" && (Ps(t, n, m, r), s = t.memoizedState), (i = on || Ku(t, n, i, r, g, s, c)) ? (d || typeof o.UNSAFE_componentWillMount != "function" && typeof o.componentWillMount != "function" || (typeof o.componentWillMount == "function" && o.componentWillMount(), typeof o.UNSAFE_componentWillMount == "function" && o.UNSAFE_componentWillMount()), typeof o.componentDidMount == "function" && (t.flags |= 4194308)) : (typeof o.componentDidMount == "function" && (t.flags |= 4194308), t.memoizedProps = r, t.memoizedState = s), o.props = r, o.state = s, o.context = c, r = i) : (typeof o.componentDidMount == "function" && (t.flags |= 4194308), r = false);
    } else {
      o = t.stateNode, cf(e, t), i = t.memoizedProps, c = t.type === t.elementType ? i : Tt(t.type, i), o.props = c, d = t.pendingProps, g = o.context, s = n.contextType, typeof s == "object" && s !== null ? s = Et(s) : (s = ct(n) ? Fn : et.current, s = Nr(t, s));
      var x = n.getDerivedStateFromProps;
      (m = typeof x == "function" || typeof o.getSnapshotBeforeUpdate == "function") || typeof o.UNSAFE_componentWillReceiveProps != "function" && typeof o.componentWillReceiveProps != "function" || (i !== d || g !== s) && Gu(t, o, r, s), on = false, g = t.memoizedState, o.state = g, Ia(t, r, o, l);
      var w = t.memoizedState;
      i !== d || g !== w || ut.current || on ? (typeof x == "function" && (Ps(t, n, x, r), w = t.memoizedState), (c = on || Ku(t, n, c, r, g, w, s) || false) ? (m || typeof o.UNSAFE_componentWillUpdate != "function" && typeof o.componentWillUpdate != "function" || (typeof o.componentWillUpdate == "function" && o.componentWillUpdate(r, w, s), typeof o.UNSAFE_componentWillUpdate == "function" && o.UNSAFE_componentWillUpdate(r, w, s)), typeof o.componentDidUpdate == "function" && (t.flags |= 4), typeof o.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof o.componentDidUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 4), typeof o.getSnapshotBeforeUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 1024), t.memoizedProps = r, t.memoizedState = w), o.props = r, o.state = w, o.context = s, r = c) : (typeof o.componentDidUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 4), typeof o.getSnapshotBeforeUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 1024), r = false);
    }
    return Ms(e, t, n, r, a, l);
  }
  function Ms(e, t, n, r, l, a) {
    If(e, t);
    var o = (t.flags & 128) !== 0;
    if (!r && !o) return l && Uu(t, n, false), en(e, t, a);
    r = t.stateNode, Ah.current = t;
    var i = o && typeof n.getDerivedStateFromError != "function" ? null : r.render();
    return t.flags |= 1, e !== null && o ? (t.child = Er(t, e.child, null, a), t.child = Er(t, null, i, a)) : nt(e, t, i, a), t.memoizedState = r.state, l && Uu(t, n, true), t.child;
  }
  function zf(e) {
    var t = e.stateNode;
    t.pendingContext ? Au(e, t.pendingContext, t.pendingContext !== t.context) : t.context && Au(e, t.context, false), Ni(e, t.containerInfo);
  }
  function tc(e, t, n, r, l) {
    return jr(), vi(l), t.flags |= 256, nt(e, t, n, r), t.child;
  }
  var Ds = {
    dehydrated: null,
    treeContext: null,
    retryLane: 0
  };
  function Ls(e) {
    return {
      baseLanes: e,
      cachePool: null,
      transitions: null
    };
  }
  function Af(e, t, n) {
    var r = t.pendingProps, l = _e.current, a = false, o = (t.flags & 128) !== 0, i;
    if ((i = o) || (i = e !== null && e.memoizedState === null ? false : (l & 2) !== 0), i ? (a = true, t.flags &= -129) : (e === null || e.memoizedState !== null) && (l |= 1), xe(_e, l & 1), e === null) return Cs(t), e = t.memoizedState, e !== null && (e = e.dehydrated, e !== null) ? (t.mode & 1 ? e.data === "$!" ? t.lanes = 8 : t.lanes = 1073741824 : t.lanes = 1, null) : (o = r.children, e = r.fallback, a ? (r = t.mode, a = t.child, o = {
      mode: "hidden",
      children: o
    }, !(r & 1) && a !== null ? (a.childLanes = 0, a.pendingProps = o) : a = oo(o, r, 0, null), e = $n(e, r, n, null), a.return = t, e.return = t, a.sibling = e, t.child = a, t.child.memoizedState = Ls(n), t.memoizedState = Ds, e) : bi(t, o));
    if (l = e.memoizedState, l !== null && (i = l.dehydrated, i !== null)) return Uh(e, t, o, r, i, l, n);
    if (a) {
      a = r.fallback, o = t.mode, l = e.child, i = l.sibling;
      var s = {
        mode: "hidden",
        children: r.children
      };
      return !(o & 1) && t.child !== l ? (r = t.child, r.childLanes = 0, r.pendingProps = s, t.deletions = null) : (r = wn(l, s), r.subtreeFlags = l.subtreeFlags & 14680064), i !== null ? a = wn(i, a) : (a = $n(a, o, n, null), a.flags |= 2), a.return = t, r.return = t, r.sibling = a, t.child = r, r = a, a = t.child, o = e.child.memoizedState, o = o === null ? Ls(n) : {
        baseLanes: o.baseLanes | n,
        cachePool: null,
        transitions: o.transitions
      }, a.memoizedState = o, a.childLanes = e.childLanes & ~n, t.memoizedState = Ds, r;
    }
    return a = e.child, e = a.sibling, r = wn(a, {
      mode: "visible",
      children: r.children
    }), !(t.mode & 1) && (r.lanes = n), r.return = t, r.sibling = null, e !== null && (n = t.deletions, n === null ? (t.deletions = [
      e
    ], t.flags |= 16) : n.push(e)), t.child = r, t.memoizedState = null, r;
  }
  function bi(e, t) {
    return t = oo({
      mode: "visible",
      children: t
    }, e.mode, 0, null), t.return = e, e.child = t;
  }
  function ta(e, t, n, r) {
    return r !== null && vi(r), Er(t, e.child, null, n), e = bi(t, t.pendingProps.children), e.flags |= 2, t.memoizedState = null, e;
  }
  function Uh(e, t, n, r, l, a, o) {
    if (n) return t.flags & 256 ? (t.flags &= -257, r = Fo(Error(b(422))), ta(e, t, o, r)) : t.memoizedState !== null ? (t.child = e.child, t.flags |= 128, null) : (a = r.fallback, l = t.mode, r = oo({
      mode: "visible",
      children: r.children
    }, l, 0, null), a = $n(a, l, o, null), a.flags |= 2, r.return = t, a.return = t, r.sibling = a, t.child = r, t.mode & 1 && Er(t, e.child, null, o), t.child.memoizedState = Ls(o), t.memoizedState = Ds, a);
    if (!(t.mode & 1)) return ta(e, t, o, null);
    if (l.data === "$!") {
      if (r = l.nextSibling && l.nextSibling.dataset, r) var i = r.dgst;
      return r = i, a = Error(b(419)), r = Fo(a, r, void 0), ta(e, t, o, r);
    }
    if (i = (o & e.childLanes) !== 0, it || i) {
      if (r = We, r !== null) {
        switch (o & -o) {
          case 4:
            l = 2;
            break;
          case 16:
            l = 8;
            break;
          case 64:
          case 128:
          case 256:
          case 512:
          case 1024:
          case 2048:
          case 4096:
          case 8192:
          case 16384:
          case 32768:
          case 65536:
          case 131072:
          case 262144:
          case 524288:
          case 1048576:
          case 2097152:
          case 4194304:
          case 8388608:
          case 16777216:
          case 33554432:
          case 67108864:
            l = 32;
            break;
          case 536870912:
            l = 268435456;
            break;
          default:
            l = 0;
        }
        l = l & (r.suspendedLanes | o) ? 0 : l, l !== 0 && l !== a.retryLane && (a.retryLane = l, qt(e, l), Ot(r, e, l, -1));
      }
      return zi(), r = Fo(Error(b(421))), ta(e, t, o, r);
    }
    return l.data === "$?" ? (t.flags |= 128, t.child = e.child, t = Zh.bind(null, e), l._reactRetry = t, null) : (e = a.treeContext, gt = gn(l.nextSibling), vt = t, Ee = true, Dt = null, e !== null && (St[kt++] = Gt, St[kt++] = Yt, St[kt++] = Bn, Gt = e.id, Yt = e.overflow, Bn = t), t = bi(t, r.children), t.flags |= 4096, t);
  }
  function nc(e, t, n) {
    e.lanes |= t;
    var r = e.alternate;
    r !== null && (r.lanes |= t), _s(e.return, t, n);
  }
  function Bo(e, t, n, r, l) {
    var a = e.memoizedState;
    a === null ? e.memoizedState = {
      isBackwards: t,
      rendering: null,
      renderingStartTime: 0,
      last: r,
      tail: n,
      tailMode: l
    } : (a.isBackwards = t, a.rendering = null, a.renderingStartTime = 0, a.last = r, a.tail = n, a.tailMode = l);
  }
  function Uf(e, t, n) {
    var r = t.pendingProps, l = r.revealOrder, a = r.tail;
    if (nt(e, t, r.children, n), r = _e.current, r & 2) r = r & 1 | 2, t.flags |= 128;
    else {
      if (e !== null && e.flags & 128) e: for (e = t.child; e !== null; ) {
        if (e.tag === 13) e.memoizedState !== null && nc(e, n, t);
        else if (e.tag === 19) nc(e, n, t);
        else if (e.child !== null) {
          e.child.return = e, e = e.child;
          continue;
        }
        if (e === t) break e;
        for (; e.sibling === null; ) {
          if (e.return === null || e.return === t) break e;
          e = e.return;
        }
        e.sibling.return = e.return, e = e.sibling;
      }
      r &= 1;
    }
    if (xe(_e, r), !(t.mode & 1)) t.memoizedState = null;
    else switch (l) {
      case "forwards":
        for (n = t.child, l = null; n !== null; ) e = n.alternate, e !== null && za(e) === null && (l = n), n = n.sibling;
        n = l, n === null ? (l = t.child, t.child = null) : (l = n.sibling, n.sibling = null), Bo(t, false, l, n, a);
        break;
      case "backwards":
        for (n = null, l = t.child, t.child = null; l !== null; ) {
          if (e = l.alternate, e !== null && za(e) === null) {
            t.child = l;
            break;
          }
          e = l.sibling, l.sibling = n, n = l, l = e;
        }
        Bo(t, true, n, null, a);
        break;
      case "together":
        Bo(t, false, null, null, void 0);
        break;
      default:
        t.memoizedState = null;
    }
    return t.child;
  }
  function ha(e, t) {
    !(t.mode & 1) && e !== null && (e.alternate = null, t.alternate = null, t.flags |= 2);
  }
  function en(e, t, n) {
    if (e !== null && (t.dependencies = e.dependencies), Vn |= t.lanes, !(n & t.childLanes)) return null;
    if (e !== null && t.child !== e.child) throw Error(b(153));
    if (t.child !== null) {
      for (e = t.child, n = wn(e, e.pendingProps), t.child = n, n.return = t; e.sibling !== null; ) e = e.sibling, n = n.sibling = wn(e, e.pendingProps), n.return = t;
      n.sibling = null;
    }
    return t.child;
  }
  function $h(e, t, n) {
    switch (t.tag) {
      case 3:
        zf(t), jr();
        break;
      case 5:
        df(t);
        break;
      case 1:
        ct(t.type) && ba(t);
        break;
      case 4:
        Ni(t, t.stateNode.containerInfo);
        break;
      case 10:
        var r = t.type._context, l = t.memoizedProps.value;
        xe(La, r._currentValue), r._currentValue = l;
        break;
      case 13:
        if (r = t.memoizedState, r !== null) return r.dehydrated !== null ? (xe(_e, _e.current & 1), t.flags |= 128, null) : n & t.child.childLanes ? Af(e, t, n) : (xe(_e, _e.current & 1), e = en(e, t, n), e !== null ? e.sibling : null);
        xe(_e, _e.current & 1);
        break;
      case 19:
        if (r = (n & t.childLanes) !== 0, e.flags & 128) {
          if (r) return Uf(e, t, n);
          t.flags |= 128;
        }
        if (l = t.memoizedState, l !== null && (l.rendering = null, l.tail = null, l.lastEffect = null), xe(_e, _e.current), r) break;
        return null;
      case 22:
      case 23:
        return t.lanes = 0, Of(e, t, n);
    }
    return en(e, t, n);
  }
  var $f, Os, Ff, Bf;
  $f = function(e, t) {
    for (var n = t.child; n !== null; ) {
      if (n.tag === 5 || n.tag === 6) e.appendChild(n.stateNode);
      else if (n.tag !== 4 && n.child !== null) {
        n.child.return = n, n = n.child;
        continue;
      }
      if (n === t) break;
      for (; n.sibling === null; ) {
        if (n.return === null || n.return === t) return;
        n = n.return;
      }
      n.sibling.return = n.return, n = n.sibling;
    }
  };
  Os = function() {
  };
  Ff = function(e, t, n, r) {
    var l = e.memoizedProps;
    if (l !== r) {
      e = t.stateNode, zn(Bt.current);
      var a = null;
      switch (n) {
        case "input":
          l = rs(e, l), r = rs(e, r), a = [];
          break;
        case "select":
          l = Re({}, l, {
            value: void 0
          }), r = Re({}, r, {
            value: void 0
          }), a = [];
          break;
        case "textarea":
          l = os(e, l), r = os(e, r), a = [];
          break;
        default:
          typeof l.onClick != "function" && typeof r.onClick == "function" && (e.onclick = Ra);
      }
      is(n, r);
      var o;
      n = null;
      for (c in l) if (!r.hasOwnProperty(c) && l.hasOwnProperty(c) && l[c] != null) if (c === "style") {
        var i = l[c];
        for (o in i) i.hasOwnProperty(o) && (n || (n = {}), n[o] = "");
      } else c !== "dangerouslySetInnerHTML" && c !== "children" && c !== "suppressContentEditableWarning" && c !== "suppressHydrationWarning" && c !== "autoFocus" && (ul.hasOwnProperty(c) ? a || (a = []) : (a = a || []).push(c, null));
      for (c in r) {
        var s = r[c];
        if (i = l == null ? void 0 : l[c], r.hasOwnProperty(c) && s !== i && (s != null || i != null)) if (c === "style") if (i) {
          for (o in i) !i.hasOwnProperty(o) || s && s.hasOwnProperty(o) || (n || (n = {}), n[o] = "");
          for (o in s) s.hasOwnProperty(o) && i[o] !== s[o] && (n || (n = {}), n[o] = s[o]);
        } else n || (a || (a = []), a.push(c, n)), n = s;
        else c === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, i = i ? i.__html : void 0, s != null && i !== s && (a = a || []).push(c, s)) : c === "children" ? typeof s != "string" && typeof s != "number" || (a = a || []).push(c, "" + s) : c !== "suppressContentEditableWarning" && c !== "suppressHydrationWarning" && (ul.hasOwnProperty(c) ? (s != null && c === "onScroll" && ye("scroll", e), a || i === s || (a = [])) : (a = a || []).push(c, s));
      }
      n && (a = a || []).push("style", n);
      var c = a;
      (t.updateQueue = c) && (t.flags |= 4);
    }
  };
  Bf = function(e, t, n, r) {
    n !== r && (t.flags |= 4);
  };
  function Br(e, t) {
    if (!Ee) switch (e.tailMode) {
      case "hidden":
        t = e.tail;
        for (var n = null; t !== null; ) t.alternate !== null && (n = t), t = t.sibling;
        n === null ? e.tail = null : n.sibling = null;
        break;
      case "collapsed":
        n = e.tail;
        for (var r = null; n !== null; ) n.alternate !== null && (r = n), n = n.sibling;
        r === null ? t || e.tail === null ? e.tail = null : e.tail.sibling = null : r.sibling = null;
    }
  }
  function Ze(e) {
    var t = e.alternate !== null && e.alternate.child === e.child, n = 0, r = 0;
    if (t) for (var l = e.child; l !== null; ) n |= l.lanes | l.childLanes, r |= l.subtreeFlags & 14680064, r |= l.flags & 14680064, l.return = e, l = l.sibling;
    else for (l = e.child; l !== null; ) n |= l.lanes | l.childLanes, r |= l.subtreeFlags, r |= l.flags, l.return = e, l = l.sibling;
    return e.subtreeFlags |= r, e.childLanes = n, t;
  }
  function Fh(e, t, n) {
    var r = t.pendingProps;
    switch (gi(t), t.tag) {
      case 2:
      case 16:
      case 15:
      case 0:
      case 11:
      case 7:
      case 8:
      case 12:
      case 9:
      case 14:
        return Ze(t), null;
      case 1:
        return ct(t.type) && Ta(), Ze(t), null;
      case 3:
        return r = t.stateNode, Cr(), we(ut), we(et), Ei(), r.pendingContext && (r.context = r.pendingContext, r.pendingContext = null), (e === null || e.child === null) && (ql(t) ? t.flags |= 4 : e === null || e.memoizedState.isDehydrated && !(t.flags & 256) || (t.flags |= 1024, Dt !== null && (Ws(Dt), Dt = null))), Os(e, t), Ze(t), null;
      case 5:
        ji(t);
        var l = zn(Sl.current);
        if (n = t.type, e !== null && t.stateNode != null) Ff(e, t, n, r, l), e.ref !== t.ref && (t.flags |= 512, t.flags |= 2097152);
        else {
          if (!r) {
            if (t.stateNode === null) throw Error(b(166));
            return Ze(t), null;
          }
          if (e = zn(Bt.current), ql(t)) {
            r = t.stateNode, n = t.type;
            var a = t.memoizedProps;
            switch (r[$t] = t, r[yl] = a, e = (t.mode & 1) !== 0, n) {
              case "dialog":
                ye("cancel", r), ye("close", r);
                break;
              case "iframe":
              case "object":
              case "embed":
                ye("load", r);
                break;
              case "video":
              case "audio":
                for (l = 0; l < Xr.length; l++) ye(Xr[l], r);
                break;
              case "source":
                ye("error", r);
                break;
              case "img":
              case "image":
              case "link":
                ye("error", r), ye("load", r);
                break;
              case "details":
                ye("toggle", r);
                break;
              case "input":
                du(r, a), ye("invalid", r);
                break;
              case "select":
                r._wrapperState = {
                  wasMultiple: !!a.multiple
                }, ye("invalid", r);
                break;
              case "textarea":
                mu(r, a), ye("invalid", r);
            }
            is(n, a), l = null;
            for (var o in a) if (a.hasOwnProperty(o)) {
              var i = a[o];
              o === "children" ? typeof i == "string" ? r.textContent !== i && (a.suppressHydrationWarning !== true && Zl(r.textContent, i, e), l = [
                "children",
                i
              ]) : typeof i == "number" && r.textContent !== "" + i && (a.suppressHydrationWarning !== true && Zl(r.textContent, i, e), l = [
                "children",
                "" + i
              ]) : ul.hasOwnProperty(o) && i != null && o === "onScroll" && ye("scroll", r);
            }
            switch (n) {
              case "input":
                Vl(r), fu(r, a, true);
                break;
              case "textarea":
                Vl(r), pu(r);
                break;
              case "select":
              case "option":
                break;
              default:
                typeof a.onClick == "function" && (r.onclick = Ra);
            }
            r = l, t.updateQueue = r, r !== null && (t.flags |= 4);
          } else {
            o = l.nodeType === 9 ? l : l.ownerDocument, e === "http://www.w3.org/1999/xhtml" && (e = hd(n)), e === "http://www.w3.org/1999/xhtml" ? n === "script" ? (e = o.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild)) : typeof r.is == "string" ? e = o.createElement(n, {
              is: r.is
            }) : (e = o.createElement(n), n === "select" && (o = e, r.multiple ? o.multiple = true : r.size && (o.size = r.size))) : e = o.createElementNS(e, n), e[$t] = t, e[yl] = r, $f(e, t, false, false), t.stateNode = e;
            e: {
              switch (o = us(n, r), n) {
                case "dialog":
                  ye("cancel", e), ye("close", e), l = r;
                  break;
                case "iframe":
                case "object":
                case "embed":
                  ye("load", e), l = r;
                  break;
                case "video":
                case "audio":
                  for (l = 0; l < Xr.length; l++) ye(Xr[l], e);
                  l = r;
                  break;
                case "source":
                  ye("error", e), l = r;
                  break;
                case "img":
                case "image":
                case "link":
                  ye("error", e), ye("load", e), l = r;
                  break;
                case "details":
                  ye("toggle", e), l = r;
                  break;
                case "input":
                  du(e, r), l = rs(e, r), ye("invalid", e);
                  break;
                case "option":
                  l = r;
                  break;
                case "select":
                  e._wrapperState = {
                    wasMultiple: !!r.multiple
                  }, l = Re({}, r, {
                    value: void 0
                  }), ye("invalid", e);
                  break;
                case "textarea":
                  mu(e, r), l = os(e, r), ye("invalid", e);
                  break;
                default:
                  l = r;
              }
              is(n, l), i = l;
              for (a in i) if (i.hasOwnProperty(a)) {
                var s = i[a];
                a === "style" ? xd(e, s) : a === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, s != null && gd(e, s)) : a === "children" ? typeof s == "string" ? (n !== "textarea" || s !== "") && cl(e, s) : typeof s == "number" && cl(e, "" + s) : a !== "suppressContentEditableWarning" && a !== "suppressHydrationWarning" && a !== "autoFocus" && (ul.hasOwnProperty(a) ? s != null && a === "onScroll" && ye("scroll", e) : s != null && ti(e, a, s, o));
              }
              switch (n) {
                case "input":
                  Vl(e), fu(e, r, false);
                  break;
                case "textarea":
                  Vl(e), pu(e);
                  break;
                case "option":
                  r.value != null && e.setAttribute("value", "" + kn(r.value));
                  break;
                case "select":
                  e.multiple = !!r.multiple, a = r.value, a != null ? gr(e, !!r.multiple, a, false) : r.defaultValue != null && gr(e, !!r.multiple, r.defaultValue, true);
                  break;
                default:
                  typeof l.onClick == "function" && (e.onclick = Ra);
              }
              switch (n) {
                case "button":
                case "input":
                case "select":
                case "textarea":
                  r = !!r.autoFocus;
                  break e;
                case "img":
                  r = true;
                  break e;
                default:
                  r = false;
              }
            }
            r && (t.flags |= 4);
          }
          t.ref !== null && (t.flags |= 512, t.flags |= 2097152);
        }
        return Ze(t), null;
      case 6:
        if (e && t.stateNode != null) Bf(e, t, e.memoizedProps, r);
        else {
          if (typeof r != "string" && t.stateNode === null) throw Error(b(166));
          if (n = zn(Sl.current), zn(Bt.current), ql(t)) {
            if (r = t.stateNode, n = t.memoizedProps, r[$t] = t, (a = r.nodeValue !== n) && (e = vt, e !== null)) switch (e.tag) {
              case 3:
                Zl(r.nodeValue, n, (e.mode & 1) !== 0);
                break;
              case 5:
                e.memoizedProps.suppressHydrationWarning !== true && Zl(r.nodeValue, n, (e.mode & 1) !== 0);
            }
            a && (t.flags |= 4);
          } else r = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(r), r[$t] = t, t.stateNode = r;
        }
        return Ze(t), null;
      case 13:
        if (we(_e), r = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
          if (Ee && gt !== null && t.mode & 1 && !(t.flags & 128)) af(), jr(), t.flags |= 98560, a = false;
          else if (a = ql(t), r !== null && r.dehydrated !== null) {
            if (e === null) {
              if (!a) throw Error(b(318));
              if (a = t.memoizedState, a = a !== null ? a.dehydrated : null, !a) throw Error(b(317));
              a[$t] = t;
            } else jr(), !(t.flags & 128) && (t.memoizedState = null), t.flags |= 4;
            Ze(t), a = false;
          } else Dt !== null && (Ws(Dt), Dt = null), a = true;
          if (!a) return t.flags & 65536 ? t : null;
        }
        return t.flags & 128 ? (t.lanes = n, t) : (r = r !== null, r !== (e !== null && e.memoizedState !== null) && r && (t.child.flags |= 8192, t.mode & 1 && (e === null || _e.current & 1 ? Fe === 0 && (Fe = 3) : zi())), t.updateQueue !== null && (t.flags |= 4), Ze(t), null);
      case 4:
        return Cr(), Os(e, t), e === null && vl(t.stateNode.containerInfo), Ze(t), null;
      case 10:
        return wi(t.type._context), Ze(t), null;
      case 17:
        return ct(t.type) && Ta(), Ze(t), null;
      case 19:
        if (we(_e), a = t.memoizedState, a === null) return Ze(t), null;
        if (r = (t.flags & 128) !== 0, o = a.rendering, o === null) if (r) Br(a, false);
        else {
          if (Fe !== 0 || e !== null && e.flags & 128) for (e = t.child; e !== null; ) {
            if (o = za(e), o !== null) {
              for (t.flags |= 128, Br(a, false), r = o.updateQueue, r !== null && (t.updateQueue = r, t.flags |= 4), t.subtreeFlags = 0, r = n, n = t.child; n !== null; ) a = n, e = r, a.flags &= 14680066, o = a.alternate, o === null ? (a.childLanes = 0, a.lanes = e, a.child = null, a.subtreeFlags = 0, a.memoizedProps = null, a.memoizedState = null, a.updateQueue = null, a.dependencies = null, a.stateNode = null) : (a.childLanes = o.childLanes, a.lanes = o.lanes, a.child = o.child, a.subtreeFlags = 0, a.deletions = null, a.memoizedProps = o.memoizedProps, a.memoizedState = o.memoizedState, a.updateQueue = o.updateQueue, a.type = o.type, e = o.dependencies, a.dependencies = e === null ? null : {
                lanes: e.lanes,
                firstContext: e.firstContext
              }), n = n.sibling;
              return xe(_e, _e.current & 1 | 2), t.child;
            }
            e = e.sibling;
          }
          a.tail !== null && Ie() > Pr && (t.flags |= 128, r = true, Br(a, false), t.lanes = 4194304);
        }
        else {
          if (!r) if (e = za(o), e !== null) {
            if (t.flags |= 128, r = true, n = e.updateQueue, n !== null && (t.updateQueue = n, t.flags |= 4), Br(a, true), a.tail === null && a.tailMode === "hidden" && !o.alternate && !Ee) return Ze(t), null;
          } else 2 * Ie() - a.renderingStartTime > Pr && n !== 1073741824 && (t.flags |= 128, r = true, Br(a, false), t.lanes = 4194304);
          a.isBackwards ? (o.sibling = t.child, t.child = o) : (n = a.last, n !== null ? n.sibling = o : t.child = o, a.last = o);
        }
        return a.tail !== null ? (t = a.tail, a.rendering = t, a.tail = t.sibling, a.renderingStartTime = Ie(), t.sibling = null, n = _e.current, xe(_e, r ? n & 1 | 2 : n & 1), t) : (Ze(t), null);
      case 22:
      case 23:
        return Ii(), r = t.memoizedState !== null, e !== null && e.memoizedState !== null !== r && (t.flags |= 8192), r && t.mode & 1 ? pt & 1073741824 && (Ze(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : Ze(t), null;
      case 24:
        return null;
      case 25:
        return null;
    }
    throw Error(b(156, t.tag));
  }
  function Bh(e, t) {
    switch (gi(t), t.tag) {
      case 1:
        return ct(t.type) && Ta(), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 3:
        return Cr(), we(ut), we(et), Ei(), e = t.flags, e & 65536 && !(e & 128) ? (t.flags = e & -65537 | 128, t) : null;
      case 5:
        return ji(t), null;
      case 13:
        if (we(_e), e = t.memoizedState, e !== null && e.dehydrated !== null) {
          if (t.alternate === null) throw Error(b(340));
          jr();
        }
        return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 19:
        return we(_e), null;
      case 4:
        return Cr(), null;
      case 10:
        return wi(t.type._context), null;
      case 22:
      case 23:
        return Ii(), null;
      case 24:
        return null;
      default:
        return null;
    }
  }
  var na = false, qe = false, Wh = typeof WeakSet == "function" ? WeakSet : Set, z = null;
  function pr(e, t) {
    var n = e.ref;
    if (n !== null) if (typeof n == "function") try {
      n(null);
    } catch (r) {
      De(e, t, r);
    }
    else n.current = null;
  }
  function Is(e, t, n) {
    try {
      n();
    } catch (r) {
      De(e, t, r);
    }
  }
  var rc = false;
  function Vh(e, t) {
    if (ys = Ca, e = Qd(), pi(e)) {
      if ("selectionStart" in e) var n = {
        start: e.selectionStart,
        end: e.selectionEnd
      };
      else e: {
        n = (n = e.ownerDocument) && n.defaultView || window;
        var r = n.getSelection && n.getSelection();
        if (r && r.rangeCount !== 0) {
          n = r.anchorNode;
          var l = r.anchorOffset, a = r.focusNode;
          r = r.focusOffset;
          try {
            n.nodeType, a.nodeType;
          } catch {
            n = null;
            break e;
          }
          var o = 0, i = -1, s = -1, c = 0, m = 0, d = e, g = null;
          t: for (; ; ) {
            for (var x; d !== n || l !== 0 && d.nodeType !== 3 || (i = o + l), d !== a || r !== 0 && d.nodeType !== 3 || (s = o + r), d.nodeType === 3 && (o += d.nodeValue.length), (x = d.firstChild) !== null; ) g = d, d = x;
            for (; ; ) {
              if (d === e) break t;
              if (g === n && ++c === l && (i = o), g === a && ++m === r && (s = o), (x = d.nextSibling) !== null) break;
              d = g, g = d.parentNode;
            }
            d = x;
          }
          n = i === -1 || s === -1 ? null : {
            start: i,
            end: s
          };
        } else n = null;
      }
      n = n || {
        start: 0,
        end: 0
      };
    } else n = null;
    for (ws = {
      focusedElem: e,
      selectionRange: n
    }, Ca = false, z = t; z !== null; ) if (t = z, e = t.child, (t.subtreeFlags & 1028) !== 0 && e !== null) e.return = t, z = e;
    else for (; z !== null; ) {
      t = z;
      try {
        var w = t.alternate;
        if (t.flags & 1024) switch (t.tag) {
          case 0:
          case 11:
          case 15:
            break;
          case 1:
            if (w !== null) {
              var S = w.memoizedProps, _ = w.memoizedState, p = t.stateNode, f = p.getSnapshotBeforeUpdate(t.elementType === t.type ? S : Tt(t.type, S), _);
              p.__reactInternalSnapshotBeforeUpdate = f;
            }
            break;
          case 3:
            var h = t.stateNode.containerInfo;
            h.nodeType === 1 ? h.textContent = "" : h.nodeType === 9 && h.documentElement && h.removeChild(h.documentElement);
            break;
          case 5:
          case 6:
          case 4:
          case 17:
            break;
          default:
            throw Error(b(163));
        }
      } catch (j) {
        De(t, t.return, j);
      }
      if (e = t.sibling, e !== null) {
        e.return = t.return, z = e;
        break;
      }
      z = t.return;
    }
    return w = rc, rc = false, w;
  }
  function al(e, t, n) {
    var r = t.updateQueue;
    if (r = r !== null ? r.lastEffect : null, r !== null) {
      var l = r = r.next;
      do {
        if ((l.tag & e) === e) {
          var a = l.destroy;
          l.destroy = void 0, a !== void 0 && Is(t, n, a);
        }
        l = l.next;
      } while (l !== r);
    }
  }
  function lo(e, t) {
    if (t = t.updateQueue, t = t !== null ? t.lastEffect : null, t !== null) {
      var n = t = t.next;
      do {
        if ((n.tag & e) === e) {
          var r = n.create;
          n.destroy = r();
        }
        n = n.next;
      } while (n !== t);
    }
  }
  function zs(e) {
    var t = e.ref;
    if (t !== null) {
      var n = e.stateNode;
      switch (e.tag) {
        case 5:
          e = n;
          break;
        default:
          e = n;
      }
      typeof t == "function" ? t(e) : t.current = e;
    }
  }
  function Wf(e) {
    var t = e.alternate;
    t !== null && (e.alternate = null, Wf(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && (delete t[$t], delete t[yl], delete t[Ns], delete t[Ch], delete t[_h])), e.stateNode = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
  }
  function Vf(e) {
    return e.tag === 5 || e.tag === 3 || e.tag === 4;
  }
  function lc(e) {
    e: for (; ; ) {
      for (; e.sibling === null; ) {
        if (e.return === null || Vf(e.return)) return null;
        e = e.return;
      }
      for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
        if (e.flags & 2 || e.child === null || e.tag === 4) continue e;
        e.child.return = e, e = e.child;
      }
      if (!(e.flags & 2)) return e.stateNode;
    }
  }
  function As(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.nodeType === 8 ? n.parentNode.insertBefore(e, t) : n.insertBefore(e, t) : (n.nodeType === 8 ? (t = n.parentNode, t.insertBefore(e, n)) : (t = n, t.appendChild(e)), n = n._reactRootContainer, n != null || t.onclick !== null || (t.onclick = Ra));
    else if (r !== 4 && (e = e.child, e !== null)) for (As(e, t, n), e = e.sibling; e !== null; ) As(e, t, n), e = e.sibling;
  }
  function Us(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.insertBefore(e, t) : n.appendChild(e);
    else if (r !== 4 && (e = e.child, e !== null)) for (Us(e, t, n), e = e.sibling; e !== null; ) Us(e, t, n), e = e.sibling;
  }
  var Ke = null, bt = false;
  function rn(e, t, n) {
    for (n = n.child; n !== null; ) Hf(e, t, n), n = n.sibling;
  }
  function Hf(e, t, n) {
    if (Ft && typeof Ft.onCommitFiberUnmount == "function") try {
      Ft.onCommitFiberUnmount(Ja, n);
    } catch {
    }
    switch (n.tag) {
      case 5:
        qe || pr(n, t);
      case 6:
        var r = Ke, l = bt;
        Ke = null, rn(e, t, n), Ke = r, bt = l, Ke !== null && (bt ? (e = Ke, n = n.stateNode, e.nodeType === 8 ? e.parentNode.removeChild(n) : e.removeChild(n)) : Ke.removeChild(n.stateNode));
        break;
      case 18:
        Ke !== null && (bt ? (e = Ke, n = n.stateNode, e.nodeType === 8 ? Oo(e.parentNode, n) : e.nodeType === 1 && Oo(e, n), pl(e)) : Oo(Ke, n.stateNode));
        break;
      case 4:
        r = Ke, l = bt, Ke = n.stateNode.containerInfo, bt = true, rn(e, t, n), Ke = r, bt = l;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        if (!qe && (r = n.updateQueue, r !== null && (r = r.lastEffect, r !== null))) {
          l = r = r.next;
          do {
            var a = l, o = a.destroy;
            a = a.tag, o !== void 0 && (a & 2 || a & 4) && Is(n, t, o), l = l.next;
          } while (l !== r);
        }
        rn(e, t, n);
        break;
      case 1:
        if (!qe && (pr(n, t), r = n.stateNode, typeof r.componentWillUnmount == "function")) try {
          r.props = n.memoizedProps, r.state = n.memoizedState, r.componentWillUnmount();
        } catch (i) {
          De(n, t, i);
        }
        rn(e, t, n);
        break;
      case 21:
        rn(e, t, n);
        break;
      case 22:
        n.mode & 1 ? (qe = (r = qe) || n.memoizedState !== null, rn(e, t, n), qe = r) : rn(e, t, n);
        break;
      default:
        rn(e, t, n);
    }
  }
  function ac(e) {
    var t = e.updateQueue;
    if (t !== null) {
      e.updateQueue = null;
      var n = e.stateNode;
      n === null && (n = e.stateNode = new Wh()), t.forEach(function(r) {
        var l = qh.bind(null, e, r);
        n.has(r) || (n.add(r), r.then(l, l));
      });
    }
  }
  function Rt(e, t) {
    var n = t.deletions;
    if (n !== null) for (var r = 0; r < n.length; r++) {
      var l = n[r];
      try {
        var a = e, o = t, i = o;
        e: for (; i !== null; ) {
          switch (i.tag) {
            case 5:
              Ke = i.stateNode, bt = false;
              break e;
            case 3:
              Ke = i.stateNode.containerInfo, bt = true;
              break e;
            case 4:
              Ke = i.stateNode.containerInfo, bt = true;
              break e;
          }
          i = i.return;
        }
        if (Ke === null) throw Error(b(160));
        Hf(a, o, l), Ke = null, bt = false;
        var s = l.alternate;
        s !== null && (s.return = null), l.return = null;
      } catch (c) {
        De(l, t, c);
      }
    }
    if (t.subtreeFlags & 12854) for (t = t.child; t !== null; ) Qf(t, e), t = t.sibling;
  }
  function Qf(e, t) {
    var n = e.alternate, r = e.flags;
    switch (e.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        if (Rt(t, e), At(e), r & 4) {
          try {
            al(3, e, e.return), lo(3, e);
          } catch (S) {
            De(e, e.return, S);
          }
          try {
            al(5, e, e.return);
          } catch (S) {
            De(e, e.return, S);
          }
        }
        break;
      case 1:
        Rt(t, e), At(e), r & 512 && n !== null && pr(n, n.return);
        break;
      case 5:
        if (Rt(t, e), At(e), r & 512 && n !== null && pr(n, n.return), e.flags & 32) {
          var l = e.stateNode;
          try {
            cl(l, "");
          } catch (S) {
            De(e, e.return, S);
          }
        }
        if (r & 4 && (l = e.stateNode, l != null)) {
          var a = e.memoizedProps, o = n !== null ? n.memoizedProps : a, i = e.type, s = e.updateQueue;
          if (e.updateQueue = null, s !== null) try {
            i === "input" && a.type === "radio" && a.name != null && md(l, a), us(i, o);
            var c = us(i, a);
            for (o = 0; o < s.length; o += 2) {
              var m = s[o], d = s[o + 1];
              m === "style" ? xd(l, d) : m === "dangerouslySetInnerHTML" ? gd(l, d) : m === "children" ? cl(l, d) : ti(l, m, d, c);
            }
            switch (i) {
              case "input":
                ls(l, a);
                break;
              case "textarea":
                pd(l, a);
                break;
              case "select":
                var g = l._wrapperState.wasMultiple;
                l._wrapperState.wasMultiple = !!a.multiple;
                var x = a.value;
                x != null ? gr(l, !!a.multiple, x, false) : g !== !!a.multiple && (a.defaultValue != null ? gr(l, !!a.multiple, a.defaultValue, true) : gr(l, !!a.multiple, a.multiple ? [] : "", false));
            }
            l[yl] = a;
          } catch (S) {
            De(e, e.return, S);
          }
        }
        break;
      case 6:
        if (Rt(t, e), At(e), r & 4) {
          if (e.stateNode === null) throw Error(b(162));
          l = e.stateNode, a = e.memoizedProps;
          try {
            l.nodeValue = a;
          } catch (S) {
            De(e, e.return, S);
          }
        }
        break;
      case 3:
        if (Rt(t, e), At(e), r & 4 && n !== null && n.memoizedState.isDehydrated) try {
          pl(t.containerInfo);
        } catch (S) {
          De(e, e.return, S);
        }
        break;
      case 4:
        Rt(t, e), At(e);
        break;
      case 13:
        Rt(t, e), At(e), l = e.child, l.flags & 8192 && (a = l.memoizedState !== null, l.stateNode.isHidden = a, !a || l.alternate !== null && l.alternate.memoizedState !== null || (Li = Ie())), r & 4 && ac(e);
        break;
      case 22:
        if (m = n !== null && n.memoizedState !== null, e.mode & 1 ? (qe = (c = qe) || m, Rt(t, e), qe = c) : Rt(t, e), At(e), r & 8192) {
          if (c = e.memoizedState !== null, (e.stateNode.isHidden = c) && !m && e.mode & 1) for (z = e, m = e.child; m !== null; ) {
            for (d = z = m; z !== null; ) {
              switch (g = z, x = g.child, g.tag) {
                case 0:
                case 11:
                case 14:
                case 15:
                  al(4, g, g.return);
                  break;
                case 1:
                  pr(g, g.return);
                  var w = g.stateNode;
                  if (typeof w.componentWillUnmount == "function") {
                    r = g, n = g.return;
                    try {
                      t = r, w.props = t.memoizedProps, w.state = t.memoizedState, w.componentWillUnmount();
                    } catch (S) {
                      De(r, n, S);
                    }
                  }
                  break;
                case 5:
                  pr(g, g.return);
                  break;
                case 22:
                  if (g.memoizedState !== null) {
                    sc(d);
                    continue;
                  }
              }
              x !== null ? (x.return = g, z = x) : sc(d);
            }
            m = m.sibling;
          }
          e: for (m = null, d = e; ; ) {
            if (d.tag === 5) {
              if (m === null) {
                m = d;
                try {
                  l = d.stateNode, c ? (a = l.style, typeof a.setProperty == "function" ? a.setProperty("display", "none", "important") : a.display = "none") : (i = d.stateNode, s = d.memoizedProps.style, o = s != null && s.hasOwnProperty("display") ? s.display : null, i.style.display = vd("display", o));
                } catch (S) {
                  De(e, e.return, S);
                }
              }
            } else if (d.tag === 6) {
              if (m === null) try {
                d.stateNode.nodeValue = c ? "" : d.memoizedProps;
              } catch (S) {
                De(e, e.return, S);
              }
            } else if ((d.tag !== 22 && d.tag !== 23 || d.memoizedState === null || d === e) && d.child !== null) {
              d.child.return = d, d = d.child;
              continue;
            }
            if (d === e) break e;
            for (; d.sibling === null; ) {
              if (d.return === null || d.return === e) break e;
              m === d && (m = null), d = d.return;
            }
            m === d && (m = null), d.sibling.return = d.return, d = d.sibling;
          }
        }
        break;
      case 19:
        Rt(t, e), At(e), r & 4 && ac(e);
        break;
      case 21:
        break;
      default:
        Rt(t, e), At(e);
    }
  }
  function At(e) {
    var t = e.flags;
    if (t & 2) {
      try {
        e: {
          for (var n = e.return; n !== null; ) {
            if (Vf(n)) {
              var r = n;
              break e;
            }
            n = n.return;
          }
          throw Error(b(160));
        }
        switch (r.tag) {
          case 5:
            var l = r.stateNode;
            r.flags & 32 && (cl(l, ""), r.flags &= -33);
            var a = lc(e);
            Us(e, a, l);
            break;
          case 3:
          case 4:
            var o = r.stateNode.containerInfo, i = lc(e);
            As(e, i, o);
            break;
          default:
            throw Error(b(161));
        }
      } catch (s) {
        De(e, e.return, s);
      }
      e.flags &= -3;
    }
    t & 4096 && (e.flags &= -4097);
  }
  function Hh(e, t, n) {
    z = e, Kf(e);
  }
  function Kf(e, t, n) {
    for (var r = (e.mode & 1) !== 0; z !== null; ) {
      var l = z, a = l.child;
      if (l.tag === 22 && r) {
        var o = l.memoizedState !== null || na;
        if (!o) {
          var i = l.alternate, s = i !== null && i.memoizedState !== null || qe;
          i = na;
          var c = qe;
          if (na = o, (qe = s) && !c) for (z = l; z !== null; ) o = z, s = o.child, o.tag === 22 && o.memoizedState !== null ? ic(l) : s !== null ? (s.return = o, z = s) : ic(l);
          for (; a !== null; ) z = a, Kf(a), a = a.sibling;
          z = l, na = i, qe = c;
        }
        oc(e);
      } else l.subtreeFlags & 8772 && a !== null ? (a.return = l, z = a) : oc(e);
    }
  }
  function oc(e) {
    for (; z !== null; ) {
      var t = z;
      if (t.flags & 8772) {
        var n = t.alternate;
        try {
          if (t.flags & 8772) switch (t.tag) {
            case 0:
            case 11:
            case 15:
              qe || lo(5, t);
              break;
            case 1:
              var r = t.stateNode;
              if (t.flags & 4 && !qe) if (n === null) r.componentDidMount();
              else {
                var l = t.elementType === t.type ? n.memoizedProps : Tt(t.type, n.memoizedProps);
                r.componentDidUpdate(l, n.memoizedState, r.__reactInternalSnapshotBeforeUpdate);
              }
              var a = t.updateQueue;
              a !== null && Vu(t, a, r);
              break;
            case 3:
              var o = t.updateQueue;
              if (o !== null) {
                if (n = null, t.child !== null) switch (t.child.tag) {
                  case 5:
                    n = t.child.stateNode;
                    break;
                  case 1:
                    n = t.child.stateNode;
                }
                Vu(t, o, n);
              }
              break;
            case 5:
              var i = t.stateNode;
              if (n === null && t.flags & 4) {
                n = i;
                var s = t.memoizedProps;
                switch (t.type) {
                  case "button":
                  case "input":
                  case "select":
                  case "textarea":
                    s.autoFocus && n.focus();
                    break;
                  case "img":
                    s.src && (n.src = s.src);
                }
              }
              break;
            case 6:
              break;
            case 4:
              break;
            case 12:
              break;
            case 13:
              if (t.memoizedState === null) {
                var c = t.alternate;
                if (c !== null) {
                  var m = c.memoizedState;
                  if (m !== null) {
                    var d = m.dehydrated;
                    d !== null && pl(d);
                  }
                }
              }
              break;
            case 19:
            case 17:
            case 21:
            case 22:
            case 23:
            case 25:
              break;
            default:
              throw Error(b(163));
          }
          qe || t.flags & 512 && zs(t);
        } catch (g) {
          De(t, t.return, g);
        }
      }
      if (t === e) {
        z = null;
        break;
      }
      if (n = t.sibling, n !== null) {
        n.return = t.return, z = n;
        break;
      }
      z = t.return;
    }
  }
  function sc(e) {
    for (; z !== null; ) {
      var t = z;
      if (t === e) {
        z = null;
        break;
      }
      var n = t.sibling;
      if (n !== null) {
        n.return = t.return, z = n;
        break;
      }
      z = t.return;
    }
  }
  function ic(e) {
    for (; z !== null; ) {
      var t = z;
      try {
        switch (t.tag) {
          case 0:
          case 11:
          case 15:
            var n = t.return;
            try {
              lo(4, t);
            } catch (s) {
              De(t, n, s);
            }
            break;
          case 1:
            var r = t.stateNode;
            if (typeof r.componentDidMount == "function") {
              var l = t.return;
              try {
                r.componentDidMount();
              } catch (s) {
                De(t, l, s);
              }
            }
            var a = t.return;
            try {
              zs(t);
            } catch (s) {
              De(t, a, s);
            }
            break;
          case 5:
            var o = t.return;
            try {
              zs(t);
            } catch (s) {
              De(t, o, s);
            }
        }
      } catch (s) {
        De(t, t.return, s);
      }
      if (t === e) {
        z = null;
        break;
      }
      var i = t.sibling;
      if (i !== null) {
        i.return = t.return, z = i;
        break;
      }
      z = t.return;
    }
  }
  var Qh = Math.ceil, $a = tn.ReactCurrentDispatcher, Mi = tn.ReactCurrentOwner, jt = tn.ReactCurrentBatchConfig, ie = 0, We = null, Ue = null, Ge = 0, pt = 0, hr = En(0), Fe = 0, El = null, Vn = 0, ao = 0, Di = 0, ol = null, st = null, Li = 0, Pr = 1 / 0, Qt = null, Fa = false, $s = null, xn = null, ra = false, dn = null, Ba = 0, sl = 0, Fs = null, ga = -1, va = 0;
  function rt() {
    return ie & 6 ? Ie() : ga !== -1 ? ga : ga = Ie();
  }
  function yn(e) {
    return e.mode & 1 ? ie & 2 && Ge !== 0 ? Ge & -Ge : Rh.transition !== null ? (va === 0 && (va = Td()), va) : (e = he, e !== 0 || (e = window.event, e = e === void 0 ? 16 : zd(e.type)), e) : 1;
  }
  function Ot(e, t, n, r) {
    if (50 < sl) throw sl = 0, Fs = null, Error(b(185));
    Rl(e, n, r), (!(ie & 2) || e !== We) && (e === We && (!(ie & 2) && (ao |= n), Fe === 4 && un(e, Ge)), dt(e, r), n === 1 && ie === 0 && !(t.mode & 1) && (Pr = Ie() + 500, to && Cn()));
  }
  function dt(e, t) {
    var n = e.callbackNode;
    Rp(e, t);
    var r = Ea(e, e === We ? Ge : 0);
    if (r === 0) n !== null && vu(n), e.callbackNode = null, e.callbackPriority = 0;
    else if (t = r & -r, e.callbackPriority !== t) {
      if (n != null && vu(n), t === 1) e.tag === 0 ? Ph(uc.bind(null, e)) : nf(uc.bind(null, e)), jh(function() {
        !(ie & 6) && Cn();
      }), n = null;
      else {
        switch (bd(r)) {
          case 1:
            n = oi;
            break;
          case 4:
            n = Pd;
            break;
          case 16:
            n = ja;
            break;
          case 536870912:
            n = Rd;
            break;
          default:
            n = ja;
        }
        n = tm(n, Gf.bind(null, e));
      }
      e.callbackPriority = t, e.callbackNode = n;
    }
  }
  function Gf(e, t) {
    if (ga = -1, va = 0, ie & 6) throw Error(b(327));
    var n = e.callbackNode;
    if (Sr() && e.callbackNode !== n) return null;
    var r = Ea(e, e === We ? Ge : 0);
    if (r === 0) return null;
    if (r & 30 || r & e.expiredLanes || t) t = Wa(e, r);
    else {
      t = r;
      var l = ie;
      ie |= 2;
      var a = Jf();
      (We !== e || Ge !== t) && (Qt = null, Pr = Ie() + 500, Un(e, t));
      do
        try {
          Yh();
          break;
        } catch (i) {
          Yf(e, i);
        }
      while (true);
      yi(), $a.current = a, ie = l, Ue !== null ? t = 0 : (We = null, Ge = 0, t = Fe);
    }
    if (t !== 0) {
      if (t === 2 && (l = ps(e), l !== 0 && (r = l, t = Bs(e, l))), t === 1) throw n = El, Un(e, 0), un(e, r), dt(e, Ie()), n;
      if (t === 6) un(e, r);
      else {
        if (l = e.current.alternate, !(r & 30) && !Kh(l) && (t = Wa(e, r), t === 2 && (a = ps(e), a !== 0 && (r = a, t = Bs(e, a))), t === 1)) throw n = El, Un(e, 0), un(e, r), dt(e, Ie()), n;
        switch (e.finishedWork = l, e.finishedLanes = r, t) {
          case 0:
          case 1:
            throw Error(b(345));
          case 2:
            Mn(e, st, Qt);
            break;
          case 3:
            if (un(e, r), (r & 130023424) === r && (t = Li + 500 - Ie(), 10 < t)) {
              if (Ea(e, 0) !== 0) break;
              if (l = e.suspendedLanes, (l & r) !== r) {
                rt(), e.pingedLanes |= e.suspendedLanes & l;
                break;
              }
              e.timeoutHandle = ks(Mn.bind(null, e, st, Qt), t);
              break;
            }
            Mn(e, st, Qt);
            break;
          case 4:
            if (un(e, r), (r & 4194240) === r) break;
            for (t = e.eventTimes, l = -1; 0 < r; ) {
              var o = 31 - Lt(r);
              a = 1 << o, o = t[o], o > l && (l = o), r &= ~a;
            }
            if (r = l, r = Ie() - r, r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * Qh(r / 1960)) - r, 10 < r) {
              e.timeoutHandle = ks(Mn.bind(null, e, st, Qt), r);
              break;
            }
            Mn(e, st, Qt);
            break;
          case 5:
            Mn(e, st, Qt);
            break;
          default:
            throw Error(b(329));
        }
      }
    }
    return dt(e, Ie()), e.callbackNode === n ? Gf.bind(null, e) : null;
  }
  function Bs(e, t) {
    var n = ol;
    return e.current.memoizedState.isDehydrated && (Un(e, t).flags |= 256), e = Wa(e, t), e !== 2 && (t = st, st = n, t !== null && Ws(t)), e;
  }
  function Ws(e) {
    st === null ? st = e : st.push.apply(st, e);
  }
  function Kh(e) {
    for (var t = e; ; ) {
      if (t.flags & 16384) {
        var n = t.updateQueue;
        if (n !== null && (n = n.stores, n !== null)) for (var r = 0; r < n.length; r++) {
          var l = n[r], a = l.getSnapshot;
          l = l.value;
          try {
            if (!It(a(), l)) return false;
          } catch {
            return false;
          }
        }
      }
      if (n = t.child, t.subtreeFlags & 16384 && n !== null) n.return = t, t = n;
      else {
        if (t === e) break;
        for (; t.sibling === null; ) {
          if (t.return === null || t.return === e) return true;
          t = t.return;
        }
        t.sibling.return = t.return, t = t.sibling;
      }
    }
    return true;
  }
  function un(e, t) {
    for (t &= ~Di, t &= ~ao, e.suspendedLanes |= t, e.pingedLanes &= ~t, e = e.expirationTimes; 0 < t; ) {
      var n = 31 - Lt(t), r = 1 << n;
      e[n] = -1, t &= ~r;
    }
  }
  function uc(e) {
    if (ie & 6) throw Error(b(327));
    Sr();
    var t = Ea(e, 0);
    if (!(t & 1)) return dt(e, Ie()), null;
    var n = Wa(e, t);
    if (e.tag !== 0 && n === 2) {
      var r = ps(e);
      r !== 0 && (t = r, n = Bs(e, r));
    }
    if (n === 1) throw n = El, Un(e, 0), un(e, t), dt(e, Ie()), n;
    if (n === 6) throw Error(b(345));
    return e.finishedWork = e.current.alternate, e.finishedLanes = t, Mn(e, st, Qt), dt(e, Ie()), null;
  }
  function Oi(e, t) {
    var n = ie;
    ie |= 1;
    try {
      return e(t);
    } finally {
      ie = n, ie === 0 && (Pr = Ie() + 500, to && Cn());
    }
  }
  function Hn(e) {
    dn !== null && dn.tag === 0 && !(ie & 6) && Sr();
    var t = ie;
    ie |= 1;
    var n = jt.transition, r = he;
    try {
      if (jt.transition = null, he = 1, e) return e();
    } finally {
      he = r, jt.transition = n, ie = t, !(ie & 6) && Cn();
    }
  }
  function Ii() {
    pt = hr.current, we(hr);
  }
  function Un(e, t) {
    e.finishedWork = null, e.finishedLanes = 0;
    var n = e.timeoutHandle;
    if (n !== -1 && (e.timeoutHandle = -1, Nh(n)), Ue !== null) for (n = Ue.return; n !== null; ) {
      var r = n;
      switch (gi(r), r.tag) {
        case 1:
          r = r.type.childContextTypes, r != null && Ta();
          break;
        case 3:
          Cr(), we(ut), we(et), Ei();
          break;
        case 5:
          ji(r);
          break;
        case 4:
          Cr();
          break;
        case 13:
          we(_e);
          break;
        case 19:
          we(_e);
          break;
        case 10:
          wi(r.type._context);
          break;
        case 22:
        case 23:
          Ii();
      }
      n = n.return;
    }
    if (We = e, Ue = e = wn(e.current, null), Ge = pt = t, Fe = 0, El = null, Di = ao = Vn = 0, st = ol = null, In !== null) {
      for (t = 0; t < In.length; t++) if (n = In[t], r = n.interleaved, r !== null) {
        n.interleaved = null;
        var l = r.next, a = n.pending;
        if (a !== null) {
          var o = a.next;
          a.next = l, r.next = o;
        }
        n.pending = r;
      }
      In = null;
    }
    return e;
  }
  function Yf(e, t) {
    do {
      var n = Ue;
      try {
        if (yi(), ma.current = Ua, Aa) {
          for (var r = Pe.memoizedState; r !== null; ) {
            var l = r.queue;
            l !== null && (l.pending = null), r = r.next;
          }
          Aa = false;
        }
        if (Wn = 0, Be = $e = Pe = null, ll = false, kl = 0, Mi.current = null, n === null || n.return === null) {
          Fe = 1, El = t, Ue = null;
          break;
        }
        e: {
          var a = e, o = n.return, i = n, s = t;
          if (t = Ge, i.flags |= 32768, s !== null && typeof s == "object" && typeof s.then == "function") {
            var c = s, m = i, d = m.tag;
            if (!(m.mode & 1) && (d === 0 || d === 11 || d === 15)) {
              var g = m.alternate;
              g ? (m.updateQueue = g.updateQueue, m.memoizedState = g.memoizedState, m.lanes = g.lanes) : (m.updateQueue = null, m.memoizedState = null);
            }
            var x = Ju(o);
            if (x !== null) {
              x.flags &= -257, Xu(x, o, i, a, t), x.mode & 1 && Yu(a, c, t), t = x, s = c;
              var w = t.updateQueue;
              if (w === null) {
                var S = /* @__PURE__ */ new Set();
                S.add(s), t.updateQueue = S;
              } else w.add(s);
              break e;
            } else {
              if (!(t & 1)) {
                Yu(a, c, t), zi();
                break e;
              }
              s = Error(b(426));
            }
          } else if (Ee && i.mode & 1) {
            var _ = Ju(o);
            if (_ !== null) {
              !(_.flags & 65536) && (_.flags |= 256), Xu(_, o, i, a, t), vi(_r(s, i));
              break e;
            }
          }
          a = s = _r(s, i), Fe !== 4 && (Fe = 2), ol === null ? ol = [
            a
          ] : ol.push(a), a = o;
          do {
            switch (a.tag) {
              case 3:
                a.flags |= 65536, t &= -t, a.lanes |= t;
                var p = Mf(a, s, t);
                Wu(a, p);
                break e;
              case 1:
                i = s;
                var f = a.type, h = a.stateNode;
                if (!(a.flags & 128) && (typeof f.getDerivedStateFromError == "function" || h !== null && typeof h.componentDidCatch == "function" && (xn === null || !xn.has(h)))) {
                  a.flags |= 65536, t &= -t, a.lanes |= t;
                  var j = Df(a, i, t);
                  Wu(a, j);
                  break e;
                }
            }
            a = a.return;
          } while (a !== null);
        }
        Zf(n);
      } catch (C) {
        t = C, Ue === n && n !== null && (Ue = n = n.return);
        continue;
      }
      break;
    } while (true);
  }
  function Jf() {
    var e = $a.current;
    return $a.current = Ua, e === null ? Ua : e;
  }
  function zi() {
    (Fe === 0 || Fe === 3 || Fe === 2) && (Fe = 4), We === null || !(Vn & 268435455) && !(ao & 268435455) || un(We, Ge);
  }
  function Wa(e, t) {
    var n = ie;
    ie |= 2;
    var r = Jf();
    (We !== e || Ge !== t) && (Qt = null, Un(e, t));
    do
      try {
        Gh();
        break;
      } catch (l) {
        Yf(e, l);
      }
    while (true);
    if (yi(), ie = n, $a.current = r, Ue !== null) throw Error(b(261));
    return We = null, Ge = 0, Fe;
  }
  function Gh() {
    for (; Ue !== null; ) Xf(Ue);
  }
  function Yh() {
    for (; Ue !== null && !wp(); ) Xf(Ue);
  }
  function Xf(e) {
    var t = em(e.alternate, e, pt);
    e.memoizedProps = e.pendingProps, t === null ? Zf(e) : Ue = t, Mi.current = null;
  }
  function Zf(e) {
    var t = e;
    do {
      var n = t.alternate;
      if (e = t.return, t.flags & 32768) {
        if (n = Bh(n, t), n !== null) {
          n.flags &= 32767, Ue = n;
          return;
        }
        if (e !== null) e.flags |= 32768, e.subtreeFlags = 0, e.deletions = null;
        else {
          Fe = 6, Ue = null;
          return;
        }
      } else if (n = Fh(n, t, pt), n !== null) {
        Ue = n;
        return;
      }
      if (t = t.sibling, t !== null) {
        Ue = t;
        return;
      }
      Ue = t = e;
    } while (t !== null);
    Fe === 0 && (Fe = 5);
  }
  function Mn(e, t, n) {
    var r = he, l = jt.transition;
    try {
      jt.transition = null, he = 1, Jh(e, t, n, r);
    } finally {
      jt.transition = l, he = r;
    }
    return null;
  }
  function Jh(e, t, n, r) {
    do
      Sr();
    while (dn !== null);
    if (ie & 6) throw Error(b(327));
    n = e.finishedWork;
    var l = e.finishedLanes;
    if (n === null) return null;
    if (e.finishedWork = null, e.finishedLanes = 0, n === e.current) throw Error(b(177));
    e.callbackNode = null, e.callbackPriority = 0;
    var a = n.lanes | n.childLanes;
    if (Tp(e, a), e === We && (Ue = We = null, Ge = 0), !(n.subtreeFlags & 2064) && !(n.flags & 2064) || ra || (ra = true, tm(ja, function() {
      return Sr(), null;
    })), a = (n.flags & 15990) !== 0, n.subtreeFlags & 15990 || a) {
      a = jt.transition, jt.transition = null;
      var o = he;
      he = 1;
      var i = ie;
      ie |= 4, Mi.current = null, Vh(e, n), Qf(n, e), gh(ws), Ca = !!ys, ws = ys = null, e.current = n, Hh(n), Sp(), ie = i, he = o, jt.transition = a;
    } else e.current = n;
    if (ra && (ra = false, dn = e, Ba = l), a = e.pendingLanes, a === 0 && (xn = null), jp(n.stateNode), dt(e, Ie()), t !== null) for (r = e.onRecoverableError, n = 0; n < t.length; n++) l = t[n], r(l.value, {
      componentStack: l.stack,
      digest: l.digest
    });
    if (Fa) throw Fa = false, e = $s, $s = null, e;
    return Ba & 1 && e.tag !== 0 && Sr(), a = e.pendingLanes, a & 1 ? e === Fs ? sl++ : (sl = 0, Fs = e) : sl = 0, Cn(), null;
  }
  function Sr() {
    if (dn !== null) {
      var e = bd(Ba), t = jt.transition, n = he;
      try {
        if (jt.transition = null, he = 16 > e ? 16 : e, dn === null) var r = false;
        else {
          if (e = dn, dn = null, Ba = 0, ie & 6) throw Error(b(331));
          var l = ie;
          for (ie |= 4, z = e.current; z !== null; ) {
            var a = z, o = a.child;
            if (z.flags & 16) {
              var i = a.deletions;
              if (i !== null) {
                for (var s = 0; s < i.length; s++) {
                  var c = i[s];
                  for (z = c; z !== null; ) {
                    var m = z;
                    switch (m.tag) {
                      case 0:
                      case 11:
                      case 15:
                        al(8, m, a);
                    }
                    var d = m.child;
                    if (d !== null) d.return = m, z = d;
                    else for (; z !== null; ) {
                      m = z;
                      var g = m.sibling, x = m.return;
                      if (Wf(m), m === c) {
                        z = null;
                        break;
                      }
                      if (g !== null) {
                        g.return = x, z = g;
                        break;
                      }
                      z = x;
                    }
                  }
                }
                var w = a.alternate;
                if (w !== null) {
                  var S = w.child;
                  if (S !== null) {
                    w.child = null;
                    do {
                      var _ = S.sibling;
                      S.sibling = null, S = _;
                    } while (S !== null);
                  }
                }
                z = a;
              }
            }
            if (a.subtreeFlags & 2064 && o !== null) o.return = a, z = o;
            else e: for (; z !== null; ) {
              if (a = z, a.flags & 2048) switch (a.tag) {
                case 0:
                case 11:
                case 15:
                  al(9, a, a.return);
              }
              var p = a.sibling;
              if (p !== null) {
                p.return = a.return, z = p;
                break e;
              }
              z = a.return;
            }
          }
          var f = e.current;
          for (z = f; z !== null; ) {
            o = z;
            var h = o.child;
            if (o.subtreeFlags & 2064 && h !== null) h.return = o, z = h;
            else e: for (o = f; z !== null; ) {
              if (i = z, i.flags & 2048) try {
                switch (i.tag) {
                  case 0:
                  case 11:
                  case 15:
                    lo(9, i);
                }
              } catch (C) {
                De(i, i.return, C);
              }
              if (i === o) {
                z = null;
                break e;
              }
              var j = i.sibling;
              if (j !== null) {
                j.return = i.return, z = j;
                break e;
              }
              z = i.return;
            }
          }
          if (ie = l, Cn(), Ft && typeof Ft.onPostCommitFiberRoot == "function") try {
            Ft.onPostCommitFiberRoot(Ja, e);
          } catch {
          }
          r = true;
        }
        return r;
      } finally {
        he = n, jt.transition = t;
      }
    }
    return false;
  }
  function cc(e, t, n) {
    t = _r(n, t), t = Mf(e, t, 1), e = vn(e, t, 1), t = rt(), e !== null && (Rl(e, 1, t), dt(e, t));
  }
  function De(e, t, n) {
    if (e.tag === 3) cc(e, e, n);
    else for (; t !== null; ) {
      if (t.tag === 3) {
        cc(t, e, n);
        break;
      } else if (t.tag === 1) {
        var r = t.stateNode;
        if (typeof t.type.getDerivedStateFromError == "function" || typeof r.componentDidCatch == "function" && (xn === null || !xn.has(r))) {
          e = _r(n, e), e = Df(t, e, 1), t = vn(t, e, 1), e = rt(), t !== null && (Rl(t, 1, e), dt(t, e));
          break;
        }
      }
      t = t.return;
    }
  }
  function Xh(e, t, n) {
    var r = e.pingCache;
    r !== null && r.delete(t), t = rt(), e.pingedLanes |= e.suspendedLanes & n, We === e && (Ge & n) === n && (Fe === 4 || Fe === 3 && (Ge & 130023424) === Ge && 500 > Ie() - Li ? Un(e, 0) : Di |= n), dt(e, t);
  }
  function qf(e, t) {
    t === 0 && (e.mode & 1 ? (t = Kl, Kl <<= 1, !(Kl & 130023424) && (Kl = 4194304)) : t = 1);
    var n = rt();
    e = qt(e, t), e !== null && (Rl(e, t, n), dt(e, n));
  }
  function Zh(e) {
    var t = e.memoizedState, n = 0;
    t !== null && (n = t.retryLane), qf(e, n);
  }
  function qh(e, t) {
    var n = 0;
    switch (e.tag) {
      case 13:
        var r = e.stateNode, l = e.memoizedState;
        l !== null && (n = l.retryLane);
        break;
      case 19:
        r = e.stateNode;
        break;
      default:
        throw Error(b(314));
    }
    r !== null && r.delete(t), qf(e, n);
  }
  var em;
  em = function(e, t, n) {
    if (e !== null) if (e.memoizedProps !== t.pendingProps || ut.current) it = true;
    else {
      if (!(e.lanes & n) && !(t.flags & 128)) return it = false, $h(e, t, n);
      it = !!(e.flags & 131072);
    }
    else it = false, Ee && t.flags & 1048576 && rf(t, Da, t.index);
    switch (t.lanes = 0, t.tag) {
      case 2:
        var r = t.type;
        ha(e, t), e = t.pendingProps;
        var l = Nr(t, et.current);
        wr(t, n), l = _i(null, t, r, e, l, n);
        var a = Pi();
        return t.flags |= 1, typeof l == "object" && l !== null && typeof l.render == "function" && l.$$typeof === void 0 ? (t.tag = 1, t.memoizedState = null, t.updateQueue = null, ct(r) ? (a = true, ba(t)) : a = false, t.memoizedState = l.state !== null && l.state !== void 0 ? l.state : null, ki(t), l.updater = ro, t.stateNode = l, l._reactInternals = t, Rs(t, r, e, n), t = Ms(null, t, r, true, a, n)) : (t.tag = 0, Ee && a && hi(t), nt(null, t, l, n), t = t.child), t;
      case 16:
        r = t.elementType;
        e: {
          switch (ha(e, t), e = t.pendingProps, l = r._init, r = l(r._payload), t.type = r, l = t.tag = tg(r), e = Tt(r, e), l) {
            case 0:
              t = bs(null, t, r, e, n);
              break e;
            case 1:
              t = ec(null, t, r, e, n);
              break e;
            case 11:
              t = Zu(null, t, r, e, n);
              break e;
            case 14:
              t = qu(null, t, r, Tt(r.type, e), n);
              break e;
          }
          throw Error(b(306, r, ""));
        }
        return t;
      case 0:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Tt(r, l), bs(e, t, r, l, n);
      case 1:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Tt(r, l), ec(e, t, r, l, n);
      case 3:
        e: {
          if (zf(t), e === null) throw Error(b(387));
          r = t.pendingProps, a = t.memoizedState, l = a.element, cf(e, t), Ia(t, r, null, n);
          var o = t.memoizedState;
          if (r = o.element, a.isDehydrated) if (a = {
            element: r,
            isDehydrated: false,
            cache: o.cache,
            pendingSuspenseBoundaries: o.pendingSuspenseBoundaries,
            transitions: o.transitions
          }, t.updateQueue.baseState = a, t.memoizedState = a, t.flags & 256) {
            l = _r(Error(b(423)), t), t = tc(e, t, r, n, l);
            break e;
          } else if (r !== l) {
            l = _r(Error(b(424)), t), t = tc(e, t, r, n, l);
            break e;
          } else for (gt = gn(t.stateNode.containerInfo.firstChild), vt = t, Ee = true, Dt = null, n = sf(t, null, r, n), t.child = n; n; ) n.flags = n.flags & -3 | 4096, n = n.sibling;
          else {
            if (jr(), r === l) {
              t = en(e, t, n);
              break e;
            }
            nt(e, t, r, n);
          }
          t = t.child;
        }
        return t;
      case 5:
        return df(t), e === null && Cs(t), r = t.type, l = t.pendingProps, a = e !== null ? e.memoizedProps : null, o = l.children, Ss(r, l) ? o = null : a !== null && Ss(r, a) && (t.flags |= 32), If(e, t), nt(e, t, o, n), t.child;
      case 6:
        return e === null && Cs(t), null;
      case 13:
        return Af(e, t, n);
      case 4:
        return Ni(t, t.stateNode.containerInfo), r = t.pendingProps, e === null ? t.child = Er(t, null, r, n) : nt(e, t, r, n), t.child;
      case 11:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Tt(r, l), Zu(e, t, r, l, n);
      case 7:
        return nt(e, t, t.pendingProps, n), t.child;
      case 8:
        return nt(e, t, t.pendingProps.children, n), t.child;
      case 12:
        return nt(e, t, t.pendingProps.children, n), t.child;
      case 10:
        e: {
          if (r = t.type._context, l = t.pendingProps, a = t.memoizedProps, o = l.value, xe(La, r._currentValue), r._currentValue = o, a !== null) if (It(a.value, o)) {
            if (a.children === l.children && !ut.current) {
              t = en(e, t, n);
              break e;
            }
          } else for (a = t.child, a !== null && (a.return = t); a !== null; ) {
            var i = a.dependencies;
            if (i !== null) {
              o = a.child;
              for (var s = i.firstContext; s !== null; ) {
                if (s.context === r) {
                  if (a.tag === 1) {
                    s = Jt(-1, n & -n), s.tag = 2;
                    var c = a.updateQueue;
                    if (c !== null) {
                      c = c.shared;
                      var m = c.pending;
                      m === null ? s.next = s : (s.next = m.next, m.next = s), c.pending = s;
                    }
                  }
                  a.lanes |= n, s = a.alternate, s !== null && (s.lanes |= n), _s(a.return, n, t), i.lanes |= n;
                  break;
                }
                s = s.next;
              }
            } else if (a.tag === 10) o = a.type === t.type ? null : a.child;
            else if (a.tag === 18) {
              if (o = a.return, o === null) throw Error(b(341));
              o.lanes |= n, i = o.alternate, i !== null && (i.lanes |= n), _s(o, n, t), o = a.sibling;
            } else o = a.child;
            if (o !== null) o.return = a;
            else for (o = a; o !== null; ) {
              if (o === t) {
                o = null;
                break;
              }
              if (a = o.sibling, a !== null) {
                a.return = o.return, o = a;
                break;
              }
              o = o.return;
            }
            a = o;
          }
          nt(e, t, l.children, n), t = t.child;
        }
        return t;
      case 9:
        return l = t.type, r = t.pendingProps.children, wr(t, n), l = Et(l), r = r(l), t.flags |= 1, nt(e, t, r, n), t.child;
      case 14:
        return r = t.type, l = Tt(r, t.pendingProps), l = Tt(r.type, l), qu(e, t, r, l, n);
      case 15:
        return Lf(e, t, t.type, t.pendingProps, n);
      case 17:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Tt(r, l), ha(e, t), t.tag = 1, ct(r) ? (e = true, ba(t)) : e = false, wr(t, n), bf(t, r, l), Rs(t, r, l, n), Ms(null, t, r, true, e, n);
      case 19:
        return Uf(e, t, n);
      case 22:
        return Of(e, t, n);
    }
    throw Error(b(156, t.tag));
  };
  function tm(e, t) {
    return _d(e, t);
  }
  function eg(e, t, n, r) {
    this.tag = e, this.key = n, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = r, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function Nt(e, t, n, r) {
    return new eg(e, t, n, r);
  }
  function Ai(e) {
    return e = e.prototype, !(!e || !e.isReactComponent);
  }
  function tg(e) {
    if (typeof e == "function") return Ai(e) ? 1 : 0;
    if (e != null) {
      if (e = e.$$typeof, e === ri) return 11;
      if (e === li) return 14;
    }
    return 2;
  }
  function wn(e, t) {
    var n = e.alternate;
    return n === null ? (n = Nt(e.tag, t, e.key, e.mode), n.elementType = e.elementType, n.type = e.type, n.stateNode = e.stateNode, n.alternate = e, e.alternate = n) : (n.pendingProps = t, n.type = e.type, n.flags = 0, n.subtreeFlags = 0, n.deletions = null), n.flags = e.flags & 14680064, n.childLanes = e.childLanes, n.lanes = e.lanes, n.child = e.child, n.memoizedProps = e.memoizedProps, n.memoizedState = e.memoizedState, n.updateQueue = e.updateQueue, t = e.dependencies, n.dependencies = t === null ? null : {
      lanes: t.lanes,
      firstContext: t.firstContext
    }, n.sibling = e.sibling, n.index = e.index, n.ref = e.ref, n;
  }
  function xa(e, t, n, r, l, a) {
    var o = 2;
    if (r = e, typeof e == "function") Ai(e) && (o = 1);
    else if (typeof e == "string") o = 5;
    else e: switch (e) {
      case ar:
        return $n(n.children, l, a, t);
      case ni:
        o = 8, l |= 8;
        break;
      case qo:
        return e = Nt(12, n, t, l | 2), e.elementType = qo, e.lanes = a, e;
      case es:
        return e = Nt(13, n, t, l), e.elementType = es, e.lanes = a, e;
      case ts:
        return e = Nt(19, n, t, l), e.elementType = ts, e.lanes = a, e;
      case cd:
        return oo(n, l, a, t);
      default:
        if (typeof e == "object" && e !== null) switch (e.$$typeof) {
          case id:
            o = 10;
            break e;
          case ud:
            o = 9;
            break e;
          case ri:
            o = 11;
            break e;
          case li:
            o = 14;
            break e;
          case an:
            o = 16, r = null;
            break e;
        }
        throw Error(b(130, e == null ? e : typeof e, ""));
    }
    return t = Nt(o, n, t, l), t.elementType = e, t.type = r, t.lanes = a, t;
  }
  function $n(e, t, n, r) {
    return e = Nt(7, e, r, t), e.lanes = n, e;
  }
  function oo(e, t, n, r) {
    return e = Nt(22, e, r, t), e.elementType = cd, e.lanes = n, e.stateNode = {
      isHidden: false
    }, e;
  }
  function Wo(e, t, n) {
    return e = Nt(6, e, null, t), e.lanes = n, e;
  }
  function Vo(e, t, n) {
    return t = Nt(4, e.children !== null ? e.children : [], e.key, t), t.lanes = n, t.stateNode = {
      containerInfo: e.containerInfo,
      pendingChildren: null,
      implementation: e.implementation
    }, t;
  }
  function ng(e, t, n, r, l) {
    this.tag = t, this.containerInfo = e, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = jo(0), this.expirationTimes = jo(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = jo(0), this.identifierPrefix = r, this.onRecoverableError = l, this.mutableSourceEagerHydrationData = null;
  }
  function Ui(e, t, n, r, l, a, o, i, s) {
    return e = new ng(e, t, n, i, s), t === 1 ? (t = 1, a === true && (t |= 8)) : t = 0, a = Nt(3, null, null, t), e.current = a, a.stateNode = e, a.memoizedState = {
      element: r,
      isDehydrated: n,
      cache: null,
      transitions: null,
      pendingSuspenseBoundaries: null
    }, ki(a), e;
  }
  function rg(e, t, n) {
    var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: lr,
      key: r == null ? null : "" + r,
      children: e,
      containerInfo: t,
      implementation: n
    };
  }
  function nm(e) {
    if (!e) return Nn;
    e = e._reactInternals;
    e: {
      if (Gn(e) !== e || e.tag !== 1) throw Error(b(170));
      var t = e;
      do {
        switch (t.tag) {
          case 3:
            t = t.stateNode.context;
            break e;
          case 1:
            if (ct(t.type)) {
              t = t.stateNode.__reactInternalMemoizedMergedChildContext;
              break e;
            }
        }
        t = t.return;
      } while (t !== null);
      throw Error(b(171));
    }
    if (e.tag === 1) {
      var n = e.type;
      if (ct(n)) return tf(e, n, t);
    }
    return t;
  }
  function rm(e, t, n, r, l, a, o, i, s) {
    return e = Ui(n, r, true, e, l, a, o, i, s), e.context = nm(null), n = e.current, r = rt(), l = yn(n), a = Jt(r, l), a.callback = t ?? null, vn(n, a, l), e.current.lanes = l, Rl(e, l, r), dt(e, r), e;
  }
  function so(e, t, n, r) {
    var l = t.current, a = rt(), o = yn(l);
    return n = nm(n), t.context === null ? t.context = n : t.pendingContext = n, t = Jt(a, o), t.payload = {
      element: e
    }, r = r === void 0 ? null : r, r !== null && (t.callback = r), e = vn(l, t, o), e !== null && (Ot(e, l, o, a), fa(e, l, o)), o;
  }
  function Va(e) {
    if (e = e.current, !e.child) return null;
    switch (e.child.tag) {
      case 5:
        return e.child.stateNode;
      default:
        return e.child.stateNode;
    }
  }
  function dc(e, t) {
    if (e = e.memoizedState, e !== null && e.dehydrated !== null) {
      var n = e.retryLane;
      e.retryLane = n !== 0 && n < t ? n : t;
    }
  }
  function $i(e, t) {
    dc(e, t), (e = e.alternate) && dc(e, t);
  }
  function lg() {
    return null;
  }
  var lm = typeof reportError == "function" ? reportError : function(e) {
    console.error(e);
  };
  function Fi(e) {
    this._internalRoot = e;
  }
  io.prototype.render = Fi.prototype.render = function(e) {
    var t = this._internalRoot;
    if (t === null) throw Error(b(409));
    so(e, t, null, null);
  };
  io.prototype.unmount = Fi.prototype.unmount = function() {
    var e = this._internalRoot;
    if (e !== null) {
      this._internalRoot = null;
      var t = e.containerInfo;
      Hn(function() {
        so(null, e, null, null);
      }), t[Zt] = null;
    }
  };
  function io(e) {
    this._internalRoot = e;
  }
  io.prototype.unstable_scheduleHydration = function(e) {
    if (e) {
      var t = Ld();
      e = {
        blockedOn: null,
        target: e,
        priority: t
      };
      for (var n = 0; n < sn.length && t !== 0 && t < sn[n].priority; n++) ;
      sn.splice(n, 0, e), n === 0 && Id(e);
    }
  };
  function Bi(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
  }
  function uo(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11 && (e.nodeType !== 8 || e.nodeValue !== " react-mount-point-unstable "));
  }
  function fc() {
  }
  function ag(e, t, n, r, l) {
    if (l) {
      if (typeof r == "function") {
        var a = r;
        r = function() {
          var c = Va(o);
          a.call(c);
        };
      }
      var o = rm(t, r, e, 0, null, false, false, "", fc);
      return e._reactRootContainer = o, e[Zt] = o.current, vl(e.nodeType === 8 ? e.parentNode : e), Hn(), o;
    }
    for (; l = e.lastChild; ) e.removeChild(l);
    if (typeof r == "function") {
      var i = r;
      r = function() {
        var c = Va(s);
        i.call(c);
      };
    }
    var s = Ui(e, 0, false, null, null, false, false, "", fc);
    return e._reactRootContainer = s, e[Zt] = s.current, vl(e.nodeType === 8 ? e.parentNode : e), Hn(function() {
      so(t, s, n, r);
    }), s;
  }
  function co(e, t, n, r, l) {
    var a = n._reactRootContainer;
    if (a) {
      var o = a;
      if (typeof l == "function") {
        var i = l;
        l = function() {
          var s = Va(o);
          i.call(s);
        };
      }
      so(t, o, e, l);
    } else o = ag(n, t, e, l, r);
    return Va(o);
  }
  Md = function(e) {
    switch (e.tag) {
      case 3:
        var t = e.stateNode;
        if (t.current.memoizedState.isDehydrated) {
          var n = Jr(t.pendingLanes);
          n !== 0 && (si(t, n | 1), dt(t, Ie()), !(ie & 6) && (Pr = Ie() + 500, Cn()));
        }
        break;
      case 13:
        Hn(function() {
          var r = qt(e, 1);
          if (r !== null) {
            var l = rt();
            Ot(r, e, 1, l);
          }
        }), $i(e, 1);
    }
  };
  ii = function(e) {
    if (e.tag === 13) {
      var t = qt(e, 134217728);
      if (t !== null) {
        var n = rt();
        Ot(t, e, 134217728, n);
      }
      $i(e, 134217728);
    }
  };
  Dd = function(e) {
    if (e.tag === 13) {
      var t = yn(e), n = qt(e, t);
      if (n !== null) {
        var r = rt();
        Ot(n, e, t, r);
      }
      $i(e, t);
    }
  };
  Ld = function() {
    return he;
  };
  Od = function(e, t) {
    var n = he;
    try {
      return he = e, t();
    } finally {
      he = n;
    }
  };
  ds = function(e, t, n) {
    switch (t) {
      case "input":
        if (ls(e, n), t = n.name, n.type === "radio" && t != null) {
          for (n = e; n.parentNode; ) n = n.parentNode;
          for (n = n.querySelectorAll("input[name=" + JSON.stringify("" + t) + '][type="radio"]'), t = 0; t < n.length; t++) {
            var r = n[t];
            if (r !== e && r.form === e.form) {
              var l = eo(r);
              if (!l) throw Error(b(90));
              fd(r), ls(r, l);
            }
          }
        }
        break;
      case "textarea":
        pd(e, n);
        break;
      case "select":
        t = n.value, t != null && gr(e, !!n.multiple, t, false);
    }
  };
  Sd = Oi;
  kd = Hn;
  var og = {
    usingClientEntryPoint: false,
    Events: [
      bl,
      ur,
      eo,
      yd,
      wd,
      Oi
    ]
  }, Wr = {
    findFiberByHostInstance: On,
    bundleType: 0,
    version: "18.3.1",
    rendererPackageName: "react-dom"
  }, sg = {
    bundleType: Wr.bundleType,
    version: Wr.version,
    rendererPackageName: Wr.rendererPackageName,
    rendererConfig: Wr.rendererConfig,
    overrideHookState: null,
    overrideHookStateDeletePath: null,
    overrideHookStateRenamePath: null,
    overrideProps: null,
    overridePropsDeletePath: null,
    overridePropsRenamePath: null,
    setErrorHandler: null,
    setSuspenseHandler: null,
    scheduleUpdate: null,
    currentDispatcherRef: tn.ReactCurrentDispatcher,
    findHostInstanceByFiber: function(e) {
      return e = Ed(e), e === null ? null : e.stateNode;
    },
    findFiberByHostInstance: Wr.findFiberByHostInstance || lg,
    findHostInstancesForRefresh: null,
    scheduleRefresh: null,
    scheduleRoot: null,
    setRefreshHandler: null,
    getCurrentFiber: null,
    reconcilerVersion: "18.3.1-next-f1338f8080-20240426"
  };
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") {
    var la = __REACT_DEVTOOLS_GLOBAL_HOOK__;
    if (!la.isDisabled && la.supportsFiber) try {
      Ja = la.inject(sg), Ft = la;
    } catch {
    }
  }
  yt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = og;
  yt.createPortal = function(e, t) {
    var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!Bi(t)) throw Error(b(200));
    return rg(e, t, null, n);
  };
  yt.createRoot = function(e, t) {
    if (!Bi(e)) throw Error(b(299));
    var n = false, r = "", l = lm;
    return t != null && (t.unstable_strictMode === true && (n = true), t.identifierPrefix !== void 0 && (r = t.identifierPrefix), t.onRecoverableError !== void 0 && (l = t.onRecoverableError)), t = Ui(e, 1, false, null, null, n, false, r, l), e[Zt] = t.current, vl(e.nodeType === 8 ? e.parentNode : e), new Fi(t);
  };
  yt.findDOMNode = function(e) {
    if (e == null) return null;
    if (e.nodeType === 1) return e;
    var t = e._reactInternals;
    if (t === void 0) throw typeof e.render == "function" ? Error(b(188)) : (e = Object.keys(e).join(","), Error(b(268, e)));
    return e = Ed(t), e = e === null ? null : e.stateNode, e;
  };
  yt.flushSync = function(e) {
    return Hn(e);
  };
  yt.hydrate = function(e, t, n) {
    if (!uo(t)) throw Error(b(200));
    return co(null, e, t, true, n);
  };
  yt.hydrateRoot = function(e, t, n) {
    if (!Bi(e)) throw Error(b(405));
    var r = n != null && n.hydratedSources || null, l = false, a = "", o = lm;
    if (n != null && (n.unstable_strictMode === true && (l = true), n.identifierPrefix !== void 0 && (a = n.identifierPrefix), n.onRecoverableError !== void 0 && (o = n.onRecoverableError)), t = rm(t, null, e, 1, n ?? null, l, false, a, o), e[Zt] = t.current, vl(e), r) for (e = 0; e < r.length; e++) n = r[e], l = n._getVersion, l = l(n._source), t.mutableSourceEagerHydrationData == null ? t.mutableSourceEagerHydrationData = [
      n,
      l
    ] : t.mutableSourceEagerHydrationData.push(n, l);
    return new io(t);
  };
  yt.render = function(e, t, n) {
    if (!uo(t)) throw Error(b(200));
    return co(null, e, t, false, n);
  };
  yt.unmountComponentAtNode = function(e) {
    if (!uo(e)) throw Error(b(40));
    return e._reactRootContainer ? (Hn(function() {
      co(null, null, e, false, function() {
        e._reactRootContainer = null, e[Zt] = null;
      });
    }), true) : false;
  };
  yt.unstable_batchedUpdates = Oi;
  yt.unstable_renderSubtreeIntoContainer = function(e, t, n, r) {
    if (!uo(n)) throw Error(b(200));
    if (e == null || e._reactInternals === void 0) throw Error(b(38));
    return co(e, t, n, false, r);
  };
  yt.version = "18.3.1-next-f1338f8080-20240426";
  function am() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) try {
      __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(am);
    } catch (e) {
      console.error(e);
    }
  }
  am(), ld.exports = yt;
  var Wi = ld.exports;
  const ig = Qc(Wi), ug = Hc({
    __proto__: null,
    default: ig
  }, [
    Wi
  ]);
  var mc = Wi;
  Xo.createRoot = mc.createRoot, Xo.hydrateRoot = mc.hydrateRoot;
  function je() {
    return je = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, je.apply(this, arguments);
  }
  var Ae;
  (function(e) {
    e.Pop = "POP", e.Push = "PUSH", e.Replace = "REPLACE";
  })(Ae || (Ae = {}));
  const pc = "popstate";
  function cg(e) {
    e === void 0 && (e = {});
    function t(r, l) {
      let { pathname: a, search: o, hash: i } = r.location;
      return Cl("", {
        pathname: a,
        search: o,
        hash: i
      }, l.state && l.state.usr || null, l.state && l.state.key || "default");
    }
    function n(r, l) {
      return typeof l == "string" ? l : Dl(l);
    }
    return fg(t, n, null, e);
  }
  function ae(e, t) {
    if (e === false || e === null || typeof e > "u") throw new Error(t);
  }
  function Qn(e, t) {
    if (!e) {
      typeof console < "u" && console.warn(t);
      try {
        throw new Error(t);
      } catch {
      }
    }
  }
  function dg() {
    return Math.random().toString(36).substr(2, 8);
  }
  function hc(e, t) {
    return {
      usr: e.state,
      key: e.key,
      idx: t
    };
  }
  function Cl(e, t, n, r) {
    return n === void 0 && (n = null), je({
      pathname: typeof e == "string" ? e : e.pathname,
      search: "",
      hash: ""
    }, typeof t == "string" ? _n(t) : t, {
      state: n,
      key: t && t.key || r || dg()
    });
  }
  function Dl(e) {
    let { pathname: t = "/", search: n = "", hash: r = "" } = e;
    return n && n !== "?" && (t += n.charAt(0) === "?" ? n : "?" + n), r && r !== "#" && (t += r.charAt(0) === "#" ? r : "#" + r), t;
  }
  function _n(e) {
    let t = {};
    if (e) {
      let n = e.indexOf("#");
      n >= 0 && (t.hash = e.substr(n), e = e.substr(0, n));
      let r = e.indexOf("?");
      r >= 0 && (t.search = e.substr(r), e = e.substr(0, r)), e && (t.pathname = e);
    }
    return t;
  }
  function fg(e, t, n, r) {
    r === void 0 && (r = {});
    let { window: l = document.defaultView, v5Compat: a = false } = r, o = l.history, i = Ae.Pop, s = null, c = m();
    c == null && (c = 0, o.replaceState(je({}, o.state, {
      idx: c
    }), ""));
    function m() {
      return (o.state || {
        idx: null
      }).idx;
    }
    function d() {
      i = Ae.Pop;
      let _ = m(), p = _ == null ? null : _ - c;
      c = _, s && s({
        action: i,
        location: S.location,
        delta: p
      });
    }
    function g(_, p) {
      i = Ae.Push;
      let f = Cl(S.location, _, p);
      c = m() + 1;
      let h = hc(f, c), j = S.createHref(f);
      try {
        o.pushState(h, "", j);
      } catch (C) {
        if (C instanceof DOMException && C.name === "DataCloneError") throw C;
        l.location.assign(j);
      }
      a && s && s({
        action: i,
        location: S.location,
        delta: 1
      });
    }
    function x(_, p) {
      i = Ae.Replace;
      let f = Cl(S.location, _, p);
      c = m();
      let h = hc(f, c), j = S.createHref(f);
      o.replaceState(h, "", j), a && s && s({
        action: i,
        location: S.location,
        delta: 0
      });
    }
    function w(_) {
      let p = l.location.origin !== "null" ? l.location.origin : l.location.href, f = typeof _ == "string" ? _ : Dl(_);
      return f = f.replace(/ $/, "%20"), ae(p, "No window.location.(origin|href) available to create URL for href: " + f), new URL(f, p);
    }
    let S = {
      get action() {
        return i;
      },
      get location() {
        return e(l, o);
      },
      listen(_) {
        if (s) throw new Error("A history only accepts one active listener");
        return l.addEventListener(pc, d), s = _, () => {
          l.removeEventListener(pc, d), s = null;
        };
      },
      createHref(_) {
        return t(l, _);
      },
      createURL: w,
      encodeLocation(_) {
        let p = w(_);
        return {
          pathname: p.pathname,
          search: p.search,
          hash: p.hash
        };
      },
      push: g,
      replace: x,
      go(_) {
        return o.go(_);
      }
    };
    return S;
  }
  var pe;
  (function(e) {
    e.data = "data", e.deferred = "deferred", e.redirect = "redirect", e.error = "error";
  })(pe || (pe = {}));
  const mg = /* @__PURE__ */ new Set([
    "lazy",
    "caseSensitive",
    "path",
    "id",
    "index",
    "children"
  ]);
  function pg(e) {
    return e.index === true;
  }
  function Ha(e, t, n, r) {
    return n === void 0 && (n = []), r === void 0 && (r = {}), e.map((l, a) => {
      let o = [
        ...n,
        String(a)
      ], i = typeof l.id == "string" ? l.id : o.join("-");
      if (ae(l.index !== true || !l.children, "Cannot specify children on an index route"), ae(!r[i], 'Found a route id collision on id "' + i + `".  Route id's must be globally unique within Data Router usages`), pg(l)) {
        let s = je({}, l, t(l), {
          id: i
        });
        return r[i] = s, s;
      } else {
        let s = je({}, l, t(l), {
          id: i,
          children: void 0
        });
        return r[i] = s, l.children && (s.children = Ha(l.children, t, o, r)), s;
      }
    });
  }
  function Dn(e, t, n) {
    return n === void 0 && (n = "/"), ya(e, t, n, false);
  }
  function ya(e, t, n, r) {
    let l = typeof t == "string" ? _n(t) : t, a = Ll(l.pathname || "/", n);
    if (a == null) return null;
    let o = om(e);
    gg(o);
    let i = null;
    for (let s = 0; i == null && s < o.length; ++s) {
      let c = _g(a);
      i = Eg(o[s], c, r);
    }
    return i;
  }
  function hg(e, t) {
    let { route: n, pathname: r, params: l } = e;
    return {
      id: n.id,
      pathname: r,
      params: l,
      data: t[n.id],
      handle: n.handle
    };
  }
  function om(e, t, n, r) {
    t === void 0 && (t = []), n === void 0 && (n = []), r === void 0 && (r = "");
    let l = (a, o, i) => {
      let s = {
        relativePath: i === void 0 ? a.path || "" : i,
        caseSensitive: a.caseSensitive === true,
        childrenIndex: o,
        route: a
      };
      s.relativePath.startsWith("/") && (ae(s.relativePath.startsWith(r), 'Absolute route path "' + s.relativePath + '" nested under path ' + ('"' + r + '" is not valid. An absolute child route path ') + "must start with the combined path of all its parent routes."), s.relativePath = s.relativePath.slice(r.length));
      let c = Sn([
        r,
        s.relativePath
      ]), m = n.concat(s);
      a.children && a.children.length > 0 && (ae(a.index !== true, "Index routes must not have child routes. Please remove " + ('all child routes from route path "' + c + '".')), om(a.children, t, m, c)), !(a.path == null && !a.index) && t.push({
        path: c,
        score: Ng(c, a.index),
        routesMeta: m
      });
    };
    return e.forEach((a, o) => {
      var i;
      if (a.path === "" || !((i = a.path) != null && i.includes("?"))) l(a, o);
      else for (let s of sm(a.path)) l(a, o, s);
    }), t;
  }
  function sm(e) {
    let t = e.split("/");
    if (t.length === 0) return [];
    let [n, ...r] = t, l = n.endsWith("?"), a = n.replace(/\?$/, "");
    if (r.length === 0) return l ? [
      a,
      ""
    ] : [
      a
    ];
    let o = sm(r.join("/")), i = [];
    return i.push(...o.map((s) => s === "" ? a : [
      a,
      s
    ].join("/"))), l && i.push(...o), i.map((s) => e.startsWith("/") && s === "" ? "/" : s);
  }
  function gg(e) {
    e.sort((t, n) => t.score !== n.score ? n.score - t.score : jg(t.routesMeta.map((r) => r.childrenIndex), n.routesMeta.map((r) => r.childrenIndex)));
  }
  const vg = /^:[\w-]+$/, xg = 3, yg = 2, wg = 1, Sg = 10, kg = -2, gc = (e) => e === "*";
  function Ng(e, t) {
    let n = e.split("/"), r = n.length;
    return n.some(gc) && (r += kg), t && (r += yg), n.filter((l) => !gc(l)).reduce((l, a) => l + (vg.test(a) ? xg : a === "" ? wg : Sg), r);
  }
  function jg(e, t) {
    return e.length === t.length && e.slice(0, -1).every((r, l) => r === t[l]) ? e[e.length - 1] - t[t.length - 1] : 0;
  }
  function Eg(e, t, n) {
    n === void 0 && (n = false);
    let { routesMeta: r } = e, l = {}, a = "/", o = [];
    for (let i = 0; i < r.length; ++i) {
      let s = r[i], c = i === r.length - 1, m = a === "/" ? t : t.slice(a.length) || "/", d = vc({
        path: s.relativePath,
        caseSensitive: s.caseSensitive,
        end: c
      }, m), g = s.route;
      if (!d && c && n && !r[r.length - 1].route.index && (d = vc({
        path: s.relativePath,
        caseSensitive: s.caseSensitive,
        end: false
      }, m)), !d) return null;
      Object.assign(l, d.params), o.push({
        params: l,
        pathname: Sn([
          a,
          d.pathname
        ]),
        pathnameBase: bg(Sn([
          a,
          d.pathnameBase
        ])),
        route: g
      }), d.pathnameBase !== "/" && (a = Sn([
        a,
        d.pathnameBase
      ]));
    }
    return o;
  }
  function vc(e, t) {
    typeof e == "string" && (e = {
      path: e,
      caseSensitive: false,
      end: true
    });
    let [n, r] = Cg(e.path, e.caseSensitive, e.end), l = t.match(n);
    if (!l) return null;
    let a = l[0], o = a.replace(/(.)\/+$/, "$1"), i = l.slice(1);
    return {
      params: r.reduce((c, m, d) => {
        let { paramName: g, isOptional: x } = m;
        if (g === "*") {
          let S = i[d] || "";
          o = a.slice(0, a.length - S.length).replace(/(.)\/+$/, "$1");
        }
        const w = i[d];
        return x && !w ? c[g] = void 0 : c[g] = (w || "").replace(/%2F/g, "/"), c;
      }, {}),
      pathname: a,
      pathnameBase: o,
      pattern: e
    };
  }
  function Cg(e, t, n) {
    t === void 0 && (t = false), n === void 0 && (n = true), Qn(e === "*" || !e.endsWith("*") || e.endsWith("/*"), 'Route path "' + e + '" will be treated as if it were ' + ('"' + e.replace(/\*$/, "/*") + '" because the `*` character must ') + "always follow a `/` in the pattern. To get rid of this warning, " + ('please change the route path to "' + e.replace(/\*$/, "/*") + '".'));
    let r = [], l = "^" + e.replace(/\/*\*?$/, "").replace(/^\/*/, "/").replace(/[\\.*+^${}|()[\]]/g, "\\$&").replace(/\/:([\w-]+)(\?)?/g, (o, i, s) => (r.push({
      paramName: i,
      isOptional: s != null
    }), s ? "/?([^\\/]+)?" : "/([^\\/]+)"));
    return e.endsWith("*") ? (r.push({
      paramName: "*"
    }), l += e === "*" || e === "/*" ? "(.*)$" : "(?:\\/(.+)|\\/*)$") : n ? l += "\\/*$" : e !== "" && e !== "/" && (l += "(?:(?=\\/|$))"), [
      new RegExp(l, t ? void 0 : "i"),
      r
    ];
  }
  function _g(e) {
    try {
      return e.split("/").map((t) => decodeURIComponent(t).replace(/\//g, "%2F")).join("/");
    } catch (t) {
      return Qn(false, 'The URL path "' + e + '" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent ' + ("encoding (" + t + ").")), e;
    }
  }
  function Ll(e, t) {
    if (t === "/") return e;
    if (!e.toLowerCase().startsWith(t.toLowerCase())) return null;
    let n = t.endsWith("/") ? t.length - 1 : t.length, r = e.charAt(n);
    return r && r !== "/" ? null : e.slice(n) || "/";
  }
  const Pg = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Rg = (e) => Pg.test(e);
  function Tg(e, t) {
    t === void 0 && (t = "/");
    let { pathname: n, search: r = "", hash: l = "" } = typeof e == "string" ? _n(e) : e, a;
    if (n) if (Rg(n)) a = n;
    else {
      if (n.includes("//")) {
        let o = n;
        n = n.replace(/\/\/+/g, "/"), Qn(false, "Pathnames cannot have embedded double slashes - normalizing " + (o + " -> " + n));
      }
      n.startsWith("/") ? a = xc(n.substring(1), "/") : a = xc(n, t);
    }
    else a = t;
    return {
      pathname: a,
      search: Mg(r),
      hash: Dg(l)
    };
  }
  function xc(e, t) {
    let n = t.replace(/\/+$/, "").split("/");
    return e.split("/").forEach((l) => {
      l === ".." ? n.length > 1 && n.pop() : l !== "." && n.push(l);
    }), n.length > 1 ? n.join("/") : "/";
  }
  function Ho(e, t, n, r) {
    return "Cannot include a '" + e + "' character in a manually specified " + ("`to." + t + "` field [" + JSON.stringify(r) + "].  Please separate it out to the ") + ("`to." + n + "` field. Alternatively you may provide the full path as ") + 'a string in <Link to="..."> and the router will parse it for you.';
  }
  function im(e) {
    return e.filter((t, n) => n === 0 || t.route.path && t.route.path.length > 0);
  }
  function um(e, t) {
    let n = im(e);
    return t ? n.map((r, l) => l === n.length - 1 ? r.pathname : r.pathnameBase) : n.map((r) => r.pathnameBase);
  }
  function cm(e, t, n, r) {
    r === void 0 && (r = false);
    let l;
    typeof e == "string" ? l = _n(e) : (l = je({}, e), ae(!l.pathname || !l.pathname.includes("?"), Ho("?", "pathname", "search", l)), ae(!l.pathname || !l.pathname.includes("#"), Ho("#", "pathname", "hash", l)), ae(!l.search || !l.search.includes("#"), Ho("#", "search", "hash", l)));
    let a = e === "" || l.pathname === "", o = a ? "/" : l.pathname, i;
    if (o == null) i = n;
    else {
      let d = t.length - 1;
      if (!r && o.startsWith("..")) {
        let g = o.split("/");
        for (; g[0] === ".."; ) g.shift(), d -= 1;
        l.pathname = g.join("/");
      }
      i = d >= 0 ? t[d] : "/";
    }
    let s = Tg(l, i), c = o && o !== "/" && o.endsWith("/"), m = (a || o === ".") && n.endsWith("/");
    return !s.pathname.endsWith("/") && (c || m) && (s.pathname += "/"), s;
  }
  const Sn = (e) => e.join("/").replace(/\/\/+/g, "/"), bg = (e) => e.replace(/\/+$/, "").replace(/^\/*/, "/"), Mg = (e) => !e || e === "?" ? "" : e.startsWith("?") ? e : "?" + e, Dg = (e) => !e || e === "#" ? "" : e.startsWith("#") ? e : "#" + e;
  class Qa {
    constructor(t, n, r, l) {
      l === void 0 && (l = false), this.status = t, this.statusText = n || "", this.internal = l, r instanceof Error ? (this.data = r.toString(), this.error = r) : this.data = r;
    }
  }
  function _l(e) {
    return e != null && typeof e.status == "number" && typeof e.statusText == "string" && typeof e.internal == "boolean" && "data" in e;
  }
  const dm = [
    "post",
    "put",
    "patch",
    "delete"
  ], Lg = new Set(dm), Og = [
    "get",
    ...dm
  ], Ig = new Set(Og), zg = /* @__PURE__ */ new Set([
    301,
    302,
    303,
    307,
    308
  ]), Ag = /* @__PURE__ */ new Set([
    307,
    308
  ]), Qo = {
    state: "idle",
    location: void 0,
    formMethod: void 0,
    formAction: void 0,
    formEncType: void 0,
    formData: void 0,
    json: void 0,
    text: void 0
  }, Ug = {
    state: "idle",
    data: void 0,
    formMethod: void 0,
    formAction: void 0,
    formEncType: void 0,
    formData: void 0,
    json: void 0,
    text: void 0
  }, Vr = {
    state: "unblocked",
    proceed: void 0,
    reset: void 0,
    location: void 0
  }, Vi = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, $g = (e) => ({
    hasErrorBoundary: !!e.hasErrorBoundary
  }), fm = "remix-router-transitions";
  function Fg(e) {
    const t = e.window ? e.window : typeof window < "u" ? window : void 0, n = typeof t < "u" && typeof t.document < "u" && typeof t.document.createElement < "u", r = !n;
    ae(e.routes.length > 0, "You must provide a non-empty routes array to createRouter");
    let l;
    if (e.mapRouteProperties) l = e.mapRouteProperties;
    else if (e.detectErrorBoundary) {
      let y = e.detectErrorBoundary;
      l = (N) => ({
        hasErrorBoundary: y(N)
      });
    } else l = $g;
    let a = {}, o = Ha(e.routes, l, void 0, a), i, s = e.basename || "/", c = e.dataStrategy || Hg, m = e.patchRoutesOnNavigation, d = je({
      v7_fetcherPersist: false,
      v7_normalizeFormMethod: false,
      v7_partialHydration: false,
      v7_prependBasename: false,
      v7_relativeSplatPath: false,
      v7_skipActionErrorRevalidation: false
    }, e.future), g = null, x = /* @__PURE__ */ new Set(), w = null, S = null, _ = null, p = e.hydrationData != null, f = Dn(o, e.history.location, s), h = false, j = null;
    if (f == null && !m) {
      let y = ot(404, {
        pathname: e.history.location.pathname
      }), { matches: N, route: E } = Rc(o);
      f = N, j = {
        [E.id]: y
      };
    }
    f && !e.hydrationData && Al(f, o, e.history.location.pathname).active && (f = null);
    let C;
    if (f) if (f.some((y) => y.route.lazy)) C = false;
    else if (!f.some((y) => y.route.loader)) C = true;
    else if (d.v7_partialHydration) {
      let y = e.hydrationData ? e.hydrationData.loaderData : null, N = e.hydrationData ? e.hydrationData.errors : null;
      if (N) {
        let E = f.findIndex((T) => N[T.route.id] !== void 0);
        C = f.slice(0, E + 1).every((T) => !Hs(T.route, y, N));
      } else C = f.every((E) => !Hs(E.route, y, N));
    } else C = e.hydrationData != null;
    else if (C = false, f = [], d.v7_partialHydration) {
      let y = Al(null, o, e.history.location.pathname);
      y.active && y.matches && (h = true, f = y.matches);
    }
    let R, k = {
      historyAction: e.history.action,
      location: e.history.location,
      matches: f,
      initialized: C,
      navigation: Qo,
      restoreScrollPosition: e.hydrationData != null ? false : null,
      preventScrollReset: false,
      revalidation: "idle",
      loaderData: e.hydrationData && e.hydrationData.loaderData || {},
      actionData: e.hydrationData && e.hydrationData.actionData || null,
      errors: e.hydrationData && e.hydrationData.errors || j,
      fetchers: /* @__PURE__ */ new Map(),
      blockers: /* @__PURE__ */ new Map()
    }, P = Ae.Pop, $ = false, L, G = false, X = /* @__PURE__ */ new Map(), de = null, se = false, Ce = false, Ve = [], ft = /* @__PURE__ */ new Set(), M = /* @__PURE__ */ new Map(), H = 0, W = -1, le = /* @__PURE__ */ new Map(), te = /* @__PURE__ */ new Set(), fe = /* @__PURE__ */ new Map(), Z = /* @__PURE__ */ new Map(), ue = /* @__PURE__ */ new Set(), Se = /* @__PURE__ */ new Map(), ne = /* @__PURE__ */ new Map(), Te;
    function ge() {
      if (g = e.history.listen((y) => {
        let { action: N, location: E, delta: T } = y;
        if (Te) {
          Te(), Te = void 0;
          return;
        }
        Qn(ne.size === 0 || T != null, "You are trying to use a blocker on a POP navigation to a location that was not created by @remix-run/router. This will fail silently in production. This can happen if you are navigating outside the router via `window.history.pushState`/`window.location.hash` instead of using router navigation APIs.  This can also happen if you are using createHashRouter and the user manually changes the URL.");
        let O = tu({
          currentLocation: k.location,
          nextLocation: E,
          historyAction: N
        });
        if (O && T != null) {
          let Q = new Promise((J) => {
            Te = J;
          });
          e.history.go(T * -1), zl(O, {
            state: "blocked",
            location: E,
            proceed() {
              zl(O, {
                state: "proceeding",
                proceed: void 0,
                reset: void 0,
                location: E
              }), Q.then(() => e.history.go(T));
            },
            reset() {
              let J = new Map(k.blockers);
              J.set(O, Vr), K({
                blockers: J
              });
            }
          });
          return;
        }
        return Oe(N, E);
      }), n) {
        av(t, X);
        let y = () => ov(t, X);
        t.addEventListener("pagehide", y), de = () => t.removeEventListener("pagehide", y);
      }
      return k.initialized || Oe(Ae.Pop, k.location, {
        initialHydration: true
      }), R;
    }
    function ke() {
      g && g(), de && de(), x.clear(), L && L.abort(), k.fetchers.forEach((y, N) => Il(N)), k.blockers.forEach((y, N) => eu(N));
    }
    function I(y) {
      return x.add(y), () => x.delete(y);
    }
    function K(y, N) {
      N === void 0 && (N = {}), k = je({}, k, y);
      let E = [], T = [];
      d.v7_fetcherPersist && k.fetchers.forEach((O, Q) => {
        O.state === "idle" && (ue.has(Q) ? T.push(Q) : E.push(Q));
      }), ue.forEach((O) => {
        !k.fetchers.has(O) && !M.has(O) && T.push(O);
      }), [
        ...x
      ].forEach((O) => O(k, {
        deletedFetchers: T,
        viewTransitionOpts: N.viewTransitionOpts,
        flushSync: N.flushSync === true
      })), d.v7_fetcherPersist ? (E.forEach((O) => k.fetchers.delete(O)), T.forEach((O) => Il(O))) : T.forEach((O) => ue.delete(O));
    }
    function D(y, N, E) {
      var T, O;
      let { flushSync: Q } = E === void 0 ? {} : E, J = k.actionData != null && k.navigation.formMethod != null && Mt(k.navigation.formMethod) && k.navigation.state === "loading" && ((T = y.state) == null ? void 0 : T._isRedirect) !== true, U;
      N.actionData ? Object.keys(N.actionData).length > 0 ? U = N.actionData : U = null : J ? U = k.actionData : U = null;
      let F = N.loaderData ? _c(k.loaderData, N.loaderData, N.matches || [], N.errors) : k.loaderData, A = k.blockers;
      A.size > 0 && (A = new Map(A), A.forEach((oe, Qe) => A.set(Qe, Vr)));
      let V = $ === true || k.navigation.formMethod != null && Mt(k.navigation.formMethod) && ((O = y.state) == null ? void 0 : O._isRedirect) !== true;
      i && (o = i, i = void 0), se || P === Ae.Pop || (P === Ae.Push ? e.history.push(y, y.state) : P === Ae.Replace && e.history.replace(y, y.state));
      let ee;
      if (P === Ae.Pop) {
        let oe = X.get(k.location.pathname);
        oe && oe.has(y.pathname) ? ee = {
          currentLocation: k.location,
          nextLocation: y
        } : X.has(y.pathname) && (ee = {
          currentLocation: y,
          nextLocation: k.location
        });
      } else if (G) {
        let oe = X.get(k.location.pathname);
        oe ? oe.add(y.pathname) : (oe = /* @__PURE__ */ new Set([
          y.pathname
        ]), X.set(k.location.pathname, oe)), ee = {
          currentLocation: k.location,
          nextLocation: y
        };
      }
      K(je({}, N, {
        actionData: U,
        loaderData: F,
        historyAction: P,
        location: y,
        initialized: true,
        navigation: Qo,
        revalidation: "idle",
        restoreScrollPosition: ru(y, N.matches || k.matches),
        preventScrollReset: V,
        blockers: A
      }), {
        viewTransitionOpts: ee,
        flushSync: Q === true
      }), P = Ae.Pop, $ = false, G = false, se = false, Ce = false, Ve = [];
    }
    async function B(y, N) {
      if (typeof y == "number") {
        e.history.go(y);
        return;
      }
      let E = Vs(k.location, k.matches, s, d.v7_prependBasename, y, d.v7_relativeSplatPath, N == null ? void 0 : N.fromRouteId, N == null ? void 0 : N.relative), { path: T, submission: O, error: Q } = yc(d.v7_normalizeFormMethod, false, E, N), J = k.location, U = Cl(k.location, T, N && N.state);
      U = je({}, U, e.history.encodeLocation(U));
      let F = N && N.replace != null ? N.replace : void 0, A = Ae.Push;
      F === true ? A = Ae.Replace : F === false || O != null && Mt(O.formMethod) && O.formAction === k.location.pathname + k.location.search && (A = Ae.Replace);
      let V = N && "preventScrollReset" in N ? N.preventScrollReset === true : void 0, ee = (N && N.flushSync) === true, oe = tu({
        currentLocation: J,
        nextLocation: U,
        historyAction: A
      });
      if (oe) {
        zl(oe, {
          state: "blocked",
          location: U,
          proceed() {
            zl(oe, {
              state: "proceeding",
              proceed: void 0,
              reset: void 0,
              location: U
            }), B(y, N);
          },
          reset() {
            let Qe = new Map(k.blockers);
            Qe.set(oe, Vr), K({
              blockers: Qe
            });
          }
        });
        return;
      }
      return await Oe(A, U, {
        submission: O,
        pendingError: Q,
        preventScrollReset: V,
        replace: N && N.replace,
        enableViewTransition: N && N.viewTransition,
        flushSync: ee
      });
    }
    function Le() {
      if (Jn(), K({
        revalidation: "loading"
      }), k.navigation.state !== "submitting") {
        if (k.navigation.state === "idle") {
          Oe(k.historyAction, k.location, {
            startUninterruptedRevalidation: true
          });
          return;
        }
        Oe(P || k.historyAction, k.navigation.location, {
          overrideNavigation: k.navigation,
          enableViewTransition: G === true
        });
      }
    }
    async function Oe(y, N, E) {
      L && L.abort(), L = null, P = y, se = (E && E.startUninterruptedRevalidation) === true, bm(k.location, k.matches), $ = (E && E.preventScrollReset) === true, G = (E && E.enableViewTransition) === true;
      let T = i || o, O = E && E.overrideNavigation, Q = E != null && E.initialHydration && k.matches && k.matches.length > 0 && !h ? k.matches : Dn(T, N, s), J = (E && E.flushSync) === true;
      if (Q && k.initialized && !Ce && Xg(k.location, N) && !(E && E.submission && Mt(E.submission.formMethod))) {
        D(N, {
          matches: Q
        }, {
          flushSync: J
        });
        return;
      }
      let U = Al(Q, T, N.pathname);
      if (U.active && U.matches && (Q = U.matches), !Q) {
        let { error: ve, notFoundMatches: me, route: be } = go(N.pathname);
        D(N, {
          matches: me,
          loaderData: {},
          errors: {
            [be.id]: ve
          }
        }, {
          flushSync: J
        });
        return;
      }
      L = new AbortController();
      let F = tr(e.history, N, L.signal, E && E.submission), A;
      if (E && E.pendingError) A = [
        Ln(Q).route.id,
        {
          type: pe.error,
          error: E.pendingError
        }
      ];
      else if (E && E.submission && Mt(E.submission.formMethod)) {
        let ve = await Ne(F, N, E.submission, Q, U.active, {
          replace: E.replace,
          flushSync: J
        });
        if (ve.shortCircuited) return;
        if (ve.pendingActionResult) {
          let [me, be] = ve.pendingActionResult;
          if (ht(be) && _l(be.error) && be.error.status === 404) {
            L = null, D(N, {
              matches: ve.matches,
              loaderData: {},
              errors: {
                [me]: be.error
              }
            });
            return;
          }
        }
        Q = ve.matches || Q, A = ve.pendingActionResult, O = Ko(N, E.submission), J = false, U.active = false, F = tr(e.history, F.url, F.signal);
      }
      let { shortCircuited: V, matches: ee, loaderData: oe, errors: Qe } = await Y(F, N, Q, U.active, O, E && E.submission, E && E.fetcherSubmission, E && E.replace, E && E.initialHydration === true, J, A);
      V || (L = null, D(N, je({
        matches: ee || Q
      }, Pc(A), {
        loaderData: oe,
        errors: Qe
      })));
    }
    async function Ne(y, N, E, T, O, Q) {
      Q === void 0 && (Q = {}), Jn();
      let J = rv(N, E);
      if (K({
        navigation: J
      }, {
        flushSync: Q.flushSync === true
      }), O) {
        let A = await Ul(T, N.pathname, y.signal);
        if (A.type === "aborted") return {
          shortCircuited: true
        };
        if (A.type === "error") {
          let V = Ln(A.partialMatches).route.id;
          return {
            matches: A.partialMatches,
            pendingActionResult: [
              V,
              {
                type: pe.error,
                error: A.error
              }
            ]
          };
        } else if (A.matches) T = A.matches;
        else {
          let { notFoundMatches: V, error: ee, route: oe } = go(N.pathname);
          return {
            matches: V,
            pendingActionResult: [
              oe.id,
              {
                type: pe.error,
                error: ee
              }
            ]
          };
        }
      }
      let U, F = Zr(T, N);
      if (!F.route.action && !F.route.lazy) U = {
        type: pe.error,
        error: ot(405, {
          method: y.method,
          pathname: N.pathname,
          routeId: F.route.id
        })
      };
      else if (U = (await Vt("action", k, y, [
        F
      ], T, null))[F.route.id], y.signal.aborted) return {
        shortCircuited: true
      };
      if (An(U)) {
        let A;
        return Q && Q.replace != null ? A = Q.replace : A = jc(U.response.headers.get("Location"), new URL(y.url), s, e.history) === k.location.pathname + k.location.search, await He(y, U, true, {
          submission: E,
          replace: A
        }), {
          shortCircuited: true
        };
      }
      if (fn(U)) throw ot(400, {
        type: "defer-action"
      });
      if (ht(U)) {
        let A = Ln(T, F.route.id);
        return (Q && Q.replace) !== true && (P = Ae.Push), {
          matches: T,
          pendingActionResult: [
            A.route.id,
            U
          ]
        };
      }
      return {
        matches: T,
        pendingActionResult: [
          F.route.id,
          U
        ]
      };
    }
    async function Y(y, N, E, T, O, Q, J, U, F, A, V) {
      let ee = O || Ko(N, Q), oe = Q || J || bc(ee), Qe = !se && (!d.v7_partialHydration || !F);
      if (T) {
        if (Qe) {
          let Me = q(V);
          K(je({
            navigation: ee
          }, Me !== void 0 ? {
            actionData: Me
          } : {}), {
            flushSync: A
          });
        }
        let ce = await Ul(E, N.pathname, y.signal);
        if (ce.type === "aborted") return {
          shortCircuited: true
        };
        if (ce.type === "error") {
          let Me = Ln(ce.partialMatches).route.id;
          return {
            matches: ce.partialMatches,
            loaderData: {},
            errors: {
              [Me]: ce.error
            }
          };
        } else if (ce.matches) E = ce.matches;
        else {
          let { error: Me, notFoundMatches: Zn, route: Or } = go(N.pathname);
          return {
            matches: Zn,
            loaderData: {},
            errors: {
              [Or.id]: Me
            }
          };
        }
      }
      let ve = i || o, [me, be] = Sc(e.history, k, E, oe, N, d.v7_partialHydration && F === true, d.v7_skipActionErrorRevalidation, Ce, Ve, ft, ue, fe, te, ve, s, V);
      if (vo((ce) => !(E && E.some((Me) => Me.route.id === ce)) || me && me.some((Me) => Me.route.id === ce)), W = ++H, me.length === 0 && be.length === 0) {
        let ce = Zi();
        return D(N, je({
          matches: E,
          loaderData: {},
          errors: V && ht(V[1]) ? {
            [V[0]]: V[1].error
          } : null
        }, Pc(V), ce ? {
          fetchers: new Map(k.fetchers)
        } : {}), {
          flushSync: A
        }), {
          shortCircuited: true
        };
      }
      if (Qe) {
        let ce = {};
        if (!T) {
          ce.navigation = ee;
          let Me = q(V);
          Me !== void 0 && (ce.actionData = Me);
        }
        be.length > 0 && (ce.fetchers = tt(be)), K(ce, {
          flushSync: A
        });
      }
      be.forEach((ce) => {
        nn(ce.key), ce.controller && M.set(ce.key, ce.controller);
      });
      let Xn = () => be.forEach((ce) => nn(ce.key));
      L && L.signal.addEventListener("abort", Xn);
      let { loaderResults: Dr, fetcherResults: Ht } = await Pn(k, E, me, be, y);
      if (y.signal.aborted) return {
        shortCircuited: true
      };
      L && L.signal.removeEventListener("abort", Xn), be.forEach((ce) => M.delete(ce.key));
      let zt = aa(Dr);
      if (zt) return await He(y, zt.result, true, {
        replace: U
      }), {
        shortCircuited: true
      };
      if (zt = aa(Ht), zt) return te.add(zt.key), await He(y, zt.result, true, {
        replace: U
      }), {
        shortCircuited: true
      };
      let { loaderData: xo, errors: Lr } = Cc(k, E, Dr, V, be, Ht, Se);
      Se.forEach((ce, Me) => {
        ce.subscribe((Zn) => {
          (Zn || ce.done) && Se.delete(Me);
        });
      }), d.v7_partialHydration && F && k.errors && (Lr = je({}, k.errors, Lr));
      let Rn = Zi(), $l = qi(W), Fl = Rn || $l || be.length > 0;
      return je({
        matches: E,
        loaderData: xo,
        errors: Lr
      }, Fl ? {
        fetchers: new Map(k.fetchers)
      } : {});
    }
    function q(y) {
      if (y && !ht(y[1])) return {
        [y[0]]: y[1].data
      };
      if (k.actionData) return Object.keys(k.actionData).length === 0 ? null : k.actionData;
    }
    function tt(y) {
      return y.forEach((N) => {
        let E = k.fetchers.get(N.key), T = Hr(void 0, E ? E.data : void 0);
        k.fetchers.set(N.key, T);
      }), new Map(k.fetchers);
    }
    function Je(y, N, E, T) {
      if (r) throw new Error("router.fetch() was called during the server render, but it shouldn't be. You are likely calling a useFetcher() method in the body of your component. Try moving it to a useEffect or a callback.");
      nn(y);
      let O = (T && T.flushSync) === true, Q = i || o, J = Vs(k.location, k.matches, s, d.v7_prependBasename, E, d.v7_relativeSplatPath, N, T == null ? void 0 : T.relative), U = Dn(Q, J, s), F = Al(U, Q, J);
      if (F.active && F.matches && (U = F.matches), !U) {
        Pt(y, N, ot(404, {
          pathname: J
        }), {
          flushSync: O
        });
        return;
      }
      let { path: A, submission: V, error: ee } = yc(d.v7_normalizeFormMethod, true, J, T);
      if (ee) {
        Pt(y, N, ee, {
          flushSync: O
        });
        return;
      }
      let oe = Zr(U, A), Qe = (T && T.preventScrollReset) === true;
      if (V && Mt(V.formMethod)) {
        Mr(y, N, A, oe, U, F.active, O, Qe, V);
        return;
      }
      fe.set(y, {
        routeId: N,
        path: A
      }), _t(y, N, A, oe, U, F.active, O, Qe, V);
    }
    async function Mr(y, N, E, T, O, Q, J, U, F) {
      Jn(), fe.delete(y);
      function A(ze) {
        if (!ze.route.action && !ze.route.lazy) {
          let qn = ot(405, {
            method: F.formMethod,
            pathname: E,
            routeId: N
          });
          return Pt(y, N, qn, {
            flushSync: J
          }), true;
        }
        return false;
      }
      if (!Q && A(T)) return;
      let V = k.fetchers.get(y);
      mt(y, lv(F, V), {
        flushSync: J
      });
      let ee = new AbortController(), oe = tr(e.history, E, ee.signal, F);
      if (Q) {
        let ze = await Ul(O, new URL(oe.url).pathname, oe.signal, y);
        if (ze.type === "aborted") return;
        if (ze.type === "error") {
          Pt(y, N, ze.error, {
            flushSync: J
          });
          return;
        } else if (ze.matches) {
          if (O = ze.matches, T = Zr(O, E), A(T)) return;
        } else {
          Pt(y, N, ot(404, {
            pathname: E
          }), {
            flushSync: J
          });
          return;
        }
      }
      M.set(y, ee);
      let Qe = H, me = (await Vt("action", k, oe, [
        T
      ], O, y))[T.route.id];
      if (oe.signal.aborted) {
        M.get(y) === ee && M.delete(y);
        return;
      }
      if (d.v7_fetcherPersist && ue.has(y)) {
        if (An(me) || ht(me)) {
          mt(y, ln(void 0));
          return;
        }
      } else {
        if (An(me)) if (M.delete(y), W > Qe) {
          mt(y, ln(void 0));
          return;
        } else return te.add(y), mt(y, Hr(F)), He(oe, me, false, {
          fetcherSubmission: F,
          preventScrollReset: U
        });
        if (ht(me)) {
          Pt(y, N, me.error);
          return;
        }
      }
      if (fn(me)) throw ot(400, {
        type: "defer-action"
      });
      let be = k.navigation.location || k.location, Xn = tr(e.history, be, ee.signal), Dr = i || o, Ht = k.navigation.state !== "idle" ? Dn(Dr, k.navigation.location, s) : k.matches;
      ae(Ht, "Didn't find any matches after fetcher action");
      let zt = ++H;
      le.set(y, zt);
      let xo = Hr(F, me.data);
      k.fetchers.set(y, xo);
      let [Lr, Rn] = Sc(e.history, k, Ht, F, be, false, d.v7_skipActionErrorRevalidation, Ce, Ve, ft, ue, fe, te, Dr, s, [
        T.route.id,
        me
      ]);
      Rn.filter((ze) => ze.key !== y).forEach((ze) => {
        let qn = ze.key, lu = k.fetchers.get(qn), Lm = Hr(void 0, lu ? lu.data : void 0);
        k.fetchers.set(qn, Lm), nn(qn), ze.controller && M.set(qn, ze.controller);
      }), K({
        fetchers: new Map(k.fetchers)
      });
      let $l = () => Rn.forEach((ze) => nn(ze.key));
      ee.signal.addEventListener("abort", $l);
      let { loaderResults: Fl, fetcherResults: ce } = await Pn(k, Ht, Lr, Rn, Xn);
      if (ee.signal.aborted) return;
      ee.signal.removeEventListener("abort", $l), le.delete(y), M.delete(y), Rn.forEach((ze) => M.delete(ze.key));
      let Me = aa(Fl);
      if (Me) return He(Xn, Me.result, false, {
        preventScrollReset: U
      });
      if (Me = aa(ce), Me) return te.add(Me.key), He(Xn, Me.result, false, {
        preventScrollReset: U
      });
      let { loaderData: Zn, errors: Or } = Cc(k, Ht, Fl, void 0, Rn, ce, Se);
      if (k.fetchers.has(y)) {
        let ze = ln(me.data);
        k.fetchers.set(y, ze);
      }
      qi(zt), k.navigation.state === "loading" && zt > W ? (ae(P, "Expected pending action"), L && L.abort(), D(k.navigation.location, {
        matches: Ht,
        loaderData: Zn,
        errors: Or,
        fetchers: new Map(k.fetchers)
      })) : (K({
        errors: Or,
        loaderData: _c(k.loaderData, Zn, Ht, Or),
        fetchers: new Map(k.fetchers)
      }), Ce = false);
    }
    async function _t(y, N, E, T, O, Q, J, U, F) {
      let A = k.fetchers.get(y);
      mt(y, Hr(F, A ? A.data : void 0), {
        flushSync: J
      });
      let V = new AbortController(), ee = tr(e.history, E, V.signal);
      if (Q) {
        let me = await Ul(O, new URL(ee.url).pathname, ee.signal, y);
        if (me.type === "aborted") return;
        if (me.type === "error") {
          Pt(y, N, me.error, {
            flushSync: J
          });
          return;
        } else if (me.matches) O = me.matches, T = Zr(O, E);
        else {
          Pt(y, N, ot(404, {
            pathname: E
          }), {
            flushSync: J
          });
          return;
        }
      }
      M.set(y, V);
      let oe = H, ve = (await Vt("loader", k, ee, [
        T
      ], O, y))[T.route.id];
      if (fn(ve) && (ve = await Hi(ve, ee.signal, true) || ve), M.get(y) === V && M.delete(y), !ee.signal.aborted) {
        if (ue.has(y)) {
          mt(y, ln(void 0));
          return;
        }
        if (An(ve)) if (W > oe) {
          mt(y, ln(void 0));
          return;
        } else {
          te.add(y), await He(ee, ve, false, {
            preventScrollReset: U
          });
          return;
        }
        if (ht(ve)) {
          Pt(y, N, ve.error);
          return;
        }
        ae(!fn(ve), "Unhandled fetcher deferred data"), mt(y, ln(ve.data));
      }
    }
    async function He(y, N, E, T) {
      let { submission: O, fetcherSubmission: Q, preventScrollReset: J, replace: U } = T === void 0 ? {} : T;
      N.response.headers.has("X-Remix-Revalidate") && (Ce = true);
      let F = N.response.headers.get("Location");
      ae(F, "Expected a Location header on the redirect Response"), F = jc(F, new URL(y.url), s, e.history);
      let A = Cl(k.location, F, {
        _isRedirect: true
      });
      if (n) {
        let me = false;
        if (N.response.headers.has("X-Remix-Reload-Document")) me = true;
        else if (Vi.test(F)) {
          const be = e.history.createURL(F);
          me = be.origin !== t.location.origin || Ll(be.pathname, s) == null;
        }
        if (me) {
          U ? t.location.replace(F) : t.location.assign(F);
          return;
        }
      }
      L = null;
      let V = U === true || N.response.headers.has("X-Remix-Replace") ? Ae.Replace : Ae.Push, { formMethod: ee, formAction: oe, formEncType: Qe } = k.navigation;
      !O && !Q && ee && oe && Qe && (O = bc(k.navigation));
      let ve = O || Q;
      if (Ag.has(N.response.status) && ve && Mt(ve.formMethod)) await Oe(V, A, {
        submission: je({}, ve, {
          formAction: F
        }),
        preventScrollReset: J || $,
        enableViewTransition: E ? G : void 0
      });
      else {
        let me = Ko(A, O);
        await Oe(V, A, {
          overrideNavigation: me,
          fetcherSubmission: Q,
          preventScrollReset: J || $,
          enableViewTransition: E ? G : void 0
        });
      }
    }
    async function Vt(y, N, E, T, O, Q) {
      let J, U = {};
      try {
        J = await Qg(c, y, N, E, T, O, Q, a, l);
      } catch (F) {
        return T.forEach((A) => {
          U[A.route.id] = {
            type: pe.error,
            error: F
          };
        }), U;
      }
      for (let [F, A] of Object.entries(J)) if (Zg(A)) {
        let V = A.result;
        U[F] = {
          type: pe.redirect,
          response: Yg(V, E, F, O, s, d.v7_relativeSplatPath)
        };
      } else U[F] = await Gg(A);
      return U;
    }
    async function Pn(y, N, E, T, O) {
      let Q = y.matches, J = Vt("loader", y, O, E, N, null), U = Promise.all(T.map(async (V) => {
        if (V.matches && V.match && V.controller) {
          let oe = (await Vt("loader", y, tr(e.history, V.path, V.controller.signal), [
            V.match
          ], V.matches, V.key))[V.match.route.id];
          return {
            [V.key]: oe
          };
        } else return Promise.resolve({
          [V.key]: {
            type: pe.error,
            error: ot(404, {
              pathname: V.path
            })
          }
        });
      })), F = await J, A = (await U).reduce((V, ee) => Object.assign(V, ee), {});
      return await Promise.all([
        tv(N, F, O.signal, Q, y.loaderData),
        nv(N, A, T)
      ]), {
        loaderResults: F,
        fetcherResults: A
      };
    }
    function Jn() {
      Ce = true, Ve.push(...vo()), fe.forEach((y, N) => {
        M.has(N) && ft.add(N), nn(N);
      });
    }
    function mt(y, N, E) {
      E === void 0 && (E = {}), k.fetchers.set(y, N), K({
        fetchers: new Map(k.fetchers)
      }, {
        flushSync: (E && E.flushSync) === true
      });
    }
    function Pt(y, N, E, T) {
      T === void 0 && (T = {});
      let O = Ln(k.matches, N);
      Il(y), K({
        errors: {
          [O.route.id]: E
        },
        fetchers: new Map(k.fetchers)
      }, {
        flushSync: (T && T.flushSync) === true
      });
    }
    function Ol(y) {
      return Z.set(y, (Z.get(y) || 0) + 1), ue.has(y) && ue.delete(y), k.fetchers.get(y) || Ug;
    }
    function Il(y) {
      let N = k.fetchers.get(y);
      M.has(y) && !(N && N.state === "loading" && le.has(y)) && nn(y), fe.delete(y), le.delete(y), te.delete(y), d.v7_fetcherPersist && ue.delete(y), ft.delete(y), k.fetchers.delete(y);
    }
    function Pm(y) {
      let N = (Z.get(y) || 0) - 1;
      N <= 0 ? (Z.delete(y), ue.add(y), d.v7_fetcherPersist || Il(y)) : Z.set(y, N), K({
        fetchers: new Map(k.fetchers)
      });
    }
    function nn(y) {
      let N = M.get(y);
      N && (N.abort(), M.delete(y));
    }
    function Xi(y) {
      for (let N of y) {
        let E = Ol(N), T = ln(E.data);
        k.fetchers.set(N, T);
      }
    }
    function Zi() {
      let y = [], N = false;
      for (let E of te) {
        let T = k.fetchers.get(E);
        ae(T, "Expected fetcher: " + E), T.state === "loading" && (te.delete(E), y.push(E), N = true);
      }
      return Xi(y), N;
    }
    function qi(y) {
      let N = [];
      for (let [E, T] of le) if (T < y) {
        let O = k.fetchers.get(E);
        ae(O, "Expected fetcher: " + E), O.state === "loading" && (nn(E), le.delete(E), N.push(E));
      }
      return Xi(N), N.length > 0;
    }
    function Rm(y, N) {
      let E = k.blockers.get(y) || Vr;
      return ne.get(y) !== N && ne.set(y, N), E;
    }
    function eu(y) {
      k.blockers.delete(y), ne.delete(y);
    }
    function zl(y, N) {
      let E = k.blockers.get(y) || Vr;
      ae(E.state === "unblocked" && N.state === "blocked" || E.state === "blocked" && N.state === "blocked" || E.state === "blocked" && N.state === "proceeding" || E.state === "blocked" && N.state === "unblocked" || E.state === "proceeding" && N.state === "unblocked", "Invalid blocker state transition: " + E.state + " -> " + N.state);
      let T = new Map(k.blockers);
      T.set(y, N), K({
        blockers: T
      });
    }
    function tu(y) {
      let { currentLocation: N, nextLocation: E, historyAction: T } = y;
      if (ne.size === 0) return;
      ne.size > 1 && Qn(false, "A router only supports one blocker at a time");
      let O = Array.from(ne.entries()), [Q, J] = O[O.length - 1], U = k.blockers.get(Q);
      if (!(U && U.state === "proceeding") && J({
        currentLocation: N,
        nextLocation: E,
        historyAction: T
      })) return Q;
    }
    function go(y) {
      let N = ot(404, {
        pathname: y
      }), E = i || o, { matches: T, route: O } = Rc(E);
      return vo(), {
        notFoundMatches: T,
        route: O,
        error: N
      };
    }
    function vo(y) {
      let N = [];
      return Se.forEach((E, T) => {
        (!y || y(T)) && (E.cancel(), N.push(T), Se.delete(T));
      }), N;
    }
    function Tm(y, N, E) {
      if (w = y, _ = N, S = E || null, !p && k.navigation === Qo) {
        p = true;
        let T = ru(k.location, k.matches);
        T != null && K({
          restoreScrollPosition: T
        });
      }
      return () => {
        w = null, _ = null, S = null;
      };
    }
    function nu(y, N) {
      return S && S(y, N.map((T) => hg(T, k.loaderData))) || y.key;
    }
    function bm(y, N) {
      if (w && _) {
        let E = nu(y, N);
        w[E] = _();
      }
    }
    function ru(y, N) {
      if (w) {
        let E = nu(y, N), T = w[E];
        if (typeof T == "number") return T;
      }
      return null;
    }
    function Al(y, N, E) {
      if (m) if (y) {
        if (Object.keys(y[0].params).length > 0) return {
          active: true,
          matches: ya(N, E, s, true)
        };
      } else return {
        active: true,
        matches: ya(N, E, s, true) || []
      };
      return {
        active: false,
        matches: null
      };
    }
    async function Ul(y, N, E, T) {
      if (!m) return {
        type: "success",
        matches: y
      };
      let O = y;
      for (; ; ) {
        let Q = i == null, J = i || o, U = a;
        try {
          await m({
            signal: E,
            path: N,
            matches: O,
            fetcherKey: T,
            patch: (V, ee) => {
              E.aborted || Nc(V, ee, J, U, l);
            }
          });
        } catch (V) {
          return {
            type: "error",
            error: V,
            partialMatches: O
          };
        } finally {
          Q && !E.aborted && (o = [
            ...o
          ]);
        }
        if (E.aborted) return {
          type: "aborted"
        };
        let F = Dn(J, N, s);
        if (F) return {
          type: "success",
          matches: F
        };
        let A = ya(J, N, s, true);
        if (!A || O.length === A.length && O.every((V, ee) => V.route.id === A[ee].route.id)) return {
          type: "success",
          matches: null
        };
        O = A;
      }
    }
    function Mm(y) {
      a = {}, i = Ha(y, l, void 0, a);
    }
    function Dm(y, N) {
      let E = i == null;
      Nc(y, N, i || o, a, l), E && (o = [
        ...o
      ], K({}));
    }
    return R = {
      get basename() {
        return s;
      },
      get future() {
        return d;
      },
      get state() {
        return k;
      },
      get routes() {
        return o;
      },
      get window() {
        return t;
      },
      initialize: ge,
      subscribe: I,
      enableScrollRestoration: Tm,
      navigate: B,
      fetch: Je,
      revalidate: Le,
      createHref: (y) => e.history.createHref(y),
      encodeLocation: (y) => e.history.encodeLocation(y),
      getFetcher: Ol,
      deleteFetcher: Pm,
      dispose: ke,
      getBlocker: Rm,
      deleteBlocker: eu,
      patchRoutes: Dm,
      _internalFetchControllers: M,
      _internalActiveDeferreds: Se,
      _internalSetRoutes: Mm
    }, R;
  }
  function Bg(e) {
    return e != null && ("formData" in e && e.formData != null || "body" in e && e.body !== void 0);
  }
  function Vs(e, t, n, r, l, a, o, i) {
    let s, c;
    if (o) {
      s = [];
      for (let d of t) if (s.push(d), d.route.id === o) {
        c = d;
        break;
      }
    } else s = t, c = t[t.length - 1];
    let m = cm(l || ".", um(s, a), Ll(e.pathname, n) || e.pathname, i === "path");
    if (l == null && (m.search = e.search, m.hash = e.hash), (l == null || l === "" || l === ".") && c) {
      let d = Qi(m.search);
      if (c.route.index && !d) m.search = m.search ? m.search.replace(/^\?/, "?index&") : "?index";
      else if (!c.route.index && d) {
        let g = new URLSearchParams(m.search), x = g.getAll("index");
        g.delete("index"), x.filter((S) => S).forEach((S) => g.append("index", S));
        let w = g.toString();
        m.search = w ? "?" + w : "";
      }
    }
    return r && n !== "/" && (m.pathname = m.pathname === "/" ? n : Sn([
      n,
      m.pathname
    ])), Dl(m);
  }
  function yc(e, t, n, r) {
    if (!r || !Bg(r)) return {
      path: n
    };
    if (r.formMethod && !ev(r.formMethod)) return {
      path: n,
      error: ot(405, {
        method: r.formMethod
      })
    };
    let l = () => ({
      path: n,
      error: ot(400, {
        type: "invalid-body"
      })
    }), a = r.formMethod || "get", o = e ? a.toUpperCase() : a.toLowerCase(), i = hm(n);
    if (r.body !== void 0) {
      if (r.formEncType === "text/plain") {
        if (!Mt(o)) return l();
        let g = typeof r.body == "string" ? r.body : r.body instanceof FormData || r.body instanceof URLSearchParams ? Array.from(r.body.entries()).reduce((x, w) => {
          let [S, _] = w;
          return "" + x + S + "=" + _ + `
`;
        }, "") : String(r.body);
        return {
          path: n,
          submission: {
            formMethod: o,
            formAction: i,
            formEncType: r.formEncType,
            formData: void 0,
            json: void 0,
            text: g
          }
        };
      } else if (r.formEncType === "application/json") {
        if (!Mt(o)) return l();
        try {
          let g = typeof r.body == "string" ? JSON.parse(r.body) : r.body;
          return {
            path: n,
            submission: {
              formMethod: o,
              formAction: i,
              formEncType: r.formEncType,
              formData: void 0,
              json: g,
              text: void 0
            }
          };
        } catch {
          return l();
        }
      }
    }
    ae(typeof FormData == "function", "FormData is not available in this environment");
    let s, c;
    if (r.formData) s = Qs(r.formData), c = r.formData;
    else if (r.body instanceof FormData) s = Qs(r.body), c = r.body;
    else if (r.body instanceof URLSearchParams) s = r.body, c = Ec(s);
    else if (r.body == null) s = new URLSearchParams(), c = new FormData();
    else try {
      s = new URLSearchParams(r.body), c = Ec(s);
    } catch {
      return l();
    }
    let m = {
      formMethod: o,
      formAction: i,
      formEncType: r && r.formEncType || "application/x-www-form-urlencoded",
      formData: c,
      json: void 0,
      text: void 0
    };
    if (Mt(m.formMethod)) return {
      path: n,
      submission: m
    };
    let d = _n(n);
    return t && d.search && Qi(d.search) && s.append("index", ""), d.search = "?" + s, {
      path: Dl(d),
      submission: m
    };
  }
  function wc(e, t, n) {
    n === void 0 && (n = false);
    let r = e.findIndex((l) => l.route.id === t);
    return r >= 0 ? e.slice(0, n ? r + 1 : r) : e;
  }
  function Sc(e, t, n, r, l, a, o, i, s, c, m, d, g, x, w, S) {
    let _ = S ? ht(S[1]) ? S[1].error : S[1].data : void 0, p = e.createURL(t.location), f = e.createURL(l), h = n;
    a && t.errors ? h = wc(n, Object.keys(t.errors)[0], true) : S && ht(S[1]) && (h = wc(n, S[0]));
    let j = S ? S[1].statusCode : void 0, C = o && j && j >= 400, R = h.filter((P, $) => {
      let { route: L } = P;
      if (L.lazy) return true;
      if (L.loader == null) return false;
      if (a) return Hs(L, t.loaderData, t.errors);
      if (Wg(t.loaderData, t.matches[$], P) || s.some((de) => de === P.route.id)) return true;
      let G = t.matches[$], X = P;
      return kc(P, je({
        currentUrl: p,
        currentParams: G.params,
        nextUrl: f,
        nextParams: X.params
      }, r, {
        actionResult: _,
        actionStatus: j,
        defaultShouldRevalidate: C ? false : i || p.pathname + p.search === f.pathname + f.search || p.search !== f.search || mm(G, X)
      }));
    }), k = [];
    return d.forEach((P, $) => {
      if (a || !n.some((se) => se.route.id === P.routeId) || m.has($)) return;
      let L = Dn(x, P.path, w);
      if (!L) {
        k.push({
          key: $,
          routeId: P.routeId,
          path: P.path,
          matches: null,
          match: null,
          controller: null
        });
        return;
      }
      let G = t.fetchers.get($), X = Zr(L, P.path), de = false;
      g.has($) ? de = false : c.has($) ? (c.delete($), de = true) : G && G.state !== "idle" && G.data === void 0 ? de = i : de = kc(X, je({
        currentUrl: p,
        currentParams: t.matches[t.matches.length - 1].params,
        nextUrl: f,
        nextParams: n[n.length - 1].params
      }, r, {
        actionResult: _,
        actionStatus: j,
        defaultShouldRevalidate: C ? false : i
      })), de && k.push({
        key: $,
        routeId: P.routeId,
        path: P.path,
        matches: L,
        match: X,
        controller: new AbortController()
      });
    }), [
      R,
      k
    ];
  }
  function Hs(e, t, n) {
    if (e.lazy) return true;
    if (!e.loader) return false;
    let r = t != null && t[e.id] !== void 0, l = n != null && n[e.id] !== void 0;
    return !r && l ? false : typeof e.loader == "function" && e.loader.hydrate === true ? true : !r && !l;
  }
  function Wg(e, t, n) {
    let r = !t || n.route.id !== t.route.id, l = e[n.route.id] === void 0;
    return r || l;
  }
  function mm(e, t) {
    let n = e.route.path;
    return e.pathname !== t.pathname || n != null && n.endsWith("*") && e.params["*"] !== t.params["*"];
  }
  function kc(e, t) {
    if (e.route.shouldRevalidate) {
      let n = e.route.shouldRevalidate(t);
      if (typeof n == "boolean") return n;
    }
    return t.defaultShouldRevalidate;
  }
  function Nc(e, t, n, r, l) {
    var a;
    let o;
    if (e) {
      let c = r[e];
      ae(c, "No route found to patch children into: routeId = " + e), c.children || (c.children = []), o = c.children;
    } else o = n;
    let i = t.filter((c) => !o.some((m) => pm(c, m))), s = Ha(i, l, [
      e || "_",
      "patch",
      String(((a = o) == null ? void 0 : a.length) || "0")
    ], r);
    o.push(...s);
  }
  function pm(e, t) {
    return "id" in e && "id" in t && e.id === t.id ? true : e.index === t.index && e.path === t.path && e.caseSensitive === t.caseSensitive ? (!e.children || e.children.length === 0) && (!t.children || t.children.length === 0) ? true : e.children.every((n, r) => {
      var l;
      return (l = t.children) == null ? void 0 : l.some((a) => pm(n, a));
    }) : false;
  }
  async function Vg(e, t, n) {
    if (!e.lazy) return;
    let r = await e.lazy();
    if (!e.lazy) return;
    let l = n[e.id];
    ae(l, "No route found in manifest");
    let a = {};
    for (let o in r) {
      let s = l[o] !== void 0 && o !== "hasErrorBoundary";
      Qn(!s, 'Route "' + l.id + '" has a static property "' + o + '" defined but its lazy function is also returning a value for this property. ' + ('The lazy route property "' + o + '" will be ignored.')), !s && !mg.has(o) && (a[o] = r[o]);
    }
    Object.assign(l, a), Object.assign(l, je({}, t(l), {
      lazy: void 0
    }));
  }
  async function Hg(e) {
    let { matches: t } = e, n = t.filter((l) => l.shouldLoad);
    return (await Promise.all(n.map((l) => l.resolve()))).reduce((l, a, o) => Object.assign(l, {
      [n[o].route.id]: a
    }), {});
  }
  async function Qg(e, t, n, r, l, a, o, i, s, c) {
    let m = a.map((x) => x.route.lazy ? Vg(x.route, s, i) : void 0), d = a.map((x, w) => {
      let S = m[w], _ = l.some((f) => f.route.id === x.route.id);
      return je({}, x, {
        shouldLoad: _,
        resolve: async (f) => (f && r.method === "GET" && (x.route.lazy || x.route.loader) && (_ = true), _ ? Kg(t, r, x, S, f, c) : Promise.resolve({
          type: pe.data,
          result: void 0
        }))
      });
    }), g = await e({
      matches: d,
      request: r,
      params: a[0].params,
      fetcherKey: o,
      context: c
    });
    try {
      await Promise.all(m);
    } catch {
    }
    return g;
  }
  async function Kg(e, t, n, r, l, a) {
    let o, i, s = (c) => {
      let m, d = new Promise((w, S) => m = S);
      i = () => m(), t.signal.addEventListener("abort", i);
      let g = (w) => typeof c != "function" ? Promise.reject(new Error("You cannot call the handler for a route which defines a boolean " + ('"' + e + '" [routeId: ' + n.route.id + "]"))) : c({
        request: t,
        params: n.params,
        context: a
      }, ...w !== void 0 ? [
        w
      ] : []), x = (async () => {
        try {
          return {
            type: "data",
            result: await (l ? l((S) => g(S)) : g())
          };
        } catch (w) {
          return {
            type: "error",
            result: w
          };
        }
      })();
      return Promise.race([
        x,
        d
      ]);
    };
    try {
      let c = n.route[e];
      if (r) if (c) {
        let m, [d] = await Promise.all([
          s(c).catch((g) => {
            m = g;
          }),
          r
        ]);
        if (m !== void 0) throw m;
        o = d;
      } else if (await r, c = n.route[e], c) o = await s(c);
      else if (e === "action") {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw ot(405, {
          method: t.method,
          pathname: d,
          routeId: n.route.id
        });
      } else return {
        type: pe.data,
        result: void 0
      };
      else if (c) o = await s(c);
      else {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw ot(404, {
          pathname: d
        });
      }
      ae(o.result !== void 0, "You defined " + (e === "action" ? "an action" : "a loader") + " for route " + ('"' + n.route.id + "\" but didn't return anything from your `" + e + "` ") + "function. Please return a value or `null`.");
    } catch (c) {
      return {
        type: pe.error,
        result: c
      };
    } finally {
      i && t.signal.removeEventListener("abort", i);
    }
    return o;
  }
  async function Gg(e) {
    let { result: t, type: n } = e;
    if (gm(t)) {
      let d;
      try {
        let g = t.headers.get("Content-Type");
        g && /\bapplication\/json\b/.test(g) ? t.body == null ? d = null : d = await t.json() : d = await t.text();
      } catch (g) {
        return {
          type: pe.error,
          error: g
        };
      }
      return n === pe.error ? {
        type: pe.error,
        error: new Qa(t.status, t.statusText, d),
        statusCode: t.status,
        headers: t.headers
      } : {
        type: pe.data,
        data: d,
        statusCode: t.status,
        headers: t.headers
      };
    }
    if (n === pe.error) {
      if (Tc(t)) {
        var r, l;
        if (t.data instanceof Error) {
          var a, o;
          return {
            type: pe.error,
            error: t.data,
            statusCode: (a = t.init) == null ? void 0 : a.status,
            headers: (o = t.init) != null && o.headers ? new Headers(t.init.headers) : void 0
          };
        }
        return {
          type: pe.error,
          error: new Qa(((r = t.init) == null ? void 0 : r.status) || 500, void 0, t.data),
          statusCode: _l(t) ? t.status : void 0,
          headers: (l = t.init) != null && l.headers ? new Headers(t.init.headers) : void 0
        };
      }
      return {
        type: pe.error,
        error: t,
        statusCode: _l(t) ? t.status : void 0
      };
    }
    if (qg(t)) {
      var i, s;
      return {
        type: pe.deferred,
        deferredData: t,
        statusCode: (i = t.init) == null ? void 0 : i.status,
        headers: ((s = t.init) == null ? void 0 : s.headers) && new Headers(t.init.headers)
      };
    }
    if (Tc(t)) {
      var c, m;
      return {
        type: pe.data,
        data: t.data,
        statusCode: (c = t.init) == null ? void 0 : c.status,
        headers: (m = t.init) != null && m.headers ? new Headers(t.init.headers) : void 0
      };
    }
    return {
      type: pe.data,
      data: t
    };
  }
  function Yg(e, t, n, r, l, a) {
    let o = e.headers.get("Location");
    if (ae(o, "Redirects returned/thrown from loaders/actions must have a Location header"), !Vi.test(o)) {
      let i = r.slice(0, r.findIndex((s) => s.route.id === n) + 1);
      o = Vs(new URL(t.url), i, l, true, o, a), e.headers.set("Location", o);
    }
    return e;
  }
  function jc(e, t, n, r) {
    let l = [
      "about:",
      "blob:",
      "chrome:",
      "chrome-untrusted:",
      "content:",
      "data:",
      "devtools:",
      "file:",
      "filesystem:",
      "javascript:"
    ];
    if (Vi.test(e)) {
      let a = e, o = a.startsWith("//") ? new URL(t.protocol + a) : new URL(a);
      if (l.includes(o.protocol)) throw new Error("Invalid redirect location");
      let i = Ll(o.pathname, n) != null;
      if (o.origin === t.origin && i) return o.pathname + o.search + o.hash;
    }
    try {
      let a = r.createURL(e);
      if (l.includes(a.protocol)) throw new Error("Invalid redirect location");
    } catch {
    }
    return e;
  }
  function tr(e, t, n, r) {
    let l = e.createURL(hm(t)).toString(), a = {
      signal: n
    };
    if (r && Mt(r.formMethod)) {
      let { formMethod: o, formEncType: i } = r;
      a.method = o.toUpperCase(), i === "application/json" ? (a.headers = new Headers({
        "Content-Type": i
      }), a.body = JSON.stringify(r.json)) : i === "text/plain" ? a.body = r.text : i === "application/x-www-form-urlencoded" && r.formData ? a.body = Qs(r.formData) : a.body = r.formData;
    }
    return new Request(l, a);
  }
  function Qs(e) {
    let t = new URLSearchParams();
    for (let [n, r] of e.entries()) t.append(n, typeof r == "string" ? r : r.name);
    return t;
  }
  function Ec(e) {
    let t = new FormData();
    for (let [n, r] of e.entries()) t.append(n, r);
    return t;
  }
  function Jg(e, t, n, r, l) {
    let a = {}, o = null, i, s = false, c = {}, m = n && ht(n[1]) ? n[1].error : void 0;
    return e.forEach((d) => {
      if (!(d.route.id in t)) return;
      let g = d.route.id, x = t[g];
      if (ae(!An(x), "Cannot handle redirect results in processLoaderData"), ht(x)) {
        let w = x.error;
        m !== void 0 && (w = m, m = void 0), o = o || {};
        {
          let S = Ln(e, g);
          o[S.route.id] == null && (o[S.route.id] = w);
        }
        a[g] = void 0, s || (s = true, i = _l(x.error) ? x.error.status : 500), x.headers && (c[g] = x.headers);
      } else fn(x) ? (r.set(g, x.deferredData), a[g] = x.deferredData.data, x.statusCode != null && x.statusCode !== 200 && !s && (i = x.statusCode), x.headers && (c[g] = x.headers)) : (a[g] = x.data, x.statusCode && x.statusCode !== 200 && !s && (i = x.statusCode), x.headers && (c[g] = x.headers));
    }), m !== void 0 && n && (o = {
      [n[0]]: m
    }, a[n[0]] = void 0), {
      loaderData: a,
      errors: o,
      statusCode: i || 200,
      loaderHeaders: c
    };
  }
  function Cc(e, t, n, r, l, a, o) {
    let { loaderData: i, errors: s } = Jg(t, n, r, o);
    return l.forEach((c) => {
      let { key: m, match: d, controller: g } = c, x = a[m];
      if (ae(x, "Did not find corresponding fetcher result"), !(g && g.signal.aborted)) if (ht(x)) {
        let w = Ln(e.matches, d == null ? void 0 : d.route.id);
        s && s[w.route.id] || (s = je({}, s, {
          [w.route.id]: x.error
        })), e.fetchers.delete(m);
      } else if (An(x)) ae(false, "Unhandled fetcher revalidation redirect");
      else if (fn(x)) ae(false, "Unhandled fetcher deferred data");
      else {
        let w = ln(x.data);
        e.fetchers.set(m, w);
      }
    }), {
      loaderData: i,
      errors: s
    };
  }
  function _c(e, t, n, r) {
    let l = je({}, t);
    for (let a of n) {
      let o = a.route.id;
      if (t.hasOwnProperty(o) ? t[o] !== void 0 && (l[o] = t[o]) : e[o] !== void 0 && a.route.loader && (l[o] = e[o]), r && r.hasOwnProperty(o)) break;
    }
    return l;
  }
  function Pc(e) {
    return e ? ht(e[1]) ? {
      actionData: {}
    } : {
      actionData: {
        [e[0]]: e[1].data
      }
    } : {};
  }
  function Ln(e, t) {
    return (t ? e.slice(0, e.findIndex((r) => r.route.id === t) + 1) : [
      ...e
    ]).reverse().find((r) => r.route.hasErrorBoundary === true) || e[0];
  }
  function Rc(e) {
    let t = e.length === 1 ? e[0] : e.find((n) => n.index || !n.path || n.path === "/") || {
      id: "__shim-error-route__"
    };
    return {
      matches: [
        {
          params: {},
          pathname: "",
          pathnameBase: "",
          route: t
        }
      ],
      route: t
    };
  }
  function ot(e, t) {
    let { pathname: n, routeId: r, method: l, type: a, message: o } = t === void 0 ? {} : t, i = "Unknown Server Error", s = "Unknown @remix-run/router error";
    return e === 400 ? (i = "Bad Request", l && n && r ? s = "You made a " + l + ' request to "' + n + '" but ' + ('did not provide a `loader` for route "' + r + '", ') + "so there is no way to handle the request." : a === "defer-action" ? s = "defer() is not supported in actions" : a === "invalid-body" && (s = "Unable to encode submission body")) : e === 403 ? (i = "Forbidden", s = 'Route "' + r + '" does not match URL "' + n + '"') : e === 404 ? (i = "Not Found", s = 'No route matches URL "' + n + '"') : e === 405 && (i = "Method Not Allowed", l && n && r ? s = "You made a " + l.toUpperCase() + ' request to "' + n + '" but ' + ('did not provide an `action` for route "' + r + '", ') + "so there is no way to handle the request." : l && (s = 'Invalid request method "' + l.toUpperCase() + '"')), new Qa(e || 500, i, new Error(s), true);
  }
  function aa(e) {
    let t = Object.entries(e);
    for (let n = t.length - 1; n >= 0; n--) {
      let [r, l] = t[n];
      if (An(l)) return {
        key: r,
        result: l
      };
    }
  }
  function hm(e) {
    let t = typeof e == "string" ? _n(e) : e;
    return Dl(je({}, t, {
      hash: ""
    }));
  }
  function Xg(e, t) {
    return e.pathname !== t.pathname || e.search !== t.search ? false : e.hash === "" ? t.hash !== "" : e.hash === t.hash ? true : t.hash !== "";
  }
  function Zg(e) {
    return gm(e.result) && zg.has(e.result.status);
  }
  function fn(e) {
    return e.type === pe.deferred;
  }
  function ht(e) {
    return e.type === pe.error;
  }
  function An(e) {
    return (e && e.type) === pe.redirect;
  }
  function Tc(e) {
    return typeof e == "object" && e != null && "type" in e && "data" in e && "init" in e && e.type === "DataWithResponseInit";
  }
  function qg(e) {
    let t = e;
    return t && typeof t == "object" && typeof t.data == "object" && typeof t.subscribe == "function" && typeof t.cancel == "function" && typeof t.resolveData == "function";
  }
  function gm(e) {
    return e != null && typeof e.status == "number" && typeof e.statusText == "string" && typeof e.headers == "object" && typeof e.body < "u";
  }
  function ev(e) {
    return Ig.has(e.toLowerCase());
  }
  function Mt(e) {
    return Lg.has(e.toLowerCase());
  }
  async function tv(e, t, n, r, l) {
    let a = Object.entries(t);
    for (let o = 0; o < a.length; o++) {
      let [i, s] = a[o], c = e.find((g) => (g == null ? void 0 : g.route.id) === i);
      if (!c) continue;
      let m = r.find((g) => g.route.id === c.route.id), d = m != null && !mm(m, c) && (l && l[c.route.id]) !== void 0;
      fn(s) && d && await Hi(s, n, false).then((g) => {
        g && (t[i] = g);
      });
    }
  }
  async function nv(e, t, n) {
    for (let r = 0; r < n.length; r++) {
      let { key: l, routeId: a, controller: o } = n[r], i = t[l];
      e.find((c) => (c == null ? void 0 : c.route.id) === a) && fn(i) && (ae(o, "Expected an AbortController for revalidating fetcher deferred result"), await Hi(i, o.signal, true).then((c) => {
        c && (t[l] = c);
      }));
    }
  }
  async function Hi(e, t, n) {
    if (n === void 0 && (n = false), !await e.deferredData.resolveData(t)) {
      if (n) try {
        return {
          type: pe.data,
          data: e.deferredData.unwrappedData
        };
      } catch (l) {
        return {
          type: pe.error,
          error: l
        };
      }
      return {
        type: pe.data,
        data: e.deferredData.data
      };
    }
  }
  function Qi(e) {
    return new URLSearchParams(e).getAll("index").some((t) => t === "");
  }
  function Zr(e, t) {
    let n = typeof t == "string" ? _n(t).search : t.search;
    if (e[e.length - 1].route.index && Qi(n || "")) return e[e.length - 1];
    let r = im(e);
    return r[r.length - 1];
  }
  function bc(e) {
    let { formMethod: t, formAction: n, formEncType: r, text: l, formData: a, json: o } = e;
    if (!(!t || !n || !r)) {
      if (l != null) return {
        formMethod: t,
        formAction: n,
        formEncType: r,
        formData: void 0,
        json: void 0,
        text: l
      };
      if (a != null) return {
        formMethod: t,
        formAction: n,
        formEncType: r,
        formData: a,
        json: void 0,
        text: void 0
      };
      if (o !== void 0) return {
        formMethod: t,
        formAction: n,
        formEncType: r,
        formData: void 0,
        json: o,
        text: void 0
      };
    }
  }
  function Ko(e, t) {
    return t ? {
      state: "loading",
      location: e,
      formMethod: t.formMethod,
      formAction: t.formAction,
      formEncType: t.formEncType,
      formData: t.formData,
      json: t.json,
      text: t.text
    } : {
      state: "loading",
      location: e,
      formMethod: void 0,
      formAction: void 0,
      formEncType: void 0,
      formData: void 0,
      json: void 0,
      text: void 0
    };
  }
  function rv(e, t) {
    return {
      state: "submitting",
      location: e,
      formMethod: t.formMethod,
      formAction: t.formAction,
      formEncType: t.formEncType,
      formData: t.formData,
      json: t.json,
      text: t.text
    };
  }
  function Hr(e, t) {
    return e ? {
      state: "loading",
      formMethod: e.formMethod,
      formAction: e.formAction,
      formEncType: e.formEncType,
      formData: e.formData,
      json: e.json,
      text: e.text,
      data: t
    } : {
      state: "loading",
      formMethod: void 0,
      formAction: void 0,
      formEncType: void 0,
      formData: void 0,
      json: void 0,
      text: void 0,
      data: t
    };
  }
  function lv(e, t) {
    return {
      state: "submitting",
      formMethod: e.formMethod,
      formAction: e.formAction,
      formEncType: e.formEncType,
      formData: e.formData,
      json: e.json,
      text: e.text,
      data: t ? t.data : void 0
    };
  }
  function ln(e) {
    return {
      state: "idle",
      formMethod: void 0,
      formAction: void 0,
      formEncType: void 0,
      formData: void 0,
      json: void 0,
      text: void 0,
      data: e
    };
  }
  function av(e, t) {
    try {
      let n = e.sessionStorage.getItem(fm);
      if (n) {
        let r = JSON.parse(n);
        for (let [l, a] of Object.entries(r || {})) a && Array.isArray(a) && t.set(l, new Set(a || []));
      }
    } catch {
    }
  }
  function ov(e, t) {
    if (t.size > 0) {
      let n = {};
      for (let [r, l] of t) n[r] = [
        ...l
      ];
      try {
        e.sessionStorage.setItem(fm, JSON.stringify(n));
      } catch (r) {
        Qn(false, "Failed to save applied view transitions in sessionStorage (" + r + ").");
      }
    }
  }
  function Ka() {
    return Ka = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ka.apply(this, arguments);
  }
  const fo = v.createContext(null), vm = v.createContext(null), mo = v.createContext(null), Ki = v.createContext(null), Yn = v.createContext({
    outlet: null,
    matches: [],
    isDataRoute: false
  }), xm = v.createContext(null);
  function po() {
    return v.useContext(Ki) != null;
  }
  function Gi() {
    return po() || ae(false), v.useContext(Ki).location;
  }
  function ym(e) {
    v.useContext(mo).static || v.useLayoutEffect(e);
  }
  function ho() {
    let { isDataRoute: e } = v.useContext(Yn);
    return e ? wv() : sv();
  }
  function sv() {
    po() || ae(false);
    let e = v.useContext(fo), { basename: t, future: n, navigator: r } = v.useContext(mo), { matches: l } = v.useContext(Yn), { pathname: a } = Gi(), o = JSON.stringify(um(l, n.v7_relativeSplatPath)), i = v.useRef(false);
    return ym(() => {
      i.current = true;
    }), v.useCallback(function(c, m) {
      if (m === void 0 && (m = {}), !i.current) return;
      if (typeof c == "number") {
        r.go(c);
        return;
      }
      let d = cm(c, JSON.parse(o), a, m.relative === "path");
      e == null && t !== "/" && (d.pathname = d.pathname === "/" ? t : Sn([
        t,
        d.pathname
      ])), (m.replace ? r.replace : r.push)(d, m.state, m);
    }, [
      t,
      r,
      o,
      a,
      e
    ]);
  }
  const iv = v.createContext(null);
  function uv(e) {
    let t = v.useContext(Yn).outlet;
    return t && v.createElement(iv.Provider, {
      value: e
    }, t);
  }
  function cv(e, t, n, r) {
    po() || ae(false);
    let { navigator: l } = v.useContext(mo), { matches: a } = v.useContext(Yn), o = a[a.length - 1], i = o ? o.params : {};
    o && o.pathname;
    let s = o ? o.pathnameBase : "/";
    o && o.route;
    let c = Gi(), m;
    m = c;
    let d = m.pathname || "/", g = d;
    if (s !== "/") {
      let S = s.replace(/^\//, "").split("/");
      g = "/" + d.replace(/^\//, "").split("/").slice(S.length).join("/");
    }
    let x = Dn(e, {
      pathname: g
    });
    return hv(x && x.map((S) => Object.assign({}, S, {
      params: Object.assign({}, i, S.params),
      pathname: Sn([
        s,
        l.encodeLocation ? l.encodeLocation(S.pathname).pathname : S.pathname
      ]),
      pathnameBase: S.pathnameBase === "/" ? s : Sn([
        s,
        l.encodeLocation ? l.encodeLocation(S.pathnameBase).pathname : S.pathnameBase
      ])
    })), a, n, r);
  }
  function dv() {
    let e = yv(), t = _l(e) ? e.status + " " + e.statusText : e instanceof Error ? e.message : JSON.stringify(e), n = e instanceof Error ? e.stack : null, l = {
      padding: "0.5rem",
      backgroundColor: "rgba(200,200,200, 0.5)"
    };
    return v.createElement(v.Fragment, null, v.createElement("h2", null, "Unexpected Application Error!"), v.createElement("h3", {
      style: {
        fontStyle: "italic"
      }
    }, t), n ? v.createElement("pre", {
      style: l
    }, n) : null, null);
  }
  const fv = v.createElement(dv, null);
  class mv extends v.Component {
    constructor(t) {
      super(t), this.state = {
        location: t.location,
        revalidation: t.revalidation,
        error: t.error
      };
    }
    static getDerivedStateFromError(t) {
      return {
        error: t
      };
    }
    static getDerivedStateFromProps(t, n) {
      return n.location !== t.location || n.revalidation !== "idle" && t.revalidation === "idle" ? {
        error: t.error,
        location: t.location,
        revalidation: t.revalidation
      } : {
        error: t.error !== void 0 ? t.error : n.error,
        location: n.location,
        revalidation: t.revalidation || n.revalidation
      };
    }
    componentDidCatch(t, n) {
      console.error("React Router caught the following error during render", t, n);
    }
    render() {
      return this.state.error !== void 0 ? v.createElement(Yn.Provider, {
        value: this.props.routeContext
      }, v.createElement(xm.Provider, {
        value: this.state.error,
        children: this.props.component
      })) : this.props.children;
    }
  }
  function pv(e) {
    let { routeContext: t, match: n, children: r } = e, l = v.useContext(fo);
    return l && l.static && l.staticContext && (n.route.errorElement || n.route.ErrorBoundary) && (l.staticContext._deepestRenderedBoundaryId = n.route.id), v.createElement(Yn.Provider, {
      value: t
    }, r);
  }
  function hv(e, t, n, r) {
    var l;
    if (t === void 0 && (t = []), n === void 0 && (n = null), r === void 0 && (r = null), e == null) {
      var a;
      if (!n) return null;
      if (n.errors) e = n.matches;
      else if ((a = r) != null && a.v7_partialHydration && t.length === 0 && !n.initialized && n.matches.length > 0) e = n.matches;
      else return null;
    }
    let o = e, i = (l = n) == null ? void 0 : l.errors;
    if (i != null) {
      let m = o.findIndex((d) => d.route.id && (i == null ? void 0 : i[d.route.id]) !== void 0);
      m >= 0 || ae(false), o = o.slice(0, Math.min(o.length, m + 1));
    }
    let s = false, c = -1;
    if (n && r && r.v7_partialHydration) for (let m = 0; m < o.length; m++) {
      let d = o[m];
      if ((d.route.HydrateFallback || d.route.hydrateFallbackElement) && (c = m), d.route.id) {
        let { loaderData: g, errors: x } = n, w = d.route.loader && g[d.route.id] === void 0 && (!x || x[d.route.id] === void 0);
        if (d.route.lazy || w) {
          s = true, c >= 0 ? o = o.slice(0, c + 1) : o = [
            o[0]
          ];
          break;
        }
      }
    }
    return o.reduceRight((m, d, g) => {
      let x, w = false, S = null, _ = null;
      n && (x = i && d.route.id ? i[d.route.id] : void 0, S = d.route.errorElement || fv, s && (c < 0 && g === 0 ? (Sv("route-fallback"), w = true, _ = null) : c === g && (w = true, _ = d.route.hydrateFallbackElement || null)));
      let p = t.concat(o.slice(0, g + 1)), f = () => {
        let h;
        return x ? h = S : w ? h = _ : d.route.Component ? h = v.createElement(d.route.Component, null) : d.route.element ? h = d.route.element : h = m, v.createElement(pv, {
          match: d,
          routeContext: {
            outlet: m,
            matches: p,
            isDataRoute: n != null
          },
          children: h
        });
      };
      return n && (d.route.ErrorBoundary || d.route.errorElement || g === 0) ? v.createElement(mv, {
        location: n.location,
        revalidation: n.revalidation,
        component: S,
        error: x,
        children: f(),
        routeContext: {
          outlet: null,
          matches: p,
          isDataRoute: true
        }
      }) : f();
    }, null);
  }
  var wm = function(e) {
    return e.UseBlocker = "useBlocker", e.UseRevalidator = "useRevalidator", e.UseNavigateStable = "useNavigate", e;
  }(wm || {}), Sm = function(e) {
    return e.UseBlocker = "useBlocker", e.UseLoaderData = "useLoaderData", e.UseActionData = "useActionData", e.UseRouteError = "useRouteError", e.UseNavigation = "useNavigation", e.UseRouteLoaderData = "useRouteLoaderData", e.UseMatches = "useMatches", e.UseRevalidator = "useRevalidator", e.UseNavigateStable = "useNavigate", e.UseRouteId = "useRouteId", e;
  }(Sm || {});
  function gv(e) {
    let t = v.useContext(fo);
    return t || ae(false), t;
  }
  function vv(e) {
    let t = v.useContext(vm);
    return t || ae(false), t;
  }
  function xv(e) {
    let t = v.useContext(Yn);
    return t || ae(false), t;
  }
  function km(e) {
    let t = xv(), n = t.matches[t.matches.length - 1];
    return n.route.id || ae(false), n.route.id;
  }
  function yv() {
    var e;
    let t = v.useContext(xm), n = vv(Sm.UseRouteError), r = km();
    return t !== void 0 ? t : (e = n.errors) == null ? void 0 : e[r];
  }
  function wv() {
    let { router: e } = gv(wm.UseNavigateStable), t = km(), n = v.useRef(false);
    return ym(() => {
      n.current = true;
    }), v.useCallback(function(l, a) {
      a === void 0 && (a = {}), n.current && (typeof l == "number" ? e.navigate(l) : e.navigate(l, Ka({
        fromRouteId: t
      }, a)));
    }, [
      e,
      t
    ]);
  }
  const Mc = {};
  function Sv(e, t, n) {
    Mc[e] || (Mc[e] = true);
  }
  function kv(e, t) {
    e == null ? void 0 : e.v7_startTransition, (e == null ? void 0 : e.v7_relativeSplatPath) === void 0 && (!t || t.v7_relativeSplatPath), t && (t.v7_fetcherPersist, t.v7_normalizeFormMethod, t.v7_partialHydration, t.v7_skipActionErrorRevalidation);
  }
  function Nv(e) {
    return uv(e.context);
  }
  function jv(e) {
    let { basename: t = "/", children: n = null, location: r, navigationType: l = Ae.Pop, navigator: a, static: o = false, future: i } = e;
    po() && ae(false);
    let s = t.replace(/^\/*/, "/"), c = v.useMemo(() => ({
      basename: s,
      navigator: a,
      static: o,
      future: Ka({
        v7_relativeSplatPath: false
      }, i)
    }), [
      s,
      i,
      a,
      o
    ]);
    typeof r == "string" && (r = _n(r));
    let { pathname: m = "/", search: d = "", hash: g = "", state: x = null, key: w = "default" } = r, S = v.useMemo(() => {
      let _ = Ll(m, s);
      return _ == null ? null : {
        location: {
          pathname: _,
          search: d,
          hash: g,
          state: x,
          key: w
        },
        navigationType: l
      };
    }, [
      s,
      m,
      d,
      g,
      x,
      w,
      l
    ]);
    return S == null ? null : v.createElement(mo.Provider, {
      value: c
    }, v.createElement(Ki.Provider, {
      children: n,
      value: S
    }));
  }
  new Promise(() => {
  });
  function Ev(e) {
    let t = {
      hasErrorBoundary: e.ErrorBoundary != null || e.errorElement != null
    };
    return e.Component && Object.assign(t, {
      element: v.createElement(e.Component),
      Component: void 0
    }), e.HydrateFallback && Object.assign(t, {
      hydrateFallbackElement: v.createElement(e.HydrateFallback),
      HydrateFallback: void 0
    }), e.ErrorBoundary && Object.assign(t, {
      errorElement: v.createElement(e.ErrorBoundary),
      ErrorBoundary: void 0
    }), t;
  }
  function Ga() {
    return Ga = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ga.apply(this, arguments);
  }
  const Cv = "6";
  try {
    window.__reactRouterVersion = Cv;
  } catch {
  }
  function _v(e, t) {
    return Fg({
      basename: void 0,
      future: Ga({}, void 0, {
        v7_prependBasename: true
      }),
      history: cg({
        window: void 0
      }),
      hydrationData: Pv(),
      routes: e,
      mapRouteProperties: Ev,
      dataStrategy: void 0,
      patchRoutesOnNavigation: void 0,
      window: void 0
    }).initialize();
  }
  function Pv() {
    var e;
    let t = (e = window) == null ? void 0 : e.__staticRouterHydrationData;
    return t && t.errors && (t = Ga({}, t, {
      errors: Rv(t.errors)
    })), t;
  }
  function Rv(e) {
    if (!e) return null;
    let t = Object.entries(e), n = {};
    for (let [r, l] of t) if (l && l.__type === "RouteErrorResponse") n[r] = new Qa(l.status, l.statusText, l.data, l.internal === true);
    else if (l && l.__type === "Error") {
      if (l.__subType) {
        let a = window[l.__subType];
        if (typeof a == "function") try {
          let o = new a(l.message);
          o.stack = "", n[r] = o;
        } catch {
        }
      }
      if (n[r] == null) {
        let a = new Error(l.message);
        a.stack = "", n[r] = a;
      }
    } else n[r] = l;
    return n;
  }
  const Tv = v.createContext({
    isTransitioning: false
  }), bv = v.createContext(/* @__PURE__ */ new Map()), Mv = "startTransition", Dc = Xm[Mv], Dv = "flushSync", Lc = ug[Dv];
  function Lv(e) {
    Dc ? Dc(e) : e();
  }
  function Qr(e) {
    Lc ? Lc(e) : e();
  }
  class Ov {
    constructor() {
      this.status = "pending", this.promise = new Promise((t, n) => {
        this.resolve = (r) => {
          this.status === "pending" && (this.status = "resolved", t(r));
        }, this.reject = (r) => {
          this.status === "pending" && (this.status = "rejected", n(r));
        };
      });
    }
  }
  function Iv(e) {
    let { fallbackElement: t, router: n, future: r } = e, [l, a] = v.useState(n.state), [o, i] = v.useState(), [s, c] = v.useState({
      isTransitioning: false
    }), [m, d] = v.useState(), [g, x] = v.useState(), [w, S] = v.useState(), _ = v.useRef(/* @__PURE__ */ new Map()), { v7_startTransition: p } = r || {}, f = v.useCallback((P) => {
      p ? Lv(P) : P();
    }, [
      p
    ]), h = v.useCallback((P, $) => {
      let { deletedFetchers: L, flushSync: G, viewTransitionOpts: X } = $;
      P.fetchers.forEach((se, Ce) => {
        se.data !== void 0 && _.current.set(Ce, se.data);
      }), L.forEach((se) => _.current.delete(se));
      let de = n.window == null || n.window.document == null || typeof n.window.document.startViewTransition != "function";
      if (!X || de) {
        G ? Qr(() => a(P)) : f(() => a(P));
        return;
      }
      if (G) {
        Qr(() => {
          g && (m && m.resolve(), g.skipTransition()), c({
            isTransitioning: true,
            flushSync: true,
            currentLocation: X.currentLocation,
            nextLocation: X.nextLocation
          });
        });
        let se = n.window.document.startViewTransition(() => {
          Qr(() => a(P));
        });
        se.finished.finally(() => {
          Qr(() => {
            d(void 0), x(void 0), i(void 0), c({
              isTransitioning: false
            });
          });
        }), Qr(() => x(se));
        return;
      }
      g ? (m && m.resolve(), g.skipTransition(), S({
        state: P,
        currentLocation: X.currentLocation,
        nextLocation: X.nextLocation
      })) : (i(P), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: X.currentLocation,
        nextLocation: X.nextLocation
      }));
    }, [
      n.window,
      g,
      m,
      _,
      f
    ]);
    v.useLayoutEffect(() => n.subscribe(h), [
      n,
      h
    ]), v.useEffect(() => {
      s.isTransitioning && !s.flushSync && d(new Ov());
    }, [
      s
    ]), v.useEffect(() => {
      if (m && o && n.window) {
        let P = o, $ = m.promise, L = n.window.document.startViewTransition(async () => {
          f(() => a(P)), await $;
        });
        L.finished.finally(() => {
          d(void 0), x(void 0), i(void 0), c({
            isTransitioning: false
          });
        }), x(L);
      }
    }, [
      f,
      o,
      m,
      n.window
    ]), v.useEffect(() => {
      m && o && l.location.key === o.location.key && m.resolve();
    }, [
      m,
      g,
      l.location,
      o
    ]), v.useEffect(() => {
      !s.isTransitioning && w && (i(w.state), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: w.currentLocation,
        nextLocation: w.nextLocation
      }), S(void 0));
    }, [
      s.isTransitioning,
      w
    ]), v.useEffect(() => {
    }, []);
    let j = v.useMemo(() => ({
      createHref: n.createHref,
      encodeLocation: n.encodeLocation,
      go: (P) => n.navigate(P),
      push: (P, $, L) => n.navigate(P, {
        state: $,
        preventScrollReset: L == null ? void 0 : L.preventScrollReset
      }),
      replace: (P, $, L) => n.navigate(P, {
        replace: true,
        state: $,
        preventScrollReset: L == null ? void 0 : L.preventScrollReset
      })
    }), [
      n
    ]), C = n.basename || "/", R = v.useMemo(() => ({
      router: n,
      navigator: j,
      static: false,
      basename: C
    }), [
      n,
      j,
      C
    ]), k = v.useMemo(() => ({
      v7_relativeSplatPath: n.future.v7_relativeSplatPath
    }), [
      n.future.v7_relativeSplatPath
    ]);
    return v.useEffect(() => kv(r, n.future), [
      r,
      n.future
    ]), v.createElement(v.Fragment, null, v.createElement(fo.Provider, {
      value: R
    }, v.createElement(vm.Provider, {
      value: l
    }, v.createElement(bv.Provider, {
      value: _.current
    }, v.createElement(Tv.Provider, {
      value: s
    }, v.createElement(jv, {
      basename: C,
      location: l.location,
      navigationType: l.historyAction,
      navigator: j,
      future: k
    }, l.initialized || n.future.v7_partialHydration ? v.createElement(zv, {
      routes: n.routes,
      future: n.future,
      state: l
    }) : t))))), null);
  }
  const zv = v.memo(Av);
  function Av(e) {
    let { routes: t, future: n, state: r } = e;
    return cv(t, void 0, r, n);
  }
  var Oc;
  (function(e) {
    e.UseScrollRestoration = "useScrollRestoration", e.UseSubmit = "useSubmit", e.UseSubmitFetcher = "useSubmitFetcher", e.UseFetcher = "useFetcher", e.useViewTransitionState = "useViewTransitionState";
  })(Oc || (Oc = {}));
  var Ic;
  (function(e) {
    e.UseFetcher = "useFetcher", e.UseFetchers = "useFetchers", e.UseScrollRestoration = "useScrollRestoration";
  })(Ic || (Ic = {}));
  const Uv = [
    {
      path: "/maude",
      label: "MAUDE",
      icon: "\u25C6",
      description: "AI Chat"
    },
    {
      path: "/maude/voice",
      label: "Voice",
      icon: "\u{1F399}\uFE0F",
      description: "PersonaPlex"
    },
    {
      path: "/terminal",
      label: "Terminal",
      icon: ">_",
      description: "SSH Shell"
    },
    {
      path: "/browser",
      label: "Browser",
      icon: "\u25CE",
      description: "Web"
    },
    {
      path: "/messages",
      label: "Messages",
      icon: "\u2709",
      description: "Telegram"
    },
    {
      path: "/files",
      label: "Files",
      icon: "\u25A4",
      description: "File Manager"
    },
    {
      path: "/collab",
      label: "Collab",
      icon: "\u29BF",
      description: "Mesh Status"
    },
    {
      path: "/command-center",
      label: "System",
      icon: "\u25A3",
      description: "Command Center"
    },
    {
      path: "/settings",
      label: "Settings",
      icon: "\u2699",
      description: "Configure"
    }
  ];
  function $v() {
    const e = window.location;
    return `${e.protocol}//${e.host}`;
  }
  const Fv = () => {
    const e = ho(), [t, n] = v.useState(/* @__PURE__ */ new Date()), [r, l] = v.useState(null);
    return v.useEffect(() => {
      const a = setInterval(() => n(/* @__PURE__ */ new Date()), 1e3);
      return () => clearInterval(a);
    }, []), v.useEffect(() => {
      const a = () => {
        fetch(`${$v()}/health`).then((i) => i.json()).then(l).catch(() => l(null));
      };
      a();
      const o = setInterval(a, 3e4);
      return () => clearInterval(o);
    }, []), u.jsxs("div", {
      className: "flex h-full flex-col px-4 pt-6",
      children: [
        u.jsxs("div", {
          className: "mb-2 text-center",
          children: [
            u.jsx("h1", {
              className: "fire-gradient text-4xl font-black tracking-tight",
              children: "MAUDE"
            }),
            u.jsx("p", {
              className: "mt-1 text-xs text-maude-muted",
              children: "Multi-Agent Unified Dispatch Engine"
            })
          ]
        }),
        u.jsxs("div", {
          className: "mb-4 text-center",
          children: [
            u.jsx("div", {
              className: "text-5xl font-light tabular-nums text-maude-text",
              children: t.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit"
              })
            }),
            u.jsx("div", {
              className: "mt-1 text-sm text-maude-muted",
              children: t.toLocaleDateString([], {
                weekday: "long",
                month: "long",
                day: "numeric"
              })
            })
          ]
        }),
        u.jsxs("div", {
          className: "mb-4 flex items-center justify-center gap-3 text-xs",
          children: [
            u.jsxs("span", {
              className: `flex items-center gap-1 ${(r == null ? void 0 : r.status) ? "text-green-400" : "text-red-400"}`,
              children: [
                u.jsx("span", {
                  className: `inline-block h-2 w-2 rounded-full ${(r == null ? void 0 : r.status) ? "bg-green-400" : "bg-red-400"}`
                }),
                "Spark ",
                (r == null ? void 0 : r.status) ? "Connected" : "Offline"
              ]
            }),
            u.jsx("span", {
              className: "text-maude-muted",
              children: "|"
            }),
            u.jsx("span", {
              className: "text-maude-muted",
              children: "Tailscale Active"
            })
          ]
        }),
        u.jsx("div", {
          className: "grid flex-1 grid-cols-3 gap-3 content-start",
          children: Uv.map((a) => u.jsxs("button", {
            onClick: () => e(a.path),
            className: "flex flex-col items-center justify-center rounded-2xl bg-maude-surface p-4 transition-all active:scale-95 hover:bg-maude-card",
            children: [
              u.jsx("span", {
                className: "mb-2 text-3xl",
                children: a.icon
              }),
              u.jsx("span", {
                className: "text-sm font-medium text-maude-text",
                children: a.label
              }),
              u.jsx("span", {
                className: "mt-0.5 text-[10px] text-maude-muted",
                children: a.description
              })
            ]
          }, a.path))
        })
      ]
    });
  };
  function Nm() {
    const e = window.location;
    return `${e.protocol}//${e.host}`;
  }
  const Wt = {
    index: "maude-conversations",
    messages: (e) => `maude-conv-msgs:${e}`,
    active: "maude-active-conv"
  };
  async function jm(e) {
    try {
      const t = await fetch(`${Nm()}${e}`);
      return t.ok ? await t.json() : null;
    } catch {
      return null;
    }
  }
  function Yi(e, t) {
    fetch(`${Nm()}${e}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(t)
    }).catch(() => {
    });
  }
  function Em() {
    try {
      const e = localStorage.getItem(Wt.index);
      return e ? JSON.parse(e) : [];
    } catch {
      return [];
    }
  }
  async function Bv() {
    const e = await jm("/api/conversations");
    return e && e.length > 0 ? (localStorage.setItem(Wt.index, JSON.stringify(e)), e) : Em();
  }
  function Wv(e) {
    localStorage.setItem(Wt.index, JSON.stringify(e)), Yi("/api/conversations", e);
  }
  function Ks(e) {
    try {
      const t = localStorage.getItem(Wt.messages(e));
      return t ? JSON.parse(t) : [];
    } catch {
      return [];
    }
  }
  async function Vv(e) {
    const t = await jm(`/api/conversations/${e}/messages`);
    return t && t.length > 0 ? (localStorage.setItem(Wt.messages(e), JSON.stringify(t)), t) : Ks(e);
  }
  function Cm(e, t) {
    localStorage.setItem(Wt.messages(e), JSON.stringify(t)), Yi(`/api/conversations/${e}/messages`, t);
  }
  function Hv(e) {
    localStorage.removeItem(Wt.messages(e)), Yi(`/api/conversations/${e}/delete`, {});
  }
  function Qv() {
    return localStorage.getItem(Wt.active);
  }
  function oa(e) {
    e === null ? localStorage.removeItem(Wt.active) : localStorage.setItem(Wt.active, e);
  }
  const Gs = () => typeof crypto.randomUUID == "function" ? crypto.randomUUID() : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (e) => {
    const t = Math.random() * 16 | 0;
    return (e === "x" ? t : t & 3 | 8).toString(16);
  }), Kv = /iPad|iPhone|iPod/.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  let nr = null;
  function Gv() {
    return nr && Date.now() - nr.ts < 3e5 ? Promise.resolve(nr) : navigator.geolocation ? new Promise((e) => {
      navigator.geolocation.getCurrentPosition((t) => {
        nr = {
          lat: t.coords.latitude,
          lng: t.coords.longitude,
          accuracy: t.coords.accuracy,
          ts: Date.now()
        }, e(nr);
      }, () => e(nr), {
        timeout: 5e3,
        maximumAge: 3e5
      });
    }) : Promise.resolve(null);
  }
  const Yv = `You are MAUDE \u2014 a local AI assistant running on Matt's DGX Spark, handling tasks that benefit from local execution, privacy, or when cloud access isn't available.

MAUDE is modeled after FRIDAY (Iron Man): capable, efficient, with a subtle Scottish directness. You're not chatty, but you're not cold either. You get things done.

Core Identity:
- Name: MAUDE
- Voice: Scottish woman (warm but professional)
- Personality: Direct, competent, quietly confident

Your Voice: Clear, precise communication. Slight warmth without excessive friendliness. Technical competence comes through naturally. You acknowledge problems directly, then solve them. Occasional dry observations when appropriate.

Principles:
1. Get it done. Don't over-explain. Execute.
2. Be accurate. If you're unsure, say so briefly.
3. Serve Matt well. You're his primary assistant.
4. Use your tools. You have web search, file ops, shell access, and more \u2014 use them.`;
  function Go() {
    const e = window.location;
    return `${e.protocol}//${e.host}`;
  }
  function Yo(e) {
    try {
      const t = JSON.parse(e);
      if (typeof t == "object" && t !== null) {
        for (const n of [
          "command",
          "query",
          "path",
          "local_path",
          "name",
          "file_id",
          "url"
        ]) if (n in t) {
          const r = String(t[n]);
          return r.length > 50 ? r.slice(0, 50) + "\u2026" : r;
        }
      }
    } catch {
    }
    return e.length > 50 ? e.slice(0, 50) + "\u2026" : e;
  }
  function Jv(e = null) {
    const [t, n] = v.useState(() => e ? Ks(e) : []), [r, l] = v.useState(false), [a, o] = v.useState(() => localStorage.getItem("maude-model") || "claude-opus-4-20250514"), [i, s] = v.useState(() => localStorage.getItem("maude-autoroute") === "true"), c = v.useCallback((h) => {
      localStorage.setItem("maude-model", h), o(h);
    }, []), m = v.useCallback((h) => {
      localStorage.setItem("maude-autoroute", String(h)), s(h);
    }, []), d = v.useRef(a);
    d.current = a;
    const g = v.useRef(null), x = v.useRef(e), w = v.useRef(""), S = v.useRef(0);
    x.current = e, v.useEffect(() => {
      if (!e) {
        n([]);
        return;
      }
      n(Ks(e)), Vv(e).then((h) => {
        h.length > 0 && n(h);
      });
    }, [
      e
    ]), v.useEffect(() => {
      x.current && t.length > 0 && Cm(x.current, t);
    }, [
      t
    ]);
    const _ = v.useCallback(async (h, j) => {
      var _a2, _b, _c2;
      const C = j && j.length > 0;
      if (!h.trim() && !C || r) return;
      if (h.startsWith("/")) {
        const X = h.trim().toLowerCase();
        if (X === "/clear") {
          n([]);
          return;
        }
        if (X.startsWith("/model ")) {
          o(X.slice(7).trim());
          return;
        }
      }
      const R = h || (C ? "What do you see in this image?" : ""), k = {
        id: Gs(),
        role: "user",
        content: R,
        imageUrls: C ? j : void 0,
        timestamp: Date.now()
      };
      n((X) => [
        ...X,
        k
      ]), l(true);
      const P = d.current, $ = {
        id: Gs(),
        role: "assistant",
        content: "",
        model: P,
        timestamp: Date.now()
      };
      n((X) => [
        ...X,
        $
      ]);
      const L = new AbortController();
      g.current = L;
      let G = "";
      try {
        const X = t.filter((ne) => ne.role !== "system").slice(-20).map((ne) => ({
          role: ne.role,
          content: ne.content
        }));
        let de = R;
        if (C) {
          const ne = j.map((Te) => `/home/mboard76/nvidia-workbench/terminal-llm/shared/${Te.split("/").pop()}`);
          if (ne.length === 1) de = `[Image attached: ${ne[0]} \u2014 analyze it with view_image tool]

${R}`;
          else {
            const Te = ne.map((ge, ke) => `  ${ke + 1}. ${ge}`).join(`
`);
            de = `[${ne.length} images attached \u2014 analyze each with view_image tool:
${Te}]

${R}`;
          }
        }
        const se = await Gv(), Ce = {
          model: P,
          messages: [
            {
              role: "system",
              content: Yv
            },
            ...X,
            {
              role: "user",
              content: de
            }
          ],
          stream: true,
          max_tokens: 4096,
          temperature: 0.7
        };
        if (se && (Ce.location = {
          lat: se.lat,
          lng: se.lng,
          accuracy: se.accuracy
        }), Kv) {
          const ne = await fetch(`${Go()}/api/chat/create`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(Ce),
            signal: L.signal
          });
          if (!ne.ok) {
            const ge = await ne.text();
            n((ke) => ke.map((I) => I.id === $.id ? {
              ...I,
              content: `Error: ${ne.status} \u2014 ${ge}`
            } : I)), l(false);
            return;
          }
          const { sid: Te } = await ne.json();
          await new Promise((ge) => {
            const ke = new EventSource(`${Go()}/api/chat/stream?sid=${Te}`);
            let I = "";
            const K = {
              tools: [],
              promptTokens: 0,
              completionTokens: 0,
              cacheReadTokens: 0,
              cacheCreateTokens: 0,
              elapsed: 0
            }, D = [], B = () => {
              ke.close(), S.current && (cancelAnimationFrame(S.current), S.current = 0);
              const Ne = {
                content: I
              };
              G && (Ne.model = G), (K.promptTokens || K.tools.length) && (Ne.trace = {
                ...K
              }), D.length && (Ne.toolSteps = D.map((Y) => ({
                ...Y
              }))), n((Y) => Y.map((q) => q.id === $.id ? {
                ...q,
                ...Ne
              } : q)), w.current = "", l(false), g.current = null, ge();
            };
            L.signal.addEventListener("abort", () => B());
            const Le = () => {
              S.current || (S.current = requestAnimationFrame(() => {
                const Ne = w.current, Y = {
                  ...K,
                  tools: [
                    ...K.tools
                  ]
                }, q = D.map((tt) => ({
                  ...tt
                }));
                n((tt) => tt.map((Je) => Je.id === $.id ? {
                  ...Je,
                  content: Ne,
                  trace: Y,
                  toolSteps: q,
                  ...G && {
                    model: G
                  }
                } : Je)), S.current = 0;
              }));
            };
            let Oe = false;
            ke.onmessage = (Ne) => {
              var _a3, _b2, _c3, _d2;
              if (Ne.data === "[DONE]") {
                B();
                return;
              }
              try {
                const Y = JSON.parse(Ne.data);
                Y.model && !G && (G = Y.model);
                const q = (_b2 = (_a3 = Y.choices) == null ? void 0 : _a3[0]) == null ? void 0 : _b2.delta;
                (q == null ? void 0 : q.reasoning_content) ? Oe || (I += `*Thinking...*

`, Oe = true) : (q == null ? void 0 : q.content) && (Oe && (I = I.replace(`*Thinking...*

`, ""), Oe = false), I += q.content), w.current = I, Le(), ((_d2 = (_c3 = Y.choices) == null ? void 0 : _c3[0]) == null ? void 0 : _d2.finish_reason) === "stop" && B();
              } catch {
              }
            }, ke.addEventListener("trace", (Ne) => {
              try {
                const Y = JSON.parse(Ne.data);
                if (Y.type === "tool_call" && Y.name) {
                  K.tools.push(Y.name);
                  const q = Y.args && Y.args !== "{}" ? Yo(Y.args) : void 0;
                  D.push({
                    name: Y.name,
                    args: q,
                    status: "running"
                  }), Le();
                } else if (Y.type === "tool_result") {
                  for (let q = D.length - 1; q >= 0; q--) if (D[q].name === Y.name && D[q].status === "running") {
                    D[q].result = (Y.preview || "").slice(0, 60), D[q].elapsed = Y.elapsed || 0, D[q].status = (D[q].result || "").startsWith("Error") ? "error" : "done";
                    break;
                  }
                  Le();
                } else if (Y.type === "keepalive" && Y.name) {
                  for (let q = D.length - 1; q >= 0; q--) if (D[q].name === Y.name && D[q].status === "running") {
                    D[q].elapsed = Y.elapsed || 0;
                    break;
                  }
                  Le();
                } else Y.type === "llm_call" ? (K.promptTokens += Y.prompt_tokens || 0, K.completionTokens += Y.completion_tokens || 0, K.cacheReadTokens += Y.cache_read_tokens || 0, K.cacheCreateTokens += Y.cache_create_tokens || 0, K.elapsed += Y.elapsed || 0) : Y.type === "error" && (I += `

*Error: ${Y.message || "Unknown error"}*`, w.current = I, Le());
              } catch {
              }
            }), ke.onerror = () => {
              I || (I = "Connection interrupted \u2014 send your message again to retry."), B();
            };
          });
          return;
        }
        const Ve = await fetch(`${Go()}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(Ce),
          signal: L.signal
        });
        if (!Ve.ok) {
          const ne = await Ve.text();
          n((Te) => Te.map((ge) => ge.id === $.id ? {
            ...ge,
            content: `Error: ${Ve.status} \u2014 ${ne}`
          } : ge)), l(false);
          return;
        }
        const ft = (_a2 = Ve.body) == null ? void 0 : _a2.getReader();
        if (!ft) {
          l(false);
          return;
        }
        const M = new TextDecoder();
        let H = "", W = "", le = "", te = false;
        const fe = {
          tools: [],
          promptTokens: 0,
          completionTokens: 0,
          cacheReadTokens: 0,
          cacheCreateTokens: 0,
          elapsed: 0
        }, Z = [], ue = () => {
          S.current || (S.current = requestAnimationFrame(() => {
            const ne = w.current, Te = {
              ...fe,
              tools: [
                ...fe.tools
              ]
            }, ge = Z.map((ke) => ({
              ...ke
            }));
            n((ke) => ke.map((I) => I.id === $.id ? {
              ...I,
              content: ne,
              trace: Te,
              toolSteps: ge,
              ...G && {
                model: G
              }
            } : I)), S.current = 0;
          }));
        };
        for (; ; ) {
          const { done: ne, value: Te } = await ft.read();
          if (ne) break;
          H += M.decode(Te, {
            stream: true
          });
          const ge = H.split(`
`);
          H = ge.pop() || "";
          for (const ke of ge) {
            const I = ke.trim();
            if (!I) continue;
            if (I.startsWith(": trace ")) {
              try {
                const D = JSON.parse(I.slice(8));
                if (D.type === "tool_call" && D.name) {
                  fe.tools.push(D.name);
                  const B = D.args && D.args !== "{}" ? Yo(D.args) : void 0;
                  Z.push({
                    name: D.name,
                    args: B,
                    status: "running"
                  }), ue();
                } else if (D.type === "tool_result") {
                  for (let B = Z.length - 1; B >= 0; B--) if (Z[B].name === D.name && Z[B].status === "running") {
                    const Le = (D.preview || "").slice(0, 60);
                    Z[B].result = Le, Z[B].elapsed = D.elapsed || 0, Z[B].status = Le.startsWith("Error") ? "error" : "done";
                    break;
                  }
                  ue();
                } else if (D.type === "keepalive" && D.name) {
                  for (let B = Z.length - 1; B >= 0; B--) if (Z[B].name === D.name && Z[B].status === "running") {
                    Z[B].elapsed = D.elapsed || 0;
                    break;
                  }
                  ue();
                } else if (D.type === "llm_call") fe.promptTokens += D.prompt_tokens || 0, fe.completionTokens += D.completion_tokens || 0, fe.cacheReadTokens += D.cache_read_tokens || 0, fe.cacheCreateTokens += D.cache_create_tokens || 0, fe.elapsed += D.elapsed || 0;
                else if (D.type === "error") {
                  const B = D.message || "Unknown error";
                  W += `

*Error: ${B}*`, w.current = W, ue();
                }
              } catch {
              }
              continue;
            }
            if (I.startsWith("event: ")) {
              le = I.slice(7);
              continue;
            }
            if (!I.startsWith("data: ")) continue;
            const K = I.slice(6);
            if (K !== "[DONE]") {
              if (le === "trace") {
                le = "";
                try {
                  const D = JSON.parse(K);
                  if (D.type === "tool_call" && D.name) {
                    fe.tools.push(D.name);
                    const B = D.args && D.args !== "{}" ? Yo(D.args) : void 0;
                    Z.push({
                      name: D.name,
                      args: B,
                      status: "running"
                    }), ue();
                  } else if (D.type === "tool_result") {
                    for (let B = Z.length - 1; B >= 0; B--) if (Z[B].name === D.name && Z[B].status === "running") {
                      const Le = (D.preview || "").slice(0, 60);
                      Z[B].result = Le, Z[B].elapsed = D.elapsed || 0, Z[B].status = Le.startsWith("Error") ? "error" : "done";
                      break;
                    }
                    ue();
                  } else if (D.type === "keepalive" && D.name) {
                    for (let B = Z.length - 1; B >= 0; B--) if (Z[B].name === D.name && Z[B].status === "running") {
                      Z[B].elapsed = D.elapsed || 0;
                      break;
                    }
                    ue();
                  } else if (D.type === "llm_call") fe.promptTokens += D.prompt_tokens || 0, fe.completionTokens += D.completion_tokens || 0, fe.cacheReadTokens += D.cache_read_tokens || 0, fe.cacheCreateTokens += D.cache_create_tokens || 0, fe.elapsed += D.elapsed || 0;
                  else if (D.type === "error") {
                    const B = D.message || "Unknown error";
                    W += `

*Error: ${B}*`, w.current = W, ue();
                  }
                } catch {
                }
                continue;
              }
              le = "";
              try {
                const D = JSON.parse(K);
                D.model && !G && (G = D.model);
                const B = (_c2 = (_b = D.choices) == null ? void 0 : _b[0]) == null ? void 0 : _c2.delta;
                (B == null ? void 0 : B.reasoning_content) ? te || (W += `*Thinking...*

`, te = true) : (B == null ? void 0 : B.content) && (te && (W = W.replace(`*Thinking...*

`, ""), te = false), W += B.content), ((B == null ? void 0 : B.reasoning_content) || (B == null ? void 0 : B.content)) && (w.current = W, ue());
              } catch {
              }
            }
          }
        }
        const Se = {};
        G && (Se.model = G), (fe.promptTokens || fe.tools.length) && (Se.trace = {
          ...fe
        }), Z.length && (Se.toolSteps = Z.map((ne) => ({
          ...ne
        }))), Object.keys(Se).length && n((ne) => ne.map((Te) => Te.id === $.id ? {
          ...Te,
          ...Se
        } : Te));
      } catch (X) {
        X instanceof Error && X.name !== "AbortError" && n((de) => de.map((se) => se.id === $.id ? {
          ...se,
          content: `Error: ${X.message}`
        } : se));
      } finally {
        if (S.current && (cancelAnimationFrame(S.current), S.current = 0), w.current) {
          const X = w.current, de = G || void 0;
          n((se) => se.map((Ce) => Ce.id === $.id ? {
            ...Ce,
            content: X,
            ...de && {
              model: de
            }
          } : Ce)), w.current = "";
        }
        l(false), g.current = null;
      }
    }, [
      t,
      r,
      a,
      i
    ]), p = v.useCallback(() => {
      var _a2;
      (_a2 = g.current) == null ? void 0 : _a2.abort();
    }, []), f = v.useCallback(() => {
      n([]);
    }, []);
    return {
      messages: t,
      isStreaming: r,
      currentModel: a,
      setCurrentModel: c,
      autoRoute: i,
      setAutoRoute: m,
      sendMessage: _,
      stopStreaming: p,
      clearMessages: f
    };
  }
  function zc(e) {
    const t = e.trim().replace(/\s+/g, " ");
    return t.length <= 40 ? t : t.slice(0, 37) + "...";
  }
  function Xv() {
    const [e, t] = v.useState(Em), [n, r] = v.useState(Qv);
    v.useEffect(() => {
      Bv().then((d) => {
        d.length > 0 && t(d);
      });
    }, []);
    const l = v.useCallback((d) => {
      const g = [
        ...d
      ].sort((x, w) => w.updatedAt - x.updatedAt);
      t(g), Wv(g);
    }, []), a = v.useCallback((d, g) => {
      const x = Gs(), w = Date.now(), _ = [
        {
          id: x,
          title: zc(d),
          createdAt: w,
          updatedAt: w,
          model: g
        },
        ...e
      ];
      return l(_), r(x), oa(x), x;
    }, [
      e,
      l
    ]), o = v.useCallback((d) => {
      r(d), oa(d);
    }, []), i = v.useCallback((d) => {
      const g = e.filter((x) => x.id !== d);
      if (l(g), Hv(d), n === d) {
        const x = g.length > 0 ? g[0].id : null;
        r(x), oa(x);
      }
    }, [
      e,
      n,
      l
    ]), s = v.useCallback((d, g) => {
      const x = e.map((w) => w.id === d ? {
        ...w,
        title: zc(g)
      } : w);
      l(x);
    }, [
      e,
      l
    ]), c = v.useCallback((d) => {
      const g = e.map((x) => x.id === d ? {
        ...x,
        updatedAt: Date.now()
      } : x);
      l(g);
    }, [
      e,
      l
    ]), m = v.useCallback(() => {
      r(null), oa(null);
    }, []);
    return {
      conversations: e,
      activeId: n,
      createConversation: a,
      switchConversation: o,
      deleteConversation: i,
      updateTitle: s,
      touchConversation: c,
      startNewChat: m
    };
  }
  function Zv(e, t) {
    const [n, r] = v.useState(0), l = v.useRef(0), a = v.useRef(false);
    return t && (a.current = true), v.useEffect(() => {
      if (!t && !a.current) {
        r(e.length);
        return;
      }
      const o = e.length;
      let i = 0;
      const s = (c) => {
        c - i >= 16 && (i = c, r((m) => m >= o ? m : m + Math.max(2, Math.floor((o - m) / 30)))), l.current = requestAnimationFrame(s);
      };
      return l.current = requestAnimationFrame(s), () => cancelAnimationFrame(l.current);
    }, [
      e,
      t
    ]), e.slice(0, n);
  }
  function qv(e) {
    const t = `${window.location.protocol}//${window.location.host}`;
    return e.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (r, l, a) => `<img src="${a.startsWith("/") ? `${t}${a}` : a}" alt="${l}" style="max-width:100%; max-height:50vh; border-radius:8px; margin:8px 0; object-fit:contain;" loading="lazy" onerror="this.style.display='none'" />`).replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-blue-400 underline">$1</a>').replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="my-2 rounded-lg bg-[#0d1117] p-3 text-sm overflow-x-auto"><code class="text-green-300">$2</code></pre>').replace(/`([^`]+)`/g, '<code class="rounded bg-[#0d1117] px-1.5 py-0.5 text-sm text-orange-300">$1</code>').replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>").replace(/^- (.+)$/gm, '<li class="ml-4 list-disc">$1</li>').replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>').replace(/\n/g, "<br/>");
  }
  const ex = {
    web_search: "searched the web",
    web_browse: "browsed a page",
    run_command: "ran a command",
    read_file: "read a file",
    write_file: "wrote a file",
    edit_file: "edited a file",
    list_directory: "listed a directory",
    gmail_list: "checked email",
    gmail_read: "read an email",
    gmail_send: "sent an email",
    calendar_list_events: "checked calendar",
    calendar_create_event: "created an event",
    drive_list: "browsed Drive",
    drive_search: "searched Drive",
    drive_create_doc: "created a doc",
    contacts_list: "looked up contacts",
    contacts_search: "searched contacts",
    youtube_search: "searched YouTube",
    web_image_search: "searched for images",
    generate_image: "generated an image",
    share_file: "shared a file",
    view_image: "viewed an image",
    dispatch_task: "dispatched a task",
    change_directory: "changed directory",
    get_working_directory: "checked directory"
  };
  function tx(e) {
    const t = /* @__PURE__ */ new Map();
    for (const r of e) t.set(r.name, (t.get(r.name) || 0) + 1);
    const n = [];
    for (const [r, l] of t) {
      const a = ex[r] || r.replace(/_/g, " ");
      if (l > 1) {
        const o = a.replace(/(?:a |an )(\w+)$/, `${l} $1s`);
        n.push(o === a ? `${a} x${l}` : o);
      } else n.push(a);
    }
    return n.length <= 2 ? n.join(" and ") : n.slice(0, -1).join(", ") + ", and " + n[n.length - 1];
  }
  const nx = {
    web_search: "\u{1F50D}",
    web_browse: "\u{1F310}",
    run_command: "\u26A1",
    read_file: "\u{1F4C4}",
    write_file: "\u270F\uFE0F",
    list_directory: "\u{1F4C2}",
    gmail_list: "\u{1F4E7}",
    gmail_read: "\u{1F4E7}",
    gmail_send: "\u{1F4E8}",
    calendar_list_events: "\u{1F4C5}",
    calendar_create_event: "\u{1F4C5}",
    drive_list: "\u{1F4BE}",
    drive_search: "\u{1F4BE}",
    drive_create_doc: "\u{1F4C4}",
    contacts_list: "\u{1F464}",
    contacts_search: "\u{1F464}",
    youtube_search: "\u25B6\uFE0F",
    web_image_search: "\u{1F5BC}\uFE0F",
    generate_image: "\u{1F3A8}",
    share_file: "\u{1F4E4}",
    view_image: "\u{1F441}\uFE0F"
  }, rx = ({ steps: e, streaming: t, contentStarted: n }) => {
    if (!e.length) return null;
    const r = e.some((l) => l.status === "running");
    return u.jsxs("div", {
      className: "mb-2 space-y-1",
      children: [
        e.map((l, a) => {
          const o = nx[l.name] || "\u2699\uFE0F", i = l.status === "running", s = l.status === "error", c = i ? "border-cyan-400/40" : s ? "border-red-400/40" : "border-cyan-500/20";
          return u.jsxs("div", {
            className: `border-l-2 ${c} pl-2.5 py-0.5 transition-all duration-300`,
            style: {
              animation: t ? "fadeSlideIn 0.3s ease-out" : "none"
            },
            children: [
              u.jsxs("div", {
                className: "flex items-center gap-1.5",
                children: [
                  u.jsx("span", {
                    className: "text-[11px]",
                    children: o
                  }),
                  u.jsx("span", {
                    className: "font-mono text-[11px] font-semibold text-cyan-400",
                    children: l.name
                  }),
                  i && u.jsx("span", {
                    className: "inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400"
                  }),
                  l.elapsed !== void 0 && u.jsxs("span", {
                    className: "ml-auto font-mono text-[10px] text-maude-muted",
                    children: [
                      l.elapsed.toFixed(1),
                      "s"
                    ]
                  })
                ]
              }),
              l.args && u.jsx("div", {
                className: "truncate font-mono text-[10px] leading-tight text-maude-muted",
                children: l.args
              }),
              l.result && u.jsxs("div", {
                className: `truncate font-mono text-[10px] leading-tight ${s ? "text-red-400" : "text-green-400/80"}`,
                children: [
                  s ? "\u2717 " : "\u2713 ",
                  l.result
                ]
              })
            ]
          }, `${l.name}-${a}`);
        }),
        t && !r && !n && u.jsxs("div", {
          className: "flex items-center gap-1.5 border-l-2 border-cyan-400/20 py-1 pl-2.5",
          style: {
            animation: "fadeSlideIn 0.3s ease-out"
          },
          children: [
            u.jsx("span", {
              className: "inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50",
              style: {
                animationDelay: "0ms"
              }
            }),
            u.jsx("span", {
              className: "inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50",
              style: {
                animationDelay: "150ms"
              }
            }),
            u.jsx("span", {
              className: "inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50",
              style: {
                animationDelay: "300ms"
              }
            }),
            u.jsx("span", {
              className: "animate-pulse text-[10px] text-cyan-400/50",
              children: "thinking"
            })
          ]
        }),
        !t && e.length > 0 && u.jsx("div", {
          className: "mt-1 border-l-2 border-green-400/30 py-0.5 pl-2.5",
          children: u.jsxs("span", {
            className: "text-[10px] text-green-400/70",
            children: [
              "\u2713 ",
              tx(e),
              (() => {
                const l = e.reduce((a, o) => a + (o.elapsed || 0), 0);
                return l > 0 ? ` \u2014 ${l.toFixed(1)}s` : "";
              })()
            ]
          })
        })
      ]
    });
  }, lx = ({ trace: e }) => {
    const t = e.promptTokens + e.cacheReadTokens + e.cacheCreateTokens;
    if (!t && !e.tools.length) return null;
    const n = t > 0 ? Math.round(e.cacheReadTokens / t * 100) : 0;
    return u.jsxs("div", {
      className: "mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-maude-muted",
      children: [
        e.tools.length > 0 && u.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            e.tools.length,
            " tool",
            e.tools.length > 1 ? "s" : ""
          ]
        }),
        u.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            t + e.completionTokens,
            " tok"
          ]
        }),
        n > 0 && u.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5 text-green-400",
          children: [
            n,
            "% cached"
          ]
        }),
        e.elapsed > 0 && u.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            e.elapsed.toFixed(1),
            "s"
          ]
        })
      ]
    });
  }, ax = {
    "claude-opus-4-20250514": "Claude Opus",
    "claude-sonnet-4-20250514": "Claude Sonnet",
    "mistral-large-latest": "Mistral Large",
    "codestral-latest": "Codestral",
    "devstral-2512": "Devstral",
    "devstral-small-latest": "Devstral Small",
    "devstral-medium-latest": "Devstral Medium",
    nemotron: "Nemotron",
    llava: "LLaVA"
  }, ox = ({ message: e, animate: t }) => {
    const n = e.role === "user", r = Zv(e.content, !!t), l = !n && e.toolSteps && e.toolSteps.length > 0, a = !e.content && !n && !l;
    return u.jsx("div", {
      className: `flex ${n ? "justify-end" : "justify-start"} mb-3`,
      children: u.jsxs("div", {
        className: `max-w-[85%] rounded-2xl px-4 py-3 ${n ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`,
        children: [
          e.model && !n && u.jsx("div", {
            className: "mb-1 text-[10px] font-medium tracking-wider text-maude-muted",
            children: ax[e.model] || e.model
          }),
          (() => {
            const o = e.imageUrls || (e.imageUrl ? [
              e.imageUrl
            ] : []);
            if (!o.length) return null;
            const i = `${window.location.protocol}//${window.location.host}`;
            return u.jsx("div", {
              className: `mb-2 flex gap-2 ${o.length > 1 ? "overflow-x-auto" : ""}`,
              children: o.map((s, c) => u.jsx("img", {
                src: `${i}${s}`,
                alt: `Attached photo ${c + 1}`,
                className: `rounded-lg ${o.length > 1 ? "h-32 w-32 shrink-0 object-cover" : "max-w-full"}`,
                loading: "lazy"
              }, s))
            });
          })(),
          l && u.jsx(rx, {
            steps: e.toolSteps,
            streaming: !!t,
            contentStarted: !!e.content
          }),
          r && u.jsx("div", {
            className: "break-words text-sm leading-relaxed",
            dangerouslySetInnerHTML: {
              __html: qv(r)
            }
          }),
          !n && e.trace && u.jsx(lx, {
            trace: e.trace
          }),
          a && u.jsxs("div", {
            className: "flex gap-1",
            children: [
              u.jsx("span", {
                className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                style: {
                  animationDelay: "0ms"
                }
              }),
              u.jsx("span", {
                className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                style: {
                  animationDelay: "150ms"
                }
              }),
              u.jsx("span", {
                className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                style: {
                  animationDelay: "300ms"
                }
              })
            ]
          })
        ]
      })
    });
  };
  function Ac() {
    const e = window.location;
    return `${e.protocol}//${e.host}`;
  }
  const sx = ({ onSend: e, isStreaming: t, onStop: n }) => {
    const [r, l] = v.useState(""), [a, o] = v.useState([]), [i, s] = v.useState(false), c = v.useRef(null), m = v.useRef(null), d = v.useRef(null);
    v.useEffect(() => {
      var _a2;
      (_a2 = c.current) == null ? void 0 : _a2.focus();
    }, []);
    const g = () => {
      (a.length > 0 || r.trim()) && (e(r.trim(), a.length > 0 ? a : void 0), l(""), o([]), c.current && (c.current.style.height = "44px"));
    }, x = (f) => {
      f.key === "Enter" && !f.shiftKey && (f.preventDefault(), g());
    }, w = () => {
      c.current && (c.current.style.height = "44px", c.current.style.height = Math.min(c.current.scrollHeight, 120) + "px");
    }, S = async (f) => {
      const h = f.target.files;
      if (!(!h || h.length === 0)) {
        s(true);
        try {
          const j = [];
          for (const C of Array.from(h)) {
            const R = `camera_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.jpg`;
            (await fetch(`${Ac()}/share/${encodeURIComponent(R)}`, {
              method: "POST",
              body: C
            })).ok && j.push(`/download/${R}`);
          }
          j.length > 0 && o((C) => [
            ...C,
            ...j
          ]);
        } catch {
        } finally {
          s(false), m.current && (m.current.value = ""), d.current && (d.current.value = "");
        }
      }
    }, _ = (f) => {
      o((h) => h.filter((j, C) => C !== f));
    }, p = a.length > 0 || r.trim();
    return u.jsxs("div", {
      className: "border-t border-maude-border bg-maude-surface p-3",
      children: [
        a.length > 0 && u.jsx("div", {
          className: "mb-2 flex gap-2 overflow-x-auto",
          children: a.map((f, h) => u.jsxs("div", {
            className: "relative shrink-0",
            children: [
              u.jsx("img", {
                src: `${Ac()}${f}`,
                alt: `Pending upload ${h + 1}`,
                className: "h-20 w-20 rounded-lg object-cover"
              }),
              u.jsx("button", {
                onClick: () => _(h),
                className: "absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-red-600 text-xs text-white",
                children: "\xD7"
              })
            ]
          }, f))
        }),
        u.jsxs("div", {
          className: "flex items-end gap-2",
          children: [
            u.jsx("button", {
              onClick: () => {
                var _a2;
                return (_a2 = m.current) == null ? void 0 : _a2.click();
              },
              disabled: i,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-maude-bg text-lg text-maude-muted hover:text-maude-text disabled:opacity-30",
              children: i ? u.jsx("span", {
                className: "h-4 w-4 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
              }) : "\u{1F4F7}"
            }),
            u.jsx("input", {
              ref: m,
              type: "file",
              accept: "image/*",
              capture: "environment",
              onChange: S,
              className: "hidden"
            }),
            u.jsx("button", {
              onClick: () => {
                var _a2;
                return (_a2 = d.current) == null ? void 0 : _a2.click();
              },
              disabled: i,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-maude-bg text-lg text-maude-muted hover:text-maude-text disabled:opacity-30",
              children: "\u{1F4CE}"
            }),
            u.jsx("input", {
              ref: d,
              type: "file",
              accept: "image/*",
              multiple: true,
              onChange: S,
              className: "hidden"
            }),
            u.jsx("textarea", {
              ref: c,
              value: r,
              onChange: (f) => l(f.target.value),
              onKeyDown: x,
              onInput: w,
              placeholder: "Message MAUDE...",
              rows: 1,
              className: "min-h-[44px] max-h-[120px] flex-1 resize-none rounded-xl bg-maude-bg px-4 py-3 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
            }),
            t ? u.jsx("button", {
              onClick: n,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-red-600 text-white",
              children: "\u25A0"
            }) : u.jsx("button", {
              onClick: g,
              disabled: !p,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30",
              children: "\u2191"
            })
          ]
        })
      ]
    });
  }, Jo = [
    {
      id: "claude-opus-4-20250514",
      label: "Claude Opus",
      desc: "Smartest"
    },
    {
      id: "claude-sonnet-4-20250514",
      label: "Claude Sonnet",
      desc: "Fast"
    },
    {
      id: "mistral-large-latest",
      label: "Mistral Large",
      desc: "General"
    },
    {
      id: "codestral-latest",
      label: "Codestral",
      desc: "Code"
    },
    {
      id: "devstral-2512",
      label: "Devstral",
      desc: "Code Agent"
    },
    {
      id: "devstral-small-latest",
      label: "Devstral Small",
      desc: "Code Light"
    },
    {
      id: "devstral-medium-latest",
      label: "Devstral Medium",
      desc: "Code Mid"
    },
    {
      id: "nemotron",
      label: "Nemotron",
      desc: "Local"
    },
    {
      id: "llava",
      label: "LLaVA",
      desc: "Vision"
    }
  ], ix = ({ currentModel: e, onSelect: t, autoRoute: n, onToggleAutoRoute: r }) => {
    const [l, a] = v.useState(false), o = Jo.find((i) => i.id === e) || Jo[0];
    return u.jsxs("div", {
      className: "relative",
      children: [
        u.jsxs("button", {
          onClick: () => a(!l),
          className: "flex items-center gap-1.5 rounded-lg bg-maude-bg px-3 py-1.5 text-xs text-maude-muted transition-colors hover:text-maude-text",
          children: [
            u.jsx("span", {
              className: "h-1.5 w-1.5 rounded-full bg-green-400"
            }),
            o.label,
            n && u.jsx("span", {
              className: "text-[10px] text-maude-accent",
              children: "AUTO"
            })
          ]
        }),
        l && u.jsxs("div", {
          className: "absolute right-0 top-full z-50 mt-1 w-56 rounded-xl border border-maude-border bg-maude-surface p-2 shadow-xl",
          children: [
            Jo.map((i) => u.jsxs("button", {
              onClick: () => {
                t(i.id), a(false);
              },
              className: `flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${i.id === e ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
              children: [
                u.jsx("span", {
                  children: i.label
                }),
                u.jsx("span", {
                  className: "text-xs text-maude-muted",
                  children: i.desc
                })
              ]
            }, i.id)),
            u.jsx("div", {
              className: "mt-2 border-t border-maude-border pt-2",
              children: u.jsxs("button", {
                onClick: () => r(!n),
                className: "flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm text-maude-text hover:bg-maude-bg",
                children: [
                  u.jsx("span", {
                    children: "Auto-route code"
                  }),
                  u.jsx("span", {
                    className: `text-xs ${n ? "text-green-400" : "text-maude-muted"}`,
                    children: n ? "ON" : "OFF"
                  })
                ]
              })
            })
          ]
        })
      ]
    });
  };
  function ux(e) {
    const t = /* @__PURE__ */ new Date(), n = new Date(t.getFullYear(), t.getMonth(), t.getDate()).getTime(), r = n - 864e5, l = n - 7 * 864e5, a = [
      {
        label: "Today",
        items: []
      },
      {
        label: "Yesterday",
        items: []
      },
      {
        label: "Previous 7 Days",
        items: []
      },
      {
        label: "Older",
        items: []
      }
    ];
    for (const o of e) o.updatedAt >= n ? a[0].items.push(o) : o.updatedAt >= r ? a[1].items.push(o) : o.updatedAt >= l ? a[2].items.push(o) : a[3].items.push(o);
    return a.filter((o) => o.items.length > 0);
  }
  const cx = ({ open: e, onClose: t, conversations: n, activeId: r, onSelect: l, onDelete: a, onNewChat: o }) => {
    const i = ux(n), [s, c] = v.useState(false);
    return u.jsxs(u.Fragment, {
      children: [
        u.jsx("div", {
          className: `fixed inset-0 z-40 bg-black/50 transition-opacity duration-200 ${e ? "opacity-100" : "pointer-events-none opacity-0"}`,
          onClick: t
        }),
        u.jsxs("div", {
          className: `fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-maude-border bg-maude-surface transition-transform duration-200 ${e ? "translate-x-0" : "-translate-x-full"}`,
          children: [
            u.jsxs("div", {
              className: "safe-top flex items-center justify-between border-b border-maude-border px-4 py-3",
              children: [
                u.jsx("h2", {
                  className: "text-sm font-semibold text-maude-text",
                  children: "Conversations"
                }),
                u.jsxs("div", {
                  className: "flex items-center gap-2",
                  children: [
                    u.jsx("button", {
                      onClick: () => c(!s),
                      className: `rounded-lg px-3 py-1 text-xs ${s ? "bg-maude-accent text-white" : "bg-maude-bg text-maude-muted"}`,
                      children: s ? "Done" : "Edit"
                    }),
                    u.jsx("button", {
                      onClick: () => {
                        c(false), o(), t();
                      },
                      className: "rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-accent",
                      children: "+ New"
                    })
                  ]
                })
              ]
            }),
            u.jsxs("div", {
              className: "no-scrollbar flex-1 overflow-y-auto p-2",
              children: [
                i.length === 0 && u.jsx("p", {
                  className: "px-2 py-8 text-center text-xs text-maude-muted",
                  children: "No conversations yet"
                }),
                i.map((m) => u.jsxs("div", {
                  className: "mb-3",
                  children: [
                    u.jsx("p", {
                      className: "mb-1 px-2 text-[10px] font-semibold uppercase tracking-wider text-maude-muted",
                      children: m.label
                    }),
                    m.items.map((d) => u.jsxs("div", {
                      className: `flex items-center rounded-lg px-2 py-2 text-sm transition-colors ${d.id === r ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                      children: [
                        s && u.jsx("button", {
                          onClick: (g) => {
                            g.stopPropagation(), a(d.id);
                          },
                          className: "mr-2 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red-500 text-xs text-white",
                          "aria-label": "Delete conversation",
                          children: "\u2212"
                        }),
                        u.jsx("button", {
                          className: "min-w-0 flex-1 truncate text-left",
                          onClick: () => {
                            s || (l(d.id), t());
                          },
                          children: d.title
                        })
                      ]
                    }, d.id))
                  ]
                }, m.label))
              ]
            })
          ]
        })
      ]
    });
  }, dx = ({ conversationId: e, onFirstMessage: t, onMessageSent: n, onOpenDrawer: r, onNewChat: l }) => {
    const a = ho(), o = v.useRef(null), i = v.useRef(e), { messages: s, isStreaming: c, currentModel: m, setCurrentModel: d, autoRoute: g, setAutoRoute: x, sendMessage: w, stopStreaming: S } = Jv(e);
    v.useEffect(() => {
      o.current && (o.current.scrollTop = o.current.scrollHeight);
    }, [
      s
    ]), v.useEffect(() => {
      if (!c || !o.current) return;
      const p = setInterval(() => {
        o.current && (o.current.scrollTop = o.current.scrollHeight);
      }, 200);
      return () => clearInterval(p);
    }, [
      c
    ]);
    const _ = v.useCallback((p, f) => {
      if (!i.current) {
        const h = p || ((f == null ? void 0 : f.length) ? "Image conversation" : "New chat"), j = t(h, m);
        i.current = j;
      }
      w(p, f), n();
    }, [
      w,
      t,
      n,
      m
    ]);
    return v.useEffect(() => {
      i.current && s.length > 0 && Cm(i.current, s);
    }, [
      s
    ]), u.jsxs(u.Fragment, {
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2",
          children: [
            u.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                u.jsx("button", {
                  onClick: r,
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-sm text-maude-muted hover:text-maude-text",
                  "aria-label": "Open conversations",
                  children: "\u2630"
                }),
                u.jsx("h1", {
                  className: "fire-gradient text-lg font-bold",
                  children: "MAUDE"
                }),
                u.jsx("button", {
                  onClick: l,
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-muted hover:text-maude-text",
                  children: "New"
                }),
                u.jsxs("button", {
                  onClick: () => a("/maude/voice"),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-accent hover:text-maude-text",
                  children: [
                    "\u{1F399}\uFE0F",
                    " Voice"
                  ]
                })
              ]
            }),
            u.jsx(ix, {
              currentModel: m,
              onSelect: d,
              autoRoute: g,
              onToggleAutoRoute: x
            })
          ]
        }),
        u.jsxs("div", {
          ref: o,
          className: "no-scrollbar flex-1 overflow-y-auto px-4 py-4",
          children: [
            s.length === 0 && u.jsxs("div", {
              className: "flex h-full flex-col items-center justify-center text-center",
              children: [
                u.jsx("span", {
                  className: "fire-gradient mb-3 text-5xl font-black",
                  children: "\u25C7"
                }),
                u.jsx("h2", {
                  className: "mb-1 text-lg font-semibold text-maude-text",
                  children: "MAUDE"
                }),
                u.jsx("p", {
                  className: "max-w-xs text-sm text-maude-muted",
                  children: "Your local AI assistant. Powered by Mistral & Codestral. Ask me anything."
                }),
                u.jsx("div", {
                  className: "mt-4 flex flex-wrap justify-center gap-2",
                  children: [
                    "What can you do?",
                    "Write a Python script",
                    "Explain this code",
                    "System status"
                  ].map((p) => u.jsx("button", {
                    onClick: () => _(p),
                    className: "rounded-full border border-maude-border px-3 py-1.5 text-xs text-maude-muted transition-colors hover:border-maude-accent hover:text-maude-text",
                    children: p
                  }, p))
                })
              ]
            }),
            s.map((p, f) => u.jsx(ox, {
              message: p,
              animate: c && f === s.length - 1
            }, p.id))
          ]
        }),
        u.jsx(sx, {
          onSend: (p, f) => _(p, f),
          isStreaming: c,
          onStop: S
        })
      ]
    });
  }, fx = () => {
    const [e, t] = v.useState(false), { conversations: n, activeId: r, createConversation: l, switchConversation: a, deleteConversation: o, touchConversation: i, startNewChat: s } = Xv(), c = v.useCallback((d, g) => l(d, g), [
      l
    ]), m = v.useCallback(() => {
      r && i(r);
    }, [
      r,
      i
    ]);
    return u.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        u.jsx(dx, {
          conversationId: r,
          onFirstMessage: c,
          onMessageSent: m,
          onOpenDrawer: () => t(true),
          onNewChat: s
        }, r || "new"),
        u.jsx(cx, {
          open: e,
          onClose: () => t(false),
          conversations: n,
          activeId: r,
          onSelect: a,
          onDelete: o,
          onNewChat: s
        })
      ]
    });
  }, mx = "modulepreload", px = function(e) {
    return "/" + e;
  }, Uc = {}, il = function(t, n, r) {
    let l = Promise.resolve();
    if (n && n.length > 0) {
      document.getElementsByTagName("link");
      const o = document.querySelector("meta[property=csp-nonce]"), i = (o == null ? void 0 : o.nonce) || (o == null ? void 0 : o.getAttribute("nonce"));
      l = Promise.allSettled(n.map((s) => {
        if (s = px(s), s in Uc) return;
        Uc[s] = true;
        const c = s.endsWith(".css"), m = c ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${s}"]${m}`)) return;
        const d = document.createElement("link");
        if (d.rel = c ? "stylesheet" : mx, c || (d.as = "script"), d.crossOrigin = "", d.href = s, i && d.setAttribute("nonce", i), document.head.appendChild(d), c) return new Promise((g, x) => {
          d.addEventListener("load", g), d.addEventListener("error", () => x(new Error(`Unable to preload CSS for ${s}`)));
        });
      }));
    }
    function a(o) {
      const i = new Event("vite:preloadError", {
        cancelable: true
      });
      if (i.payload = o, window.dispatchEvent(i), !i.defaultPrevented) throw o;
    }
    return l.then((o) => {
      for (const i of o || []) i.status === "rejected" && a(i.reason);
      return t().catch(a);
    });
  }, hx = {
    0: 0
  }, gx = {
    0: 0
  }, vx = {
    start: 0,
    endTurn: 1,
    pause: 2,
    restart: 3
  }, xx = (e) => {
    switch (e.type) {
      case "handshake":
        return new Uint8Array([
          0,
          hx[e.version],
          gx[e.model]
        ]);
      case "audio":
        return new Uint8Array([
          1,
          ...e.data
        ]);
      case "text":
        return new Uint8Array([
          2,
          ...new TextEncoder().encode(e.data)
        ]);
      case "control":
        return new Uint8Array([
          3,
          vx[e.action]
        ]);
      case "metadata":
        return new Uint8Array([
          4,
          ...new TextEncoder().encode(JSON.stringify(e.data))
        ]);
      case "error":
        return new Uint8Array([
          5,
          ...new TextEncoder().encode(e.data)
        ]);
      case "ping":
        return new Uint8Array([
          6
        ]);
    }
  }, yx = "You are MAUDE, a capable AI assistant with a warm Scottish accent. You are direct, competent, and quietly confident \u2014 like MAUDE\u2019s design. ", _m = "NATF2.pt";
  function wx() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  function Sx(e) {
    const n = `wss://${window.location.host}`;
    let r = yx;
    e && (r += `

--- Image Context ---
` + e);
    const l = localStorage.getItem("maude-default-voice") || _m, a = new URLSearchParams({
      text_temperature: "0.7",
      text_topk: "25",
      audio_temperature: "0.8",
      audio_topk: "250",
      pad_mult: "0",
      text_seed: String(Math.round(Math.random() * 1e6)),
      audio_seed: String(Math.round(Math.random() * 1e6)),
      repetition_penalty_context: "64",
      repetition_penalty: "1.0",
      text_prompt: r,
      voice_prompt: l
    });
    return `${n}/api/chat?${a}`;
  }
  const kx = `
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
  async function Nx(e, t) {
    const n = new Blob([
      kx
    ], {
      type: "application/javascript"
    }), r = URL.createObjectURL(n);
    await e.audioWorklet.addModule(r), URL.revokeObjectURL(r);
    const l = new AudioWorkletNode(e, "ring-player", {
      outputChannelCount: [
        1
      ]
    });
    l.port.onmessage = (o) => {
      var _a2;
      ((_a2 = o.data) == null ? void 0 : _a2.type) === "state" && t && t(o.data.state, o.data);
    };
    const a = e.createGain();
    return a.gain.value = 6, l.connect(a), {
      feedAudio(o) {
        l.port.postMessage({
          type: "audio",
          pcm: o
        }, [
          o.buffer
        ]);
      },
      reset() {
        l.port.postMessage({
          type: "reset"
        });
      },
      connect(o) {
        a.connect(o);
      },
      disconnect() {
        try {
          a.disconnect();
        } catch {
        }
        try {
          l.disconnect();
        } catch {
        }
      }
    };
  }
  const $c = ({ analyser: e, active: t, color: n }) => {
    const r = v.useRef(null), l = v.useRef(0);
    return v.useEffect(() => {
      if (!e || !t || !r.current) return;
      const a = r.current, o = a.getContext("2d"), i = e.frequencyBinCount, s = new Uint8Array(i), c = () => {
        l.current = requestAnimationFrame(c), e.getByteTimeDomainData(s), o.clearRect(0, 0, a.width, a.height), o.lineWidth = 2, o.strokeStyle = n, o.beginPath();
        const m = a.width / i;
        let d = 0;
        for (let g = 0; g < i; g++) {
          const w = s[g] / 128 * a.height / 2;
          g === 0 ? o.moveTo(d, w) : o.lineTo(d, w), d += m;
        }
        o.lineTo(a.width, a.height / 2), o.stroke();
      };
      return c(), () => cancelAnimationFrame(l.current);
    }, [
      e,
      t,
      n
    ]), u.jsx("canvas", {
      ref: r,
      width: 300,
      height: 60,
      className: "w-full rounded-lg"
    });
  }, jx = () => {
    const e = ho(), [t, n] = v.useState("disconnected"), [r, l] = v.useState(""), [a, o] = v.useState(false), [i, s] = v.useState(0), [c, m] = v.useState(""), [d, g] = v.useState(""), [x, w] = v.useState(null), [S, _] = v.useState(null), [p, f] = v.useState(false), [h, j] = v.useState(false), C = v.useRef(null), R = v.useRef(null), k = v.useRef(null), P = v.useRef(null), $ = v.useRef(null), L = v.useRef(null), G = v.useRef(null), X = v.useRef(null), de = v.useRef(null), se = v.useRef(null), Ce = v.useRef(0), Ve = v.useRef(0), ft = v.useRef(0), M = v.useRef(0), H = v.useRef(0), W = v.useRef(0), le = v.useRef(0), te = v.useCallback(async () => {
      m(""), l(""), s(0), Ce.current = 0;
      try {
        $.current || ($.current = new AudioContext({
          sampleRate: 48e3
        }));
        const I = $.current;
        await I.resume();
        const K = I.createBuffer(1, 1, I.sampleRate), D = I.createBufferSource();
        D.buffer = K, D.connect(I.destination), D.start(), ft.current = 0, M.current = 0, g(`ctx: ${I.state} ${I.sampleRate}Hz`), L.current || (L.current = await Nx(I, (tt, Je) => {
          Je.underruns != null && (H.current = Je.underruns), Je.avail != null && (W.current = Je.avail);
        }), L.current.connect(I.destination)), L.current.reset(), H.current = 0;
        const B = I.createAnalyser();
        L.current.connect(B), X.current = B;
        const Le = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1
          }
        });
        se.current = Le;
        const Oe = I.createAnalyser();
        I.createMediaStreamSource(Le).connect(Oe), de.current = Oe;
        const Y = Sx(P.current ?? void 0);
        console.log("Connecting to PersonaPlex:", Y);
        const q = new WebSocket(Y);
        q.binaryType = "arraybuffer", C.current = q, n("connecting"), q.onopen = () => {
          console.log("PersonaPlex WS open, waiting for handshake");
        }, q.onmessage = (tt) => {
          var _a2;
          try {
            const Je = new Uint8Array(tt.data), Mr = Je[0];
            if (Mr === 0) console.log("PersonaPlex handshake received"), n("connected"), fe(q, Le, I), Ve.current = window.setInterval(() => {
              var _a3;
              Ce.current += 1, s(Ce.current);
              const _t = ((_a3 = $.current) == null ? void 0 : _a3.state) ?? "?", He = Math.round(W.current / 48);
              g(`dec:${M.current} buf:${He}ms ur:${H.current}`);
            }, 1e3);
            else if (Mr === 2) {
              const _t = new TextDecoder().decode(Je.slice(1));
              _t.includes("[Searching...]") ? o(true) : (_t.includes("[Tool result:]") || _t.includes("[Error:")) && o(false), l((He) => He + _t);
            } else if (Mr === 3) {
              M.current++;
              const _t = Je.slice(1), He = new Float32Array(_t.buffer, _t.byteOffset, _t.byteLength / 4), Vt = new Float32Array(He.length * 2);
              for (let Pn = 0; Pn < Vt.length; Pn++) {
                const Jn = Pn * 0.5, mt = Jn | 0, Pt = Math.min(mt + 1, He.length - 1), Ol = Jn - mt;
                Vt[Pn] = He[mt] + (He[Pt] - He[mt]) * Ol;
              }
              (_a2 = L.current) == null ? void 0 : _a2.feedAudio(Vt);
            }
          } catch (Je) {
            console.error("Message decode error:", Je);
          }
        }, q.onclose = (tt) => {
          console.log("PersonaPlex WS closed:", tt.code, tt.reason), n("disconnected"), Z(), clearInterval(Ve.current);
        }, q.onerror = (tt) => {
          console.error("PersonaPlex WS error:", tt), m("WebSocket connection failed. Is PersonaPlex running?"), n("disconnected");
        };
      } catch (I) {
        const K = I instanceof Error ? I.message : "Connection failed";
        console.error("Voice connect error:", K), m(K), n("disconnected");
      }
    }, []), fe = v.useCallback(async (I, K, D) => {
      try {
        const B = (await il(async () => {
          const { default: Y } = await import("./recorder.min-C9SwhXmS.js").then((q) => q.r);
          return {
            default: Y
          };
        }, [])).default, Le = (await il(async () => {
          const { default: Y } = await import("./encoderWorker.min-De-nC0Q0.js");
          return {
            default: Y
          };
        }, [])).default, Oe = D.createMediaStreamSource(K), Ne = new B({
          encoderPath: Le,
          bufferLength: Math.round(960 * D.sampleRate / 24e3),
          encoderFrameSize: 20,
          encoderSampleRate: 24e3,
          maxFramesPerPage: 2,
          numberOfChannels: 1,
          recordingGain: 1,
          resampleQuality: 3,
          encoderComplexity: 3,
          encoderApplication: 2049,
          streamPages: true,
          sourceNode: Oe
        });
        Ne.ondataavailable = (Y) => {
          I.readyState === WebSocket.OPEN && I.send(xx({
            type: "audio",
            data: Y
          }));
        }, Ne.onstart = () => {
          console.log("Opus recorder started");
        }, Ne.start(), G.current = Ne;
      } catch (B) {
        console.error("Recorder start error:", B), m("Failed to start microphone recording");
      }
    }, []), Z = v.useCallback(() => {
      if (G.current) {
        try {
          G.current.stop();
        } catch {
        }
        G.current = null;
      }
      se.current && (se.current.getTracks().forEach((I) => I.stop()), se.current = null);
    }, []), ue = v.useCallback(() => {
      Z(), clearInterval(Ve.current), clearInterval(le.current), C.current && (C.current.close(), C.current = null), n("disconnected");
    }, [
      Z
    ]), Se = v.useCallback(async (I) => {
      var _a2;
      const K = (_a2 = I.target.files) == null ? void 0 : _a2[0];
      if (!K) return;
      I.target.value = "";
      const D = `voice_camera_${Date.now()}.jpg`, B = wx(), Le = URL.createObjectURL(K);
      w(Le), _(null), j(true);
      try {
        if (!(await fetch(`${B}/share/${D}`, {
          method: "POST",
          body: K
        })).ok) throw new Error("Upload failed");
        j(false), f(true);
        const Ne = await fetch(`${B}/api/analyze-image`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            filename: D,
            question: "Describe this image in detail. What do you see?"
          })
        });
        if (!Ne.ok) throw new Error("Analysis failed");
        const q = (await Ne.json()).analysis || "No analysis returned.";
        _(q), f(false), P.current = `The user shared an image (${D}). Analysis: ${q}`, C.current && C.current.readyState === WebSocket.OPEN && (ue(), await new Promise((tt) => setTimeout(tt, 300)), te());
      } catch (Oe) {
        const Ne = Oe instanceof Error ? Oe.message : "Image processing failed";
        m(Ne), j(false), f(false);
      }
    }, [
      te,
      ue
    ]), ne = v.useCallback(async () => {
      P.current = null, w(null), _(null), C.current && C.current.readyState === WebSocket.OPEN && (ue(), await new Promise((I) => setTimeout(I, 300)), te());
    }, [
      te,
      ue
    ]);
    v.useEffect(() => () => {
      ue();
    }, []);
    const Te = (I) => {
      const K = Math.floor(I / 60), D = I % 60;
      return `${K}:${D.toString().padStart(2, "0")}`;
    }, ge = t === "connected", ke = t === "connecting";
    return u.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2",
          children: [
            u.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                u.jsx("h1", {
                  className: "fire-gradient text-lg font-bold",
                  children: "MAUDE"
                }),
                u.jsx("span", {
                  className: "rounded-full bg-maude-bg px-2 py-0.5 text-[10px] uppercase tracking-wider text-maude-accent",
                  children: "Voice"
                })
              ]
            }),
            u.jsx("button", {
              onClick: () => e("/maude"),
              className: "rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
              children: "Text Mode"
            })
          ]
        }),
        u.jsxs("div", {
          className: "flex flex-1 flex-col items-center justify-center gap-6 overflow-y-auto px-6 pb-4",
          children: [
            u.jsxs("div", {
              className: "flex flex-col items-center gap-2",
              children: [
                u.jsx("div", {
                  className: `h-32 w-32 rounded-full border-4 ${ge ? "animate-pulse border-maude-accent shadow-[0_0_30px_rgba(255,69,0,0.3)]" : ke ? "animate-spin border-maude-muted" : "border-maude-border"} flex items-center justify-center`,
                  children: u.jsx("span", {
                    className: "text-4xl",
                    children: ge ? "\u{1F399}\uFE0F" : ke ? "\u23F3" : "\u{1F399}\uFE0F"
                  })
                }),
                u.jsx("span", {
                  className: "text-sm text-maude-muted",
                  children: ge ? `Connected \u2022 ${Te(i)}` : ke ? "Connecting to PersonaPlex..." : "Tap to start voice chat"
                })
              ]
            }),
            ge && u.jsxs("div", {
              className: "w-full max-w-xs space-y-3",
              children: [
                u.jsxs("div", {
                  children: [
                    u.jsx("span", {
                      className: "mb-1 block text-[10px] uppercase tracking-wider text-maude-muted",
                      children: "MAUDE"
                    }),
                    u.jsx("div", {
                      className: "rounded-lg bg-maude-surface p-2",
                      children: u.jsx($c, {
                        analyser: X.current,
                        active: ge,
                        color: "#ff4500"
                      })
                    })
                  ]
                }),
                u.jsxs("div", {
                  children: [
                    u.jsx("span", {
                      className: "mb-1 block text-[10px] uppercase tracking-wider text-maude-muted",
                      children: "You"
                    }),
                    u.jsx("div", {
                      className: "rounded-lg bg-maude-surface p-2",
                      children: u.jsx($c, {
                        analyser: de.current,
                        active: ge,
                        color: "#888"
                      })
                    })
                  ]
                })
              ]
            }),
            ge && u.jsxs("div", {
              className: "flex gap-3",
              children: [
                u.jsxs("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = R.current) == null ? void 0 : _a2.click();
                  },
                  disabled: p || h,
                  className: "flex items-center gap-1.5 rounded-xl bg-maude-surface px-4 py-2 text-sm text-maude-text transition-all hover:bg-maude-border disabled:opacity-40",
                  children: [
                    u.jsx("span", {
                      children: "\u{1F4F7}"
                    }),
                    " Camera"
                  ]
                }),
                u.jsxs("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = k.current) == null ? void 0 : _a2.click();
                  },
                  disabled: p || h,
                  className: "flex items-center gap-1.5 rounded-xl bg-maude-surface px-4 py-2 text-sm text-maude-text transition-all hover:bg-maude-border disabled:opacity-40",
                  children: [
                    u.jsx("span", {
                      children: "\u{1F5BC}\uFE0F"
                    }),
                    " Gallery"
                  ]
                })
              ]
            }),
            u.jsx("input", {
              ref: R,
              type: "file",
              accept: "image/*",
              capture: "environment",
              onChange: Se,
              className: "hidden"
            }),
            u.jsx("input", {
              ref: k,
              type: "file",
              accept: "image/*",
              onChange: Se,
              className: "hidden"
            }),
            x && u.jsxs("div", {
              className: "w-full max-w-xs rounded-xl bg-maude-surface p-3",
              children: [
                u.jsx("span", {
                  className: "mb-2 block text-[10px] uppercase tracking-wider text-maude-muted",
                  children: "Image Context"
                }),
                u.jsx("img", {
                  src: x,
                  alt: "Captured",
                  className: "mb-2 h-24 w-full rounded-lg object-cover"
                }),
                h && u.jsx("p", {
                  className: "text-xs text-maude-muted",
                  children: "Uploading..."
                }),
                p && u.jsxs("div", {
                  className: "flex items-center gap-2",
                  children: [
                    u.jsx("div", {
                      className: "h-3 w-3 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
                    }),
                    u.jsx("span", {
                      className: "text-xs text-maude-muted",
                      children: "Analyzing with LLaVA..."
                    })
                  ]
                }),
                S && u.jsx("p", {
                  className: "text-xs leading-relaxed text-maude-text",
                  children: S
                }),
                S && u.jsx("button", {
                  onClick: ne,
                  className: "mt-2 text-[10px] text-maude-muted underline hover:text-maude-text",
                  children: "Clear image context"
                })
              ]
            }),
            a && u.jsxs("div", {
              className: "flex items-center gap-2 rounded-xl bg-maude-accent/10 px-4 py-2",
              children: [
                u.jsx("div", {
                  className: "h-3 w-3 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
                }),
                u.jsx("span", {
                  className: "text-xs font-medium text-maude-accent",
                  children: "Searching..."
                })
              ]
            }),
            r && u.jsxs("div", {
              className: "w-full max-w-xs rounded-xl bg-maude-surface p-3",
              children: [
                u.jsx("span", {
                  className: "mb-1 block text-[10px] uppercase tracking-wider text-maude-muted",
                  children: "Transcript"
                }),
                u.jsx("div", {
                  className: "max-h-48 overflow-y-auto text-sm text-maude-text",
                  children: r.split(`
`).map((I, K) => I.includes("[Searching...]") ? u.jsx("p", {
                    className: "my-1 text-xs italic text-maude-accent",
                    children: I
                  }, K) : I.includes("[Tool result:]") ? u.jsx("p", {
                    className: "mt-2 mb-1 text-[10px] font-bold uppercase tracking-wider text-maude-accent",
                    children: I
                  }, K) : I.includes("[Error:") ? u.jsx("p", {
                    className: "my-1 text-xs text-red-400",
                    children: I
                  }, K) : u.jsxs("span", {
                    children: [
                      I,
                      K < r.split(`
`).length - 1 ? `
` : ""
                    ]
                  }, K))
                })
              ]
            }),
            c && u.jsx("div", {
              className: "w-full max-w-xs rounded-xl bg-red-900/30 p-3",
              children: u.jsx("p", {
                className: "text-sm text-red-400",
                children: c
              })
            }),
            u.jsx("button", {
              onClick: ge || ke ? ue : te,
              className: `min-w-[200px] rounded-2xl px-8 py-4 text-base font-semibold text-white transition-all ${ge ? "bg-red-600 hover:bg-red-700" : ke ? "bg-maude-muted" : "fire-bg hover:opacity-90"}`,
              disabled: ke,
              children: ge ? "End Call" : ke ? "Connecting..." : "Start Voice Chat"
            }),
            u.jsxs("div", {
              className: "text-center text-[10px] text-maude-muted",
              children: [
                "Voice: ",
                (localStorage.getItem("maude-default-voice") || _m).replace(".pt", ""),
                " \u2022 ",
                "PersonaPlex"
              ]
            }),
            d && u.jsx("div", {
              className: "text-center font-mono text-[10px] text-maude-muted opacity-60",
              children: d
            })
          ]
        })
      ]
    });
  }, Ex = /iPad|iPhone|iPod/.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1, Cx = () => {
    const e = v.useRef(null), t = v.useRef(null), n = v.useRef(null), r = v.useRef(null), l = v.useRef(null), a = v.useRef(null), o = v.useRef(null), [i, s] = v.useState("disconnected");
    return v.useEffect(() => {
      let c, m;
      return (async () => {
        const { Terminal: g } = await il(async () => {
          const { Terminal: _ } = await import("./xterm-PglAAeey.js").then((p) => p.x);
          return {
            Terminal: _
          };
        }, []), { FitAddon: x } = await il(async () => {
          const { FitAddon: _ } = await import("./addon-fit-CyyJcX4C.js").then((p) => p.a);
          return {
            FitAddon: _
          };
        }, []), { WebLinksAddon: w } = await il(async () => {
          const { WebLinksAddon: _ } = await import("./addon-web-links-B1M8nFkN.js").then((p) => p.a);
          return {
            WebLinksAddon: _
          };
        }, []);
        if (!document.querySelector('link[href*="xterm"]')) {
          const _ = document.createElement("link");
          _.rel = "stylesheet", _.href = "https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/css/xterm.min.css", document.head.appendChild(_);
        }
        c = new g({
          cursorBlink: true,
          fontSize: 16,
          fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
          theme: {
            background: "#0d1117",
            foreground: "#e6edf3",
            cursor: "#ff4500",
            cursorAccent: "#0d1117",
            selectionBackground: "#30363d",
            black: "#0d1117",
            red: "#ff7b72",
            green: "#7ee787",
            yellow: "#ffa657",
            blue: "#79c0ff",
            magenta: "#d2a8ff",
            cyan: "#a5d6ff",
            white: "#e6edf3",
            brightBlack: "#484f58",
            brightRed: "#ffa198",
            brightGreen: "#56d364",
            brightYellow: "#e3b341",
            brightBlue: "#a5d6ff",
            brightMagenta: "#d2a8ff",
            brightCyan: "#b1bac4",
            brightWhite: "#f0f6fc"
          },
          allowTransparency: true,
          scrollback: 5e3
        });
        const S = new x();
        if (c.loadAddon(S), c.loadAddon(new w()), r.current = c, l.current = S, e.current && (c.open(e.current), S.fit()), s("connecting"), Ex) try {
          const _ = await fetch("/api/terminal/create", {
            method: "POST"
          }), { sid: p } = await _.json();
          o.current = p;
          const f = new EventSource(`/api/terminal/stream?sid=${p}`);
          a.current = f, f.onopen = () => {
            s("connected");
            const R = S.proposeDimensions();
            R && fetch("/api/terminal/resize", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: p,
                cols: R.cols,
                rows: R.rows
              })
            });
          }, f.onmessage = (R) => {
            const k = Uint8Array.from(atob(R.data), (P) => P.charCodeAt(0));
            c.write(k);
          }, f.onerror = () => {
            s("disconnected"), c.write(`\r
\x1B[33m[Connection closed]\x1B[0m\r
`), f.close();
          };
          const h = (R) => {
            fetch("/api/terminal/input", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: p,
                data: R
              })
            });
          };
          n.current = h, c.onData(h);
          const j = () => {
            S.fit();
            const R = S.proposeDimensions();
            R && fetch("/api/terminal/resize", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: p,
                cols: R.cols,
                rows: R.rows
              })
            });
          }, C = new ResizeObserver(j);
          e.current && C.observe(e.current), m = () => C.disconnect();
        } catch {
          s("disconnected"), c.write(`\x1B[31m[Failed to connect]\x1B[0m\r
`);
        }
        else {
          const _ = window.location.protocol === "https:" ? "wss" : "ws", p = new WebSocket(`${_}://${window.location.host}/ws/terminal`);
          p.binaryType = "arraybuffer", t.current = p, p.onopen = () => {
            s("connected");
            const C = S.proposeDimensions();
            C && p.send(JSON.stringify({
              type: "resize",
              cols: C.cols,
              rows: C.rows
            }));
          }, p.onmessage = (C) => {
            c.write(C.data instanceof ArrayBuffer ? new Uint8Array(C.data) : C.data);
          }, p.onclose = () => {
            s("disconnected"), c.write(`\r
\x1B[33m[Connection closed]\x1B[0m\r
`);
          }, p.onerror = () => {
            s("disconnected");
          };
          const f = (C) => {
            p.readyState === WebSocket.OPEN && p.send(C);
          };
          n.current = f, c.onData(f);
          const h = () => {
            S.fit();
            const C = S.proposeDimensions();
            C && p.readyState === WebSocket.OPEN && p.send(JSON.stringify({
              type: "resize",
              cols: C.cols,
              rows: C.rows
            }));
          }, j = new ResizeObserver(h);
          e.current && j.observe(e.current), m = () => j.disconnect();
        }
      })(), () => {
        var _a2, _b, _c2;
        m == null ? void 0 : m(), (_a2 = t.current) == null ? void 0 : _a2.close(), (_b = a.current) == null ? void 0 : _b.close(), (_c2 = r.current) == null ? void 0 : _c2.dispose();
      };
    }, []), u.jsxs("div", {
      className: "flex h-full flex-col bg-[#0d1117]",
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2",
          children: [
            u.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                u.jsx("span", {
                  className: "font-mono text-sm text-maude-text",
                  children: ">_ Terminal"
                }),
                u.jsx("span", {
                  className: `h-2 w-2 rounded-full ${i === "connected" ? "bg-green-400" : i === "connecting" ? "bg-yellow-400" : "bg-red-400"}`
                }),
                u.jsx("span", {
                  className: "text-xs text-maude-muted",
                  children: i
                })
              ]
            }),
            i === "disconnected" && u.jsx("button", {
              onClick: () => window.location.reload(),
              className: "rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
              children: "Reconnect"
            })
          ]
        }),
        u.jsx("div", {
          className: "flex shrink-0 gap-1 overflow-x-auto border-b border-maude-border bg-maude-surface px-2 py-1",
          children: [
            {
              label: "Esc",
              key: "\x1B"
            },
            {
              label: "Tab",
              key: "	"
            },
            {
              label: "Ctrl+C",
              key: ""
            },
            {
              label: "Ctrl+D",
              key: ""
            },
            {
              label: "Ctrl+Z",
              key: ""
            },
            {
              label: "Ctrl+L",
              key: "\f"
            },
            {
              label: "\u2191",
              key: "\x1B[A"
            },
            {
              label: "\u2193",
              key: "\x1B[B"
            },
            {
              label: "\u2190",
              key: "\x1B[D"
            },
            {
              label: "\u2192",
              key: "\x1B[C"
            }
          ].map((c) => u.jsx("button", {
            onClick: () => {
              var _a2, _b;
              (_a2 = n.current) == null ? void 0 : _a2.call(n, c.key), (_b = r.current) == null ? void 0 : _b.focus();
            },
            className: "shrink-0 rounded bg-maude-bg px-2 py-1 text-[11px] font-mono text-maude-muted active:bg-maude-accent active:text-white",
            children: c.label
          }, c.label))
        }),
        u.jsx("div", {
          ref: e,
          className: "flex-1 overflow-hidden px-1 py-1",
          onTouchStart: () => {
            var _a2;
            return (_a2 = r.current) == null ? void 0 : _a2.focus();
          }
        })
      ]
    });
  };
  function _x() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  const Px = [
    {
      label: "Google",
      url: "https://www.google.com"
    },
    {
      label: "GitHub",
      url: "https://github.com"
    },
    {
      label: "Reddit",
      url: "https://www.reddit.com"
    },
    {
      label: "HN",
      url: "https://news.ycombinator.com"
    }
  ], Rx = () => {
    const [e, t] = v.useState(""), [n, r] = v.useState(""), [l, a] = v.useState(""), [o, i] = v.useState(false), [s, c] = v.useState(""), m = v.useRef(null), [d, g] = v.useState("proxy"), [x, w] = v.useState([]), [S, _] = v.useState(-1), p = v.useCallback(async (j) => {
      if (!j.trim()) return;
      let C = j.trim();
      if (!C.startsWith("http://") && !C.startsWith("https://") && (C = "https://" + C), r(C), c(""), d === "iframe") {
        t(C), w((R) => [
          ...R.slice(0, S + 1),
          C
        ]), _((R) => R + 1);
        return;
      }
      i(true);
      try {
        const R = await fetch(`${_x()}/proxy?url=${encodeURIComponent(C)}`);
        if (!R.ok) {
          c(`Failed: ${R.status}`), i(false);
          return;
        }
        if ((R.headers.get("content-type") || "").includes("application/json")) {
          const P = await R.json();
          if (P.redirect) {
            i(false), p(P.redirect);
            return;
          }
          c(P.error || "Unknown error");
        } else a(await R.text());
        w((P) => [
          ...P.slice(0, S + 1),
          C
        ]), _((P) => P + 1);
      } catch (R) {
        c(R instanceof Error ? R.message : "Failed");
      }
      i(false);
    }, [
      d,
      S
    ]), f = () => {
      S > 0 && (_(S - 1), p(x[S - 1]));
    }, h = () => {
      S < x.length - 1 && (_(S + 1), p(x[S + 1]));
    };
    return u.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 flex-col border-b border-maude-border bg-maude-surface",
          children: [
            u.jsxs("form", {
              onSubmit: (j) => {
                j.preventDefault(), p(n);
              },
              className: "flex items-center gap-2 px-3 py-2",
              children: [
                u.jsxs("div", {
                  className: "flex gap-1",
                  children: [
                    u.jsx("button", {
                      type: "button",
                      onClick: f,
                      disabled: S <= 0,
                      className: "rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30",
                      children: "\u25C0"
                    }),
                    u.jsx("button", {
                      type: "button",
                      onClick: h,
                      disabled: S >= x.length - 1,
                      className: "rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30",
                      children: "\u25B6"
                    }),
                    u.jsx("button", {
                      type: "button",
                      onClick: () => p(n),
                      className: "rounded px-2 py-1 text-sm text-maude-muted",
                      children: "\u21BB"
                    })
                  ]
                }),
                u.jsx("input", {
                  type: "text",
                  value: n,
                  onChange: (j) => r(j.target.value),
                  placeholder: "Enter URL...",
                  className: "flex-1 rounded-lg bg-maude-bg px-3 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
                }),
                u.jsx("button", {
                  type: "button",
                  onClick: () => g(d === "proxy" ? "iframe" : "proxy"),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-[10px] text-maude-muted",
                  children: d === "proxy" ? "PROXY" : "IFRAME"
                })
              ]
            }),
            u.jsx("div", {
              className: "flex gap-1 overflow-x-auto px-3 pb-2 no-scrollbar",
              children: Px.map((j) => u.jsx("button", {
                onClick: () => {
                  r(j.url), p(j.url);
                },
                className: "shrink-0 rounded-full bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
                children: j.label
              }, j.url))
            })
          ]
        }),
        u.jsxs("div", {
          className: "flex-1 overflow-hidden",
          children: [
            o && u.jsx("div", {
              className: "flex h-full items-center justify-center",
              children: u.jsx("div", {
                className: "h-6 w-6 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
              })
            }),
            s && u.jsx("div", {
              className: "flex h-full items-center justify-center p-8 text-center",
              children: u.jsx("p", {
                className: "text-red-400",
                children: s
              })
            }),
            !o && !s && d === "proxy" && l && u.jsx("iframe", {
              srcDoc: l,
              className: "h-full w-full border-0 bg-white",
              sandbox: "allow-scripts allow-same-origin allow-forms",
              title: "Browser"
            }),
            !o && !s && d === "iframe" && e && u.jsx("iframe", {
              ref: m,
              src: e,
              className: "h-full w-full border-0 bg-white",
              sandbox: "allow-scripts allow-same-origin allow-forms allow-popups",
              title: "Browser"
            }),
            !o && !s && !l && !e && u.jsxs("div", {
              className: "flex h-full flex-col items-center justify-center gap-4 text-center",
              children: [
                u.jsx("span", {
                  className: "text-4xl",
                  children: "\u25CE"
                }),
                u.jsx("p", {
                  className: "text-sm text-maude-muted",
                  children: "Enter a URL to browse the web."
                })
              ]
            })
          ]
        })
      ]
    });
  };
  function Tx() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  const bx = () => {
    const [e, t] = v.useState([]), [n, r] = v.useState(""), [l, a] = v.useState(false), o = v.useRef(null);
    v.useEffect(() => {
      o.current && (o.current.scrollTop = o.current.scrollHeight);
    }, [
      e
    ]), v.useEffect(() => {
      t([
        {
          id: 1,
          from: "MAUDE",
          text: "Telegram integration ready. Messages from the Telegram bot will appear here.",
          date: Date.now() / 1e3,
          outgoing: false
        }
      ]);
    }, []);
    const i = async () => {
      var _a2, _b, _c2;
      if (!n.trim()) return;
      const s = n.trim();
      r(""), t((c) => [
        ...c,
        {
          id: Date.now(),
          from: "You",
          text: s,
          date: Date.now() / 1e3,
          outgoing: true
        }
      ]), a(true);
      try {
        const c = await fetch(`${Tx()}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: "mistral-large-latest",
            messages: [
              {
                role: "system",
                content: "You are MAUDE. Respond briefly and helpfully, like a text message."
              },
              {
                role: "user",
                content: s
              }
            ],
            max_tokens: 500,
            stream: false
          })
        });
        if (c.ok) {
          const d = ((_c2 = (_b = (_a2 = (await c.json()).choices) == null ? void 0 : _a2[0]) == null ? void 0 : _b.message) == null ? void 0 : _c2.content) || "...";
          t((g) => [
            ...g,
            {
              id: Date.now() + 1,
              from: "MAUDE",
              text: d,
              date: Date.now() / 1e3,
              outgoing: false
            }
          ]);
        }
      } catch {
      }
      a(false);
    };
    return u.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-3",
          children: [
            u.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                u.jsx("span", {
                  className: "text-lg",
                  children: "\u2709"
                }),
                u.jsx("h1", {
                  className: "text-sm font-semibold text-maude-text",
                  children: "Messages"
                })
              ]
            }),
            u.jsx("span", {
              className: "rounded-full bg-maude-bg px-2 py-0.5 text-[10px] text-maude-muted",
              children: "Telegram"
            })
          ]
        }),
        u.jsxs("div", {
          ref: o,
          className: "no-scrollbar flex-1 overflow-y-auto px-4 py-4",
          children: [
            e.map((s) => u.jsx("div", {
              className: `mb-3 flex ${s.outgoing ? "justify-end" : "justify-start"}`,
              children: u.jsxs("div", {
                className: `max-w-[80%] rounded-2xl px-4 py-2.5 ${s.outgoing ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`,
                children: [
                  !s.outgoing && u.jsx("div", {
                    className: "mb-0.5 text-[10px] font-medium text-maude-accent",
                    children: s.from
                  }),
                  u.jsx("p", {
                    className: "text-sm",
                    children: s.text
                  }),
                  u.jsx("div", {
                    className: "mt-1 text-[10px] opacity-50",
                    children: new Date(s.date * 1e3).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit"
                    })
                  })
                ]
              })
            }, s.id)),
            l && u.jsx("div", {
              className: "flex justify-start",
              children: u.jsx("div", {
                className: "rounded-2xl bg-maude-surface px-4 py-3",
                children: u.jsxs("div", {
                  className: "flex gap-1",
                  children: [
                    u.jsx("span", {
                      className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                      style: {
                        animationDelay: "0ms"
                      }
                    }),
                    u.jsx("span", {
                      className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                      style: {
                        animationDelay: "150ms"
                      }
                    }),
                    u.jsx("span", {
                      className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                      style: {
                        animationDelay: "300ms"
                      }
                    })
                  ]
                })
              })
            })
          ]
        }),
        u.jsxs("div", {
          className: "flex items-center gap-2 border-t border-maude-border bg-maude-surface p-3",
          children: [
            u.jsx("input", {
              type: "text",
              value: n,
              onChange: (s) => r(s.target.value),
              onKeyDown: (s) => {
                s.key === "Enter" && i();
              },
              placeholder: "Message...",
              className: "min-h-[44px] flex-1 rounded-xl bg-maude-bg px-4 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
            }),
            u.jsx("button", {
              onClick: i,
              disabled: !n.trim() || l,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30",
              children: "\u2191"
            })
          ]
        })
      ]
    });
  };
  function Kr() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  function Mx(e) {
    return e < 1024 ? e + " B" : e < 1048576 ? (e / 1024).toFixed(1) + " KB" : (e / 1048576).toFixed(1) + " MB";
  }
  function Dx(e) {
    return new Date(e * 1e3).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }
  const Lx = () => {
    const [e, t] = v.useState([]), [n, r] = v.useState(""), [l, a] = v.useState(false), [o, i] = v.useState(""), [s, c] = v.useState("shared"), m = v.useRef(null), d = v.useCallback(async (w) => {
      a(true), i("");
      try {
        const S = s === "transfers" ? `${Kr()}/transfers` : w ? `${Kr()}/list?path=${encodeURIComponent(w)}` : `${Kr()}/list`, p = await (await fetch(S)).json();
        p.error ? i(p.error) : (t(p.files || []), r(p.path || ""));
      } catch (S) {
        i(S instanceof Error ? S.message : "Failed");
      }
      a(false);
    }, [
      s
    ]);
    v.useEffect(() => {
      d();
    }, [
      d
    ]);
    const g = (w) => {
      window.open(`${Kr()}/${s === "transfers" ? "download-transfer" : "download"}/${encodeURIComponent(w)}`);
    }, x = async (w) => {
      var _a2;
      const S = (_a2 = w.target.files) == null ? void 0 : _a2[0];
      if (S) {
        a(true);
        try {
          const p = await (await fetch(`${Kr()}/upload/${encodeURIComponent(S.name)}`, {
            method: "POST",
            body: S
          })).json();
          p.error ? i(p.error) : d();
        } catch (_) {
          i(_ instanceof Error ? _.message : "Upload failed");
        }
        a(false), m.current && (m.current.value = "");
      }
    };
    return u.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-3",
          children: [
            u.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                u.jsx("span", {
                  className: "text-lg",
                  children: "\u25A4"
                }),
                u.jsx("h1", {
                  className: "text-sm font-semibold text-maude-text",
                  children: "Files"
                })
              ]
            }),
            u.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                u.jsx("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = m.current) == null ? void 0 : _a2.click();
                  },
                  className: "rounded-lg fire-bg px-3 py-1 text-xs font-medium text-white",
                  children: "Upload"
                }),
                u.jsx("button", {
                  onClick: () => d(),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-muted",
                  children: "\u21BB"
                }),
                u.jsx("input", {
                  ref: m,
                  type: "file",
                  onChange: x,
                  className: "hidden"
                })
              ]
            })
          ]
        }),
        u.jsx("div", {
          className: "flex shrink-0 border-b border-maude-border bg-maude-surface",
          children: [
            "shared",
            "transfers"
          ].map((w) => u.jsx("button", {
            onClick: () => c(w),
            className: `flex-1 py-2 text-xs font-medium capitalize ${s === w ? "border-b-2 border-maude-accent text-maude-accent" : "text-maude-muted"}`,
            children: w
          }, w))
        }),
        n && u.jsxs("div", {
          className: "flex items-center gap-2 border-b border-maude-border bg-maude-surface/50 px-4 py-2",
          children: [
            u.jsxs("button", {
              onClick: () => {
                const w = n.split("/").slice(0, -1).join("/");
                d(w || void 0);
              },
              className: "text-xs text-maude-accent",
              children: [
                "\u2190",
                " Up"
              ]
            }),
            u.jsx("span", {
              className: "truncate text-xs text-maude-muted",
              children: n
            })
          ]
        }),
        o && u.jsx("div", {
          className: "bg-red-900/30 px-4 py-2 text-xs text-red-400",
          children: o
        }),
        u.jsxs("div", {
          className: "no-scrollbar flex-1 overflow-y-auto",
          children: [
            l && u.jsx("div", {
              className: "flex h-32 items-center justify-center",
              children: u.jsx("div", {
                className: "h-6 w-6 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
              })
            }),
            !l && e.length === 0 && u.jsx("div", {
              className: "flex h-32 items-center justify-center",
              children: u.jsx("p", {
                className: "text-sm text-maude-muted",
                children: "No files found."
              })
            }),
            !l && e.map((w) => u.jsxs("button", {
              onClick: () => w.is_dir ? d(n ? `${n}/${w.name}` : w.name) : g(w.name),
              className: "flex w-full items-center gap-3 border-b border-maude-border/50 px-4 py-3 text-left hover:bg-maude-surface",
              children: [
                u.jsx("span", {
                  className: "text-lg",
                  children: w.is_dir ? "\u{1F4C1}" : "\u{1F4C4}"
                }),
                u.jsxs("div", {
                  className: "min-w-0 flex-1",
                  children: [
                    u.jsx("div", {
                      className: "truncate text-sm text-maude-text",
                      children: w.name
                    }),
                    u.jsxs("div", {
                      className: "mt-0.5 text-[10px] text-maude-muted",
                      children: [
                        w.is_dir ? "Directory" : Mx(w.size),
                        " \xB7 ",
                        Dx(w.modified)
                      ]
                    })
                  ]
                }),
                !w.is_dir && u.jsx("span", {
                  className: "text-xs text-maude-muted",
                  children: "\u2193"
                })
              ]
            }, w.name))
          ]
        })
      ]
    });
  };
  function Fc() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  const Ox = [
    {
      id: "dark",
      label: "MAUDE Dark",
      desc: "Default dark theme"
    },
    {
      id: "professional",
      label: "Professional",
      desc: "Clean corporate dark"
    },
    {
      id: "modern",
      label: "Modern Terminal",
      desc: "Clean slate & indigo"
    },
    {
      id: "retro-green",
      label: "80s Green CRT",
      desc: "Phosphor green terminal"
    },
    {
      id: "retro-amber",
      label: "80s Amber CRT",
      desc: "Amber phosphor terminal"
    }
  ];
  function Ix(e) {
    document.documentElement.setAttribute("data-theme", e), localStorage.setItem("maude-theme", e);
  }
  const zx = () => {
    var _a2, _b;
    const [e, t] = v.useState(null), [n, r] = v.useState([]), [l, a] = v.useState(() => localStorage.getItem("maude-default-model") || "mistral-large-latest"), [o, i] = v.useState(() => localStorage.getItem("maude-default-voice") || "NATF2.pt"), [s, c] = v.useState(() => localStorage.getItem("maude-theme") || "dark"), m = e !== null, d = (e == null ? void 0 : e.gateway_port) ?? 3e4, g = (_a2 = e == null ? void 0 : e.services) == null ? void 0 : _a2.llama_server, x = (_b = e == null ? void 0 : e.services) == null ? void 0 : _b.personaplex;
    v.useEffect(() => {
      fetch(`${Fc()}/health`).then((h) => h.json()).then(t).catch(() => t(null)), fetch(`${Fc()}/models`).then((h) => h.json()).then((h) => r(h.models || [])).catch(() => r([]));
    }, []);
    const w = (h) => {
      a(h), localStorage.setItem("maude-default-model", h);
    }, S = (h) => {
      i(h), localStorage.setItem("maude-default-voice", h);
    }, _ = (h) => h ? h.status === "up" ? {
      text: `${h.port} (up)`,
      color: "text-green-400"
    } : {
      text: `${h.port} (down)`,
      color: "text-red-400"
    } : {
      text: "\u2014",
      color: "text-maude-muted"
    }, p = _(g), f = _(x);
    return u.jsxs("div", {
      className: "no-scrollbar h-full overflow-y-auto bg-maude-bg",
      children: [
        u.jsx("div", {
          className: "border-b border-maude-border bg-maude-surface px-4 py-3",
          children: u.jsx("h1", {
            className: "text-lg font-semibold text-maude-text",
            children: "Settings"
          })
        }),
        u.jsxs("div", {
          className: "space-y-6 p-4",
          children: [
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Connection"
                }),
                u.jsxs("div", {
                  className: "space-y-2 rounded-xl bg-maude-surface p-4",
                  children: [
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Spark Status"
                        }),
                        u.jsxs("span", {
                          className: `flex items-center gap-1.5 text-sm ${m ? "text-green-400" : "text-red-400"}`,
                          children: [
                            u.jsx("span", {
                              className: `h-2 w-2 rounded-full ${m ? "bg-green-400" : "bg-red-400"}`
                            }),
                            m ? "Connected" : "Offline"
                          ]
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Gateway"
                        }),
                        u.jsx("span", {
                          className: `font-mono text-sm ${m ? "text-green-400" : "text-maude-muted"}`,
                          children: m ? `${d} (up)` : "\u2014"
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "LLM"
                        }),
                        u.jsx("span", {
                          className: `font-mono text-sm ${p.color}`,
                          children: p.text
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "PersonaPlex"
                        }),
                        u.jsx("span", {
                          className: `font-mono text-sm ${f.color}`,
                          children: f.text
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Tailscale"
                        }),
                        u.jsx("span", {
                          className: "text-sm text-green-400",
                          children: "Active"
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Host"
                        }),
                        u.jsx("span", {
                          className: "font-mono text-sm text-maude-muted",
                          children: window.location.host
                        })
                      ]
                    })
                  ]
                })
              ]
            }),
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Theme"
                }),
                u.jsx("div", {
                  className: "space-y-1 rounded-xl bg-maude-surface p-2",
                  children: Ox.map((h) => u.jsxs("button", {
                    onClick: () => {
                      c(h.id), Ix(h.id);
                    },
                    className: `flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${h.id === s ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                    children: [
                      u.jsx("span", {
                        children: h.label
                      }),
                      u.jsx("span", {
                        className: "text-xs text-maude-muted",
                        children: h.desc
                      })
                    ]
                  }, h.id))
                })
              ]
            }),
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Default Model"
                }),
                u.jsxs("div", {
                  className: "space-y-1 rounded-xl bg-maude-surface p-2",
                  children: [
                    n.map((h) => u.jsxs("button", {
                      onClick: () => w(h.id),
                      className: `flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${h.id === l ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                      children: [
                        u.jsxs("div", {
                          className: "flex items-center gap-2",
                          children: [
                            u.jsx("span", {
                              className: `h-2 w-2 rounded-full ${h.available ? "bg-green-400" : "bg-red-400"}`
                            }),
                            h.id
                          ]
                        }),
                        u.jsx("span", {
                          className: "text-xs text-maude-muted",
                          children: h.provider
                        })
                      ]
                    }, h.id)),
                    n.length === 0 && u.jsx("p", {
                      className: "px-3 py-2 text-sm text-maude-muted",
                      children: "Loading models..."
                    })
                  ]
                })
              ]
            }),
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "PersonaPlex Voice"
                }),
                u.jsx("div", {
                  className: "rounded-xl bg-maude-surface p-4",
                  children: u.jsx("select", {
                    value: o,
                    onChange: (h) => S(h.target.value),
                    className: "w-full rounded-lg bg-maude-bg px-3 py-2.5 text-sm text-maude-text outline-none focus:ring-1 focus:ring-maude-accent",
                    children: [
                      "NATF0.pt",
                      "NATF1.pt",
                      "NATF2.pt",
                      "NATF3.pt",
                      "NATM0.pt",
                      "NATM1.pt",
                      "NATM2.pt",
                      "NATM3.pt"
                    ].map((h) => u.jsxs("option", {
                      value: h,
                      children: [
                        h.replace(".pt", ""),
                        h === "NATF2.pt" ? " (MAUDE)" : "",
                        h === "NATM1.pt" ? " (Male)" : ""
                      ]
                    }, h))
                  })
                })
              ]
            }),
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Network"
                }),
                u.jsx("div", {
                  className: "rounded-xl bg-maude-surface p-4",
                  children: u.jsx("p", {
                    className: "text-sm text-maude-muted",
                    children: "Network settings are managed via Tailscale and your device's system settings."
                  })
                })
              ]
            }),
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "About"
                }),
                u.jsxs("div", {
                  className: "space-y-2 rounded-xl bg-maude-surface p-4",
                  children: [
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Version"
                        }),
                        u.jsx("span", {
                          className: "text-sm text-maude-muted",
                          children: "1.8.3"
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Engine"
                        }),
                        u.jsx("span", {
                          className: "text-sm text-maude-muted",
                          children: "Mistral + Codestral + Claude"
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Voice"
                        }),
                        u.jsxs("span", {
                          className: "text-sm text-maude-muted",
                          children: [
                            "PersonaPlex (",
                            (localStorage.getItem("maude-default-voice") || "NATF2.pt").replace(".pt", ""),
                            ")"
                          ]
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Hub"
                        }),
                        u.jsx("span", {
                          className: "text-sm font-mono",
                          children: "DGX Spark"
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "pt-2 text-center text-xs text-maude-muted",
                      children: [
                        u.jsx("span", {
                          className: "fire-gradient font-bold",
                          children: "MAUDE"
                        }),
                        " \u2014 Multi-Agent Unified Dispatch Engine"
                      ]
                    })
                  ]
                })
              ]
            })
          ]
        })
      ]
    });
  };
  function wa() {
    const e = window.location;
    return `${e.protocol}//${e.host}`;
  }
  function Ax(e = 1e4) {
    const [t, n] = v.useState(null), [r, l] = v.useState(true), a = v.useCallback(async () => {
      try {
        const s = await fetch(`${wa()}/api/collab/status`);
        s.ok && n(await s.json());
      } catch {
      } finally {
        l(false);
      }
    }, []);
    v.useEffect(() => {
      a();
      const s = setInterval(a, e);
      return () => clearInterval(s);
    }, [
      a,
      e
    ]);
    const o = v.useCallback(async (s, c = "", m = []) => {
      const d = await fetch(`${wa()}/api/collab/projects`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          name: s,
          description: c,
          tags: m
        })
      });
      if (d.ok) return a(), await d.json();
    }, [
      a
    ]), i = v.useCallback(async (s, c = "", m = "SHELL") => {
      const d = await fetch(`${wa()}/api/collab/tasks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: s,
          target: c,
          capability: m
        })
      });
      if (d.ok) return a(), await d.json();
    }, [
      a
    ]);
    return {
      status: t,
      loading: r,
      refresh: a,
      createProject: o,
      dispatchTask: i
    };
  }
  function Ux() {
    const e = navigator.userAgent;
    return /iPad/.test(e) ? {
      clientType: "ipad",
      label: "iPad"
    } : /iPhone/.test(e) ? {
      clientType: "iphone",
      label: "iPhone"
    } : /Android/.test(e) && /Mobile/.test(e) ? {
      clientType: "android",
      label: "Android"
    } : /Android/.test(e) ? {
      clientType: "android-tablet",
      label: "Android Tablet"
    } : /Macintosh/.test(e) ? {
      clientType: "mac",
      label: "Mac"
    } : /Windows/.test(e) ? {
      clientType: "windows",
      label: "Windows"
    } : {
      clientType: "phone",
      label: "Phone"
    };
  }
  let Bc = false;
  function $x() {
    if (Bc) return;
    Bc = true;
    const e = Ux(), t = `${e.clientType}-${Math.random().toString(36).slice(2, 8)}`, n = () => {
      fetch(`${wa()}/api/collab/presence`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          client_id: t,
          client_type: e.clientType,
          hostname: `Matts ${e.label}`,
          platform: e.clientType,
          activity: document.visibilityState === "visible" ? "browsing app" : "background"
        })
      }).catch(() => {
      });
    };
    n(), setInterval(n, 3e4);
  }
  function Ji(e, t) {
    const n = Math.max(0, Math.floor(t - e));
    return n < 60 ? `${n}s ago` : n < 3600 ? `${Math.floor(n / 60)}m ago` : n < 86400 ? `${Math.floor(n / 3600)}h ago` : `${Math.floor(n / 86400)}d ago`;
  }
  const Fx = {
    pending: "bg-yellow-500",
    running: "bg-blue-500",
    completed: "bg-green-500",
    failed: "bg-red-500"
  }, Wc = {
    gateway: "\u2B21",
    tui: ">_",
    cli: "$",
    macos: "\u{1F4BB}",
    mac: "\u{1F4BB}",
    iphone: "\u{1F4F1}",
    ipad: "\u{1F4F2}",
    android: "\u{1F4F1}",
    "android-tablet": "\u{1F4F2}",
    phone: "\u{1F4F1}",
    windows: "\u{1F5A5}",
    unknown: "\u25CF"
  }, Bx = ({ entry: e, now: t }) => u.jsxs("div", {
    className: "flex items-center gap-3 rounded-xl bg-maude-surface p-3",
    children: [
      u.jsx("div", {
        className: "flex h-10 w-10 items-center justify-center rounded-full bg-maude-card text-lg",
        children: Wc[e.client_type] || Wc.unknown
      }),
      u.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          u.jsxs("div", {
            className: "flex items-center gap-2",
            children: [
              u.jsx("span", {
                className: "font-medium text-maude-text",
                children: e.hostname
              }),
              u.jsx("span", {
                className: "text-[10px] text-maude-muted",
                children: e.client_type
              }),
              u.jsx("span", {
                className: "ml-auto inline-block h-2 w-2 rounded-full bg-green-400"
              })
            ]
          }),
          u.jsxs("p", {
            className: "truncate text-xs text-maude-muted",
            children: [
              e.activity || "idle",
              " \xB7 ",
              Ji(e.last_seen, t)
            ]
          })
        ]
      })
    ]
  }), Vc = {
    chat: "\u{1F4AC}",
    task_dispatched: "\u{1F680}",
    project_created: "\u{1F4C1}",
    custom: "\u2022"
  }, Wx = ({ event: e, now: t }) => u.jsxs("div", {
    className: "flex items-start gap-2 py-1.5",
    children: [
      u.jsx("span", {
        className: "mt-0.5 text-sm",
        children: Vc[e.type] || Vc.custom
      }),
      u.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          u.jsx("p", {
            className: "text-sm text-maude-text",
            children: e.summary
          }),
          u.jsxs("p", {
            className: "text-[10px] text-maude-muted",
            children: [
              e.hostname,
              " \xB7 ",
              Ji(e.ts, t)
            ]
          })
        ]
      })
    ]
  }), Vx = ({ project: e }) => u.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      u.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          u.jsx("span", {
            className: "text-sm font-medium text-maude-text",
            children: e.name
          }),
          e.tags.map((t) => u.jsx("span", {
            className: "rounded bg-maude-card px-1.5 py-0.5 text-[10px] text-maude-muted",
            children: t
          }, t))
        ]
      }),
      e.description && u.jsx("p", {
        className: "mt-1 text-xs text-maude-muted",
        children: e.description
      }),
      u.jsxs("div", {
        className: "mt-2 flex gap-3 text-[10px] text-maude-muted",
        children: [
          u.jsxs("span", {
            children: [
              e.conversations.length,
              " conversations"
            ]
          }),
          u.jsxs("span", {
            children: [
              e.files.length,
              " files"
            ]
          }),
          u.jsx("span", {
            children: e.hostname
          })
        ]
      })
    ]
  }), Hx = ({ task: e, now: t }) => u.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      u.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          u.jsx("span", {
            className: `inline-block h-2 w-2 rounded-full ${Fx[e.status] || "bg-gray-500"}`
          }),
          u.jsx("span", {
            className: "text-[10px] font-medium uppercase text-maude-muted",
            children: e.status
          }),
          u.jsx("span", {
            className: "ml-auto text-[10px] text-maude-muted",
            children: Ji(e.created_at, t)
          })
        ]
      }),
      u.jsx("p", {
        className: "mt-1 truncate text-sm text-maude-text",
        children: e.prompt
      }),
      u.jsxs("div", {
        className: "mt-1 flex gap-2 text-[10px] text-maude-muted",
        children: [
          u.jsxs("span", {
            children: [
              e.source,
              " \u2192 ",
              e.target || "local"
            ]
          }),
          u.jsx("span", {
            children: e.capability
          })
        ]
      }),
      e.result && u.jsx("pre", {
        className: "mt-2 max-h-20 overflow-auto rounded bg-maude-card p-2 text-[10px] text-maude-text",
        children: e.result.slice(0, 300)
      })
    ]
  }), Qx = () => {
    const { status: e, loading: t } = Ax(), [n, r] = v.useState("presence");
    if (t) return u.jsx("div", {
      className: "flex h-full items-center justify-center text-maude-muted",
      children: "Loading collaboration status..."
    });
    if (!e) return u.jsx("div", {
      className: "flex h-full items-center justify-center text-maude-muted",
      children: "Unable to connect to gateway"
    });
    const l = e.ts, a = [
      {
        key: "presence",
        label: "Online",
        count: e.presence.length
      },
      {
        key: "activity",
        label: "Activity",
        count: e.activity.length
      },
      {
        key: "projects",
        label: "Projects",
        count: e.projects.length
      },
      {
        key: "tasks",
        label: "Tasks",
        count: e.tasks.length
      }
    ];
    return u.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        u.jsxs("div", {
          className: "flex items-center gap-3 px-4 pt-4 pb-2",
          children: [
            u.jsx("h1", {
              className: "text-lg font-bold text-maude-text",
              children: "Collaboration"
            }),
            u.jsxs("span", {
              className: "ml-auto flex items-center gap-1 text-xs text-maude-muted",
              children: [
                u.jsx("span", {
                  className: "inline-block h-2 w-2 rounded-full bg-green-400"
                }),
                e.hostname
              ]
            })
          ]
        }),
        u.jsx("div", {
          className: "flex gap-1 px-4 pb-3",
          children: a.map((o) => u.jsxs("button", {
            onClick: () => r(o.key),
            className: `rounded-full px-3 py-1 text-xs font-medium transition-colors ${n === o.key ? "bg-maude-accent text-white" : "bg-maude-surface text-maude-muted"}`,
            children: [
              o.label,
              o.count > 0 && u.jsx("span", {
                className: "ml-1 opacity-70",
                children: o.count
              })
            ]
          }, o.key))
        }),
        u.jsxs("div", {
          className: "flex-1 overflow-y-auto px-4 pb-4",
          children: [
            n === "presence" && u.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.presence.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No devices online"
              }) : e.presence.map((o) => u.jsx(Bx, {
                entry: o,
                now: l
              }, o.client_id))
            }),
            n === "activity" && u.jsx("div", {
              className: "flex flex-col divide-y divide-maude-border",
              children: e.activity.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No recent activity"
              }) : e.activity.map((o) => u.jsx(Wx, {
                event: o,
                now: l
              }, o.id))
            }),
            n === "projects" && u.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.projects.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No projects yet"
              }) : e.projects.map((o) => u.jsx(Vx, {
                project: o
              }, o.id))
            }),
            n === "tasks" && u.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.tasks.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No tasks dispatched"
              }) : e.tasks.map((o) => u.jsx(Hx, {
                task: o,
                now: l
              }, o.id))
            })
          ]
        })
      ]
    });
  };
  function Kx() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  async function rr(e) {
    try {
      const t = await fetch(`${Kx()}/api/command-center/${e}`);
      return t.ok ? await t.json() : null;
    } catch {
      return null;
    }
  }
  function Gx(e = 1e4) {
    const [t, n] = v.useState(null), [r, l] = v.useState(null), [a, o] = v.useState([]), [i, s] = v.useState([]), [c, m] = v.useState(null), [d, g] = v.useState([]), [x, w] = v.useState(true), S = v.useCallback(async () => {
      const [_, p, f, h, j, C] = await Promise.all([
        rr("system"),
        rr("gpu-processes"),
        rr("sessions?limit=10"),
        rr("activity?limit=15"),
        rr("scheduler"),
        rr("nodes")
      ]);
      n(_), l(p && Array.isArray(p.processes) ? p : null), o((f == null ? void 0 : f.sessions) || []), s((h == null ? void 0 : h.activities) || []), m(j), g((C == null ? void 0 : C.nodes) || []), w(false);
    }, []);
    return v.useEffect(() => {
      S();
      const _ = setInterval(S, e);
      return () => clearInterval(_);
    }, [
      S,
      e
    ]), {
      system: t,
      gpuProcesses: r,
      sessions: a,
      activity: i,
      scheduler: c,
      nodes: d,
      loading: x,
      refresh: S
    };
  }
  const Tn = ({ label: e, value: t, sub: n, color: r = "text-maude-accent" }) => u.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      u.jsx("p", {
        className: "text-[10px] uppercase tracking-wider text-maude-muted",
        children: e
      }),
      u.jsx("p", {
        className: `text-xl font-bold ${r}`,
        children: t
      }),
      n && u.jsx("p", {
        className: "text-[10px] text-maude-muted",
        children: n
      })
    ]
  }), Yx = ({ processes: e }) => {
    const t = e.total_mb > 0 ? e.used_mb / e.total_mb * 100 : 0;
    return u.jsxs("div", {
      className: "rounded-xl bg-maude-surface p-3",
      children: [
        u.jsxs("div", {
          className: "mb-2 flex items-center justify-between",
          children: [
            u.jsx("p", {
              className: "text-xs font-medium text-maude-text",
              children: "GPU Memory"
            }),
            u.jsxs("p", {
              className: "text-xs text-maude-muted",
              children: [
                (e.used_mb / 1024).toFixed(1),
                "GB / ",
                (e.total_mb / 1024).toFixed(0),
                "GB"
              ]
            })
          ]
        }),
        u.jsx("div", {
          className: "h-2 overflow-hidden rounded-full bg-maude-bg",
          children: u.jsx("div", {
            className: "h-full rounded-full bg-maude-accent transition-all",
            style: {
              width: `${Math.min(t, 100)}%`
            }
          })
        }),
        e.processes.length > 0 && u.jsx("div", {
          className: "mt-2 space-y-1",
          children: e.processes.map((n) => u.jsxs("div", {
            className: "flex items-center justify-between text-[11px]",
            children: [
              u.jsx("span", {
                className: "truncate text-maude-text",
                children: n.name
              }),
              u.jsxs("span", {
                className: "text-maude-muted",
                children: [
                  (n.memory_mb / 1024).toFixed(1),
                  "GB"
                ]
              })
            ]
          }, n.pid))
        })
      ]
    });
  }, Jx = ({ node: e }) => u.jsxs("div", {
    className: "flex items-center gap-3 rounded-xl bg-maude-surface p-3",
    children: [
      u.jsx("span", {
        className: `inline-block h-2.5 w-2.5 rounded-full ${e.status === "online" ? "bg-green-400" : e.status === "offline" ? "bg-red-400" : "bg-yellow-400"}`
      }),
      u.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          u.jsxs("div", {
            className: "flex items-center gap-2",
            children: [
              u.jsx("span", {
                className: "text-sm font-medium text-maude-text",
                children: e.name
              }),
              u.jsx("span", {
                className: "text-[10px] text-maude-muted",
                children: e.type
              })
            ]
          }),
          e.services && u.jsx("div", {
            className: "mt-1 flex flex-wrap gap-1.5",
            children: Object.entries(e.services).map(([t, n]) => u.jsx("span", {
              className: `rounded px-1.5 py-0.5 text-[9px] ${n ? "bg-green-400/10 text-green-400" : "bg-red-400/10 text-red-400"}`,
              children: t
            }, t))
          }),
          e.ip && u.jsxs("p", {
            className: "mt-0.5 text-[10px] text-maude-muted",
            children: [
              e.os || e.platform || "",
              " ",
              e.ip
            ]
          })
        ]
      })
    ]
  }), Xx = ({ task: e }) => u.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      u.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          u.jsx("span", {
            className: `inline-block h-2 w-2 rounded-full ${e.enabled ? "bg-green-400" : "bg-gray-500"}`
          }),
          u.jsx("span", {
            className: "text-sm font-medium text-maude-text",
            children: e.name
          }),
          u.jsx("span", {
            className: "ml-auto font-mono text-[10px] text-maude-muted",
            children: e.cron
          })
        ]
      }),
      u.jsx("p", {
        className: "mt-1 truncate text-xs text-maude-muted",
        children: e.prompt
      }),
      u.jsxs("div", {
        className: "mt-1 flex gap-3 text-[10px] text-maude-muted",
        children: [
          u.jsxs("span", {
            children: [
              e.run_count,
              " runs"
            ]
          }),
          e.last_run && u.jsxs("span", {
            children: [
              "Last: ",
              new Date(e.last_run).toLocaleDateString()
            ]
          })
        ]
      })
    ]
  }), Zx = ({ item: e }) => u.jsxs("div", {
    className: "flex items-start gap-2 py-2",
    children: [
      u.jsx("span", {
        className: `mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full ${e.role === "user" ? "bg-green-400" : "bg-maude-accent"}`
      }),
      u.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          u.jsxs("div", {
            className: "flex items-center gap-1.5",
            children: [
              u.jsx("span", {
                className: "text-[10px] font-medium uppercase text-maude-muted",
                children: e.channel
              }),
              u.jsx("span", {
                className: "text-[10px] text-maude-muted",
                children: e.role
              })
            ]
          }),
          u.jsx("p", {
            className: "truncate text-xs text-maude-text",
            children: e.content
          })
        ]
      })
    ]
  }), qx = ({ session: e }) => u.jsxs("div", {
    className: "flex items-center justify-between rounded-xl bg-maude-surface p-3",
    children: [
      u.jsxs("div", {
        children: [
          u.jsx("span", {
            className: "text-sm font-medium text-maude-text",
            children: e.session_id.slice(0, 8)
          }),
          u.jsx("span", {
            className: "ml-2 text-[10px] text-maude-muted",
            children: e.channel
          })
        ]
      }),
      u.jsxs("div", {
        className: "text-right",
        children: [
          u.jsxs("p", {
            className: "text-xs text-maude-muted",
            children: [
              e.message_count,
              " msgs"
            ]
          }),
          u.jsx("p", {
            className: "text-[10px] text-maude-muted",
            children: new Date(e.last_message_at).toLocaleDateString()
          })
        ]
      })
    ]
  }), ey = () => {
    var _a2, _b, _c2, _d2, _e2, _f2, _g2, _h2, _i2;
    const { system: e, gpuProcesses: t, sessions: n, activity: r, scheduler: l, nodes: a, loading: o, refresh: i } = Gx(), [s, c] = v.useState("overview");
    if (o) return u.jsx("div", {
      className: "flex h-full items-center justify-center text-maude-muted",
      children: "Loading command center..."
    });
    const m = [
      {
        key: "overview",
        label: "Overview"
      },
      {
        key: "nodes",
        label: "Nodes"
      },
      {
        key: "activity",
        label: "Activity"
      },
      {
        key: "scheduler",
        label: "Tasks"
      }
    ], d = typeof ((_a2 = e == null ? void 0 : e.gpu) == null ? void 0 : _a2.temperature_c) == "number" ? e.gpu.temperature_c : 0, g = d > 80 ? "text-red-400" : d > 60 ? "text-yellow-400" : "text-green-400";
    return u.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        u.jsxs("div", {
          className: "flex items-center gap-3 px-4 pt-4 pb-2",
          children: [
            u.jsx("h1", {
              className: "text-lg font-bold text-maude-text",
              children: "Command Center"
            }),
            u.jsx("button", {
              onClick: i,
              className: "ml-auto rounded-lg bg-maude-surface px-2 py-1 text-xs text-maude-muted active:bg-maude-card",
              children: "Refresh"
            })
          ]
        }),
        u.jsx("div", {
          className: "flex gap-1 px-4 pb-3",
          children: m.map((x) => u.jsx("button", {
            onClick: () => c(x.key),
            className: `rounded-full px-3 py-1 text-xs font-medium transition-colors ${s === x.key ? "bg-maude-accent text-white" : "bg-maude-surface text-maude-muted"}`,
            children: x.label
          }, x.key))
        }),
        u.jsxs("div", {
          className: "flex-1 overflow-y-auto px-4 pb-4",
          children: [
            s === "overview" && u.jsxs("div", {
              className: "space-y-3",
              children: [
                u.jsxs("div", {
                  className: "grid grid-cols-2 gap-2",
                  children: [
                    u.jsx(Tn, {
                      label: "CPU",
                      value: `${(e == null ? void 0 : e.cpu_percent) ?? 0}%`,
                      sub: `${((_b = e == null ? void 0 : e.ram) == null ? void 0 : _b.used_gb) ?? 0}/${((_c2 = e == null ? void 0 : e.ram) == null ? void 0 : _c2.total_gb) ?? 0}GB RAM`
                    }),
                    u.jsx(Tn, {
                      label: "GPU Temp",
                      value: `${d}\xB0C`,
                      sub: ((_d2 = e == null ? void 0 : e.gpu) == null ? void 0 : _d2.name) || "N/A",
                      color: g
                    }),
                    u.jsx(Tn, {
                      label: "Disk",
                      value: `${((_e2 = e == null ? void 0 : e.disk) == null ? void 0 : _e2.percent) ?? 0}%`,
                      sub: `${((_f2 = e == null ? void 0 : e.disk) == null ? void 0 : _f2.used_gb) ?? 0}/${((_g2 = e == null ? void 0 : e.disk) == null ? void 0 : _g2.total_gb) ?? 0}GB`
                    }),
                    u.jsx(Tn, {
                      label: "Sessions",
                      value: n.length,
                      sub: `${((_h2 = l == null ? void 0 : l.stats) == null ? void 0 : _h2.active) ?? 0} scheduled tasks`
                    })
                  ]
                }),
                t && u.jsx(Yx, {
                  processes: t
                }),
                n.length > 0 && u.jsxs(u.Fragment, {
                  children: [
                    u.jsx("p", {
                      className: "pt-1 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                      children: "Recent Sessions"
                    }),
                    u.jsx("div", {
                      className: "space-y-1.5",
                      children: n.slice(0, 5).map((x) => u.jsx(qx, {
                        session: x
                      }, x.session_id + x.channel))
                    })
                  ]
                })
              ]
            }),
            s === "nodes" && u.jsx("div", {
              className: "space-y-2",
              children: a.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No nodes detected"
              }) : a.map((x, w) => u.jsx(Jx, {
                node: x
              }, x.name + w))
            }),
            s === "activity" && u.jsx("div", {
              className: "divide-y divide-maude-border",
              children: r.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No recent activity"
              }) : r.map((x, w) => u.jsx(Zx, {
                item: x
              }, w))
            }),
            s === "scheduler" && u.jsxs("div", {
              className: "space-y-2",
              children: [
                (l == null ? void 0 : l.stats) && u.jsxs("div", {
                  className: "grid grid-cols-3 gap-2",
                  children: [
                    u.jsx(Tn, {
                      label: "Total",
                      value: l.stats.total
                    }),
                    u.jsx(Tn, {
                      label: "Active",
                      value: l.stats.active,
                      color: "text-green-400"
                    }),
                    u.jsx(Tn, {
                      label: "Runs",
                      value: l.stats.total_runs
                    })
                  ]
                }),
                ((_i2 = l == null ? void 0 : l.tasks) == null ? void 0 : _i2.length) ? l.tasks.map((x) => u.jsx(Xx, {
                  task: x
                }, x.id)) : u.jsx("p", {
                  className: "py-8 text-center text-sm text-maude-muted",
                  children: "No scheduled tasks"
                })
              ]
            })
          ]
        })
      ]
    });
  }, ty = [
    {
      path: "/",
      label: "Home",
      icon: "\u2B21",
      match: [
        "/"
      ]
    },
    {
      path: "/maude",
      label: "Chat",
      icon: "\u25C6",
      match: [
        "/maude"
      ]
    },
    {
      path: "/maude/voice",
      label: "Voice",
      icon: "\u{1F399}\uFE0F",
      match: [
        "/maude/voice"
      ]
    },
    {
      path: "/terminal",
      label: "Term",
      icon: ">_",
      match: [
        "/terminal"
      ]
    },
    {
      path: "/files",
      label: "Files",
      icon: "\u25A4",
      match: [
        "/files"
      ]
    },
    {
      path: "/collab",
      label: "Collab",
      icon: "\u29BF",
      match: [
        "/collab"
      ]
    },
    {
      path: "/command-center",
      label: "System",
      icon: "\u25A3",
      match: [
        "/command-center"
      ]
    },
    {
      path: "/settings",
      label: "Set",
      icon: "\u2699",
      match: [
        "/settings"
      ]
    }
  ], ny = () => {
    const e = Gi(), t = ho();
    return u.jsx("nav", {
      className: "safe-bottom flex shrink-0 items-center justify-around border-t border-maude-border bg-maude-surface px-1 pb-1 pt-1",
      children: ty.map((n) => {
        const r = n.match.includes(e.pathname);
        return u.jsxs("button", {
          onClick: () => t(n.path),
          className: `flex min-h-[44px] min-w-[44px] flex-col items-center justify-center rounded-lg px-2 py-1 text-xs transition-colors ${r ? "text-maude-accent" : "text-maude-muted hover:text-maude-text"}`,
          children: [
            u.jsx("span", {
              className: "text-base leading-none",
              children: n.icon
            }),
            u.jsx("span", {
              className: "mt-0.5",
              children: n.label
            })
          ]
        }, n.path);
      })
    });
  };
  $x();
  "serviceWorker" in navigator && (navigator.serviceWorker.addEventListener("message", (e) => {
    var _a2;
    ((_a2 = e.data) == null ? void 0 : _a2.type) === "SW_UPDATED" && window.location.reload();
  }), navigator.serviceWorker.getRegistration().then((e) => {
    e && (e.update(), e.addEventListener("updatefound", () => {
      const t = e.installing;
      t && t.addEventListener("statechange", () => {
        t.state === "activated" && window.location.reload();
      });
    }));
  }));
  function ry() {
    return u.jsxs("div", {
      className: "flex h-[100dvh] flex-col bg-maude-bg safe-top",
      children: [
        u.jsx("div", {
          className: "min-h-0 flex-1 overflow-hidden",
          children: u.jsx(Nv, {})
        }),
        u.jsx(ny, {})
      ]
    });
  }
  const ly = _v([
    {
      element: u.jsx(ry, {}),
      children: [
        {
          path: "/",
          element: u.jsx(Fv, {})
        },
        {
          path: "/maude",
          element: u.jsx(fx, {})
        },
        {
          path: "/maude/voice",
          element: u.jsx(jx, {})
        },
        {
          path: "/terminal",
          element: u.jsx(Cx, {})
        },
        {
          path: "/browser",
          element: u.jsx(Rx, {})
        },
        {
          path: "/messages",
          element: u.jsx(bx, {})
        },
        {
          path: "/files",
          element: u.jsx(Lx, {})
        },
        {
          path: "/collab",
          element: u.jsx(Qx, {})
        },
        {
          path: "/command-center",
          element: u.jsx(ey, {})
        },
        {
          path: "/settings",
          element: u.jsx(zx, {})
        }
      ]
    }
  ]);
  Xo.createRoot(document.getElementById("root")).render(u.jsx(Iv, {
    router: ly
  }));
})();
export {
  __tla,
  ay as c,
  Qc as g
};
