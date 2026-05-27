let gy, Qc;
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
      for (const a of l) if (a.type === "childList") for (const s of a.addedNodes) s.tagName === "LINK" && s.rel === "modulepreload" && r(s);
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
  gy = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
  Qc = function(e) {
    return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
  };
  var Kc = {
    exports: {}
  }, Za = {}, Gc = {
    exports: {}
  }, Z = {};
  var bl = Symbol.for("react.element"), Am = Symbol.for("react.portal"), Im = Symbol.for("react.fragment"), zm = Symbol.for("react.strict_mode"), Um = Symbol.for("react.profiler"), $m = Symbol.for("react.provider"), Fm = Symbol.for("react.context"), Bm = Symbol.for("react.forward_ref"), Vm = Symbol.for("react.suspense"), Wm = Symbol.for("react.memo"), Hm = Symbol.for("react.lazy"), ou = Symbol.iterator;
  function Qm(e) {
    return e === null || typeof e != "object" ? null : (e = ou && e[ou] || e["@@iterator"], typeof e == "function" ? e : null);
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
  function Pr(e, t, n) {
    this.props = e, this.context = t, this.refs = Xc, this.updater = n || Yc;
  }
  Pr.prototype.isReactComponent = {};
  Pr.prototype.setState = function(e, t) {
    if (typeof e != "object" && typeof e != "function" && e != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
    this.updater.enqueueSetState(this, e, t, "setState");
  };
  Pr.prototype.forceUpdate = function(e) {
    this.updater.enqueueForceUpdate(this, e, "forceUpdate");
  };
  function Zc() {
  }
  Zc.prototype = Pr.prototype;
  function Xo(e, t, n) {
    this.props = e, this.context = t, this.refs = Xc, this.updater = n || Yc;
  }
  var Zo = Xo.prototype = new Zc();
  Zo.constructor = Xo;
  Jc(Zo, Pr.prototype);
  Zo.isPureReactComponent = true;
  var iu = Array.isArray, qc = Object.prototype.hasOwnProperty, qo = {
    current: null
  }, ed = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function td(e, t, n) {
    var r, l = {}, a = null, s = null;
    if (t != null) for (r in t.ref !== void 0 && (s = t.ref), t.key !== void 0 && (a = "" + t.key), t) qc.call(t, r) && !ed.hasOwnProperty(r) && (l[r] = t[r]);
    var u = arguments.length - 2;
    if (u === 1) l.children = n;
    else if (1 < u) {
      for (var i = Array(u), c = 0; c < u; c++) i[c] = arguments[c + 2];
      l.children = i;
    }
    if (e && e.defaultProps) for (r in u = e.defaultProps, u) l[r] === void 0 && (l[r] = u[r]);
    return {
      $$typeof: bl,
      type: e,
      key: a,
      ref: s,
      props: l,
      _owner: qo.current
    };
  }
  function Km(e, t) {
    return {
      $$typeof: bl,
      type: e.type,
      key: t,
      ref: e.ref,
      props: e.props,
      _owner: e._owner
    };
  }
  function ei(e) {
    return typeof e == "object" && e !== null && e.$$typeof === bl;
  }
  function Gm(e) {
    var t = {
      "=": "=0",
      ":": "=2"
    };
    return "$" + e.replace(/[=:]/g, function(n) {
      return t[n];
    });
  }
  var uu = /\/+/g;
  function Ss(e, t) {
    return typeof e == "object" && e !== null && e.key != null ? Gm("" + e.key) : t.toString(36);
  }
  function ua(e, t, n, r, l) {
    var a = typeof e;
    (a === "undefined" || a === "boolean") && (e = null);
    var s = false;
    if (e === null) s = true;
    else switch (a) {
      case "string":
      case "number":
        s = true;
        break;
      case "object":
        switch (e.$$typeof) {
          case bl:
          case Am:
            s = true;
        }
    }
    if (s) return s = e, l = l(s), e = r === "" ? "." + Ss(s, 0) : r, iu(l) ? (n = "", e != null && (n = e.replace(uu, "$&/") + "/"), ua(l, t, n, "", function(c) {
      return c;
    })) : l != null && (ei(l) && (l = Km(l, n + (!l.key || s && s.key === l.key ? "" : ("" + l.key).replace(uu, "$&/") + "/") + e)), t.push(l)), 1;
    if (s = 0, r = r === "" ? "." : r + ":", iu(e)) for (var u = 0; u < e.length; u++) {
      a = e[u];
      var i = r + Ss(a, u);
      s += ua(a, t, n, i, l);
    }
    else if (i = Qm(e), typeof i == "function") for (e = i.call(e), u = 0; !(a = e.next()).done; ) a = a.value, i = r + Ss(a, u++), s += ua(a, t, n, i, l);
    else if (a === "object") throw t = String(e), Error("Objects are not valid as a React child (found: " + (t === "[object Object]" ? "object with keys {" + Object.keys(e).join(", ") + "}" : t) + "). If you meant to render a collection of children, use an array instead.");
    return s;
  }
  function Bl(e, t, n) {
    if (e == null) return e;
    var r = [], l = 0;
    return ua(e, r, "", "", function(a) {
      return t.call(n, a, l++);
    }), r;
  }
  function Ym(e) {
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
  var ot = {
    current: null
  }, ca = {
    transition: null
  }, Jm = {
    ReactCurrentDispatcher: ot,
    ReactCurrentBatchConfig: ca,
    ReactCurrentOwner: qo
  };
  function nd() {
    throw Error("act(...) is not supported in production builds of React.");
  }
  Z.Children = {
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
      if (!ei(e)) throw Error("React.Children.only expected to receive a single React element child.");
      return e;
    }
  };
  Z.Component = Pr;
  Z.Fragment = Im;
  Z.Profiler = Um;
  Z.PureComponent = Xo;
  Z.StrictMode = zm;
  Z.Suspense = Vm;
  Z.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Jm;
  Z.act = nd;
  Z.cloneElement = function(e, t, n) {
    if (e == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + e + ".");
    var r = Jc({}, e.props), l = e.key, a = e.ref, s = e._owner;
    if (t != null) {
      if (t.ref !== void 0 && (a = t.ref, s = qo.current), t.key !== void 0 && (l = "" + t.key), e.type && e.type.defaultProps) var u = e.type.defaultProps;
      for (i in t) qc.call(t, i) && !ed.hasOwnProperty(i) && (r[i] = t[i] === void 0 && u !== void 0 ? u[i] : t[i]);
    }
    var i = arguments.length - 2;
    if (i === 1) r.children = n;
    else if (1 < i) {
      u = Array(i);
      for (var c = 0; c < i; c++) u[c] = arguments[c + 2];
      r.children = u;
    }
    return {
      $$typeof: bl,
      type: e.type,
      key: l,
      ref: a,
      props: r,
      _owner: s
    };
  };
  Z.createContext = function(e) {
    return e = {
      $$typeof: Fm,
      _currentValue: e,
      _currentValue2: e,
      _threadCount: 0,
      Provider: null,
      Consumer: null,
      _defaultValue: null,
      _globalName: null
    }, e.Provider = {
      $$typeof: $m,
      _context: e
    }, e.Consumer = e;
  };
  Z.createElement = td;
  Z.createFactory = function(e) {
    var t = td.bind(null, e);
    return t.type = e, t;
  };
  Z.createRef = function() {
    return {
      current: null
    };
  };
  Z.forwardRef = function(e) {
    return {
      $$typeof: Bm,
      render: e
    };
  };
  Z.isValidElement = ei;
  Z.lazy = function(e) {
    return {
      $$typeof: Hm,
      _payload: {
        _status: -1,
        _result: e
      },
      _init: Ym
    };
  };
  Z.memo = function(e, t) {
    return {
      $$typeof: Wm,
      type: e,
      compare: t === void 0 ? null : t
    };
  };
  Z.startTransition = function(e) {
    var t = ca.transition;
    ca.transition = {};
    try {
      e();
    } finally {
      ca.transition = t;
    }
  };
  Z.unstable_act = nd;
  Z.useCallback = function(e, t) {
    return ot.current.useCallback(e, t);
  };
  Z.useContext = function(e) {
    return ot.current.useContext(e);
  };
  Z.useDebugValue = function() {
  };
  Z.useDeferredValue = function(e) {
    return ot.current.useDeferredValue(e);
  };
  Z.useEffect = function(e, t) {
    return ot.current.useEffect(e, t);
  };
  Z.useId = function() {
    return ot.current.useId();
  };
  Z.useImperativeHandle = function(e, t, n) {
    return ot.current.useImperativeHandle(e, t, n);
  };
  Z.useInsertionEffect = function(e, t) {
    return ot.current.useInsertionEffect(e, t);
  };
  Z.useLayoutEffect = function(e, t) {
    return ot.current.useLayoutEffect(e, t);
  };
  Z.useMemo = function(e, t) {
    return ot.current.useMemo(e, t);
  };
  Z.useReducer = function(e, t, n) {
    return ot.current.useReducer(e, t, n);
  };
  Z.useRef = function(e) {
    return ot.current.useRef(e);
  };
  Z.useState = function(e) {
    return ot.current.useState(e);
  };
  Z.useSyncExternalStore = function(e, t, n) {
    return ot.current.useSyncExternalStore(e, t, n);
  };
  Z.useTransition = function() {
    return ot.current.useTransition();
  };
  Z.version = "18.3.1";
  Gc.exports = Z;
  var g = Gc.exports;
  const Xm = Qc(g), Zm = Hc({
    __proto__: null,
    default: Xm
  }, [
    g
  ]);
  var qm = g, ep = Symbol.for("react.element"), tp = Symbol.for("react.fragment"), np = Object.prototype.hasOwnProperty, rp = qm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, lp = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function rd(e, t, n) {
    var r, l = {}, a = null, s = null;
    n !== void 0 && (a = "" + n), t.key !== void 0 && (a = "" + t.key), t.ref !== void 0 && (s = t.ref);
    for (r in t) np.call(t, r) && !lp.hasOwnProperty(r) && (l[r] = t[r]);
    if (e && e.defaultProps) for (r in t = e.defaultProps, t) l[r] === void 0 && (l[r] = t[r]);
    return {
      $$typeof: ep,
      type: e,
      key: a,
      ref: s,
      props: l,
      _owner: rp.current
    };
  }
  Za.Fragment = tp;
  Za.jsx = rd;
  Za.jsxs = rd;
  Kc.exports = Za;
  var o = Kc.exports, Zs = {}, ld = {
    exports: {}
  }, Nt = {}, ad = {
    exports: {}
  }, sd = {};
  (function(e) {
    function t(M, W) {
      var H = M.length;
      M.push(W);
      e: for (; 0 < H; ) {
        var Y = H - 1 >>> 1, X = M[Y];
        if (0 < l(X, W)) M[Y] = W, M[H] = X, H = Y;
        else break e;
      }
    }
    function n(M) {
      return M.length === 0 ? null : M[0];
    }
    function r(M) {
      if (M.length === 0) return null;
      var W = M[0], H = M.pop();
      if (H !== W) {
        M[0] = H;
        e: for (var Y = 0, X = M.length, Fe = X >>> 1; Y < Fe; ) {
          var ve = 2 * (Y + 1) - 1, fe = M[ve], Te = ve + 1, Oe = M[Te];
          if (0 > l(fe, H)) Te < X && 0 > l(Oe, fe) ? (M[Y] = Oe, M[Te] = H, Y = Te) : (M[Y] = fe, M[ve] = H, Y = ve);
          else if (Te < X && 0 > l(Oe, H)) M[Y] = Oe, M[Te] = H, Y = Te;
          else break e;
        }
      }
      return W;
    }
    function l(M, W) {
      var H = M.sortIndex - W.sortIndex;
      return H !== 0 ? H : M.id - W.id;
    }
    if (typeof performance == "object" && typeof performance.now == "function") {
      var a = performance;
      e.unstable_now = function() {
        return a.now();
      };
    } else {
      var s = Date, u = s.now();
      e.unstable_now = function() {
        return s.now() - u;
      };
    }
    var i = [], c = [], m = 1, d = null, p = 3, S = false, w = false, y = false, b = typeof setTimeout == "function" ? setTimeout : null, h = typeof clearTimeout == "function" ? clearTimeout : null, f = typeof setImmediate < "u" ? setImmediate : null;
    typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
    function v(M) {
      for (var W = n(c); W !== null; ) {
        if (W.callback === null) r(c);
        else if (W.startTime <= M) r(c), W.sortIndex = W.expirationTime, t(i, W);
        else break;
        W = n(c);
      }
    }
    function E(M) {
      if (y = false, v(M), !w) if (n(i) !== null) w = true, lt(_);
      else {
        var W = n(c);
        W !== null && Qe(E, W.startTime - M);
      }
    }
    function _(M, W) {
      w = false, y && (y = false, h(j), j = -1), S = true;
      var H = p;
      try {
        for (v(W), d = n(i); d !== null && (!(d.expirationTime > W) || M && !Q()); ) {
          var Y = d.callback;
          if (typeof Y == "function") {
            d.callback = null, p = d.priorityLevel;
            var X = Y(d.expirationTime <= W);
            W = e.unstable_now(), typeof X == "function" ? d.callback = X : d === n(i) && r(i), v(W);
          } else r(i);
          d = n(i);
        }
        if (d !== null) var Fe = true;
        else {
          var ve = n(c);
          ve !== null && Qe(E, ve.startTime - W), Fe = false;
        }
        return Fe;
      } finally {
        d = null, p = H, S = false;
      }
    }
    var R = false, k = null, j = -1, I = 5, D = -1;
    function Q() {
      return !(e.unstable_now() - D < I);
    }
    function K() {
      if (k !== null) {
        var M = e.unstable_now();
        D = M;
        var W = true;
        try {
          W = k(true, M);
        } finally {
          W ? ae() : (R = false, k = null);
        }
      } else R = false;
    }
    var ae;
    if (typeof f == "function") ae = function() {
      f(K);
    };
    else if (typeof MessageChannel < "u") {
      var de = new MessageChannel(), ge = de.port2;
      de.port1.onmessage = K, ae = function() {
        ge.postMessage(null);
      };
    } else ae = function() {
      b(K, 0);
    };
    function lt(M) {
      k = M, R || (R = true, ae());
    }
    function Qe(M, W) {
      j = b(function() {
        M(e.unstable_now());
      }, W);
    }
    e.unstable_IdlePriority = 5, e.unstable_ImmediatePriority = 1, e.unstable_LowPriority = 4, e.unstable_NormalPriority = 3, e.unstable_Profiling = null, e.unstable_UserBlockingPriority = 2, e.unstable_cancelCallback = function(M) {
      M.callback = null;
    }, e.unstable_continueExecution = function() {
      w || S || (w = true, lt(_));
    }, e.unstable_forceFrameRate = function(M) {
      0 > M || 125 < M ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : I = 0 < M ? Math.floor(1e3 / M) : 5;
    }, e.unstable_getCurrentPriorityLevel = function() {
      return p;
    }, e.unstable_getFirstCallbackNode = function() {
      return n(i);
    }, e.unstable_next = function(M) {
      switch (p) {
        case 1:
        case 2:
        case 3:
          var W = 3;
          break;
        default:
          W = p;
      }
      var H = p;
      p = W;
      try {
        return M();
      } finally {
        p = H;
      }
    }, e.unstable_pauseExecution = function() {
    }, e.unstable_requestPaint = function() {
    }, e.unstable_runWithPriority = function(M, W) {
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
      var H = p;
      p = M;
      try {
        return W();
      } finally {
        p = H;
      }
    }, e.unstable_scheduleCallback = function(M, W, H) {
      var Y = e.unstable_now();
      switch (typeof H == "object" && H !== null ? (H = H.delay, H = typeof H == "number" && 0 < H ? Y + H : Y) : H = Y, M) {
        case 1:
          var X = -1;
          break;
        case 2:
          X = 250;
          break;
        case 5:
          X = 1073741823;
          break;
        case 4:
          X = 1e4;
          break;
        default:
          X = 5e3;
      }
      return X = H + X, M = {
        id: m++,
        callback: W,
        priorityLevel: M,
        startTime: H,
        expirationTime: X,
        sortIndex: -1
      }, H > Y ? (M.sortIndex = H, t(c, M), n(i) === null && M === n(c) && (y ? (h(j), j = -1) : y = true, Qe(E, H - Y))) : (M.sortIndex = X, t(i, M), w || S || (w = true, lt(_))), M;
    }, e.unstable_shouldYield = Q, e.unstable_wrapCallback = function(M) {
      var W = p;
      return function() {
        var H = p;
        p = W;
        try {
          return M.apply(this, arguments);
        } finally {
          p = H;
        }
      };
    };
  })(sd);
  ad.exports = sd;
  var ap = ad.exports;
  var sp = g, kt = ap;
  function P(e) {
    for (var t = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, n = 1; n < arguments.length; n++) t += "&args[]=" + encodeURIComponent(arguments[n]);
    return "Minified React error #" + e + "; visit " + t + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  var od = /* @__PURE__ */ new Set(), ul = {};
  function Xn(e, t) {
    jr(e, t), jr(e + "Capture", t);
  }
  function jr(e, t) {
    for (ul[e] = t, e = 0; e < t.length; e++) od.add(t[e]);
  }
  var tn = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), qs = Object.prototype.hasOwnProperty, op = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, cu = {}, du = {};
  function ip(e) {
    return qs.call(du, e) ? true : qs.call(cu, e) ? false : op.test(e) ? du[e] = true : (cu[e] = true, false);
  }
  function up(e, t, n, r) {
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
  function cp(e, t, n, r) {
    if (t === null || typeof t > "u" || up(e, t, n, r)) return true;
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
  function it(e, t, n, r, l, a, s) {
    this.acceptsBooleans = t === 2 || t === 3 || t === 4, this.attributeName = r, this.attributeNamespace = l, this.mustUseProperty = n, this.propertyName = e, this.type = t, this.sanitizeURL = a, this.removeEmptyString = s;
  }
  var Xe = {};
  "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e) {
    Xe[e] = new it(e, 0, false, e, null, false, false);
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
    Xe[t] = new it(t, 1, false, e[1], null, false, false);
  });
  [
    "contentEditable",
    "draggable",
    "spellCheck",
    "value"
  ].forEach(function(e) {
    Xe[e] = new it(e, 2, false, e.toLowerCase(), null, false, false);
  });
  [
    "autoReverse",
    "externalResourcesRequired",
    "focusable",
    "preserveAlpha"
  ].forEach(function(e) {
    Xe[e] = new it(e, 2, false, e, null, false, false);
  });
  "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e) {
    Xe[e] = new it(e, 3, false, e.toLowerCase(), null, false, false);
  });
  [
    "checked",
    "multiple",
    "muted",
    "selected"
  ].forEach(function(e) {
    Xe[e] = new it(e, 3, true, e, null, false, false);
  });
  [
    "capture",
    "download"
  ].forEach(function(e) {
    Xe[e] = new it(e, 4, false, e, null, false, false);
  });
  [
    "cols",
    "rows",
    "size",
    "span"
  ].forEach(function(e) {
    Xe[e] = new it(e, 6, false, e, null, false, false);
  });
  [
    "rowSpan",
    "start"
  ].forEach(function(e) {
    Xe[e] = new it(e, 5, false, e.toLowerCase(), null, false, false);
  });
  var ti = /[\-:]([a-z])/g;
  function ni(e) {
    return e[1].toUpperCase();
  }
  "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e) {
    var t = e.replace(ti, ni);
    Xe[t] = new it(t, 1, false, e, null, false, false);
  });
  "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e) {
    var t = e.replace(ti, ni);
    Xe[t] = new it(t, 1, false, e, "http://www.w3.org/1999/xlink", false, false);
  });
  [
    "xml:base",
    "xml:lang",
    "xml:space"
  ].forEach(function(e) {
    var t = e.replace(ti, ni);
    Xe[t] = new it(t, 1, false, e, "http://www.w3.org/XML/1998/namespace", false, false);
  });
  [
    "tabIndex",
    "crossOrigin"
  ].forEach(function(e) {
    Xe[e] = new it(e, 1, false, e.toLowerCase(), null, false, false);
  });
  Xe.xlinkHref = new it("xlinkHref", 1, false, "xlink:href", "http://www.w3.org/1999/xlink", true, false);
  [
    "src",
    "href",
    "action",
    "formAction"
  ].forEach(function(e) {
    Xe[e] = new it(e, 1, false, e.toLowerCase(), null, true, true);
  });
  function ri(e, t, n, r) {
    var l = Xe.hasOwnProperty(t) ? Xe[t] : null;
    (l !== null ? l.type !== 0 : r || !(2 < t.length) || t[0] !== "o" && t[0] !== "O" || t[1] !== "n" && t[1] !== "N") && (cp(t, n, l, r) && (n = null), r || l === null ? ip(t) && (n === null ? e.removeAttribute(t) : e.setAttribute(t, "" + n)) : l.mustUseProperty ? e[l.propertyName] = n === null ? l.type === 3 ? false : "" : n : (t = l.attributeName, r = l.attributeNamespace, n === null ? e.removeAttribute(t) : (l = l.type, n = l === 3 || l === 4 && n === true ? "" : "" + n, r ? e.setAttributeNS(r, t, n) : e.setAttribute(t, n))));
  }
  var an = sp.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, Vl = Symbol.for("react.element"), sr = Symbol.for("react.portal"), or = Symbol.for("react.fragment"), li = Symbol.for("react.strict_mode"), eo = Symbol.for("react.profiler"), id = Symbol.for("react.provider"), ud = Symbol.for("react.context"), ai = Symbol.for("react.forward_ref"), to = Symbol.for("react.suspense"), no = Symbol.for("react.suspense_list"), si = Symbol.for("react.memo"), cn = Symbol.for("react.lazy"), cd = Symbol.for("react.offscreen"), fu = Symbol.iterator;
  function Ir(e) {
    return e === null || typeof e != "object" ? null : (e = fu && e[fu] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var Re = Object.assign, ks;
  function Gr(e) {
    if (ks === void 0) try {
      throw Error();
    } catch (n) {
      var t = n.stack.trim().match(/\n( *(at )?)/);
      ks = t && t[1] || "";
    }
    return `
` + ks + e;
  }
  var Ns = false;
  function js(e, t) {
    if (!e || Ns) return "";
    Ns = true;
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
`), s = l.length - 1, u = a.length - 1; 1 <= s && 0 <= u && l[s] !== a[u]; ) u--;
        for (; 1 <= s && 0 <= u; s--, u--) if (l[s] !== a[u]) {
          if (s !== 1 || u !== 1) do
            if (s--, u--, 0 > u || l[s] !== a[u]) {
              var i = `
` + l[s].replace(" at new ", " at ");
              return e.displayName && i.includes("<anonymous>") && (i = i.replace("<anonymous>", e.displayName)), i;
            }
          while (1 <= s && 0 <= u);
          break;
        }
      }
    } finally {
      Ns = false, Error.prepareStackTrace = n;
    }
    return (e = e ? e.displayName || e.name : "") ? Gr(e) : "";
  }
  function dp(e) {
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
        return e = js(e.type, false), e;
      case 11:
        return e = js(e.type.render, false), e;
      case 1:
        return e = js(e.type, true), e;
      default:
        return "";
    }
  }
  function ro(e) {
    if (e == null) return null;
    if (typeof e == "function") return e.displayName || e.name || null;
    if (typeof e == "string") return e;
    switch (e) {
      case or:
        return "Fragment";
      case sr:
        return "Portal";
      case eo:
        return "Profiler";
      case li:
        return "StrictMode";
      case to:
        return "Suspense";
      case no:
        return "SuspenseList";
    }
    if (typeof e == "object") switch (e.$$typeof) {
      case ud:
        return (e.displayName || "Context") + ".Consumer";
      case id:
        return (e._context.displayName || "Context") + ".Provider";
      case ai:
        var t = e.render;
        return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
      case si:
        return t = e.displayName || null, t !== null ? t : ro(e.type) || "Memo";
      case cn:
        t = e._payload, e = e._init;
        try {
          return ro(e(t));
        } catch {
        }
    }
    return null;
  }
  function fp(e) {
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
        return ro(t);
      case 8:
        return t === li ? "StrictMode" : "Mode";
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
  function Cn(e) {
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
  function mp(e) {
    var t = dd(e) ? "checked" : "value", n = Object.getOwnPropertyDescriptor(e.constructor.prototype, t), r = "" + e[t];
    if (!e.hasOwnProperty(t) && typeof n < "u" && typeof n.get == "function" && typeof n.set == "function") {
      var l = n.get, a = n.set;
      return Object.defineProperty(e, t, {
        configurable: true,
        get: function() {
          return l.call(this);
        },
        set: function(s) {
          r = "" + s, a.call(this, s);
        }
      }), Object.defineProperty(e, t, {
        enumerable: n.enumerable
      }), {
        getValue: function() {
          return r;
        },
        setValue: function(s) {
          r = "" + s;
        },
        stopTracking: function() {
          e._valueTracker = null, delete e[t];
        }
      };
    }
  }
  function Wl(e) {
    e._valueTracker || (e._valueTracker = mp(e));
  }
  function fd(e) {
    if (!e) return false;
    var t = e._valueTracker;
    if (!t) return true;
    var n = t.getValue(), r = "";
    return e && (r = dd(e) ? e.checked ? "true" : "false" : e.value), e = r, e !== n ? (t.setValue(e), true) : false;
  }
  function ja(e) {
    if (e = e || (typeof document < "u" ? document : void 0), typeof e > "u") return null;
    try {
      return e.activeElement || e.body;
    } catch {
      return e.body;
    }
  }
  function lo(e, t) {
    var n = t.checked;
    return Re({}, t, {
      defaultChecked: void 0,
      defaultValue: void 0,
      value: void 0,
      checked: n ?? e._wrapperState.initialChecked
    });
  }
  function mu(e, t) {
    var n = t.defaultValue == null ? "" : t.defaultValue, r = t.checked != null ? t.checked : t.defaultChecked;
    n = Cn(t.value != null ? t.value : n), e._wrapperState = {
      initialChecked: r,
      initialValue: n,
      controlled: t.type === "checkbox" || t.type === "radio" ? t.checked != null : t.value != null
    };
  }
  function md(e, t) {
    t = t.checked, t != null && ri(e, "checked", t, false);
  }
  function ao(e, t) {
    md(e, t);
    var n = Cn(t.value), r = t.type;
    if (n != null) r === "number" ? (n === 0 && e.value === "" || e.value != n) && (e.value = "" + n) : e.value !== "" + n && (e.value = "" + n);
    else if (r === "submit" || r === "reset") {
      e.removeAttribute("value");
      return;
    }
    t.hasOwnProperty("value") ? so(e, t.type, n) : t.hasOwnProperty("defaultValue") && so(e, t.type, Cn(t.defaultValue)), t.checked == null && t.defaultChecked != null && (e.defaultChecked = !!t.defaultChecked);
  }
  function pu(e, t, n) {
    if (t.hasOwnProperty("value") || t.hasOwnProperty("defaultValue")) {
      var r = t.type;
      if (!(r !== "submit" && r !== "reset" || t.value !== void 0 && t.value !== null)) return;
      t = "" + e._wrapperState.initialValue, n || t === e.value || (e.value = t), e.defaultValue = t;
    }
    n = e.name, n !== "" && (e.name = ""), e.defaultChecked = !!e._wrapperState.initialChecked, n !== "" && (e.name = n);
  }
  function so(e, t, n) {
    (t !== "number" || ja(e.ownerDocument) !== e) && (n == null ? e.defaultValue = "" + e._wrapperState.initialValue : e.defaultValue !== "" + n && (e.defaultValue = "" + n));
  }
  var Yr = Array.isArray;
  function xr(e, t, n, r) {
    if (e = e.options, t) {
      t = {};
      for (var l = 0; l < n.length; l++) t["$" + n[l]] = true;
      for (n = 0; n < e.length; n++) l = t.hasOwnProperty("$" + e[n].value), e[n].selected !== l && (e[n].selected = l), l && r && (e[n].defaultSelected = true);
    } else {
      for (n = "" + Cn(n), t = null, l = 0; l < e.length; l++) {
        if (e[l].value === n) {
          e[l].selected = true, r && (e[l].defaultSelected = true);
          return;
        }
        t !== null || e[l].disabled || (t = e[l]);
      }
      t !== null && (t.selected = true);
    }
  }
  function oo(e, t) {
    if (t.dangerouslySetInnerHTML != null) throw Error(P(91));
    return Re({}, t, {
      value: void 0,
      defaultValue: void 0,
      children: "" + e._wrapperState.initialValue
    });
  }
  function hu(e, t) {
    var n = t.value;
    if (n == null) {
      if (n = t.children, t = t.defaultValue, n != null) {
        if (t != null) throw Error(P(92));
        if (Yr(n)) {
          if (1 < n.length) throw Error(P(93));
          n = n[0];
        }
        t = n;
      }
      t == null && (t = ""), n = t;
    }
    e._wrapperState = {
      initialValue: Cn(n)
    };
  }
  function pd(e, t) {
    var n = Cn(t.value), r = Cn(t.defaultValue);
    n != null && (n = "" + n, n !== e.value && (e.value = n), t.defaultValue == null && e.defaultValue !== n && (e.defaultValue = n)), r != null && (e.defaultValue = "" + r);
  }
  function gu(e) {
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
  function io(e, t) {
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
  }, pp = [
    "Webkit",
    "ms",
    "Moz",
    "O"
  ];
  Object.keys(qr).forEach(function(e) {
    pp.forEach(function(t) {
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
  var hp = Re({
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
  function uo(e, t) {
    if (t) {
      if (hp[e] && (t.children != null || t.dangerouslySetInnerHTML != null)) throw Error(P(137, e));
      if (t.dangerouslySetInnerHTML != null) {
        if (t.children != null) throw Error(P(60));
        if (typeof t.dangerouslySetInnerHTML != "object" || !("__html" in t.dangerouslySetInnerHTML)) throw Error(P(61));
      }
      if (t.style != null && typeof t.style != "object") throw Error(P(62));
    }
  }
  function co(e, t) {
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
  var fo = null;
  function oi(e) {
    return e = e.target || e.srcElement || window, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
  }
  var mo = null, yr = null, wr = null;
  function vu(e) {
    if (e = Pl(e)) {
      if (typeof mo != "function") throw Error(P(280));
      var t = e.stateNode;
      t && (t = rs(t), mo(e.stateNode, e.type, t));
    }
  }
  function yd(e) {
    yr ? wr ? wr.push(e) : wr = [
      e
    ] : yr = e;
  }
  function wd() {
    if (yr) {
      var e = yr, t = wr;
      if (wr = yr = null, vu(e), t) for (e = 0; e < t.length; e++) vu(t[e]);
    }
  }
  function Sd(e, t) {
    return e(t);
  }
  function kd() {
  }
  var Es = false;
  function Nd(e, t, n) {
    if (Es) return e(t, n);
    Es = true;
    try {
      return Sd(e, t, n);
    } finally {
      Es = false, (yr !== null || wr !== null) && (kd(), wd());
    }
  }
  function dl(e, t) {
    var n = e.stateNode;
    if (n === null) return null;
    var r = rs(n);
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
    if (n && typeof n != "function") throw Error(P(231, t, typeof n));
    return n;
  }
  var po = false;
  if (tn) try {
    var zr = {};
    Object.defineProperty(zr, "passive", {
      get: function() {
        po = true;
      }
    }), window.addEventListener("test", zr, zr), window.removeEventListener("test", zr, zr);
  } catch {
    po = false;
  }
  function gp(e, t, n, r, l, a, s, u, i) {
    var c = Array.prototype.slice.call(arguments, 3);
    try {
      t.apply(n, c);
    } catch (m) {
      this.onError(m);
    }
  }
  var el = false, Ea = null, Ca = false, ho = null, vp = {
    onError: function(e) {
      el = true, Ea = e;
    }
  };
  function xp(e, t, n, r, l, a, s, u, i) {
    el = false, Ea = null, gp.apply(vp, arguments);
  }
  function yp(e, t, n, r, l, a, s, u, i) {
    if (xp.apply(this, arguments), el) {
      if (el) {
        var c = Ea;
        el = false, Ea = null;
      } else throw Error(P(198));
      Ca || (Ca = true, ho = c);
    }
  }
  function Zn(e) {
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
  function xu(e) {
    if (Zn(e) !== e) throw Error(P(188));
  }
  function wp(e) {
    var t = e.alternate;
    if (!t) {
      if (t = Zn(e), t === null) throw Error(P(188));
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
          if (a === n) return xu(l), e;
          if (a === r) return xu(l), t;
          a = a.sibling;
        }
        throw Error(P(188));
      }
      if (n.return !== r.return) n = l, r = a;
      else {
        for (var s = false, u = l.child; u; ) {
          if (u === n) {
            s = true, n = l, r = a;
            break;
          }
          if (u === r) {
            s = true, r = l, n = a;
            break;
          }
          u = u.sibling;
        }
        if (!s) {
          for (u = a.child; u; ) {
            if (u === n) {
              s = true, n = a, r = l;
              break;
            }
            if (u === r) {
              s = true, r = a, n = l;
              break;
            }
            u = u.sibling;
          }
          if (!s) throw Error(P(189));
        }
      }
      if (n.alternate !== r) throw Error(P(190));
    }
    if (n.tag !== 3) throw Error(P(188));
    return n.stateNode.current === n ? e : t;
  }
  function Ed(e) {
    return e = wp(e), e !== null ? Cd(e) : null;
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
  var _d = kt.unstable_scheduleCallback, yu = kt.unstable_cancelCallback, Sp = kt.unstable_shouldYield, kp = kt.unstable_requestPaint, Le = kt.unstable_now, Np = kt.unstable_getCurrentPriorityLevel, ii = kt.unstable_ImmediatePriority, bd = kt.unstable_UserBlockingPriority, _a = kt.unstable_NormalPriority, jp = kt.unstable_LowPriority, Rd = kt.unstable_IdlePriority, qa = null, Qt = null;
  function Ep(e) {
    if (Qt && typeof Qt.onCommitFiberRoot == "function") try {
      Qt.onCommitFiberRoot(qa, e, void 0, (e.current.flags & 128) === 128);
    } catch {
    }
  }
  var Ut = Math.clz32 ? Math.clz32 : bp, Cp = Math.log, _p = Math.LN2;
  function bp(e) {
    return e >>>= 0, e === 0 ? 32 : 31 - (Cp(e) / _p | 0) | 0;
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
  function ba(e, t) {
    var n = e.pendingLanes;
    if (n === 0) return 0;
    var r = 0, l = e.suspendedLanes, a = e.pingedLanes, s = n & 268435455;
    if (s !== 0) {
      var u = s & ~l;
      u !== 0 ? r = Jr(u) : (a &= s, a !== 0 && (r = Jr(a)));
    } else s = n & ~l, s !== 0 ? r = Jr(s) : a !== 0 && (r = Jr(a));
    if (r === 0) return 0;
    if (t !== 0 && t !== r && !(t & l) && (l = r & -r, a = t & -t, l >= a || l === 16 && (a & 4194240) !== 0)) return t;
    if (r & 4 && (r |= n & 16), t = e.entangledLanes, t !== 0) for (e = e.entanglements, t &= r; 0 < t; ) n = 31 - Ut(t), l = 1 << n, r |= e[n], t &= ~l;
    return r;
  }
  function Rp(e, t) {
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
  function Tp(e, t) {
    for (var n = e.suspendedLanes, r = e.pingedLanes, l = e.expirationTimes, a = e.pendingLanes; 0 < a; ) {
      var s = 31 - Ut(a), u = 1 << s, i = l[s];
      i === -1 ? (!(u & n) || u & r) && (l[s] = Rp(u, t)) : i <= t && (e.expiredLanes |= u), a &= ~u;
    }
  }
  function go(e) {
    return e = e.pendingLanes & -1073741825, e !== 0 ? e : e & 1073741824 ? 1073741824 : 0;
  }
  function Td() {
    var e = Ql;
    return Ql <<= 1, !(Ql & 4194240) && (Ql = 64), e;
  }
  function Cs(e) {
    for (var t = [], n = 0; 31 > n; n++) t.push(e);
    return t;
  }
  function Rl(e, t, n) {
    e.pendingLanes |= t, t !== 536870912 && (e.suspendedLanes = 0, e.pingedLanes = 0), e = e.eventTimes, t = 31 - Ut(t), e[t] = n;
  }
  function Pp(e, t) {
    var n = e.pendingLanes & ~t;
    e.pendingLanes = t, e.suspendedLanes = 0, e.pingedLanes = 0, e.expiredLanes &= t, e.mutableReadLanes &= t, e.entangledLanes &= t, t = e.entanglements;
    var r = e.eventTimes;
    for (e = e.expirationTimes; 0 < n; ) {
      var l = 31 - Ut(n), a = 1 << l;
      t[l] = 0, r[l] = -1, e[l] = -1, n &= ~a;
    }
  }
  function ui(e, t) {
    var n = e.entangledLanes |= t;
    for (e = e.entanglements; n; ) {
      var r = 31 - Ut(n), l = 1 << r;
      l & t | e[r] & t && (e[r] |= t), n &= ~l;
    }
  }
  var ce = 0;
  function Pd(e) {
    return e &= -e, 1 < e ? 4 < e ? e & 268435455 ? 16 : 536870912 : 4 : 1;
  }
  var Md, ci, Dd, Ld, Od, vo = false, Gl = [], vn = null, xn = null, yn = null, fl = /* @__PURE__ */ new Map(), ml = /* @__PURE__ */ new Map(), fn = [], Mp = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
  function wu(e, t) {
    switch (e) {
      case "focusin":
      case "focusout":
        vn = null;
        break;
      case "dragenter":
      case "dragleave":
        xn = null;
        break;
      case "mouseover":
      case "mouseout":
        yn = null;
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
  function Ur(e, t, n, r, l, a) {
    return e === null || e.nativeEvent !== a ? (e = {
      blockedOn: t,
      domEventName: n,
      eventSystemFlags: r,
      nativeEvent: a,
      targetContainers: [
        l
      ]
    }, t !== null && (t = Pl(t), t !== null && ci(t)), e) : (e.eventSystemFlags |= r, t = e.targetContainers, l !== null && t.indexOf(l) === -1 && t.push(l), e);
  }
  function Dp(e, t, n, r, l) {
    switch (t) {
      case "focusin":
        return vn = Ur(vn, e, t, n, r, l), true;
      case "dragenter":
        return xn = Ur(xn, e, t, n, r, l), true;
      case "mouseover":
        return yn = Ur(yn, e, t, n, r, l), true;
      case "pointerover":
        var a = l.pointerId;
        return fl.set(a, Ur(fl.get(a) || null, e, t, n, r, l)), true;
      case "gotpointercapture":
        return a = l.pointerId, ml.set(a, Ur(ml.get(a) || null, e, t, n, r, l)), true;
    }
    return false;
  }
  function Ad(e) {
    var t = Un(e.target);
    if (t !== null) {
      var n = Zn(t);
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
  function da(e) {
    if (e.blockedOn !== null) return false;
    for (var t = e.targetContainers; 0 < t.length; ) {
      var n = xo(e.domEventName, e.eventSystemFlags, t[0], e.nativeEvent);
      if (n === null) {
        n = e.nativeEvent;
        var r = new n.constructor(n.type, n);
        fo = r, n.target.dispatchEvent(r), fo = null;
      } else return t = Pl(n), t !== null && ci(t), e.blockedOn = n, false;
      t.shift();
    }
    return true;
  }
  function Su(e, t, n) {
    da(e) && n.delete(t);
  }
  function Lp() {
    vo = false, vn !== null && da(vn) && (vn = null), xn !== null && da(xn) && (xn = null), yn !== null && da(yn) && (yn = null), fl.forEach(Su), ml.forEach(Su);
  }
  function $r(e, t) {
    e.blockedOn === t && (e.blockedOn = null, vo || (vo = true, kt.unstable_scheduleCallback(kt.unstable_NormalPriority, Lp)));
  }
  function pl(e) {
    function t(l) {
      return $r(l, e);
    }
    if (0 < Gl.length) {
      $r(Gl[0], e);
      for (var n = 1; n < Gl.length; n++) {
        var r = Gl[n];
        r.blockedOn === e && (r.blockedOn = null);
      }
    }
    for (vn !== null && $r(vn, e), xn !== null && $r(xn, e), yn !== null && $r(yn, e), fl.forEach(t), ml.forEach(t), n = 0; n < fn.length; n++) r = fn[n], r.blockedOn === e && (r.blockedOn = null);
    for (; 0 < fn.length && (n = fn[0], n.blockedOn === null); ) Ad(n), n.blockedOn === null && fn.shift();
  }
  var Sr = an.ReactCurrentBatchConfig, Ra = true;
  function Op(e, t, n, r) {
    var l = ce, a = Sr.transition;
    Sr.transition = null;
    try {
      ce = 1, di(e, t, n, r);
    } finally {
      ce = l, Sr.transition = a;
    }
  }
  function Ap(e, t, n, r) {
    var l = ce, a = Sr.transition;
    Sr.transition = null;
    try {
      ce = 4, di(e, t, n, r);
    } finally {
      ce = l, Sr.transition = a;
    }
  }
  function di(e, t, n, r) {
    if (Ra) {
      var l = xo(e, t, n, r);
      if (l === null) As(e, t, r, Ta, n), wu(e, r);
      else if (Dp(l, e, t, n, r)) r.stopPropagation();
      else if (wu(e, r), t & 4 && -1 < Mp.indexOf(e)) {
        for (; l !== null; ) {
          var a = Pl(l);
          if (a !== null && Md(a), a = xo(e, t, n, r), a === null && As(e, t, r, Ta, n), a === l) break;
          l = a;
        }
        l !== null && r.stopPropagation();
      } else As(e, t, r, null, n);
    }
  }
  var Ta = null;
  function xo(e, t, n, r) {
    if (Ta = null, e = oi(r), e = Un(e), e !== null) if (t = Zn(e), t === null) e = null;
    else if (n = t.tag, n === 13) {
      if (e = jd(t), e !== null) return e;
      e = null;
    } else if (n === 3) {
      if (t.stateNode.current.memoizedState.isDehydrated) return t.tag === 3 ? t.stateNode.containerInfo : null;
      e = null;
    } else t !== e && (e = null);
    return Ta = e, null;
  }
  function Id(e) {
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
        switch (Np()) {
          case ii:
            return 1;
          case bd:
            return 4;
          case _a:
          case jp:
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
  var pn = null, fi = null, fa = null;
  function zd() {
    if (fa) return fa;
    var e, t = fi, n = t.length, r, l = "value" in pn ? pn.value : pn.textContent, a = l.length;
    for (e = 0; e < n && t[e] === l[e]; e++) ;
    var s = n - e;
    for (r = 1; r <= s && t[n - r] === l[a - r]; r++) ;
    return fa = l.slice(e, 1 < r ? 1 - r : void 0);
  }
  function ma(e) {
    var t = e.keyCode;
    return "charCode" in e ? (e = e.charCode, e === 0 && t === 13 && (e = 13)) : e = t, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
  }
  function Yl() {
    return true;
  }
  function ku() {
    return false;
  }
  function jt(e) {
    function t(n, r, l, a, s) {
      this._reactName = n, this._targetInst = l, this.type = r, this.nativeEvent = a, this.target = s, this.currentTarget = null;
      for (var u in e) e.hasOwnProperty(u) && (n = e[u], this[u] = n ? n(a) : a[u]);
      return this.isDefaultPrevented = (a.defaultPrevented != null ? a.defaultPrevented : a.returnValue === false) ? Yl : ku, this.isPropagationStopped = ku, this;
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
  var Mr = {
    eventPhase: 0,
    bubbles: 0,
    cancelable: 0,
    timeStamp: function(e) {
      return e.timeStamp || Date.now();
    },
    defaultPrevented: 0,
    isTrusted: 0
  }, mi = jt(Mr), Tl = Re({}, Mr, {
    view: 0,
    detail: 0
  }), Ip = jt(Tl), _s, bs, Fr, es = Re({}, Tl, {
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
    getModifierState: pi,
    button: 0,
    buttons: 0,
    relatedTarget: function(e) {
      return e.relatedTarget === void 0 ? e.fromElement === e.srcElement ? e.toElement : e.fromElement : e.relatedTarget;
    },
    movementX: function(e) {
      return "movementX" in e ? e.movementX : (e !== Fr && (Fr && e.type === "mousemove" ? (_s = e.screenX - Fr.screenX, bs = e.screenY - Fr.screenY) : bs = _s = 0, Fr = e), _s);
    },
    movementY: function(e) {
      return "movementY" in e ? e.movementY : bs;
    }
  }), Nu = jt(es), zp = Re({}, es, {
    dataTransfer: 0
  }), Up = jt(zp), $p = Re({}, Tl, {
    relatedTarget: 0
  }), Rs = jt($p), Fp = Re({}, Mr, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Bp = jt(Fp), Vp = Re({}, Mr, {
    clipboardData: function(e) {
      return "clipboardData" in e ? e.clipboardData : window.clipboardData;
    }
  }), Wp = jt(Vp), Hp = Re({}, Mr, {
    data: 0
  }), ju = jt(Hp), Qp = {
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
  }, Kp = {
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
  }, Gp = {
    Alt: "altKey",
    Control: "ctrlKey",
    Meta: "metaKey",
    Shift: "shiftKey"
  };
  function Yp(e) {
    var t = this.nativeEvent;
    return t.getModifierState ? t.getModifierState(e) : (e = Gp[e]) ? !!t[e] : false;
  }
  function pi() {
    return Yp;
  }
  var Jp = Re({}, Tl, {
    key: function(e) {
      if (e.key) {
        var t = Qp[e.key] || e.key;
        if (t !== "Unidentified") return t;
      }
      return e.type === "keypress" ? (e = ma(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? Kp[e.keyCode] || "Unidentified" : "";
    },
    code: 0,
    location: 0,
    ctrlKey: 0,
    shiftKey: 0,
    altKey: 0,
    metaKey: 0,
    repeat: 0,
    locale: 0,
    getModifierState: pi,
    charCode: function(e) {
      return e.type === "keypress" ? ma(e) : 0;
    },
    keyCode: function(e) {
      return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
    },
    which: function(e) {
      return e.type === "keypress" ? ma(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
    }
  }), Xp = jt(Jp), Zp = Re({}, es, {
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
  }), Eu = jt(Zp), qp = Re({}, Tl, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: pi
  }), eh = jt(qp), th = Re({}, Mr, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), nh = jt(th), rh = Re({}, es, {
    deltaX: function(e) {
      return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
    },
    deltaY: function(e) {
      return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), lh = jt(rh), ah = [
    9,
    13,
    27,
    32
  ], hi = tn && "CompositionEvent" in window, tl = null;
  tn && "documentMode" in document && (tl = document.documentMode);
  var sh = tn && "TextEvent" in window && !tl, Ud = tn && (!hi || tl && 8 < tl && 11 >= tl), Cu = " ", _u = false;
  function $d(e, t) {
    switch (e) {
      case "keyup":
        return ah.indexOf(t.keyCode) !== -1;
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
  var ir = false;
  function oh(e, t) {
    switch (e) {
      case "compositionend":
        return Fd(t);
      case "keypress":
        return t.which !== 32 ? null : (_u = true, Cu);
      case "textInput":
        return e = t.data, e === Cu && _u ? null : e;
      default:
        return null;
    }
  }
  function ih(e, t) {
    if (ir) return e === "compositionend" || !hi && $d(e, t) ? (e = zd(), fa = fi = pn = null, ir = false, e) : null;
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
  var uh = {
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
  function bu(e) {
    var t = e && e.nodeName && e.nodeName.toLowerCase();
    return t === "input" ? !!uh[e.type] : t === "textarea";
  }
  function Bd(e, t, n, r) {
    yd(r), t = Pa(t, "onChange"), 0 < t.length && (n = new mi("onChange", "change", null, n, r), e.push({
      event: n,
      listeners: t
    }));
  }
  var nl = null, hl = null;
  function ch(e) {
    qd(e, 0);
  }
  function ts(e) {
    var t = dr(e);
    if (fd(t)) return e;
  }
  function dh(e, t) {
    if (e === "change") return t;
  }
  var Vd = false;
  if (tn) {
    var Ts;
    if (tn) {
      var Ps = "oninput" in document;
      if (!Ps) {
        var Ru = document.createElement("div");
        Ru.setAttribute("oninput", "return;"), Ps = typeof Ru.oninput == "function";
      }
      Ts = Ps;
    } else Ts = false;
    Vd = Ts && (!document.documentMode || 9 < document.documentMode);
  }
  function Tu() {
    nl && (nl.detachEvent("onpropertychange", Wd), hl = nl = null);
  }
  function Wd(e) {
    if (e.propertyName === "value" && ts(hl)) {
      var t = [];
      Bd(t, hl, e, oi(e)), Nd(ch, t);
    }
  }
  function fh(e, t, n) {
    e === "focusin" ? (Tu(), nl = t, hl = n, nl.attachEvent("onpropertychange", Wd)) : e === "focusout" && Tu();
  }
  function mh(e) {
    if (e === "selectionchange" || e === "keyup" || e === "keydown") return ts(hl);
  }
  function ph(e, t) {
    if (e === "click") return ts(t);
  }
  function hh(e, t) {
    if (e === "input" || e === "change") return ts(t);
  }
  function gh(e, t) {
    return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
  }
  var Ft = typeof Object.is == "function" ? Object.is : gh;
  function gl(e, t) {
    if (Ft(e, t)) return true;
    if (typeof e != "object" || e === null || typeof t != "object" || t === null) return false;
    var n = Object.keys(e), r = Object.keys(t);
    if (n.length !== r.length) return false;
    for (r = 0; r < n.length; r++) {
      var l = n[r];
      if (!qs.call(t, l) || !Ft(e[l], t[l])) return false;
    }
    return true;
  }
  function Pu(e) {
    for (; e && e.firstChild; ) e = e.firstChild;
    return e;
  }
  function Mu(e, t) {
    var n = Pu(e);
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
      n = Pu(n);
    }
  }
  function Hd(e, t) {
    return e && t ? e === t ? true : e && e.nodeType === 3 ? false : t && t.nodeType === 3 ? Hd(e, t.parentNode) : "contains" in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : false : false;
  }
  function Qd() {
    for (var e = window, t = ja(); t instanceof e.HTMLIFrameElement; ) {
      try {
        var n = typeof t.contentWindow.location.href == "string";
      } catch {
        n = false;
      }
      if (n) e = t.contentWindow;
      else break;
      t = ja(e.document);
    }
    return t;
  }
  function gi(e) {
    var t = e && e.nodeName && e.nodeName.toLowerCase();
    return t && (t === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || t === "textarea" || e.contentEditable === "true");
  }
  function vh(e) {
    var t = Qd(), n = e.focusedElem, r = e.selectionRange;
    if (t !== n && n && n.ownerDocument && Hd(n.ownerDocument.documentElement, n)) {
      if (r !== null && gi(n)) {
        if (t = r.start, e = r.end, e === void 0 && (e = t), "selectionStart" in n) n.selectionStart = t, n.selectionEnd = Math.min(e, n.value.length);
        else if (e = (t = n.ownerDocument || document) && t.defaultView || window, e.getSelection) {
          e = e.getSelection();
          var l = n.textContent.length, a = Math.min(r.start, l);
          r = r.end === void 0 ? a : Math.min(r.end, l), !e.extend && a > r && (l = r, r = a, a = l), l = Mu(n, a);
          var s = Mu(n, r);
          l && s && (e.rangeCount !== 1 || e.anchorNode !== l.node || e.anchorOffset !== l.offset || e.focusNode !== s.node || e.focusOffset !== s.offset) && (t = t.createRange(), t.setStart(l.node, l.offset), e.removeAllRanges(), a > r ? (e.addRange(t), e.extend(s.node, s.offset)) : (t.setEnd(s.node, s.offset), e.addRange(t)));
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
  var xh = tn && "documentMode" in document && 11 >= document.documentMode, ur = null, yo = null, rl = null, wo = false;
  function Du(e, t, n) {
    var r = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
    wo || ur == null || ur !== ja(r) || (r = ur, "selectionStart" in r && gi(r) ? r = {
      start: r.selectionStart,
      end: r.selectionEnd
    } : (r = (r.ownerDocument && r.ownerDocument.defaultView || window).getSelection(), r = {
      anchorNode: r.anchorNode,
      anchorOffset: r.anchorOffset,
      focusNode: r.focusNode,
      focusOffset: r.focusOffset
    }), rl && gl(rl, r) || (rl = r, r = Pa(yo, "onSelect"), 0 < r.length && (t = new mi("onSelect", "select", null, t, n), e.push({
      event: t,
      listeners: r
    }), t.target = ur)));
  }
  function Jl(e, t) {
    var n = {};
    return n[e.toLowerCase()] = t.toLowerCase(), n["Webkit" + e] = "webkit" + t, n["Moz" + e] = "moz" + t, n;
  }
  var cr = {
    animationend: Jl("Animation", "AnimationEnd"),
    animationiteration: Jl("Animation", "AnimationIteration"),
    animationstart: Jl("Animation", "AnimationStart"),
    transitionend: Jl("Transition", "TransitionEnd")
  }, Ms = {}, Kd = {};
  tn && (Kd = document.createElement("div").style, "AnimationEvent" in window || (delete cr.animationend.animation, delete cr.animationiteration.animation, delete cr.animationstart.animation), "TransitionEvent" in window || delete cr.transitionend.transition);
  function ns(e) {
    if (Ms[e]) return Ms[e];
    if (!cr[e]) return e;
    var t = cr[e], n;
    for (n in t) if (t.hasOwnProperty(n) && n in Kd) return Ms[e] = t[n];
    return e;
  }
  var Gd = ns("animationend"), Yd = ns("animationiteration"), Jd = ns("animationstart"), Xd = ns("transitionend"), Zd = /* @__PURE__ */ new Map(), Lu = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
  function bn(e, t) {
    Zd.set(e, t), Xn(t, [
      e
    ]);
  }
  for (var Ds = 0; Ds < Lu.length; Ds++) {
    var Ls = Lu[Ds], yh = Ls.toLowerCase(), wh = Ls[0].toUpperCase() + Ls.slice(1);
    bn(yh, "on" + wh);
  }
  bn(Gd, "onAnimationEnd");
  bn(Yd, "onAnimationIteration");
  bn(Jd, "onAnimationStart");
  bn("dblclick", "onDoubleClick");
  bn("focusin", "onFocus");
  bn("focusout", "onBlur");
  bn(Xd, "onTransitionEnd");
  jr("onMouseEnter", [
    "mouseout",
    "mouseover"
  ]);
  jr("onMouseLeave", [
    "mouseout",
    "mouseover"
  ]);
  jr("onPointerEnter", [
    "pointerout",
    "pointerover"
  ]);
  jr("onPointerLeave", [
    "pointerout",
    "pointerover"
  ]);
  Xn("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
  Xn("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
  Xn("onBeforeInput", [
    "compositionend",
    "keypress",
    "textInput",
    "paste"
  ]);
  Xn("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
  Xn("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
  Xn("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
  var Xr = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), Sh = new Set("cancel close invalid load scroll toggle".split(" ").concat(Xr));
  function Ou(e, t, n) {
    var r = e.type || "unknown-event";
    e.currentTarget = n, yp(r, t, void 0, e), e.currentTarget = null;
  }
  function qd(e, t) {
    t = (t & 4) !== 0;
    for (var n = 0; n < e.length; n++) {
      var r = e[n], l = r.event;
      r = r.listeners;
      e: {
        var a = void 0;
        if (t) for (var s = r.length - 1; 0 <= s; s--) {
          var u = r[s], i = u.instance, c = u.currentTarget;
          if (u = u.listener, i !== a && l.isPropagationStopped()) break e;
          Ou(l, u, c), a = i;
        }
        else for (s = 0; s < r.length; s++) {
          if (u = r[s], i = u.instance, c = u.currentTarget, u = u.listener, i !== a && l.isPropagationStopped()) break e;
          Ou(l, u, c), a = i;
        }
      }
    }
    if (Ca) throw e = ho, Ca = false, ho = null, e;
  }
  function Ne(e, t) {
    var n = t[Eo];
    n === void 0 && (n = t[Eo] = /* @__PURE__ */ new Set());
    var r = e + "__bubble";
    n.has(r) || (ef(t, e, 2, false), n.add(r));
  }
  function Os(e, t, n) {
    var r = 0;
    t && (r |= 4), ef(n, e, r, t);
  }
  var Xl = "_reactListening" + Math.random().toString(36).slice(2);
  function vl(e) {
    if (!e[Xl]) {
      e[Xl] = true, od.forEach(function(n) {
        n !== "selectionchange" && (Sh.has(n) || Os(n, false, e), Os(n, true, e));
      });
      var t = e.nodeType === 9 ? e : e.ownerDocument;
      t === null || t[Xl] || (t[Xl] = true, Os("selectionchange", false, t));
    }
  }
  function ef(e, t, n, r) {
    switch (Id(t)) {
      case 1:
        var l = Op;
        break;
      case 4:
        l = Ap;
        break;
      default:
        l = di;
    }
    n = l.bind(null, t, n, e), l = void 0, !po || t !== "touchstart" && t !== "touchmove" && t !== "wheel" || (l = true), r ? l !== void 0 ? e.addEventListener(t, n, {
      capture: true,
      passive: l
    }) : e.addEventListener(t, n, true) : l !== void 0 ? e.addEventListener(t, n, {
      passive: l
    }) : e.addEventListener(t, n, false);
  }
  function As(e, t, n, r, l) {
    var a = r;
    if (!(t & 1) && !(t & 2) && r !== null) e: for (; ; ) {
      if (r === null) return;
      var s = r.tag;
      if (s === 3 || s === 4) {
        var u = r.stateNode.containerInfo;
        if (u === l || u.nodeType === 8 && u.parentNode === l) break;
        if (s === 4) for (s = r.return; s !== null; ) {
          var i = s.tag;
          if ((i === 3 || i === 4) && (i = s.stateNode.containerInfo, i === l || i.nodeType === 8 && i.parentNode === l)) return;
          s = s.return;
        }
        for (; u !== null; ) {
          if (s = Un(u), s === null) return;
          if (i = s.tag, i === 5 || i === 6) {
            r = a = s;
            continue e;
          }
          u = u.parentNode;
        }
      }
      r = r.return;
    }
    Nd(function() {
      var c = a, m = oi(n), d = [];
      e: {
        var p = Zd.get(e);
        if (p !== void 0) {
          var S = mi, w = e;
          switch (e) {
            case "keypress":
              if (ma(n) === 0) break e;
            case "keydown":
            case "keyup":
              S = Xp;
              break;
            case "focusin":
              w = "focus", S = Rs;
              break;
            case "focusout":
              w = "blur", S = Rs;
              break;
            case "beforeblur":
            case "afterblur":
              S = Rs;
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
              S = Nu;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              S = Up;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              S = eh;
              break;
            case Gd:
            case Yd:
            case Jd:
              S = Bp;
              break;
            case Xd:
              S = nh;
              break;
            case "scroll":
              S = Ip;
              break;
            case "wheel":
              S = lh;
              break;
            case "copy":
            case "cut":
            case "paste":
              S = Wp;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              S = Eu;
          }
          var y = (t & 4) !== 0, b = !y && e === "scroll", h = y ? p !== null ? p + "Capture" : null : p;
          y = [];
          for (var f = c, v; f !== null; ) {
            v = f;
            var E = v.stateNode;
            if (v.tag === 5 && E !== null && (v = E, h !== null && (E = dl(f, h), E != null && y.push(xl(f, E, v)))), b) break;
            f = f.return;
          }
          0 < y.length && (p = new S(p, w, null, n, m), d.push({
            event: p,
            listeners: y
          }));
        }
      }
      if (!(t & 7)) {
        e: {
          if (p = e === "mouseover" || e === "pointerover", S = e === "mouseout" || e === "pointerout", p && n !== fo && (w = n.relatedTarget || n.fromElement) && (Un(w) || w[nn])) break e;
          if ((S || p) && (p = m.window === m ? m : (p = m.ownerDocument) ? p.defaultView || p.parentWindow : window, S ? (w = n.relatedTarget || n.toElement, S = c, w = w ? Un(w) : null, w !== null && (b = Zn(w), w !== b || w.tag !== 5 && w.tag !== 6) && (w = null)) : (S = null, w = c), S !== w)) {
            if (y = Nu, E = "onMouseLeave", h = "onMouseEnter", f = "mouse", (e === "pointerout" || e === "pointerover") && (y = Eu, E = "onPointerLeave", h = "onPointerEnter", f = "pointer"), b = S == null ? p : dr(S), v = w == null ? p : dr(w), p = new y(E, f + "leave", S, n, m), p.target = b, p.relatedTarget = v, E = null, Un(m) === c && (y = new y(h, f + "enter", w, n, m), y.target = v, y.relatedTarget = b, E = y), b = E, S && w) t: {
              for (y = S, h = w, f = 0, v = y; v; v = rr(v)) f++;
              for (v = 0, E = h; E; E = rr(E)) v++;
              for (; 0 < f - v; ) y = rr(y), f--;
              for (; 0 < v - f; ) h = rr(h), v--;
              for (; f--; ) {
                if (y === h || h !== null && y === h.alternate) break t;
                y = rr(y), h = rr(h);
              }
              y = null;
            }
            else y = null;
            S !== null && Au(d, p, S, y, false), w !== null && b !== null && Au(d, b, w, y, true);
          }
        }
        e: {
          if (p = c ? dr(c) : window, S = p.nodeName && p.nodeName.toLowerCase(), S === "select" || S === "input" && p.type === "file") var _ = dh;
          else if (bu(p)) if (Vd) _ = hh;
          else {
            _ = mh;
            var R = fh;
          }
          else (S = p.nodeName) && S.toLowerCase() === "input" && (p.type === "checkbox" || p.type === "radio") && (_ = ph);
          if (_ && (_ = _(e, c))) {
            Bd(d, _, n, m);
            break e;
          }
          R && R(e, p, c), e === "focusout" && (R = p._wrapperState) && R.controlled && p.type === "number" && so(p, "number", p.value);
        }
        switch (R = c ? dr(c) : window, e) {
          case "focusin":
            (bu(R) || R.contentEditable === "true") && (ur = R, yo = c, rl = null);
            break;
          case "focusout":
            rl = yo = ur = null;
            break;
          case "mousedown":
            wo = true;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            wo = false, Du(d, n, m);
            break;
          case "selectionchange":
            if (xh) break;
          case "keydown":
          case "keyup":
            Du(d, n, m);
        }
        var k;
        if (hi) e: {
          switch (e) {
            case "compositionstart":
              var j = "onCompositionStart";
              break e;
            case "compositionend":
              j = "onCompositionEnd";
              break e;
            case "compositionupdate":
              j = "onCompositionUpdate";
              break e;
          }
          j = void 0;
        }
        else ir ? $d(e, n) && (j = "onCompositionEnd") : e === "keydown" && n.keyCode === 229 && (j = "onCompositionStart");
        j && (Ud && n.locale !== "ko" && (ir || j !== "onCompositionStart" ? j === "onCompositionEnd" && ir && (k = zd()) : (pn = m, fi = "value" in pn ? pn.value : pn.textContent, ir = true)), R = Pa(c, j), 0 < R.length && (j = new ju(j, e, null, n, m), d.push({
          event: j,
          listeners: R
        }), k ? j.data = k : (k = Fd(n), k !== null && (j.data = k)))), (k = sh ? oh(e, n) : ih(e, n)) && (c = Pa(c, "onBeforeInput"), 0 < c.length && (m = new ju("onBeforeInput", "beforeinput", null, n, m), d.push({
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
  function rr(e) {
    if (e === null) return null;
    do
      e = e.return;
    while (e && e.tag !== 5);
    return e || null;
  }
  function Au(e, t, n, r, l) {
    for (var a = t._reactName, s = []; n !== null && n !== r; ) {
      var u = n, i = u.alternate, c = u.stateNode;
      if (i !== null && i === r) break;
      u.tag === 5 && c !== null && (u = c, l ? (i = dl(n, a), i != null && s.unshift(xl(n, i, u))) : l || (i = dl(n, a), i != null && s.push(xl(n, i, u)))), n = n.return;
    }
    s.length !== 0 && e.push({
      event: t,
      listeners: s
    });
  }
  var kh = /\r\n?/g, Nh = /\u0000|\uFFFD/g;
  function Iu(e) {
    return (typeof e == "string" ? e : "" + e).replace(kh, `
`).replace(Nh, "");
  }
  function Zl(e, t, n) {
    if (t = Iu(t), Iu(e) !== t && n) throw Error(P(425));
  }
  function Ma() {
  }
  var So = null, ko = null;
  function No(e, t) {
    return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
  }
  var jo = typeof setTimeout == "function" ? setTimeout : void 0, jh = typeof clearTimeout == "function" ? clearTimeout : void 0, zu = typeof Promise == "function" ? Promise : void 0, Eh = typeof queueMicrotask == "function" ? queueMicrotask : typeof zu < "u" ? function(e) {
    return zu.resolve(null).then(e).catch(Ch);
  } : jo;
  function Ch(e) {
    setTimeout(function() {
      throw e;
    });
  }
  function Is(e, t) {
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
  function wn(e) {
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
  function Uu(e) {
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
  var Dr = Math.random().toString(36).slice(2), Ht = "__reactFiber$" + Dr, yl = "__reactProps$" + Dr, nn = "__reactContainer$" + Dr, Eo = "__reactEvents$" + Dr, _h = "__reactListeners$" + Dr, bh = "__reactHandles$" + Dr;
  function Un(e) {
    var t = e[Ht];
    if (t) return t;
    for (var n = e.parentNode; n; ) {
      if (t = n[nn] || n[Ht]) {
        if (n = t.alternate, t.child !== null || n !== null && n.child !== null) for (e = Uu(e); e !== null; ) {
          if (n = e[Ht]) return n;
          e = Uu(e);
        }
        return t;
      }
      e = n, n = e.parentNode;
    }
    return null;
  }
  function Pl(e) {
    return e = e[Ht] || e[nn], !e || e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3 ? null : e;
  }
  function dr(e) {
    if (e.tag === 5 || e.tag === 6) return e.stateNode;
    throw Error(P(33));
  }
  function rs(e) {
    return e[yl] || null;
  }
  var Co = [], fr = -1;
  function Rn(e) {
    return {
      current: e
    };
  }
  function je(e) {
    0 > fr || (e.current = Co[fr], Co[fr] = null, fr--);
  }
  function we(e, t) {
    fr++, Co[fr] = e.current, e.current = t;
  }
  var _n = {}, rt = Rn(_n), ht = Rn(false), Hn = _n;
  function Er(e, t) {
    var n = e.type.contextTypes;
    if (!n) return _n;
    var r = e.stateNode;
    if (r && r.__reactInternalMemoizedUnmaskedChildContext === t) return r.__reactInternalMemoizedMaskedChildContext;
    var l = {}, a;
    for (a in n) l[a] = t[a];
    return r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = t, e.__reactInternalMemoizedMaskedChildContext = l), l;
  }
  function gt(e) {
    return e = e.childContextTypes, e != null;
  }
  function Da() {
    je(ht), je(rt);
  }
  function $u(e, t, n) {
    if (rt.current !== _n) throw Error(P(168));
    we(rt, t), we(ht, n);
  }
  function tf(e, t, n) {
    var r = e.stateNode;
    if (t = t.childContextTypes, typeof r.getChildContext != "function") return n;
    r = r.getChildContext();
    for (var l in r) if (!(l in t)) throw Error(P(108, fp(e) || "Unknown", l));
    return Re({}, n, r);
  }
  function La(e) {
    return e = (e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext || _n, Hn = rt.current, we(rt, e), we(ht, ht.current), true;
  }
  function Fu(e, t, n) {
    var r = e.stateNode;
    if (!r) throw Error(P(169));
    n ? (e = tf(e, t, Hn), r.__reactInternalMemoizedMergedChildContext = e, je(ht), je(rt), we(rt, e)) : je(ht), we(ht, n);
  }
  var Xt = null, ls = false, zs = false;
  function nf(e) {
    Xt === null ? Xt = [
      e
    ] : Xt.push(e);
  }
  function Rh(e) {
    ls = true, nf(e);
  }
  function Tn() {
    if (!zs && Xt !== null) {
      zs = true;
      var e = 0, t = ce;
      try {
        var n = Xt;
        for (ce = 1; e < n.length; e++) {
          var r = n[e];
          do
            r = r(true);
          while (r !== null);
        }
        Xt = null, ls = false;
      } catch (l) {
        throw Xt !== null && (Xt = Xt.slice(e + 1)), _d(ii, Tn), l;
      } finally {
        ce = t, zs = false;
      }
    }
    return null;
  }
  var mr = [], pr = 0, Oa = null, Aa = 0, Ct = [], _t = 0, Qn = null, Zt = 1, qt = "";
  function On(e, t) {
    mr[pr++] = Aa, mr[pr++] = Oa, Oa = e, Aa = t;
  }
  function rf(e, t, n) {
    Ct[_t++] = Zt, Ct[_t++] = qt, Ct[_t++] = Qn, Qn = e;
    var r = Zt;
    e = qt;
    var l = 32 - Ut(r) - 1;
    r &= ~(1 << l), n += 1;
    var a = 32 - Ut(t) + l;
    if (30 < a) {
      var s = l - l % 5;
      a = (r & (1 << s) - 1).toString(32), r >>= s, l -= s, Zt = 1 << 32 - Ut(t) + l | n << l | r, qt = a + e;
    } else Zt = 1 << a | n << l | r, qt = e;
  }
  function vi(e) {
    e.return !== null && (On(e, 1), rf(e, 1, 0));
  }
  function xi(e) {
    for (; e === Oa; ) Oa = mr[--pr], mr[pr] = null, Aa = mr[--pr], mr[pr] = null;
    for (; e === Qn; ) Qn = Ct[--_t], Ct[_t] = null, qt = Ct[--_t], Ct[_t] = null, Zt = Ct[--_t], Ct[_t] = null;
  }
  var St = null, wt = null, Ce = false, zt = null;
  function lf(e, t) {
    var n = bt(5, null, null, 0);
    n.elementType = "DELETED", n.stateNode = t, n.return = e, t = e.deletions, t === null ? (e.deletions = [
      n
    ], e.flags |= 16) : t.push(n);
  }
  function Bu(e, t) {
    switch (e.tag) {
      case 5:
        var n = e.type;
        return t = t.nodeType !== 1 || n.toLowerCase() !== t.nodeName.toLowerCase() ? null : t, t !== null ? (e.stateNode = t, St = e, wt = wn(t.firstChild), true) : false;
      case 6:
        return t = e.pendingProps === "" || t.nodeType !== 3 ? null : t, t !== null ? (e.stateNode = t, St = e, wt = null, true) : false;
      case 13:
        return t = t.nodeType !== 8 ? null : t, t !== null ? (n = Qn !== null ? {
          id: Zt,
          overflow: qt
        } : null, e.memoizedState = {
          dehydrated: t,
          treeContext: n,
          retryLane: 1073741824
        }, n = bt(18, null, null, 0), n.stateNode = t, n.return = e, e.child = n, St = e, wt = null, true) : false;
      default:
        return false;
    }
  }
  function _o(e) {
    return (e.mode & 1) !== 0 && (e.flags & 128) === 0;
  }
  function bo(e) {
    if (Ce) {
      var t = wt;
      if (t) {
        var n = t;
        if (!Bu(e, t)) {
          if (_o(e)) throw Error(P(418));
          t = wn(n.nextSibling);
          var r = St;
          t && Bu(e, t) ? lf(r, n) : (e.flags = e.flags & -4097 | 2, Ce = false, St = e);
        }
      } else {
        if (_o(e)) throw Error(P(418));
        e.flags = e.flags & -4097 | 2, Ce = false, St = e;
      }
    }
  }
  function Vu(e) {
    for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13; ) e = e.return;
    St = e;
  }
  function ql(e) {
    if (e !== St) return false;
    if (!Ce) return Vu(e), Ce = true, false;
    var t;
    if ((t = e.tag !== 3) && !(t = e.tag !== 5) && (t = e.type, t = t !== "head" && t !== "body" && !No(e.type, e.memoizedProps)), t && (t = wt)) {
      if (_o(e)) throw af(), Error(P(418));
      for (; t; ) lf(e, t), t = wn(t.nextSibling);
    }
    if (Vu(e), e.tag === 13) {
      if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e) throw Error(P(317));
      e: {
        for (e = e.nextSibling, t = 0; e; ) {
          if (e.nodeType === 8) {
            var n = e.data;
            if (n === "/$") {
              if (t === 0) {
                wt = wn(e.nextSibling);
                break e;
              }
              t--;
            } else n !== "$" && n !== "$!" && n !== "$?" || t++;
          }
          e = e.nextSibling;
        }
        wt = null;
      }
    } else wt = St ? wn(e.stateNode.nextSibling) : null;
    return true;
  }
  function af() {
    for (var e = wt; e; ) e = wn(e.nextSibling);
  }
  function Cr() {
    wt = St = null, Ce = false;
  }
  function yi(e) {
    zt === null ? zt = [
      e
    ] : zt.push(e);
  }
  var Th = an.ReactCurrentBatchConfig;
  function Br(e, t, n) {
    if (e = n.ref, e !== null && typeof e != "function" && typeof e != "object") {
      if (n._owner) {
        if (n = n._owner, n) {
          if (n.tag !== 1) throw Error(P(309));
          var r = n.stateNode;
        }
        if (!r) throw Error(P(147, e));
        var l = r, a = "" + e;
        return t !== null && t.ref !== null && typeof t.ref == "function" && t.ref._stringRef === a ? t.ref : (t = function(s) {
          var u = l.refs;
          s === null ? delete u[a] : u[a] = s;
        }, t._stringRef = a, t);
      }
      if (typeof e != "string") throw Error(P(284));
      if (!n._owner) throw Error(P(290, e));
    }
    return e;
  }
  function ea(e, t) {
    throw e = Object.prototype.toString.call(t), Error(P(31, e === "[object Object]" ? "object with keys {" + Object.keys(t).join(", ") + "}" : e));
  }
  function Wu(e) {
    var t = e._init;
    return t(e._payload);
  }
  function sf(e) {
    function t(h, f) {
      if (e) {
        var v = h.deletions;
        v === null ? (h.deletions = [
          f
        ], h.flags |= 16) : v.push(f);
      }
    }
    function n(h, f) {
      if (!e) return null;
      for (; f !== null; ) t(h, f), f = f.sibling;
      return null;
    }
    function r(h, f) {
      for (h = /* @__PURE__ */ new Map(); f !== null; ) f.key !== null ? h.set(f.key, f) : h.set(f.index, f), f = f.sibling;
      return h;
    }
    function l(h, f) {
      return h = jn(h, f), h.index = 0, h.sibling = null, h;
    }
    function a(h, f, v) {
      return h.index = v, e ? (v = h.alternate, v !== null ? (v = v.index, v < f ? (h.flags |= 2, f) : v) : (h.flags |= 2, f)) : (h.flags |= 1048576, f);
    }
    function s(h) {
      return e && h.alternate === null && (h.flags |= 2), h;
    }
    function u(h, f, v, E) {
      return f === null || f.tag !== 6 ? (f = Hs(v, h.mode, E), f.return = h, f) : (f = l(f, v), f.return = h, f);
    }
    function i(h, f, v, E) {
      var _ = v.type;
      return _ === or ? m(h, f, v.props.children, E, v.key) : f !== null && (f.elementType === _ || typeof _ == "object" && _ !== null && _.$$typeof === cn && Wu(_) === f.type) ? (E = l(f, v.props), E.ref = Br(h, f, v), E.return = h, E) : (E = wa(v.type, v.key, v.props, null, h.mode, E), E.ref = Br(h, f, v), E.return = h, E);
    }
    function c(h, f, v, E) {
      return f === null || f.tag !== 4 || f.stateNode.containerInfo !== v.containerInfo || f.stateNode.implementation !== v.implementation ? (f = Qs(v, h.mode, E), f.return = h, f) : (f = l(f, v.children || []), f.return = h, f);
    }
    function m(h, f, v, E, _) {
      return f === null || f.tag !== 7 ? (f = Wn(v, h.mode, E, _), f.return = h, f) : (f = l(f, v), f.return = h, f);
    }
    function d(h, f, v) {
      if (typeof f == "string" && f !== "" || typeof f == "number") return f = Hs("" + f, h.mode, v), f.return = h, f;
      if (typeof f == "object" && f !== null) {
        switch (f.$$typeof) {
          case Vl:
            return v = wa(f.type, f.key, f.props, null, h.mode, v), v.ref = Br(h, null, f), v.return = h, v;
          case sr:
            return f = Qs(f, h.mode, v), f.return = h, f;
          case cn:
            var E = f._init;
            return d(h, E(f._payload), v);
        }
        if (Yr(f) || Ir(f)) return f = Wn(f, h.mode, v, null), f.return = h, f;
        ea(h, f);
      }
      return null;
    }
    function p(h, f, v, E) {
      var _ = f !== null ? f.key : null;
      if (typeof v == "string" && v !== "" || typeof v == "number") return _ !== null ? null : u(h, f, "" + v, E);
      if (typeof v == "object" && v !== null) {
        switch (v.$$typeof) {
          case Vl:
            return v.key === _ ? i(h, f, v, E) : null;
          case sr:
            return v.key === _ ? c(h, f, v, E) : null;
          case cn:
            return _ = v._init, p(h, f, _(v._payload), E);
        }
        if (Yr(v) || Ir(v)) return _ !== null ? null : m(h, f, v, E, null);
        ea(h, v);
      }
      return null;
    }
    function S(h, f, v, E, _) {
      if (typeof E == "string" && E !== "" || typeof E == "number") return h = h.get(v) || null, u(f, h, "" + E, _);
      if (typeof E == "object" && E !== null) {
        switch (E.$$typeof) {
          case Vl:
            return h = h.get(E.key === null ? v : E.key) || null, i(f, h, E, _);
          case sr:
            return h = h.get(E.key === null ? v : E.key) || null, c(f, h, E, _);
          case cn:
            var R = E._init;
            return S(h, f, v, R(E._payload), _);
        }
        if (Yr(E) || Ir(E)) return h = h.get(v) || null, m(f, h, E, _, null);
        ea(f, E);
      }
      return null;
    }
    function w(h, f, v, E) {
      for (var _ = null, R = null, k = f, j = f = 0, I = null; k !== null && j < v.length; j++) {
        k.index > j ? (I = k, k = null) : I = k.sibling;
        var D = p(h, k, v[j], E);
        if (D === null) {
          k === null && (k = I);
          break;
        }
        e && k && D.alternate === null && t(h, k), f = a(D, f, j), R === null ? _ = D : R.sibling = D, R = D, k = I;
      }
      if (j === v.length) return n(h, k), Ce && On(h, j), _;
      if (k === null) {
        for (; j < v.length; j++) k = d(h, v[j], E), k !== null && (f = a(k, f, j), R === null ? _ = k : R.sibling = k, R = k);
        return Ce && On(h, j), _;
      }
      for (k = r(h, k); j < v.length; j++) I = S(k, h, j, v[j], E), I !== null && (e && I.alternate !== null && k.delete(I.key === null ? j : I.key), f = a(I, f, j), R === null ? _ = I : R.sibling = I, R = I);
      return e && k.forEach(function(Q) {
        return t(h, Q);
      }), Ce && On(h, j), _;
    }
    function y(h, f, v, E) {
      var _ = Ir(v);
      if (typeof _ != "function") throw Error(P(150));
      if (v = _.call(v), v == null) throw Error(P(151));
      for (var R = _ = null, k = f, j = f = 0, I = null, D = v.next(); k !== null && !D.done; j++, D = v.next()) {
        k.index > j ? (I = k, k = null) : I = k.sibling;
        var Q = p(h, k, D.value, E);
        if (Q === null) {
          k === null && (k = I);
          break;
        }
        e && k && Q.alternate === null && t(h, k), f = a(Q, f, j), R === null ? _ = Q : R.sibling = Q, R = Q, k = I;
      }
      if (D.done) return n(h, k), Ce && On(h, j), _;
      if (k === null) {
        for (; !D.done; j++, D = v.next()) D = d(h, D.value, E), D !== null && (f = a(D, f, j), R === null ? _ = D : R.sibling = D, R = D);
        return Ce && On(h, j), _;
      }
      for (k = r(h, k); !D.done; j++, D = v.next()) D = S(k, h, j, D.value, E), D !== null && (e && D.alternate !== null && k.delete(D.key === null ? j : D.key), f = a(D, f, j), R === null ? _ = D : R.sibling = D, R = D);
      return e && k.forEach(function(K) {
        return t(h, K);
      }), Ce && On(h, j), _;
    }
    function b(h, f, v, E) {
      if (typeof v == "object" && v !== null && v.type === or && v.key === null && (v = v.props.children), typeof v == "object" && v !== null) {
        switch (v.$$typeof) {
          case Vl:
            e: {
              for (var _ = v.key, R = f; R !== null; ) {
                if (R.key === _) {
                  if (_ = v.type, _ === or) {
                    if (R.tag === 7) {
                      n(h, R.sibling), f = l(R, v.props.children), f.return = h, h = f;
                      break e;
                    }
                  } else if (R.elementType === _ || typeof _ == "object" && _ !== null && _.$$typeof === cn && Wu(_) === R.type) {
                    n(h, R.sibling), f = l(R, v.props), f.ref = Br(h, R, v), f.return = h, h = f;
                    break e;
                  }
                  n(h, R);
                  break;
                } else t(h, R);
                R = R.sibling;
              }
              v.type === or ? (f = Wn(v.props.children, h.mode, E, v.key), f.return = h, h = f) : (E = wa(v.type, v.key, v.props, null, h.mode, E), E.ref = Br(h, f, v), E.return = h, h = E);
            }
            return s(h);
          case sr:
            e: {
              for (R = v.key; f !== null; ) {
                if (f.key === R) if (f.tag === 4 && f.stateNode.containerInfo === v.containerInfo && f.stateNode.implementation === v.implementation) {
                  n(h, f.sibling), f = l(f, v.children || []), f.return = h, h = f;
                  break e;
                } else {
                  n(h, f);
                  break;
                }
                else t(h, f);
                f = f.sibling;
              }
              f = Qs(v, h.mode, E), f.return = h, h = f;
            }
            return s(h);
          case cn:
            return R = v._init, b(h, f, R(v._payload), E);
        }
        if (Yr(v)) return w(h, f, v, E);
        if (Ir(v)) return y(h, f, v, E);
        ea(h, v);
      }
      return typeof v == "string" && v !== "" || typeof v == "number" ? (v = "" + v, f !== null && f.tag === 6 ? (n(h, f.sibling), f = l(f, v), f.return = h, h = f) : (n(h, f), f = Hs(v, h.mode, E), f.return = h, h = f), s(h)) : n(h, f);
    }
    return b;
  }
  var _r = sf(true), of = sf(false), Ia = Rn(null), za = null, hr = null, wi = null;
  function Si() {
    wi = hr = za = null;
  }
  function ki(e) {
    var t = Ia.current;
    je(Ia), e._currentValue = t;
  }
  function Ro(e, t, n) {
    for (; e !== null; ) {
      var r = e.alternate;
      if ((e.childLanes & t) !== t ? (e.childLanes |= t, r !== null && (r.childLanes |= t)) : r !== null && (r.childLanes & t) !== t && (r.childLanes |= t), e === n) break;
      e = e.return;
    }
  }
  function kr(e, t) {
    za = e, wi = hr = null, e = e.dependencies, e !== null && e.firstContext !== null && (e.lanes & t && (pt = true), e.firstContext = null);
  }
  function Tt(e) {
    var t = e._currentValue;
    if (wi !== e) if (e = {
      context: e,
      memoizedValue: t,
      next: null
    }, hr === null) {
      if (za === null) throw Error(P(308));
      hr = e, za.dependencies = {
        lanes: 0,
        firstContext: e
      };
    } else hr = hr.next = e;
    return t;
  }
  var $n = null;
  function Ni(e) {
    $n === null ? $n = [
      e
    ] : $n.push(e);
  }
  function uf(e, t, n, r) {
    var l = t.interleaved;
    return l === null ? (n.next = n, Ni(t)) : (n.next = l.next, l.next = n), t.interleaved = n, rn(e, r);
  }
  function rn(e, t) {
    e.lanes |= t;
    var n = e.alternate;
    for (n !== null && (n.lanes |= t), n = e, e = e.return; e !== null; ) e.childLanes |= t, n = e.alternate, n !== null && (n.childLanes |= t), n = e, e = e.return;
    return n.tag === 3 ? n.stateNode : null;
  }
  var dn = false;
  function ji(e) {
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
  function en(e, t) {
    return {
      eventTime: e,
      lane: t,
      tag: 0,
      payload: null,
      callback: null,
      next: null
    };
  }
  function Sn(e, t, n) {
    var r = e.updateQueue;
    if (r === null) return null;
    if (r = r.shared, re & 2) {
      var l = r.pending;
      return l === null ? t.next = t : (t.next = l.next, l.next = t), r.pending = t, rn(e, n);
    }
    return l = r.interleaved, l === null ? (t.next = t, Ni(r)) : (t.next = l.next, l.next = t), r.interleaved = t, rn(e, n);
  }
  function pa(e, t, n) {
    if (t = t.updateQueue, t !== null && (t = t.shared, (n & 4194240) !== 0)) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, ui(e, n);
    }
  }
  function Hu(e, t) {
    var n = e.updateQueue, r = e.alternate;
    if (r !== null && (r = r.updateQueue, n === r)) {
      var l = null, a = null;
      if (n = n.firstBaseUpdate, n !== null) {
        do {
          var s = {
            eventTime: n.eventTime,
            lane: n.lane,
            tag: n.tag,
            payload: n.payload,
            callback: n.callback,
            next: null
          };
          a === null ? l = a = s : a = a.next = s, n = n.next;
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
  function Ua(e, t, n, r) {
    var l = e.updateQueue;
    dn = false;
    var a = l.firstBaseUpdate, s = l.lastBaseUpdate, u = l.shared.pending;
    if (u !== null) {
      l.shared.pending = null;
      var i = u, c = i.next;
      i.next = null, s === null ? a = c : s.next = c, s = i;
      var m = e.alternate;
      m !== null && (m = m.updateQueue, u = m.lastBaseUpdate, u !== s && (u === null ? m.firstBaseUpdate = c : u.next = c, m.lastBaseUpdate = i));
    }
    if (a !== null) {
      var d = l.baseState;
      s = 0, m = c = i = null, u = a;
      do {
        var p = u.lane, S = u.eventTime;
        if ((r & p) === p) {
          m !== null && (m = m.next = {
            eventTime: S,
            lane: 0,
            tag: u.tag,
            payload: u.payload,
            callback: u.callback,
            next: null
          });
          e: {
            var w = e, y = u;
            switch (p = t, S = n, y.tag) {
              case 1:
                if (w = y.payload, typeof w == "function") {
                  d = w.call(S, d, p);
                  break e;
                }
                d = w;
                break e;
              case 3:
                w.flags = w.flags & -65537 | 128;
              case 0:
                if (w = y.payload, p = typeof w == "function" ? w.call(S, d, p) : w, p == null) break e;
                d = Re({}, d, p);
                break e;
              case 2:
                dn = true;
            }
          }
          u.callback !== null && u.lane !== 0 && (e.flags |= 64, p = l.effects, p === null ? l.effects = [
            u
          ] : p.push(u));
        } else S = {
          eventTime: S,
          lane: p,
          tag: u.tag,
          payload: u.payload,
          callback: u.callback,
          next: null
        }, m === null ? (c = m = S, i = d) : m = m.next = S, s |= p;
        if (u = u.next, u === null) {
          if (u = l.shared.pending, u === null) break;
          p = u, u = p.next, p.next = null, l.lastBaseUpdate = p, l.shared.pending = null;
        }
      } while (true);
      if (m === null && (i = d), l.baseState = i, l.firstBaseUpdate = c, l.lastBaseUpdate = m, t = l.shared.interleaved, t !== null) {
        l = t;
        do
          s |= l.lane, l = l.next;
        while (l !== t);
      } else a === null && (l.shared.lanes = 0);
      Gn |= s, e.lanes = s, e.memoizedState = d;
    }
  }
  function Qu(e, t, n) {
    if (e = t.effects, t.effects = null, e !== null) for (t = 0; t < e.length; t++) {
      var r = e[t], l = r.callback;
      if (l !== null) {
        if (r.callback = null, r = n, typeof l != "function") throw Error(P(191, l));
        l.call(r);
      }
    }
  }
  var Ml = {}, Kt = Rn(Ml), wl = Rn(Ml), Sl = Rn(Ml);
  function Fn(e) {
    if (e === Ml) throw Error(P(174));
    return e;
  }
  function Ei(e, t) {
    switch (we(Sl, t), we(wl, e), we(Kt, Ml), e = t.nodeType, e) {
      case 9:
      case 11:
        t = (t = t.documentElement) ? t.namespaceURI : io(null, "");
        break;
      default:
        e = e === 8 ? t.parentNode : t, t = e.namespaceURI || null, e = e.tagName, t = io(t, e);
    }
    je(Kt), we(Kt, t);
  }
  function br() {
    je(Kt), je(wl), je(Sl);
  }
  function df(e) {
    Fn(Sl.current);
    var t = Fn(Kt.current), n = io(t, e.type);
    t !== n && (we(wl, e), we(Kt, n));
  }
  function Ci(e) {
    wl.current === e && (je(Kt), je(wl));
  }
  var _e = Rn(0);
  function $a(e) {
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
  var Us = [];
  function _i() {
    for (var e = 0; e < Us.length; e++) Us[e]._workInProgressVersionPrimary = null;
    Us.length = 0;
  }
  var ha = an.ReactCurrentDispatcher, $s = an.ReactCurrentBatchConfig, Kn = 0, be = null, Ue = null, We = null, Fa = false, ll = false, kl = 0, Ph = 0;
  function et() {
    throw Error(P(321));
  }
  function bi(e, t) {
    if (t === null) return false;
    for (var n = 0; n < t.length && n < e.length; n++) if (!Ft(e[n], t[n])) return false;
    return true;
  }
  function Ri(e, t, n, r, l, a) {
    if (Kn = a, be = t, t.memoizedState = null, t.updateQueue = null, t.lanes = 0, ha.current = e === null || e.memoizedState === null ? Oh : Ah, e = n(r, l), ll) {
      a = 0;
      do {
        if (ll = false, kl = 0, 25 <= a) throw Error(P(301));
        a += 1, We = Ue = null, t.updateQueue = null, ha.current = Ih, e = n(r, l);
      } while (ll);
    }
    if (ha.current = Ba, t = Ue !== null && Ue.next !== null, Kn = 0, We = Ue = be = null, Fa = false, t) throw Error(P(300));
    return e;
  }
  function Ti() {
    var e = kl !== 0;
    return kl = 0, e;
  }
  function Wt() {
    var e = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return We === null ? be.memoizedState = We = e : We = We.next = e, We;
  }
  function Pt() {
    if (Ue === null) {
      var e = be.alternate;
      e = e !== null ? e.memoizedState : null;
    } else e = Ue.next;
    var t = We === null ? be.memoizedState : We.next;
    if (t !== null) We = t, Ue = e;
    else {
      if (e === null) throw Error(P(310));
      Ue = e, e = {
        memoizedState: Ue.memoizedState,
        baseState: Ue.baseState,
        baseQueue: Ue.baseQueue,
        queue: Ue.queue,
        next: null
      }, We === null ? be.memoizedState = We = e : We = We.next = e;
    }
    return We;
  }
  function Nl(e, t) {
    return typeof t == "function" ? t(e) : t;
  }
  function Fs(e) {
    var t = Pt(), n = t.queue;
    if (n === null) throw Error(P(311));
    n.lastRenderedReducer = e;
    var r = Ue, l = r.baseQueue, a = n.pending;
    if (a !== null) {
      if (l !== null) {
        var s = l.next;
        l.next = a.next, a.next = s;
      }
      r.baseQueue = l = a, n.pending = null;
    }
    if (l !== null) {
      a = l.next, r = r.baseState;
      var u = s = null, i = null, c = a;
      do {
        var m = c.lane;
        if ((Kn & m) === m) i !== null && (i = i.next = {
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
          i === null ? (u = i = d, s = r) : i = i.next = d, be.lanes |= m, Gn |= m;
        }
        c = c.next;
      } while (c !== null && c !== a);
      i === null ? s = r : i.next = u, Ft(r, t.memoizedState) || (pt = true), t.memoizedState = r, t.baseState = s, t.baseQueue = i, n.lastRenderedState = r;
    }
    if (e = n.interleaved, e !== null) {
      l = e;
      do
        a = l.lane, be.lanes |= a, Gn |= a, l = l.next;
      while (l !== e);
    } else l === null && (n.lanes = 0);
    return [
      t.memoizedState,
      n.dispatch
    ];
  }
  function Bs(e) {
    var t = Pt(), n = t.queue;
    if (n === null) throw Error(P(311));
    n.lastRenderedReducer = e;
    var r = n.dispatch, l = n.pending, a = t.memoizedState;
    if (l !== null) {
      n.pending = null;
      var s = l = l.next;
      do
        a = e(a, s.action), s = s.next;
      while (s !== l);
      Ft(a, t.memoizedState) || (pt = true), t.memoizedState = a, t.baseQueue === null && (t.baseState = a), n.lastRenderedState = a;
    }
    return [
      a,
      r
    ];
  }
  function ff() {
  }
  function mf(e, t) {
    var n = be, r = Pt(), l = t(), a = !Ft(r.memoizedState, l);
    if (a && (r.memoizedState = l, pt = true), r = r.queue, Pi(gf.bind(null, n, r, e), [
      e
    ]), r.getSnapshot !== t || a || We !== null && We.memoizedState.tag & 1) {
      if (n.flags |= 2048, jl(9, hf.bind(null, n, r, l, t), void 0, null), He === null) throw Error(P(349));
      Kn & 30 || pf(n, t, l);
    }
    return l;
  }
  function pf(e, t, n) {
    e.flags |= 16384, e = {
      getSnapshot: t,
      value: n
    }, t = be.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, be.updateQueue = t, t.stores = [
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
      return !Ft(e, n);
    } catch {
      return true;
    }
  }
  function xf(e) {
    var t = rn(e, 1);
    t !== null && $t(t, e, 1, -1);
  }
  function Ku(e) {
    var t = Wt();
    return typeof e == "function" && (e = e()), t.memoizedState = t.baseState = e, e = {
      pending: null,
      interleaved: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Nl,
      lastRenderedState: e
    }, t.queue = e, e = e.dispatch = Lh.bind(null, be, e), [
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
    }, t = be.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, be.updateQueue = t, t.lastEffect = e.next = e) : (n = t.lastEffect, n === null ? t.lastEffect = e.next = e : (r = n.next, n.next = e, e.next = r, t.lastEffect = e)), e;
  }
  function yf() {
    return Pt().memoizedState;
  }
  function ga(e, t, n, r) {
    var l = Wt();
    be.flags |= e, l.memoizedState = jl(1 | t, n, void 0, r === void 0 ? null : r);
  }
  function as(e, t, n, r) {
    var l = Pt();
    r = r === void 0 ? null : r;
    var a = void 0;
    if (Ue !== null) {
      var s = Ue.memoizedState;
      if (a = s.destroy, r !== null && bi(r, s.deps)) {
        l.memoizedState = jl(t, n, a, r);
        return;
      }
    }
    be.flags |= e, l.memoizedState = jl(1 | t, n, a, r);
  }
  function Gu(e, t) {
    return ga(8390656, 8, e, t);
  }
  function Pi(e, t) {
    return as(2048, 8, e, t);
  }
  function wf(e, t) {
    return as(4, 2, e, t);
  }
  function Sf(e, t) {
    return as(4, 4, e, t);
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
    ]) : null, as(4, 4, kf.bind(null, t, e), n);
  }
  function Mi() {
  }
  function jf(e, t) {
    var n = Pt();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && bi(t, r[1]) ? r[0] : (n.memoizedState = [
      e,
      t
    ], e);
  }
  function Ef(e, t) {
    var n = Pt();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && bi(t, r[1]) ? r[0] : (e = e(), n.memoizedState = [
      e,
      t
    ], e);
  }
  function Cf(e, t, n) {
    return Kn & 21 ? (Ft(n, t) || (n = Td(), be.lanes |= n, Gn |= n, e.baseState = true), t) : (e.baseState && (e.baseState = false, pt = true), e.memoizedState = n);
  }
  function Mh(e, t) {
    var n = ce;
    ce = n !== 0 && 4 > n ? n : 4, e(true);
    var r = $s.transition;
    $s.transition = {};
    try {
      e(false), t();
    } finally {
      ce = n, $s.transition = r;
    }
  }
  function _f() {
    return Pt().memoizedState;
  }
  function Dh(e, t, n) {
    var r = Nn(e);
    if (n = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    }, bf(e)) Rf(t, n);
    else if (n = uf(e, t, n, r), n !== null) {
      var l = st();
      $t(n, e, r, l), Tf(n, t, r);
    }
  }
  function Lh(e, t, n) {
    var r = Nn(e), l = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    };
    if (bf(e)) Rf(t, l);
    else {
      var a = e.alternate;
      if (e.lanes === 0 && (a === null || a.lanes === 0) && (a = t.lastRenderedReducer, a !== null)) try {
        var s = t.lastRenderedState, u = a(s, n);
        if (l.hasEagerState = true, l.eagerState = u, Ft(u, s)) {
          var i = t.interleaved;
          i === null ? (l.next = l, Ni(t)) : (l.next = i.next, i.next = l), t.interleaved = l;
          return;
        }
      } catch {
      } finally {
      }
      n = uf(e, t, l, r), n !== null && (l = st(), $t(n, e, r, l), Tf(n, t, r));
    }
  }
  function bf(e) {
    var t = e.alternate;
    return e === be || t !== null && t === be;
  }
  function Rf(e, t) {
    ll = Fa = true;
    var n = e.pending;
    n === null ? t.next = t : (t.next = n.next, n.next = t), e.pending = t;
  }
  function Tf(e, t, n) {
    if (n & 4194240) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, ui(e, n);
    }
  }
  var Ba = {
    readContext: Tt,
    useCallback: et,
    useContext: et,
    useEffect: et,
    useImperativeHandle: et,
    useInsertionEffect: et,
    useLayoutEffect: et,
    useMemo: et,
    useReducer: et,
    useRef: et,
    useState: et,
    useDebugValue: et,
    useDeferredValue: et,
    useTransition: et,
    useMutableSource: et,
    useSyncExternalStore: et,
    useId: et,
    unstable_isNewReconciler: false
  }, Oh = {
    readContext: Tt,
    useCallback: function(e, t) {
      return Wt().memoizedState = [
        e,
        t === void 0 ? null : t
      ], e;
    },
    useContext: Tt,
    useEffect: Gu,
    useImperativeHandle: function(e, t, n) {
      return n = n != null ? n.concat([
        e
      ]) : null, ga(4194308, 4, kf.bind(null, t, e), n);
    },
    useLayoutEffect: function(e, t) {
      return ga(4194308, 4, e, t);
    },
    useInsertionEffect: function(e, t) {
      return ga(4, 2, e, t);
    },
    useMemo: function(e, t) {
      var n = Wt();
      return t = t === void 0 ? null : t, e = e(), n.memoizedState = [
        e,
        t
      ], e;
    },
    useReducer: function(e, t, n) {
      var r = Wt();
      return t = n !== void 0 ? n(t) : t, r.memoizedState = r.baseState = t, e = {
        pending: null,
        interleaved: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: e,
        lastRenderedState: t
      }, r.queue = e, e = e.dispatch = Dh.bind(null, be, e), [
        r.memoizedState,
        e
      ];
    },
    useRef: function(e) {
      var t = Wt();
      return e = {
        current: e
      }, t.memoizedState = e;
    },
    useState: Ku,
    useDebugValue: Mi,
    useDeferredValue: function(e) {
      return Wt().memoizedState = e;
    },
    useTransition: function() {
      var e = Ku(false), t = e[0];
      return e = Mh.bind(null, e[1]), Wt().memoizedState = e, [
        t,
        e
      ];
    },
    useMutableSource: function() {
    },
    useSyncExternalStore: function(e, t, n) {
      var r = be, l = Wt();
      if (Ce) {
        if (n === void 0) throw Error(P(407));
        n = n();
      } else {
        if (n = t(), He === null) throw Error(P(349));
        Kn & 30 || pf(r, t, n);
      }
      l.memoizedState = n;
      var a = {
        value: n,
        getSnapshot: t
      };
      return l.queue = a, Gu(gf.bind(null, r, a, e), [
        e
      ]), r.flags |= 2048, jl(9, hf.bind(null, r, a, n, t), void 0, null), n;
    },
    useId: function() {
      var e = Wt(), t = He.identifierPrefix;
      if (Ce) {
        var n = qt, r = Zt;
        n = (r & ~(1 << 32 - Ut(r) - 1)).toString(32) + n, t = ":" + t + "R" + n, n = kl++, 0 < n && (t += "H" + n.toString(32)), t += ":";
      } else n = Ph++, t = ":" + t + "r" + n.toString(32) + ":";
      return e.memoizedState = t;
    },
    unstable_isNewReconciler: false
  }, Ah = {
    readContext: Tt,
    useCallback: jf,
    useContext: Tt,
    useEffect: Pi,
    useImperativeHandle: Nf,
    useInsertionEffect: wf,
    useLayoutEffect: Sf,
    useMemo: Ef,
    useReducer: Fs,
    useRef: yf,
    useState: function() {
      return Fs(Nl);
    },
    useDebugValue: Mi,
    useDeferredValue: function(e) {
      var t = Pt();
      return Cf(t, Ue.memoizedState, e);
    },
    useTransition: function() {
      var e = Fs(Nl)[0], t = Pt().memoizedState;
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
    readContext: Tt,
    useCallback: jf,
    useContext: Tt,
    useEffect: Pi,
    useImperativeHandle: Nf,
    useInsertionEffect: wf,
    useLayoutEffect: Sf,
    useMemo: Ef,
    useReducer: Bs,
    useRef: yf,
    useState: function() {
      return Bs(Nl);
    },
    useDebugValue: Mi,
    useDeferredValue: function(e) {
      var t = Pt();
      return Ue === null ? t.memoizedState = e : Cf(t, Ue.memoizedState, e);
    },
    useTransition: function() {
      var e = Bs(Nl)[0], t = Pt().memoizedState;
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
  function Ot(e, t) {
    if (e && e.defaultProps) {
      t = Re({}, t), e = e.defaultProps;
      for (var n in e) t[n] === void 0 && (t[n] = e[n]);
      return t;
    }
    return t;
  }
  function To(e, t, n, r) {
    t = e.memoizedState, n = n(r, t), n = n == null ? t : Re({}, t, n), e.memoizedState = n, e.lanes === 0 && (e.updateQueue.baseState = n);
  }
  var ss = {
    isMounted: function(e) {
      return (e = e._reactInternals) ? Zn(e) === e : false;
    },
    enqueueSetState: function(e, t, n) {
      e = e._reactInternals;
      var r = st(), l = Nn(e), a = en(r, l);
      a.payload = t, n != null && (a.callback = n), t = Sn(e, a, l), t !== null && ($t(t, e, l, r), pa(t, e, l));
    },
    enqueueReplaceState: function(e, t, n) {
      e = e._reactInternals;
      var r = st(), l = Nn(e), a = en(r, l);
      a.tag = 1, a.payload = t, n != null && (a.callback = n), t = Sn(e, a, l), t !== null && ($t(t, e, l, r), pa(t, e, l));
    },
    enqueueForceUpdate: function(e, t) {
      e = e._reactInternals;
      var n = st(), r = Nn(e), l = en(n, r);
      l.tag = 2, t != null && (l.callback = t), t = Sn(e, l, r), t !== null && ($t(t, e, r, n), pa(t, e, r));
    }
  };
  function Yu(e, t, n, r, l, a, s) {
    return e = e.stateNode, typeof e.shouldComponentUpdate == "function" ? e.shouldComponentUpdate(r, a, s) : t.prototype && t.prototype.isPureReactComponent ? !gl(n, r) || !gl(l, a) : true;
  }
  function Pf(e, t, n) {
    var r = false, l = _n, a = t.contextType;
    return typeof a == "object" && a !== null ? a = Tt(a) : (l = gt(t) ? Hn : rt.current, r = t.contextTypes, a = (r = r != null) ? Er(e, l) : _n), t = new t(n, a), e.memoizedState = t.state !== null && t.state !== void 0 ? t.state : null, t.updater = ss, e.stateNode = t, t._reactInternals = e, r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = l, e.__reactInternalMemoizedMaskedChildContext = a), t;
  }
  function Ju(e, t, n, r) {
    e = t.state, typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(n, r), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(n, r), t.state !== e && ss.enqueueReplaceState(t, t.state, null);
  }
  function Po(e, t, n, r) {
    var l = e.stateNode;
    l.props = n, l.state = e.memoizedState, l.refs = {}, ji(e);
    var a = t.contextType;
    typeof a == "object" && a !== null ? l.context = Tt(a) : (a = gt(t) ? Hn : rt.current, l.context = Er(e, a)), l.state = e.memoizedState, a = t.getDerivedStateFromProps, typeof a == "function" && (To(e, t, a, n), l.state = e.memoizedState), typeof t.getDerivedStateFromProps == "function" || typeof l.getSnapshotBeforeUpdate == "function" || typeof l.UNSAFE_componentWillMount != "function" && typeof l.componentWillMount != "function" || (t = l.state, typeof l.componentWillMount == "function" && l.componentWillMount(), typeof l.UNSAFE_componentWillMount == "function" && l.UNSAFE_componentWillMount(), t !== l.state && ss.enqueueReplaceState(l, l.state, null), Ua(e, n, l, r), l.state = e.memoizedState), typeof l.componentDidMount == "function" && (e.flags |= 4194308);
  }
  function Rr(e, t) {
    try {
      var n = "", r = t;
      do
        n += dp(r), r = r.return;
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
  function Vs(e, t, n) {
    return {
      value: e,
      source: null,
      stack: n ?? null,
      digest: t ?? null
    };
  }
  function Mo(e, t) {
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
    n = en(-1, n), n.tag = 3, n.payload = {
      element: null
    };
    var r = t.value;
    return n.callback = function() {
      Wa || (Wa = true, Bo = r), Mo(e, t);
    }, n;
  }
  function Df(e, t, n) {
    n = en(-1, n), n.tag = 3;
    var r = e.type.getDerivedStateFromError;
    if (typeof r == "function") {
      var l = t.value;
      n.payload = function() {
        return r(l);
      }, n.callback = function() {
        Mo(e, t);
      };
    }
    var a = e.stateNode;
    return a !== null && typeof a.componentDidCatch == "function" && (n.callback = function() {
      Mo(e, t), typeof r != "function" && (kn === null ? kn = /* @__PURE__ */ new Set([
        this
      ]) : kn.add(this));
      var s = t.stack;
      this.componentDidCatch(t.value, {
        componentStack: s !== null ? s : ""
      });
    }), n;
  }
  function Xu(e, t, n) {
    var r = e.pingCache;
    if (r === null) {
      r = e.pingCache = new zh();
      var l = /* @__PURE__ */ new Set();
      r.set(t, l);
    } else l = r.get(t), l === void 0 && (l = /* @__PURE__ */ new Set(), r.set(t, l));
    l.has(n) || (l.add(n), e = Zh.bind(null, e, t, n), t.then(e, e));
  }
  function Zu(e) {
    do {
      var t;
      if ((t = e.tag === 13) && (t = e.memoizedState, t = t !== null ? t.dehydrated !== null : true), t) return e;
      e = e.return;
    } while (e !== null);
    return null;
  }
  function qu(e, t, n, r, l) {
    return e.mode & 1 ? (e.flags |= 65536, e.lanes = l, e) : (e === t ? e.flags |= 65536 : (e.flags |= 128, n.flags |= 131072, n.flags &= -52805, n.tag === 1 && (n.alternate === null ? n.tag = 17 : (t = en(-1, 1), t.tag = 2, Sn(n, t, 1))), n.lanes |= 1), e);
  }
  var Uh = an.ReactCurrentOwner, pt = false;
  function at(e, t, n, r) {
    t.child = e === null ? of(t, null, n, r) : _r(t, e.child, n, r);
  }
  function ec(e, t, n, r, l) {
    n = n.render;
    var a = t.ref;
    return kr(t, l), r = Ri(e, t, n, r, a, l), n = Ti(), e !== null && !pt ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, ln(e, t, l)) : (Ce && n && vi(t), t.flags |= 1, at(e, t, r, l), t.child);
  }
  function tc(e, t, n, r, l) {
    if (e === null) {
      var a = n.type;
      return typeof a == "function" && !$i(a) && a.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0 ? (t.tag = 15, t.type = a, Lf(e, t, a, r, l)) : (e = wa(n.type, null, r, t, t.mode, l), e.ref = t.ref, e.return = t, t.child = e);
    }
    if (a = e.child, !(e.lanes & l)) {
      var s = a.memoizedProps;
      if (n = n.compare, n = n !== null ? n : gl, n(s, r) && e.ref === t.ref) return ln(e, t, l);
    }
    return t.flags |= 1, e = jn(a, r), e.ref = t.ref, e.return = t, t.child = e;
  }
  function Lf(e, t, n, r, l) {
    if (e !== null) {
      var a = e.memoizedProps;
      if (gl(a, r) && e.ref === t.ref) if (pt = false, t.pendingProps = r = a, (e.lanes & l) !== 0) e.flags & 131072 && (pt = true);
      else return t.lanes = e.lanes, ln(e, t, l);
    }
    return Do(e, t, n, r, l);
  }
  function Of(e, t, n) {
    var r = t.pendingProps, l = r.children, a = e !== null ? e.memoizedState : null;
    if (r.mode === "hidden") if (!(t.mode & 1)) t.memoizedState = {
      baseLanes: 0,
      cachePool: null,
      transitions: null
    }, we(vr, xt), xt |= n;
    else {
      if (!(n & 1073741824)) return e = a !== null ? a.baseLanes | n : n, t.lanes = t.childLanes = 1073741824, t.memoizedState = {
        baseLanes: e,
        cachePool: null,
        transitions: null
      }, t.updateQueue = null, we(vr, xt), xt |= e, null;
      t.memoizedState = {
        baseLanes: 0,
        cachePool: null,
        transitions: null
      }, r = a !== null ? a.baseLanes : n, we(vr, xt), xt |= r;
    }
    else a !== null ? (r = a.baseLanes | n, t.memoizedState = null) : r = n, we(vr, xt), xt |= r;
    return at(e, t, l, n), t.child;
  }
  function Af(e, t) {
    var n = t.ref;
    (e === null && n !== null || e !== null && e.ref !== n) && (t.flags |= 512, t.flags |= 2097152);
  }
  function Do(e, t, n, r, l) {
    var a = gt(n) ? Hn : rt.current;
    return a = Er(t, a), kr(t, l), n = Ri(e, t, n, r, a, l), r = Ti(), e !== null && !pt ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, ln(e, t, l)) : (Ce && r && vi(t), t.flags |= 1, at(e, t, n, l), t.child);
  }
  function nc(e, t, n, r, l) {
    if (gt(n)) {
      var a = true;
      La(t);
    } else a = false;
    if (kr(t, l), t.stateNode === null) va(e, t), Pf(t, n, r), Po(t, n, r, l), r = true;
    else if (e === null) {
      var s = t.stateNode, u = t.memoizedProps;
      s.props = u;
      var i = s.context, c = n.contextType;
      typeof c == "object" && c !== null ? c = Tt(c) : (c = gt(n) ? Hn : rt.current, c = Er(t, c));
      var m = n.getDerivedStateFromProps, d = typeof m == "function" || typeof s.getSnapshotBeforeUpdate == "function";
      d || typeof s.UNSAFE_componentWillReceiveProps != "function" && typeof s.componentWillReceiveProps != "function" || (u !== r || i !== c) && Ju(t, s, r, c), dn = false;
      var p = t.memoizedState;
      s.state = p, Ua(t, r, s, l), i = t.memoizedState, u !== r || p !== i || ht.current || dn ? (typeof m == "function" && (To(t, n, m, r), i = t.memoizedState), (u = dn || Yu(t, n, u, r, p, i, c)) ? (d || typeof s.UNSAFE_componentWillMount != "function" && typeof s.componentWillMount != "function" || (typeof s.componentWillMount == "function" && s.componentWillMount(), typeof s.UNSAFE_componentWillMount == "function" && s.UNSAFE_componentWillMount()), typeof s.componentDidMount == "function" && (t.flags |= 4194308)) : (typeof s.componentDidMount == "function" && (t.flags |= 4194308), t.memoizedProps = r, t.memoizedState = i), s.props = r, s.state = i, s.context = c, r = u) : (typeof s.componentDidMount == "function" && (t.flags |= 4194308), r = false);
    } else {
      s = t.stateNode, cf(e, t), u = t.memoizedProps, c = t.type === t.elementType ? u : Ot(t.type, u), s.props = c, d = t.pendingProps, p = s.context, i = n.contextType, typeof i == "object" && i !== null ? i = Tt(i) : (i = gt(n) ? Hn : rt.current, i = Er(t, i));
      var S = n.getDerivedStateFromProps;
      (m = typeof S == "function" || typeof s.getSnapshotBeforeUpdate == "function") || typeof s.UNSAFE_componentWillReceiveProps != "function" && typeof s.componentWillReceiveProps != "function" || (u !== d || p !== i) && Ju(t, s, r, i), dn = false, p = t.memoizedState, s.state = p, Ua(t, r, s, l);
      var w = t.memoizedState;
      u !== d || p !== w || ht.current || dn ? (typeof S == "function" && (To(t, n, S, r), w = t.memoizedState), (c = dn || Yu(t, n, c, r, p, w, i) || false) ? (m || typeof s.UNSAFE_componentWillUpdate != "function" && typeof s.componentWillUpdate != "function" || (typeof s.componentWillUpdate == "function" && s.componentWillUpdate(r, w, i), typeof s.UNSAFE_componentWillUpdate == "function" && s.UNSAFE_componentWillUpdate(r, w, i)), typeof s.componentDidUpdate == "function" && (t.flags |= 4), typeof s.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof s.componentDidUpdate != "function" || u === e.memoizedProps && p === e.memoizedState || (t.flags |= 4), typeof s.getSnapshotBeforeUpdate != "function" || u === e.memoizedProps && p === e.memoizedState || (t.flags |= 1024), t.memoizedProps = r, t.memoizedState = w), s.props = r, s.state = w, s.context = i, r = c) : (typeof s.componentDidUpdate != "function" || u === e.memoizedProps && p === e.memoizedState || (t.flags |= 4), typeof s.getSnapshotBeforeUpdate != "function" || u === e.memoizedProps && p === e.memoizedState || (t.flags |= 1024), r = false);
    }
    return Lo(e, t, n, r, a, l);
  }
  function Lo(e, t, n, r, l, a) {
    Af(e, t);
    var s = (t.flags & 128) !== 0;
    if (!r && !s) return l && Fu(t, n, false), ln(e, t, a);
    r = t.stateNode, Uh.current = t;
    var u = s && typeof n.getDerivedStateFromError != "function" ? null : r.render();
    return t.flags |= 1, e !== null && s ? (t.child = _r(t, e.child, null, a), t.child = _r(t, null, u, a)) : at(e, t, u, a), t.memoizedState = r.state, l && Fu(t, n, true), t.child;
  }
  function If(e) {
    var t = e.stateNode;
    t.pendingContext ? $u(e, t.pendingContext, t.pendingContext !== t.context) : t.context && $u(e, t.context, false), Ei(e, t.containerInfo);
  }
  function rc(e, t, n, r, l) {
    return Cr(), yi(l), t.flags |= 256, at(e, t, n, r), t.child;
  }
  var Oo = {
    dehydrated: null,
    treeContext: null,
    retryLane: 0
  };
  function Ao(e) {
    return {
      baseLanes: e,
      cachePool: null,
      transitions: null
    };
  }
  function zf(e, t, n) {
    var r = t.pendingProps, l = _e.current, a = false, s = (t.flags & 128) !== 0, u;
    if ((u = s) || (u = e !== null && e.memoizedState === null ? false : (l & 2) !== 0), u ? (a = true, t.flags &= -129) : (e === null || e.memoizedState !== null) && (l |= 1), we(_e, l & 1), e === null) return bo(t), e = t.memoizedState, e !== null && (e = e.dehydrated, e !== null) ? (t.mode & 1 ? e.data === "$!" ? t.lanes = 8 : t.lanes = 1073741824 : t.lanes = 1, null) : (s = r.children, e = r.fallback, a ? (r = t.mode, a = t.child, s = {
      mode: "hidden",
      children: s
    }, !(r & 1) && a !== null ? (a.childLanes = 0, a.pendingProps = s) : a = us(s, r, 0, null), e = Wn(e, r, n, null), a.return = t, e.return = t, a.sibling = e, t.child = a, t.child.memoizedState = Ao(n), t.memoizedState = Oo, e) : Di(t, s));
    if (l = e.memoizedState, l !== null && (u = l.dehydrated, u !== null)) return $h(e, t, s, r, u, l, n);
    if (a) {
      a = r.fallback, s = t.mode, l = e.child, u = l.sibling;
      var i = {
        mode: "hidden",
        children: r.children
      };
      return !(s & 1) && t.child !== l ? (r = t.child, r.childLanes = 0, r.pendingProps = i, t.deletions = null) : (r = jn(l, i), r.subtreeFlags = l.subtreeFlags & 14680064), u !== null ? a = jn(u, a) : (a = Wn(a, s, n, null), a.flags |= 2), a.return = t, r.return = t, r.sibling = a, t.child = r, r = a, a = t.child, s = e.child.memoizedState, s = s === null ? Ao(n) : {
        baseLanes: s.baseLanes | n,
        cachePool: null,
        transitions: s.transitions
      }, a.memoizedState = s, a.childLanes = e.childLanes & ~n, t.memoizedState = Oo, r;
    }
    return a = e.child, e = a.sibling, r = jn(a, {
      mode: "visible",
      children: r.children
    }), !(t.mode & 1) && (r.lanes = n), r.return = t, r.sibling = null, e !== null && (n = t.deletions, n === null ? (t.deletions = [
      e
    ], t.flags |= 16) : n.push(e)), t.child = r, t.memoizedState = null, r;
  }
  function Di(e, t) {
    return t = us({
      mode: "visible",
      children: t
    }, e.mode, 0, null), t.return = e, e.child = t;
  }
  function ta(e, t, n, r) {
    return r !== null && yi(r), _r(t, e.child, null, n), e = Di(t, t.pendingProps.children), e.flags |= 2, t.memoizedState = null, e;
  }
  function $h(e, t, n, r, l, a, s) {
    if (n) return t.flags & 256 ? (t.flags &= -257, r = Vs(Error(P(422))), ta(e, t, s, r)) : t.memoizedState !== null ? (t.child = e.child, t.flags |= 128, null) : (a = r.fallback, l = t.mode, r = us({
      mode: "visible",
      children: r.children
    }, l, 0, null), a = Wn(a, l, s, null), a.flags |= 2, r.return = t, a.return = t, r.sibling = a, t.child = r, t.mode & 1 && _r(t, e.child, null, s), t.child.memoizedState = Ao(s), t.memoizedState = Oo, a);
    if (!(t.mode & 1)) return ta(e, t, s, null);
    if (l.data === "$!") {
      if (r = l.nextSibling && l.nextSibling.dataset, r) var u = r.dgst;
      return r = u, a = Error(P(419)), r = Vs(a, r, void 0), ta(e, t, s, r);
    }
    if (u = (s & e.childLanes) !== 0, pt || u) {
      if (r = He, r !== null) {
        switch (s & -s) {
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
        l = l & (r.suspendedLanes | s) ? 0 : l, l !== 0 && l !== a.retryLane && (a.retryLane = l, rn(e, l), $t(r, e, l, -1));
      }
      return Ui(), r = Vs(Error(P(421))), ta(e, t, s, r);
    }
    return l.data === "$?" ? (t.flags |= 128, t.child = e.child, t = qh.bind(null, e), l._reactRetry = t, null) : (e = a.treeContext, wt = wn(l.nextSibling), St = t, Ce = true, zt = null, e !== null && (Ct[_t++] = Zt, Ct[_t++] = qt, Ct[_t++] = Qn, Zt = e.id, qt = e.overflow, Qn = t), t = Di(t, r.children), t.flags |= 4096, t);
  }
  function lc(e, t, n) {
    e.lanes |= t;
    var r = e.alternate;
    r !== null && (r.lanes |= t), Ro(e.return, t, n);
  }
  function Ws(e, t, n, r, l) {
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
    if (at(e, t, r.children, n), r = _e.current, r & 2) r = r & 1 | 2, t.flags |= 128;
    else {
      if (e !== null && e.flags & 128) e: for (e = t.child; e !== null; ) {
        if (e.tag === 13) e.memoizedState !== null && lc(e, n, t);
        else if (e.tag === 19) lc(e, n, t);
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
    if (we(_e, r), !(t.mode & 1)) t.memoizedState = null;
    else switch (l) {
      case "forwards":
        for (n = t.child, l = null; n !== null; ) e = n.alternate, e !== null && $a(e) === null && (l = n), n = n.sibling;
        n = l, n === null ? (l = t.child, t.child = null) : (l = n.sibling, n.sibling = null), Ws(t, false, l, n, a);
        break;
      case "backwards":
        for (n = null, l = t.child, t.child = null; l !== null; ) {
          if (e = l.alternate, e !== null && $a(e) === null) {
            t.child = l;
            break;
          }
          e = l.sibling, l.sibling = n, n = l, l = e;
        }
        Ws(t, true, n, null, a);
        break;
      case "together":
        Ws(t, false, null, null, void 0);
        break;
      default:
        t.memoizedState = null;
    }
    return t.child;
  }
  function va(e, t) {
    !(t.mode & 1) && e !== null && (e.alternate = null, t.alternate = null, t.flags |= 2);
  }
  function ln(e, t, n) {
    if (e !== null && (t.dependencies = e.dependencies), Gn |= t.lanes, !(n & t.childLanes)) return null;
    if (e !== null && t.child !== e.child) throw Error(P(153));
    if (t.child !== null) {
      for (e = t.child, n = jn(e, e.pendingProps), t.child = n, n.return = t; e.sibling !== null; ) e = e.sibling, n = n.sibling = jn(e, e.pendingProps), n.return = t;
      n.sibling = null;
    }
    return t.child;
  }
  function Fh(e, t, n) {
    switch (t.tag) {
      case 3:
        If(t), Cr();
        break;
      case 5:
        df(t);
        break;
      case 1:
        gt(t.type) && La(t);
        break;
      case 4:
        Ei(t, t.stateNode.containerInfo);
        break;
      case 10:
        var r = t.type._context, l = t.memoizedProps.value;
        we(Ia, r._currentValue), r._currentValue = l;
        break;
      case 13:
        if (r = t.memoizedState, r !== null) return r.dehydrated !== null ? (we(_e, _e.current & 1), t.flags |= 128, null) : n & t.child.childLanes ? zf(e, t, n) : (we(_e, _e.current & 1), e = ln(e, t, n), e !== null ? e.sibling : null);
        we(_e, _e.current & 1);
        break;
      case 19:
        if (r = (n & t.childLanes) !== 0, e.flags & 128) {
          if (r) return Uf(e, t, n);
          t.flags |= 128;
        }
        if (l = t.memoizedState, l !== null && (l.rendering = null, l.tail = null, l.lastEffect = null), we(_e, _e.current), r) break;
        return null;
      case 22:
      case 23:
        return t.lanes = 0, Of(e, t, n);
    }
    return ln(e, t, n);
  }
  var $f, Io, Ff, Bf;
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
  Io = function() {
  };
  Ff = function(e, t, n, r) {
    var l = e.memoizedProps;
    if (l !== r) {
      e = t.stateNode, Fn(Kt.current);
      var a = null;
      switch (n) {
        case "input":
          l = lo(e, l), r = lo(e, r), a = [];
          break;
        case "select":
          l = Re({}, l, {
            value: void 0
          }), r = Re({}, r, {
            value: void 0
          }), a = [];
          break;
        case "textarea":
          l = oo(e, l), r = oo(e, r), a = [];
          break;
        default:
          typeof l.onClick != "function" && typeof r.onClick == "function" && (e.onclick = Ma);
      }
      uo(n, r);
      var s;
      n = null;
      for (c in l) if (!r.hasOwnProperty(c) && l.hasOwnProperty(c) && l[c] != null) if (c === "style") {
        var u = l[c];
        for (s in u) u.hasOwnProperty(s) && (n || (n = {}), n[s] = "");
      } else c !== "dangerouslySetInnerHTML" && c !== "children" && c !== "suppressContentEditableWarning" && c !== "suppressHydrationWarning" && c !== "autoFocus" && (ul.hasOwnProperty(c) ? a || (a = []) : (a = a || []).push(c, null));
      for (c in r) {
        var i = r[c];
        if (u = l == null ? void 0 : l[c], r.hasOwnProperty(c) && i !== u && (i != null || u != null)) if (c === "style") if (u) {
          for (s in u) !u.hasOwnProperty(s) || i && i.hasOwnProperty(s) || (n || (n = {}), n[s] = "");
          for (s in i) i.hasOwnProperty(s) && u[s] !== i[s] && (n || (n = {}), n[s] = i[s]);
        } else n || (a || (a = []), a.push(c, n)), n = i;
        else c === "dangerouslySetInnerHTML" ? (i = i ? i.__html : void 0, u = u ? u.__html : void 0, i != null && u !== i && (a = a || []).push(c, i)) : c === "children" ? typeof i != "string" && typeof i != "number" || (a = a || []).push(c, "" + i) : c !== "suppressContentEditableWarning" && c !== "suppressHydrationWarning" && (ul.hasOwnProperty(c) ? (i != null && c === "onScroll" && Ne("scroll", e), a || u === i || (a = [])) : (a = a || []).push(c, i));
      }
      n && (a = a || []).push("style", n);
      var c = a;
      (t.updateQueue = c) && (t.flags |= 4);
    }
  };
  Bf = function(e, t, n, r) {
    n !== r && (t.flags |= 4);
  };
  function Vr(e, t) {
    if (!Ce) switch (e.tailMode) {
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
  function tt(e) {
    var t = e.alternate !== null && e.alternate.child === e.child, n = 0, r = 0;
    if (t) for (var l = e.child; l !== null; ) n |= l.lanes | l.childLanes, r |= l.subtreeFlags & 14680064, r |= l.flags & 14680064, l.return = e, l = l.sibling;
    else for (l = e.child; l !== null; ) n |= l.lanes | l.childLanes, r |= l.subtreeFlags, r |= l.flags, l.return = e, l = l.sibling;
    return e.subtreeFlags |= r, e.childLanes = n, t;
  }
  function Bh(e, t, n) {
    var r = t.pendingProps;
    switch (xi(t), t.tag) {
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
        return tt(t), null;
      case 1:
        return gt(t.type) && Da(), tt(t), null;
      case 3:
        return r = t.stateNode, br(), je(ht), je(rt), _i(), r.pendingContext && (r.context = r.pendingContext, r.pendingContext = null), (e === null || e.child === null) && (ql(t) ? t.flags |= 4 : e === null || e.memoizedState.isDehydrated && !(t.flags & 256) || (t.flags |= 1024, zt !== null && (Ho(zt), zt = null))), Io(e, t), tt(t), null;
      case 5:
        Ci(t);
        var l = Fn(Sl.current);
        if (n = t.type, e !== null && t.stateNode != null) Ff(e, t, n, r, l), e.ref !== t.ref && (t.flags |= 512, t.flags |= 2097152);
        else {
          if (!r) {
            if (t.stateNode === null) throw Error(P(166));
            return tt(t), null;
          }
          if (e = Fn(Kt.current), ql(t)) {
            r = t.stateNode, n = t.type;
            var a = t.memoizedProps;
            switch (r[Ht] = t, r[yl] = a, e = (t.mode & 1) !== 0, n) {
              case "dialog":
                Ne("cancel", r), Ne("close", r);
                break;
              case "iframe":
              case "object":
              case "embed":
                Ne("load", r);
                break;
              case "video":
              case "audio":
                for (l = 0; l < Xr.length; l++) Ne(Xr[l], r);
                break;
              case "source":
                Ne("error", r);
                break;
              case "img":
              case "image":
              case "link":
                Ne("error", r), Ne("load", r);
                break;
              case "details":
                Ne("toggle", r);
                break;
              case "input":
                mu(r, a), Ne("invalid", r);
                break;
              case "select":
                r._wrapperState = {
                  wasMultiple: !!a.multiple
                }, Ne("invalid", r);
                break;
              case "textarea":
                hu(r, a), Ne("invalid", r);
            }
            uo(n, a), l = null;
            for (var s in a) if (a.hasOwnProperty(s)) {
              var u = a[s];
              s === "children" ? typeof u == "string" ? r.textContent !== u && (a.suppressHydrationWarning !== true && Zl(r.textContent, u, e), l = [
                "children",
                u
              ]) : typeof u == "number" && r.textContent !== "" + u && (a.suppressHydrationWarning !== true && Zl(r.textContent, u, e), l = [
                "children",
                "" + u
              ]) : ul.hasOwnProperty(s) && u != null && s === "onScroll" && Ne("scroll", r);
            }
            switch (n) {
              case "input":
                Wl(r), pu(r, a, true);
                break;
              case "textarea":
                Wl(r), gu(r);
                break;
              case "select":
              case "option":
                break;
              default:
                typeof a.onClick == "function" && (r.onclick = Ma);
            }
            r = l, t.updateQueue = r, r !== null && (t.flags |= 4);
          } else {
            s = l.nodeType === 9 ? l : l.ownerDocument, e === "http://www.w3.org/1999/xhtml" && (e = hd(n)), e === "http://www.w3.org/1999/xhtml" ? n === "script" ? (e = s.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild)) : typeof r.is == "string" ? e = s.createElement(n, {
              is: r.is
            }) : (e = s.createElement(n), n === "select" && (s = e, r.multiple ? s.multiple = true : r.size && (s.size = r.size))) : e = s.createElementNS(e, n), e[Ht] = t, e[yl] = r, $f(e, t, false, false), t.stateNode = e;
            e: {
              switch (s = co(n, r), n) {
                case "dialog":
                  Ne("cancel", e), Ne("close", e), l = r;
                  break;
                case "iframe":
                case "object":
                case "embed":
                  Ne("load", e), l = r;
                  break;
                case "video":
                case "audio":
                  for (l = 0; l < Xr.length; l++) Ne(Xr[l], e);
                  l = r;
                  break;
                case "source":
                  Ne("error", e), l = r;
                  break;
                case "img":
                case "image":
                case "link":
                  Ne("error", e), Ne("load", e), l = r;
                  break;
                case "details":
                  Ne("toggle", e), l = r;
                  break;
                case "input":
                  mu(e, r), l = lo(e, r), Ne("invalid", e);
                  break;
                case "option":
                  l = r;
                  break;
                case "select":
                  e._wrapperState = {
                    wasMultiple: !!r.multiple
                  }, l = Re({}, r, {
                    value: void 0
                  }), Ne("invalid", e);
                  break;
                case "textarea":
                  hu(e, r), l = oo(e, r), Ne("invalid", e);
                  break;
                default:
                  l = r;
              }
              uo(n, l), u = l;
              for (a in u) if (u.hasOwnProperty(a)) {
                var i = u[a];
                a === "style" ? xd(e, i) : a === "dangerouslySetInnerHTML" ? (i = i ? i.__html : void 0, i != null && gd(e, i)) : a === "children" ? typeof i == "string" ? (n !== "textarea" || i !== "") && cl(e, i) : typeof i == "number" && cl(e, "" + i) : a !== "suppressContentEditableWarning" && a !== "suppressHydrationWarning" && a !== "autoFocus" && (ul.hasOwnProperty(a) ? i != null && a === "onScroll" && Ne("scroll", e) : i != null && ri(e, a, i, s));
              }
              switch (n) {
                case "input":
                  Wl(e), pu(e, r, false);
                  break;
                case "textarea":
                  Wl(e), gu(e);
                  break;
                case "option":
                  r.value != null && e.setAttribute("value", "" + Cn(r.value));
                  break;
                case "select":
                  e.multiple = !!r.multiple, a = r.value, a != null ? xr(e, !!r.multiple, a, false) : r.defaultValue != null && xr(e, !!r.multiple, r.defaultValue, true);
                  break;
                default:
                  typeof l.onClick == "function" && (e.onclick = Ma);
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
        return tt(t), null;
      case 6:
        if (e && t.stateNode != null) Bf(e, t, e.memoizedProps, r);
        else {
          if (typeof r != "string" && t.stateNode === null) throw Error(P(166));
          if (n = Fn(Sl.current), Fn(Kt.current), ql(t)) {
            if (r = t.stateNode, n = t.memoizedProps, r[Ht] = t, (a = r.nodeValue !== n) && (e = St, e !== null)) switch (e.tag) {
              case 3:
                Zl(r.nodeValue, n, (e.mode & 1) !== 0);
                break;
              case 5:
                e.memoizedProps.suppressHydrationWarning !== true && Zl(r.nodeValue, n, (e.mode & 1) !== 0);
            }
            a && (t.flags |= 4);
          } else r = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(r), r[Ht] = t, t.stateNode = r;
        }
        return tt(t), null;
      case 13:
        if (je(_e), r = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
          if (Ce && wt !== null && t.mode & 1 && !(t.flags & 128)) af(), Cr(), t.flags |= 98560, a = false;
          else if (a = ql(t), r !== null && r.dehydrated !== null) {
            if (e === null) {
              if (!a) throw Error(P(318));
              if (a = t.memoizedState, a = a !== null ? a.dehydrated : null, !a) throw Error(P(317));
              a[Ht] = t;
            } else Cr(), !(t.flags & 128) && (t.memoizedState = null), t.flags |= 4;
            tt(t), a = false;
          } else zt !== null && (Ho(zt), zt = null), a = true;
          if (!a) return t.flags & 65536 ? t : null;
        }
        return t.flags & 128 ? (t.lanes = n, t) : (r = r !== null, r !== (e !== null && e.memoizedState !== null) && r && (t.child.flags |= 8192, t.mode & 1 && (e === null || _e.current & 1 ? $e === 0 && ($e = 3) : Ui())), t.updateQueue !== null && (t.flags |= 4), tt(t), null);
      case 4:
        return br(), Io(e, t), e === null && vl(t.stateNode.containerInfo), tt(t), null;
      case 10:
        return ki(t.type._context), tt(t), null;
      case 17:
        return gt(t.type) && Da(), tt(t), null;
      case 19:
        if (je(_e), a = t.memoizedState, a === null) return tt(t), null;
        if (r = (t.flags & 128) !== 0, s = a.rendering, s === null) if (r) Vr(a, false);
        else {
          if ($e !== 0 || e !== null && e.flags & 128) for (e = t.child; e !== null; ) {
            if (s = $a(e), s !== null) {
              for (t.flags |= 128, Vr(a, false), r = s.updateQueue, r !== null && (t.updateQueue = r, t.flags |= 4), t.subtreeFlags = 0, r = n, n = t.child; n !== null; ) a = n, e = r, a.flags &= 14680066, s = a.alternate, s === null ? (a.childLanes = 0, a.lanes = e, a.child = null, a.subtreeFlags = 0, a.memoizedProps = null, a.memoizedState = null, a.updateQueue = null, a.dependencies = null, a.stateNode = null) : (a.childLanes = s.childLanes, a.lanes = s.lanes, a.child = s.child, a.subtreeFlags = 0, a.deletions = null, a.memoizedProps = s.memoizedProps, a.memoizedState = s.memoizedState, a.updateQueue = s.updateQueue, a.type = s.type, e = s.dependencies, a.dependencies = e === null ? null : {
                lanes: e.lanes,
                firstContext: e.firstContext
              }), n = n.sibling;
              return we(_e, _e.current & 1 | 2), t.child;
            }
            e = e.sibling;
          }
          a.tail !== null && Le() > Tr && (t.flags |= 128, r = true, Vr(a, false), t.lanes = 4194304);
        }
        else {
          if (!r) if (e = $a(s), e !== null) {
            if (t.flags |= 128, r = true, n = e.updateQueue, n !== null && (t.updateQueue = n, t.flags |= 4), Vr(a, true), a.tail === null && a.tailMode === "hidden" && !s.alternate && !Ce) return tt(t), null;
          } else 2 * Le() - a.renderingStartTime > Tr && n !== 1073741824 && (t.flags |= 128, r = true, Vr(a, false), t.lanes = 4194304);
          a.isBackwards ? (s.sibling = t.child, t.child = s) : (n = a.last, n !== null ? n.sibling = s : t.child = s, a.last = s);
        }
        return a.tail !== null ? (t = a.tail, a.rendering = t, a.tail = t.sibling, a.renderingStartTime = Le(), t.sibling = null, n = _e.current, we(_e, r ? n & 1 | 2 : n & 1), t) : (tt(t), null);
      case 22:
      case 23:
        return zi(), r = t.memoizedState !== null, e !== null && e.memoizedState !== null !== r && (t.flags |= 8192), r && t.mode & 1 ? xt & 1073741824 && (tt(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : tt(t), null;
      case 24:
        return null;
      case 25:
        return null;
    }
    throw Error(P(156, t.tag));
  }
  function Vh(e, t) {
    switch (xi(t), t.tag) {
      case 1:
        return gt(t.type) && Da(), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 3:
        return br(), je(ht), je(rt), _i(), e = t.flags, e & 65536 && !(e & 128) ? (t.flags = e & -65537 | 128, t) : null;
      case 5:
        return Ci(t), null;
      case 13:
        if (je(_e), e = t.memoizedState, e !== null && e.dehydrated !== null) {
          if (t.alternate === null) throw Error(P(340));
          Cr();
        }
        return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 19:
        return je(_e), null;
      case 4:
        return br(), null;
      case 10:
        return ki(t.type._context), null;
      case 22:
      case 23:
        return zi(), null;
      case 24:
        return null;
      default:
        return null;
    }
  }
  var na = false, nt = false, Wh = typeof WeakSet == "function" ? WeakSet : Set, O = null;
  function gr(e, t) {
    var n = e.ref;
    if (n !== null) if (typeof n == "function") try {
      n(null);
    } catch (r) {
      De(e, t, r);
    }
    else n.current = null;
  }
  function zo(e, t, n) {
    try {
      n();
    } catch (r) {
      De(e, t, r);
    }
  }
  var ac = false;
  function Hh(e, t) {
    if (So = Ra, e = Qd(), gi(e)) {
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
          var s = 0, u = -1, i = -1, c = 0, m = 0, d = e, p = null;
          t: for (; ; ) {
            for (var S; d !== n || l !== 0 && d.nodeType !== 3 || (u = s + l), d !== a || r !== 0 && d.nodeType !== 3 || (i = s + r), d.nodeType === 3 && (s += d.nodeValue.length), (S = d.firstChild) !== null; ) p = d, d = S;
            for (; ; ) {
              if (d === e) break t;
              if (p === n && ++c === l && (u = s), p === a && ++m === r && (i = s), (S = d.nextSibling) !== null) break;
              d = p, p = d.parentNode;
            }
            d = S;
          }
          n = u === -1 || i === -1 ? null : {
            start: u,
            end: i
          };
        } else n = null;
      }
      n = n || {
        start: 0,
        end: 0
      };
    } else n = null;
    for (ko = {
      focusedElem: e,
      selectionRange: n
    }, Ra = false, O = t; O !== null; ) if (t = O, e = t.child, (t.subtreeFlags & 1028) !== 0 && e !== null) e.return = t, O = e;
    else for (; O !== null; ) {
      t = O;
      try {
        var w = t.alternate;
        if (t.flags & 1024) switch (t.tag) {
          case 0:
          case 11:
          case 15:
            break;
          case 1:
            if (w !== null) {
              var y = w.memoizedProps, b = w.memoizedState, h = t.stateNode, f = h.getSnapshotBeforeUpdate(t.elementType === t.type ? y : Ot(t.type, y), b);
              h.__reactInternalSnapshotBeforeUpdate = f;
            }
            break;
          case 3:
            var v = t.stateNode.containerInfo;
            v.nodeType === 1 ? v.textContent = "" : v.nodeType === 9 && v.documentElement && v.removeChild(v.documentElement);
            break;
          case 5:
          case 6:
          case 4:
          case 17:
            break;
          default:
            throw Error(P(163));
        }
      } catch (E) {
        De(t, t.return, E);
      }
      if (e = t.sibling, e !== null) {
        e.return = t.return, O = e;
        break;
      }
      O = t.return;
    }
    return w = ac, ac = false, w;
  }
  function al(e, t, n) {
    var r = t.updateQueue;
    if (r = r !== null ? r.lastEffect : null, r !== null) {
      var l = r = r.next;
      do {
        if ((l.tag & e) === e) {
          var a = l.destroy;
          l.destroy = void 0, a !== void 0 && zo(t, n, a);
        }
        l = l.next;
      } while (l !== r);
    }
  }
  function os(e, t) {
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
  function Uo(e) {
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
  function Vf(e) {
    var t = e.alternate;
    t !== null && (e.alternate = null, Vf(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && (delete t[Ht], delete t[yl], delete t[Eo], delete t[_h], delete t[bh])), e.stateNode = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
  }
  function Wf(e) {
    return e.tag === 5 || e.tag === 3 || e.tag === 4;
  }
  function sc(e) {
    e: for (; ; ) {
      for (; e.sibling === null; ) {
        if (e.return === null || Wf(e.return)) return null;
        e = e.return;
      }
      for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
        if (e.flags & 2 || e.child === null || e.tag === 4) continue e;
        e.child.return = e, e = e.child;
      }
      if (!(e.flags & 2)) return e.stateNode;
    }
  }
  function $o(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.nodeType === 8 ? n.parentNode.insertBefore(e, t) : n.insertBefore(e, t) : (n.nodeType === 8 ? (t = n.parentNode, t.insertBefore(e, n)) : (t = n, t.appendChild(e)), n = n._reactRootContainer, n != null || t.onclick !== null || (t.onclick = Ma));
    else if (r !== 4 && (e = e.child, e !== null)) for ($o(e, t, n), e = e.sibling; e !== null; ) $o(e, t, n), e = e.sibling;
  }
  function Fo(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.insertBefore(e, t) : n.appendChild(e);
    else if (r !== 4 && (e = e.child, e !== null)) for (Fo(e, t, n), e = e.sibling; e !== null; ) Fo(e, t, n), e = e.sibling;
  }
  var Ye = null, At = false;
  function on(e, t, n) {
    for (n = n.child; n !== null; ) Hf(e, t, n), n = n.sibling;
  }
  function Hf(e, t, n) {
    if (Qt && typeof Qt.onCommitFiberUnmount == "function") try {
      Qt.onCommitFiberUnmount(qa, n);
    } catch {
    }
    switch (n.tag) {
      case 5:
        nt || gr(n, t);
      case 6:
        var r = Ye, l = At;
        Ye = null, on(e, t, n), Ye = r, At = l, Ye !== null && (At ? (e = Ye, n = n.stateNode, e.nodeType === 8 ? e.parentNode.removeChild(n) : e.removeChild(n)) : Ye.removeChild(n.stateNode));
        break;
      case 18:
        Ye !== null && (At ? (e = Ye, n = n.stateNode, e.nodeType === 8 ? Is(e.parentNode, n) : e.nodeType === 1 && Is(e, n), pl(e)) : Is(Ye, n.stateNode));
        break;
      case 4:
        r = Ye, l = At, Ye = n.stateNode.containerInfo, At = true, on(e, t, n), Ye = r, At = l;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        if (!nt && (r = n.updateQueue, r !== null && (r = r.lastEffect, r !== null))) {
          l = r = r.next;
          do {
            var a = l, s = a.destroy;
            a = a.tag, s !== void 0 && (a & 2 || a & 4) && zo(n, t, s), l = l.next;
          } while (l !== r);
        }
        on(e, t, n);
        break;
      case 1:
        if (!nt && (gr(n, t), r = n.stateNode, typeof r.componentWillUnmount == "function")) try {
          r.props = n.memoizedProps, r.state = n.memoizedState, r.componentWillUnmount();
        } catch (u) {
          De(n, t, u);
        }
        on(e, t, n);
        break;
      case 21:
        on(e, t, n);
        break;
      case 22:
        n.mode & 1 ? (nt = (r = nt) || n.memoizedState !== null, on(e, t, n), nt = r) : on(e, t, n);
        break;
      default:
        on(e, t, n);
    }
  }
  function oc(e) {
    var t = e.updateQueue;
    if (t !== null) {
      e.updateQueue = null;
      var n = e.stateNode;
      n === null && (n = e.stateNode = new Wh()), t.forEach(function(r) {
        var l = eg.bind(null, e, r);
        n.has(r) || (n.add(r), r.then(l, l));
      });
    }
  }
  function Dt(e, t) {
    var n = t.deletions;
    if (n !== null) for (var r = 0; r < n.length; r++) {
      var l = n[r];
      try {
        var a = e, s = t, u = s;
        e: for (; u !== null; ) {
          switch (u.tag) {
            case 5:
              Ye = u.stateNode, At = false;
              break e;
            case 3:
              Ye = u.stateNode.containerInfo, At = true;
              break e;
            case 4:
              Ye = u.stateNode.containerInfo, At = true;
              break e;
          }
          u = u.return;
        }
        if (Ye === null) throw Error(P(160));
        Hf(a, s, l), Ye = null, At = false;
        var i = l.alternate;
        i !== null && (i.return = null), l.return = null;
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
        if (Dt(t, e), Vt(e), r & 4) {
          try {
            al(3, e, e.return), os(3, e);
          } catch (y) {
            De(e, e.return, y);
          }
          try {
            al(5, e, e.return);
          } catch (y) {
            De(e, e.return, y);
          }
        }
        break;
      case 1:
        Dt(t, e), Vt(e), r & 512 && n !== null && gr(n, n.return);
        break;
      case 5:
        if (Dt(t, e), Vt(e), r & 512 && n !== null && gr(n, n.return), e.flags & 32) {
          var l = e.stateNode;
          try {
            cl(l, "");
          } catch (y) {
            De(e, e.return, y);
          }
        }
        if (r & 4 && (l = e.stateNode, l != null)) {
          var a = e.memoizedProps, s = n !== null ? n.memoizedProps : a, u = e.type, i = e.updateQueue;
          if (e.updateQueue = null, i !== null) try {
            u === "input" && a.type === "radio" && a.name != null && md(l, a), co(u, s);
            var c = co(u, a);
            for (s = 0; s < i.length; s += 2) {
              var m = i[s], d = i[s + 1];
              m === "style" ? xd(l, d) : m === "dangerouslySetInnerHTML" ? gd(l, d) : m === "children" ? cl(l, d) : ri(l, m, d, c);
            }
            switch (u) {
              case "input":
                ao(l, a);
                break;
              case "textarea":
                pd(l, a);
                break;
              case "select":
                var p = l._wrapperState.wasMultiple;
                l._wrapperState.wasMultiple = !!a.multiple;
                var S = a.value;
                S != null ? xr(l, !!a.multiple, S, false) : p !== !!a.multiple && (a.defaultValue != null ? xr(l, !!a.multiple, a.defaultValue, true) : xr(l, !!a.multiple, a.multiple ? [] : "", false));
            }
            l[yl] = a;
          } catch (y) {
            De(e, e.return, y);
          }
        }
        break;
      case 6:
        if (Dt(t, e), Vt(e), r & 4) {
          if (e.stateNode === null) throw Error(P(162));
          l = e.stateNode, a = e.memoizedProps;
          try {
            l.nodeValue = a;
          } catch (y) {
            De(e, e.return, y);
          }
        }
        break;
      case 3:
        if (Dt(t, e), Vt(e), r & 4 && n !== null && n.memoizedState.isDehydrated) try {
          pl(t.containerInfo);
        } catch (y) {
          De(e, e.return, y);
        }
        break;
      case 4:
        Dt(t, e), Vt(e);
        break;
      case 13:
        Dt(t, e), Vt(e), l = e.child, l.flags & 8192 && (a = l.memoizedState !== null, l.stateNode.isHidden = a, !a || l.alternate !== null && l.alternate.memoizedState !== null || (Ai = Le())), r & 4 && oc(e);
        break;
      case 22:
        if (m = n !== null && n.memoizedState !== null, e.mode & 1 ? (nt = (c = nt) || m, Dt(t, e), nt = c) : Dt(t, e), Vt(e), r & 8192) {
          if (c = e.memoizedState !== null, (e.stateNode.isHidden = c) && !m && e.mode & 1) for (O = e, m = e.child; m !== null; ) {
            for (d = O = m; O !== null; ) {
              switch (p = O, S = p.child, p.tag) {
                case 0:
                case 11:
                case 14:
                case 15:
                  al(4, p, p.return);
                  break;
                case 1:
                  gr(p, p.return);
                  var w = p.stateNode;
                  if (typeof w.componentWillUnmount == "function") {
                    r = p, n = p.return;
                    try {
                      t = r, w.props = t.memoizedProps, w.state = t.memoizedState, w.componentWillUnmount();
                    } catch (y) {
                      De(r, n, y);
                    }
                  }
                  break;
                case 5:
                  gr(p, p.return);
                  break;
                case 22:
                  if (p.memoizedState !== null) {
                    uc(d);
                    continue;
                  }
              }
              S !== null ? (S.return = p, O = S) : uc(d);
            }
            m = m.sibling;
          }
          e: for (m = null, d = e; ; ) {
            if (d.tag === 5) {
              if (m === null) {
                m = d;
                try {
                  l = d.stateNode, c ? (a = l.style, typeof a.setProperty == "function" ? a.setProperty("display", "none", "important") : a.display = "none") : (u = d.stateNode, i = d.memoizedProps.style, s = i != null && i.hasOwnProperty("display") ? i.display : null, u.style.display = vd("display", s));
                } catch (y) {
                  De(e, e.return, y);
                }
              }
            } else if (d.tag === 6) {
              if (m === null) try {
                d.stateNode.nodeValue = c ? "" : d.memoizedProps;
              } catch (y) {
                De(e, e.return, y);
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
        Dt(t, e), Vt(e), r & 4 && oc(e);
        break;
      case 21:
        break;
      default:
        Dt(t, e), Vt(e);
    }
  }
  function Vt(e) {
    var t = e.flags;
    if (t & 2) {
      try {
        e: {
          for (var n = e.return; n !== null; ) {
            if (Wf(n)) {
              var r = n;
              break e;
            }
            n = n.return;
          }
          throw Error(P(160));
        }
        switch (r.tag) {
          case 5:
            var l = r.stateNode;
            r.flags & 32 && (cl(l, ""), r.flags &= -33);
            var a = sc(e);
            Fo(e, a, l);
            break;
          case 3:
          case 4:
            var s = r.stateNode.containerInfo, u = sc(e);
            $o(e, u, s);
            break;
          default:
            throw Error(P(161));
        }
      } catch (i) {
        De(e, e.return, i);
      }
      e.flags &= -3;
    }
    t & 4096 && (e.flags &= -4097);
  }
  function Qh(e, t, n) {
    O = e, Kf(e);
  }
  function Kf(e, t, n) {
    for (var r = (e.mode & 1) !== 0; O !== null; ) {
      var l = O, a = l.child;
      if (l.tag === 22 && r) {
        var s = l.memoizedState !== null || na;
        if (!s) {
          var u = l.alternate, i = u !== null && u.memoizedState !== null || nt;
          u = na;
          var c = nt;
          if (na = s, (nt = i) && !c) for (O = l; O !== null; ) s = O, i = s.child, s.tag === 22 && s.memoizedState !== null ? cc(l) : i !== null ? (i.return = s, O = i) : cc(l);
          for (; a !== null; ) O = a, Kf(a), a = a.sibling;
          O = l, na = u, nt = c;
        }
        ic(e);
      } else l.subtreeFlags & 8772 && a !== null ? (a.return = l, O = a) : ic(e);
    }
  }
  function ic(e) {
    for (; O !== null; ) {
      var t = O;
      if (t.flags & 8772) {
        var n = t.alternate;
        try {
          if (t.flags & 8772) switch (t.tag) {
            case 0:
            case 11:
            case 15:
              nt || os(5, t);
              break;
            case 1:
              var r = t.stateNode;
              if (t.flags & 4 && !nt) if (n === null) r.componentDidMount();
              else {
                var l = t.elementType === t.type ? n.memoizedProps : Ot(t.type, n.memoizedProps);
                r.componentDidUpdate(l, n.memoizedState, r.__reactInternalSnapshotBeforeUpdate);
              }
              var a = t.updateQueue;
              a !== null && Qu(t, a, r);
              break;
            case 3:
              var s = t.updateQueue;
              if (s !== null) {
                if (n = null, t.child !== null) switch (t.child.tag) {
                  case 5:
                    n = t.child.stateNode;
                    break;
                  case 1:
                    n = t.child.stateNode;
                }
                Qu(t, s, n);
              }
              break;
            case 5:
              var u = t.stateNode;
              if (n === null && t.flags & 4) {
                n = u;
                var i = t.memoizedProps;
                switch (t.type) {
                  case "button":
                  case "input":
                  case "select":
                  case "textarea":
                    i.autoFocus && n.focus();
                    break;
                  case "img":
                    i.src && (n.src = i.src);
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
              throw Error(P(163));
          }
          nt || t.flags & 512 && Uo(t);
        } catch (p) {
          De(t, t.return, p);
        }
      }
      if (t === e) {
        O = null;
        break;
      }
      if (n = t.sibling, n !== null) {
        n.return = t.return, O = n;
        break;
      }
      O = t.return;
    }
  }
  function uc(e) {
    for (; O !== null; ) {
      var t = O;
      if (t === e) {
        O = null;
        break;
      }
      var n = t.sibling;
      if (n !== null) {
        n.return = t.return, O = n;
        break;
      }
      O = t.return;
    }
  }
  function cc(e) {
    for (; O !== null; ) {
      var t = O;
      try {
        switch (t.tag) {
          case 0:
          case 11:
          case 15:
            var n = t.return;
            try {
              os(4, t);
            } catch (i) {
              De(t, n, i);
            }
            break;
          case 1:
            var r = t.stateNode;
            if (typeof r.componentDidMount == "function") {
              var l = t.return;
              try {
                r.componentDidMount();
              } catch (i) {
                De(t, l, i);
              }
            }
            var a = t.return;
            try {
              Uo(t);
            } catch (i) {
              De(t, a, i);
            }
            break;
          case 5:
            var s = t.return;
            try {
              Uo(t);
            } catch (i) {
              De(t, s, i);
            }
        }
      } catch (i) {
        De(t, t.return, i);
      }
      if (t === e) {
        O = null;
        break;
      }
      var u = t.sibling;
      if (u !== null) {
        u.return = t.return, O = u;
        break;
      }
      O = t.return;
    }
  }
  var Kh = Math.ceil, Va = an.ReactCurrentDispatcher, Li = an.ReactCurrentOwner, Rt = an.ReactCurrentBatchConfig, re = 0, He = null, ze = null, Je = 0, xt = 0, vr = Rn(0), $e = 0, El = null, Gn = 0, is = 0, Oi = 0, sl = null, mt = null, Ai = 0, Tr = 1 / 0, Jt = null, Wa = false, Bo = null, kn = null, ra = false, hn = null, Ha = 0, ol = 0, Vo = null, xa = -1, ya = 0;
  function st() {
    return re & 6 ? Le() : xa !== -1 ? xa : xa = Le();
  }
  function Nn(e) {
    return e.mode & 1 ? re & 2 && Je !== 0 ? Je & -Je : Th.transition !== null ? (ya === 0 && (ya = Td()), ya) : (e = ce, e !== 0 || (e = window.event, e = e === void 0 ? 16 : Id(e.type)), e) : 1;
  }
  function $t(e, t, n, r) {
    if (50 < ol) throw ol = 0, Vo = null, Error(P(185));
    Rl(e, n, r), (!(re & 2) || e !== He) && (e === He && (!(re & 2) && (is |= n), $e === 4 && mn(e, Je)), vt(e, r), n === 1 && re === 0 && !(t.mode & 1) && (Tr = Le() + 500, ls && Tn()));
  }
  function vt(e, t) {
    var n = e.callbackNode;
    Tp(e, t);
    var r = ba(e, e === He ? Je : 0);
    if (r === 0) n !== null && yu(n), e.callbackNode = null, e.callbackPriority = 0;
    else if (t = r & -r, e.callbackPriority !== t) {
      if (n != null && yu(n), t === 1) e.tag === 0 ? Rh(dc.bind(null, e)) : nf(dc.bind(null, e)), Eh(function() {
        !(re & 6) && Tn();
      }), n = null;
      else {
        switch (Pd(r)) {
          case 1:
            n = ii;
            break;
          case 4:
            n = bd;
            break;
          case 16:
            n = _a;
            break;
          case 536870912:
            n = Rd;
            break;
          default:
            n = _a;
        }
        n = tm(n, Gf.bind(null, e));
      }
      e.callbackPriority = t, e.callbackNode = n;
    }
  }
  function Gf(e, t) {
    if (xa = -1, ya = 0, re & 6) throw Error(P(327));
    var n = e.callbackNode;
    if (Nr() && e.callbackNode !== n) return null;
    var r = ba(e, e === He ? Je : 0);
    if (r === 0) return null;
    if (r & 30 || r & e.expiredLanes || t) t = Qa(e, r);
    else {
      t = r;
      var l = re;
      re |= 2;
      var a = Jf();
      (He !== e || Je !== t) && (Jt = null, Tr = Le() + 500, Vn(e, t));
      do
        try {
          Jh();
          break;
        } catch (u) {
          Yf(e, u);
        }
      while (true);
      Si(), Va.current = a, re = l, ze !== null ? t = 0 : (He = null, Je = 0, t = $e);
    }
    if (t !== 0) {
      if (t === 2 && (l = go(e), l !== 0 && (r = l, t = Wo(e, l))), t === 1) throw n = El, Vn(e, 0), mn(e, r), vt(e, Le()), n;
      if (t === 6) mn(e, r);
      else {
        if (l = e.current.alternate, !(r & 30) && !Gh(l) && (t = Qa(e, r), t === 2 && (a = go(e), a !== 0 && (r = a, t = Wo(e, a))), t === 1)) throw n = El, Vn(e, 0), mn(e, r), vt(e, Le()), n;
        switch (e.finishedWork = l, e.finishedLanes = r, t) {
          case 0:
          case 1:
            throw Error(P(345));
          case 2:
            An(e, mt, Jt);
            break;
          case 3:
            if (mn(e, r), (r & 130023424) === r && (t = Ai + 500 - Le(), 10 < t)) {
              if (ba(e, 0) !== 0) break;
              if (l = e.suspendedLanes, (l & r) !== r) {
                st(), e.pingedLanes |= e.suspendedLanes & l;
                break;
              }
              e.timeoutHandle = jo(An.bind(null, e, mt, Jt), t);
              break;
            }
            An(e, mt, Jt);
            break;
          case 4:
            if (mn(e, r), (r & 4194240) === r) break;
            for (t = e.eventTimes, l = -1; 0 < r; ) {
              var s = 31 - Ut(r);
              a = 1 << s, s = t[s], s > l && (l = s), r &= ~a;
            }
            if (r = l, r = Le() - r, r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * Kh(r / 1960)) - r, 10 < r) {
              e.timeoutHandle = jo(An.bind(null, e, mt, Jt), r);
              break;
            }
            An(e, mt, Jt);
            break;
          case 5:
            An(e, mt, Jt);
            break;
          default:
            throw Error(P(329));
        }
      }
    }
    return vt(e, Le()), e.callbackNode === n ? Gf.bind(null, e) : null;
  }
  function Wo(e, t) {
    var n = sl;
    return e.current.memoizedState.isDehydrated && (Vn(e, t).flags |= 256), e = Qa(e, t), e !== 2 && (t = mt, mt = n, t !== null && Ho(t)), e;
  }
  function Ho(e) {
    mt === null ? mt = e : mt.push.apply(mt, e);
  }
  function Gh(e) {
    for (var t = e; ; ) {
      if (t.flags & 16384) {
        var n = t.updateQueue;
        if (n !== null && (n = n.stores, n !== null)) for (var r = 0; r < n.length; r++) {
          var l = n[r], a = l.getSnapshot;
          l = l.value;
          try {
            if (!Ft(a(), l)) return false;
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
  function mn(e, t) {
    for (t &= ~Oi, t &= ~is, e.suspendedLanes |= t, e.pingedLanes &= ~t, e = e.expirationTimes; 0 < t; ) {
      var n = 31 - Ut(t), r = 1 << n;
      e[n] = -1, t &= ~r;
    }
  }
  function dc(e) {
    if (re & 6) throw Error(P(327));
    Nr();
    var t = ba(e, 0);
    if (!(t & 1)) return vt(e, Le()), null;
    var n = Qa(e, t);
    if (e.tag !== 0 && n === 2) {
      var r = go(e);
      r !== 0 && (t = r, n = Wo(e, r));
    }
    if (n === 1) throw n = El, Vn(e, 0), mn(e, t), vt(e, Le()), n;
    if (n === 6) throw Error(P(345));
    return e.finishedWork = e.current.alternate, e.finishedLanes = t, An(e, mt, Jt), vt(e, Le()), null;
  }
  function Ii(e, t) {
    var n = re;
    re |= 1;
    try {
      return e(t);
    } finally {
      re = n, re === 0 && (Tr = Le() + 500, ls && Tn());
    }
  }
  function Yn(e) {
    hn !== null && hn.tag === 0 && !(re & 6) && Nr();
    var t = re;
    re |= 1;
    var n = Rt.transition, r = ce;
    try {
      if (Rt.transition = null, ce = 1, e) return e();
    } finally {
      ce = r, Rt.transition = n, re = t, !(re & 6) && Tn();
    }
  }
  function zi() {
    xt = vr.current, je(vr);
  }
  function Vn(e, t) {
    e.finishedWork = null, e.finishedLanes = 0;
    var n = e.timeoutHandle;
    if (n !== -1 && (e.timeoutHandle = -1, jh(n)), ze !== null) for (n = ze.return; n !== null; ) {
      var r = n;
      switch (xi(r), r.tag) {
        case 1:
          r = r.type.childContextTypes, r != null && Da();
          break;
        case 3:
          br(), je(ht), je(rt), _i();
          break;
        case 5:
          Ci(r);
          break;
        case 4:
          br();
          break;
        case 13:
          je(_e);
          break;
        case 19:
          je(_e);
          break;
        case 10:
          ki(r.type._context);
          break;
        case 22:
        case 23:
          zi();
      }
      n = n.return;
    }
    if (He = e, ze = e = jn(e.current, null), Je = xt = t, $e = 0, El = null, Oi = is = Gn = 0, mt = sl = null, $n !== null) {
      for (t = 0; t < $n.length; t++) if (n = $n[t], r = n.interleaved, r !== null) {
        n.interleaved = null;
        var l = r.next, a = n.pending;
        if (a !== null) {
          var s = a.next;
          a.next = l, r.next = s;
        }
        n.pending = r;
      }
      $n = null;
    }
    return e;
  }
  function Yf(e, t) {
    do {
      var n = ze;
      try {
        if (Si(), ha.current = Ba, Fa) {
          for (var r = be.memoizedState; r !== null; ) {
            var l = r.queue;
            l !== null && (l.pending = null), r = r.next;
          }
          Fa = false;
        }
        if (Kn = 0, We = Ue = be = null, ll = false, kl = 0, Li.current = null, n === null || n.return === null) {
          $e = 1, El = t, ze = null;
          break;
        }
        e: {
          var a = e, s = n.return, u = n, i = t;
          if (t = Je, u.flags |= 32768, i !== null && typeof i == "object" && typeof i.then == "function") {
            var c = i, m = u, d = m.tag;
            if (!(m.mode & 1) && (d === 0 || d === 11 || d === 15)) {
              var p = m.alternate;
              p ? (m.updateQueue = p.updateQueue, m.memoizedState = p.memoizedState, m.lanes = p.lanes) : (m.updateQueue = null, m.memoizedState = null);
            }
            var S = Zu(s);
            if (S !== null) {
              S.flags &= -257, qu(S, s, u, a, t), S.mode & 1 && Xu(a, c, t), t = S, i = c;
              var w = t.updateQueue;
              if (w === null) {
                var y = /* @__PURE__ */ new Set();
                y.add(i), t.updateQueue = y;
              } else w.add(i);
              break e;
            } else {
              if (!(t & 1)) {
                Xu(a, c, t), Ui();
                break e;
              }
              i = Error(P(426));
            }
          } else if (Ce && u.mode & 1) {
            var b = Zu(s);
            if (b !== null) {
              !(b.flags & 65536) && (b.flags |= 256), qu(b, s, u, a, t), yi(Rr(i, u));
              break e;
            }
          }
          a = i = Rr(i, u), $e !== 4 && ($e = 2), sl === null ? sl = [
            a
          ] : sl.push(a), a = s;
          do {
            switch (a.tag) {
              case 3:
                a.flags |= 65536, t &= -t, a.lanes |= t;
                var h = Mf(a, i, t);
                Hu(a, h);
                break e;
              case 1:
                u = i;
                var f = a.type, v = a.stateNode;
                if (!(a.flags & 128) && (typeof f.getDerivedStateFromError == "function" || v !== null && typeof v.componentDidCatch == "function" && (kn === null || !kn.has(v)))) {
                  a.flags |= 65536, t &= -t, a.lanes |= t;
                  var E = Df(a, u, t);
                  Hu(a, E);
                  break e;
                }
            }
            a = a.return;
          } while (a !== null);
        }
        Zf(n);
      } catch (_) {
        t = _, ze === n && n !== null && (ze = n = n.return);
        continue;
      }
      break;
    } while (true);
  }
  function Jf() {
    var e = Va.current;
    return Va.current = Ba, e === null ? Ba : e;
  }
  function Ui() {
    ($e === 0 || $e === 3 || $e === 2) && ($e = 4), He === null || !(Gn & 268435455) && !(is & 268435455) || mn(He, Je);
  }
  function Qa(e, t) {
    var n = re;
    re |= 2;
    var r = Jf();
    (He !== e || Je !== t) && (Jt = null, Vn(e, t));
    do
      try {
        Yh();
        break;
      } catch (l) {
        Yf(e, l);
      }
    while (true);
    if (Si(), re = n, Va.current = r, ze !== null) throw Error(P(261));
    return He = null, Je = 0, $e;
  }
  function Yh() {
    for (; ze !== null; ) Xf(ze);
  }
  function Jh() {
    for (; ze !== null && !Sp(); ) Xf(ze);
  }
  function Xf(e) {
    var t = em(e.alternate, e, xt);
    e.memoizedProps = e.pendingProps, t === null ? Zf(e) : ze = t, Li.current = null;
  }
  function Zf(e) {
    var t = e;
    do {
      var n = t.alternate;
      if (e = t.return, t.flags & 32768) {
        if (n = Vh(n, t), n !== null) {
          n.flags &= 32767, ze = n;
          return;
        }
        if (e !== null) e.flags |= 32768, e.subtreeFlags = 0, e.deletions = null;
        else {
          $e = 6, ze = null;
          return;
        }
      } else if (n = Bh(n, t, xt), n !== null) {
        ze = n;
        return;
      }
      if (t = t.sibling, t !== null) {
        ze = t;
        return;
      }
      ze = t = e;
    } while (t !== null);
    $e === 0 && ($e = 5);
  }
  function An(e, t, n) {
    var r = ce, l = Rt.transition;
    try {
      Rt.transition = null, ce = 1, Xh(e, t, n, r);
    } finally {
      Rt.transition = l, ce = r;
    }
    return null;
  }
  function Xh(e, t, n, r) {
    do
      Nr();
    while (hn !== null);
    if (re & 6) throw Error(P(327));
    n = e.finishedWork;
    var l = e.finishedLanes;
    if (n === null) return null;
    if (e.finishedWork = null, e.finishedLanes = 0, n === e.current) throw Error(P(177));
    e.callbackNode = null, e.callbackPriority = 0;
    var a = n.lanes | n.childLanes;
    if (Pp(e, a), e === He && (ze = He = null, Je = 0), !(n.subtreeFlags & 2064) && !(n.flags & 2064) || ra || (ra = true, tm(_a, function() {
      return Nr(), null;
    })), a = (n.flags & 15990) !== 0, n.subtreeFlags & 15990 || a) {
      a = Rt.transition, Rt.transition = null;
      var s = ce;
      ce = 1;
      var u = re;
      re |= 4, Li.current = null, Hh(e, n), Qf(n, e), vh(ko), Ra = !!So, ko = So = null, e.current = n, Qh(n), kp(), re = u, ce = s, Rt.transition = a;
    } else e.current = n;
    if (ra && (ra = false, hn = e, Ha = l), a = e.pendingLanes, a === 0 && (kn = null), Ep(n.stateNode), vt(e, Le()), t !== null) for (r = e.onRecoverableError, n = 0; n < t.length; n++) l = t[n], r(l.value, {
      componentStack: l.stack,
      digest: l.digest
    });
    if (Wa) throw Wa = false, e = Bo, Bo = null, e;
    return Ha & 1 && e.tag !== 0 && Nr(), a = e.pendingLanes, a & 1 ? e === Vo ? ol++ : (ol = 0, Vo = e) : ol = 0, Tn(), null;
  }
  function Nr() {
    if (hn !== null) {
      var e = Pd(Ha), t = Rt.transition, n = ce;
      try {
        if (Rt.transition = null, ce = 16 > e ? 16 : e, hn === null) var r = false;
        else {
          if (e = hn, hn = null, Ha = 0, re & 6) throw Error(P(331));
          var l = re;
          for (re |= 4, O = e.current; O !== null; ) {
            var a = O, s = a.child;
            if (O.flags & 16) {
              var u = a.deletions;
              if (u !== null) {
                for (var i = 0; i < u.length; i++) {
                  var c = u[i];
                  for (O = c; O !== null; ) {
                    var m = O;
                    switch (m.tag) {
                      case 0:
                      case 11:
                      case 15:
                        al(8, m, a);
                    }
                    var d = m.child;
                    if (d !== null) d.return = m, O = d;
                    else for (; O !== null; ) {
                      m = O;
                      var p = m.sibling, S = m.return;
                      if (Vf(m), m === c) {
                        O = null;
                        break;
                      }
                      if (p !== null) {
                        p.return = S, O = p;
                        break;
                      }
                      O = S;
                    }
                  }
                }
                var w = a.alternate;
                if (w !== null) {
                  var y = w.child;
                  if (y !== null) {
                    w.child = null;
                    do {
                      var b = y.sibling;
                      y.sibling = null, y = b;
                    } while (y !== null);
                  }
                }
                O = a;
              }
            }
            if (a.subtreeFlags & 2064 && s !== null) s.return = a, O = s;
            else e: for (; O !== null; ) {
              if (a = O, a.flags & 2048) switch (a.tag) {
                case 0:
                case 11:
                case 15:
                  al(9, a, a.return);
              }
              var h = a.sibling;
              if (h !== null) {
                h.return = a.return, O = h;
                break e;
              }
              O = a.return;
            }
          }
          var f = e.current;
          for (O = f; O !== null; ) {
            s = O;
            var v = s.child;
            if (s.subtreeFlags & 2064 && v !== null) v.return = s, O = v;
            else e: for (s = f; O !== null; ) {
              if (u = O, u.flags & 2048) try {
                switch (u.tag) {
                  case 0:
                  case 11:
                  case 15:
                    os(9, u);
                }
              } catch (_) {
                De(u, u.return, _);
              }
              if (u === s) {
                O = null;
                break e;
              }
              var E = u.sibling;
              if (E !== null) {
                E.return = u.return, O = E;
                break e;
              }
              O = u.return;
            }
          }
          if (re = l, Tn(), Qt && typeof Qt.onPostCommitFiberRoot == "function") try {
            Qt.onPostCommitFiberRoot(qa, e);
          } catch {
          }
          r = true;
        }
        return r;
      } finally {
        ce = n, Rt.transition = t;
      }
    }
    return false;
  }
  function fc(e, t, n) {
    t = Rr(n, t), t = Mf(e, t, 1), e = Sn(e, t, 1), t = st(), e !== null && (Rl(e, 1, t), vt(e, t));
  }
  function De(e, t, n) {
    if (e.tag === 3) fc(e, e, n);
    else for (; t !== null; ) {
      if (t.tag === 3) {
        fc(t, e, n);
        break;
      } else if (t.tag === 1) {
        var r = t.stateNode;
        if (typeof t.type.getDerivedStateFromError == "function" || typeof r.componentDidCatch == "function" && (kn === null || !kn.has(r))) {
          e = Rr(n, e), e = Df(t, e, 1), t = Sn(t, e, 1), e = st(), t !== null && (Rl(t, 1, e), vt(t, e));
          break;
        }
      }
      t = t.return;
    }
  }
  function Zh(e, t, n) {
    var r = e.pingCache;
    r !== null && r.delete(t), t = st(), e.pingedLanes |= e.suspendedLanes & n, He === e && (Je & n) === n && ($e === 4 || $e === 3 && (Je & 130023424) === Je && 500 > Le() - Ai ? Vn(e, 0) : Oi |= n), vt(e, t);
  }
  function qf(e, t) {
    t === 0 && (e.mode & 1 ? (t = Kl, Kl <<= 1, !(Kl & 130023424) && (Kl = 4194304)) : t = 1);
    var n = st();
    e = rn(e, t), e !== null && (Rl(e, t, n), vt(e, n));
  }
  function qh(e) {
    var t = e.memoizedState, n = 0;
    t !== null && (n = t.retryLane), qf(e, n);
  }
  function eg(e, t) {
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
        throw Error(P(314));
    }
    r !== null && r.delete(t), qf(e, n);
  }
  var em;
  em = function(e, t, n) {
    if (e !== null) if (e.memoizedProps !== t.pendingProps || ht.current) pt = true;
    else {
      if (!(e.lanes & n) && !(t.flags & 128)) return pt = false, Fh(e, t, n);
      pt = !!(e.flags & 131072);
    }
    else pt = false, Ce && t.flags & 1048576 && rf(t, Aa, t.index);
    switch (t.lanes = 0, t.tag) {
      case 2:
        var r = t.type;
        va(e, t), e = t.pendingProps;
        var l = Er(t, rt.current);
        kr(t, n), l = Ri(null, t, r, e, l, n);
        var a = Ti();
        return t.flags |= 1, typeof l == "object" && l !== null && typeof l.render == "function" && l.$$typeof === void 0 ? (t.tag = 1, t.memoizedState = null, t.updateQueue = null, gt(r) ? (a = true, La(t)) : a = false, t.memoizedState = l.state !== null && l.state !== void 0 ? l.state : null, ji(t), l.updater = ss, t.stateNode = l, l._reactInternals = t, Po(t, r, e, n), t = Lo(null, t, r, true, a, n)) : (t.tag = 0, Ce && a && vi(t), at(null, t, l, n), t = t.child), t;
      case 16:
        r = t.elementType;
        e: {
          switch (va(e, t), e = t.pendingProps, l = r._init, r = l(r._payload), t.type = r, l = t.tag = ng(r), e = Ot(r, e), l) {
            case 0:
              t = Do(null, t, r, e, n);
              break e;
            case 1:
              t = nc(null, t, r, e, n);
              break e;
            case 11:
              t = ec(null, t, r, e, n);
              break e;
            case 14:
              t = tc(null, t, r, Ot(r.type, e), n);
              break e;
          }
          throw Error(P(306, r, ""));
        }
        return t;
      case 0:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Ot(r, l), Do(e, t, r, l, n);
      case 1:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Ot(r, l), nc(e, t, r, l, n);
      case 3:
        e: {
          if (If(t), e === null) throw Error(P(387));
          r = t.pendingProps, a = t.memoizedState, l = a.element, cf(e, t), Ua(t, r, null, n);
          var s = t.memoizedState;
          if (r = s.element, a.isDehydrated) if (a = {
            element: r,
            isDehydrated: false,
            cache: s.cache,
            pendingSuspenseBoundaries: s.pendingSuspenseBoundaries,
            transitions: s.transitions
          }, t.updateQueue.baseState = a, t.memoizedState = a, t.flags & 256) {
            l = Rr(Error(P(423)), t), t = rc(e, t, r, n, l);
            break e;
          } else if (r !== l) {
            l = Rr(Error(P(424)), t), t = rc(e, t, r, n, l);
            break e;
          } else for (wt = wn(t.stateNode.containerInfo.firstChild), St = t, Ce = true, zt = null, n = of(t, null, r, n), t.child = n; n; ) n.flags = n.flags & -3 | 4096, n = n.sibling;
          else {
            if (Cr(), r === l) {
              t = ln(e, t, n);
              break e;
            }
            at(e, t, r, n);
          }
          t = t.child;
        }
        return t;
      case 5:
        return df(t), e === null && bo(t), r = t.type, l = t.pendingProps, a = e !== null ? e.memoizedProps : null, s = l.children, No(r, l) ? s = null : a !== null && No(r, a) && (t.flags |= 32), Af(e, t), at(e, t, s, n), t.child;
      case 6:
        return e === null && bo(t), null;
      case 13:
        return zf(e, t, n);
      case 4:
        return Ei(t, t.stateNode.containerInfo), r = t.pendingProps, e === null ? t.child = _r(t, null, r, n) : at(e, t, r, n), t.child;
      case 11:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Ot(r, l), ec(e, t, r, l, n);
      case 7:
        return at(e, t, t.pendingProps, n), t.child;
      case 8:
        return at(e, t, t.pendingProps.children, n), t.child;
      case 12:
        return at(e, t, t.pendingProps.children, n), t.child;
      case 10:
        e: {
          if (r = t.type._context, l = t.pendingProps, a = t.memoizedProps, s = l.value, we(Ia, r._currentValue), r._currentValue = s, a !== null) if (Ft(a.value, s)) {
            if (a.children === l.children && !ht.current) {
              t = ln(e, t, n);
              break e;
            }
          } else for (a = t.child, a !== null && (a.return = t); a !== null; ) {
            var u = a.dependencies;
            if (u !== null) {
              s = a.child;
              for (var i = u.firstContext; i !== null; ) {
                if (i.context === r) {
                  if (a.tag === 1) {
                    i = en(-1, n & -n), i.tag = 2;
                    var c = a.updateQueue;
                    if (c !== null) {
                      c = c.shared;
                      var m = c.pending;
                      m === null ? i.next = i : (i.next = m.next, m.next = i), c.pending = i;
                    }
                  }
                  a.lanes |= n, i = a.alternate, i !== null && (i.lanes |= n), Ro(a.return, n, t), u.lanes |= n;
                  break;
                }
                i = i.next;
              }
            } else if (a.tag === 10) s = a.type === t.type ? null : a.child;
            else if (a.tag === 18) {
              if (s = a.return, s === null) throw Error(P(341));
              s.lanes |= n, u = s.alternate, u !== null && (u.lanes |= n), Ro(s, n, t), s = a.sibling;
            } else s = a.child;
            if (s !== null) s.return = a;
            else for (s = a; s !== null; ) {
              if (s === t) {
                s = null;
                break;
              }
              if (a = s.sibling, a !== null) {
                a.return = s.return, s = a;
                break;
              }
              s = s.return;
            }
            a = s;
          }
          at(e, t, l.children, n), t = t.child;
        }
        return t;
      case 9:
        return l = t.type, r = t.pendingProps.children, kr(t, n), l = Tt(l), r = r(l), t.flags |= 1, at(e, t, r, n), t.child;
      case 14:
        return r = t.type, l = Ot(r, t.pendingProps), l = Ot(r.type, l), tc(e, t, r, l, n);
      case 15:
        return Lf(e, t, t.type, t.pendingProps, n);
      case 17:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Ot(r, l), va(e, t), t.tag = 1, gt(r) ? (e = true, La(t)) : e = false, kr(t, n), Pf(t, r, l), Po(t, r, l, n), Lo(null, t, r, true, e, n);
      case 19:
        return Uf(e, t, n);
      case 22:
        return Of(e, t, n);
    }
    throw Error(P(156, t.tag));
  };
  function tm(e, t) {
    return _d(e, t);
  }
  function tg(e, t, n, r) {
    this.tag = e, this.key = n, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = r, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function bt(e, t, n, r) {
    return new tg(e, t, n, r);
  }
  function $i(e) {
    return e = e.prototype, !(!e || !e.isReactComponent);
  }
  function ng(e) {
    if (typeof e == "function") return $i(e) ? 1 : 0;
    if (e != null) {
      if (e = e.$$typeof, e === ai) return 11;
      if (e === si) return 14;
    }
    return 2;
  }
  function jn(e, t) {
    var n = e.alternate;
    return n === null ? (n = bt(e.tag, t, e.key, e.mode), n.elementType = e.elementType, n.type = e.type, n.stateNode = e.stateNode, n.alternate = e, e.alternate = n) : (n.pendingProps = t, n.type = e.type, n.flags = 0, n.subtreeFlags = 0, n.deletions = null), n.flags = e.flags & 14680064, n.childLanes = e.childLanes, n.lanes = e.lanes, n.child = e.child, n.memoizedProps = e.memoizedProps, n.memoizedState = e.memoizedState, n.updateQueue = e.updateQueue, t = e.dependencies, n.dependencies = t === null ? null : {
      lanes: t.lanes,
      firstContext: t.firstContext
    }, n.sibling = e.sibling, n.index = e.index, n.ref = e.ref, n;
  }
  function wa(e, t, n, r, l, a) {
    var s = 2;
    if (r = e, typeof e == "function") $i(e) && (s = 1);
    else if (typeof e == "string") s = 5;
    else e: switch (e) {
      case or:
        return Wn(n.children, l, a, t);
      case li:
        s = 8, l |= 8;
        break;
      case eo:
        return e = bt(12, n, t, l | 2), e.elementType = eo, e.lanes = a, e;
      case to:
        return e = bt(13, n, t, l), e.elementType = to, e.lanes = a, e;
      case no:
        return e = bt(19, n, t, l), e.elementType = no, e.lanes = a, e;
      case cd:
        return us(n, l, a, t);
      default:
        if (typeof e == "object" && e !== null) switch (e.$$typeof) {
          case id:
            s = 10;
            break e;
          case ud:
            s = 9;
            break e;
          case ai:
            s = 11;
            break e;
          case si:
            s = 14;
            break e;
          case cn:
            s = 16, r = null;
            break e;
        }
        throw Error(P(130, e == null ? e : typeof e, ""));
    }
    return t = bt(s, n, t, l), t.elementType = e, t.type = r, t.lanes = a, t;
  }
  function Wn(e, t, n, r) {
    return e = bt(7, e, r, t), e.lanes = n, e;
  }
  function us(e, t, n, r) {
    return e = bt(22, e, r, t), e.elementType = cd, e.lanes = n, e.stateNode = {
      isHidden: false
    }, e;
  }
  function Hs(e, t, n) {
    return e = bt(6, e, null, t), e.lanes = n, e;
  }
  function Qs(e, t, n) {
    return t = bt(4, e.children !== null ? e.children : [], e.key, t), t.lanes = n, t.stateNode = {
      containerInfo: e.containerInfo,
      pendingChildren: null,
      implementation: e.implementation
    }, t;
  }
  function rg(e, t, n, r, l) {
    this.tag = t, this.containerInfo = e, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = Cs(0), this.expirationTimes = Cs(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = Cs(0), this.identifierPrefix = r, this.onRecoverableError = l, this.mutableSourceEagerHydrationData = null;
  }
  function Fi(e, t, n, r, l, a, s, u, i) {
    return e = new rg(e, t, n, u, i), t === 1 ? (t = 1, a === true && (t |= 8)) : t = 0, a = bt(3, null, null, t), e.current = a, a.stateNode = e, a.memoizedState = {
      element: r,
      isDehydrated: n,
      cache: null,
      transitions: null,
      pendingSuspenseBoundaries: null
    }, ji(a), e;
  }
  function lg(e, t, n) {
    var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: sr,
      key: r == null ? null : "" + r,
      children: e,
      containerInfo: t,
      implementation: n
    };
  }
  function nm(e) {
    if (!e) return _n;
    e = e._reactInternals;
    e: {
      if (Zn(e) !== e || e.tag !== 1) throw Error(P(170));
      var t = e;
      do {
        switch (t.tag) {
          case 3:
            t = t.stateNode.context;
            break e;
          case 1:
            if (gt(t.type)) {
              t = t.stateNode.__reactInternalMemoizedMergedChildContext;
              break e;
            }
        }
        t = t.return;
      } while (t !== null);
      throw Error(P(171));
    }
    if (e.tag === 1) {
      var n = e.type;
      if (gt(n)) return tf(e, n, t);
    }
    return t;
  }
  function rm(e, t, n, r, l, a, s, u, i) {
    return e = Fi(n, r, true, e, l, a, s, u, i), e.context = nm(null), n = e.current, r = st(), l = Nn(n), a = en(r, l), a.callback = t ?? null, Sn(n, a, l), e.current.lanes = l, Rl(e, l, r), vt(e, r), e;
  }
  function cs(e, t, n, r) {
    var l = t.current, a = st(), s = Nn(l);
    return n = nm(n), t.context === null ? t.context = n : t.pendingContext = n, t = en(a, s), t.payload = {
      element: e
    }, r = r === void 0 ? null : r, r !== null && (t.callback = r), e = Sn(l, t, s), e !== null && ($t(e, l, s, a), pa(e, l, s)), s;
  }
  function Ka(e) {
    if (e = e.current, !e.child) return null;
    switch (e.child.tag) {
      case 5:
        return e.child.stateNode;
      default:
        return e.child.stateNode;
    }
  }
  function mc(e, t) {
    if (e = e.memoizedState, e !== null && e.dehydrated !== null) {
      var n = e.retryLane;
      e.retryLane = n !== 0 && n < t ? n : t;
    }
  }
  function Bi(e, t) {
    mc(e, t), (e = e.alternate) && mc(e, t);
  }
  function ag() {
    return null;
  }
  var lm = typeof reportError == "function" ? reportError : function(e) {
    console.error(e);
  };
  function Vi(e) {
    this._internalRoot = e;
  }
  ds.prototype.render = Vi.prototype.render = function(e) {
    var t = this._internalRoot;
    if (t === null) throw Error(P(409));
    cs(e, t, null, null);
  };
  ds.prototype.unmount = Vi.prototype.unmount = function() {
    var e = this._internalRoot;
    if (e !== null) {
      this._internalRoot = null;
      var t = e.containerInfo;
      Yn(function() {
        cs(null, e, null, null);
      }), t[nn] = null;
    }
  };
  function ds(e) {
    this._internalRoot = e;
  }
  ds.prototype.unstable_scheduleHydration = function(e) {
    if (e) {
      var t = Ld();
      e = {
        blockedOn: null,
        target: e,
        priority: t
      };
      for (var n = 0; n < fn.length && t !== 0 && t < fn[n].priority; n++) ;
      fn.splice(n, 0, e), n === 0 && Ad(e);
    }
  };
  function Wi(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
  }
  function fs(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11 && (e.nodeType !== 8 || e.nodeValue !== " react-mount-point-unstable "));
  }
  function pc() {
  }
  function sg(e, t, n, r, l) {
    if (l) {
      if (typeof r == "function") {
        var a = r;
        r = function() {
          var c = Ka(s);
          a.call(c);
        };
      }
      var s = rm(t, r, e, 0, null, false, false, "", pc);
      return e._reactRootContainer = s, e[nn] = s.current, vl(e.nodeType === 8 ? e.parentNode : e), Yn(), s;
    }
    for (; l = e.lastChild; ) e.removeChild(l);
    if (typeof r == "function") {
      var u = r;
      r = function() {
        var c = Ka(i);
        u.call(c);
      };
    }
    var i = Fi(e, 0, false, null, null, false, false, "", pc);
    return e._reactRootContainer = i, e[nn] = i.current, vl(e.nodeType === 8 ? e.parentNode : e), Yn(function() {
      cs(t, i, n, r);
    }), i;
  }
  function ms(e, t, n, r, l) {
    var a = n._reactRootContainer;
    if (a) {
      var s = a;
      if (typeof l == "function") {
        var u = l;
        l = function() {
          var i = Ka(s);
          u.call(i);
        };
      }
      cs(t, s, e, l);
    } else s = sg(n, t, e, l, r);
    return Ka(s);
  }
  Md = function(e) {
    switch (e.tag) {
      case 3:
        var t = e.stateNode;
        if (t.current.memoizedState.isDehydrated) {
          var n = Jr(t.pendingLanes);
          n !== 0 && (ui(t, n | 1), vt(t, Le()), !(re & 6) && (Tr = Le() + 500, Tn()));
        }
        break;
      case 13:
        Yn(function() {
          var r = rn(e, 1);
          if (r !== null) {
            var l = st();
            $t(r, e, 1, l);
          }
        }), Bi(e, 1);
    }
  };
  ci = function(e) {
    if (e.tag === 13) {
      var t = rn(e, 134217728);
      if (t !== null) {
        var n = st();
        $t(t, e, 134217728, n);
      }
      Bi(e, 134217728);
    }
  };
  Dd = function(e) {
    if (e.tag === 13) {
      var t = Nn(e), n = rn(e, t);
      if (n !== null) {
        var r = st();
        $t(n, e, t, r);
      }
      Bi(e, t);
    }
  };
  Ld = function() {
    return ce;
  };
  Od = function(e, t) {
    var n = ce;
    try {
      return ce = e, t();
    } finally {
      ce = n;
    }
  };
  mo = function(e, t, n) {
    switch (t) {
      case "input":
        if (ao(e, n), t = n.name, n.type === "radio" && t != null) {
          for (n = e; n.parentNode; ) n = n.parentNode;
          for (n = n.querySelectorAll("input[name=" + JSON.stringify("" + t) + '][type="radio"]'), t = 0; t < n.length; t++) {
            var r = n[t];
            if (r !== e && r.form === e.form) {
              var l = rs(r);
              if (!l) throw Error(P(90));
              fd(r), ao(r, l);
            }
          }
        }
        break;
      case "textarea":
        pd(e, n);
        break;
      case "select":
        t = n.value, t != null && xr(e, !!n.multiple, t, false);
    }
  };
  Sd = Ii;
  kd = Yn;
  var og = {
    usingClientEntryPoint: false,
    Events: [
      Pl,
      dr,
      rs,
      yd,
      wd,
      Ii
    ]
  }, Wr = {
    findFiberByHostInstance: Un,
    bundleType: 0,
    version: "18.3.1",
    rendererPackageName: "react-dom"
  }, ig = {
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
    currentDispatcherRef: an.ReactCurrentDispatcher,
    findHostInstanceByFiber: function(e) {
      return e = Ed(e), e === null ? null : e.stateNode;
    },
    findFiberByHostInstance: Wr.findFiberByHostInstance || ag,
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
      qa = la.inject(ig), Qt = la;
    } catch {
    }
  }
  Nt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = og;
  Nt.createPortal = function(e, t) {
    var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!Wi(t)) throw Error(P(200));
    return lg(e, t, null, n);
  };
  Nt.createRoot = function(e, t) {
    if (!Wi(e)) throw Error(P(299));
    var n = false, r = "", l = lm;
    return t != null && (t.unstable_strictMode === true && (n = true), t.identifierPrefix !== void 0 && (r = t.identifierPrefix), t.onRecoverableError !== void 0 && (l = t.onRecoverableError)), t = Fi(e, 1, false, null, null, n, false, r, l), e[nn] = t.current, vl(e.nodeType === 8 ? e.parentNode : e), new Vi(t);
  };
  Nt.findDOMNode = function(e) {
    if (e == null) return null;
    if (e.nodeType === 1) return e;
    var t = e._reactInternals;
    if (t === void 0) throw typeof e.render == "function" ? Error(P(188)) : (e = Object.keys(e).join(","), Error(P(268, e)));
    return e = Ed(t), e = e === null ? null : e.stateNode, e;
  };
  Nt.flushSync = function(e) {
    return Yn(e);
  };
  Nt.hydrate = function(e, t, n) {
    if (!fs(t)) throw Error(P(200));
    return ms(null, e, t, true, n);
  };
  Nt.hydrateRoot = function(e, t, n) {
    if (!Wi(e)) throw Error(P(405));
    var r = n != null && n.hydratedSources || null, l = false, a = "", s = lm;
    if (n != null && (n.unstable_strictMode === true && (l = true), n.identifierPrefix !== void 0 && (a = n.identifierPrefix), n.onRecoverableError !== void 0 && (s = n.onRecoverableError)), t = rm(t, null, e, 1, n ?? null, l, false, a, s), e[nn] = t.current, vl(e), r) for (e = 0; e < r.length; e++) n = r[e], l = n._getVersion, l = l(n._source), t.mutableSourceEagerHydrationData == null ? t.mutableSourceEagerHydrationData = [
      n,
      l
    ] : t.mutableSourceEagerHydrationData.push(n, l);
    return new ds(t);
  };
  Nt.render = function(e, t, n) {
    if (!fs(t)) throw Error(P(200));
    return ms(null, e, t, false, n);
  };
  Nt.unmountComponentAtNode = function(e) {
    if (!fs(e)) throw Error(P(40));
    return e._reactRootContainer ? (Yn(function() {
      ms(null, null, e, false, function() {
        e._reactRootContainer = null, e[nn] = null;
      });
    }), true) : false;
  };
  Nt.unstable_batchedUpdates = Ii;
  Nt.unstable_renderSubtreeIntoContainer = function(e, t, n, r) {
    if (!fs(n)) throw Error(P(200));
    if (e == null || e._reactInternals === void 0) throw Error(P(38));
    return ms(e, t, n, false, r);
  };
  Nt.version = "18.3.1-next-f1338f8080-20240426";
  function am() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) try {
      __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(am);
    } catch (e) {
      console.error(e);
    }
  }
  am(), ld.exports = Nt;
  var Hi = ld.exports;
  const ug = Qc(Hi), cg = Hc({
    __proto__: null,
    default: ug
  }, [
    Hi
  ]);
  var hc = Hi;
  Zs.createRoot = hc.createRoot, Zs.hydrateRoot = hc.hydrateRoot;
  function Ee() {
    return Ee = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ee.apply(this, arguments);
  }
  var Ie;
  (function(e) {
    e.Pop = "POP", e.Push = "PUSH", e.Replace = "REPLACE";
  })(Ie || (Ie = {}));
  const gc = "popstate";
  function dg(e) {
    e === void 0 && (e = {});
    function t(r, l) {
      let { pathname: a, search: s, hash: u } = r.location;
      return Cl("", {
        pathname: a,
        search: s,
        hash: u
      }, l.state && l.state.usr || null, l.state && l.state.key || "default");
    }
    function n(r, l) {
      return typeof l == "string" ? l : Dl(l);
    }
    return mg(t, n, null, e);
  }
  function ee(e, t) {
    if (e === false || e === null || typeof e > "u") throw new Error(t);
  }
  function Jn(e, t) {
    if (!e) {
      typeof console < "u" && console.warn(t);
      try {
        throw new Error(t);
      } catch {
      }
    }
  }
  function fg() {
    return Math.random().toString(36).substr(2, 8);
  }
  function vc(e, t) {
    return {
      usr: e.state,
      key: e.key,
      idx: t
    };
  }
  function Cl(e, t, n, r) {
    return n === void 0 && (n = null), Ee({
      pathname: typeof e == "string" ? e : e.pathname,
      search: "",
      hash: ""
    }, typeof t == "string" ? Pn(t) : t, {
      state: n,
      key: t && t.key || r || fg()
    });
  }
  function Dl(e) {
    let { pathname: t = "/", search: n = "", hash: r = "" } = e;
    return n && n !== "?" && (t += n.charAt(0) === "?" ? n : "?" + n), r && r !== "#" && (t += r.charAt(0) === "#" ? r : "#" + r), t;
  }
  function Pn(e) {
    let t = {};
    if (e) {
      let n = e.indexOf("#");
      n >= 0 && (t.hash = e.substr(n), e = e.substr(0, n));
      let r = e.indexOf("?");
      r >= 0 && (t.search = e.substr(r), e = e.substr(0, r)), e && (t.pathname = e);
    }
    return t;
  }
  function mg(e, t, n, r) {
    r === void 0 && (r = {});
    let { window: l = document.defaultView, v5Compat: a = false } = r, s = l.history, u = Ie.Pop, i = null, c = m();
    c == null && (c = 0, s.replaceState(Ee({}, s.state, {
      idx: c
    }), ""));
    function m() {
      return (s.state || {
        idx: null
      }).idx;
    }
    function d() {
      u = Ie.Pop;
      let b = m(), h = b == null ? null : b - c;
      c = b, i && i({
        action: u,
        location: y.location,
        delta: h
      });
    }
    function p(b, h) {
      u = Ie.Push;
      let f = Cl(y.location, b, h);
      c = m() + 1;
      let v = vc(f, c), E = y.createHref(f);
      try {
        s.pushState(v, "", E);
      } catch (_) {
        if (_ instanceof DOMException && _.name === "DataCloneError") throw _;
        l.location.assign(E);
      }
      a && i && i({
        action: u,
        location: y.location,
        delta: 1
      });
    }
    function S(b, h) {
      u = Ie.Replace;
      let f = Cl(y.location, b, h);
      c = m();
      let v = vc(f, c), E = y.createHref(f);
      s.replaceState(v, "", E), a && i && i({
        action: u,
        location: y.location,
        delta: 0
      });
    }
    function w(b) {
      let h = l.location.origin !== "null" ? l.location.origin : l.location.href, f = typeof b == "string" ? b : Dl(b);
      return f = f.replace(/ $/, "%20"), ee(h, "No window.location.(origin|href) available to create URL for href: " + f), new URL(f, h);
    }
    let y = {
      get action() {
        return u;
      },
      get location() {
        return e(l, s);
      },
      listen(b) {
        if (i) throw new Error("A history only accepts one active listener");
        return l.addEventListener(gc, d), i = b, () => {
          l.removeEventListener(gc, d), i = null;
        };
      },
      createHref(b) {
        return t(l, b);
      },
      createURL: w,
      encodeLocation(b) {
        let h = w(b);
        return {
          pathname: h.pathname,
          search: h.search,
          hash: h.hash
        };
      },
      push: p,
      replace: S,
      go(b) {
        return s.go(b);
      }
    };
    return y;
  }
  var ue;
  (function(e) {
    e.data = "data", e.deferred = "deferred", e.redirect = "redirect", e.error = "error";
  })(ue || (ue = {}));
  const pg = /* @__PURE__ */ new Set([
    "lazy",
    "caseSensitive",
    "path",
    "id",
    "index",
    "children"
  ]);
  function hg(e) {
    return e.index === true;
  }
  function Ga(e, t, n, r) {
    return n === void 0 && (n = []), r === void 0 && (r = {}), e.map((l, a) => {
      let s = [
        ...n,
        String(a)
      ], u = typeof l.id == "string" ? l.id : s.join("-");
      if (ee(l.index !== true || !l.children, "Cannot specify children on an index route"), ee(!r[u], 'Found a route id collision on id "' + u + `".  Route id's must be globally unique within Data Router usages`), hg(l)) {
        let i = Ee({}, l, t(l), {
          id: u
        });
        return r[u] = i, i;
      } else {
        let i = Ee({}, l, t(l), {
          id: u,
          children: void 0
        });
        return r[u] = i, l.children && (i.children = Ga(l.children, t, s, r)), i;
      }
    });
  }
  function In(e, t, n) {
    return n === void 0 && (n = "/"), Sa(e, t, n, false);
  }
  function Sa(e, t, n, r) {
    let l = typeof t == "string" ? Pn(t) : t, a = Ll(l.pathname || "/", n);
    if (a == null) return null;
    let s = sm(e);
    vg(s);
    let u = null;
    for (let i = 0; u == null && i < s.length; ++i) {
      let c = bg(a);
      u = Cg(s[i], c, r);
    }
    return u;
  }
  function gg(e, t) {
    let { route: n, pathname: r, params: l } = e;
    return {
      id: n.id,
      pathname: r,
      params: l,
      data: t[n.id],
      handle: n.handle
    };
  }
  function sm(e, t, n, r) {
    t === void 0 && (t = []), n === void 0 && (n = []), r === void 0 && (r = "");
    let l = (a, s, u) => {
      let i = {
        relativePath: u === void 0 ? a.path || "" : u,
        caseSensitive: a.caseSensitive === true,
        childrenIndex: s,
        route: a
      };
      i.relativePath.startsWith("/") && (ee(i.relativePath.startsWith(r), 'Absolute route path "' + i.relativePath + '" nested under path ' + ('"' + r + '" is not valid. An absolute child route path ') + "must start with the combined path of all its parent routes."), i.relativePath = i.relativePath.slice(r.length));
      let c = En([
        r,
        i.relativePath
      ]), m = n.concat(i);
      a.children && a.children.length > 0 && (ee(a.index !== true, "Index routes must not have child routes. Please remove " + ('all child routes from route path "' + c + '".')), sm(a.children, t, m, c)), !(a.path == null && !a.index) && t.push({
        path: c,
        score: jg(c, a.index),
        routesMeta: m
      });
    };
    return e.forEach((a, s) => {
      var u;
      if (a.path === "" || !((u = a.path) != null && u.includes("?"))) l(a, s);
      else for (let i of om(a.path)) l(a, s, i);
    }), t;
  }
  function om(e) {
    let t = e.split("/");
    if (t.length === 0) return [];
    let [n, ...r] = t, l = n.endsWith("?"), a = n.replace(/\?$/, "");
    if (r.length === 0) return l ? [
      a,
      ""
    ] : [
      a
    ];
    let s = om(r.join("/")), u = [];
    return u.push(...s.map((i) => i === "" ? a : [
      a,
      i
    ].join("/"))), l && u.push(...s), u.map((i) => e.startsWith("/") && i === "" ? "/" : i);
  }
  function vg(e) {
    e.sort((t, n) => t.score !== n.score ? n.score - t.score : Eg(t.routesMeta.map((r) => r.childrenIndex), n.routesMeta.map((r) => r.childrenIndex)));
  }
  const xg = /^:[\w-]+$/, yg = 3, wg = 2, Sg = 1, kg = 10, Ng = -2, xc = (e) => e === "*";
  function jg(e, t) {
    let n = e.split("/"), r = n.length;
    return n.some(xc) && (r += Ng), t && (r += wg), n.filter((l) => !xc(l)).reduce((l, a) => l + (xg.test(a) ? yg : a === "" ? Sg : kg), r);
  }
  function Eg(e, t) {
    return e.length === t.length && e.slice(0, -1).every((r, l) => r === t[l]) ? e[e.length - 1] - t[t.length - 1] : 0;
  }
  function Cg(e, t, n) {
    n === void 0 && (n = false);
    let { routesMeta: r } = e, l = {}, a = "/", s = [];
    for (let u = 0; u < r.length; ++u) {
      let i = r[u], c = u === r.length - 1, m = a === "/" ? t : t.slice(a.length) || "/", d = yc({
        path: i.relativePath,
        caseSensitive: i.caseSensitive,
        end: c
      }, m), p = i.route;
      if (!d && c && n && !r[r.length - 1].route.index && (d = yc({
        path: i.relativePath,
        caseSensitive: i.caseSensitive,
        end: false
      }, m)), !d) return null;
      Object.assign(l, d.params), s.push({
        params: l,
        pathname: En([
          a,
          d.pathname
        ]),
        pathnameBase: Mg(En([
          a,
          d.pathnameBase
        ])),
        route: p
      }), d.pathnameBase !== "/" && (a = En([
        a,
        d.pathnameBase
      ]));
    }
    return s;
  }
  function yc(e, t) {
    typeof e == "string" && (e = {
      path: e,
      caseSensitive: false,
      end: true
    });
    let [n, r] = _g(e.path, e.caseSensitive, e.end), l = t.match(n);
    if (!l) return null;
    let a = l[0], s = a.replace(/(.)\/+$/, "$1"), u = l.slice(1);
    return {
      params: r.reduce((c, m, d) => {
        let { paramName: p, isOptional: S } = m;
        if (p === "*") {
          let y = u[d] || "";
          s = a.slice(0, a.length - y.length).replace(/(.)\/+$/, "$1");
        }
        const w = u[d];
        return S && !w ? c[p] = void 0 : c[p] = (w || "").replace(/%2F/g, "/"), c;
      }, {}),
      pathname: a,
      pathnameBase: s,
      pattern: e
    };
  }
  function _g(e, t, n) {
    t === void 0 && (t = false), n === void 0 && (n = true), Jn(e === "*" || !e.endsWith("*") || e.endsWith("/*"), 'Route path "' + e + '" will be treated as if it were ' + ('"' + e.replace(/\*$/, "/*") + '" because the `*` character must ') + "always follow a `/` in the pattern. To get rid of this warning, " + ('please change the route path to "' + e.replace(/\*$/, "/*") + '".'));
    let r = [], l = "^" + e.replace(/\/*\*?$/, "").replace(/^\/*/, "/").replace(/[\\.*+^${}|()[\]]/g, "\\$&").replace(/\/:([\w-]+)(\?)?/g, (s, u, i) => (r.push({
      paramName: u,
      isOptional: i != null
    }), i ? "/?([^\\/]+)?" : "/([^\\/]+)"));
    return e.endsWith("*") ? (r.push({
      paramName: "*"
    }), l += e === "*" || e === "/*" ? "(.*)$" : "(?:\\/(.+)|\\/*)$") : n ? l += "\\/*$" : e !== "" && e !== "/" && (l += "(?:(?=\\/|$))"), [
      new RegExp(l, t ? void 0 : "i"),
      r
    ];
  }
  function bg(e) {
    try {
      return e.split("/").map((t) => decodeURIComponent(t).replace(/\//g, "%2F")).join("/");
    } catch (t) {
      return Jn(false, 'The URL path "' + e + '" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent ' + ("encoding (" + t + ").")), e;
    }
  }
  function Ll(e, t) {
    if (t === "/") return e;
    if (!e.toLowerCase().startsWith(t.toLowerCase())) return null;
    let n = t.endsWith("/") ? t.length - 1 : t.length, r = e.charAt(n);
    return r && r !== "/" ? null : e.slice(n) || "/";
  }
  const Rg = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Tg = (e) => Rg.test(e);
  function Pg(e, t) {
    t === void 0 && (t = "/");
    let { pathname: n, search: r = "", hash: l = "" } = typeof e == "string" ? Pn(e) : e, a;
    if (n) if (Tg(n)) a = n;
    else {
      if (n.includes("//")) {
        let s = n;
        n = n.replace(/\/\/+/g, "/"), Jn(false, "Pathnames cannot have embedded double slashes - normalizing " + (s + " -> " + n));
      }
      n.startsWith("/") ? a = wc(n.substring(1), "/") : a = wc(n, t);
    }
    else a = t;
    return {
      pathname: a,
      search: Dg(r),
      hash: Lg(l)
    };
  }
  function wc(e, t) {
    let n = t.replace(/\/+$/, "").split("/");
    return e.split("/").forEach((l) => {
      l === ".." ? n.length > 1 && n.pop() : l !== "." && n.push(l);
    }), n.length > 1 ? n.join("/") : "/";
  }
  function Ks(e, t, n, r) {
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
    typeof e == "string" ? l = Pn(e) : (l = Ee({}, e), ee(!l.pathname || !l.pathname.includes("?"), Ks("?", "pathname", "search", l)), ee(!l.pathname || !l.pathname.includes("#"), Ks("#", "pathname", "hash", l)), ee(!l.search || !l.search.includes("#"), Ks("#", "search", "hash", l)));
    let a = e === "" || l.pathname === "", s = a ? "/" : l.pathname, u;
    if (s == null) u = n;
    else {
      let d = t.length - 1;
      if (!r && s.startsWith("..")) {
        let p = s.split("/");
        for (; p[0] === ".."; ) p.shift(), d -= 1;
        l.pathname = p.join("/");
      }
      u = d >= 0 ? t[d] : "/";
    }
    let i = Pg(l, u), c = s && s !== "/" && s.endsWith("/"), m = (a || s === ".") && n.endsWith("/");
    return !i.pathname.endsWith("/") && (c || m) && (i.pathname += "/"), i;
  }
  const En = (e) => e.join("/").replace(/\/\/+/g, "/"), Mg = (e) => e.replace(/\/+$/, "").replace(/^\/*/, "/"), Dg = (e) => !e || e === "?" ? "" : e.startsWith("?") ? e : "?" + e, Lg = (e) => !e || e === "#" ? "" : e.startsWith("#") ? e : "#" + e;
  class Ya {
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
  ], Og = new Set(dm), Ag = [
    "get",
    ...dm
  ], Ig = new Set(Ag), zg = /* @__PURE__ */ new Set([
    301,
    302,
    303,
    307,
    308
  ]), Ug = /* @__PURE__ */ new Set([
    307,
    308
  ]), Gs = {
    state: "idle",
    location: void 0,
    formMethod: void 0,
    formAction: void 0,
    formEncType: void 0,
    formData: void 0,
    json: void 0,
    text: void 0
  }, $g = {
    state: "idle",
    data: void 0,
    formMethod: void 0,
    formAction: void 0,
    formEncType: void 0,
    formData: void 0,
    json: void 0,
    text: void 0
  }, Hr = {
    state: "unblocked",
    proceed: void 0,
    reset: void 0,
    location: void 0
  }, Qi = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Fg = (e) => ({
    hasErrorBoundary: !!e.hasErrorBoundary
  }), fm = "remix-router-transitions";
  function Bg(e) {
    const t = e.window ? e.window : typeof window < "u" ? window : void 0, n = typeof t < "u" && typeof t.document < "u" && typeof t.document.createElement < "u", r = !n;
    ee(e.routes.length > 0, "You must provide a non-empty routes array to createRouter");
    let l;
    if (e.mapRouteProperties) l = e.mapRouteProperties;
    else if (e.detectErrorBoundary) {
      let x = e.detectErrorBoundary;
      l = (N) => ({
        hasErrorBoundary: x(N)
      });
    } else l = Fg;
    let a = {}, s = Ga(e.routes, l, void 0, a), u, i = e.basename || "/", c = e.dataStrategy || Qg, m = e.patchRoutesOnNavigation, d = Ee({
      v7_fetcherPersist: false,
      v7_normalizeFormMethod: false,
      v7_partialHydration: false,
      v7_prependBasename: false,
      v7_relativeSplatPath: false,
      v7_skipActionErrorRevalidation: false
    }, e.future), p = null, S = /* @__PURE__ */ new Set(), w = null, y = null, b = null, h = e.hydrationData != null, f = In(s, e.history.location, i), v = false, E = null;
    if (f == null && !m) {
      let x = ft(404, {
        pathname: e.history.location.pathname
      }), { matches: N, route: C } = Pc(s);
      f = N, E = {
        [C.id]: x
      };
    }
    f && !e.hydrationData && zl(f, s, e.history.location.pathname).active && (f = null);
    let _;
    if (f) if (f.some((x) => x.route.lazy)) _ = false;
    else if (!f.some((x) => x.route.loader)) _ = true;
    else if (d.v7_partialHydration) {
      let x = e.hydrationData ? e.hydrationData.loaderData : null, N = e.hydrationData ? e.hydrationData.errors : null;
      if (N) {
        let C = f.findIndex((T) => N[T.route.id] !== void 0);
        _ = f.slice(0, C + 1).every((T) => !Ko(T.route, x, N));
      } else _ = f.every((C) => !Ko(C.route, x, N));
    } else _ = e.hydrationData != null;
    else if (_ = false, f = [], d.v7_partialHydration) {
      let x = zl(null, s, e.history.location.pathname);
      x.active && x.matches && (v = true, f = x.matches);
    }
    let R, k = {
      historyAction: e.history.action,
      location: e.history.location,
      matches: f,
      initialized: _,
      navigation: Gs,
      restoreScrollPosition: e.hydrationData != null ? false : null,
      preventScrollReset: false,
      revalidation: "idle",
      loaderData: e.hydrationData && e.hydrationData.loaderData || {},
      actionData: e.hydrationData && e.hydrationData.actionData || null,
      errors: e.hydrationData && e.hydrationData.errors || E,
      fetchers: /* @__PURE__ */ new Map(),
      blockers: /* @__PURE__ */ new Map()
    }, j = Ie.Pop, I = false, D, Q = false, K = /* @__PURE__ */ new Map(), ae = null, de = false, ge = false, lt = [], Qe = /* @__PURE__ */ new Set(), M = /* @__PURE__ */ new Map(), W = 0, H = -1, Y = /* @__PURE__ */ new Map(), X = /* @__PURE__ */ new Set(), Fe = /* @__PURE__ */ new Map(), ve = /* @__PURE__ */ new Map(), fe = /* @__PURE__ */ new Set(), Te = /* @__PURE__ */ new Map(), Oe = /* @__PURE__ */ new Map(), q;
    function me() {
      if (p = e.history.listen((x) => {
        let { action: N, location: C, delta: T } = x;
        if (q) {
          q(), q = void 0;
          return;
        }
        Jn(Oe.size === 0 || T != null, "You are trying to use a blocker on a POP navigation to a location that was not created by @remix-run/router. This will fail silently in production. This can happen if you are navigating outside the router via `window.history.pushState`/`window.location.hash` instead of using router navigation APIs.  This can also happen if you are using createHashRouter and the user manually changes the URL.");
        let L = ru({
          currentLocation: k.location,
          nextLocation: C,
          historyAction: N
        });
        if (L && T != null) {
          let V = new Promise((G) => {
            q = G;
          });
          e.history.go(T * -1), Il(L, {
            state: "blocked",
            location: C,
            proceed() {
              Il(L, {
                state: "proceeding",
                proceed: void 0,
                reset: void 0,
                location: C
              }), V.then(() => e.history.go(T));
            },
            reset() {
              let G = new Map(k.blockers);
              G.set(L, Hr), B({
                blockers: G
              });
            }
          });
          return;
        }
        return ke(N, C);
      }), n) {
        sv(t, K);
        let x = () => ov(t, K);
        t.addEventListener("pagehide", x), ae = () => t.removeEventListener("pagehide", x);
      }
      return k.initialized || ke(Ie.Pop, k.location, {
        initialHydration: true
      }), R;
    }
    function he() {
      p && p(), ae && ae(), S.clear(), D && D.abort(), k.fetchers.forEach((x, N) => Al(N)), k.blockers.forEach((x, N) => nu(N));
    }
    function $(x) {
      return S.add(x), () => S.delete(x);
    }
    function B(x, N) {
      N === void 0 && (N = {}), k = Ee({}, k, x);
      let C = [], T = [];
      d.v7_fetcherPersist && k.fetchers.forEach((L, V) => {
        L.state === "idle" && (fe.has(V) ? T.push(V) : C.push(V));
      }), fe.forEach((L) => {
        !k.fetchers.has(L) && !M.has(L) && T.push(L);
      }), [
        ...S
      ].forEach((L) => L(k, {
        deletedFetchers: T,
        viewTransitionOpts: N.viewTransitionOpts,
        flushSync: N.flushSync === true
      })), d.v7_fetcherPersist ? (C.forEach((L) => k.fetchers.delete(L)), T.forEach((L) => Al(L))) : T.forEach((L) => fe.delete(L));
    }
    function pe(x, N, C) {
      var T, L;
      let { flushSync: V } = C === void 0 ? {} : C, G = k.actionData != null && k.navigation.formMethod != null && It(k.navigation.formMethod) && k.navigation.state === "loading" && ((T = x.state) == null ? void 0 : T._isRedirect) !== true, z;
      N.actionData ? Object.keys(N.actionData).length > 0 ? z = N.actionData : z = null : G ? z = k.actionData : z = null;
      let U = N.loaderData ? Rc(k.loaderData, N.loaderData, N.matches || [], N.errors) : k.loaderData, A = k.blockers;
      A.size > 0 && (A = new Map(A), A.forEach((te, Ge) => A.set(Ge, Hr)));
      let F = I === true || k.navigation.formMethod != null && It(k.navigation.formMethod) && ((L = x.state) == null ? void 0 : L._isRedirect) !== true;
      u && (s = u, u = void 0), de || j === Ie.Pop || (j === Ie.Push ? e.history.push(x, x.state) : j === Ie.Replace && e.history.replace(x, x.state));
      let J;
      if (j === Ie.Pop) {
        let te = K.get(k.location.pathname);
        te && te.has(x.pathname) ? J = {
          currentLocation: k.location,
          nextLocation: x
        } : K.has(x.pathname) && (J = {
          currentLocation: x,
          nextLocation: k.location
        });
      } else if (Q) {
        let te = K.get(k.location.pathname);
        te ? te.add(x.pathname) : (te = /* @__PURE__ */ new Set([
          x.pathname
        ]), K.set(k.location.pathname, te)), J = {
          currentLocation: k.location,
          nextLocation: x
        };
      }
      B(Ee({}, N, {
        actionData: z,
        loaderData: U,
        historyAction: j,
        location: x,
        initialized: true,
        navigation: Gs,
        revalidation: "idle",
        restoreScrollPosition: au(x, N.matches || k.matches),
        preventScrollReset: F,
        blockers: A
      }), {
        viewTransitionOpts: J,
        flushSync: V === true
      }), j = Ie.Pop, I = false, Q = false, de = false, ge = false, lt = [];
    }
    async function se(x, N) {
      if (typeof x == "number") {
        e.history.go(x);
        return;
      }
      let C = Qo(k.location, k.matches, i, d.v7_prependBasename, x, d.v7_relativeSplatPath, N == null ? void 0 : N.fromRouteId, N == null ? void 0 : N.relative), { path: T, submission: L, error: V } = Sc(d.v7_normalizeFormMethod, false, C, N), G = k.location, z = Cl(k.location, T, N && N.state);
      z = Ee({}, z, e.history.encodeLocation(z));
      let U = N && N.replace != null ? N.replace : void 0, A = Ie.Push;
      U === true ? A = Ie.Replace : U === false || L != null && It(L.formMethod) && L.formAction === k.location.pathname + k.location.search && (A = Ie.Replace);
      let F = N && "preventScrollReset" in N ? N.preventScrollReset === true : void 0, J = (N && N.flushSync) === true, te = ru({
        currentLocation: G,
        nextLocation: z,
        historyAction: A
      });
      if (te) {
        Il(te, {
          state: "blocked",
          location: z,
          proceed() {
            Il(te, {
              state: "proceeding",
              proceed: void 0,
              reset: void 0,
              location: z
            }), se(x, N);
          },
          reset() {
            let Ge = new Map(k.blockers);
            Ge.set(te, Hr), B({
              blockers: Ge
            });
          }
        });
        return;
      }
      return await ke(A, z, {
        submission: L,
        pendingError: V,
        preventScrollReset: F,
        replace: N && N.replace,
        enableViewTransition: N && N.viewTransition,
        flushSync: J
      });
    }
    function Se() {
      if (Et(), B({
        revalidation: "loading"
      }), k.navigation.state !== "submitting") {
        if (k.navigation.state === "idle") {
          ke(k.historyAction, k.location, {
            startUninterruptedRevalidation: true
          });
          return;
        }
        ke(j || k.historyAction, k.navigation.location, {
          overrideNavigation: k.navigation,
          enableViewTransition: Q === true
        });
      }
    }
    async function ke(x, N, C) {
      D && D.abort(), D = null, j = x, de = (C && C.startUninterruptedRevalidation) === true, Mm(k.location, k.matches), I = (C && C.preventScrollReset) === true, Q = (C && C.enableViewTransition) === true;
      let T = u || s, L = C && C.overrideNavigation, V = C != null && C.initialHydration && k.matches && k.matches.length > 0 && !v ? k.matches : In(T, N, i), G = (C && C.flushSync) === true;
      if (V && k.initialized && !ge && Zg(k.location, N) && !(C && C.submission && It(C.submission.formMethod))) {
        pe(N, {
          matches: V
        }, {
          flushSync: G
        });
        return;
      }
      let z = zl(V, T, N.pathname);
      if (z.active && z.matches && (V = z.matches), !V) {
        let { error: ye, notFoundMatches: ie, route: Pe } = xs(N.pathname);
        pe(N, {
          matches: ie,
          loaderData: {},
          errors: {
            [Pe.id]: ye
          }
        }, {
          flushSync: G
        });
        return;
      }
      D = new AbortController();
      let U = lr(e.history, N, D.signal, C && C.submission), A;
      if (C && C.pendingError) A = [
        zn(V).route.id,
        {
          type: ue.error,
          error: C.pendingError
        }
      ];
      else if (C && C.submission && It(C.submission.formMethod)) {
        let ye = await Ze(U, N, C.submission, V, z.active, {
          replace: C.replace,
          flushSync: G
        });
        if (ye.shortCircuited) return;
        if (ye.pendingActionResult) {
          let [ie, Pe] = ye.pendingActionResult;
          if (yt(Pe) && _l(Pe.error) && Pe.error.status === 404) {
            D = null, pe(N, {
              matches: ye.matches,
              loaderData: {},
              errors: {
                [ie]: Pe.error
              }
            });
            return;
          }
        }
        V = ye.matches || V, A = ye.pendingActionResult, L = Ys(N, C.submission), G = false, z.active = false, U = lr(e.history, U.url, U.signal);
      }
      let { shortCircuited: F, matches: J, loaderData: te, errors: Ge } = await ut(U, N, V, z.active, L, C && C.submission, C && C.fetcherSubmission, C && C.replace, C && C.initialHydration === true, G, A);
      F || (D = null, pe(N, Ee({
        matches: J || V
      }, Tc(A), {
        loaderData: te,
        errors: Ge
      })));
    }
    async function Ze(x, N, C, T, L, V) {
      V === void 0 && (V = {}), Et();
      let G = lv(N, C);
      if (B({
        navigation: G
      }, {
        flushSync: V.flushSync === true
      }), L) {
        let A = await Ul(T, N.pathname, x.signal);
        if (A.type === "aborted") return {
          shortCircuited: true
        };
        if (A.type === "error") {
          let F = zn(A.partialMatches).route.id;
          return {
            matches: A.partialMatches,
            pendingActionResult: [
              F,
              {
                type: ue.error,
                error: A.error
              }
            ]
          };
        } else if (A.matches) T = A.matches;
        else {
          let { notFoundMatches: F, error: J, route: te } = xs(N.pathname);
          return {
            matches: F,
            pendingActionResult: [
              te.id,
              {
                type: ue.error,
                error: J
              }
            ]
          };
        }
      }
      let z, U = Zr(T, N);
      if (!U.route.action && !U.route.lazy) z = {
        type: ue.error,
        error: ft(405, {
          method: x.method,
          pathname: N.pathname,
          routeId: U.route.id
        })
      };
      else if (z = (await xe("action", k, x, [
        U
      ], T, null))[U.route.id], x.signal.aborted) return {
        shortCircuited: true
      };
      if (Bn(z)) {
        let A;
        return V && V.replace != null ? A = V.replace : A = Cc(z.response.headers.get("Location"), new URL(x.url), i, e.history) === k.location.pathname + k.location.search, await ne(x, z, true, {
          submission: C,
          replace: A
        }), {
          shortCircuited: true
        };
      }
      if (gn(z)) throw ft(400, {
        type: "defer-action"
      });
      if (yt(z)) {
        let A = zn(T, U.route.id);
        return (V && V.replace) !== true && (j = Ie.Push), {
          matches: T,
          pendingActionResult: [
            A.route.id,
            z
          ]
        };
      }
      return {
        matches: T,
        pendingActionResult: [
          U.route.id,
          z
        ]
      };
    }
    async function ut(x, N, C, T, L, V, G, z, U, A, F) {
      let J = L || Ys(N, V), te = V || G || Dc(J), Ge = !de && (!d.v7_partialHydration || !U);
      if (T) {
        if (Ge) {
          let Me = Be(F);
          B(Ee({
            navigation: J
          }, Me !== void 0 ? {
            actionData: Me
          } : {}), {
            flushSync: A
          });
        }
        let oe = await Ul(C, N.pathname, x.signal);
        if (oe.type === "aborted") return {
          shortCircuited: true
        };
        if (oe.type === "error") {
          let Me = zn(oe.partialMatches).route.id;
          return {
            matches: oe.partialMatches,
            loaderData: {},
            errors: {
              [Me]: oe.error
            }
          };
        } else if (oe.matches) C = oe.matches;
        else {
          let { error: Me, notFoundMatches: tr, route: Ar } = xs(N.pathname);
          return {
            matches: tr,
            loaderData: {},
            errors: {
              [Ar.id]: Me
            }
          };
        }
      }
      let ye = u || s, [ie, Pe] = Nc(e.history, k, C, te, N, d.v7_partialHydration && U === true, d.v7_skipActionErrorRevalidation, ge, lt, Qe, fe, Fe, X, ye, i, F);
      if (ys((oe) => !(C && C.some((Me) => Me.route.id === oe)) || ie && ie.some((Me) => Me.route.id === oe)), H = ++W, ie.length === 0 && Pe.length === 0) {
        let oe = eu();
        return pe(N, Ee({
          matches: C,
          loaderData: {},
          errors: F && yt(F[1]) ? {
            [F[0]]: F[1].error
          } : null
        }, Tc(F), oe ? {
          fetchers: new Map(k.fetchers)
        } : {}), {
          flushSync: A
        }), {
          shortCircuited: true
        };
      }
      if (Ge) {
        let oe = {};
        if (!T) {
          oe.navigation = J;
          let Me = Be(F);
          Me !== void 0 && (oe.actionData = Me);
        }
        Pe.length > 0 && (oe.fetchers = qe(Pe)), B(oe, {
          flushSync: A
        });
      }
      Pe.forEach((oe) => {
        sn(oe.key), oe.controller && M.set(oe.key, oe.controller);
      });
      let er = () => Pe.forEach((oe) => sn(oe.key));
      D && D.signal.addEventListener("abort", er);
      let { loaderResults: Lr, fetcherResults: Yt } = await Ve(k, C, ie, Pe, x);
      if (x.signal.aborted) return {
        shortCircuited: true
      };
      D && D.signal.removeEventListener("abort", er), Pe.forEach((oe) => M.delete(oe.key));
      let Bt = aa(Lr);
      if (Bt) return await ne(x, Bt.result, true, {
        replace: z
      }), {
        shortCircuited: true
      };
      if (Bt = aa(Yt), Bt) return X.add(Bt.key), await ne(x, Bt.result, true, {
        replace: z
      }), {
        shortCircuited: true
      };
      let { loaderData: ws, errors: Or } = bc(k, C, Lr, F, Pe, Yt, Te);
      Te.forEach((oe, Me) => {
        oe.subscribe((tr) => {
          (tr || oe.done) && Te.delete(Me);
        });
      }), d.v7_partialHydration && U && k.errors && (Or = Ee({}, k.errors, Or));
      let Dn = eu(), $l = tu(H), Fl = Dn || $l || Pe.length > 0;
      return Ee({
        matches: C,
        loaderData: ws,
        errors: Or
      }, Fl ? {
        fetchers: new Map(k.fetchers)
      } : {});
    }
    function Be(x) {
      if (x && !yt(x[1])) return {
        [x[0]]: x[1].data
      };
      if (k.actionData) return Object.keys(k.actionData).length === 0 ? null : k.actionData;
    }
    function qe(x) {
      return x.forEach((N) => {
        let C = k.fetchers.get(N.key), T = Qr(void 0, C ? C.data : void 0);
        k.fetchers.set(N.key, T);
      }), new Map(k.fetchers);
    }
    function ct(x, N, C, T) {
      if (r) throw new Error("router.fetch() was called during the server render, but it shouldn't be. You are likely calling a useFetcher() method in the body of your component. Try moving it to a useEffect or a callback.");
      sn(x);
      let L = (T && T.flushSync) === true, V = u || s, G = Qo(k.location, k.matches, i, d.v7_prependBasename, C, d.v7_relativeSplatPath, N, T == null ? void 0 : T.relative), z = In(V, G, i), U = zl(z, V, G);
      if (U.active && U.matches && (z = U.matches), !z) {
        Mt(x, N, ft(404, {
          pathname: G
        }), {
          flushSync: L
        });
        return;
      }
      let { path: A, submission: F, error: J } = Sc(d.v7_normalizeFormMethod, true, G, T);
      if (J) {
        Mt(x, N, J, {
          flushSync: L
        });
        return;
      }
      let te = Zr(z, A), Ge = (T && T.preventScrollReset) === true;
      if (F && It(F.formMethod)) {
        Mn(x, N, A, te, z, U.active, L, Ge, F);
        return;
      }
      Fe.set(x, {
        routeId: N,
        path: A
      }), dt(x, N, A, te, z, U.active, L, Ge, F);
    }
    async function Mn(x, N, C, T, L, V, G, z, U) {
      Et(), Fe.delete(x);
      function A(Ae) {
        if (!Ae.route.action && !Ae.route.lazy) {
          let nr = ft(405, {
            method: U.formMethod,
            pathname: C,
            routeId: N
          });
          return Mt(x, N, nr, {
            flushSync: G
          }), true;
        }
        return false;
      }
      if (!V && A(T)) return;
      let F = k.fetchers.get(x);
      Ke(x, av(U, F), {
        flushSync: G
      });
      let J = new AbortController(), te = lr(e.history, C, J.signal, U);
      if (V) {
        let Ae = await Ul(L, new URL(te.url).pathname, te.signal, x);
        if (Ae.type === "aborted") return;
        if (Ae.type === "error") {
          Mt(x, N, Ae.error, {
            flushSync: G
          });
          return;
        } else if (Ae.matches) {
          if (L = Ae.matches, T = Zr(L, C), A(T)) return;
        } else {
          Mt(x, N, ft(404, {
            pathname: C
          }), {
            flushSync: G
          });
          return;
        }
      }
      M.set(x, J);
      let Ge = W, ie = (await xe("action", k, te, [
        T
      ], L, x))[T.route.id];
      if (te.signal.aborted) {
        M.get(x) === J && M.delete(x);
        return;
      }
      if (d.v7_fetcherPersist && fe.has(x)) {
        if (Bn(ie) || yt(ie)) {
          Ke(x, un(void 0));
          return;
        }
      } else {
        if (Bn(ie)) if (M.delete(x), H > Ge) {
          Ke(x, un(void 0));
          return;
        } else return X.add(x), Ke(x, Qr(U)), ne(te, ie, false, {
          fetcherSubmission: U,
          preventScrollReset: z
        });
        if (yt(ie)) {
          Mt(x, N, ie.error);
          return;
        }
      }
      if (gn(ie)) throw ft(400, {
        type: "defer-action"
      });
      let Pe = k.navigation.location || k.location, er = lr(e.history, Pe, J.signal), Lr = u || s, Yt = k.navigation.state !== "idle" ? In(Lr, k.navigation.location, i) : k.matches;
      ee(Yt, "Didn't find any matches after fetcher action");
      let Bt = ++W;
      Y.set(x, Bt);
      let ws = Qr(U, ie.data);
      k.fetchers.set(x, ws);
      let [Or, Dn] = Nc(e.history, k, Yt, U, Pe, false, d.v7_skipActionErrorRevalidation, ge, lt, Qe, fe, Fe, X, Lr, i, [
        T.route.id,
        ie
      ]);
      Dn.filter((Ae) => Ae.key !== x).forEach((Ae) => {
        let nr = Ae.key, su = k.fetchers.get(nr), Om = Qr(void 0, su ? su.data : void 0);
        k.fetchers.set(nr, Om), sn(nr), Ae.controller && M.set(nr, Ae.controller);
      }), B({
        fetchers: new Map(k.fetchers)
      });
      let $l = () => Dn.forEach((Ae) => sn(Ae.key));
      J.signal.addEventListener("abort", $l);
      let { loaderResults: Fl, fetcherResults: oe } = await Ve(k, Yt, Or, Dn, er);
      if (J.signal.aborted) return;
      J.signal.removeEventListener("abort", $l), Y.delete(x), M.delete(x), Dn.forEach((Ae) => M.delete(Ae.key));
      let Me = aa(Fl);
      if (Me) return ne(er, Me.result, false, {
        preventScrollReset: z
      });
      if (Me = aa(oe), Me) return X.add(Me.key), ne(er, Me.result, false, {
        preventScrollReset: z
      });
      let { loaderData: tr, errors: Ar } = bc(k, Yt, Fl, void 0, Dn, oe, Te);
      if (k.fetchers.has(x)) {
        let Ae = un(ie.data);
        k.fetchers.set(x, Ae);
      }
      tu(Bt), k.navigation.state === "loading" && Bt > H ? (ee(j, "Expected pending action"), D && D.abort(), pe(k.navigation.location, {
        matches: Yt,
        loaderData: tr,
        errors: Ar,
        fetchers: new Map(k.fetchers)
      })) : (B({
        errors: Ar,
        loaderData: Rc(k.loaderData, tr, Yt, Ar),
        fetchers: new Map(k.fetchers)
      }), ge = false);
    }
    async function dt(x, N, C, T, L, V, G, z, U) {
      let A = k.fetchers.get(x);
      Ke(x, Qr(U, A ? A.data : void 0), {
        flushSync: G
      });
      let F = new AbortController(), J = lr(e.history, C, F.signal);
      if (V) {
        let ie = await Ul(L, new URL(J.url).pathname, J.signal, x);
        if (ie.type === "aborted") return;
        if (ie.type === "error") {
          Mt(x, N, ie.error, {
            flushSync: G
          });
          return;
        } else if (ie.matches) L = ie.matches, T = Zr(L, C);
        else {
          Mt(x, N, ft(404, {
            pathname: C
          }), {
            flushSync: G
          });
          return;
        }
      }
      M.set(x, F);
      let te = W, ye = (await xe("loader", k, J, [
        T
      ], L, x))[T.route.id];
      if (gn(ye) && (ye = await Ki(ye, J.signal, true) || ye), M.get(x) === F && M.delete(x), !J.signal.aborted) {
        if (fe.has(x)) {
          Ke(x, un(void 0));
          return;
        }
        if (Bn(ye)) if (H > te) {
          Ke(x, un(void 0));
          return;
        } else {
          X.add(x), await ne(J, ye, false, {
            preventScrollReset: z
          });
          return;
        }
        if (yt(ye)) {
          Mt(x, N, ye.error);
          return;
        }
        ee(!gn(ye), "Unhandled fetcher deferred data"), Ke(x, un(ye.data));
      }
    }
    async function ne(x, N, C, T) {
      let { submission: L, fetcherSubmission: V, preventScrollReset: G, replace: z } = T === void 0 ? {} : T;
      N.response.headers.has("X-Remix-Revalidate") && (ge = true);
      let U = N.response.headers.get("Location");
      ee(U, "Expected a Location header on the redirect Response"), U = Cc(U, new URL(x.url), i, e.history);
      let A = Cl(k.location, U, {
        _isRedirect: true
      });
      if (n) {
        let ie = false;
        if (N.response.headers.has("X-Remix-Reload-Document")) ie = true;
        else if (Qi.test(U)) {
          const Pe = e.history.createURL(U);
          ie = Pe.origin !== t.location.origin || Ll(Pe.pathname, i) == null;
        }
        if (ie) {
          z ? t.location.replace(U) : t.location.assign(U);
          return;
        }
      }
      D = null;
      let F = z === true || N.response.headers.has("X-Remix-Replace") ? Ie.Replace : Ie.Push, { formMethod: J, formAction: te, formEncType: Ge } = k.navigation;
      !L && !V && J && te && Ge && (L = Dc(k.navigation));
      let ye = L || V;
      if (Ug.has(N.response.status) && ye && It(ye.formMethod)) await ke(F, A, {
        submission: Ee({}, ye, {
          formAction: U
        }),
        preventScrollReset: G || I,
        enableViewTransition: C ? Q : void 0
      });
      else {
        let ie = Ys(A, L);
        await ke(F, A, {
          overrideNavigation: ie,
          fetcherSubmission: V,
          preventScrollReset: G || I,
          enableViewTransition: C ? Q : void 0
        });
      }
    }
    async function xe(x, N, C, T, L, V) {
      let G, z = {};
      try {
        G = await Kg(c, x, N, C, T, L, V, a, l);
      } catch (U) {
        return T.forEach((A) => {
          z[A.route.id] = {
            type: ue.error,
            error: U
          };
        }), z;
      }
      for (let [U, A] of Object.entries(G)) if (qg(A)) {
        let F = A.result;
        z[U] = {
          type: ue.redirect,
          response: Jg(F, C, U, L, i, d.v7_relativeSplatPath)
        };
      } else z[U] = await Yg(A);
      return z;
    }
    async function Ve(x, N, C, T, L) {
      let V = x.matches, G = xe("loader", x, L, C, N, null), z = Promise.all(T.map(async (F) => {
        if (F.matches && F.match && F.controller) {
          let te = (await xe("loader", x, lr(e.history, F.path, F.controller.signal), [
            F.match
          ], F.matches, F.key))[F.match.route.id];
          return {
            [F.key]: te
          };
        } else return Promise.resolve({
          [F.key]: {
            type: ue.error,
            error: ft(404, {
              pathname: F.path
            })
          }
        });
      })), U = await G, A = (await z).reduce((F, J) => Object.assign(F, J), {});
      return await Promise.all([
        nv(N, U, L.signal, V, x.loaderData),
        rv(N, A, T)
      ]), {
        loaderResults: U,
        fetcherResults: A
      };
    }
    function Et() {
      ge = true, lt.push(...ys()), Fe.forEach((x, N) => {
        M.has(N) && Qe.add(N), sn(N);
      });
    }
    function Ke(x, N, C) {
      C === void 0 && (C = {}), k.fetchers.set(x, N), B({
        fetchers: new Map(k.fetchers)
      }, {
        flushSync: (C && C.flushSync) === true
      });
    }
    function Mt(x, N, C, T) {
      T === void 0 && (T = {});
      let L = zn(k.matches, N);
      Al(x), B({
        errors: {
          [L.route.id]: C
        },
        fetchers: new Map(k.fetchers)
      }, {
        flushSync: (T && T.flushSync) === true
      });
    }
    function Ol(x) {
      return ve.set(x, (ve.get(x) || 0) + 1), fe.has(x) && fe.delete(x), k.fetchers.get(x) || $g;
    }
    function Al(x) {
      let N = k.fetchers.get(x);
      M.has(x) && !(N && N.state === "loading" && Y.has(x)) && sn(x), Fe.delete(x), Y.delete(x), X.delete(x), d.v7_fetcherPersist && fe.delete(x), Qe.delete(x), k.fetchers.delete(x);
    }
    function Rm(x) {
      let N = (ve.get(x) || 0) - 1;
      N <= 0 ? (ve.delete(x), fe.add(x), d.v7_fetcherPersist || Al(x)) : ve.set(x, N), B({
        fetchers: new Map(k.fetchers)
      });
    }
    function sn(x) {
      let N = M.get(x);
      N && (N.abort(), M.delete(x));
    }
    function qi(x) {
      for (let N of x) {
        let C = Ol(N), T = un(C.data);
        k.fetchers.set(N, T);
      }
    }
    function eu() {
      let x = [], N = false;
      for (let C of X) {
        let T = k.fetchers.get(C);
        ee(T, "Expected fetcher: " + C), T.state === "loading" && (X.delete(C), x.push(C), N = true);
      }
      return qi(x), N;
    }
    function tu(x) {
      let N = [];
      for (let [C, T] of Y) if (T < x) {
        let L = k.fetchers.get(C);
        ee(L, "Expected fetcher: " + C), L.state === "loading" && (sn(C), Y.delete(C), N.push(C));
      }
      return qi(N), N.length > 0;
    }
    function Tm(x, N) {
      let C = k.blockers.get(x) || Hr;
      return Oe.get(x) !== N && Oe.set(x, N), C;
    }
    function nu(x) {
      k.blockers.delete(x), Oe.delete(x);
    }
    function Il(x, N) {
      let C = k.blockers.get(x) || Hr;
      ee(C.state === "unblocked" && N.state === "blocked" || C.state === "blocked" && N.state === "blocked" || C.state === "blocked" && N.state === "proceeding" || C.state === "blocked" && N.state === "unblocked" || C.state === "proceeding" && N.state === "unblocked", "Invalid blocker state transition: " + C.state + " -> " + N.state);
      let T = new Map(k.blockers);
      T.set(x, N), B({
        blockers: T
      });
    }
    function ru(x) {
      let { currentLocation: N, nextLocation: C, historyAction: T } = x;
      if (Oe.size === 0) return;
      Oe.size > 1 && Jn(false, "A router only supports one blocker at a time");
      let L = Array.from(Oe.entries()), [V, G] = L[L.length - 1], z = k.blockers.get(V);
      if (!(z && z.state === "proceeding") && G({
        currentLocation: N,
        nextLocation: C,
        historyAction: T
      })) return V;
    }
    function xs(x) {
      let N = ft(404, {
        pathname: x
      }), C = u || s, { matches: T, route: L } = Pc(C);
      return ys(), {
        notFoundMatches: T,
        route: L,
        error: N
      };
    }
    function ys(x) {
      let N = [];
      return Te.forEach((C, T) => {
        (!x || x(T)) && (C.cancel(), N.push(T), Te.delete(T));
      }), N;
    }
    function Pm(x, N, C) {
      if (w = x, b = N, y = C || null, !h && k.navigation === Gs) {
        h = true;
        let T = au(k.location, k.matches);
        T != null && B({
          restoreScrollPosition: T
        });
      }
      return () => {
        w = null, b = null, y = null;
      };
    }
    function lu(x, N) {
      return y && y(x, N.map((T) => gg(T, k.loaderData))) || x.key;
    }
    function Mm(x, N) {
      if (w && b) {
        let C = lu(x, N);
        w[C] = b();
      }
    }
    function au(x, N) {
      if (w) {
        let C = lu(x, N), T = w[C];
        if (typeof T == "number") return T;
      }
      return null;
    }
    function zl(x, N, C) {
      if (m) if (x) {
        if (Object.keys(x[0].params).length > 0) return {
          active: true,
          matches: Sa(N, C, i, true)
        };
      } else return {
        active: true,
        matches: Sa(N, C, i, true) || []
      };
      return {
        active: false,
        matches: null
      };
    }
    async function Ul(x, N, C, T) {
      if (!m) return {
        type: "success",
        matches: x
      };
      let L = x;
      for (; ; ) {
        let V = u == null, G = u || s, z = a;
        try {
          await m({
            signal: C,
            path: N,
            matches: L,
            fetcherKey: T,
            patch: (F, J) => {
              C.aborted || Ec(F, J, G, z, l);
            }
          });
        } catch (F) {
          return {
            type: "error",
            error: F,
            partialMatches: L
          };
        } finally {
          V && !C.aborted && (s = [
            ...s
          ]);
        }
        if (C.aborted) return {
          type: "aborted"
        };
        let U = In(G, N, i);
        if (U) return {
          type: "success",
          matches: U
        };
        let A = Sa(G, N, i, true);
        if (!A || L.length === A.length && L.every((F, J) => F.route.id === A[J].route.id)) return {
          type: "success",
          matches: null
        };
        L = A;
      }
    }
    function Dm(x) {
      a = {}, u = Ga(x, l, void 0, a);
    }
    function Lm(x, N) {
      let C = u == null;
      Ec(x, N, u || s, a, l), C && (s = [
        ...s
      ], B({}));
    }
    return R = {
      get basename() {
        return i;
      },
      get future() {
        return d;
      },
      get state() {
        return k;
      },
      get routes() {
        return s;
      },
      get window() {
        return t;
      },
      initialize: me,
      subscribe: $,
      enableScrollRestoration: Pm,
      navigate: se,
      fetch: ct,
      revalidate: Se,
      createHref: (x) => e.history.createHref(x),
      encodeLocation: (x) => e.history.encodeLocation(x),
      getFetcher: Ol,
      deleteFetcher: Rm,
      dispose: he,
      getBlocker: Tm,
      deleteBlocker: nu,
      patchRoutes: Lm,
      _internalFetchControllers: M,
      _internalActiveDeferreds: Te,
      _internalSetRoutes: Dm
    }, R;
  }
  function Vg(e) {
    return e != null && ("formData" in e && e.formData != null || "body" in e && e.body !== void 0);
  }
  function Qo(e, t, n, r, l, a, s, u) {
    let i, c;
    if (s) {
      i = [];
      for (let d of t) if (i.push(d), d.route.id === s) {
        c = d;
        break;
      }
    } else i = t, c = t[t.length - 1];
    let m = cm(l || ".", um(i, a), Ll(e.pathname, n) || e.pathname, u === "path");
    if (l == null && (m.search = e.search, m.hash = e.hash), (l == null || l === "" || l === ".") && c) {
      let d = Gi(m.search);
      if (c.route.index && !d) m.search = m.search ? m.search.replace(/^\?/, "?index&") : "?index";
      else if (!c.route.index && d) {
        let p = new URLSearchParams(m.search), S = p.getAll("index");
        p.delete("index"), S.filter((y) => y).forEach((y) => p.append("index", y));
        let w = p.toString();
        m.search = w ? "?" + w : "";
      }
    }
    return r && n !== "/" && (m.pathname = m.pathname === "/" ? n : En([
      n,
      m.pathname
    ])), Dl(m);
  }
  function Sc(e, t, n, r) {
    if (!r || !Vg(r)) return {
      path: n
    };
    if (r.formMethod && !tv(r.formMethod)) return {
      path: n,
      error: ft(405, {
        method: r.formMethod
      })
    };
    let l = () => ({
      path: n,
      error: ft(400, {
        type: "invalid-body"
      })
    }), a = r.formMethod || "get", s = e ? a.toUpperCase() : a.toLowerCase(), u = hm(n);
    if (r.body !== void 0) {
      if (r.formEncType === "text/plain") {
        if (!It(s)) return l();
        let p = typeof r.body == "string" ? r.body : r.body instanceof FormData || r.body instanceof URLSearchParams ? Array.from(r.body.entries()).reduce((S, w) => {
          let [y, b] = w;
          return "" + S + y + "=" + b + `
`;
        }, "") : String(r.body);
        return {
          path: n,
          submission: {
            formMethod: s,
            formAction: u,
            formEncType: r.formEncType,
            formData: void 0,
            json: void 0,
            text: p
          }
        };
      } else if (r.formEncType === "application/json") {
        if (!It(s)) return l();
        try {
          let p = typeof r.body == "string" ? JSON.parse(r.body) : r.body;
          return {
            path: n,
            submission: {
              formMethod: s,
              formAction: u,
              formEncType: r.formEncType,
              formData: void 0,
              json: p,
              text: void 0
            }
          };
        } catch {
          return l();
        }
      }
    }
    ee(typeof FormData == "function", "FormData is not available in this environment");
    let i, c;
    if (r.formData) i = Go(r.formData), c = r.formData;
    else if (r.body instanceof FormData) i = Go(r.body), c = r.body;
    else if (r.body instanceof URLSearchParams) i = r.body, c = _c(i);
    else if (r.body == null) i = new URLSearchParams(), c = new FormData();
    else try {
      i = new URLSearchParams(r.body), c = _c(i);
    } catch {
      return l();
    }
    let m = {
      formMethod: s,
      formAction: u,
      formEncType: r && r.formEncType || "application/x-www-form-urlencoded",
      formData: c,
      json: void 0,
      text: void 0
    };
    if (It(m.formMethod)) return {
      path: n,
      submission: m
    };
    let d = Pn(n);
    return t && d.search && Gi(d.search) && i.append("index", ""), d.search = "?" + i, {
      path: Dl(d),
      submission: m
    };
  }
  function kc(e, t, n) {
    n === void 0 && (n = false);
    let r = e.findIndex((l) => l.route.id === t);
    return r >= 0 ? e.slice(0, n ? r + 1 : r) : e;
  }
  function Nc(e, t, n, r, l, a, s, u, i, c, m, d, p, S, w, y) {
    let b = y ? yt(y[1]) ? y[1].error : y[1].data : void 0, h = e.createURL(t.location), f = e.createURL(l), v = n;
    a && t.errors ? v = kc(n, Object.keys(t.errors)[0], true) : y && yt(y[1]) && (v = kc(n, y[0]));
    let E = y ? y[1].statusCode : void 0, _ = s && E && E >= 400, R = v.filter((j, I) => {
      let { route: D } = j;
      if (D.lazy) return true;
      if (D.loader == null) return false;
      if (a) return Ko(D, t.loaderData, t.errors);
      if (Wg(t.loaderData, t.matches[I], j) || i.some((ae) => ae === j.route.id)) return true;
      let Q = t.matches[I], K = j;
      return jc(j, Ee({
        currentUrl: h,
        currentParams: Q.params,
        nextUrl: f,
        nextParams: K.params
      }, r, {
        actionResult: b,
        actionStatus: E,
        defaultShouldRevalidate: _ ? false : u || h.pathname + h.search === f.pathname + f.search || h.search !== f.search || mm(Q, K)
      }));
    }), k = [];
    return d.forEach((j, I) => {
      if (a || !n.some((de) => de.route.id === j.routeId) || m.has(I)) return;
      let D = In(S, j.path, w);
      if (!D) {
        k.push({
          key: I,
          routeId: j.routeId,
          path: j.path,
          matches: null,
          match: null,
          controller: null
        });
        return;
      }
      let Q = t.fetchers.get(I), K = Zr(D, j.path), ae = false;
      p.has(I) ? ae = false : c.has(I) ? (c.delete(I), ae = true) : Q && Q.state !== "idle" && Q.data === void 0 ? ae = u : ae = jc(K, Ee({
        currentUrl: h,
        currentParams: t.matches[t.matches.length - 1].params,
        nextUrl: f,
        nextParams: n[n.length - 1].params
      }, r, {
        actionResult: b,
        actionStatus: E,
        defaultShouldRevalidate: _ ? false : u
      })), ae && k.push({
        key: I,
        routeId: j.routeId,
        path: j.path,
        matches: D,
        match: K,
        controller: new AbortController()
      });
    }), [
      R,
      k
    ];
  }
  function Ko(e, t, n) {
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
  function jc(e, t) {
    if (e.route.shouldRevalidate) {
      let n = e.route.shouldRevalidate(t);
      if (typeof n == "boolean") return n;
    }
    return t.defaultShouldRevalidate;
  }
  function Ec(e, t, n, r, l) {
    var a;
    let s;
    if (e) {
      let c = r[e];
      ee(c, "No route found to patch children into: routeId = " + e), c.children || (c.children = []), s = c.children;
    } else s = n;
    let u = t.filter((c) => !s.some((m) => pm(c, m))), i = Ga(u, l, [
      e || "_",
      "patch",
      String(((a = s) == null ? void 0 : a.length) || "0")
    ], r);
    s.push(...i);
  }
  function pm(e, t) {
    return "id" in e && "id" in t && e.id === t.id ? true : e.index === t.index && e.path === t.path && e.caseSensitive === t.caseSensitive ? (!e.children || e.children.length === 0) && (!t.children || t.children.length === 0) ? true : e.children.every((n, r) => {
      var l;
      return (l = t.children) == null ? void 0 : l.some((a) => pm(n, a));
    }) : false;
  }
  async function Hg(e, t, n) {
    if (!e.lazy) return;
    let r = await e.lazy();
    if (!e.lazy) return;
    let l = n[e.id];
    ee(l, "No route found in manifest");
    let a = {};
    for (let s in r) {
      let i = l[s] !== void 0 && s !== "hasErrorBoundary";
      Jn(!i, 'Route "' + l.id + '" has a static property "' + s + '" defined but its lazy function is also returning a value for this property. ' + ('The lazy route property "' + s + '" will be ignored.')), !i && !pg.has(s) && (a[s] = r[s]);
    }
    Object.assign(l, a), Object.assign(l, Ee({}, t(l), {
      lazy: void 0
    }));
  }
  async function Qg(e) {
    let { matches: t } = e, n = t.filter((l) => l.shouldLoad);
    return (await Promise.all(n.map((l) => l.resolve()))).reduce((l, a, s) => Object.assign(l, {
      [n[s].route.id]: a
    }), {});
  }
  async function Kg(e, t, n, r, l, a, s, u, i, c) {
    let m = a.map((S) => S.route.lazy ? Hg(S.route, i, u) : void 0), d = a.map((S, w) => {
      let y = m[w], b = l.some((f) => f.route.id === S.route.id);
      return Ee({}, S, {
        shouldLoad: b,
        resolve: async (f) => (f && r.method === "GET" && (S.route.lazy || S.route.loader) && (b = true), b ? Gg(t, r, S, y, f, c) : Promise.resolve({
          type: ue.data,
          result: void 0
        }))
      });
    }), p = await e({
      matches: d,
      request: r,
      params: a[0].params,
      fetcherKey: s,
      context: c
    });
    try {
      await Promise.all(m);
    } catch {
    }
    return p;
  }
  async function Gg(e, t, n, r, l, a) {
    let s, u, i = (c) => {
      let m, d = new Promise((w, y) => m = y);
      u = () => m(), t.signal.addEventListener("abort", u);
      let p = (w) => typeof c != "function" ? Promise.reject(new Error("You cannot call the handler for a route which defines a boolean " + ('"' + e + '" [routeId: ' + n.route.id + "]"))) : c({
        request: t,
        params: n.params,
        context: a
      }, ...w !== void 0 ? [
        w
      ] : []), S = (async () => {
        try {
          return {
            type: "data",
            result: await (l ? l((y) => p(y)) : p())
          };
        } catch (w) {
          return {
            type: "error",
            result: w
          };
        }
      })();
      return Promise.race([
        S,
        d
      ]);
    };
    try {
      let c = n.route[e];
      if (r) if (c) {
        let m, [d] = await Promise.all([
          i(c).catch((p) => {
            m = p;
          }),
          r
        ]);
        if (m !== void 0) throw m;
        s = d;
      } else if (await r, c = n.route[e], c) s = await i(c);
      else if (e === "action") {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw ft(405, {
          method: t.method,
          pathname: d,
          routeId: n.route.id
        });
      } else return {
        type: ue.data,
        result: void 0
      };
      else if (c) s = await i(c);
      else {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw ft(404, {
          pathname: d
        });
      }
      ee(s.result !== void 0, "You defined " + (e === "action" ? "an action" : "a loader") + " for route " + ('"' + n.route.id + "\" but didn't return anything from your `" + e + "` ") + "function. Please return a value or `null`.");
    } catch (c) {
      return {
        type: ue.error,
        result: c
      };
    } finally {
      u && t.signal.removeEventListener("abort", u);
    }
    return s;
  }
  async function Yg(e) {
    let { result: t, type: n } = e;
    if (gm(t)) {
      let d;
      try {
        let p = t.headers.get("Content-Type");
        p && /\bapplication\/json\b/.test(p) ? t.body == null ? d = null : d = await t.json() : d = await t.text();
      } catch (p) {
        return {
          type: ue.error,
          error: p
        };
      }
      return n === ue.error ? {
        type: ue.error,
        error: new Ya(t.status, t.statusText, d),
        statusCode: t.status,
        headers: t.headers
      } : {
        type: ue.data,
        data: d,
        statusCode: t.status,
        headers: t.headers
      };
    }
    if (n === ue.error) {
      if (Mc(t)) {
        var r, l;
        if (t.data instanceof Error) {
          var a, s;
          return {
            type: ue.error,
            error: t.data,
            statusCode: (a = t.init) == null ? void 0 : a.status,
            headers: (s = t.init) != null && s.headers ? new Headers(t.init.headers) : void 0
          };
        }
        return {
          type: ue.error,
          error: new Ya(((r = t.init) == null ? void 0 : r.status) || 500, void 0, t.data),
          statusCode: _l(t) ? t.status : void 0,
          headers: (l = t.init) != null && l.headers ? new Headers(t.init.headers) : void 0
        };
      }
      return {
        type: ue.error,
        error: t,
        statusCode: _l(t) ? t.status : void 0
      };
    }
    if (ev(t)) {
      var u, i;
      return {
        type: ue.deferred,
        deferredData: t,
        statusCode: (u = t.init) == null ? void 0 : u.status,
        headers: ((i = t.init) == null ? void 0 : i.headers) && new Headers(t.init.headers)
      };
    }
    if (Mc(t)) {
      var c, m;
      return {
        type: ue.data,
        data: t.data,
        statusCode: (c = t.init) == null ? void 0 : c.status,
        headers: (m = t.init) != null && m.headers ? new Headers(t.init.headers) : void 0
      };
    }
    return {
      type: ue.data,
      data: t
    };
  }
  function Jg(e, t, n, r, l, a) {
    let s = e.headers.get("Location");
    if (ee(s, "Redirects returned/thrown from loaders/actions must have a Location header"), !Qi.test(s)) {
      let u = r.slice(0, r.findIndex((i) => i.route.id === n) + 1);
      s = Qo(new URL(t.url), u, l, true, s, a), e.headers.set("Location", s);
    }
    return e;
  }
  function Cc(e, t, n, r) {
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
    if (Qi.test(e)) {
      let a = e, s = a.startsWith("//") ? new URL(t.protocol + a) : new URL(a);
      if (l.includes(s.protocol)) throw new Error("Invalid redirect location");
      let u = Ll(s.pathname, n) != null;
      if (s.origin === t.origin && u) return s.pathname + s.search + s.hash;
    }
    try {
      let a = r.createURL(e);
      if (l.includes(a.protocol)) throw new Error("Invalid redirect location");
    } catch {
    }
    return e;
  }
  function lr(e, t, n, r) {
    let l = e.createURL(hm(t)).toString(), a = {
      signal: n
    };
    if (r && It(r.formMethod)) {
      let { formMethod: s, formEncType: u } = r;
      a.method = s.toUpperCase(), u === "application/json" ? (a.headers = new Headers({
        "Content-Type": u
      }), a.body = JSON.stringify(r.json)) : u === "text/plain" ? a.body = r.text : u === "application/x-www-form-urlencoded" && r.formData ? a.body = Go(r.formData) : a.body = r.formData;
    }
    return new Request(l, a);
  }
  function Go(e) {
    let t = new URLSearchParams();
    for (let [n, r] of e.entries()) t.append(n, typeof r == "string" ? r : r.name);
    return t;
  }
  function _c(e) {
    let t = new FormData();
    for (let [n, r] of e.entries()) t.append(n, r);
    return t;
  }
  function Xg(e, t, n, r, l) {
    let a = {}, s = null, u, i = false, c = {}, m = n && yt(n[1]) ? n[1].error : void 0;
    return e.forEach((d) => {
      if (!(d.route.id in t)) return;
      let p = d.route.id, S = t[p];
      if (ee(!Bn(S), "Cannot handle redirect results in processLoaderData"), yt(S)) {
        let w = S.error;
        m !== void 0 && (w = m, m = void 0), s = s || {};
        {
          let y = zn(e, p);
          s[y.route.id] == null && (s[y.route.id] = w);
        }
        a[p] = void 0, i || (i = true, u = _l(S.error) ? S.error.status : 500), S.headers && (c[p] = S.headers);
      } else gn(S) ? (r.set(p, S.deferredData), a[p] = S.deferredData.data, S.statusCode != null && S.statusCode !== 200 && !i && (u = S.statusCode), S.headers && (c[p] = S.headers)) : (a[p] = S.data, S.statusCode && S.statusCode !== 200 && !i && (u = S.statusCode), S.headers && (c[p] = S.headers));
    }), m !== void 0 && n && (s = {
      [n[0]]: m
    }, a[n[0]] = void 0), {
      loaderData: a,
      errors: s,
      statusCode: u || 200,
      loaderHeaders: c
    };
  }
  function bc(e, t, n, r, l, a, s) {
    let { loaderData: u, errors: i } = Xg(t, n, r, s);
    return l.forEach((c) => {
      let { key: m, match: d, controller: p } = c, S = a[m];
      if (ee(S, "Did not find corresponding fetcher result"), !(p && p.signal.aborted)) if (yt(S)) {
        let w = zn(e.matches, d == null ? void 0 : d.route.id);
        i && i[w.route.id] || (i = Ee({}, i, {
          [w.route.id]: S.error
        })), e.fetchers.delete(m);
      } else if (Bn(S)) ee(false, "Unhandled fetcher revalidation redirect");
      else if (gn(S)) ee(false, "Unhandled fetcher deferred data");
      else {
        let w = un(S.data);
        e.fetchers.set(m, w);
      }
    }), {
      loaderData: u,
      errors: i
    };
  }
  function Rc(e, t, n, r) {
    let l = Ee({}, t);
    for (let a of n) {
      let s = a.route.id;
      if (t.hasOwnProperty(s) ? t[s] !== void 0 && (l[s] = t[s]) : e[s] !== void 0 && a.route.loader && (l[s] = e[s]), r && r.hasOwnProperty(s)) break;
    }
    return l;
  }
  function Tc(e) {
    return e ? yt(e[1]) ? {
      actionData: {}
    } : {
      actionData: {
        [e[0]]: e[1].data
      }
    } : {};
  }
  function zn(e, t) {
    return (t ? e.slice(0, e.findIndex((r) => r.route.id === t) + 1) : [
      ...e
    ]).reverse().find((r) => r.route.hasErrorBoundary === true) || e[0];
  }
  function Pc(e) {
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
  function ft(e, t) {
    let { pathname: n, routeId: r, method: l, type: a, message: s } = t === void 0 ? {} : t, u = "Unknown Server Error", i = "Unknown @remix-run/router error";
    return e === 400 ? (u = "Bad Request", l && n && r ? i = "You made a " + l + ' request to "' + n + '" but ' + ('did not provide a `loader` for route "' + r + '", ') + "so there is no way to handle the request." : a === "defer-action" ? i = "defer() is not supported in actions" : a === "invalid-body" && (i = "Unable to encode submission body")) : e === 403 ? (u = "Forbidden", i = 'Route "' + r + '" does not match URL "' + n + '"') : e === 404 ? (u = "Not Found", i = 'No route matches URL "' + n + '"') : e === 405 && (u = "Method Not Allowed", l && n && r ? i = "You made a " + l.toUpperCase() + ' request to "' + n + '" but ' + ('did not provide an `action` for route "' + r + '", ') + "so there is no way to handle the request." : l && (i = 'Invalid request method "' + l.toUpperCase() + '"')), new Ya(e || 500, u, new Error(i), true);
  }
  function aa(e) {
    let t = Object.entries(e);
    for (let n = t.length - 1; n >= 0; n--) {
      let [r, l] = t[n];
      if (Bn(l)) return {
        key: r,
        result: l
      };
    }
  }
  function hm(e) {
    let t = typeof e == "string" ? Pn(e) : e;
    return Dl(Ee({}, t, {
      hash: ""
    }));
  }
  function Zg(e, t) {
    return e.pathname !== t.pathname || e.search !== t.search ? false : e.hash === "" ? t.hash !== "" : e.hash === t.hash ? true : t.hash !== "";
  }
  function qg(e) {
    return gm(e.result) && zg.has(e.result.status);
  }
  function gn(e) {
    return e.type === ue.deferred;
  }
  function yt(e) {
    return e.type === ue.error;
  }
  function Bn(e) {
    return (e && e.type) === ue.redirect;
  }
  function Mc(e) {
    return typeof e == "object" && e != null && "type" in e && "data" in e && "init" in e && e.type === "DataWithResponseInit";
  }
  function ev(e) {
    let t = e;
    return t && typeof t == "object" && typeof t.data == "object" && typeof t.subscribe == "function" && typeof t.cancel == "function" && typeof t.resolveData == "function";
  }
  function gm(e) {
    return e != null && typeof e.status == "number" && typeof e.statusText == "string" && typeof e.headers == "object" && typeof e.body < "u";
  }
  function tv(e) {
    return Ig.has(e.toLowerCase());
  }
  function It(e) {
    return Og.has(e.toLowerCase());
  }
  async function nv(e, t, n, r, l) {
    let a = Object.entries(t);
    for (let s = 0; s < a.length; s++) {
      let [u, i] = a[s], c = e.find((p) => (p == null ? void 0 : p.route.id) === u);
      if (!c) continue;
      let m = r.find((p) => p.route.id === c.route.id), d = m != null && !mm(m, c) && (l && l[c.route.id]) !== void 0;
      gn(i) && d && await Ki(i, n, false).then((p) => {
        p && (t[u] = p);
      });
    }
  }
  async function rv(e, t, n) {
    for (let r = 0; r < n.length; r++) {
      let { key: l, routeId: a, controller: s } = n[r], u = t[l];
      e.find((c) => (c == null ? void 0 : c.route.id) === a) && gn(u) && (ee(s, "Expected an AbortController for revalidating fetcher deferred result"), await Ki(u, s.signal, true).then((c) => {
        c && (t[l] = c);
      }));
    }
  }
  async function Ki(e, t, n) {
    if (n === void 0 && (n = false), !await e.deferredData.resolveData(t)) {
      if (n) try {
        return {
          type: ue.data,
          data: e.deferredData.unwrappedData
        };
      } catch (l) {
        return {
          type: ue.error,
          error: l
        };
      }
      return {
        type: ue.data,
        data: e.deferredData.data
      };
    }
  }
  function Gi(e) {
    return new URLSearchParams(e).getAll("index").some((t) => t === "");
  }
  function Zr(e, t) {
    let n = typeof t == "string" ? Pn(t).search : t.search;
    if (e[e.length - 1].route.index && Gi(n || "")) return e[e.length - 1];
    let r = im(e);
    return r[r.length - 1];
  }
  function Dc(e) {
    let { formMethod: t, formAction: n, formEncType: r, text: l, formData: a, json: s } = e;
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
      if (s !== void 0) return {
        formMethod: t,
        formAction: n,
        formEncType: r,
        formData: void 0,
        json: s,
        text: void 0
      };
    }
  }
  function Ys(e, t) {
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
  function lv(e, t) {
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
  function Qr(e, t) {
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
  function av(e, t) {
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
  function un(e) {
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
  function sv(e, t) {
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
        Jn(false, "Failed to save applied view transitions in sessionStorage (" + r + ").");
      }
    }
  }
  function Ja() {
    return Ja = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ja.apply(this, arguments);
  }
  const ps = g.createContext(null), vm = g.createContext(null), hs = g.createContext(null), Yi = g.createContext(null), qn = g.createContext({
    outlet: null,
    matches: [],
    isDataRoute: false
  }), xm = g.createContext(null);
  function gs() {
    return g.useContext(Yi) != null;
  }
  function Ji() {
    return gs() || ee(false), g.useContext(Yi).location;
  }
  function ym(e) {
    g.useContext(hs).static || g.useLayoutEffect(e);
  }
  function vs() {
    let { isDataRoute: e } = g.useContext(qn);
    return e ? Sv() : iv();
  }
  function iv() {
    gs() || ee(false);
    let e = g.useContext(ps), { basename: t, future: n, navigator: r } = g.useContext(hs), { matches: l } = g.useContext(qn), { pathname: a } = Ji(), s = JSON.stringify(um(l, n.v7_relativeSplatPath)), u = g.useRef(false);
    return ym(() => {
      u.current = true;
    }), g.useCallback(function(c, m) {
      if (m === void 0 && (m = {}), !u.current) return;
      if (typeof c == "number") {
        r.go(c);
        return;
      }
      let d = cm(c, JSON.parse(s), a, m.relative === "path");
      e == null && t !== "/" && (d.pathname = d.pathname === "/" ? t : En([
        t,
        d.pathname
      ])), (m.replace ? r.replace : r.push)(d, m.state, m);
    }, [
      t,
      r,
      s,
      a,
      e
    ]);
  }
  const uv = g.createContext(null);
  function cv(e) {
    let t = g.useContext(qn).outlet;
    return t && g.createElement(uv.Provider, {
      value: e
    }, t);
  }
  function dv(e, t, n, r) {
    gs() || ee(false);
    let { navigator: l } = g.useContext(hs), { matches: a } = g.useContext(qn), s = a[a.length - 1], u = s ? s.params : {};
    s && s.pathname;
    let i = s ? s.pathnameBase : "/";
    s && s.route;
    let c = Ji(), m;
    m = c;
    let d = m.pathname || "/", p = d;
    if (i !== "/") {
      let y = i.replace(/^\//, "").split("/");
      p = "/" + d.replace(/^\//, "").split("/").slice(y.length).join("/");
    }
    let S = In(e, {
      pathname: p
    });
    return gv(S && S.map((y) => Object.assign({}, y, {
      params: Object.assign({}, u, y.params),
      pathname: En([
        i,
        l.encodeLocation ? l.encodeLocation(y.pathname).pathname : y.pathname
      ]),
      pathnameBase: y.pathnameBase === "/" ? i : En([
        i,
        l.encodeLocation ? l.encodeLocation(y.pathnameBase).pathname : y.pathnameBase
      ])
    })), a, n, r);
  }
  function fv() {
    let e = wv(), t = _l(e) ? e.status + " " + e.statusText : e instanceof Error ? e.message : JSON.stringify(e), n = e instanceof Error ? e.stack : null, l = {
      padding: "0.5rem",
      backgroundColor: "rgba(200,200,200, 0.5)"
    };
    return g.createElement(g.Fragment, null, g.createElement("h2", null, "Unexpected Application Error!"), g.createElement("h3", {
      style: {
        fontStyle: "italic"
      }
    }, t), n ? g.createElement("pre", {
      style: l
    }, n) : null, null);
  }
  const mv = g.createElement(fv, null);
  class pv extends g.Component {
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
      return this.state.error !== void 0 ? g.createElement(qn.Provider, {
        value: this.props.routeContext
      }, g.createElement(xm.Provider, {
        value: this.state.error,
        children: this.props.component
      })) : this.props.children;
    }
  }
  function hv(e) {
    let { routeContext: t, match: n, children: r } = e, l = g.useContext(ps);
    return l && l.static && l.staticContext && (n.route.errorElement || n.route.ErrorBoundary) && (l.staticContext._deepestRenderedBoundaryId = n.route.id), g.createElement(qn.Provider, {
      value: t
    }, r);
  }
  function gv(e, t, n, r) {
    var l;
    if (t === void 0 && (t = []), n === void 0 && (n = null), r === void 0 && (r = null), e == null) {
      var a;
      if (!n) return null;
      if (n.errors) e = n.matches;
      else if ((a = r) != null && a.v7_partialHydration && t.length === 0 && !n.initialized && n.matches.length > 0) e = n.matches;
      else return null;
    }
    let s = e, u = (l = n) == null ? void 0 : l.errors;
    if (u != null) {
      let m = s.findIndex((d) => d.route.id && (u == null ? void 0 : u[d.route.id]) !== void 0);
      m >= 0 || ee(false), s = s.slice(0, Math.min(s.length, m + 1));
    }
    let i = false, c = -1;
    if (n && r && r.v7_partialHydration) for (let m = 0; m < s.length; m++) {
      let d = s[m];
      if ((d.route.HydrateFallback || d.route.hydrateFallbackElement) && (c = m), d.route.id) {
        let { loaderData: p, errors: S } = n, w = d.route.loader && p[d.route.id] === void 0 && (!S || S[d.route.id] === void 0);
        if (d.route.lazy || w) {
          i = true, c >= 0 ? s = s.slice(0, c + 1) : s = [
            s[0]
          ];
          break;
        }
      }
    }
    return s.reduceRight((m, d, p) => {
      let S, w = false, y = null, b = null;
      n && (S = u && d.route.id ? u[d.route.id] : void 0, y = d.route.errorElement || mv, i && (c < 0 && p === 0 ? (kv("route-fallback"), w = true, b = null) : c === p && (w = true, b = d.route.hydrateFallbackElement || null)));
      let h = t.concat(s.slice(0, p + 1)), f = () => {
        let v;
        return S ? v = y : w ? v = b : d.route.Component ? v = g.createElement(d.route.Component, null) : d.route.element ? v = d.route.element : v = m, g.createElement(hv, {
          match: d,
          routeContext: {
            outlet: m,
            matches: h,
            isDataRoute: n != null
          },
          children: v
        });
      };
      return n && (d.route.ErrorBoundary || d.route.errorElement || p === 0) ? g.createElement(pv, {
        location: n.location,
        revalidation: n.revalidation,
        component: y,
        error: S,
        children: f(),
        routeContext: {
          outlet: null,
          matches: h,
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
  function vv(e) {
    let t = g.useContext(ps);
    return t || ee(false), t;
  }
  function xv(e) {
    let t = g.useContext(vm);
    return t || ee(false), t;
  }
  function yv(e) {
    let t = g.useContext(qn);
    return t || ee(false), t;
  }
  function km(e) {
    let t = yv(), n = t.matches[t.matches.length - 1];
    return n.route.id || ee(false), n.route.id;
  }
  function wv() {
    var e;
    let t = g.useContext(xm), n = xv(Sm.UseRouteError), r = km();
    return t !== void 0 ? t : (e = n.errors) == null ? void 0 : e[r];
  }
  function Sv() {
    let { router: e } = vv(wm.UseNavigateStable), t = km(), n = g.useRef(false);
    return ym(() => {
      n.current = true;
    }), g.useCallback(function(l, a) {
      a === void 0 && (a = {}), n.current && (typeof l == "number" ? e.navigate(l) : e.navigate(l, Ja({
        fromRouteId: t
      }, a)));
    }, [
      e,
      t
    ]);
  }
  const Lc = {};
  function kv(e, t, n) {
    Lc[e] || (Lc[e] = true);
  }
  function Nv(e, t) {
    e == null ? void 0 : e.v7_startTransition, (e == null ? void 0 : e.v7_relativeSplatPath) === void 0 && (!t || t.v7_relativeSplatPath), t && (t.v7_fetcherPersist, t.v7_normalizeFormMethod, t.v7_partialHydration, t.v7_skipActionErrorRevalidation);
  }
  function jv(e) {
    return cv(e.context);
  }
  function Ev(e) {
    let { basename: t = "/", children: n = null, location: r, navigationType: l = Ie.Pop, navigator: a, static: s = false, future: u } = e;
    gs() && ee(false);
    let i = t.replace(/^\/*/, "/"), c = g.useMemo(() => ({
      basename: i,
      navigator: a,
      static: s,
      future: Ja({
        v7_relativeSplatPath: false
      }, u)
    }), [
      i,
      u,
      a,
      s
    ]);
    typeof r == "string" && (r = Pn(r));
    let { pathname: m = "/", search: d = "", hash: p = "", state: S = null, key: w = "default" } = r, y = g.useMemo(() => {
      let b = Ll(m, i);
      return b == null ? null : {
        location: {
          pathname: b,
          search: d,
          hash: p,
          state: S,
          key: w
        },
        navigationType: l
      };
    }, [
      i,
      m,
      d,
      p,
      S,
      w,
      l
    ]);
    return y == null ? null : g.createElement(hs.Provider, {
      value: c
    }, g.createElement(Yi.Provider, {
      children: n,
      value: y
    }));
  }
  new Promise(() => {
  });
  function Cv(e) {
    let t = {
      hasErrorBoundary: e.ErrorBoundary != null || e.errorElement != null
    };
    return e.Component && Object.assign(t, {
      element: g.createElement(e.Component),
      Component: void 0
    }), e.HydrateFallback && Object.assign(t, {
      hydrateFallbackElement: g.createElement(e.HydrateFallback),
      HydrateFallback: void 0
    }), e.ErrorBoundary && Object.assign(t, {
      errorElement: g.createElement(e.ErrorBoundary),
      ErrorBoundary: void 0
    }), t;
  }
  function Xa() {
    return Xa = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Xa.apply(this, arguments);
  }
  const _v = "6";
  try {
    window.__reactRouterVersion = _v;
  } catch {
  }
  function bv(e, t) {
    return Bg({
      basename: void 0,
      future: Xa({}, void 0, {
        v7_prependBasename: true
      }),
      history: dg({
        window: void 0
      }),
      hydrationData: Rv(),
      routes: e,
      mapRouteProperties: Cv,
      dataStrategy: void 0,
      patchRoutesOnNavigation: void 0,
      window: void 0
    }).initialize();
  }
  function Rv() {
    var e;
    let t = (e = window) == null ? void 0 : e.__staticRouterHydrationData;
    return t && t.errors && (t = Xa({}, t, {
      errors: Tv(t.errors)
    })), t;
  }
  function Tv(e) {
    if (!e) return null;
    let t = Object.entries(e), n = {};
    for (let [r, l] of t) if (l && l.__type === "RouteErrorResponse") n[r] = new Ya(l.status, l.statusText, l.data, l.internal === true);
    else if (l && l.__type === "Error") {
      if (l.__subType) {
        let a = window[l.__subType];
        if (typeof a == "function") try {
          let s = new a(l.message);
          s.stack = "", n[r] = s;
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
  const Pv = g.createContext({
    isTransitioning: false
  }), Mv = g.createContext(/* @__PURE__ */ new Map()), Dv = "startTransition", Oc = Zm[Dv], Lv = "flushSync", Ac = cg[Lv];
  function Ov(e) {
    Oc ? Oc(e) : e();
  }
  function Kr(e) {
    Ac ? Ac(e) : e();
  }
  class Av {
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
    let { fallbackElement: t, router: n, future: r } = e, [l, a] = g.useState(n.state), [s, u] = g.useState(), [i, c] = g.useState({
      isTransitioning: false
    }), [m, d] = g.useState(), [p, S] = g.useState(), [w, y] = g.useState(), b = g.useRef(/* @__PURE__ */ new Map()), { v7_startTransition: h } = r || {}, f = g.useCallback((j) => {
      h ? Ov(j) : j();
    }, [
      h
    ]), v = g.useCallback((j, I) => {
      let { deletedFetchers: D, flushSync: Q, viewTransitionOpts: K } = I;
      j.fetchers.forEach((de, ge) => {
        de.data !== void 0 && b.current.set(ge, de.data);
      }), D.forEach((de) => b.current.delete(de));
      let ae = n.window == null || n.window.document == null || typeof n.window.document.startViewTransition != "function";
      if (!K || ae) {
        Q ? Kr(() => a(j)) : f(() => a(j));
        return;
      }
      if (Q) {
        Kr(() => {
          p && (m && m.resolve(), p.skipTransition()), c({
            isTransitioning: true,
            flushSync: true,
            currentLocation: K.currentLocation,
            nextLocation: K.nextLocation
          });
        });
        let de = n.window.document.startViewTransition(() => {
          Kr(() => a(j));
        });
        de.finished.finally(() => {
          Kr(() => {
            d(void 0), S(void 0), u(void 0), c({
              isTransitioning: false
            });
          });
        }), Kr(() => S(de));
        return;
      }
      p ? (m && m.resolve(), p.skipTransition(), y({
        state: j,
        currentLocation: K.currentLocation,
        nextLocation: K.nextLocation
      })) : (u(j), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: K.currentLocation,
        nextLocation: K.nextLocation
      }));
    }, [
      n.window,
      p,
      m,
      b,
      f
    ]);
    g.useLayoutEffect(() => n.subscribe(v), [
      n,
      v
    ]), g.useEffect(() => {
      i.isTransitioning && !i.flushSync && d(new Av());
    }, [
      i
    ]), g.useEffect(() => {
      if (m && s && n.window) {
        let j = s, I = m.promise, D = n.window.document.startViewTransition(async () => {
          f(() => a(j)), await I;
        });
        D.finished.finally(() => {
          d(void 0), S(void 0), u(void 0), c({
            isTransitioning: false
          });
        }), S(D);
      }
    }, [
      f,
      s,
      m,
      n.window
    ]), g.useEffect(() => {
      m && s && l.location.key === s.location.key && m.resolve();
    }, [
      m,
      p,
      l.location,
      s
    ]), g.useEffect(() => {
      !i.isTransitioning && w && (u(w.state), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: w.currentLocation,
        nextLocation: w.nextLocation
      }), y(void 0));
    }, [
      i.isTransitioning,
      w
    ]), g.useEffect(() => {
    }, []);
    let E = g.useMemo(() => ({
      createHref: n.createHref,
      encodeLocation: n.encodeLocation,
      go: (j) => n.navigate(j),
      push: (j, I, D) => n.navigate(j, {
        state: I,
        preventScrollReset: D == null ? void 0 : D.preventScrollReset
      }),
      replace: (j, I, D) => n.navigate(j, {
        replace: true,
        state: I,
        preventScrollReset: D == null ? void 0 : D.preventScrollReset
      })
    }), [
      n
    ]), _ = n.basename || "/", R = g.useMemo(() => ({
      router: n,
      navigator: E,
      static: false,
      basename: _
    }), [
      n,
      E,
      _
    ]), k = g.useMemo(() => ({
      v7_relativeSplatPath: n.future.v7_relativeSplatPath
    }), [
      n.future.v7_relativeSplatPath
    ]);
    return g.useEffect(() => Nv(r, n.future), [
      r,
      n.future
    ]), g.createElement(g.Fragment, null, g.createElement(ps.Provider, {
      value: R
    }, g.createElement(vm.Provider, {
      value: l
    }, g.createElement(Mv.Provider, {
      value: b.current
    }, g.createElement(Pv.Provider, {
      value: i
    }, g.createElement(Ev, {
      basename: _,
      location: l.location,
      navigationType: l.historyAction,
      navigator: E,
      future: k
    }, l.initialized || n.future.v7_partialHydration ? g.createElement(zv, {
      routes: n.routes,
      future: n.future,
      state: l
    }) : t))))), null);
  }
  const zv = g.memo(Uv);
  function Uv(e) {
    let { routes: t, future: n, state: r } = e;
    return dv(t, void 0, r, n);
  }
  var Ic;
  (function(e) {
    e.UseScrollRestoration = "useScrollRestoration", e.UseSubmit = "useSubmit", e.UseSubmitFetcher = "useSubmitFetcher", e.UseFetcher = "useFetcher", e.useViewTransitionState = "useViewTransitionState";
  })(Ic || (Ic = {}));
  var zc;
  (function(e) {
    e.UseFetcher = "useFetcher", e.UseFetchers = "useFetchers", e.UseScrollRestoration = "useScrollRestoration";
  })(zc || (zc = {}));
  const $v = "https://100.107.132.16:30000";
  function Fv() {
    const e = window.location;
    return e.protocol === "file:" || e.protocol === "capacitor:" || e.protocol === "ionic:" || e.hostname === "localhost" || e.hostname === "127.0.0.1" || e.hostname === "";
  }
  function le() {
    const e = window.location;
    return !Fv() && (e.protocol === "http:" || e.protocol === "https:") ? `${e.protocol}//${e.host}` : $v;
  }
  function Bv(e) {
    const t = new URL(le());
    return t.protocol = t.protocol === "https:" ? "wss:" : "ws:", `${t.protocol}//${t.host}${e}`;
  }
  async function Vv() {
    try {
      if ("serviceWorker" in navigator) {
        const e = await navigator.serviceWorker.getRegistrations();
        await Promise.all(e.map((t) => t.unregister()));
      }
    } catch {
    }
    try {
      if ("caches" in window) {
        const e = await caches.keys();
        await Promise.all(e.map((t) => caches.delete(t)));
      }
    } catch {
    }
  }
  function Wv(e) {
    if (!(e instanceof Error)) return false;
    const t = e.message.toLowerCase();
    return e.name === "TypeError" && (t.includes("load failed") || t.includes("failed to fetch"));
  }
  async function Nm(e, t = {}, n = 1e4) {
    const r = e.startsWith("http") ? e : `${le()}${e}`, l = async () => {
      const a = new AbortController(), s = window.setTimeout(() => a.abort(), n);
      try {
        return await fetch(r, {
          ...t,
          cache: "no-store",
          signal: t.signal ?? a.signal
        });
      } finally {
        window.clearTimeout(s);
      }
    };
    try {
      return await l();
    } catch (a) {
      if (!Wv(a)) throw a;
      return await Vv(), l();
    }
  }
  const Hv = [
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
      description: "Voice Chat"
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
  ], Qv = () => {
    const e = vs(), [t, n] = g.useState(/* @__PURE__ */ new Date()), [r, l] = g.useState(null);
    return g.useEffect(() => {
      const a = setInterval(() => n(/* @__PURE__ */ new Date()), 1e3);
      return () => clearInterval(a);
    }, []), g.useEffect(() => {
      const a = () => {
        fetch(`${le()}/health`).then((u) => u.json()).then(l).catch(() => l(null));
      };
      a();
      const s = setInterval(a, 3e4);
      return () => clearInterval(s);
    }, []), o.jsxs("div", {
      className: "flex h-full flex-col px-4 pt-6",
      children: [
        o.jsxs("div", {
          className: "mb-2 text-center",
          children: [
            o.jsx("h1", {
              className: "fire-gradient text-4xl font-black tracking-tight",
              children: "MAUDE"
            }),
            o.jsx("p", {
              className: "mt-1 text-xs text-maude-muted",
              children: "Multi-Agent Unified Dispatch Engine"
            })
          ]
        }),
        o.jsxs("div", {
          className: "mb-4 text-center",
          children: [
            o.jsx("div", {
              className: "text-5xl font-light tabular-nums text-maude-text",
              children: t.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit"
              })
            }),
            o.jsx("div", {
              className: "mt-1 text-sm text-maude-muted",
              children: t.toLocaleDateString([], {
                weekday: "long",
                month: "long",
                day: "numeric"
              })
            })
          ]
        }),
        o.jsxs("div", {
          className: "mb-4 flex items-center justify-center gap-3 text-xs",
          children: [
            o.jsxs("span", {
              className: `flex items-center gap-1 ${(r == null ? void 0 : r.status) ? "text-green-400" : "text-red-400"}`,
              children: [
                o.jsx("span", {
                  className: `inline-block h-2 w-2 rounded-full ${(r == null ? void 0 : r.status) ? "bg-green-400" : "bg-red-400"}`
                }),
                "Spark ",
                (r == null ? void 0 : r.status) ? "Connected" : "Offline"
              ]
            }),
            o.jsx("span", {
              className: "text-maude-muted",
              children: "|"
            }),
            o.jsx("span", {
              className: "text-maude-muted",
              children: "Tailscale Active"
            })
          ]
        }),
        o.jsx("div", {
          className: "grid flex-1 grid-cols-3 gap-3 content-start",
          children: Hv.map((a) => o.jsxs("button", {
            onClick: () => e(a.path),
            className: "flex flex-col items-center justify-center rounded-2xl bg-maude-surface p-4 transition-all active:scale-95 hover:bg-maude-card",
            children: [
              o.jsx("span", {
                className: "mb-2 text-3xl",
                children: a.icon
              }),
              o.jsx("span", {
                className: "text-sm font-medium text-maude-text",
                children: a.label
              }),
              o.jsx("span", {
                className: "mt-0.5 text-[10px] text-maude-muted",
                children: a.description
              })
            ]
          }, a.path))
        })
      ]
    });
  };
  function jm() {
    return le();
  }
  const Gt = {
    index: "maude-conversations",
    messages: (e) => `maude-conv-msgs:${e}`,
    active: "maude-active-conv"
  };
  async function Em(e) {
    try {
      const t = await fetch(`${jm()}${e}`);
      return t.ok ? await t.json() : null;
    } catch {
      return null;
    }
  }
  function Xi(e, t) {
    fetch(`${jm()}${e}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(t)
    }).catch(() => {
    });
  }
  function Cm() {
    try {
      const e = localStorage.getItem(Gt.index);
      return e ? JSON.parse(e) : [];
    } catch {
      return [];
    }
  }
  async function Kv() {
    const e = await Em("/api/conversations");
    return e && e.length > 0 ? (localStorage.setItem(Gt.index, JSON.stringify(e)), e) : Cm();
  }
  function Gv(e) {
    localStorage.setItem(Gt.index, JSON.stringify(e)), Xi("/api/conversations", e);
  }
  function Yo(e) {
    try {
      const t = localStorage.getItem(Gt.messages(e));
      return t ? JSON.parse(t) : [];
    } catch {
      return [];
    }
  }
  async function Yv(e) {
    const t = await Em(`/api/conversations/${e}/messages`);
    return t && t.length > 0 ? (localStorage.setItem(Gt.messages(e), JSON.stringify(t)), t) : Yo(e);
  }
  function _m(e, t) {
    localStorage.setItem(Gt.messages(e), JSON.stringify(t)), Xi(`/api/conversations/${e}/messages`, t);
  }
  function Jv(e) {
    localStorage.removeItem(Gt.messages(e)), Xi(`/api/conversations/${e}/delete`, {});
  }
  function Xv() {
    return localStorage.getItem(Gt.active);
  }
  function sa(e) {
    e === null ? localStorage.removeItem(Gt.active) : localStorage.setItem(Gt.active, e);
  }
  const Jo = () => typeof crypto.randomUUID == "function" ? crypto.randomUUID() : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (e) => {
    const t = Math.random() * 16 | 0;
    return (e === "x" ? t : t & 3 | 8).toString(16);
  }), Zv = /iPad|iPhone|iPod/.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1, qv = {
    "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
    "nvidia/nemotron-3-nano-30b-a3b": "nemotron-a3b",
    "nemotron-nano": "nemotron-a3b",
    a3b: "nemotron-a3b",
    "codex-cli": "codex"
  };
  function oa(e) {
    const t = (e || "").trim();
    return !t || t === "claude-opus-4-20250514" ? "nemotron-super" : qv[t] || t;
  }
  let ar = null;
  function ex() {
    return ar && Date.now() - ar.ts < 3e5 ? Promise.resolve(ar) : navigator.geolocation ? new Promise((e) => {
      navigator.geolocation.getCurrentPosition((t) => {
        ar = {
          lat: t.coords.latitude,
          lng: t.coords.longitude,
          accuracy: t.coords.accuracy,
          ts: Date.now()
        }, e(ar);
      }, () => e(ar), {
        timeout: 5e3,
        maximumAge: 3e5
      });
    }) : Promise.resolve(null);
  }
  async function tx() {
    try {
      if ("serviceWorker" in navigator) {
        const e = await navigator.serviceWorker.getRegistrations();
        await Promise.all(e.map((t) => t.unregister()));
      }
    } catch {
    }
    try {
      if ("caches" in window) {
        const e = await caches.keys();
        await Promise.all(e.map((t) => caches.delete(t)));
      }
    } catch {
    }
  }
  function ka(e) {
    if (!(e instanceof Error)) return false;
    const t = e.message.toLowerCase();
    return e.name === "TypeError" && (t.includes("load failed") || t.includes("failed to fetch"));
  }
  async function ia(e, t) {
    try {
      return await fetch(e, {
        ...t,
        cache: "no-store"
      });
    } catch (n) {
      if (!ka(n)) throw n;
      return await tx(), fetch(e, {
        ...t,
        cache: "no-store"
      });
    }
  }
  const nx = `You are MAUDE \u2014 a local AI assistant running on Matt's DGX Spark, handling tasks that benefit from local execution, privacy, or when cloud access isn't available.

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
  function rx(e) {
    return e.length <= 4e3 ? e : `${e.slice(0, 1800)}

... [older content trimmed for mobile reliability] ...

${e.slice(-1800)}`;
  }
  function bm(e, t, n, r) {
    const l = {
      model: e,
      messages: [
        {
          role: "system",
          content: nx
        },
        ...t,
        {
          role: "user",
          content: n
        }
      ],
      stream: true,
      max_tokens: 4096,
      temperature: 0.7
    };
    return r && (l.location = {
      lat: r.lat,
      lng: r.lng,
      accuracy: r.accuracy
    }), l;
  }
  function lx(e, t) {
    return bm(e, [], t, null);
  }
  function ax(e) {
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
  function Js(e, t, n, r) {
    if (e.type === "model_route") {
      t.route = {
        requestedModel: e.requested_model || "",
        resolvedModel: e.resolved_model || "",
        provider: e.provider || "unknown",
        endpoint: e.endpoint,
        maxContext: e.max_context,
        routeKind: e.route_kind,
        toolMode: e.tool_mode
      };
      const l = e.summary || e.resolved_model || e.requested_model || "model route", a = [
        e.provider,
        e.endpoint
      ].filter(Boolean).join(" via ");
      return n.push({
        name: "model_route",
        kind: "route",
        task: `Route: ${l}`,
        args: a || void 0,
        result: e.max_context ? `${Number(e.max_context).toLocaleString()} ctx` : void 0,
        status: "done"
      }), true;
    }
    if (e.type === "parallel_start") {
      const l = Array.isArray(e.tools) ? e.tools.join(", ") : "";
      return n.push({
        name: "parallel_start",
        kind: "parallel",
        task: `Running ${e.count || 0} tools in parallel`,
        args: l || void 0,
        status: "done"
      }), true;
    }
    if (e.type === "context_trim") return n.push({
      name: "context_trim",
      kind: "context",
      task: `Trimmed ${e.removed || 0} old messages`,
      result: e.max_tokens ? `${Number(e.max_tokens).toLocaleString()} token budget` : void 0,
      status: "done"
    }), true;
    if (e.type === "tool_call" && e.name) {
      t.tools.push(e.name);
      const l = e.args && e.args !== "{}" ? ax(e.args) : void 0;
      return n.push({
        name: e.name,
        kind: "tool",
        task: e.task,
        args: l,
        status: "running"
      }), true;
    }
    if (e.type === "tool_result") {
      for (let l = n.length - 1; l >= 0; l--) if (n[l].name === e.name && n[l].status === "running") {
        const a = (e.preview || "").slice(0, 60);
        n[l].result = a, n[l].elapsed = e.elapsed || 0, n[l].status = a.startsWith("Error") ? "error" : "done";
        break;
      }
      return true;
    }
    if (e.type === "keepalive" && e.name) {
      for (let l = n.length - 1; l >= 0; l--) if (n[l].name === e.name && n[l].status === "running") {
        n[l].elapsed = e.elapsed || 0;
        break;
      }
      return true;
    }
    return e.type === "llm_call" ? (t.promptTokens += e.prompt_tokens || 0, t.completionTokens += e.completion_tokens || 0, t.cacheReadTokens += e.cache_read_tokens || 0, t.cacheCreateTokens += e.cache_create_tokens || 0, t.elapsed += e.elapsed || 0, false) : e.type === "error" ? (r(e.message || "Unknown error"), true) : false;
  }
  function sx(e = null) {
    const [t, n] = g.useState(() => e ? Yo(e) : []), [r, l] = g.useState(false), [a, s] = g.useState(() => {
      const v = localStorage.getItem("maude-model"), E = oa(v);
      return v !== E && localStorage.setItem("maude-model", E), E;
    }), [u, i] = g.useState(() => localStorage.getItem("maude-autoroute") === "true"), c = g.useCallback((v) => {
      const E = oa(v);
      localStorage.setItem("maude-model", E), s(E);
    }, []), m = g.useCallback((v) => {
      localStorage.setItem("maude-autoroute", String(v)), i(v);
    }, []), d = g.useRef(a);
    d.current = a;
    const p = g.useRef(null), S = g.useRef(e), w = g.useRef(""), y = g.useRef(0);
    S.current = e, g.useEffect(() => {
      if (!e) {
        n([]);
        return;
      }
      n(Yo(e)), Yv(e).then((v) => {
        v.length > 0 && n(v);
      });
    }, [
      e
    ]), g.useEffect(() => {
      S.current && t.length > 0 && _m(S.current, t);
    }, [
      t
    ]);
    const b = g.useCallback(async (v, E) => {
      var _a2, _b, _c2;
      const _ = E && E.length > 0;
      if (!v.trim() && !_ || r) return;
      if (v.startsWith("/")) {
        const K = v.trim().toLowerCase();
        if (K === "/clear") {
          n([]);
          return;
        }
        if (K.startsWith("/model ")) {
          c(K.slice(7).trim());
          return;
        }
      }
      const R = v || (_ ? "What do you see in this image?" : ""), k = {
        id: Jo(),
        role: "user",
        content: R,
        imageUrls: _ ? E : void 0,
        timestamp: Date.now()
      };
      n((K) => [
        ...K,
        k
      ]), l(true);
      const j = d.current, I = {
        id: Jo(),
        role: "assistant",
        content: "",
        model: j,
        timestamp: Date.now()
      };
      n((K) => [
        ...K,
        I
      ]);
      const D = new AbortController();
      p.current = D;
      let Q = "";
      try {
        const K = t.filter((q) => q.role !== "system").slice(-8).map((q) => ({
          role: q.role,
          content: rx(q.content)
        }));
        let ae = R;
        if (_) {
          const q = E.map((me) => `/home/mboard76/nvidia-workbench/terminal-llm/shared/${me.split("/").pop()}`);
          if (q.length === 1) ae = `[Image attached: ${q[0]} \u2014 analyze it with view_image tool]

${R}`;
          else {
            const me = q.map((he, $) => `  ${$ + 1}. ${he}`).join(`
`);
            ae = `[${q.length} images attached \u2014 analyze each with view_image tool:
${me}]

${R}`;
          }
        }
        const de = await ex(), ge = bm(j, K, ae, de), lt = lx(j, ae);
        if (Zv) {
          let q;
          try {
            q = await ia(`${le()}/api/chat/create`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify(ge),
              signal: D.signal
            });
          } catch (he) {
            if (!ka(he)) throw he;
            q = await ia(`${le()}/api/chat/create`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify(lt),
              signal: D.signal
            });
          }
          if (!q.ok) {
            const he = await q.text();
            n(($) => $.map((B) => B.id === I.id ? {
              ...B,
              content: `Error: ${q.status} \u2014 ${he}`
            } : B)), l(false);
            return;
          }
          const { sid: me } = await q.json();
          await new Promise((he) => {
            let $ = null, B = 0, pe = false, se = "";
            const Se = {
              tools: [],
              promptTokens: 0,
              completionTokens: 0,
              cacheReadTokens: 0,
              cacheCreateTokens: 0,
              elapsed: 0
            }, ke = [], Ze = () => {
              if (pe) return;
              pe = true, $ == null ? void 0 : $.close(), y.current && (cancelAnimationFrame(y.current), y.current = 0);
              const ne = {
                content: se
              };
              Q && (ne.model = Q), (Se.promptTokens || Se.tools.length || Se.route) && (ne.trace = {
                ...Se
              }), ke.length && (ne.toolSteps = ke.map((xe) => ({
                ...xe
              }))), n((xe) => xe.map((Ve) => Ve.id === I.id ? {
                ...Ve,
                ...ne
              } : Ve)), w.current = "", l(false), p.current = null, he();
            };
            D.signal.addEventListener("abort", () => Ze());
            const ut = (ne) => {
              const xe = Number(ne.lastEventId);
              B = Number.isFinite(xe) ? xe + 1 : B + 1;
            }, Be = () => {
              y.current || (y.current = requestAnimationFrame(() => {
                const ne = w.current, xe = {
                  ...Se,
                  tools: [
                    ...Se.tools
                  ]
                }, Ve = ke.map((Et) => ({
                  ...Et
                }));
                n((Et) => Et.map((Ke) => Ke.id === I.id ? {
                  ...Ke,
                  content: ne,
                  trace: xe,
                  toolSteps: Ve,
                  ...Q && {
                    model: Q
                  }
                } : Ke)), y.current = 0;
              }));
            };
            let qe = false;
            const ct = (ne) => {
              var _a3, _b2, _c3, _d2;
              if (ut(ne), ne.data === "[DONE]") {
                Ze();
                return;
              }
              try {
                const xe = JSON.parse(ne.data);
                xe.model && !Q && (Q = oa(xe.model));
                const Ve = (_b2 = (_a3 = xe.choices) == null ? void 0 : _a3[0]) == null ? void 0 : _b2.delta;
                (Ve == null ? void 0 : Ve.reasoning_content) ? qe || (se += `*Thinking...*

`, qe = true) : (Ve == null ? void 0 : Ve.content) && (qe && (se = se.replace(`*Thinking...*

`, ""), qe = false), se += Ve.content), w.current = se, Be(), ((_d2 = (_c3 = xe.choices) == null ? void 0 : _c3[0]) == null ? void 0 : _d2.finish_reason) === "stop" && Ze();
              } catch {
              }
            }, Mn = (ne) => {
              ut(ne);
              try {
                const xe = JSON.parse(ne.data);
                Js(xe, Se, ke, (Et) => {
                  se += `

*Error: ${Et}*`, w.current = se;
                }) && (xe.type !== "error" && (w.current = se), Be());
              } catch {
              }
            }, dt = () => {
              pe || D.signal.aborted || ($ == null ? void 0 : $.close(), $ = new EventSource(`${le()}/api/chat/stream?sid=${me}&offset=${B}`), $.onmessage = ct, $.addEventListener("trace", Mn), $.onerror = () => {
                $ == null ? void 0 : $.close(), !pe && !D.signal.aborted && window.setTimeout(dt, document.visibilityState === "visible" ? 1e3 : 3e3);
              });
            };
            dt();
          });
          return;
        }
        let Qe;
        try {
          Qe = await ia(`${le()}/v1/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(ge),
            signal: D.signal
          });
        } catch (q) {
          if (!ka(q)) throw q;
          Qe = await ia(`${le()}/v1/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(lt),
            signal: D.signal
          });
        }
        if (!Qe.ok) {
          const q = await Qe.text();
          n((me) => me.map((he) => he.id === I.id ? {
            ...he,
            content: `Error: ${Qe.status} \u2014 ${q}`
          } : he)), l(false);
          return;
        }
        const M = (_a2 = Qe.body) == null ? void 0 : _a2.getReader();
        if (!M) {
          l(false);
          return;
        }
        const W = new TextDecoder();
        let H = "", Y = "", X = "", Fe = false;
        const ve = {
          tools: [],
          promptTokens: 0,
          completionTokens: 0,
          cacheReadTokens: 0,
          cacheCreateTokens: 0,
          elapsed: 0
        }, fe = [], Te = () => {
          y.current || (y.current = requestAnimationFrame(() => {
            const q = w.current, me = {
              ...ve,
              tools: [
                ...ve.tools
              ]
            }, he = fe.map(($) => ({
              ...$
            }));
            n(($) => $.map((B) => B.id === I.id ? {
              ...B,
              content: q,
              trace: me,
              toolSteps: he,
              ...Q && {
                model: Q
              }
            } : B)), y.current = 0;
          }));
        };
        for (; ; ) {
          const { done: q, value: me } = await M.read();
          if (q) break;
          H += W.decode(me, {
            stream: true
          });
          const he = H.split(`
`);
          H = he.pop() || "";
          for (const $ of he) {
            const B = $.trim();
            if (!B) continue;
            if (B.startsWith(": trace ")) {
              try {
                const se = JSON.parse(B.slice(8));
                Js(se, ve, fe, (ke) => {
                  Y += `

*Error: ${ke}*`, w.current = Y;
                }) && (se.type !== "error" && (w.current = Y), Te());
              } catch {
              }
              continue;
            }
            if (B.startsWith("event: ")) {
              X = B.slice(7);
              continue;
            }
            if (!B.startsWith("data: ")) continue;
            const pe = B.slice(6);
            if (pe !== "[DONE]") {
              if (X === "trace") {
                X = "";
                try {
                  const se = JSON.parse(pe);
                  Js(se, ve, fe, (ke) => {
                    Y += `

*Error: ${ke}*`, w.current = Y;
                  }) && (se.type !== "error" && (w.current = Y), Te());
                } catch {
                }
                continue;
              }
              X = "";
              try {
                const se = JSON.parse(pe);
                se.model && !Q && (Q = oa(se.model));
                const Se = (_c2 = (_b = se.choices) == null ? void 0 : _b[0]) == null ? void 0 : _c2.delta;
                (Se == null ? void 0 : Se.reasoning_content) ? Fe || (Y += `*Thinking...*

`, Fe = true) : (Se == null ? void 0 : Se.content) && (Fe && (Y = Y.replace(`*Thinking...*

`, ""), Fe = false), Y += Se.content), ((Se == null ? void 0 : Se.reasoning_content) || (Se == null ? void 0 : Se.content)) && (w.current = Y, Te());
              } catch {
              }
            }
          }
        }
        const Oe = {};
        Q && (Oe.model = Q), (ve.promptTokens || ve.tools.length || ve.route) && (Oe.trace = {
          ...ve
        }), fe.length && (Oe.toolSteps = fe.map((q) => ({
          ...q
        }))), Object.keys(Oe).length && n((q) => q.map((me) => me.id === I.id ? {
          ...me,
          ...Oe
        } : me));
      } catch (K) {
        if (K instanceof Error && K.name !== "AbortError") {
          const ae = ka(K) ? "Could not reach the MAUDE gateway after clearing stale app cache. Check Tailscale/VPN and reopen MAUDE." : K.message;
          n((de) => de.map((ge) => ge.id === I.id ? {
            ...ge,
            content: `Error: ${ae}`
          } : ge));
        }
      } finally {
        if (y.current && (cancelAnimationFrame(y.current), y.current = 0), w.current) {
          const K = w.current, ae = Q || void 0;
          n((de) => de.map((ge) => ge.id === I.id ? {
            ...ge,
            content: K,
            ...ae && {
              model: ae
            }
          } : ge)), w.current = "";
        }
        l(false), p.current = null;
      }
    }, [
      t,
      r,
      a,
      u,
      c
    ]), h = g.useCallback(() => {
      var _a2;
      (_a2 = p.current) == null ? void 0 : _a2.abort();
    }, []), f = g.useCallback(() => {
      n([]);
    }, []);
    return {
      messages: t,
      isStreaming: r,
      currentModel: a,
      setCurrentModel: c,
      autoRoute: u,
      setAutoRoute: m,
      sendMessage: b,
      stopStreaming: h,
      clearMessages: f
    };
  }
  function Uc(e) {
    const t = e.trim().replace(/\s+/g, " ");
    return t.length <= 40 ? t : t.slice(0, 37) + "...";
  }
  function ox() {
    const [e, t] = g.useState(Cm), [n, r] = g.useState(Xv);
    g.useEffect(() => {
      Kv().then((d) => {
        d.length > 0 && t(d);
      });
    }, []);
    const l = g.useCallback((d) => {
      const p = [
        ...d
      ].sort((S, w) => w.updatedAt - S.updatedAt);
      t(p), Gv(p);
    }, []), a = g.useCallback((d, p) => {
      const S = Jo(), w = Date.now(), b = [
        {
          id: S,
          title: Uc(d),
          createdAt: w,
          updatedAt: w,
          model: p
        },
        ...e
      ];
      return l(b), r(S), sa(S), S;
    }, [
      e,
      l
    ]), s = g.useCallback((d) => {
      r(d), sa(d);
    }, []), u = g.useCallback((d) => {
      const p = e.filter((S) => S.id !== d);
      if (l(p), Jv(d), n === d) {
        const S = p.length > 0 ? p[0].id : null;
        r(S), sa(S);
      }
    }, [
      e,
      n,
      l
    ]), i = g.useCallback((d, p) => {
      const S = e.map((w) => w.id === d ? {
        ...w,
        title: Uc(p)
      } : w);
      l(S);
    }, [
      e,
      l
    ]), c = g.useCallback((d) => {
      const p = e.map((S) => S.id === d ? {
        ...S,
        updatedAt: Date.now()
      } : S);
      l(p);
    }, [
      e,
      l
    ]), m = g.useCallback(() => {
      r(null), sa(null);
    }, []);
    return {
      conversations: e,
      activeId: n,
      createConversation: a,
      switchConversation: s,
      deleteConversation: u,
      updateTitle: i,
      touchConversation: c,
      startNewChat: m
    };
  }
  function ix(e, t) {
    const [n, r] = g.useState(0), l = g.useRef(0), a = g.useRef(false);
    return t && (a.current = true), g.useEffect(() => {
      if (!t && !a.current) {
        r(e.length);
        return;
      }
      const s = e.length;
      let u = 0;
      const i = (c) => {
        c - u >= 16 && (u = c, r((m) => m >= s ? m : m + Math.max(2, Math.floor((s - m) / 30)))), l.current = requestAnimationFrame(i);
      };
      return l.current = requestAnimationFrame(i), () => cancelAnimationFrame(l.current);
    }, [
      e,
      t
    ]), e.slice(0, n);
  }
  function ux(e) {
    const t = le();
    return e.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (r, l, a) => `<img src="${a.startsWith("/") ? `${t}${a}` : a}" alt="${l}" style="max-width:100%; max-height:50vh; border-radius:8px; margin:8px 0; object-fit:contain;" loading="lazy" onerror="this.style.display='none'" />`).replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-blue-400 underline">$1</a>').replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="my-2 rounded-lg bg-[#0d1117] p-3 text-sm overflow-x-auto"><code class="text-green-300">$2</code></pre>').replace(/`([^`]+)`/g, '<code class="rounded bg-[#0d1117] px-1.5 py-0.5 text-sm text-orange-300">$1</code>').replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>").replace(/^- (.+)$/gm, '<li class="ml-4 list-disc">$1</li>').replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>').replace(/\n/g, "<br/>");
  }
  const cx = {
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
    run_agent: "spawned an agent",
    run_agents: "spawned agents",
    execute_plan: "ran plan mode",
    change_directory: "changed directory",
    get_working_directory: "checked directory"
  };
  function dx(e) {
    const t = /* @__PURE__ */ new Map();
    for (const r of e.filter((l) => !l.kind || l.kind === "tool")) t.set(r.name, (t.get(r.name) || 0) + 1);
    const n = [];
    for (const [r, l] of t) {
      const a = cx[r] || r.replace(/_/g, " ");
      if (l > 1) {
        const s = a.replace(/(?:a |an )(\w+)$/, `${l} $1s`);
        n.push(s === a ? `${a} x${l}` : s);
      } else n.push(a);
    }
    return n.length <= 2 ? n.join(" and ") : n.slice(0, -1).join(", ") + ", and " + n[n.length - 1];
  }
  const fx = {
    model_route: "\u21C4",
    parallel_start: "\u2225",
    context_trim: "\u25F1",
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
  }, mx = ({ steps: e, streaming: t, contentStarted: n }) => {
    if (!e.length) return null;
    const r = e.some((a) => a.status === "running"), l = dx(e);
    return o.jsxs("div", {
      className: "mb-2 space-y-1",
      children: [
        e.map((a, s) => {
          const u = fx[a.name] || "\u2699\uFE0F", i = a.status === "running", c = a.status === "error", m = i ? "border-cyan-400/40" : c ? "border-red-400/40" : "border-cyan-500/20";
          return o.jsxs("div", {
            className: `border-l-2 ${m} pl-2.5 py-0.5 transition-all duration-300`,
            style: {
              animation: t ? "fadeSlideIn 0.3s ease-out" : "none"
            },
            children: [
              o.jsxs("div", {
                className: "flex items-center gap-1.5",
                children: [
                  i && t ? o.jsx("span", {
                    className: "inline-block h-3 w-3 animate-spin rounded-full border-2 border-cyan-300/30 border-t-cyan-300"
                  }) : o.jsx("span", {
                    className: "text-[11px]",
                    children: u
                  }),
                  o.jsx("span", {
                    className: "text-[11px] font-semibold text-cyan-300",
                    children: a.task || a.name
                  }),
                  i && o.jsx("span", {
                    className: "animate-pulse text-[10px] font-medium text-cyan-300",
                    children: "still working"
                  }),
                  a.elapsed !== void 0 && o.jsxs("span", {
                    className: "ml-auto font-mono text-[10px] text-maude-muted",
                    children: [
                      a.elapsed.toFixed(1),
                      "s"
                    ]
                  })
                ]
              }),
              a.task && (!a.kind || a.kind === "tool") && o.jsx("div", {
                className: "truncate font-mono text-[10px] leading-tight text-maude-muted",
                children: a.name
              }),
              a.args && o.jsx("div", {
                className: "truncate font-mono text-[10px] leading-tight text-maude-muted",
                children: a.args
              }),
              a.result && o.jsxs("div", {
                className: `truncate font-mono text-[10px] leading-tight ${c ? "text-red-400" : "text-green-400/80"}`,
                children: [
                  c ? "\u2717 " : "\u2713 ",
                  a.result
                ]
              })
            ]
          }, `${a.name}-${s}`);
        }),
        t && !r && !n && o.jsxs("div", {
          className: "flex items-center gap-1.5 border-l-2 border-cyan-400/20 py-1 pl-2.5",
          style: {
            animation: "fadeSlideIn 0.3s ease-out"
          },
          children: [
            o.jsx("span", {
              className: "inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50",
              style: {
                animationDelay: "0ms"
              }
            }),
            o.jsx("span", {
              className: "inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50",
              style: {
                animationDelay: "150ms"
              }
            }),
            o.jsx("span", {
              className: "inline-block h-1 w-1 animate-bounce rounded-full bg-cyan-400/50",
              style: {
                animationDelay: "300ms"
              }
            }),
            o.jsx("span", {
              className: "animate-pulse text-[10px] text-cyan-400/50",
              children: "thinking"
            })
          ]
        }),
        !t && l && o.jsx("div", {
          className: "mt-1 border-l-2 border-green-400/30 py-0.5 pl-2.5",
          children: o.jsxs("span", {
            className: "text-[10px] text-green-400/70",
            children: [
              "\u2713 ",
              l,
              (() => {
                const a = e.reduce((s, u) => s + (u.elapsed || 0), 0);
                return a > 0 ? ` \u2014 ${a.toFixed(1)}s` : "";
              })()
            ]
          })
        })
      ]
    });
  }, px = ({ trace: e }) => {
    const t = e.promptTokens + e.cacheReadTokens + e.cacheCreateTokens;
    if (!t && !e.tools.length && !e.route) return null;
    const n = t > 0 ? Math.round(e.cacheReadTokens / t * 100) : 0;
    return o.jsxs("div", {
      className: "mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-maude-muted",
      children: [
        e.route && o.jsx("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5 text-cyan-300",
          children: e.route.requestedModel && e.route.requestedModel !== e.route.resolvedModel ? `${e.route.requestedModel} -> ${e.route.resolvedModel}` : e.route.resolvedModel || e.route.requestedModel
        }),
        e.tools.length > 0 && o.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            e.tools.length,
            " tool",
            e.tools.length > 1 ? "s" : ""
          ]
        }),
        t + e.completionTokens > 0 && o.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            t + e.completionTokens,
            " tok"
          ]
        }),
        n > 0 && o.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5 text-green-400",
          children: [
            n,
            "% cached"
          ]
        }),
        e.elapsed > 0 && o.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            e.elapsed.toFixed(1),
            "s"
          ]
        })
      ]
    });
  }, hx = {
    "claude-opus-4-20250514": "Claude Opus",
    "claude-sonnet-4-20250514": "Claude Sonnet",
    "mistral-large-latest": "Mistral Large",
    "codestral-latest": "Codestral",
    "devstral-2512": "Devstral",
    "devstral-small-latest": "Devstral Small",
    "devstral-medium-latest": "Devstral Medium",
    nemotron: "Nemotron",
    "nemotron-super": "Nemotron Super",
    "nemotron-a3b": "Nemotron A3B",
    "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron Super",
    "nvidia/nemotron-3-nano-30b-a3b": "Nemotron A3B",
    "nemotron-nano": "Nemotron A3B",
    a3b: "Nemotron A3B",
    "gemma-4-31b": "Gemma 4",
    llava: "LLaVA"
  }, gx = ({ message: e, animate: t }) => {
    const n = e.role === "user", r = ix(e.content, !!t), l = !n && e.toolSteps && e.toolSteps.length > 0, a = !e.content && !n && !l;
    return o.jsx("div", {
      className: `flex ${n ? "justify-end" : "justify-start"} mb-3`,
      children: o.jsxs("div", {
        className: `max-w-[85%] rounded-2xl px-4 py-3 ${n ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`,
        children: [
          e.model && !n && o.jsx("div", {
            className: "mb-1 text-[10px] font-medium tracking-wider text-maude-muted",
            children: hx[e.model] || e.model
          }),
          (() => {
            const s = e.imageUrls || (e.imageUrl ? [
              e.imageUrl
            ] : []);
            if (!s.length) return null;
            const u = le();
            return o.jsx("div", {
              className: `mb-2 flex gap-2 ${s.length > 1 ? "overflow-x-auto" : ""}`,
              children: s.map((i, c) => o.jsx("img", {
                src: `${u}${i}`,
                alt: `Attached photo ${c + 1}`,
                className: `rounded-lg ${s.length > 1 ? "h-32 w-32 shrink-0 object-cover" : "max-w-full"}`,
                loading: "lazy"
              }, i))
            });
          })(),
          l && o.jsx(mx, {
            steps: e.toolSteps,
            streaming: !!t,
            contentStarted: !!e.content
          }),
          r && o.jsx("div", {
            className: "break-words text-sm leading-relaxed",
            dangerouslySetInnerHTML: {
              __html: ux(r)
            }
          }),
          !n && e.trace && o.jsx(px, {
            trace: e.trace
          }),
          a && o.jsxs("div", {
            className: "flex gap-1",
            children: [
              o.jsx("span", {
                className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                style: {
                  animationDelay: "0ms"
                }
              }),
              o.jsx("span", {
                className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                style: {
                  animationDelay: "150ms"
                }
              }),
              o.jsx("span", {
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
  }, vx = ({ onSend: e, isStreaming: t, onStop: n }) => {
    const [r, l] = g.useState(""), [a, s] = g.useState([]), [u, i] = g.useState(false), c = g.useRef(null), m = g.useRef(null), d = g.useRef(null);
    g.useEffect(() => {
      var _a2;
      (_a2 = c.current) == null ? void 0 : _a2.focus();
    }, []);
    const p = () => {
      (a.length > 0 || r.trim()) && (e(r.trim(), a.length > 0 ? a : void 0), l(""), s([]), c.current && (c.current.style.height = "44px"));
    }, S = (f) => {
      f.key === "Enter" && !f.shiftKey && (f.preventDefault(), p());
    }, w = () => {
      c.current && (c.current.style.height = "44px", c.current.style.height = Math.min(c.current.scrollHeight, 120) + "px");
    }, y = async (f) => {
      const v = f.target.files;
      if (!(!v || v.length === 0)) {
        i(true);
        try {
          const E = [];
          for (const _ of Array.from(v)) {
            const R = `camera_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.jpg`;
            (await fetch(`${le()}/share/${encodeURIComponent(R)}`, {
              method: "POST",
              body: _
            })).ok && E.push(`/download/${R}`);
          }
          E.length > 0 && s((_) => [
            ..._,
            ...E
          ]);
        } catch {
        } finally {
          i(false), m.current && (m.current.value = ""), d.current && (d.current.value = "");
        }
      }
    }, b = (f) => {
      s((v) => v.filter((E, _) => _ !== f));
    }, h = a.length > 0 || r.trim();
    return o.jsxs("div", {
      className: "border-t border-maude-border bg-maude-surface p-3",
      children: [
        a.length > 0 && o.jsx("div", {
          className: "mb-2 flex gap-2 overflow-x-auto",
          children: a.map((f, v) => o.jsxs("div", {
            className: "relative shrink-0",
            children: [
              o.jsx("img", {
                src: `${le()}${f}`,
                alt: `Pending upload ${v + 1}`,
                className: "h-20 w-20 rounded-lg object-cover"
              }),
              o.jsx("button", {
                onClick: () => b(v),
                className: "absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-red-600 text-xs text-white",
                children: "\xD7"
              })
            ]
          }, f))
        }),
        o.jsxs("div", {
          className: "flex items-end gap-2",
          children: [
            o.jsx("button", {
              onClick: () => {
                var _a2;
                return (_a2 = m.current) == null ? void 0 : _a2.click();
              },
              disabled: u,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-maude-bg text-lg text-maude-muted hover:text-maude-text disabled:opacity-30",
              children: u ? o.jsx("span", {
                className: "h-4 w-4 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
              }) : "\u{1F4F7}"
            }),
            o.jsx("input", {
              ref: m,
              type: "file",
              accept: "image/*",
              capture: "environment",
              onChange: y,
              className: "hidden"
            }),
            o.jsx("button", {
              onClick: () => {
                var _a2;
                return (_a2 = d.current) == null ? void 0 : _a2.click();
              },
              disabled: u,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-maude-bg text-lg text-maude-muted hover:text-maude-text disabled:opacity-30",
              children: "\u{1F4CE}"
            }),
            o.jsx("input", {
              ref: d,
              type: "file",
              accept: "image/*",
              multiple: true,
              onChange: y,
              className: "hidden"
            }),
            o.jsx("textarea", {
              ref: c,
              value: r,
              onChange: (f) => l(f.target.value),
              onKeyDown: S,
              onInput: w,
              placeholder: "Message MAUDE...",
              rows: 1,
              className: "min-h-[44px] max-h-[120px] flex-1 resize-none rounded-xl bg-maude-bg px-4 py-3 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
            }),
            t ? o.jsx("button", {
              onClick: n,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl bg-red-600 text-white",
              children: "\u25A0"
            }) : o.jsx("button", {
              onClick: p,
              disabled: !h,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30",
              children: "\u2191"
            })
          ]
        })
      ]
    });
  }, Xs = [
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
      id: "codex",
      label: "Codex",
      desc: "CLI"
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
      id: "nemotron-super",
      label: "Nemotron Super",
      desc: "OpenRouter 120B"
    },
    {
      id: "nemotron-a3b",
      label: "Nemotron A3B",
      desc: "OpenRouter 30B"
    },
    {
      id: "gemma-4-31b",
      label: "Gemma 4",
      desc: "Local 31B"
    },
    {
      id: "llava",
      label: "LLaVA",
      desc: "Vision"
    }
  ], xx = {
    "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
    "nvidia/nemotron-3-nano-30b-a3b": "nemotron-a3b",
    "nemotron-nano": "nemotron-a3b",
    a3b: "nemotron-a3b",
    "codex-cli": "codex"
  }, yx = ({ currentModel: e, onSelect: t, autoRoute: n, onToggleAutoRoute: r }) => {
    const [l, a] = g.useState(false), s = xx[e] || e, u = Xs.find((i) => i.id === s) || Xs[0];
    return o.jsxs("div", {
      className: "relative",
      children: [
        o.jsxs("button", {
          onClick: () => a(!l),
          className: "flex items-center gap-1.5 rounded-lg bg-maude-bg px-3 py-1.5 text-xs text-maude-muted transition-colors hover:text-maude-text",
          children: [
            o.jsx("span", {
              className: "h-1.5 w-1.5 rounded-full bg-green-400"
            }),
            u.label,
            n && o.jsx("span", {
              className: "text-[10px] text-maude-accent",
              children: "AUTO"
            })
          ]
        }),
        l && o.jsxs("div", {
          className: "absolute right-0 top-full z-50 mt-1 w-56 rounded-xl border border-maude-border bg-maude-surface p-2 shadow-xl",
          children: [
            Xs.map((i) => o.jsxs("button", {
              onClick: () => {
                t(i.id), a(false);
              },
              className: `flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${i.id === s ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
              children: [
                o.jsx("span", {
                  children: i.label
                }),
                o.jsx("span", {
                  className: "text-xs text-maude-muted",
                  children: i.desc
                })
              ]
            }, i.id)),
            o.jsx("div", {
              className: "mt-2 border-t border-maude-border pt-2",
              children: o.jsxs("button", {
                onClick: () => r(!n),
                className: "flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm text-maude-text hover:bg-maude-bg",
                children: [
                  o.jsx("span", {
                    children: "Auto-route code"
                  }),
                  o.jsx("span", {
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
  function wx(e) {
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
    for (const s of e) s.updatedAt >= n ? a[0].items.push(s) : s.updatedAt >= r ? a[1].items.push(s) : s.updatedAt >= l ? a[2].items.push(s) : a[3].items.push(s);
    return a.filter((s) => s.items.length > 0);
  }
  const Sx = ({ open: e, onClose: t, conversations: n, activeId: r, onSelect: l, onDelete: a, onNewChat: s }) => {
    const u = wx(n), [i, c] = g.useState(false);
    return o.jsxs(o.Fragment, {
      children: [
        o.jsx("div", {
          className: `fixed inset-0 z-40 bg-black/50 transition-opacity duration-200 ${e ? "opacity-100" : "pointer-events-none opacity-0"}`,
          onClick: t
        }),
        o.jsxs("div", {
          className: `fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-maude-border bg-maude-surface transition-transform duration-200 ${e ? "translate-x-0" : "-translate-x-full"}`,
          children: [
            o.jsxs("div", {
              className: "safe-top flex items-center justify-between border-b border-maude-border px-4 py-3",
              children: [
                o.jsx("h2", {
                  className: "text-sm font-semibold text-maude-text",
                  children: "Conversations"
                }),
                o.jsxs("div", {
                  className: "flex items-center gap-2",
                  children: [
                    o.jsx("button", {
                      onClick: () => c(!i),
                      className: `rounded-lg px-3 py-1 text-xs ${i ? "bg-maude-accent text-white" : "bg-maude-bg text-maude-muted"}`,
                      children: i ? "Done" : "Edit"
                    }),
                    o.jsx("button", {
                      onClick: () => {
                        c(false), s(), t();
                      },
                      className: "rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-accent",
                      children: "+ New"
                    })
                  ]
                })
              ]
            }),
            o.jsxs("div", {
              className: "no-scrollbar flex-1 overflow-y-auto p-2",
              children: [
                u.length === 0 && o.jsx("p", {
                  className: "px-2 py-8 text-center text-xs text-maude-muted",
                  children: "No conversations yet"
                }),
                u.map((m) => o.jsxs("div", {
                  className: "mb-3",
                  children: [
                    o.jsx("p", {
                      className: "mb-1 px-2 text-[10px] font-semibold uppercase tracking-wider text-maude-muted",
                      children: m.label
                    }),
                    m.items.map((d) => o.jsxs("div", {
                      className: `flex items-center rounded-lg px-2 py-2 text-sm transition-colors ${d.id === r ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                      children: [
                        i && o.jsx("button", {
                          onClick: (p) => {
                            p.stopPropagation(), a(d.id);
                          },
                          className: "mr-2 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red-500 text-xs text-white",
                          "aria-label": "Delete conversation",
                          children: "\u2212"
                        }),
                        o.jsx("button", {
                          className: "min-w-0 flex-1 truncate text-left",
                          onClick: () => {
                            i || (l(d.id), t());
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
  }, kx = ({ conversationId: e, onFirstMessage: t, onMessageSent: n, onOpenDrawer: r, onNewChat: l }) => {
    const a = vs(), s = g.useRef(null), u = g.useRef(e), { messages: i, isStreaming: c, currentModel: m, setCurrentModel: d, autoRoute: p, setAutoRoute: S, sendMessage: w, stopStreaming: y } = sx(e);
    g.useEffect(() => {
      s.current && (s.current.scrollTop = s.current.scrollHeight);
    }, [
      i
    ]), g.useEffect(() => {
      if (!c || !s.current) return;
      const h = setInterval(() => {
        s.current && (s.current.scrollTop = s.current.scrollHeight);
      }, 200);
      return () => clearInterval(h);
    }, [
      c
    ]);
    const b = g.useCallback((h, f) => {
      if (!u.current) {
        const v = h || ((f == null ? void 0 : f.length) ? "Image conversation" : "New chat"), E = t(v, m);
        u.current = E;
      }
      w(h, f), n();
    }, [
      w,
      t,
      n,
      m
    ]);
    return g.useEffect(() => {
      u.current && i.length > 0 && _m(u.current, i);
    }, [
      i
    ]), o.jsxs(o.Fragment, {
      children: [
        o.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2",
          children: [
            o.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                o.jsx("button", {
                  onClick: r,
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-sm text-maude-muted hover:text-maude-text",
                  "aria-label": "Open conversations",
                  children: "\u2630"
                }),
                o.jsx("h1", {
                  className: "fire-gradient text-lg font-bold",
                  children: "MAUDE"
                }),
                o.jsx("button", {
                  onClick: l,
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-muted hover:text-maude-text",
                  children: "New"
                }),
                o.jsxs("button", {
                  onClick: () => a("/maude/voice"),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-accent hover:text-maude-text",
                  children: [
                    "\u{1F399}\uFE0F",
                    " Voice"
                  ]
                })
              ]
            }),
            o.jsx(yx, {
              currentModel: m,
              onSelect: d,
              autoRoute: p,
              onToggleAutoRoute: S
            })
          ]
        }),
        o.jsxs("div", {
          ref: s,
          className: "no-scrollbar flex-1 overflow-y-auto px-4 py-4",
          children: [
            i.length === 0 && o.jsxs("div", {
              className: "flex h-full flex-col items-center justify-center text-center",
              children: [
                o.jsx("span", {
                  className: "fire-gradient mb-3 text-5xl font-black",
                  children: "\u25C7"
                }),
                o.jsx("h2", {
                  className: "mb-1 text-lg font-semibold text-maude-text",
                  children: "MAUDE"
                }),
                o.jsx("p", {
                  className: "max-w-xs text-sm text-maude-muted",
                  children: "Your local AI assistant. Ask me anything."
                }),
                o.jsx("div", {
                  className: "mt-4 flex flex-wrap justify-center gap-2",
                  children: [
                    "What can you do?",
                    "Write a Python script",
                    "Explain this code",
                    "System status"
                  ].map((h) => o.jsx("button", {
                    onClick: () => b(h),
                    className: "rounded-full border border-maude-border px-3 py-1.5 text-xs text-maude-muted transition-colors hover:border-maude-accent hover:text-maude-text",
                    children: h
                  }, h))
                })
              ]
            }),
            i.map((h, f) => o.jsx(gx, {
              message: h,
              animate: c && f === i.length - 1
            }, h.id))
          ]
        }),
        o.jsx(vx, {
          onSend: (h, f) => b(h, f),
          isStreaming: c,
          onStop: y
        })
      ]
    });
  }, Nx = () => {
    const [e, t] = g.useState(false), { conversations: n, activeId: r, createConversation: l, switchConversation: a, deleteConversation: s, touchConversation: u, startNewChat: i } = ox(), c = g.useCallback((d, p) => l(d, p), [
      l
    ]), m = g.useCallback(() => {
      r && u(r);
    }, [
      r,
      u
    ]);
    return o.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        o.jsx(kx, {
          conversationId: r,
          onFirstMessage: c,
          onMessageSent: m,
          onOpenDrawer: () => t(true),
          onNewChat: i
        }, r || "new"),
        o.jsx(Sx, {
          open: e,
          onClose: () => t(false),
          conversations: n,
          activeId: r,
          onSelect: a,
          onDelete: s,
          onNewChat: i
        })
      ]
    });
  }, jx = "modulepreload", Ex = function(e) {
    return "/" + e;
  }, $c = {}, il = function(t, n, r) {
    let l = Promise.resolve();
    if (n && n.length > 0) {
      document.getElementsByTagName("link");
      const s = document.querySelector("meta[property=csp-nonce]"), u = (s == null ? void 0 : s.nonce) || (s == null ? void 0 : s.getAttribute("nonce"));
      l = Promise.allSettled(n.map((i) => {
        if (i = Ex(i), i in $c) return;
        $c[i] = true;
        const c = i.endsWith(".css"), m = c ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${i}"]${m}`)) return;
        const d = document.createElement("link");
        if (d.rel = c ? "stylesheet" : jx, c || (d.as = "script"), d.crossOrigin = "", d.href = i, u && d.setAttribute("nonce", u), document.head.appendChild(d), c) return new Promise((p, S) => {
          d.addEventListener("load", p), d.addEventListener("error", () => S(new Error(`Unable to preload CSS for ${i}`)));
        });
      }));
    }
    function a(s) {
      const u = new Event("vite:preloadError", {
        cancelable: true
      });
      if (u.payload = s, window.dispatchEvent(u), !u.defaultPrevented) throw s;
    }
    return l.then((s) => {
      for (const u of s || []) u.status === "rejected" && a(u.reason);
      return t().catch(a);
    });
  }, Cx = {
    0: 0
  }, _x = {
    0: 0
  }, bx = {
    start: 0,
    endTurn: 1,
    pause: 2,
    restart: 3
  }, Rx = (e) => {
    switch (e.type) {
      case "handshake":
        return new Uint8Array([
          0,
          Cx[e.version],
          _x[e.model]
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
          bx[e.action]
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
  }, Tx = "You are MAUDE, a capable AI assistant with a warm Scottish accent. You are direct, competent, and quietly confident. Keep responses concise and natural for voice conversation. You run locally on Matt\u2019s DGX Spark workstation.", Px = "NATF2.pt";
  function Mx(e) {
    const t = Bv("");
    let n = Tx;
    e && (n += `

--- Image Context ---
` + e);
    const r = new URLSearchParams({
      text_prompt: n
    });
    return `${t}/api/chat?${r}`;
  }
  const Dx = `
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
  async function Lx(e, t) {
    const n = new Blob([
      Dx
    ], {
      type: "application/javascript"
    }), r = URL.createObjectURL(n);
    await e.audioWorklet.addModule(r), URL.revokeObjectURL(r);
    const l = new AudioWorkletNode(e, "ring-player", {
      outputChannelCount: [
        1
      ]
    });
    l.port.onmessage = (s) => {
      var _a2;
      ((_a2 = s.data) == null ? void 0 : _a2.type) === "state" && t && t(s.data.state, s.data);
    };
    const a = e.createGain();
    return a.gain.value = 6, l.connect(a), {
      feedAudio(s) {
        l.port.postMessage({
          type: "audio",
          pcm: s
        }, [
          s.buffer
        ]);
      },
      reset() {
        l.port.postMessage({
          type: "reset"
        });
      },
      connect(s) {
        a.connect(s);
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
  const Fc = ({ analyser: e, active: t, color: n }) => {
    const r = g.useRef(null), l = g.useRef(0);
    return g.useEffect(() => {
      if (!e || !t || !r.current) return;
      const a = r.current, s = a.getContext("2d"), u = e.frequencyBinCount, i = new Uint8Array(u), c = () => {
        l.current = requestAnimationFrame(c), e.getByteTimeDomainData(i), s.clearRect(0, 0, a.width, a.height), s.lineWidth = 2, s.strokeStyle = n, s.beginPath();
        const m = a.width / u;
        let d = 0;
        for (let p = 0; p < u; p++) {
          const w = i[p] / 128 * a.height / 2;
          p === 0 ? s.moveTo(d, w) : s.lineTo(d, w), d += m;
        }
        s.lineTo(a.width, a.height / 2), s.stroke();
      };
      return c(), () => cancelAnimationFrame(l.current);
    }, [
      e,
      t,
      n
    ]), o.jsx("canvas", {
      ref: r,
      width: 300,
      height: 60,
      className: "w-full rounded-lg"
    });
  }, Ox = () => {
    const e = vs(), [t, n] = g.useState("disconnected"), [r, l] = g.useState(""), [a, s] = g.useState(false), [u, i] = g.useState(0), [c, m] = g.useState(""), [d, p] = g.useState(""), [S, w] = g.useState(null), [y, b] = g.useState(null), [h, f] = g.useState(false), [v, E] = g.useState(false), _ = g.useRef(null), R = g.useRef(null), k = g.useRef(null), j = g.useRef(null), I = g.useRef(null), D = g.useRef(null), Q = g.useRef(null), K = g.useRef(null), ae = g.useRef(null), de = g.useRef(null), ge = g.useRef(0), lt = g.useRef(0), Qe = g.useRef(0), M = g.useRef(0), W = g.useRef(0), H = g.useRef(0), Y = g.useRef(0), X = g.useCallback(async () => {
      m(""), l(""), i(0), ge.current = 0;
      try {
        I.current || (I.current = new AudioContext({
          sampleRate: 48e3
        }));
        const $ = I.current;
        await $.resume();
        const B = $.createBuffer(1, 1, $.sampleRate), pe = $.createBufferSource();
        pe.buffer = B, pe.connect($.destination), pe.start(), Qe.current = 0, M.current = 0, p(`ctx: ${$.state} ${$.sampleRate}Hz`), D.current || (D.current = await Lx($, (qe, ct) => {
          ct.underruns != null && (W.current = ct.underruns), ct.avail != null && (H.current = ct.avail);
        }), D.current.connect($.destination)), D.current.reset(), W.current = 0;
        const se = $.createAnalyser();
        D.current.connect(se), K.current = se;
        const Se = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1
          }
        });
        de.current = Se;
        const ke = $.createAnalyser();
        $.createMediaStreamSource(Se).connect(ke), ae.current = ke;
        const ut = Mx(j.current ?? void 0);
        console.log("Connecting to voice server:", ut);
        const Be = new WebSocket(ut);
        Be.binaryType = "arraybuffer", _.current = Be, n("connecting"), Be.onopen = () => {
          console.log("voice server WS open, waiting for handshake");
        }, Be.onmessage = (qe) => {
          var _a2;
          try {
            const ct = new Uint8Array(qe.data), Mn = ct[0];
            if (Mn === 0) console.log("voice server handshake received"), n("connected"), Fe(Be, Se, $), lt.current = window.setInterval(() => {
              var _a3;
              ge.current += 1, i(ge.current);
              const dt = ((_a3 = I.current) == null ? void 0 : _a3.state) ?? "?", ne = Math.round(H.current / 48);
              p(`dec:${M.current} buf:${ne}ms ur:${W.current}`);
            }, 1e3);
            else if (Mn === 2) {
              const dt = new TextDecoder().decode(ct.slice(1));
              dt.includes("[Searching...]") ? s(true) : (dt.includes("[Tool result:]") || dt.includes("[Error:")) && s(false), l((ne) => ne + dt);
            } else if (Mn === 3) {
              M.current++;
              const dt = ct.slice(1), ne = new Float32Array(dt.buffer, dt.byteOffset, dt.byteLength / 4), xe = new Float32Array(ne.length * 2);
              for (let Ve = 0; Ve < xe.length; Ve++) {
                const Et = Ve * 0.5, Ke = Et | 0, Mt = Math.min(Ke + 1, ne.length - 1), Ol = Et - Ke;
                xe[Ve] = ne[Ke] + (ne[Mt] - ne[Ke]) * Ol;
              }
              (_a2 = D.current) == null ? void 0 : _a2.feedAudio(xe);
            }
          } catch (ct) {
            console.error("Message decode error:", ct);
          }
        }, Be.onclose = (qe) => {
          console.log("voice server WS closed:", qe.code, qe.reason), n("disconnected"), ve(), clearInterval(lt.current);
        }, Be.onerror = (qe) => {
          console.error("voice server WS error:", qe), m("WebSocket connection failed. Is voice server running?"), n("disconnected");
        };
      } catch ($) {
        const B = $ instanceof Error ? $.message : "Connection failed";
        console.error("Voice connect error:", B), m(B), n("disconnected");
      }
    }, []), Fe = g.useCallback(async ($, B, pe) => {
      try {
        const se = (await il(async () => {
          const { default: ut } = await import("./recorder.min-Cu_XpJPj.js").then((Be) => Be.r);
          return {
            default: ut
          };
        }, [])).default, Se = (await il(async () => {
          const { default: ut } = await import("./encoderWorker.min-De-nC0Q0.js");
          return {
            default: ut
          };
        }, [])).default, ke = pe.createMediaStreamSource(B), Ze = new se({
          encoderPath: Se,
          bufferLength: Math.round(960 * pe.sampleRate / 24e3),
          encoderFrameSize: 20,
          encoderSampleRate: 24e3,
          maxFramesPerPage: 2,
          numberOfChannels: 1,
          recordingGain: 1,
          resampleQuality: 3,
          encoderComplexity: 3,
          encoderApplication: 2049,
          streamPages: true,
          sourceNode: ke
        });
        Ze.ondataavailable = (ut) => {
          $.readyState === WebSocket.OPEN && $.send(Rx({
            type: "audio",
            data: ut
          }));
        }, Ze.onstart = () => {
          console.log("Opus recorder started");
        }, Ze.start(), Q.current = Ze;
      } catch (se) {
        console.error("Recorder start error:", se), m("Failed to start microphone recording");
      }
    }, []), ve = g.useCallback(() => {
      if (Q.current) {
        try {
          Q.current.stop();
        } catch {
        }
        Q.current = null;
      }
      de.current && (de.current.getTracks().forEach(($) => $.stop()), de.current = null);
    }, []), fe = g.useCallback(() => {
      ve(), clearInterval(lt.current), clearInterval(Y.current), _.current && (_.current.close(), _.current = null), n("disconnected");
    }, [
      ve
    ]), Te = g.useCallback(async ($) => {
      var _a2;
      const B = (_a2 = $.target.files) == null ? void 0 : _a2[0];
      if (!B) return;
      $.target.value = "";
      const pe = `voice_camera_${Date.now()}.jpg`, se = le(), Se = URL.createObjectURL(B);
      w(Se), b(null), E(true);
      try {
        if (!(await fetch(`${se}/share/${pe}`, {
          method: "POST",
          body: B
        })).ok) throw new Error("Upload failed");
        E(false), f(true);
        const Ze = await fetch(`${se}/api/analyze-image`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            filename: pe,
            question: "Describe this image in detail. What do you see?"
          })
        });
        if (!Ze.ok) throw new Error("Analysis failed");
        const Be = (await Ze.json()).analysis || "No analysis returned.";
        b(Be), f(false), j.current = `The user shared an image (${pe}). Analysis: ${Be}`, _.current && _.current.readyState === WebSocket.OPEN && (fe(), await new Promise((qe) => setTimeout(qe, 300)), X());
      } catch (ke) {
        const Ze = ke instanceof Error ? ke.message : "Image processing failed";
        m(Ze), E(false), f(false);
      }
    }, [
      X,
      fe
    ]), Oe = g.useCallback(async () => {
      j.current = null, w(null), b(null), _.current && _.current.readyState === WebSocket.OPEN && (fe(), await new Promise(($) => setTimeout($, 300)), X());
    }, [
      X,
      fe
    ]);
    g.useEffect(() => () => {
      fe();
    }, []);
    const q = ($) => {
      const B = Math.floor($ / 60), pe = $ % 60;
      return `${B}:${pe.toString().padStart(2, "0")}`;
    }, me = t === "connected", he = t === "connecting";
    return o.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        o.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2",
          children: [
            o.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                o.jsx("h1", {
                  className: "fire-gradient text-lg font-bold",
                  children: "MAUDE"
                }),
                o.jsx("span", {
                  className: "rounded-full bg-maude-bg px-2 py-0.5 text-[10px] uppercase tracking-wider text-maude-accent",
                  children: "Voice"
                })
              ]
            }),
            o.jsx("button", {
              onClick: () => e("/maude"),
              className: "rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
              children: "Text Mode"
            })
          ]
        }),
        o.jsxs("div", {
          className: "flex flex-1 flex-col items-center justify-center gap-6 overflow-y-auto px-6 pb-4",
          children: [
            o.jsxs("div", {
              className: "flex flex-col items-center gap-2",
              children: [
                o.jsx("div", {
                  className: `h-32 w-32 rounded-full border-4 ${me ? "animate-pulse border-maude-accent shadow-[0_0_30px_rgba(255,69,0,0.3)]" : he ? "animate-spin border-maude-muted" : "border-maude-border"} flex items-center justify-center`,
                  children: o.jsx("span", {
                    className: "text-4xl",
                    children: me ? "\u{1F399}\uFE0F" : he ? "\u23F3" : "\u{1F399}\uFE0F"
                  })
                }),
                o.jsx("span", {
                  className: "text-sm text-maude-muted",
                  children: me ? `Connected \u2022 ${q(u)}` : he ? "Connecting to MAUDE Voice..." : "Tap to start voice chat"
                })
              ]
            }),
            me && o.jsxs("div", {
              className: "w-full max-w-xs space-y-3",
              children: [
                o.jsxs("div", {
                  children: [
                    o.jsx("span", {
                      className: "mb-1 block text-[10px] uppercase tracking-wider text-maude-muted",
                      children: "MAUDE"
                    }),
                    o.jsx("div", {
                      className: "rounded-lg bg-maude-surface p-2",
                      children: o.jsx(Fc, {
                        analyser: K.current,
                        active: me,
                        color: "#ff4500"
                      })
                    })
                  ]
                }),
                o.jsxs("div", {
                  children: [
                    o.jsx("span", {
                      className: "mb-1 block text-[10px] uppercase tracking-wider text-maude-muted",
                      children: "You"
                    }),
                    o.jsx("div", {
                      className: "rounded-lg bg-maude-surface p-2",
                      children: o.jsx(Fc, {
                        analyser: ae.current,
                        active: me,
                        color: "#888"
                      })
                    })
                  ]
                })
              ]
            }),
            me && o.jsxs("div", {
              className: "flex gap-3",
              children: [
                o.jsxs("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = R.current) == null ? void 0 : _a2.click();
                  },
                  disabled: h || v,
                  className: "flex items-center gap-1.5 rounded-xl bg-maude-surface px-4 py-2 text-sm text-maude-text transition-all hover:bg-maude-border disabled:opacity-40",
                  children: [
                    o.jsx("span", {
                      children: "\u{1F4F7}"
                    }),
                    " Camera"
                  ]
                }),
                o.jsxs("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = k.current) == null ? void 0 : _a2.click();
                  },
                  disabled: h || v,
                  className: "flex items-center gap-1.5 rounded-xl bg-maude-surface px-4 py-2 text-sm text-maude-text transition-all hover:bg-maude-border disabled:opacity-40",
                  children: [
                    o.jsx("span", {
                      children: "\u{1F5BC}\uFE0F"
                    }),
                    " Gallery"
                  ]
                })
              ]
            }),
            o.jsx("input", {
              ref: R,
              type: "file",
              accept: "image/*",
              capture: "environment",
              onChange: Te,
              className: "hidden"
            }),
            o.jsx("input", {
              ref: k,
              type: "file",
              accept: "image/*",
              onChange: Te,
              className: "hidden"
            }),
            S && o.jsxs("div", {
              className: "w-full max-w-xs rounded-xl bg-maude-surface p-3",
              children: [
                o.jsx("span", {
                  className: "mb-2 block text-[10px] uppercase tracking-wider text-maude-muted",
                  children: "Image Context"
                }),
                o.jsx("img", {
                  src: S,
                  alt: "Captured",
                  className: "mb-2 h-24 w-full rounded-lg object-cover"
                }),
                v && o.jsx("p", {
                  className: "text-xs text-maude-muted",
                  children: "Uploading..."
                }),
                h && o.jsxs("div", {
                  className: "flex items-center gap-2",
                  children: [
                    o.jsx("div", {
                      className: "h-3 w-3 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
                    }),
                    o.jsx("span", {
                      className: "text-xs text-maude-muted",
                      children: "Analyzing with LLaVA..."
                    })
                  ]
                }),
                y && o.jsx("p", {
                  className: "text-xs leading-relaxed text-maude-text",
                  children: y
                }),
                y && o.jsx("button", {
                  onClick: Oe,
                  className: "mt-2 text-[10px] text-maude-muted underline hover:text-maude-text",
                  children: "Clear image context"
                })
              ]
            }),
            a && o.jsxs("div", {
              className: "flex items-center gap-2 rounded-xl bg-maude-accent/10 px-4 py-2",
              children: [
                o.jsx("div", {
                  className: "h-3 w-3 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
                }),
                o.jsx("span", {
                  className: "text-xs font-medium text-maude-accent",
                  children: "Searching..."
                })
              ]
            }),
            r && o.jsxs("div", {
              className: "w-full max-w-xs rounded-xl bg-maude-surface p-3",
              children: [
                o.jsx("span", {
                  className: "mb-1 block text-[10px] uppercase tracking-wider text-maude-muted",
                  children: "Transcript"
                }),
                o.jsx("div", {
                  className: "max-h-48 overflow-y-auto text-sm text-maude-text",
                  children: r.split(`
`).map(($, B) => $.includes("[Searching...]") ? o.jsx("p", {
                    className: "my-1 text-xs italic text-maude-accent",
                    children: $
                  }, B) : $.includes("[Tool result:]") ? o.jsx("p", {
                    className: "mt-2 mb-1 text-[10px] font-bold uppercase tracking-wider text-maude-accent",
                    children: $
                  }, B) : $.includes("[Error:") ? o.jsx("p", {
                    className: "my-1 text-xs text-red-400",
                    children: $
                  }, B) : o.jsxs("span", {
                    children: [
                      $,
                      B < r.split(`
`).length - 1 ? `
` : ""
                    ]
                  }, B))
                })
              ]
            }),
            c && o.jsx("div", {
              className: "w-full max-w-xs rounded-xl bg-red-900/30 p-3",
              children: o.jsx("p", {
                className: "text-sm text-red-400",
                children: c
              })
            }),
            o.jsx("button", {
              onClick: me || he ? fe : X,
              className: `min-w-[200px] rounded-2xl px-8 py-4 text-base font-semibold text-white transition-all ${me ? "bg-red-600 hover:bg-red-700" : he ? "bg-maude-muted" : "fire-bg hover:opacity-90"}`,
              disabled: he,
              children: me ? "End Call" : he ? "Connecting..." : "Start Voice Chat"
            }),
            o.jsxs("div", {
              className: "text-center text-[10px] text-maude-muted",
              children: [
                "Voice: ",
                (localStorage.getItem("maude-default-voice") || Px).replace(".pt", ""),
                " \u2022 ",
                "MAUDE Voice"
              ]
            }),
            d && o.jsx("div", {
              className: "text-center font-mono text-[10px] text-maude-muted opacity-60",
              children: d
            })
          ]
        })
      ]
    });
  }, Ax = () => {
    const e = g.useRef(null), t = g.useRef(null), n = g.useRef(null), r = g.useRef(null), l = g.useRef(null), a = g.useRef(null), s = g.useRef(null), [u, i] = g.useState("disconnected");
    return g.useEffect(() => {
      let c, m;
      return (async () => {
        const { Terminal: p } = await il(async () => {
          const { Terminal: b } = await import("./xterm-PglAAeey.js").then((h) => h.x);
          return {
            Terminal: b
          };
        }, []), { FitAddon: S } = await il(async () => {
          const { FitAddon: b } = await import("./addon-fit-CyyJcX4C.js").then((h) => h.a);
          return {
            FitAddon: b
          };
        }, []), { WebLinksAddon: w } = await il(async () => {
          const { WebLinksAddon: b } = await import("./addon-web-links-B1M8nFkN.js").then((h) => h.a);
          return {
            WebLinksAddon: b
          };
        }, []);
        if (!document.querySelector('link[href*="xterm"]')) {
          const b = document.createElement("link");
          b.rel = "stylesheet", b.href = "https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/css/xterm.min.css", document.head.appendChild(b);
        }
        c = new p({
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
        const y = new S();
        c.loadAddon(y), c.loadAddon(new w()), r.current = c, l.current = y, e.current && (c.open(e.current), y.fit()), i("connecting");
        try {
          const b = await fetch(`${le()}/api/terminal/create`, {
            method: "POST"
          }), { sid: h } = await b.json();
          s.current = h;
          const f = new EventSource(`${le()}/api/terminal/stream?sid=${h}`);
          a.current = f, f.onopen = () => {
            i("connected");
            const R = y.proposeDimensions();
            R && fetch(`${le()}/api/terminal/resize`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: h,
                cols: R.cols,
                rows: R.rows
              })
            });
          }, f.onmessage = (R) => {
            const k = Uint8Array.from(atob(R.data), (j) => j.charCodeAt(0));
            c.write(k);
          }, f.onerror = () => {
            i("disconnected"), c.write(`\r
\x1B[33m[Connection closed]\x1B[0m\r
`), f.close();
          };
          const v = (R) => {
            fetch(`${le()}/api/terminal/input`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: h,
                data: R
              })
            });
          };
          n.current = v, c.onData(v);
          const E = () => {
            y.fit();
            const R = y.proposeDimensions();
            R && fetch(`${le()}/api/terminal/resize`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: h,
                cols: R.cols,
                rows: R.rows
              })
            });
          }, _ = new ResizeObserver(E);
          e.current && _.observe(e.current), m = () => _.disconnect();
        } catch {
          i("disconnected"), c.write(`\x1B[31m[Failed to connect]\x1B[0m\r
`);
        }
      })(), () => {
        var _a2, _b, _c2;
        m == null ? void 0 : m(), (_a2 = t.current) == null ? void 0 : _a2.close(), (_b = a.current) == null ? void 0 : _b.close(), (_c2 = r.current) == null ? void 0 : _c2.dispose();
      };
    }, []), o.jsxs("div", {
      className: "flex h-full flex-col bg-[#0d1117]",
      children: [
        o.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2",
          children: [
            o.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                o.jsx("span", {
                  className: "font-mono text-sm text-maude-text",
                  children: ">_ Terminal"
                }),
                o.jsx("span", {
                  className: `h-2 w-2 rounded-full ${u === "connected" ? "bg-green-400" : u === "connecting" ? "bg-yellow-400" : "bg-red-400"}`
                }),
                o.jsx("span", {
                  className: "text-xs text-maude-muted",
                  children: u
                })
              ]
            }),
            u === "disconnected" && o.jsx("button", {
              onClick: () => window.location.reload(),
              className: "rounded-lg bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
              children: "Reconnect"
            })
          ]
        }),
        o.jsx("div", {
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
          ].map((c) => o.jsx("button", {
            onClick: () => {
              var _a2, _b;
              (_a2 = n.current) == null ? void 0 : _a2.call(n, c.key), (_b = r.current) == null ? void 0 : _b.focus();
            },
            className: "shrink-0 rounded bg-maude-bg px-2 py-1 text-[11px] font-mono text-maude-muted active:bg-maude-accent active:text-white",
            children: c.label
          }, c.label))
        }),
        o.jsx("div", {
          ref: e,
          className: "flex-1 overflow-hidden px-1 py-1",
          onTouchStart: () => {
            var _a2;
            return (_a2 = r.current) == null ? void 0 : _a2.focus();
          }
        })
      ]
    });
  }, Ix = [
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
  ], zx = () => {
    const [e, t] = g.useState(""), [n, r] = g.useState(""), [l, a] = g.useState(""), [s, u] = g.useState(false), [i, c] = g.useState(""), m = g.useRef(null), [d, p] = g.useState("proxy"), [S, w] = g.useState([]), [y, b] = g.useState(-1), h = g.useCallback(async (E) => {
      if (!E.trim()) return;
      let _ = E.trim();
      if (!_.startsWith("http://") && !_.startsWith("https://") && (_ = "https://" + _), r(_), c(""), d === "iframe") {
        t(_), w((R) => [
          ...R.slice(0, y + 1),
          _
        ]), b((R) => R + 1);
        return;
      }
      u(true);
      try {
        const R = await fetch(`${le()}/proxy?url=${encodeURIComponent(_)}`);
        if (!R.ok) {
          c(`Failed: ${R.status}`), u(false);
          return;
        }
        if ((R.headers.get("content-type") || "").includes("application/json")) {
          const j = await R.json();
          if (j.redirect) {
            u(false), h(j.redirect);
            return;
          }
          c(j.error || "Unknown error");
        } else a(await R.text());
        w((j) => [
          ...j.slice(0, y + 1),
          _
        ]), b((j) => j + 1);
      } catch (R) {
        c(R instanceof Error ? R.message : "Failed");
      }
      u(false);
    }, [
      d,
      y
    ]), f = () => {
      y > 0 && (b(y - 1), h(S[y - 1]));
    }, v = () => {
      y < S.length - 1 && (b(y + 1), h(S[y + 1]));
    };
    return o.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        o.jsxs("div", {
          className: "flex shrink-0 flex-col border-b border-maude-border bg-maude-surface",
          children: [
            o.jsxs("form", {
              onSubmit: (E) => {
                E.preventDefault(), h(n);
              },
              className: "flex items-center gap-2 px-3 py-2",
              children: [
                o.jsxs("div", {
                  className: "flex gap-1",
                  children: [
                    o.jsx("button", {
                      type: "button",
                      onClick: f,
                      disabled: y <= 0,
                      className: "rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30",
                      children: "\u25C0"
                    }),
                    o.jsx("button", {
                      type: "button",
                      onClick: v,
                      disabled: y >= S.length - 1,
                      className: "rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30",
                      children: "\u25B6"
                    }),
                    o.jsx("button", {
                      type: "button",
                      onClick: () => h(n),
                      className: "rounded px-2 py-1 text-sm text-maude-muted",
                      children: "\u21BB"
                    })
                  ]
                }),
                o.jsx("input", {
                  type: "text",
                  value: n,
                  onChange: (E) => r(E.target.value),
                  placeholder: "Enter URL...",
                  className: "flex-1 rounded-lg bg-maude-bg px-3 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
                }),
                o.jsx("button", {
                  type: "button",
                  onClick: () => p(d === "proxy" ? "iframe" : "proxy"),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-[10px] text-maude-muted",
                  children: d === "proxy" ? "PROXY" : "IFRAME"
                })
              ]
            }),
            o.jsx("div", {
              className: "flex gap-1 overflow-x-auto px-3 pb-2 no-scrollbar",
              children: Ix.map((E) => o.jsx("button", {
                onClick: () => {
                  r(E.url), h(E.url);
                },
                className: "shrink-0 rounded-full bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
                children: E.label
              }, E.url))
            })
          ]
        }),
        o.jsxs("div", {
          className: "flex-1 overflow-hidden",
          children: [
            s && o.jsx("div", {
              className: "flex h-full items-center justify-center",
              children: o.jsx("div", {
                className: "h-6 w-6 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
              })
            }),
            i && o.jsx("div", {
              className: "flex h-full items-center justify-center p-8 text-center",
              children: o.jsx("p", {
                className: "text-red-400",
                children: i
              })
            }),
            !s && !i && d === "proxy" && l && o.jsx("iframe", {
              srcDoc: l,
              className: "h-full w-full border-0 bg-white",
              sandbox: "allow-scripts allow-same-origin allow-forms",
              title: "Browser"
            }),
            !s && !i && d === "iframe" && e && o.jsx("iframe", {
              ref: m,
              src: e,
              className: "h-full w-full border-0 bg-white",
              sandbox: "allow-scripts allow-same-origin allow-forms allow-popups",
              title: "Browser"
            }),
            !s && !i && !l && !e && o.jsxs("div", {
              className: "flex h-full flex-col items-center justify-center gap-4 text-center",
              children: [
                o.jsx("span", {
                  className: "text-4xl",
                  children: "\u25CE"
                }),
                o.jsx("p", {
                  className: "text-sm text-maude-muted",
                  children: "Enter a URL to browse the web."
                })
              ]
            })
          ]
        })
      ]
    });
  }, Ux = () => {
    const [e, t] = g.useState([]), [n, r] = g.useState(""), [l, a] = g.useState(false), s = g.useRef(null);
    g.useEffect(() => {
      s.current && (s.current.scrollTop = s.current.scrollHeight);
    }, [
      e
    ]), g.useEffect(() => {
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
    const u = async () => {
      var _a2, _b, _c2;
      if (!n.trim()) return;
      const i = n.trim();
      r(""), t((c) => [
        ...c,
        {
          id: Date.now(),
          from: "You",
          text: i,
          date: Date.now() / 1e3,
          outgoing: true
        }
      ]), a(true);
      try {
        const c = await fetch(`${le()}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: "nemotron-super",
            messages: [
              {
                role: "system",
                content: "You are MAUDE. Respond briefly and helpfully, like a text message."
              },
              {
                role: "user",
                content: i
              }
            ],
            max_tokens: 500,
            stream: false
          })
        });
        if (c.ok) {
          const d = ((_c2 = (_b = (_a2 = (await c.json()).choices) == null ? void 0 : _a2[0]) == null ? void 0 : _b.message) == null ? void 0 : _c2.content) || "...";
          t((p) => [
            ...p,
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
    return o.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        o.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-3",
          children: [
            o.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                o.jsx("span", {
                  className: "text-lg",
                  children: "\u2709"
                }),
                o.jsx("h1", {
                  className: "text-sm font-semibold text-maude-text",
                  children: "Messages"
                })
              ]
            }),
            o.jsx("span", {
              className: "rounded-full bg-maude-bg px-2 py-0.5 text-[10px] text-maude-muted",
              children: "Telegram"
            })
          ]
        }),
        o.jsxs("div", {
          ref: s,
          className: "no-scrollbar flex-1 overflow-y-auto px-4 py-4",
          children: [
            e.map((i) => o.jsx("div", {
              className: `mb-3 flex ${i.outgoing ? "justify-end" : "justify-start"}`,
              children: o.jsxs("div", {
                className: `max-w-[80%] rounded-2xl px-4 py-2.5 ${i.outgoing ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`,
                children: [
                  !i.outgoing && o.jsx("div", {
                    className: "mb-0.5 text-[10px] font-medium text-maude-accent",
                    children: i.from
                  }),
                  o.jsx("p", {
                    className: "text-sm",
                    children: i.text
                  }),
                  o.jsx("div", {
                    className: "mt-1 text-[10px] opacity-50",
                    children: new Date(i.date * 1e3).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit"
                    })
                  })
                ]
              })
            }, i.id)),
            l && o.jsx("div", {
              className: "flex justify-start",
              children: o.jsx("div", {
                className: "rounded-2xl bg-maude-surface px-4 py-3",
                children: o.jsxs("div", {
                  className: "flex gap-1",
                  children: [
                    o.jsx("span", {
                      className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                      style: {
                        animationDelay: "0ms"
                      }
                    }),
                    o.jsx("span", {
                      className: "h-2 w-2 animate-bounce rounded-full bg-maude-muted",
                      style: {
                        animationDelay: "150ms"
                      }
                    }),
                    o.jsx("span", {
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
        o.jsxs("div", {
          className: "flex items-center gap-2 border-t border-maude-border bg-maude-surface p-3",
          children: [
            o.jsx("input", {
              type: "text",
              value: n,
              onChange: (i) => r(i.target.value),
              onKeyDown: (i) => {
                i.key === "Enter" && u();
              },
              placeholder: "Message...",
              className: "min-h-[44px] flex-1 rounded-xl bg-maude-bg px-4 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
            }),
            o.jsx("button", {
              onClick: u,
              disabled: !n.trim() || l,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30",
              children: "\u2191"
            })
          ]
        })
      ]
    });
  };
  function $x(e) {
    return e < 1024 ? e + " B" : e < 1048576 ? (e / 1024).toFixed(1) + " KB" : (e / 1048576).toFixed(1) + " MB";
  }
  function Fx(e) {
    return new Date(e * 1e3).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }
  const Bx = () => {
    const [e, t] = g.useState([]), [n, r] = g.useState(""), [l, a] = g.useState(false), [s, u] = g.useState(""), [i, c] = g.useState("shared"), m = g.useRef(null), d = g.useCallback(async (w) => {
      a(true), u("");
      try {
        const y = i === "transfers" ? `${le()}/transfers` : w ? `${le()}/list?path=${encodeURIComponent(w)}` : `${le()}/list`, h = await (await fetch(y)).json();
        h.error ? u(h.error) : (t(h.files || []), r(h.path || ""));
      } catch (y) {
        u(y instanceof Error ? y.message : "Failed");
      }
      a(false);
    }, [
      i
    ]);
    g.useEffect(() => {
      d();
    }, [
      d
    ]);
    const p = (w) => {
      window.open(`${le()}/${i === "transfers" ? "download-transfer" : "download"}/${encodeURIComponent(w)}`);
    }, S = async (w) => {
      var _a2;
      const y = (_a2 = w.target.files) == null ? void 0 : _a2[0];
      if (y) {
        a(true);
        try {
          const h = await (await fetch(`${le()}/upload/${encodeURIComponent(y.name)}`, {
            method: "POST",
            body: y
          })).json();
          h.error ? u(h.error) : d();
        } catch (b) {
          u(b instanceof Error ? b.message : "Upload failed");
        }
        a(false), m.current && (m.current.value = "");
      }
    };
    return o.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        o.jsxs("div", {
          className: "flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-3",
          children: [
            o.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                o.jsx("span", {
                  className: "text-lg",
                  children: "\u25A4"
                }),
                o.jsx("h1", {
                  className: "text-sm font-semibold text-maude-text",
                  children: "Files"
                })
              ]
            }),
            o.jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                o.jsx("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = m.current) == null ? void 0 : _a2.click();
                  },
                  className: "rounded-lg fire-bg px-3 py-1 text-xs font-medium text-white",
                  children: "Upload"
                }),
                o.jsx("button", {
                  onClick: () => d(),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-muted",
                  children: "\u21BB"
                }),
                o.jsx("input", {
                  ref: m,
                  type: "file",
                  onChange: S,
                  className: "hidden"
                })
              ]
            })
          ]
        }),
        o.jsx("div", {
          className: "flex shrink-0 border-b border-maude-border bg-maude-surface",
          children: [
            "shared",
            "transfers"
          ].map((w) => o.jsx("button", {
            onClick: () => c(w),
            className: `flex-1 py-2 text-xs font-medium capitalize ${i === w ? "border-b-2 border-maude-accent text-maude-accent" : "text-maude-muted"}`,
            children: w
          }, w))
        }),
        n && o.jsxs("div", {
          className: "flex items-center gap-2 border-b border-maude-border bg-maude-surface/50 px-4 py-2",
          children: [
            o.jsxs("button", {
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
            o.jsx("span", {
              className: "truncate text-xs text-maude-muted",
              children: n
            })
          ]
        }),
        s && o.jsx("div", {
          className: "bg-red-900/30 px-4 py-2 text-xs text-red-400",
          children: s
        }),
        o.jsxs("div", {
          className: "no-scrollbar flex-1 overflow-y-auto",
          children: [
            l && o.jsx("div", {
              className: "flex h-32 items-center justify-center",
              children: o.jsx("div", {
                className: "h-6 w-6 animate-spin rounded-full border-2 border-maude-accent border-t-transparent"
              })
            }),
            !l && e.length === 0 && o.jsx("div", {
              className: "flex h-32 items-center justify-center",
              children: o.jsx("p", {
                className: "text-sm text-maude-muted",
                children: "No files found."
              })
            }),
            !l && e.map((w) => o.jsxs("button", {
              onClick: () => w.is_dir ? d(n ? `${n}/${w.name}` : w.name) : p(w.name),
              className: "flex w-full items-center gap-3 border-b border-maude-border/50 px-4 py-3 text-left hover:bg-maude-surface",
              children: [
                o.jsx("span", {
                  className: "text-lg",
                  children: w.is_dir ? "\u{1F4C1}" : "\u{1F4C4}"
                }),
                o.jsxs("div", {
                  className: "min-w-0 flex-1",
                  children: [
                    o.jsx("div", {
                      className: "truncate text-sm text-maude-text",
                      children: w.name
                    }),
                    o.jsxs("div", {
                      className: "mt-0.5 text-[10px] text-maude-muted",
                      children: [
                        w.is_dir ? "Directory" : $x(w.size),
                        " \xB7 ",
                        Fx(w.modified)
                      ]
                    })
                  ]
                }),
                !w.is_dir && o.jsx("span", {
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
  async function Vx() {
    const e = [];
    if ("serviceWorker" in navigator) {
      const t = await navigator.serviceWorker.getRegistrations();
      await Promise.all(t.map((n) => n.unregister())), e.push(`Unregistered ${t.length} service worker${t.length === 1 ? "" : "s"}`);
    } else e.push("Service workers unavailable");
    if ("caches" in window) {
      const t = await caches.keys();
      await Promise.all(t.map((n) => caches.delete(n))), e.push(`Deleted ${t.length} cache${t.length === 1 ? "" : "s"}`);
    } else e.push("Cache API unavailable");
    return localStorage.clear(), sessionStorage.clear(), e.push("Cleared local app state"), e;
  }
  async function Wx() {
    await Vx(), window.location.replace(`/?fresh=${Date.now()}`);
  }
  const Hx = [
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
  function Qx(e) {
    document.documentElement.setAttribute("data-theme", e), localStorage.setItem("maude-theme", e);
  }
  const Kx = () => {
    var _a2, _b;
    const [e, t] = g.useState(null), [n, r] = g.useState([]), [l, a] = g.useState(() => {
      const j = localStorage.getItem("maude-default-model");
      return !j || j === "mistral-large-latest" ? "nemotron-super" : j;
    }), [s, u] = g.useState(() => localStorage.getItem("maude-default-voice") || "NATF2.pt"), [i, c] = g.useState(() => localStorage.getItem("maude-theme") || "dark"), [m, d] = g.useState(false), [p, S] = g.useState(""), w = e !== null, y = (e == null ? void 0 : e.gateway_port) ?? (new URL(le()).port || "30080"), b = (_a2 = e == null ? void 0 : e.services) == null ? void 0 : _a2.llama_server, h = (_b = e == null ? void 0 : e.services) == null ? void 0 : _b.voice_server;
    g.useEffect(() => {
      fetch(`${le()}/health`).then((j) => j.json()).then(t).catch(() => t(null)), fetch(`${le()}/models`).then((j) => j.json()).then((j) => r(j.models || [])).catch(() => r([]));
    }, []);
    const f = (j) => {
      a(j), localStorage.setItem("maude-default-model", j);
    }, v = (j) => {
      u(j), localStorage.setItem("maude-default-voice", j);
    }, E = async () => {
      d(true), S("");
      try {
        await Wx();
      } catch (j) {
        S(j instanceof Error ? j.message : "Reset failed"), d(false);
      }
    }, _ = (j) => j ? j.status === "up" || j.status === "ok" ? {
      text: `${j.port} (${j.status})`,
      color: "text-green-400"
    } : {
      text: `${j.port} (down)`,
      color: "text-red-400"
    } : {
      text: "\u2014",
      color: "text-maude-muted"
    }, R = _(b), k = _(h);
    return o.jsxs("div", {
      className: "no-scrollbar h-full overflow-y-auto bg-maude-bg",
      children: [
        o.jsx("div", {
          className: "border-b border-maude-border bg-maude-surface px-4 py-3",
          children: o.jsx("h1", {
            className: "text-lg font-semibold text-maude-text",
            children: "Settings"
          })
        }),
        o.jsxs("div", {
          className: "space-y-6 p-4",
          children: [
            o.jsxs("section", {
              children: [
                o.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Connection"
                }),
                o.jsxs("div", {
                  className: "space-y-2 rounded-xl bg-maude-surface p-4",
                  children: [
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Spark Status"
                        }),
                        o.jsxs("span", {
                          className: `flex items-center gap-1.5 text-sm ${w ? "text-green-400" : "text-red-400"}`,
                          children: [
                            o.jsx("span", {
                              className: `h-2 w-2 rounded-full ${w ? "bg-green-400" : "bg-red-400"}`
                            }),
                            w ? "Connected" : "Offline"
                          ]
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Gateway"
                        }),
                        o.jsx("span", {
                          className: `font-mono text-sm ${w ? "text-green-400" : "text-maude-muted"}`,
                          children: w ? `${y} (up)` : "\u2014"
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "LLM"
                        }),
                        o.jsx("span", {
                          className: `font-mono text-sm ${R.color}`,
                          children: R.text
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Voice Server"
                        }),
                        o.jsx("span", {
                          className: `font-mono text-sm ${k.color}`,
                          children: k.text
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Tailscale"
                        }),
                        o.jsx("span", {
                          className: "text-sm text-green-400",
                          children: "Active"
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Host"
                        }),
                        o.jsx("span", {
                          className: "font-mono text-sm text-maude-muted",
                          children: le().replace(/^https?:\/\//, "")
                        })
                      ]
                    })
                  ]
                })
              ]
            }),
            o.jsxs("section", {
              children: [
                o.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Theme"
                }),
                o.jsx("div", {
                  className: "space-y-1 rounded-xl bg-maude-surface p-2",
                  children: Hx.map((j) => o.jsxs("button", {
                    onClick: () => {
                      c(j.id), Qx(j.id);
                    },
                    className: `flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${j.id === i ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                    children: [
                      o.jsx("span", {
                        children: j.label
                      }),
                      o.jsx("span", {
                        className: "text-xs text-maude-muted",
                        children: j.desc
                      })
                    ]
                  }, j.id))
                })
              ]
            }),
            o.jsxs("section", {
              children: [
                o.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Default Model"
                }),
                o.jsxs("div", {
                  className: "space-y-1 rounded-xl bg-maude-surface p-2",
                  children: [
                    n.map((j) => o.jsxs("button", {
                      onClick: () => f(j.id),
                      className: `flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${j.id === l ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                      children: [
                        o.jsxs("div", {
                          className: "flex items-center gap-2",
                          children: [
                            o.jsx("span", {
                              className: `h-2 w-2 rounded-full ${j.available ? "bg-green-400" : "bg-red-400"}`
                            }),
                            j.id
                          ]
                        }),
                        o.jsx("span", {
                          className: "text-xs text-maude-muted",
                          children: j.provider
                        })
                      ]
                    }, j.id)),
                    n.length === 0 && o.jsx("p", {
                      className: "px-3 py-2 text-sm text-maude-muted",
                      children: "Loading models..."
                    })
                  ]
                })
              ]
            }),
            o.jsxs("section", {
              children: [
                o.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Voice"
                }),
                o.jsx("div", {
                  className: "rounded-xl bg-maude-surface p-4",
                  children: o.jsx("select", {
                    value: s,
                    onChange: (j) => v(j.target.value),
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
                    ].map((j) => o.jsxs("option", {
                      value: j,
                      children: [
                        j.replace(".pt", ""),
                        j === "NATF2.pt" ? " (MAUDE)" : "",
                        j === "NATM1.pt" ? " (Male)" : ""
                      ]
                    }, j))
                  })
                })
              ]
            }),
            o.jsxs("section", {
              children: [
                o.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Network"
                }),
                o.jsxs("div", {
                  className: "space-y-3 rounded-xl bg-maude-surface p-4",
                  children: [
                    o.jsx("p", {
                      className: "text-sm text-maude-muted",
                      children: "Network settings are managed via Tailscale and your device's system settings."
                    }),
                    o.jsx("button", {
                      onClick: E,
                      disabled: m,
                      className: "w-full rounded-lg bg-maude-bg px-3 py-2.5 text-sm font-medium text-maude-text transition-colors hover:text-maude-accent disabled:opacity-50",
                      children: m ? "Resetting..." : "Reset App Cache"
                    }),
                    p && o.jsx("p", {
                      className: "text-xs text-red-400",
                      children: p
                    })
                  ]
                })
              ]
            }),
            o.jsxs("section", {
              children: [
                o.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "About"
                }),
                o.jsxs("div", {
                  className: "space-y-2 rounded-xl bg-maude-surface p-4",
                  children: [
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Version"
                        }),
                        o.jsx("span", {
                          className: "text-sm text-maude-muted",
                          children: "1.0.0"
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Build"
                        }),
                        o.jsx("span", {
                          className: "text-right font-mono text-[11px] text-maude-muted",
                          children: (/* @__PURE__ */ new Date("2026-05-27T19:35:38.781Z")).toLocaleString()
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Engine"
                        }),
                        o.jsx("span", {
                          className: "text-sm text-maude-muted",
                          children: "Mistral + Codestral + Claude"
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Voice"
                        }),
                        o.jsxs("span", {
                          className: "text-sm text-maude-muted",
                          children: [
                            "MAUDE Voice (",
                            (localStorage.getItem("maude-default-voice") || "NATF2.pt").replace(".pt", ""),
                            ")"
                          ]
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        o.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Hub"
                        }),
                        o.jsx("span", {
                          className: "text-sm font-mono",
                          children: "DGX Spark"
                        })
                      ]
                    }),
                    o.jsxs("div", {
                      className: "pt-2 text-center text-xs text-maude-muted",
                      children: [
                        o.jsx("span", {
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
  function Na() {
    return le();
  }
  function Gx(e = 1e4) {
    const [t, n] = g.useState(null), [r, l] = g.useState(true), a = g.useCallback(async () => {
      try {
        const i = await fetch(`${Na()}/api/collab/status`);
        i.ok && n(await i.json());
      } catch {
      } finally {
        l(false);
      }
    }, []);
    g.useEffect(() => {
      a();
      const i = setInterval(a, e);
      return () => clearInterval(i);
    }, [
      a,
      e
    ]);
    const s = g.useCallback(async (i, c = "", m = []) => {
      const d = await fetch(`${Na()}/api/collab/projects`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          name: i,
          description: c,
          tags: m
        })
      });
      if (d.ok) return a(), await d.json();
    }, [
      a
    ]), u = g.useCallback(async (i, c = "", m = "SHELL") => {
      const d = await fetch(`${Na()}/api/collab/tasks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: i,
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
      createProject: s,
      dispatchTask: u
    };
  }
  function Yx() {
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
  function Jx() {
    if (Bc) return;
    Bc = true;
    const e = Yx(), t = `${e.clientType}-${Math.random().toString(36).slice(2, 8)}`, n = () => {
      fetch(`${Na()}/api/collab/presence`, {
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
  function Zi(e, t) {
    const n = Math.max(0, Math.floor(t - e));
    return n < 60 ? `${n}s ago` : n < 3600 ? `${Math.floor(n / 60)}m ago` : n < 86400 ? `${Math.floor(n / 3600)}h ago` : `${Math.floor(n / 86400)}d ago`;
  }
  const Xx = {
    pending: "bg-yellow-500",
    running: "bg-blue-500",
    completed: "bg-green-500",
    failed: "bg-red-500"
  }, Vc = {
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
  }, Zx = ({ entry: e, now: t }) => o.jsxs("div", {
    className: "flex items-center gap-3 rounded-xl bg-maude-surface p-3",
    children: [
      o.jsx("div", {
        className: "flex h-10 w-10 items-center justify-center rounded-full bg-maude-card text-lg",
        children: Vc[e.client_type] || Vc.unknown
      }),
      o.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          o.jsxs("div", {
            className: "flex items-center gap-2",
            children: [
              o.jsx("span", {
                className: "font-medium text-maude-text",
                children: e.hostname
              }),
              o.jsx("span", {
                className: "text-[10px] text-maude-muted",
                children: e.client_type
              }),
              o.jsx("span", {
                className: "ml-auto inline-block h-2 w-2 rounded-full bg-green-400"
              })
            ]
          }),
          o.jsxs("p", {
            className: "truncate text-xs text-maude-muted",
            children: [
              e.activity || "idle",
              " \xB7 ",
              Zi(e.last_seen, t)
            ]
          })
        ]
      })
    ]
  }), Wc = {
    chat: "\u{1F4AC}",
    task_dispatched: "\u{1F680}",
    project_created: "\u{1F4C1}",
    custom: "\u2022"
  }, qx = ({ event: e, now: t }) => o.jsxs("div", {
    className: "flex items-start gap-2 py-1.5",
    children: [
      o.jsx("span", {
        className: "mt-0.5 text-sm",
        children: Wc[e.type] || Wc.custom
      }),
      o.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          o.jsx("p", {
            className: "text-sm text-maude-text",
            children: e.summary
          }),
          o.jsxs("p", {
            className: "text-[10px] text-maude-muted",
            children: [
              e.hostname,
              " \xB7 ",
              Zi(e.ts, t)
            ]
          })
        ]
      })
    ]
  }), ey = ({ project: e }) => o.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      o.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          o.jsx("span", {
            className: "text-sm font-medium text-maude-text",
            children: e.name
          }),
          e.tags.map((t) => o.jsx("span", {
            className: "rounded bg-maude-card px-1.5 py-0.5 text-[10px] text-maude-muted",
            children: t
          }, t))
        ]
      }),
      e.description && o.jsx("p", {
        className: "mt-1 text-xs text-maude-muted",
        children: e.description
      }),
      o.jsxs("div", {
        className: "mt-2 flex gap-3 text-[10px] text-maude-muted",
        children: [
          o.jsxs("span", {
            children: [
              e.conversations.length,
              " conversations"
            ]
          }),
          o.jsxs("span", {
            children: [
              e.files.length,
              " files"
            ]
          }),
          o.jsx("span", {
            children: e.hostname
          })
        ]
      })
    ]
  }), ty = ({ task: e, now: t }) => o.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      o.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          o.jsx("span", {
            className: `inline-block h-2 w-2 rounded-full ${Xx[e.status] || "bg-gray-500"}`
          }),
          o.jsx("span", {
            className: "text-[10px] font-medium uppercase text-maude-muted",
            children: e.status
          }),
          o.jsx("span", {
            className: "ml-auto text-[10px] text-maude-muted",
            children: Zi(e.created_at, t)
          })
        ]
      }),
      o.jsx("p", {
        className: "mt-1 truncate text-sm text-maude-text",
        children: e.prompt
      }),
      o.jsxs("div", {
        className: "mt-1 flex gap-2 text-[10px] text-maude-muted",
        children: [
          o.jsxs("span", {
            children: [
              e.source,
              " \u2192 ",
              e.target || "local"
            ]
          }),
          o.jsx("span", {
            children: e.capability
          })
        ]
      }),
      e.result && o.jsx("pre", {
        className: "mt-2 max-h-20 overflow-auto rounded bg-maude-card p-2 text-[10px] text-maude-text",
        children: e.result.slice(0, 300)
      })
    ]
  }), ny = () => {
    const { status: e, loading: t } = Gx(), [n, r] = g.useState("presence");
    if (t) return o.jsx("div", {
      className: "flex h-full items-center justify-center text-maude-muted",
      children: "Loading collaboration status..."
    });
    if (!e) return o.jsx("div", {
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
    return o.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        o.jsxs("div", {
          className: "flex items-center gap-3 px-4 pt-4 pb-2",
          children: [
            o.jsx("h1", {
              className: "text-lg font-bold text-maude-text",
              children: "Collaboration"
            }),
            o.jsxs("span", {
              className: "ml-auto flex items-center gap-1 text-xs text-maude-muted",
              children: [
                o.jsx("span", {
                  className: "inline-block h-2 w-2 rounded-full bg-green-400"
                }),
                e.hostname
              ]
            })
          ]
        }),
        o.jsx("div", {
          className: "flex gap-1 px-4 pb-3",
          children: a.map((s) => o.jsxs("button", {
            onClick: () => r(s.key),
            className: `rounded-full px-3 py-1 text-xs font-medium transition-colors ${n === s.key ? "bg-maude-accent text-white" : "bg-maude-surface text-maude-muted"}`,
            children: [
              s.label,
              s.count > 0 && o.jsx("span", {
                className: "ml-1 opacity-70",
                children: s.count
              })
            ]
          }, s.key))
        }),
        o.jsxs("div", {
          className: "flex-1 overflow-y-auto px-4 pb-4",
          children: [
            n === "presence" && o.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.presence.length === 0 ? o.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No devices online"
              }) : e.presence.map((s) => o.jsx(Zx, {
                entry: s,
                now: l
              }, s.client_id))
            }),
            n === "activity" && o.jsx("div", {
              className: "flex flex-col divide-y divide-maude-border",
              children: e.activity.length === 0 ? o.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No recent activity"
              }) : e.activity.map((s) => o.jsx(qx, {
                event: s,
                now: l
              }, s.id))
            }),
            n === "projects" && o.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.projects.length === 0 ? o.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No projects yet"
              }) : e.projects.map((s) => o.jsx(ey, {
                project: s
              }, s.id))
            }),
            n === "tasks" && o.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.tasks.length === 0 ? o.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No tasks dispatched"
              }) : e.tasks.map((s) => o.jsx(ty, {
                task: s,
                now: l
              }, s.id))
            })
          ]
        })
      ]
    });
  };
  async function Ln(e) {
    const t = await Nm(`/api/command-center/${e}`, {}, 7e3);
    if (!t.ok) throw new Error(`${e}: HTTP ${t.status}`);
    return await t.json();
  }
  async function ry() {
    const e = le();
    try {
      const t = await Nm("/api/ping", {}, 5e3);
      return t.ok ? {
        ok: true,
        url: e,
        checked_at: Date.now()
      } : {
        ok: false,
        url: e,
        error: `HTTP ${t.status}`,
        checked_at: Date.now()
      };
    } catch (t) {
      const n = t instanceof Error ? `${t.name}: ${t.message}` : String(t);
      return {
        ok: false,
        url: e,
        error: n,
        checked_at: Date.now()
      };
    }
  }
  function ly(e = 1e4) {
    const [t, n] = g.useState(null), [r, l] = g.useState(null), [a, s] = g.useState([]), [u, i] = g.useState([]), [c, m] = g.useState(null), [d, p] = g.useState(null), [S, w] = g.useState([]), [y, b] = g.useState(true), [h, f] = g.useState({
      ok: false,
      url: le()
    }), v = g.useCallback(async () => {
      const E = await ry();
      f(E);
      const [_, R, k, j, I, D, Q] = await Promise.all([
        Ln("system").catch(() => null),
        Ln("gpu-processes").catch(() => null),
        Ln("sessions?limit=10").catch(() => null),
        Ln("activity?limit=15").catch(() => null),
        Ln("scheduler").catch(() => null),
        Ln("missions?limit=20").catch(() => null),
        Ln("nodes").catch(() => null)
      ]);
      n(_), l(R && Array.isArray(R.processes) ? R : null), s((k == null ? void 0 : k.sessions) || []), i((j == null ? void 0 : j.activities) || []), m(I), p(D), w((Q == null ? void 0 : Q.nodes) || []), b(false);
    }, []);
    return g.useEffect(() => {
      v();
      const E = setInterval(v, e);
      return () => clearInterval(E);
    }, [
      v,
      e
    ]), {
      system: t,
      gpuProcesses: r,
      sessions: a,
      activity: u,
      scheduler: c,
      missions: d,
      nodes: S,
      gatewayStatus: h,
      loading: y,
      refresh: v
    };
  }
  const Lt = ({ label: e, value: t, sub: n, color: r = "text-maude-accent" }) => o.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      o.jsx("p", {
        className: "text-[10px] uppercase tracking-wider text-maude-muted",
        children: e
      }),
      o.jsx("p", {
        className: `text-xl font-bold ${r}`,
        children: t
      }),
      n && o.jsx("p", {
        className: "text-[10px] text-maude-muted",
        children: n
      })
    ]
  }), ay = ({ processes: e }) => {
    const t = e.total_mb > 0 ? e.used_mb / e.total_mb * 100 : 0;
    return o.jsxs("div", {
      className: "rounded-xl bg-maude-surface p-3",
      children: [
        o.jsxs("div", {
          className: "mb-2 flex items-center justify-between",
          children: [
            o.jsx("p", {
              className: "text-xs font-medium text-maude-text",
              children: "GPU Memory"
            }),
            o.jsxs("p", {
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
        o.jsx("div", {
          className: "h-2 overflow-hidden rounded-full bg-maude-bg",
          children: o.jsx("div", {
            className: "h-full rounded-full bg-maude-accent transition-all",
            style: {
              width: `${Math.min(t, 100)}%`
            }
          })
        }),
        e.processes.length > 0 && o.jsx("div", {
          className: "mt-2 space-y-1",
          children: e.processes.map((n) => o.jsxs("div", {
            className: "flex items-center justify-between text-[11px]",
            children: [
              o.jsx("span", {
                className: "truncate text-maude-text",
                children: n.name
              }),
              o.jsxs("span", {
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
  }, sy = ({ node: e }) => o.jsxs("div", {
    className: "flex items-center gap-3 rounded-xl bg-maude-surface p-3",
    children: [
      o.jsx("span", {
        className: `inline-block h-2.5 w-2.5 rounded-full ${e.status === "online" ? "bg-green-400" : e.status === "offline" ? "bg-red-400" : "bg-yellow-400"}`
      }),
      o.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          o.jsxs("div", {
            className: "flex items-center gap-2",
            children: [
              o.jsx("span", {
                className: "text-sm font-medium text-maude-text",
                children: e.name
              }),
              o.jsx("span", {
                className: "text-[10px] text-maude-muted",
                children: e.type
              })
            ]
          }),
          e.services && o.jsx("div", {
            className: "mt-1 flex flex-wrap gap-1.5",
            children: Object.entries(e.services).map(([t, n]) => o.jsx("span", {
              className: `rounded px-1.5 py-0.5 text-[9px] ${n ? "bg-green-400/10 text-green-400" : "bg-red-400/10 text-red-400"}`,
              children: t
            }, t))
          }),
          e.ip && o.jsxs("p", {
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
  }), oy = ({ task: e }) => o.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      o.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          o.jsx("span", {
            className: `inline-block h-2 w-2 rounded-full ${e.enabled ? "bg-green-400" : "bg-gray-500"}`
          }),
          o.jsx("span", {
            className: "text-sm font-medium text-maude-text",
            children: e.name
          }),
          o.jsx("span", {
            className: "ml-auto font-mono text-[10px] text-maude-muted",
            children: e.cron
          })
        ]
      }),
      o.jsx("p", {
        className: "mt-1 truncate text-xs text-maude-muted",
        children: e.prompt
      }),
      o.jsxs("div", {
        className: "mt-1 flex gap-3 text-[10px] text-maude-muted",
        children: [
          o.jsxs("span", {
            children: [
              e.run_count,
              " runs"
            ]
          }),
          e.last_run && o.jsxs("span", {
            children: [
              "Last: ",
              new Date(e.last_run).toLocaleDateString()
            ]
          })
        ]
      })
    ]
  }), iy = ({ mission: e }) => {
    var _a2, _b, _c2;
    const t = e.progress.total || 0, n = e.progress.done || 0, r = t > 0 ? n / t * 100 : 0, l = e.status === "blocked" ? "bg-red-400" : e.status === "complete" ? "bg-green-400" : e.status === "paused" ? "bg-yellow-400" : "bg-maude-accent";
    return o.jsxs("div", {
      className: "rounded-xl bg-maude-surface p-3",
      children: [
        o.jsxs("div", {
          className: "flex items-start gap-2",
          children: [
            o.jsx("span", {
              className: `mt-1 inline-block h-2.5 w-2.5 shrink-0 rounded-full ${l}`
            }),
            o.jsxs("div", {
              className: "min-w-0 flex-1",
              children: [
                o.jsxs("div", {
                  className: "flex items-center gap-2",
                  children: [
                    o.jsx("span", {
                      className: "truncate text-sm font-medium text-maude-text",
                      children: e.title
                    }),
                    o.jsx("span", {
                      className: "ml-auto shrink-0 text-[10px] uppercase text-maude-muted",
                      children: e.status
                    })
                  ]
                }),
                o.jsx("p", {
                  className: "mt-1 line-clamp-2 text-xs text-maude-muted",
                  children: e.objective
                })
              ]
            })
          ]
        }),
        o.jsxs("div", {
          className: "mt-3",
          children: [
            o.jsxs("div", {
              className: "mb-1 flex items-center justify-between text-[10px] text-maude-muted",
              children: [
                o.jsxs("span", {
                  children: [
                    n,
                    "/",
                    t,
                    " steps"
                  ]
                }),
                e.cadence && o.jsx("span", {
                  className: "truncate pl-2",
                  children: e.cadence
                })
              ]
            }),
            o.jsx("div", {
              className: "h-1.5 overflow-hidden rounded-full bg-maude-bg",
              children: o.jsx("div", {
                className: "h-full rounded-full bg-maude-accent transition-all",
                style: {
                  width: `${Math.min(r, 100)}%`
                }
              })
            })
          ]
        }),
        e.next_action && o.jsxs("div", {
          className: "mt-3 rounded-lg bg-maude-bg px-2 py-1.5",
          children: [
            o.jsx("p", {
              className: "text-[10px] uppercase tracking-wider text-maude-muted",
              children: "Next"
            }),
            o.jsx("p", {
              className: "line-clamp-2 text-xs text-maude-text",
              children: e.next_action
            })
          ]
        }),
        !!((_a2 = e.blockers) == null ? void 0 : _a2.length) && o.jsx("div", {
          className: "mt-2 space-y-1",
          children: e.blockers.slice(0, 2).map((a) => o.jsx("p", {
            className: "rounded bg-red-400/10 px-2 py-1 text-[11px] text-red-300",
            children: a
          }, a))
        }),
        o.jsxs("div", {
          className: "mt-2 flex flex-wrap gap-1.5 text-[10px]",
          children: [
            ((_b = e.schedule) == null ? void 0 : _b.task_id) && o.jsx("span", {
              className: "rounded bg-green-400/10 px-1.5 py-0.5 text-green-400",
              children: e.schedule.cron || "scheduled"
            }),
            !!((_c2 = e.artifacts) == null ? void 0 : _c2.length) && o.jsxs("span", {
              className: "rounded bg-maude-bg px-1.5 py-0.5 text-maude-muted",
              children: [
                e.artifacts.length,
                " artifacts"
              ]
            }),
            e.updated_at && o.jsx("span", {
              className: "rounded bg-maude-bg px-1.5 py-0.5 text-maude-muted",
              children: new Date(e.updated_at).toLocaleDateString()
            })
          ]
        })
      ]
    });
  }, uy = ({ item: e }) => o.jsxs("div", {
    className: "flex items-start gap-2 py-2",
    children: [
      o.jsx("span", {
        className: `mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full ${e.role === "user" ? "bg-green-400" : "bg-maude-accent"}`
      }),
      o.jsxs("div", {
        className: "min-w-0 flex-1",
        children: [
          o.jsxs("div", {
            className: "flex items-center gap-1.5",
            children: [
              o.jsx("span", {
                className: "text-[10px] font-medium uppercase text-maude-muted",
                children: e.channel
              }),
              o.jsx("span", {
                className: "text-[10px] text-maude-muted",
                children: e.role
              })
            ]
          }),
          o.jsx("p", {
            className: "truncate text-xs text-maude-text",
            children: e.content
          })
        ]
      })
    ]
  }), cy = ({ session: e }) => o.jsxs("div", {
    className: "flex items-center justify-between rounded-xl bg-maude-surface p-3",
    children: [
      o.jsxs("div", {
        children: [
          o.jsx("span", {
            className: "text-sm font-medium text-maude-text",
            children: e.session_id.slice(0, 8)
          }),
          o.jsx("span", {
            className: "ml-2 text-[10px] text-maude-muted",
            children: e.channel
          })
        ]
      }),
      o.jsxs("div", {
        className: "text-right",
        children: [
          o.jsxs("p", {
            className: "text-xs text-maude-muted",
            children: [
              e.message_count,
              " msgs"
            ]
          }),
          o.jsx("p", {
            className: "text-[10px] text-maude-muted",
            children: new Date(e.last_message_at).toLocaleDateString()
          })
        ]
      })
    ]
  }), dy = () => {
    var _a2, _b, _c2, _d2, _e2, _f2, _g2, _h2, _i2, _j, _k;
    const { system: e, gpuProcesses: t, sessions: n, activity: r, scheduler: l, missions: a, nodes: s, gatewayStatus: u, loading: i, refresh: c } = ly(), [m, d] = g.useState("overview");
    if (i) return o.jsx("div", {
      className: "flex h-full items-center justify-center text-maude-muted",
      children: "Loading command center..."
    });
    const p = [
      {
        key: "overview",
        label: "Overview"
      },
      {
        key: "missions",
        label: "Missions"
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
    ], S = typeof ((_a2 = e == null ? void 0 : e.gpu) == null ? void 0 : _a2.temperature_c) == "number" ? e.gpu.temperature_c : 0, w = S > 80 ? "text-red-400" : S > 60 ? "text-yellow-400" : "text-green-400";
    return o.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        o.jsxs("div", {
          className: "flex items-center gap-3 px-4 pt-4 pb-2",
          children: [
            o.jsx("h1", {
              className: "text-lg font-bold text-maude-text",
              children: "Command Center"
            }),
            o.jsx("button", {
              onClick: c,
              className: "ml-auto rounded-lg bg-maude-surface px-2 py-1 text-xs text-maude-muted active:bg-maude-card",
              children: "Refresh"
            })
          ]
        }),
        !u.ok && o.jsxs("div", {
          className: "mx-4 mb-3 rounded-lg border border-red-400/30 bg-red-400/10 p-3",
          children: [
            o.jsx("p", {
              className: "text-sm font-medium text-red-300",
              children: "Gateway not reachable"
            }),
            o.jsx("p", {
              className: "mt-1 break-all font-mono text-[11px] text-red-200",
              children: u.url
            }),
            u.error && o.jsx("p", {
              className: "mt-1 text-xs text-red-200",
              children: u.error
            })
          ]
        }),
        o.jsx("div", {
          className: "flex gap-1 px-4 pb-3",
          children: p.map((y) => o.jsx("button", {
            onClick: () => d(y.key),
            className: `rounded-full px-3 py-1 text-xs font-medium transition-colors ${m === y.key ? "bg-maude-accent text-white" : "bg-maude-surface text-maude-muted"}`,
            children: y.label
          }, y.key))
        }),
        o.jsxs("div", {
          className: "flex-1 overflow-y-auto px-4 pb-4",
          children: [
            m === "overview" && o.jsxs("div", {
              className: "space-y-3",
              children: [
                o.jsxs("div", {
                  className: "grid grid-cols-2 gap-2",
                  children: [
                    o.jsx(Lt, {
                      label: "CPU",
                      value: `${(e == null ? void 0 : e.cpu_percent) ?? 0}%`,
                      sub: `${((_b = e == null ? void 0 : e.ram) == null ? void 0 : _b.used_gb) ?? 0}/${((_c2 = e == null ? void 0 : e.ram) == null ? void 0 : _c2.total_gb) ?? 0}GB RAM`
                    }),
                    o.jsx(Lt, {
                      label: "GPU Temp",
                      value: `${S}\xB0C`,
                      sub: ((_d2 = e == null ? void 0 : e.gpu) == null ? void 0 : _d2.name) || "N/A",
                      color: w
                    }),
                    o.jsx(Lt, {
                      label: "Disk",
                      value: `${((_e2 = e == null ? void 0 : e.disk) == null ? void 0 : _e2.percent) ?? 0}%`,
                      sub: `${((_f2 = e == null ? void 0 : e.disk) == null ? void 0 : _f2.used_gb) ?? 0}/${((_g2 = e == null ? void 0 : e.disk) == null ? void 0 : _g2.total_gb) ?? 0}GB`
                    }),
                    o.jsx(Lt, {
                      label: "Missions",
                      value: ((_h2 = a == null ? void 0 : a.stats) == null ? void 0 : _h2.active) ?? 0,
                      sub: `${((_i2 = l == null ? void 0 : l.stats) == null ? void 0 : _i2.active) ?? 0} scheduled tasks`
                    })
                  ]
                }),
                t && o.jsx(ay, {
                  processes: t
                }),
                n.length > 0 && o.jsxs(o.Fragment, {
                  children: [
                    o.jsx("p", {
                      className: "pt-1 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                      children: "Recent Sessions"
                    }),
                    o.jsx("div", {
                      className: "space-y-1.5",
                      children: n.slice(0, 5).map((y) => o.jsx(cy, {
                        session: y
                      }, y.session_id + y.channel))
                    })
                  ]
                })
              ]
            }),
            m === "missions" && o.jsxs("div", {
              className: "space-y-2",
              children: [
                (a == null ? void 0 : a.stats) && o.jsxs("div", {
                  className: "grid grid-cols-4 gap-2",
                  children: [
                    o.jsx(Lt, {
                      label: "Total",
                      value: a.stats.total
                    }),
                    o.jsx(Lt, {
                      label: "Active",
                      value: a.stats.active,
                      color: "text-green-400"
                    }),
                    o.jsx(Lt, {
                      label: "Blocked",
                      value: a.stats.blocked,
                      color: "text-red-400"
                    }),
                    o.jsx(Lt, {
                      label: "Sched",
                      value: a.stats.scheduled
                    })
                  ]
                }),
                ((_j = a == null ? void 0 : a.missions) == null ? void 0 : _j.length) ? a.missions.map((y) => o.jsx(iy, {
                  mission: y
                }, y.id)) : o.jsx("p", {
                  className: "py-8 text-center text-sm text-maude-muted",
                  children: "No missions"
                })
              ]
            }),
            m === "nodes" && o.jsx("div", {
              className: "space-y-2",
              children: s.length === 0 ? o.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No nodes detected"
              }) : s.map((y, b) => o.jsx(sy, {
                node: y
              }, y.name + b))
            }),
            m === "activity" && o.jsx("div", {
              className: "divide-y divide-maude-border",
              children: r.length === 0 ? o.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No recent activity"
              }) : r.map((y, b) => o.jsx(uy, {
                item: y
              }, b))
            }),
            m === "scheduler" && o.jsxs("div", {
              className: "space-y-2",
              children: [
                (l == null ? void 0 : l.stats) && o.jsxs("div", {
                  className: "grid grid-cols-3 gap-2",
                  children: [
                    o.jsx(Lt, {
                      label: "Total",
                      value: l.stats.total
                    }),
                    o.jsx(Lt, {
                      label: "Active",
                      value: l.stats.active,
                      color: "text-green-400"
                    }),
                    o.jsx(Lt, {
                      label: "Runs",
                      value: l.stats.total_runs
                    })
                  ]
                }),
                ((_k = l == null ? void 0 : l.tasks) == null ? void 0 : _k.length) ? l.tasks.map((y) => o.jsx(oy, {
                  task: y
                }, y.id)) : o.jsx("p", {
                  className: "py-8 text-center text-sm text-maude-muted",
                  children: "No scheduled tasks"
                })
              ]
            })
          ]
        })
      ]
    });
  }, fy = [
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
  ], my = () => {
    const e = Ji(), t = vs();
    return o.jsx("nav", {
      className: "safe-bottom flex shrink-0 items-center justify-around border-t border-maude-border bg-maude-surface px-1 pb-1 pt-1",
      children: fy.map((n) => {
        const r = n.match.includes(e.pathname);
        return o.jsxs("button", {
          onClick: () => t(n.path),
          className: `flex min-h-[44px] min-w-[44px] flex-col items-center justify-center rounded-lg px-2 py-1 text-xs transition-colors ${r ? "text-maude-accent" : "text-maude-muted hover:text-maude-text"}`,
          children: [
            o.jsx("span", {
              className: "text-base leading-none",
              children: n.icon
            }),
            o.jsx("span", {
              className: "mt-0.5",
              children: n.label
            })
          ]
        }, n.path);
      })
    });
  };
  Jx();
  "serviceWorker" in navigator && navigator.serviceWorker.getRegistrations().then((e) => Promise.all(e.map((t) => t.unregister()))).catch(() => {
  });
  function py() {
    return o.jsxs("div", {
      className: "flex h-[100dvh] flex-col bg-maude-bg safe-top",
      children: [
        o.jsx("div", {
          className: "min-h-0 flex-1 overflow-hidden",
          children: o.jsx(jv, {})
        }),
        o.jsx(my, {})
      ]
    });
  }
  const hy = bv([
    {
      element: o.jsx(py, {}),
      children: [
        {
          path: "/",
          element: o.jsx(Qv, {})
        },
        {
          path: "/maude",
          element: o.jsx(Nx, {})
        },
        {
          path: "/maude/voice",
          element: o.jsx(Ox, {})
        },
        {
          path: "/terminal",
          element: o.jsx(Ax, {})
        },
        {
          path: "/browser",
          element: o.jsx(zx, {})
        },
        {
          path: "/messages",
          element: o.jsx(Ux, {})
        },
        {
          path: "/files",
          element: o.jsx(Bx, {})
        },
        {
          path: "/collab",
          element: o.jsx(ny, {})
        },
        {
          path: "/command-center",
          element: o.jsx(dy, {})
        },
        {
          path: "/settings",
          element: o.jsx(Kx, {})
        }
      ]
    }
  ]);
  Zs.createRoot(document.getElementById("root")).render(o.jsx(Iv, {
    router: hy
  }));
})();
export {
  __tla,
  gy as c,
  Qc as g
};
