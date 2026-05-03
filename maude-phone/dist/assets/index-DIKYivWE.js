let ax, Qc;
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
  ax = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
  Qc = function(e) {
    return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
  };
  var Gc = {
    exports: {}
  }, Ya = {}, Kc = {
    exports: {}
  }, ne = {};
  var Rl = Symbol.for("react.element"), Lm = Symbol.for("react.portal"), Om = Symbol.for("react.fragment"), Im = Symbol.for("react.strict_mode"), zm = Symbol.for("react.profiler"), Am = Symbol.for("react.provider"), Um = Symbol.for("react.context"), $m = Symbol.for("react.forward_ref"), Fm = Symbol.for("react.suspense"), Bm = Symbol.for("react.memo"), Vm = Symbol.for("react.lazy"), au = Symbol.iterator;
  function Wm(e) {
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
  function Ys(e, t, n) {
    this.props = e, this.context = t, this.refs = Xc, this.updater = n || Yc;
  }
  var Js = Ys.prototype = new Zc();
  Js.constructor = Ys;
  Jc(Js, Pr.prototype);
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
      $$typeof: Rl,
      type: e,
      key: a,
      ref: o,
      props: l,
      _owner: Xs.current
    };
  }
  function Hm(e, t) {
    return {
      $$typeof: Rl,
      type: e.type,
      key: t,
      ref: e.ref,
      props: e.props,
      _owner: e._owner
    };
  }
  function Zs(e) {
    return typeof e == "object" && e !== null && e.$$typeof === Rl;
  }
  function Qm(e) {
    var t = {
      "=": "=0",
      ":": "=2"
    };
    return "$" + e.replace(/[=:]/g, function(n) {
      return t[n];
    });
  }
  var su = /\/+/g;
  function xo(e, t) {
    return typeof e == "object" && e !== null && e.key != null ? Qm("" + e.key) : t.toString(36);
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
          case Rl:
          case Lm:
            o = true;
        }
    }
    if (o) return o = e, l = l(o), e = r === "" ? "." + xo(o, 0) : r, ou(l) ? (n = "", e != null && (n = e.replace(su, "$&/") + "/"), sa(l, t, n, "", function(c) {
      return c;
    })) : l != null && (Zs(l) && (l = Hm(l, n + (!l.key || o && o.key === l.key ? "" : ("" + l.key).replace(su, "$&/") + "/") + e)), t.push(l)), 1;
    if (o = 0, r = r === "" ? "." : r + ":", ou(e)) for (var i = 0; i < e.length; i++) {
      a = e[i];
      var s = r + xo(a, i);
      o += sa(a, t, n, s, l);
    }
    else if (s = Wm(e), typeof s == "function") for (e = s.call(e), i = 0; !(a = e.next()).done; ) a = a.value, s = r + xo(a, i++), o += sa(a, t, n, s, l);
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
  }, Km = {
    ReactCurrentDispatcher: lt,
    ReactCurrentBatchConfig: ia,
    ReactCurrentOwner: Xs
  };
  function nd() {
    throw Error("act(...) is not supported in production builds of React.");
  }
  ne.Children = {
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
  ne.Component = Pr;
  ne.Fragment = Om;
  ne.Profiler = zm;
  ne.PureComponent = Ys;
  ne.StrictMode = Im;
  ne.Suspense = Fm;
  ne.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Km;
  ne.act = nd;
  ne.cloneElement = function(e, t, n) {
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
      $$typeof: Rl,
      type: e.type,
      key: l,
      ref: a,
      props: r,
      _owner: o
    };
  };
  ne.createContext = function(e) {
    return e = {
      $$typeof: Um,
      _currentValue: e,
      _currentValue2: e,
      _threadCount: 0,
      Provider: null,
      Consumer: null,
      _defaultValue: null,
      _globalName: null
    }, e.Provider = {
      $$typeof: Am,
      _context: e
    }, e.Consumer = e;
  };
  ne.createElement = td;
  ne.createFactory = function(e) {
    var t = td.bind(null, e);
    return t.type = e, t;
  };
  ne.createRef = function() {
    return {
      current: null
    };
  };
  ne.forwardRef = function(e) {
    return {
      $$typeof: $m,
      render: e
    };
  };
  ne.isValidElement = Zs;
  ne.lazy = function(e) {
    return {
      $$typeof: Vm,
      _payload: {
        _status: -1,
        _result: e
      },
      _init: Gm
    };
  };
  ne.memo = function(e, t) {
    return {
      $$typeof: Bm,
      type: e,
      compare: t === void 0 ? null : t
    };
  };
  ne.startTransition = function(e) {
    var t = ia.transition;
    ia.transition = {};
    try {
      e();
    } finally {
      ia.transition = t;
    }
  };
  ne.unstable_act = nd;
  ne.useCallback = function(e, t) {
    return lt.current.useCallback(e, t);
  };
  ne.useContext = function(e) {
    return lt.current.useContext(e);
  };
  ne.useDebugValue = function() {
  };
  ne.useDeferredValue = function(e) {
    return lt.current.useDeferredValue(e);
  };
  ne.useEffect = function(e, t) {
    return lt.current.useEffect(e, t);
  };
  ne.useId = function() {
    return lt.current.useId();
  };
  ne.useImperativeHandle = function(e, t, n) {
    return lt.current.useImperativeHandle(e, t, n);
  };
  ne.useInsertionEffect = function(e, t) {
    return lt.current.useInsertionEffect(e, t);
  };
  ne.useLayoutEffect = function(e, t) {
    return lt.current.useLayoutEffect(e, t);
  };
  ne.useMemo = function(e, t) {
    return lt.current.useMemo(e, t);
  };
  ne.useReducer = function(e, t, n) {
    return lt.current.useReducer(e, t, n);
  };
  ne.useRef = function(e) {
    return lt.current.useRef(e);
  };
  ne.useState = function(e) {
    return lt.current.useState(e);
  };
  ne.useSyncExternalStore = function(e, t, n) {
    return lt.current.useSyncExternalStore(e, t, n);
  };
  ne.useTransition = function() {
    return lt.current.useTransition();
  };
  ne.version = "18.3.1";
  Kc.exports = ne;
  var v = Kc.exports;
  const Ym = Qc(v), Jm = Hc({
    __proto__: null,
    default: Ym
  }, [
    v
  ]);
  var Xm = v, Zm = Symbol.for("react.element"), qm = Symbol.for("react.fragment"), ep = Object.prototype.hasOwnProperty, tp = Xm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, np = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function rd(e, t, n) {
    var r, l = {}, a = null, o = null;
    n !== void 0 && (a = "" + n), t.key !== void 0 && (a = "" + t.key), t.ref !== void 0 && (o = t.ref);
    for (r in t) ep.call(t, r) && !np.hasOwnProperty(r) && (l[r] = t[r]);
    if (e && e.defaultProps) for (r in t = e.defaultProps, t) l[r] === void 0 && (l[r] = t[r]);
    return {
      $$typeof: Zm,
      type: e,
      key: a,
      ref: o,
      props: l,
      _owner: tp.current
    };
  }
  Ya.Fragment = qm;
  Ya.jsx = rd;
  Ya.jsxs = rd;
  Gc.exports = Ya;
  var u = Gc.exports, Xo = {}, ld = {
    exports: {}
  }, St = {}, ad = {
    exports: {}
  }, od = {};
  (function(e) {
    function t(D, Q) {
      var V = D.length;
      D.push(Q);
      e: for (; 0 < V; ) {
        var le = V - 1 >>> 1, ee = D[le];
        if (0 < l(ee, Q)) D[le] = Q, D[V] = ee, V = le;
        else break e;
      }
    }
    function n(D) {
      return D.length === 0 ? null : D[0];
    }
    function r(D) {
      if (D.length === 0) return null;
      var Q = D[0], V = D.pop();
      if (V !== Q) {
        D[0] = V;
        e: for (var le = 0, ee = D.length, fe = ee >>> 1; le < fe; ) {
          var X = 2 * (le + 1) - 1, ue = D[X], je = X + 1, te = D[je];
          if (0 > l(ue, V)) je < ee && 0 > l(te, ue) ? (D[le] = te, D[je] = V, le = je) : (D[le] = ue, D[X] = V, le = X);
          else if (je < ee && 0 > l(te, V)) D[le] = te, D[je] = V, le = je;
          else break e;
        }
      }
      return Q;
    }
    function l(D, Q) {
      var V = D.sortIndex - Q.sortIndex;
      return V !== 0 ? V : D.id - Q.id;
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
    var s = [], c = [], m = 1, d = null, g = 3, y = false, w = false, S = false, _ = typeof setTimeout == "function" ? setTimeout : null, p = typeof clearTimeout == "function" ? clearTimeout : null, f = typeof setImmediate < "u" ? setImmediate : null;
    typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
    function h(D) {
      for (var Q = n(c); Q !== null; ) {
        if (Q.callback === null) r(c);
        else if (Q.startTime <= D) r(c), Q.sortIndex = Q.expirationTime, t(s, Q);
        else break;
        Q = n(c);
      }
    }
    function j(D) {
      if (S = false, h(D), !w) if (n(s) !== null) w = true, He(C);
      else {
        var Q = n(c);
        Q !== null && mt(j, Q.startTime - D);
      }
    }
    function C(D, Q) {
      w = false, S && (S = false, p(R), R = -1), y = true;
      var V = g;
      try {
        for (h(Q), d = n(s); d !== null && (!(d.expirationTime > Q) || D && !K()); ) {
          var le = d.callback;
          if (typeof le == "function") {
            d.callback = null, g = d.priorityLevel;
            var ee = le(d.expirationTime <= Q);
            Q = e.unstable_now(), typeof ee == "function" ? d.callback = ee : d === n(s) && r(s), h(Q);
          } else r(s);
          d = n(s);
        }
        if (d !== null) var fe = true;
        else {
          var X = n(c);
          X !== null && mt(j, X.startTime - Q), fe = false;
        }
        return fe;
      } finally {
        d = null, g = V, y = false;
      }
    }
    var T = false, k = null, R = -1, $ = 5, M = -1;
    function K() {
      return !(e.unstable_now() - M < $);
    }
    function J() {
      if (k !== null) {
        var D = e.unstable_now();
        M = D;
        var Q = true;
        try {
          Q = k(true, D);
        } finally {
          Q ? de() : (T = false, k = null);
        }
      } else T = false;
    }
    var de;
    if (typeof f == "function") de = function() {
      f(J);
    };
    else if (typeof MessageChannel < "u") {
      var se = new MessageChannel(), _e = se.port2;
      se.port1.onmessage = J, de = function() {
        _e.postMessage(null);
      };
    } else de = function() {
      _(J, 0);
    };
    function He(D) {
      k = D, T || (T = true, de());
    }
    function mt(D, Q) {
      R = _(function() {
        D(e.unstable_now());
      }, Q);
    }
    e.unstable_IdlePriority = 5, e.unstable_ImmediatePriority = 1, e.unstable_LowPriority = 4, e.unstable_NormalPriority = 3, e.unstable_Profiling = null, e.unstable_UserBlockingPriority = 2, e.unstable_cancelCallback = function(D) {
      D.callback = null;
    }, e.unstable_continueExecution = function() {
      w || y || (w = true, He(C));
    }, e.unstable_forceFrameRate = function(D) {
      0 > D || 125 < D ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : $ = 0 < D ? Math.floor(1e3 / D) : 5;
    }, e.unstable_getCurrentPriorityLevel = function() {
      return g;
    }, e.unstable_getFirstCallbackNode = function() {
      return n(s);
    }, e.unstable_next = function(D) {
      switch (g) {
        case 1:
        case 2:
        case 3:
          var Q = 3;
          break;
        default:
          Q = g;
      }
      var V = g;
      g = Q;
      try {
        return D();
      } finally {
        g = V;
      }
    }, e.unstable_pauseExecution = function() {
    }, e.unstable_requestPaint = function() {
    }, e.unstable_runWithPriority = function(D, Q) {
      switch (D) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          break;
        default:
          D = 3;
      }
      var V = g;
      g = D;
      try {
        return Q();
      } finally {
        g = V;
      }
    }, e.unstable_scheduleCallback = function(D, Q, V) {
      var le = e.unstable_now();
      switch (typeof V == "object" && V !== null ? (V = V.delay, V = typeof V == "number" && 0 < V ? le + V : le) : V = le, D) {
        case 1:
          var ee = -1;
          break;
        case 2:
          ee = 250;
          break;
        case 5:
          ee = 1073741823;
          break;
        case 4:
          ee = 1e4;
          break;
        default:
          ee = 5e3;
      }
      return ee = V + ee, D = {
        id: m++,
        callback: Q,
        priorityLevel: D,
        startTime: V,
        expirationTime: ee,
        sortIndex: -1
      }, V > le ? (D.sortIndex = V, t(c, D), n(s) === null && D === n(c) && (S ? (p(R), R = -1) : S = true, mt(j, V - le))) : (D.sortIndex = ee, t(s, D), w || y || (w = true, He(C))), D;
    }, e.unstable_shouldYield = K, e.unstable_wrapCallback = function(D) {
      var Q = g;
      return function() {
        var V = g;
        g = Q;
        try {
          return D.apply(this, arguments);
        } finally {
          g = V;
        }
      };
    };
  })(od);
  ad.exports = od;
  var rp = ad.exports;
  var lp = v, wt = rp;
  function b(e) {
    for (var t = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, n = 1; n < arguments.length; n++) t += "&args[]=" + encodeURIComponent(arguments[n]);
    return "Minified React error #" + e + "; visit " + t + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  var sd = /* @__PURE__ */ new Set(), ul = {};
  function Yn(e, t) {
    Nr(e, t), Nr(e + "Capture", t);
  }
  function Nr(e, t) {
    for (ul[e] = t, e = 0; e < t.length; e++) sd.add(t[e]);
  }
  var qt = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), Zo = Object.prototype.hasOwnProperty, ap = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, iu = {}, uu = {};
  function op(e) {
    return Zo.call(uu, e) ? true : Zo.call(iu, e) ? false : ap.test(e) ? uu[e] = true : (iu[e] = true, false);
  }
  function sp(e, t, n, r) {
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
  function ip(e, t, n, r) {
    if (t === null || typeof t > "u" || sp(e, t, n, r)) return true;
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
  var Je = {};
  "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e) {
    Je[e] = new at(e, 0, false, e, null, false, false);
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
    Je[t] = new at(t, 1, false, e[1], null, false, false);
  });
  [
    "contentEditable",
    "draggable",
    "spellCheck",
    "value"
  ].forEach(function(e) {
    Je[e] = new at(e, 2, false, e.toLowerCase(), null, false, false);
  });
  [
    "autoReverse",
    "externalResourcesRequired",
    "focusable",
    "preserveAlpha"
  ].forEach(function(e) {
    Je[e] = new at(e, 2, false, e, null, false, false);
  });
  "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e) {
    Je[e] = new at(e, 3, false, e.toLowerCase(), null, false, false);
  });
  [
    "checked",
    "multiple",
    "muted",
    "selected"
  ].forEach(function(e) {
    Je[e] = new at(e, 3, true, e, null, false, false);
  });
  [
    "capture",
    "download"
  ].forEach(function(e) {
    Je[e] = new at(e, 4, false, e, null, false, false);
  });
  [
    "cols",
    "rows",
    "size",
    "span"
  ].forEach(function(e) {
    Je[e] = new at(e, 6, false, e, null, false, false);
  });
  [
    "rowSpan",
    "start"
  ].forEach(function(e) {
    Je[e] = new at(e, 5, false, e.toLowerCase(), null, false, false);
  });
  var qs = /[\-:]([a-z])/g;
  function ei(e) {
    return e[1].toUpperCase();
  }
  "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e) {
    var t = e.replace(qs, ei);
    Je[t] = new at(t, 1, false, e, null, false, false);
  });
  "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e) {
    var t = e.replace(qs, ei);
    Je[t] = new at(t, 1, false, e, "http://www.w3.org/1999/xlink", false, false);
  });
  [
    "xml:base",
    "xml:lang",
    "xml:space"
  ].forEach(function(e) {
    var t = e.replace(qs, ei);
    Je[t] = new at(t, 1, false, e, "http://www.w3.org/XML/1998/namespace", false, false);
  });
  [
    "tabIndex",
    "crossOrigin"
  ].forEach(function(e) {
    Je[e] = new at(e, 1, false, e.toLowerCase(), null, false, false);
  });
  Je.xlinkHref = new at("xlinkHref", 1, false, "xlink:href", "http://www.w3.org/1999/xlink", true, false);
  [
    "src",
    "href",
    "action",
    "formAction"
  ].forEach(function(e) {
    Je[e] = new at(e, 1, false, e.toLowerCase(), null, true, true);
  });
  function ti(e, t, n, r) {
    var l = Je.hasOwnProperty(t) ? Je[t] : null;
    (l !== null ? l.type !== 0 : r || !(2 < t.length) || t[0] !== "o" && t[0] !== "O" || t[1] !== "n" && t[1] !== "N") && (ip(t, n, l, r) && (n = null), r || l === null ? op(t) && (n === null ? e.removeAttribute(t) : e.setAttribute(t, "" + n)) : l.mustUseProperty ? e[l.propertyName] = n === null ? l.type === 3 ? false : "" : n : (t = l.attributeName, r = l.attributeNamespace, n === null ? e.removeAttribute(t) : (l = l.type, n = l === 3 || l === 4 && n === true ? "" : "" + n, r ? e.setAttributeNS(r, t, n) : e.setAttribute(t, n))));
  }
  var rn = lp.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, Vl = Symbol.for("react.element"), ar = Symbol.for("react.portal"), or = Symbol.for("react.fragment"), ni = Symbol.for("react.strict_mode"), qo = Symbol.for("react.profiler"), id = Symbol.for("react.provider"), ud = Symbol.for("react.context"), ri = Symbol.for("react.forward_ref"), es = Symbol.for("react.suspense"), ts = Symbol.for("react.suspense_list"), li = Symbol.for("react.memo"), un = Symbol.for("react.lazy"), cd = Symbol.for("react.offscreen"), cu = Symbol.iterator;
  function Ir(e) {
    return e === null || typeof e != "object" ? null : (e = cu && e[cu] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var Pe = Object.assign, wo;
  function Kr(e) {
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
    return (e = e ? e.displayName || e.name : "") ? Kr(e) : "";
  }
  function up(e) {
    switch (e.tag) {
      case 5:
        return Kr(e.type);
      case 16:
        return Kr("Lazy");
      case 13:
        return Kr("Suspense");
      case 19:
        return Kr("SuspenseList");
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
      case or:
        return "Fragment";
      case ar:
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
      case un:
        t = e._payload, e = e._init;
        try {
          return ns(e(t));
        } catch {
        }
    }
    return null;
  }
  function cp(e) {
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
  function En(e) {
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
  function dp(e) {
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
  function Wl(e) {
    e._valueTracker || (e._valueTracker = dp(e));
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
    return Pe({}, t, {
      defaultChecked: void 0,
      defaultValue: void 0,
      value: void 0,
      checked: n ?? e._wrapperState.initialChecked
    });
  }
  function du(e, t) {
    var n = t.defaultValue == null ? "" : t.defaultValue, r = t.checked != null ? t.checked : t.defaultChecked;
    n = En(t.value != null ? t.value : n), e._wrapperState = {
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
    var n = En(t.value), r = t.type;
    if (n != null) r === "number" ? (n === 0 && e.value === "" || e.value != n) && (e.value = "" + n) : e.value !== "" + n && (e.value = "" + n);
    else if (r === "submit" || r === "reset") {
      e.removeAttribute("value");
      return;
    }
    t.hasOwnProperty("value") ? as(e, t.type, n) : t.hasOwnProperty("defaultValue") && as(e, t.type, En(t.defaultValue)), t.checked == null && t.defaultChecked != null && (e.defaultChecked = !!t.defaultChecked);
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
  function vr(e, t, n, r) {
    if (e = e.options, t) {
      t = {};
      for (var l = 0; l < n.length; l++) t["$" + n[l]] = true;
      for (n = 0; n < e.length; n++) l = t.hasOwnProperty("$" + e[n].value), e[n].selected !== l && (e[n].selected = l), l && r && (e[n].defaultSelected = true);
    } else {
      for (n = "" + En(n), t = null, l = 0; l < e.length; l++) {
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
    return Pe({}, t, {
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
      initialValue: En(n)
    };
  }
  function pd(e, t) {
    var n = En(t.value), r = En(t.defaultValue);
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
  }, fp = [
    "Webkit",
    "ms",
    "Moz",
    "O"
  ];
  Object.keys(qr).forEach(function(e) {
    fp.forEach(function(t) {
      t = t + e.charAt(0).toUpperCase() + e.substring(1), qr[t] = qr[e];
    });
  });
  function vd(e, t, n) {
    return t == null || typeof t == "boolean" || t === "" ? "" : n || typeof t != "number" || t === 0 || qr.hasOwnProperty(e) && qr[e] ? ("" + t).trim() : t + "px";
  }
  function yd(e, t) {
    e = e.style;
    for (var n in t) if (t.hasOwnProperty(n)) {
      var r = n.indexOf("--") === 0, l = vd(n, t[n], r);
      n === "float" && (n = "cssFloat"), r ? e.setProperty(n, l) : e[n] = l;
    }
  }
  var mp = Pe({
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
      if (mp[e] && (t.children != null || t.dangerouslySetInnerHTML != null)) throw Error(b(137, e));
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
  var ds = null, yr = null, xr = null;
  function hu(e) {
    if (e = bl(e)) {
      if (typeof ds != "function") throw Error(b(280));
      var t = e.stateNode;
      t && (t = eo(t), ds(e.stateNode, e.type, t));
    }
  }
  function xd(e) {
    yr ? xr ? xr.push(e) : xr = [
      e
    ] : yr = e;
  }
  function wd() {
    if (yr) {
      var e = yr, t = xr;
      if (xr = yr = null, hu(e), t) for (e = 0; e < t.length; e++) hu(t[e]);
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
      No = false, (yr !== null || xr !== null) && (kd(), wd());
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
  if (qt) try {
    var zr = {};
    Object.defineProperty(zr, "passive", {
      get: function() {
        fs = true;
      }
    }), window.addEventListener("test", zr, zr), window.removeEventListener("test", zr, zr);
  } catch {
    fs = false;
  }
  function pp(e, t, n, r, l, a, o, i, s) {
    var c = Array.prototype.slice.call(arguments, 3);
    try {
      t.apply(n, c);
    } catch (m) {
      this.onError(m);
    }
  }
  var el = false, ka = null, Na = false, ms = null, hp = {
    onError: function(e) {
      el = true, ka = e;
    }
  };
  function gp(e, t, n, r, l, a, o, i, s) {
    el = false, ka = null, pp.apply(hp, arguments);
  }
  function vp(e, t, n, r, l, a, o, i, s) {
    if (gp.apply(this, arguments), el) {
      if (el) {
        var c = ka;
        el = false, ka = null;
      } else throw Error(b(198));
      Na || (Na = true, ms = c);
    }
  }
  function Jn(e) {
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
    if (Jn(e) !== e) throw Error(b(188));
  }
  function yp(e) {
    var t = e.alternate;
    if (!t) {
      if (t = Jn(e), t === null) throw Error(b(188));
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
  var _d = wt.unstable_scheduleCallback, vu = wt.unstable_cancelCallback, xp = wt.unstable_shouldYield, wp = wt.unstable_requestPaint, Oe = wt.unstable_now, Sp = wt.unstable_getCurrentPriorityLevel, oi = wt.unstable_ImmediatePriority, Rd = wt.unstable_UserBlockingPriority, ja = wt.unstable_NormalPriority, kp = wt.unstable_LowPriority, Td = wt.unstable_IdlePriority, Ja = null, Wt = null;
  function Np(e) {
    if (Wt && typeof Wt.onCommitFiberRoot == "function") try {
      Wt.onCommitFiberRoot(Ja, e, void 0, (e.current.flags & 128) === 128);
    } catch {
    }
  }
  var It = Math.clz32 ? Math.clz32 : Cp, jp = Math.log, Ep = Math.LN2;
  function Cp(e) {
    return e >>>= 0, e === 0 ? 32 : 31 - (jp(e) / Ep | 0) | 0;
  }
  var Ql = 64, Gl = 4194304;
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
    if (r & 4 && (r |= n & 16), t = e.entangledLanes, t !== 0) for (e = e.entanglements, t &= r; 0 < t; ) n = 31 - It(t), l = 1 << n, r |= e[n], t &= ~l;
    return r;
  }
  function _p(e, t) {
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
      var o = 31 - It(a), i = 1 << o, s = l[o];
      s === -1 ? (!(i & n) || i & r) && (l[o] = _p(i, t)) : s <= t && (e.expiredLanes |= i), a &= ~i;
    }
  }
  function ps(e) {
    return e = e.pendingLanes & -1073741825, e !== 0 ? e : e & 1073741824 ? 1073741824 : 0;
  }
  function Pd() {
    var e = Ql;
    return Ql <<= 1, !(Ql & 4194240) && (Ql = 64), e;
  }
  function jo(e) {
    for (var t = [], n = 0; 31 > n; n++) t.push(e);
    return t;
  }
  function Tl(e, t, n) {
    e.pendingLanes |= t, t !== 536870912 && (e.suspendedLanes = 0, e.pingedLanes = 0), e = e.eventTimes, t = 31 - It(t), e[t] = n;
  }
  function Tp(e, t) {
    var n = e.pendingLanes & ~t;
    e.pendingLanes = t, e.suspendedLanes = 0, e.pingedLanes = 0, e.expiredLanes &= t, e.mutableReadLanes &= t, e.entangledLanes &= t, t = e.entanglements;
    var r = e.eventTimes;
    for (e = e.expirationTimes; 0 < n; ) {
      var l = 31 - It(n), a = 1 << l;
      t[l] = 0, r[l] = -1, e[l] = -1, n &= ~a;
    }
  }
  function si(e, t) {
    var n = e.entangledLanes |= t;
    for (e = e.entanglements; n; ) {
      var r = 31 - It(n), l = 1 << r;
      l & t | e[r] & t && (e[r] |= t), n &= ~l;
    }
  }
  var ge = 0;
  function bd(e) {
    return e &= -e, 1 < e ? 4 < e ? e & 268435455 ? 16 : 536870912 : 4 : 1;
  }
  var Dd, ii, Md, Ld, Od, hs = false, Kl = [], gn = null, vn = null, yn = null, fl = /* @__PURE__ */ new Map(), ml = /* @__PURE__ */ new Map(), dn = [], Pp = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
  function yu(e, t) {
    switch (e) {
      case "focusin":
      case "focusout":
        gn = null;
        break;
      case "dragenter":
      case "dragleave":
        vn = null;
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
  function bp(e, t, n, r, l) {
    switch (t) {
      case "focusin":
        return gn = Ar(gn, e, t, n, r, l), true;
      case "dragenter":
        return vn = Ar(vn, e, t, n, r, l), true;
      case "mouseover":
        return yn = Ar(yn, e, t, n, r, l), true;
      case "pointerover":
        var a = l.pointerId;
        return fl.set(a, Ar(fl.get(a) || null, e, t, n, r, l)), true;
      case "gotpointercapture":
        return a = l.pointerId, ml.set(a, Ar(ml.get(a) || null, e, t, n, r, l)), true;
    }
    return false;
  }
  function Id(e) {
    var t = zn(e.target);
    if (t !== null) {
      var n = Jn(t);
      if (n !== null) {
        if (t = n.tag, t === 13) {
          if (t = jd(n), t !== null) {
            e.blockedOn = t, Od(e.priority, function() {
              Md(n);
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
  function xu(e, t, n) {
    ua(e) && n.delete(t);
  }
  function Dp() {
    hs = false, gn !== null && ua(gn) && (gn = null), vn !== null && ua(vn) && (vn = null), yn !== null && ua(yn) && (yn = null), fl.forEach(xu), ml.forEach(xu);
  }
  function Ur(e, t) {
    e.blockedOn === t && (e.blockedOn = null, hs || (hs = true, wt.unstable_scheduleCallback(wt.unstable_NormalPriority, Dp)));
  }
  function pl(e) {
    function t(l) {
      return Ur(l, e);
    }
    if (0 < Kl.length) {
      Ur(Kl[0], e);
      for (var n = 1; n < Kl.length; n++) {
        var r = Kl[n];
        r.blockedOn === e && (r.blockedOn = null);
      }
    }
    for (gn !== null && Ur(gn, e), vn !== null && Ur(vn, e), yn !== null && Ur(yn, e), fl.forEach(t), ml.forEach(t), n = 0; n < dn.length; n++) r = dn[n], r.blockedOn === e && (r.blockedOn = null);
    for (; 0 < dn.length && (n = dn[0], n.blockedOn === null); ) Id(n), n.blockedOn === null && dn.shift();
  }
  var wr = rn.ReactCurrentBatchConfig, Ca = true;
  function Mp(e, t, n, r) {
    var l = ge, a = wr.transition;
    wr.transition = null;
    try {
      ge = 1, ui(e, t, n, r);
    } finally {
      ge = l, wr.transition = a;
    }
  }
  function Lp(e, t, n, r) {
    var l = ge, a = wr.transition;
    wr.transition = null;
    try {
      ge = 4, ui(e, t, n, r);
    } finally {
      ge = l, wr.transition = a;
    }
  }
  function ui(e, t, n, r) {
    if (Ca) {
      var l = gs(e, t, n, r);
      if (l === null) Lo(e, t, r, _a, n), yu(e, r);
      else if (bp(l, e, t, n, r)) r.stopPropagation();
      else if (yu(e, r), t & 4 && -1 < Pp.indexOf(e)) {
        for (; l !== null; ) {
          var a = bl(l);
          if (a !== null && Dd(a), a = gs(e, t, n, r), a === null && Lo(e, t, r, _a, n), a === l) break;
          l = a;
        }
        l !== null && r.stopPropagation();
      } else Lo(e, t, r, null, n);
    }
  }
  var _a = null;
  function gs(e, t, n, r) {
    if (_a = null, e = ai(r), e = zn(e), e !== null) if (t = Jn(e), t === null) e = null;
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
        switch (Sp()) {
          case oi:
            return 1;
          case Rd:
            return 4;
          case ja:
          case kp:
            return 16;
          case Td:
            return 536870912;
          default:
            return 16;
        }
      default:
        return 16;
    }
  }
  var mn = null, ci = null, ca = null;
  function Ad() {
    if (ca) return ca;
    var e, t = ci, n = t.length, r, l = "value" in mn ? mn.value : mn.textContent, a = l.length;
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
  function kt(e) {
    function t(n, r, l, a, o) {
      this._reactName = n, this._targetInst = l, this.type = r, this.nativeEvent = a, this.target = o, this.currentTarget = null;
      for (var i in e) e.hasOwnProperty(i) && (n = e[i], this[i] = n ? n(a) : a[i]);
      return this.isDefaultPrevented = (a.defaultPrevented != null ? a.defaultPrevented : a.returnValue === false) ? Yl : wu, this.isPropagationStopped = wu, this;
    }
    return Pe(t.prototype, {
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
  var br = {
    eventPhase: 0,
    bubbles: 0,
    cancelable: 0,
    timeStamp: function(e) {
      return e.timeStamp || Date.now();
    },
    defaultPrevented: 0,
    isTrusted: 0
  }, di = kt(br), Pl = Pe({}, br, {
    view: 0,
    detail: 0
  }), Op = kt(Pl), Eo, Co, $r, Xa = Pe({}, Pl, {
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
  }), Su = kt(Xa), Ip = Pe({}, Xa, {
    dataTransfer: 0
  }), zp = kt(Ip), Ap = Pe({}, Pl, {
    relatedTarget: 0
  }), _o = kt(Ap), Up = Pe({}, br, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), $p = kt(Up), Fp = Pe({}, br, {
    clipboardData: function(e) {
      return "clipboardData" in e ? e.clipboardData : window.clipboardData;
    }
  }), Bp = kt(Fp), Vp = Pe({}, br, {
    data: 0
  }), ku = kt(Vp), Wp = {
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
  }, Hp = {
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
  }, Qp = {
    Alt: "altKey",
    Control: "ctrlKey",
    Meta: "metaKey",
    Shift: "shiftKey"
  };
  function Gp(e) {
    var t = this.nativeEvent;
    return t.getModifierState ? t.getModifierState(e) : (e = Qp[e]) ? !!t[e] : false;
  }
  function fi() {
    return Gp;
  }
  var Kp = Pe({}, Pl, {
    key: function(e) {
      if (e.key) {
        var t = Wp[e.key] || e.key;
        if (t !== "Unidentified") return t;
      }
      return e.type === "keypress" ? (e = da(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? Hp[e.keyCode] || "Unidentified" : "";
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
  }), Yp = kt(Kp), Jp = Pe({}, Xa, {
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
  }), Nu = kt(Jp), Xp = Pe({}, Pl, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: fi
  }), Zp = kt(Xp), qp = Pe({}, br, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), eh = kt(qp), th = Pe({}, Xa, {
    deltaX: function(e) {
      return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
    },
    deltaY: function(e) {
      return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), nh = kt(th), rh = [
    9,
    13,
    27,
    32
  ], mi = qt && "CompositionEvent" in window, tl = null;
  qt && "documentMode" in document && (tl = document.documentMode);
  var lh = qt && "TextEvent" in window && !tl, Ud = qt && (!mi || tl && 8 < tl && 11 >= tl), ju = " ", Eu = false;
  function $d(e, t) {
    switch (e) {
      case "keyup":
        return rh.indexOf(t.keyCode) !== -1;
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
  var sr = false;
  function ah(e, t) {
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
  function oh(e, t) {
    if (sr) return e === "compositionend" || !mi && $d(e, t) ? (e = Ad(), ca = ci = mn = null, sr = false, e) : null;
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
  var sh = {
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
    return t === "input" ? !!sh[e.type] : t === "textarea";
  }
  function Bd(e, t, n, r) {
    xd(r), t = Ra(t, "onChange"), 0 < t.length && (n = new di("onChange", "change", null, n, r), e.push({
      event: n,
      listeners: t
    }));
  }
  var nl = null, hl = null;
  function ih(e) {
    qd(e, 0);
  }
  function Za(e) {
    var t = cr(e);
    if (fd(t)) return e;
  }
  function uh(e, t) {
    if (e === "change") return t;
  }
  var Vd = false;
  if (qt) {
    var Ro;
    if (qt) {
      var To = "oninput" in document;
      if (!To) {
        var _u = document.createElement("div");
        _u.setAttribute("oninput", "return;"), To = typeof _u.oninput == "function";
      }
      Ro = To;
    } else Ro = false;
    Vd = Ro && (!document.documentMode || 9 < document.documentMode);
  }
  function Ru() {
    nl && (nl.detachEvent("onpropertychange", Wd), hl = nl = null);
  }
  function Wd(e) {
    if (e.propertyName === "value" && Za(hl)) {
      var t = [];
      Bd(t, hl, e, ai(e)), Nd(ih, t);
    }
  }
  function ch(e, t, n) {
    e === "focusin" ? (Ru(), nl = t, hl = n, nl.attachEvent("onpropertychange", Wd)) : e === "focusout" && Ru();
  }
  function dh(e) {
    if (e === "selectionchange" || e === "keyup" || e === "keydown") return Za(hl);
  }
  function fh(e, t) {
    if (e === "click") return Za(t);
  }
  function mh(e, t) {
    if (e === "input" || e === "change") return Za(t);
  }
  function ph(e, t) {
    return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
  }
  var At = typeof Object.is == "function" ? Object.is : ph;
  function gl(e, t) {
    if (At(e, t)) return true;
    if (typeof e != "object" || e === null || typeof t != "object" || t === null) return false;
    var n = Object.keys(e), r = Object.keys(t);
    if (n.length !== r.length) return false;
    for (r = 0; r < n.length; r++) {
      var l = n[r];
      if (!Zo.call(t, l) || !At(e[l], t[l])) return false;
    }
    return true;
  }
  function Tu(e) {
    for (; e && e.firstChild; ) e = e.firstChild;
    return e;
  }
  function Pu(e, t) {
    var n = Tu(e);
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
      n = Tu(n);
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
  function hh(e) {
    var t = Qd(), n = e.focusedElem, r = e.selectionRange;
    if (t !== n && n && n.ownerDocument && Hd(n.ownerDocument.documentElement, n)) {
      if (r !== null && pi(n)) {
        if (t = r.start, e = r.end, e === void 0 && (e = t), "selectionStart" in n) n.selectionStart = t, n.selectionEnd = Math.min(e, n.value.length);
        else if (e = (t = n.ownerDocument || document) && t.defaultView || window, e.getSelection) {
          e = e.getSelection();
          var l = n.textContent.length, a = Math.min(r.start, l);
          r = r.end === void 0 ? a : Math.min(r.end, l), !e.extend && a > r && (l = r, r = a, a = l), l = Pu(n, a);
          var o = Pu(n, r);
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
  var gh = qt && "documentMode" in document && 11 >= document.documentMode, ir = null, vs = null, rl = null, ys = false;
  function bu(e, t, n) {
    var r = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
    ys || ir == null || ir !== Sa(r) || (r = ir, "selectionStart" in r && pi(r) ? r = {
      start: r.selectionStart,
      end: r.selectionEnd
    } : (r = (r.ownerDocument && r.ownerDocument.defaultView || window).getSelection(), r = {
      anchorNode: r.anchorNode,
      anchorOffset: r.anchorOffset,
      focusNode: r.focusNode,
      focusOffset: r.focusOffset
    }), rl && gl(rl, r) || (rl = r, r = Ra(vs, "onSelect"), 0 < r.length && (t = new di("onSelect", "select", null, t, n), e.push({
      event: t,
      listeners: r
    }), t.target = ir)));
  }
  function Jl(e, t) {
    var n = {};
    return n[e.toLowerCase()] = t.toLowerCase(), n["Webkit" + e] = "webkit" + t, n["Moz" + e] = "moz" + t, n;
  }
  var ur = {
    animationend: Jl("Animation", "AnimationEnd"),
    animationiteration: Jl("Animation", "AnimationIteration"),
    animationstart: Jl("Animation", "AnimationStart"),
    transitionend: Jl("Transition", "TransitionEnd")
  }, Po = {}, Gd = {};
  qt && (Gd = document.createElement("div").style, "AnimationEvent" in window || (delete ur.animationend.animation, delete ur.animationiteration.animation, delete ur.animationstart.animation), "TransitionEvent" in window || delete ur.transitionend.transition);
  function qa(e) {
    if (Po[e]) return Po[e];
    if (!ur[e]) return e;
    var t = ur[e], n;
    for (n in t) if (t.hasOwnProperty(n) && n in Gd) return Po[e] = t[n];
    return e;
  }
  var Kd = qa("animationend"), Yd = qa("animationiteration"), Jd = qa("animationstart"), Xd = qa("transitionend"), Zd = /* @__PURE__ */ new Map(), Du = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
  function _n(e, t) {
    Zd.set(e, t), Yn(t, [
      e
    ]);
  }
  for (var bo = 0; bo < Du.length; bo++) {
    var Do = Du[bo], vh = Do.toLowerCase(), yh = Do[0].toUpperCase() + Do.slice(1);
    _n(vh, "on" + yh);
  }
  _n(Kd, "onAnimationEnd");
  _n(Yd, "onAnimationIteration");
  _n(Jd, "onAnimationStart");
  _n("dblclick", "onDoubleClick");
  _n("focusin", "onFocus");
  _n("focusout", "onBlur");
  _n(Xd, "onTransitionEnd");
  Nr("onMouseEnter", [
    "mouseout",
    "mouseover"
  ]);
  Nr("onMouseLeave", [
    "mouseout",
    "mouseover"
  ]);
  Nr("onPointerEnter", [
    "pointerout",
    "pointerover"
  ]);
  Nr("onPointerLeave", [
    "pointerout",
    "pointerover"
  ]);
  Yn("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
  Yn("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
  Yn("onBeforeInput", [
    "compositionend",
    "keypress",
    "textInput",
    "paste"
  ]);
  Yn("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
  Yn("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
  Yn("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
  var Xr = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), xh = new Set("cancel close invalid load scroll toggle".split(" ").concat(Xr));
  function Mu(e, t, n) {
    var r = e.type || "unknown-event";
    e.currentTarget = n, vp(r, t, void 0, e), e.currentTarget = null;
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
          Mu(l, i, c), a = s;
        }
        else for (o = 0; o < r.length; o++) {
          if (i = r[o], s = i.instance, c = i.currentTarget, i = i.listener, s !== a && l.isPropagationStopped()) break e;
          Mu(l, i, c), a = s;
        }
      }
    }
    if (Na) throw e = ms, Na = false, ms = null, e;
  }
  function ke(e, t) {
    var n = t[Ns];
    n === void 0 && (n = t[Ns] = /* @__PURE__ */ new Set());
    var r = e + "__bubble";
    n.has(r) || (ef(t, e, 2, false), n.add(r));
  }
  function Mo(e, t, n) {
    var r = 0;
    t && (r |= 4), ef(n, e, r, t);
  }
  var Xl = "_reactListening" + Math.random().toString(36).slice(2);
  function vl(e) {
    if (!e[Xl]) {
      e[Xl] = true, sd.forEach(function(n) {
        n !== "selectionchange" && (xh.has(n) || Mo(n, false, e), Mo(n, true, e));
      });
      var t = e.nodeType === 9 ? e : e.ownerDocument;
      t === null || t[Xl] || (t[Xl] = true, Mo("selectionchange", false, t));
    }
  }
  function ef(e, t, n, r) {
    switch (zd(t)) {
      case 1:
        var l = Mp;
        break;
      case 4:
        l = Lp;
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
          if (o = zn(i), o === null) return;
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
          var y = di, w = e;
          switch (e) {
            case "keypress":
              if (da(n) === 0) break e;
            case "keydown":
            case "keyup":
              y = Yp;
              break;
            case "focusin":
              w = "focus", y = _o;
              break;
            case "focusout":
              w = "blur", y = _o;
              break;
            case "beforeblur":
            case "afterblur":
              y = _o;
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
              y = Su;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              y = zp;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              y = Zp;
              break;
            case Kd:
            case Yd:
            case Jd:
              y = $p;
              break;
            case Xd:
              y = eh;
              break;
            case "scroll":
              y = Op;
              break;
            case "wheel":
              y = nh;
              break;
            case "copy":
            case "cut":
            case "paste":
              y = Bp;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              y = Nu;
          }
          var S = (t & 4) !== 0, _ = !S && e === "scroll", p = S ? g !== null ? g + "Capture" : null : g;
          S = [];
          for (var f = c, h; f !== null; ) {
            h = f;
            var j = h.stateNode;
            if (h.tag === 5 && j !== null && (h = j, p !== null && (j = dl(f, p), j != null && S.push(yl(f, j, h)))), _) break;
            f = f.return;
          }
          0 < S.length && (g = new y(g, w, null, n, m), d.push({
            event: g,
            listeners: S
          }));
        }
      }
      if (!(t & 7)) {
        e: {
          if (g = e === "mouseover" || e === "pointerover", y = e === "mouseout" || e === "pointerout", g && n !== cs && (w = n.relatedTarget || n.fromElement) && (zn(w) || w[en])) break e;
          if ((y || g) && (g = m.window === m ? m : (g = m.ownerDocument) ? g.defaultView || g.parentWindow : window, y ? (w = n.relatedTarget || n.toElement, y = c, w = w ? zn(w) : null, w !== null && (_ = Jn(w), w !== _ || w.tag !== 5 && w.tag !== 6) && (w = null)) : (y = null, w = c), y !== w)) {
            if (S = Su, j = "onMouseLeave", p = "onMouseEnter", f = "mouse", (e === "pointerout" || e === "pointerover") && (S = Nu, j = "onPointerLeave", p = "onPointerEnter", f = "pointer"), _ = y == null ? g : cr(y), h = w == null ? g : cr(w), g = new S(j, f + "leave", y, n, m), g.target = _, g.relatedTarget = h, j = null, zn(m) === c && (S = new S(p, f + "enter", w, n, m), S.target = h, S.relatedTarget = _, j = S), _ = j, y && w) t: {
              for (S = y, p = w, f = 0, h = S; h; h = tr(h)) f++;
              for (h = 0, j = p; j; j = tr(j)) h++;
              for (; 0 < f - h; ) S = tr(S), f--;
              for (; 0 < h - f; ) p = tr(p), h--;
              for (; f--; ) {
                if (S === p || p !== null && S === p.alternate) break t;
                S = tr(S), p = tr(p);
              }
              S = null;
            }
            else S = null;
            y !== null && Lu(d, g, y, S, false), w !== null && _ !== null && Lu(d, _, w, S, true);
          }
        }
        e: {
          if (g = c ? cr(c) : window, y = g.nodeName && g.nodeName.toLowerCase(), y === "select" || y === "input" && g.type === "file") var C = uh;
          else if (Cu(g)) if (Vd) C = mh;
          else {
            C = dh;
            var T = ch;
          }
          else (y = g.nodeName) && y.toLowerCase() === "input" && (g.type === "checkbox" || g.type === "radio") && (C = fh);
          if (C && (C = C(e, c))) {
            Bd(d, C, n, m);
            break e;
          }
          T && T(e, g, c), e === "focusout" && (T = g._wrapperState) && T.controlled && g.type === "number" && as(g, "number", g.value);
        }
        switch (T = c ? cr(c) : window, e) {
          case "focusin":
            (Cu(T) || T.contentEditable === "true") && (ir = T, vs = c, rl = null);
            break;
          case "focusout":
            rl = vs = ir = null;
            break;
          case "mousedown":
            ys = true;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            ys = false, bu(d, n, m);
            break;
          case "selectionchange":
            if (gh) break;
          case "keydown":
          case "keyup":
            bu(d, n, m);
        }
        var k;
        if (mi) e: {
          switch (e) {
            case "compositionstart":
              var R = "onCompositionStart";
              break e;
            case "compositionend":
              R = "onCompositionEnd";
              break e;
            case "compositionupdate":
              R = "onCompositionUpdate";
              break e;
          }
          R = void 0;
        }
        else sr ? $d(e, n) && (R = "onCompositionEnd") : e === "keydown" && n.keyCode === 229 && (R = "onCompositionStart");
        R && (Ud && n.locale !== "ko" && (sr || R !== "onCompositionStart" ? R === "onCompositionEnd" && sr && (k = Ad()) : (mn = m, ci = "value" in mn ? mn.value : mn.textContent, sr = true)), T = Ra(c, R), 0 < T.length && (R = new ku(R, e, null, n, m), d.push({
          event: R,
          listeners: T
        }), k ? R.data = k : (k = Fd(n), k !== null && (R.data = k)))), (k = lh ? ah(e, n) : oh(e, n)) && (c = Ra(c, "onBeforeInput"), 0 < c.length && (m = new ku("onBeforeInput", "beforeinput", null, n, m), d.push({
          event: m,
          listeners: c
        }), m.data = k));
      }
      qd(d, t);
    });
  }
  function yl(e, t, n) {
    return {
      instance: e,
      listener: t,
      currentTarget: n
    };
  }
  function Ra(e, t) {
    for (var n = t + "Capture", r = []; e !== null; ) {
      var l = e, a = l.stateNode;
      l.tag === 5 && a !== null && (l = a, a = dl(e, n), a != null && r.unshift(yl(e, a, l)), a = dl(e, t), a != null && r.push(yl(e, a, l))), e = e.return;
    }
    return r;
  }
  function tr(e) {
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
      i.tag === 5 && c !== null && (i = c, l ? (s = dl(n, a), s != null && o.unshift(yl(n, s, i))) : l || (s = dl(n, a), s != null && o.push(yl(n, s, i)))), n = n.return;
    }
    o.length !== 0 && e.push({
      event: t,
      listeners: o
    });
  }
  var wh = /\r\n?/g, Sh = /\u0000|\uFFFD/g;
  function Ou(e) {
    return (typeof e == "string" ? e : "" + e).replace(wh, `
`).replace(Sh, "");
  }
  function Zl(e, t, n) {
    if (t = Ou(t), Ou(e) !== t && n) throw Error(b(425));
  }
  function Ta() {
  }
  var xs = null, ws = null;
  function Ss(e, t) {
    return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
  }
  var ks = typeof setTimeout == "function" ? setTimeout : void 0, kh = typeof clearTimeout == "function" ? clearTimeout : void 0, Iu = typeof Promise == "function" ? Promise : void 0, Nh = typeof queueMicrotask == "function" ? queueMicrotask : typeof Iu < "u" ? function(e) {
    return Iu.resolve(null).then(e).catch(jh);
  } : ks;
  function jh(e) {
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
  function xn(e) {
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
  var Dr = Math.random().toString(36).slice(2), Vt = "__reactFiber$" + Dr, xl = "__reactProps$" + Dr, en = "__reactContainer$" + Dr, Ns = "__reactEvents$" + Dr, Eh = "__reactListeners$" + Dr, Ch = "__reactHandles$" + Dr;
  function zn(e) {
    var t = e[Vt];
    if (t) return t;
    for (var n = e.parentNode; n; ) {
      if (t = n[en] || n[Vt]) {
        if (n = t.alternate, t.child !== null || n !== null && n.child !== null) for (e = zu(e); e !== null; ) {
          if (n = e[Vt]) return n;
          e = zu(e);
        }
        return t;
      }
      e = n, n = e.parentNode;
    }
    return null;
  }
  function bl(e) {
    return e = e[Vt] || e[en], !e || e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3 ? null : e;
  }
  function cr(e) {
    if (e.tag === 5 || e.tag === 6) return e.stateNode;
    throw Error(b(33));
  }
  function eo(e) {
    return e[xl] || null;
  }
  var js = [], dr = -1;
  function Rn(e) {
    return {
      current: e
    };
  }
  function Ne(e) {
    0 > dr || (e.current = js[dr], js[dr] = null, dr--);
  }
  function Se(e, t) {
    dr++, js[dr] = e.current, e.current = t;
  }
  var Cn = {}, et = Rn(Cn), ct = Rn(false), Vn = Cn;
  function jr(e, t) {
    var n = e.type.contextTypes;
    if (!n) return Cn;
    var r = e.stateNode;
    if (r && r.__reactInternalMemoizedUnmaskedChildContext === t) return r.__reactInternalMemoizedMaskedChildContext;
    var l = {}, a;
    for (a in n) l[a] = t[a];
    return r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = t, e.__reactInternalMemoizedMaskedChildContext = l), l;
  }
  function dt(e) {
    return e = e.childContextTypes, e != null;
  }
  function Pa() {
    Ne(ct), Ne(et);
  }
  function Au(e, t, n) {
    if (et.current !== Cn) throw Error(b(168));
    Se(et, t), Se(ct, n);
  }
  function tf(e, t, n) {
    var r = e.stateNode;
    if (t = t.childContextTypes, typeof r.getChildContext != "function") return n;
    r = r.getChildContext();
    for (var l in r) if (!(l in t)) throw Error(b(108, cp(e) || "Unknown", l));
    return Pe({}, n, r);
  }
  function ba(e) {
    return e = (e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext || Cn, Vn = et.current, Se(et, e), Se(ct, ct.current), true;
  }
  function Uu(e, t, n) {
    var r = e.stateNode;
    if (!r) throw Error(b(169));
    n ? (e = tf(e, t, Vn), r.__reactInternalMemoizedMergedChildContext = e, Ne(ct), Ne(et), Se(et, e)) : Ne(ct), Se(ct, n);
  }
  var Yt = null, to = false, Io = false;
  function nf(e) {
    Yt === null ? Yt = [
      e
    ] : Yt.push(e);
  }
  function _h(e) {
    to = true, nf(e);
  }
  function Tn() {
    if (!Io && Yt !== null) {
      Io = true;
      var e = 0, t = ge;
      try {
        var n = Yt;
        for (ge = 1; e < n.length; e++) {
          var r = n[e];
          do
            r = r(true);
          while (r !== null);
        }
        Yt = null, to = false;
      } catch (l) {
        throw Yt !== null && (Yt = Yt.slice(e + 1)), _d(oi, Tn), l;
      } finally {
        ge = t, Io = false;
      }
    }
    return null;
  }
  var fr = [], mr = 0, Da = null, Ma = 0, Nt = [], jt = 0, Wn = null, Jt = 1, Xt = "";
  function Mn(e, t) {
    fr[mr++] = Ma, fr[mr++] = Da, Da = e, Ma = t;
  }
  function rf(e, t, n) {
    Nt[jt++] = Jt, Nt[jt++] = Xt, Nt[jt++] = Wn, Wn = e;
    var r = Jt;
    e = Xt;
    var l = 32 - It(r) - 1;
    r &= ~(1 << l), n += 1;
    var a = 32 - It(t) + l;
    if (30 < a) {
      var o = l - l % 5;
      a = (r & (1 << o) - 1).toString(32), r >>= o, l -= o, Jt = 1 << 32 - It(t) + l | n << l | r, Xt = a + e;
    } else Jt = 1 << a | n << l | r, Xt = e;
  }
  function hi(e) {
    e.return !== null && (Mn(e, 1), rf(e, 1, 0));
  }
  function gi(e) {
    for (; e === Da; ) Da = fr[--mr], fr[mr] = null, Ma = fr[--mr], fr[mr] = null;
    for (; e === Wn; ) Wn = Nt[--jt], Nt[jt] = null, Xt = Nt[--jt], Nt[jt] = null, Jt = Nt[--jt], Nt[jt] = null;
  }
  var xt = null, yt = null, Ce = false, Ot = null;
  function lf(e, t) {
    var n = Et(5, null, null, 0);
    n.elementType = "DELETED", n.stateNode = t, n.return = e, t = e.deletions, t === null ? (e.deletions = [
      n
    ], e.flags |= 16) : t.push(n);
  }
  function $u(e, t) {
    switch (e.tag) {
      case 5:
        var n = e.type;
        return t = t.nodeType !== 1 || n.toLowerCase() !== t.nodeName.toLowerCase() ? null : t, t !== null ? (e.stateNode = t, xt = e, yt = xn(t.firstChild), true) : false;
      case 6:
        return t = e.pendingProps === "" || t.nodeType !== 3 ? null : t, t !== null ? (e.stateNode = t, xt = e, yt = null, true) : false;
      case 13:
        return t = t.nodeType !== 8 ? null : t, t !== null ? (n = Wn !== null ? {
          id: Jt,
          overflow: Xt
        } : null, e.memoizedState = {
          dehydrated: t,
          treeContext: n,
          retryLane: 1073741824
        }, n = Et(18, null, null, 0), n.stateNode = t, n.return = e, e.child = n, xt = e, yt = null, true) : false;
      default:
        return false;
    }
  }
  function Es(e) {
    return (e.mode & 1) !== 0 && (e.flags & 128) === 0;
  }
  function Cs(e) {
    if (Ce) {
      var t = yt;
      if (t) {
        var n = t;
        if (!$u(e, t)) {
          if (Es(e)) throw Error(b(418));
          t = xn(n.nextSibling);
          var r = xt;
          t && $u(e, t) ? lf(r, n) : (e.flags = e.flags & -4097 | 2, Ce = false, xt = e);
        }
      } else {
        if (Es(e)) throw Error(b(418));
        e.flags = e.flags & -4097 | 2, Ce = false, xt = e;
      }
    }
  }
  function Fu(e) {
    for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13; ) e = e.return;
    xt = e;
  }
  function ql(e) {
    if (e !== xt) return false;
    if (!Ce) return Fu(e), Ce = true, false;
    var t;
    if ((t = e.tag !== 3) && !(t = e.tag !== 5) && (t = e.type, t = t !== "head" && t !== "body" && !Ss(e.type, e.memoizedProps)), t && (t = yt)) {
      if (Es(e)) throw af(), Error(b(418));
      for (; t; ) lf(e, t), t = xn(t.nextSibling);
    }
    if (Fu(e), e.tag === 13) {
      if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e) throw Error(b(317));
      e: {
        for (e = e.nextSibling, t = 0; e; ) {
          if (e.nodeType === 8) {
            var n = e.data;
            if (n === "/$") {
              if (t === 0) {
                yt = xn(e.nextSibling);
                break e;
              }
              t--;
            } else n !== "$" && n !== "$!" && n !== "$?" || t++;
          }
          e = e.nextSibling;
        }
        yt = null;
      }
    } else yt = xt ? xn(e.stateNode.nextSibling) : null;
    return true;
  }
  function af() {
    for (var e = yt; e; ) e = xn(e.nextSibling);
  }
  function Er() {
    yt = xt = null, Ce = false;
  }
  function vi(e) {
    Ot === null ? Ot = [
      e
    ] : Ot.push(e);
  }
  var Rh = rn.ReactCurrentBatchConfig;
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
      return p = Nn(p, f), p.index = 0, p.sibling = null, p;
    }
    function a(p, f, h) {
      return p.index = h, e ? (h = p.alternate, h !== null ? (h = h.index, h < f ? (p.flags |= 2, f) : h) : (p.flags |= 2, f)) : (p.flags |= 1048576, f);
    }
    function o(p) {
      return e && p.alternate === null && (p.flags |= 2), p;
    }
    function i(p, f, h, j) {
      return f === null || f.tag !== 6 ? (f = Vo(h, p.mode, j), f.return = p, f) : (f = l(f, h), f.return = p, f);
    }
    function s(p, f, h, j) {
      var C = h.type;
      return C === or ? m(p, f, h.props.children, j, h.key) : f !== null && (f.elementType === C || typeof C == "object" && C !== null && C.$$typeof === un && Bu(C) === f.type) ? (j = l(f, h.props), j.ref = Fr(p, f, h), j.return = p, j) : (j = ya(h.type, h.key, h.props, null, p.mode, j), j.ref = Fr(p, f, h), j.return = p, j);
    }
    function c(p, f, h, j) {
      return f === null || f.tag !== 4 || f.stateNode.containerInfo !== h.containerInfo || f.stateNode.implementation !== h.implementation ? (f = Wo(h, p.mode, j), f.return = p, f) : (f = l(f, h.children || []), f.return = p, f);
    }
    function m(p, f, h, j, C) {
      return f === null || f.tag !== 7 ? (f = Bn(h, p.mode, j, C), f.return = p, f) : (f = l(f, h), f.return = p, f);
    }
    function d(p, f, h) {
      if (typeof f == "string" && f !== "" || typeof f == "number") return f = Vo("" + f, p.mode, h), f.return = p, f;
      if (typeof f == "object" && f !== null) {
        switch (f.$$typeof) {
          case Vl:
            return h = ya(f.type, f.key, f.props, null, p.mode, h), h.ref = Fr(p, null, f), h.return = p, h;
          case ar:
            return f = Wo(f, p.mode, h), f.return = p, f;
          case un:
            var j = f._init;
            return d(p, j(f._payload), h);
        }
        if (Yr(f) || Ir(f)) return f = Bn(f, p.mode, h, null), f.return = p, f;
        ea(p, f);
      }
      return null;
    }
    function g(p, f, h, j) {
      var C = f !== null ? f.key : null;
      if (typeof h == "string" && h !== "" || typeof h == "number") return C !== null ? null : i(p, f, "" + h, j);
      if (typeof h == "object" && h !== null) {
        switch (h.$$typeof) {
          case Vl:
            return h.key === C ? s(p, f, h, j) : null;
          case ar:
            return h.key === C ? c(p, f, h, j) : null;
          case un:
            return C = h._init, g(p, f, C(h._payload), j);
        }
        if (Yr(h) || Ir(h)) return C !== null ? null : m(p, f, h, j, null);
        ea(p, h);
      }
      return null;
    }
    function y(p, f, h, j, C) {
      if (typeof j == "string" && j !== "" || typeof j == "number") return p = p.get(h) || null, i(f, p, "" + j, C);
      if (typeof j == "object" && j !== null) {
        switch (j.$$typeof) {
          case Vl:
            return p = p.get(j.key === null ? h : j.key) || null, s(f, p, j, C);
          case ar:
            return p = p.get(j.key === null ? h : j.key) || null, c(f, p, j, C);
          case un:
            var T = j._init;
            return y(p, f, h, T(j._payload), C);
        }
        if (Yr(j) || Ir(j)) return p = p.get(h) || null, m(f, p, j, C, null);
        ea(f, j);
      }
      return null;
    }
    function w(p, f, h, j) {
      for (var C = null, T = null, k = f, R = f = 0, $ = null; k !== null && R < h.length; R++) {
        k.index > R ? ($ = k, k = null) : $ = k.sibling;
        var M = g(p, k, h[R], j);
        if (M === null) {
          k === null && (k = $);
          break;
        }
        e && k && M.alternate === null && t(p, k), f = a(M, f, R), T === null ? C = M : T.sibling = M, T = M, k = $;
      }
      if (R === h.length) return n(p, k), Ce && Mn(p, R), C;
      if (k === null) {
        for (; R < h.length; R++) k = d(p, h[R], j), k !== null && (f = a(k, f, R), T === null ? C = k : T.sibling = k, T = k);
        return Ce && Mn(p, R), C;
      }
      for (k = r(p, k); R < h.length; R++) $ = y(k, p, R, h[R], j), $ !== null && (e && $.alternate !== null && k.delete($.key === null ? R : $.key), f = a($, f, R), T === null ? C = $ : T.sibling = $, T = $);
      return e && k.forEach(function(K) {
        return t(p, K);
      }), Ce && Mn(p, R), C;
    }
    function S(p, f, h, j) {
      var C = Ir(h);
      if (typeof C != "function") throw Error(b(150));
      if (h = C.call(h), h == null) throw Error(b(151));
      for (var T = C = null, k = f, R = f = 0, $ = null, M = h.next(); k !== null && !M.done; R++, M = h.next()) {
        k.index > R ? ($ = k, k = null) : $ = k.sibling;
        var K = g(p, k, M.value, j);
        if (K === null) {
          k === null && (k = $);
          break;
        }
        e && k && K.alternate === null && t(p, k), f = a(K, f, R), T === null ? C = K : T.sibling = K, T = K, k = $;
      }
      if (M.done) return n(p, k), Ce && Mn(p, R), C;
      if (k === null) {
        for (; !M.done; R++, M = h.next()) M = d(p, M.value, j), M !== null && (f = a(M, f, R), T === null ? C = M : T.sibling = M, T = M);
        return Ce && Mn(p, R), C;
      }
      for (k = r(p, k); !M.done; R++, M = h.next()) M = y(k, p, R, M.value, j), M !== null && (e && M.alternate !== null && k.delete(M.key === null ? R : M.key), f = a(M, f, R), T === null ? C = M : T.sibling = M, T = M);
      return e && k.forEach(function(J) {
        return t(p, J);
      }), Ce && Mn(p, R), C;
    }
    function _(p, f, h, j) {
      if (typeof h == "object" && h !== null && h.type === or && h.key === null && (h = h.props.children), typeof h == "object" && h !== null) {
        switch (h.$$typeof) {
          case Vl:
            e: {
              for (var C = h.key, T = f; T !== null; ) {
                if (T.key === C) {
                  if (C = h.type, C === or) {
                    if (T.tag === 7) {
                      n(p, T.sibling), f = l(T, h.props.children), f.return = p, p = f;
                      break e;
                    }
                  } else if (T.elementType === C || typeof C == "object" && C !== null && C.$$typeof === un && Bu(C) === T.type) {
                    n(p, T.sibling), f = l(T, h.props), f.ref = Fr(p, T, h), f.return = p, p = f;
                    break e;
                  }
                  n(p, T);
                  break;
                } else t(p, T);
                T = T.sibling;
              }
              h.type === or ? (f = Bn(h.props.children, p.mode, j, h.key), f.return = p, p = f) : (j = ya(h.type, h.key, h.props, null, p.mode, j), j.ref = Fr(p, f, h), j.return = p, p = j);
            }
            return o(p);
          case ar:
            e: {
              for (T = h.key; f !== null; ) {
                if (f.key === T) if (f.tag === 4 && f.stateNode.containerInfo === h.containerInfo && f.stateNode.implementation === h.implementation) {
                  n(p, f.sibling), f = l(f, h.children || []), f.return = p, p = f;
                  break e;
                } else {
                  n(p, f);
                  break;
                }
                else t(p, f);
                f = f.sibling;
              }
              f = Wo(h, p.mode, j), f.return = p, p = f;
            }
            return o(p);
          case un:
            return T = h._init, _(p, f, T(h._payload), j);
        }
        if (Yr(h)) return w(p, f, h, j);
        if (Ir(h)) return S(p, f, h, j);
        ea(p, h);
      }
      return typeof h == "string" && h !== "" || typeof h == "number" ? (h = "" + h, f !== null && f.tag === 6 ? (n(p, f.sibling), f = l(f, h), f.return = p, p = f) : (n(p, f), f = Vo(h, p.mode, j), f.return = p, p = f), o(p)) : n(p, f);
    }
    return _;
  }
  var Cr = of(true), sf = of(false), La = Rn(null), Oa = null, pr = null, yi = null;
  function xi() {
    yi = pr = Oa = null;
  }
  function wi(e) {
    var t = La.current;
    Ne(La), e._currentValue = t;
  }
  function _s(e, t, n) {
    for (; e !== null; ) {
      var r = e.alternate;
      if ((e.childLanes & t) !== t ? (e.childLanes |= t, r !== null && (r.childLanes |= t)) : r !== null && (r.childLanes & t) !== t && (r.childLanes |= t), e === n) break;
      e = e.return;
    }
  }
  function Sr(e, t) {
    Oa = e, yi = pr = null, e = e.dependencies, e !== null && e.firstContext !== null && (e.lanes & t && (ut = true), e.firstContext = null);
  }
  function _t(e) {
    var t = e._currentValue;
    if (yi !== e) if (e = {
      context: e,
      memoizedValue: t,
      next: null
    }, pr === null) {
      if (Oa === null) throw Error(b(308));
      pr = e, Oa.dependencies = {
        lanes: 0,
        firstContext: e
      };
    } else pr = pr.next = e;
    return t;
  }
  var An = null;
  function Si(e) {
    An === null ? An = [
      e
    ] : An.push(e);
  }
  function uf(e, t, n, r) {
    var l = t.interleaved;
    return l === null ? (n.next = n, Si(t)) : (n.next = l.next, l.next = n), t.interleaved = n, tn(e, r);
  }
  function tn(e, t) {
    e.lanes |= t;
    var n = e.alternate;
    for (n !== null && (n.lanes |= t), n = e, e = e.return; e !== null; ) e.childLanes |= t, n = e.alternate, n !== null && (n.childLanes |= t), n = e, e = e.return;
    return n.tag === 3 ? n.stateNode : null;
  }
  var cn = false;
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
  function Zt(e, t) {
    return {
      eventTime: e,
      lane: t,
      tag: 0,
      payload: null,
      callback: null,
      next: null
    };
  }
  function wn(e, t, n) {
    var r = e.updateQueue;
    if (r === null) return null;
    if (r = r.shared, ie & 2) {
      var l = r.pending;
      return l === null ? t.next = t : (t.next = l.next, l.next = t), r.pending = t, tn(e, n);
    }
    return l = r.interleaved, l === null ? (t.next = t, Si(r)) : (t.next = l.next, l.next = t), r.interleaved = t, tn(e, n);
  }
  function fa(e, t, n) {
    if (t = t.updateQueue, t !== null && (t = t.shared, (n & 4194240) !== 0)) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, si(e, n);
    }
  }
  function Vu(e, t) {
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
    cn = false;
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
        var g = i.lane, y = i.eventTime;
        if ((r & g) === g) {
          m !== null && (m = m.next = {
            eventTime: y,
            lane: 0,
            tag: i.tag,
            payload: i.payload,
            callback: i.callback,
            next: null
          });
          e: {
            var w = e, S = i;
            switch (g = t, y = n, S.tag) {
              case 1:
                if (w = S.payload, typeof w == "function") {
                  d = w.call(y, d, g);
                  break e;
                }
                d = w;
                break e;
              case 3:
                w.flags = w.flags & -65537 | 128;
              case 0:
                if (w = S.payload, g = typeof w == "function" ? w.call(y, d, g) : w, g == null) break e;
                d = Pe({}, d, g);
                break e;
              case 2:
                cn = true;
            }
          }
          i.callback !== null && i.lane !== 0 && (e.flags |= 64, g = l.effects, g === null ? l.effects = [
            i
          ] : g.push(i));
        } else y = {
          eventTime: y,
          lane: g,
          tag: i.tag,
          payload: i.payload,
          callback: i.callback,
          next: null
        }, m === null ? (c = m = y, s = d) : m = m.next = y, o |= g;
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
      Qn |= o, e.lanes = o, e.memoizedState = d;
    }
  }
  function Wu(e, t, n) {
    if (e = t.effects, t.effects = null, e !== null) for (t = 0; t < e.length; t++) {
      var r = e[t], l = r.callback;
      if (l !== null) {
        if (r.callback = null, r = n, typeof l != "function") throw Error(b(191, l));
        l.call(r);
      }
    }
  }
  var Dl = {}, Ht = Rn(Dl), wl = Rn(Dl), Sl = Rn(Dl);
  function Un(e) {
    if (e === Dl) throw Error(b(174));
    return e;
  }
  function Ni(e, t) {
    switch (Se(Sl, t), Se(wl, e), Se(Ht, Dl), e = t.nodeType, e) {
      case 9:
      case 11:
        t = (t = t.documentElement) ? t.namespaceURI : ss(null, "");
        break;
      default:
        e = e === 8 ? t.parentNode : t, t = e.namespaceURI || null, e = e.tagName, t = ss(t, e);
    }
    Ne(Ht), Se(Ht, t);
  }
  function _r() {
    Ne(Ht), Ne(wl), Ne(Sl);
  }
  function df(e) {
    Un(Sl.current);
    var t = Un(Ht.current), n = ss(t, e.type);
    t !== n && (Se(wl, e), Se(Ht, n));
  }
  function ji(e) {
    wl.current === e && (Ne(Ht), Ne(wl));
  }
  var Re = Rn(0);
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
  var ma = rn.ReactCurrentDispatcher, Ao = rn.ReactCurrentBatchConfig, Hn = 0, Te = null, Fe = null, Ve = null, Aa = false, ll = false, kl = 0, Th = 0;
  function Xe() {
    throw Error(b(321));
  }
  function Ci(e, t) {
    if (t === null) return false;
    for (var n = 0; n < t.length && n < e.length; n++) if (!At(e[n], t[n])) return false;
    return true;
  }
  function _i(e, t, n, r, l, a) {
    if (Hn = a, Te = t, t.memoizedState = null, t.updateQueue = null, t.lanes = 0, ma.current = e === null || e.memoizedState === null ? Mh : Lh, e = n(r, l), ll) {
      a = 0;
      do {
        if (ll = false, kl = 0, 25 <= a) throw Error(b(301));
        a += 1, Ve = Fe = null, t.updateQueue = null, ma.current = Oh, e = n(r, l);
      } while (ll);
    }
    if (ma.current = Ua, t = Fe !== null && Fe.next !== null, Hn = 0, Ve = Fe = Te = null, Aa = false, t) throw Error(b(300));
    return e;
  }
  function Ri() {
    var e = kl !== 0;
    return kl = 0, e;
  }
  function Bt() {
    var e = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return Ve === null ? Te.memoizedState = Ve = e : Ve = Ve.next = e, Ve;
  }
  function Rt() {
    if (Fe === null) {
      var e = Te.alternate;
      e = e !== null ? e.memoizedState : null;
    } else e = Fe.next;
    var t = Ve === null ? Te.memoizedState : Ve.next;
    if (t !== null) Ve = t, Fe = e;
    else {
      if (e === null) throw Error(b(310));
      Fe = e, e = {
        memoizedState: Fe.memoizedState,
        baseState: Fe.baseState,
        baseQueue: Fe.baseQueue,
        queue: Fe.queue,
        next: null
      }, Ve === null ? Te.memoizedState = Ve = e : Ve = Ve.next = e;
    }
    return Ve;
  }
  function Nl(e, t) {
    return typeof t == "function" ? t(e) : t;
  }
  function Uo(e) {
    var t = Rt(), n = t.queue;
    if (n === null) throw Error(b(311));
    n.lastRenderedReducer = e;
    var r = Fe, l = r.baseQueue, a = n.pending;
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
        if ((Hn & m) === m) s !== null && (s = s.next = {
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
          s === null ? (i = s = d, o = r) : s = s.next = d, Te.lanes |= m, Qn |= m;
        }
        c = c.next;
      } while (c !== null && c !== a);
      s === null ? o = r : s.next = i, At(r, t.memoizedState) || (ut = true), t.memoizedState = r, t.baseState = o, t.baseQueue = s, n.lastRenderedState = r;
    }
    if (e = n.interleaved, e !== null) {
      l = e;
      do
        a = l.lane, Te.lanes |= a, Qn |= a, l = l.next;
      while (l !== e);
    } else l === null && (n.lanes = 0);
    return [
      t.memoizedState,
      n.dispatch
    ];
  }
  function $o(e) {
    var t = Rt(), n = t.queue;
    if (n === null) throw Error(b(311));
    n.lastRenderedReducer = e;
    var r = n.dispatch, l = n.pending, a = t.memoizedState;
    if (l !== null) {
      n.pending = null;
      var o = l = l.next;
      do
        a = e(a, o.action), o = o.next;
      while (o !== l);
      At(a, t.memoizedState) || (ut = true), t.memoizedState = a, t.baseQueue === null && (t.baseState = a), n.lastRenderedState = a;
    }
    return [
      a,
      r
    ];
  }
  function ff() {
  }
  function mf(e, t) {
    var n = Te, r = Rt(), l = t(), a = !At(r.memoizedState, l);
    if (a && (r.memoizedState = l, ut = true), r = r.queue, Ti(gf.bind(null, n, r, e), [
      e
    ]), r.getSnapshot !== t || a || Ve !== null && Ve.memoizedState.tag & 1) {
      if (n.flags |= 2048, jl(9, hf.bind(null, n, r, l, t), void 0, null), We === null) throw Error(b(349));
      Hn & 30 || pf(n, t, l);
    }
    return l;
  }
  function pf(e, t, n) {
    e.flags |= 16384, e = {
      getSnapshot: t,
      value: n
    }, t = Te.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, Te.updateQueue = t, t.stores = [
      e
    ]) : (n = t.stores, n === null ? t.stores = [
      e
    ] : n.push(e));
  }
  function hf(e, t, n, r) {
    t.value = n, t.getSnapshot = r, vf(t) && yf(e);
  }
  function gf(e, t, n) {
    return n(function() {
      vf(t) && yf(e);
    });
  }
  function vf(e) {
    var t = e.getSnapshot;
    e = e.value;
    try {
      var n = t();
      return !At(e, n);
    } catch {
      return true;
    }
  }
  function yf(e) {
    var t = tn(e, 1);
    t !== null && zt(t, e, 1, -1);
  }
  function Hu(e) {
    var t = Bt();
    return typeof e == "function" && (e = e()), t.memoizedState = t.baseState = e, e = {
      pending: null,
      interleaved: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Nl,
      lastRenderedState: e
    }, t.queue = e, e = e.dispatch = Dh.bind(null, Te, e), [
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
    }, t = Te.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, Te.updateQueue = t, t.lastEffect = e.next = e) : (n = t.lastEffect, n === null ? t.lastEffect = e.next = e : (r = n.next, n.next = e, e.next = r, t.lastEffect = e)), e;
  }
  function xf() {
    return Rt().memoizedState;
  }
  function pa(e, t, n, r) {
    var l = Bt();
    Te.flags |= e, l.memoizedState = jl(1 | t, n, void 0, r === void 0 ? null : r);
  }
  function no(e, t, n, r) {
    var l = Rt();
    r = r === void 0 ? null : r;
    var a = void 0;
    if (Fe !== null) {
      var o = Fe.memoizedState;
      if (a = o.destroy, r !== null && Ci(r, o.deps)) {
        l.memoizedState = jl(t, n, a, r);
        return;
      }
    }
    Te.flags |= e, l.memoizedState = jl(1 | t, n, a, r);
  }
  function Qu(e, t) {
    return pa(8390656, 8, e, t);
  }
  function Ti(e, t) {
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
  function Pi() {
  }
  function jf(e, t) {
    var n = Rt();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Ci(t, r[1]) ? r[0] : (n.memoizedState = [
      e,
      t
    ], e);
  }
  function Ef(e, t) {
    var n = Rt();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Ci(t, r[1]) ? r[0] : (e = e(), n.memoizedState = [
      e,
      t
    ], e);
  }
  function Cf(e, t, n) {
    return Hn & 21 ? (At(n, t) || (n = Pd(), Te.lanes |= n, Qn |= n, e.baseState = true), t) : (e.baseState && (e.baseState = false, ut = true), e.memoizedState = n);
  }
  function Ph(e, t) {
    var n = ge;
    ge = n !== 0 && 4 > n ? n : 4, e(true);
    var r = Ao.transition;
    Ao.transition = {};
    try {
      e(false), t();
    } finally {
      ge = n, Ao.transition = r;
    }
  }
  function _f() {
    return Rt().memoizedState;
  }
  function bh(e, t, n) {
    var r = kn(e);
    if (n = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    }, Rf(e)) Tf(t, n);
    else if (n = uf(e, t, n, r), n !== null) {
      var l = rt();
      zt(n, e, r, l), Pf(n, t, r);
    }
  }
  function Dh(e, t, n) {
    var r = kn(e), l = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    };
    if (Rf(e)) Tf(t, l);
    else {
      var a = e.alternate;
      if (e.lanes === 0 && (a === null || a.lanes === 0) && (a = t.lastRenderedReducer, a !== null)) try {
        var o = t.lastRenderedState, i = a(o, n);
        if (l.hasEagerState = true, l.eagerState = i, At(i, o)) {
          var s = t.interleaved;
          s === null ? (l.next = l, Si(t)) : (l.next = s.next, s.next = l), t.interleaved = l;
          return;
        }
      } catch {
      } finally {
      }
      n = uf(e, t, l, r), n !== null && (l = rt(), zt(n, e, r, l), Pf(n, t, r));
    }
  }
  function Rf(e) {
    var t = e.alternate;
    return e === Te || t !== null && t === Te;
  }
  function Tf(e, t) {
    ll = Aa = true;
    var n = e.pending;
    n === null ? t.next = t : (t.next = n.next, n.next = t), e.pending = t;
  }
  function Pf(e, t, n) {
    if (n & 4194240) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, si(e, n);
    }
  }
  var Ua = {
    readContext: _t,
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
  }, Mh = {
    readContext: _t,
    useCallback: function(e, t) {
      return Bt().memoizedState = [
        e,
        t === void 0 ? null : t
      ], e;
    },
    useContext: _t,
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
      var n = Bt();
      return t = t === void 0 ? null : t, e = e(), n.memoizedState = [
        e,
        t
      ], e;
    },
    useReducer: function(e, t, n) {
      var r = Bt();
      return t = n !== void 0 ? n(t) : t, r.memoizedState = r.baseState = t, e = {
        pending: null,
        interleaved: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: e,
        lastRenderedState: t
      }, r.queue = e, e = e.dispatch = bh.bind(null, Te, e), [
        r.memoizedState,
        e
      ];
    },
    useRef: function(e) {
      var t = Bt();
      return e = {
        current: e
      }, t.memoizedState = e;
    },
    useState: Hu,
    useDebugValue: Pi,
    useDeferredValue: function(e) {
      return Bt().memoizedState = e;
    },
    useTransition: function() {
      var e = Hu(false), t = e[0];
      return e = Ph.bind(null, e[1]), Bt().memoizedState = e, [
        t,
        e
      ];
    },
    useMutableSource: function() {
    },
    useSyncExternalStore: function(e, t, n) {
      var r = Te, l = Bt();
      if (Ce) {
        if (n === void 0) throw Error(b(407));
        n = n();
      } else {
        if (n = t(), We === null) throw Error(b(349));
        Hn & 30 || pf(r, t, n);
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
      var e = Bt(), t = We.identifierPrefix;
      if (Ce) {
        var n = Xt, r = Jt;
        n = (r & ~(1 << 32 - It(r) - 1)).toString(32) + n, t = ":" + t + "R" + n, n = kl++, 0 < n && (t += "H" + n.toString(32)), t += ":";
      } else n = Th++, t = ":" + t + "r" + n.toString(32) + ":";
      return e.memoizedState = t;
    },
    unstable_isNewReconciler: false
  }, Lh = {
    readContext: _t,
    useCallback: jf,
    useContext: _t,
    useEffect: Ti,
    useImperativeHandle: Nf,
    useInsertionEffect: wf,
    useLayoutEffect: Sf,
    useMemo: Ef,
    useReducer: Uo,
    useRef: xf,
    useState: function() {
      return Uo(Nl);
    },
    useDebugValue: Pi,
    useDeferredValue: function(e) {
      var t = Rt();
      return Cf(t, Fe.memoizedState, e);
    },
    useTransition: function() {
      var e = Uo(Nl)[0], t = Rt().memoizedState;
      return [
        e,
        t
      ];
    },
    useMutableSource: ff,
    useSyncExternalStore: mf,
    useId: _f,
    unstable_isNewReconciler: false
  }, Oh = {
    readContext: _t,
    useCallback: jf,
    useContext: _t,
    useEffect: Ti,
    useImperativeHandle: Nf,
    useInsertionEffect: wf,
    useLayoutEffect: Sf,
    useMemo: Ef,
    useReducer: $o,
    useRef: xf,
    useState: function() {
      return $o(Nl);
    },
    useDebugValue: Pi,
    useDeferredValue: function(e) {
      var t = Rt();
      return Fe === null ? t.memoizedState = e : Cf(t, Fe.memoizedState, e);
    },
    useTransition: function() {
      var e = $o(Nl)[0], t = Rt().memoizedState;
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
  function Dt(e, t) {
    if (e && e.defaultProps) {
      t = Pe({}, t), e = e.defaultProps;
      for (var n in e) t[n] === void 0 && (t[n] = e[n]);
      return t;
    }
    return t;
  }
  function Rs(e, t, n, r) {
    t = e.memoizedState, n = n(r, t), n = n == null ? t : Pe({}, t, n), e.memoizedState = n, e.lanes === 0 && (e.updateQueue.baseState = n);
  }
  var ro = {
    isMounted: function(e) {
      return (e = e._reactInternals) ? Jn(e) === e : false;
    },
    enqueueSetState: function(e, t, n) {
      e = e._reactInternals;
      var r = rt(), l = kn(e), a = Zt(r, l);
      a.payload = t, n != null && (a.callback = n), t = wn(e, a, l), t !== null && (zt(t, e, l, r), fa(t, e, l));
    },
    enqueueReplaceState: function(e, t, n) {
      e = e._reactInternals;
      var r = rt(), l = kn(e), a = Zt(r, l);
      a.tag = 1, a.payload = t, n != null && (a.callback = n), t = wn(e, a, l), t !== null && (zt(t, e, l, r), fa(t, e, l));
    },
    enqueueForceUpdate: function(e, t) {
      e = e._reactInternals;
      var n = rt(), r = kn(e), l = Zt(n, r);
      l.tag = 2, t != null && (l.callback = t), t = wn(e, l, r), t !== null && (zt(t, e, r, n), fa(t, e, r));
    }
  };
  function Gu(e, t, n, r, l, a, o) {
    return e = e.stateNode, typeof e.shouldComponentUpdate == "function" ? e.shouldComponentUpdate(r, a, o) : t.prototype && t.prototype.isPureReactComponent ? !gl(n, r) || !gl(l, a) : true;
  }
  function bf(e, t, n) {
    var r = false, l = Cn, a = t.contextType;
    return typeof a == "object" && a !== null ? a = _t(a) : (l = dt(t) ? Vn : et.current, r = t.contextTypes, a = (r = r != null) ? jr(e, l) : Cn), t = new t(n, a), e.memoizedState = t.state !== null && t.state !== void 0 ? t.state : null, t.updater = ro, e.stateNode = t, t._reactInternals = e, r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = l, e.__reactInternalMemoizedMaskedChildContext = a), t;
  }
  function Ku(e, t, n, r) {
    e = t.state, typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(n, r), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(n, r), t.state !== e && ro.enqueueReplaceState(t, t.state, null);
  }
  function Ts(e, t, n, r) {
    var l = e.stateNode;
    l.props = n, l.state = e.memoizedState, l.refs = {}, ki(e);
    var a = t.contextType;
    typeof a == "object" && a !== null ? l.context = _t(a) : (a = dt(t) ? Vn : et.current, l.context = jr(e, a)), l.state = e.memoizedState, a = t.getDerivedStateFromProps, typeof a == "function" && (Rs(e, t, a, n), l.state = e.memoizedState), typeof t.getDerivedStateFromProps == "function" || typeof l.getSnapshotBeforeUpdate == "function" || typeof l.UNSAFE_componentWillMount != "function" && typeof l.componentWillMount != "function" || (t = l.state, typeof l.componentWillMount == "function" && l.componentWillMount(), typeof l.UNSAFE_componentWillMount == "function" && l.UNSAFE_componentWillMount(), t !== l.state && ro.enqueueReplaceState(l, l.state, null), Ia(e, n, l, r), l.state = e.memoizedState), typeof l.componentDidMount == "function" && (e.flags |= 4194308);
  }
  function Rr(e, t) {
    try {
      var n = "", r = t;
      do
        n += up(r), r = r.return;
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
  function Ps(e, t) {
    try {
      console.error(t.value);
    } catch (n) {
      setTimeout(function() {
        throw n;
      });
    }
  }
  var Ih = typeof WeakMap == "function" ? WeakMap : Map;
  function Df(e, t, n) {
    n = Zt(-1, n), n.tag = 3, n.payload = {
      element: null
    };
    var r = t.value;
    return n.callback = function() {
      Fa || (Fa = true, $s = r), Ps(e, t);
    }, n;
  }
  function Mf(e, t, n) {
    n = Zt(-1, n), n.tag = 3;
    var r = e.type.getDerivedStateFromError;
    if (typeof r == "function") {
      var l = t.value;
      n.payload = function() {
        return r(l);
      }, n.callback = function() {
        Ps(e, t);
      };
    }
    var a = e.stateNode;
    return a !== null && typeof a.componentDidCatch == "function" && (n.callback = function() {
      Ps(e, t), typeof r != "function" && (Sn === null ? Sn = /* @__PURE__ */ new Set([
        this
      ]) : Sn.add(this));
      var o = t.stack;
      this.componentDidCatch(t.value, {
        componentStack: o !== null ? o : ""
      });
    }), n;
  }
  function Yu(e, t, n) {
    var r = e.pingCache;
    if (r === null) {
      r = e.pingCache = new Ih();
      var l = /* @__PURE__ */ new Set();
      r.set(t, l);
    } else l = r.get(t), l === void 0 && (l = /* @__PURE__ */ new Set(), r.set(t, l));
    l.has(n) || (l.add(n), e = Jh.bind(null, e, t, n), t.then(e, e));
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
    return e.mode & 1 ? (e.flags |= 65536, e.lanes = l, e) : (e === t ? e.flags |= 65536 : (e.flags |= 128, n.flags |= 131072, n.flags &= -52805, n.tag === 1 && (n.alternate === null ? n.tag = 17 : (t = Zt(-1, 1), t.tag = 2, wn(n, t, 1))), n.lanes |= 1), e);
  }
  var zh = rn.ReactCurrentOwner, ut = false;
  function nt(e, t, n, r) {
    t.child = e === null ? sf(t, null, n, r) : Cr(t, e.child, n, r);
  }
  function Zu(e, t, n, r, l) {
    n = n.render;
    var a = t.ref;
    return Sr(t, l), r = _i(e, t, n, r, a, l), n = Ri(), e !== null && !ut ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, nn(e, t, l)) : (Ce && n && hi(t), t.flags |= 1, nt(e, t, r, l), t.child);
  }
  function qu(e, t, n, r, l) {
    if (e === null) {
      var a = n.type;
      return typeof a == "function" && !Ai(a) && a.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0 ? (t.tag = 15, t.type = a, Lf(e, t, a, r, l)) : (e = ya(n.type, null, r, t, t.mode, l), e.ref = t.ref, e.return = t, t.child = e);
    }
    if (a = e.child, !(e.lanes & l)) {
      var o = a.memoizedProps;
      if (n = n.compare, n = n !== null ? n : gl, n(o, r) && e.ref === t.ref) return nn(e, t, l);
    }
    return t.flags |= 1, e = Nn(a, r), e.ref = t.ref, e.return = t, t.child = e;
  }
  function Lf(e, t, n, r, l) {
    if (e !== null) {
      var a = e.memoizedProps;
      if (gl(a, r) && e.ref === t.ref) if (ut = false, t.pendingProps = r = a, (e.lanes & l) !== 0) e.flags & 131072 && (ut = true);
      else return t.lanes = e.lanes, nn(e, t, l);
    }
    return bs(e, t, n, r, l);
  }
  function Of(e, t, n) {
    var r = t.pendingProps, l = r.children, a = e !== null ? e.memoizedState : null;
    if (r.mode === "hidden") if (!(t.mode & 1)) t.memoizedState = {
      baseLanes: 0,
      cachePool: null,
      transitions: null
    }, Se(gr, gt), gt |= n;
    else {
      if (!(n & 1073741824)) return e = a !== null ? a.baseLanes | n : n, t.lanes = t.childLanes = 1073741824, t.memoizedState = {
        baseLanes: e,
        cachePool: null,
        transitions: null
      }, t.updateQueue = null, Se(gr, gt), gt |= e, null;
      t.memoizedState = {
        baseLanes: 0,
        cachePool: null,
        transitions: null
      }, r = a !== null ? a.baseLanes : n, Se(gr, gt), gt |= r;
    }
    else a !== null ? (r = a.baseLanes | n, t.memoizedState = null) : r = n, Se(gr, gt), gt |= r;
    return nt(e, t, l, n), t.child;
  }
  function If(e, t) {
    var n = t.ref;
    (e === null && n !== null || e !== null && e.ref !== n) && (t.flags |= 512, t.flags |= 2097152);
  }
  function bs(e, t, n, r, l) {
    var a = dt(n) ? Vn : et.current;
    return a = jr(t, a), Sr(t, l), n = _i(e, t, n, r, a, l), r = Ri(), e !== null && !ut ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, nn(e, t, l)) : (Ce && r && hi(t), t.flags |= 1, nt(e, t, n, l), t.child);
  }
  function ec(e, t, n, r, l) {
    if (dt(n)) {
      var a = true;
      ba(t);
    } else a = false;
    if (Sr(t, l), t.stateNode === null) ha(e, t), bf(t, n, r), Ts(t, n, r, l), r = true;
    else if (e === null) {
      var o = t.stateNode, i = t.memoizedProps;
      o.props = i;
      var s = o.context, c = n.contextType;
      typeof c == "object" && c !== null ? c = _t(c) : (c = dt(n) ? Vn : et.current, c = jr(t, c));
      var m = n.getDerivedStateFromProps, d = typeof m == "function" || typeof o.getSnapshotBeforeUpdate == "function";
      d || typeof o.UNSAFE_componentWillReceiveProps != "function" && typeof o.componentWillReceiveProps != "function" || (i !== r || s !== c) && Ku(t, o, r, c), cn = false;
      var g = t.memoizedState;
      o.state = g, Ia(t, r, o, l), s = t.memoizedState, i !== r || g !== s || ct.current || cn ? (typeof m == "function" && (Rs(t, n, m, r), s = t.memoizedState), (i = cn || Gu(t, n, i, r, g, s, c)) ? (d || typeof o.UNSAFE_componentWillMount != "function" && typeof o.componentWillMount != "function" || (typeof o.componentWillMount == "function" && o.componentWillMount(), typeof o.UNSAFE_componentWillMount == "function" && o.UNSAFE_componentWillMount()), typeof o.componentDidMount == "function" && (t.flags |= 4194308)) : (typeof o.componentDidMount == "function" && (t.flags |= 4194308), t.memoizedProps = r, t.memoizedState = s), o.props = r, o.state = s, o.context = c, r = i) : (typeof o.componentDidMount == "function" && (t.flags |= 4194308), r = false);
    } else {
      o = t.stateNode, cf(e, t), i = t.memoizedProps, c = t.type === t.elementType ? i : Dt(t.type, i), o.props = c, d = t.pendingProps, g = o.context, s = n.contextType, typeof s == "object" && s !== null ? s = _t(s) : (s = dt(n) ? Vn : et.current, s = jr(t, s));
      var y = n.getDerivedStateFromProps;
      (m = typeof y == "function" || typeof o.getSnapshotBeforeUpdate == "function") || typeof o.UNSAFE_componentWillReceiveProps != "function" && typeof o.componentWillReceiveProps != "function" || (i !== d || g !== s) && Ku(t, o, r, s), cn = false, g = t.memoizedState, o.state = g, Ia(t, r, o, l);
      var w = t.memoizedState;
      i !== d || g !== w || ct.current || cn ? (typeof y == "function" && (Rs(t, n, y, r), w = t.memoizedState), (c = cn || Gu(t, n, c, r, g, w, s) || false) ? (m || typeof o.UNSAFE_componentWillUpdate != "function" && typeof o.componentWillUpdate != "function" || (typeof o.componentWillUpdate == "function" && o.componentWillUpdate(r, w, s), typeof o.UNSAFE_componentWillUpdate == "function" && o.UNSAFE_componentWillUpdate(r, w, s)), typeof o.componentDidUpdate == "function" && (t.flags |= 4), typeof o.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof o.componentDidUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 4), typeof o.getSnapshotBeforeUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 1024), t.memoizedProps = r, t.memoizedState = w), o.props = r, o.state = w, o.context = s, r = c) : (typeof o.componentDidUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 4), typeof o.getSnapshotBeforeUpdate != "function" || i === e.memoizedProps && g === e.memoizedState || (t.flags |= 1024), r = false);
    }
    return Ds(e, t, n, r, a, l);
  }
  function Ds(e, t, n, r, l, a) {
    If(e, t);
    var o = (t.flags & 128) !== 0;
    if (!r && !o) return l && Uu(t, n, false), nn(e, t, a);
    r = t.stateNode, zh.current = t;
    var i = o && typeof n.getDerivedStateFromError != "function" ? null : r.render();
    return t.flags |= 1, e !== null && o ? (t.child = Cr(t, e.child, null, a), t.child = Cr(t, null, i, a)) : nt(e, t, i, a), t.memoizedState = r.state, l && Uu(t, n, true), t.child;
  }
  function zf(e) {
    var t = e.stateNode;
    t.pendingContext ? Au(e, t.pendingContext, t.pendingContext !== t.context) : t.context && Au(e, t.context, false), Ni(e, t.containerInfo);
  }
  function tc(e, t, n, r, l) {
    return Er(), vi(l), t.flags |= 256, nt(e, t, n, r), t.child;
  }
  var Ms = {
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
    var r = t.pendingProps, l = Re.current, a = false, o = (t.flags & 128) !== 0, i;
    if ((i = o) || (i = e !== null && e.memoizedState === null ? false : (l & 2) !== 0), i ? (a = true, t.flags &= -129) : (e === null || e.memoizedState !== null) && (l |= 1), Se(Re, l & 1), e === null) return Cs(t), e = t.memoizedState, e !== null && (e = e.dehydrated, e !== null) ? (t.mode & 1 ? e.data === "$!" ? t.lanes = 8 : t.lanes = 1073741824 : t.lanes = 1, null) : (o = r.children, e = r.fallback, a ? (r = t.mode, a = t.child, o = {
      mode: "hidden",
      children: o
    }, !(r & 1) && a !== null ? (a.childLanes = 0, a.pendingProps = o) : a = oo(o, r, 0, null), e = Bn(e, r, n, null), a.return = t, e.return = t, a.sibling = e, t.child = a, t.child.memoizedState = Ls(n), t.memoizedState = Ms, e) : bi(t, o));
    if (l = e.memoizedState, l !== null && (i = l.dehydrated, i !== null)) return Ah(e, t, o, r, i, l, n);
    if (a) {
      a = r.fallback, o = t.mode, l = e.child, i = l.sibling;
      var s = {
        mode: "hidden",
        children: r.children
      };
      return !(o & 1) && t.child !== l ? (r = t.child, r.childLanes = 0, r.pendingProps = s, t.deletions = null) : (r = Nn(l, s), r.subtreeFlags = l.subtreeFlags & 14680064), i !== null ? a = Nn(i, a) : (a = Bn(a, o, n, null), a.flags |= 2), a.return = t, r.return = t, r.sibling = a, t.child = r, r = a, a = t.child, o = e.child.memoizedState, o = o === null ? Ls(n) : {
        baseLanes: o.baseLanes | n,
        cachePool: null,
        transitions: o.transitions
      }, a.memoizedState = o, a.childLanes = e.childLanes & ~n, t.memoizedState = Ms, r;
    }
    return a = e.child, e = a.sibling, r = Nn(a, {
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
    return r !== null && vi(r), Cr(t, e.child, null, n), e = bi(t, t.pendingProps.children), e.flags |= 2, t.memoizedState = null, e;
  }
  function Ah(e, t, n, r, l, a, o) {
    if (n) return t.flags & 256 ? (t.flags &= -257, r = Fo(Error(b(422))), ta(e, t, o, r)) : t.memoizedState !== null ? (t.child = e.child, t.flags |= 128, null) : (a = r.fallback, l = t.mode, r = oo({
      mode: "visible",
      children: r.children
    }, l, 0, null), a = Bn(a, l, o, null), a.flags |= 2, r.return = t, a.return = t, r.sibling = a, t.child = r, t.mode & 1 && Cr(t, e.child, null, o), t.child.memoizedState = Ls(o), t.memoizedState = Ms, a);
    if (!(t.mode & 1)) return ta(e, t, o, null);
    if (l.data === "$!") {
      if (r = l.nextSibling && l.nextSibling.dataset, r) var i = r.dgst;
      return r = i, a = Error(b(419)), r = Fo(a, r, void 0), ta(e, t, o, r);
    }
    if (i = (o & e.childLanes) !== 0, ut || i) {
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
        l = l & (r.suspendedLanes | o) ? 0 : l, l !== 0 && l !== a.retryLane && (a.retryLane = l, tn(e, l), zt(r, e, l, -1));
      }
      return zi(), r = Fo(Error(b(421))), ta(e, t, o, r);
    }
    return l.data === "$?" ? (t.flags |= 128, t.child = e.child, t = Xh.bind(null, e), l._reactRetry = t, null) : (e = a.treeContext, yt = xn(l.nextSibling), xt = t, Ce = true, Ot = null, e !== null && (Nt[jt++] = Jt, Nt[jt++] = Xt, Nt[jt++] = Wn, Jt = e.id, Xt = e.overflow, Wn = t), t = bi(t, r.children), t.flags |= 4096, t);
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
    if (nt(e, t, r.children, n), r = Re.current, r & 2) r = r & 1 | 2, t.flags |= 128;
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
    if (Se(Re, r), !(t.mode & 1)) t.memoizedState = null;
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
  function nn(e, t, n) {
    if (e !== null && (t.dependencies = e.dependencies), Qn |= t.lanes, !(n & t.childLanes)) return null;
    if (e !== null && t.child !== e.child) throw Error(b(153));
    if (t.child !== null) {
      for (e = t.child, n = Nn(e, e.pendingProps), t.child = n, n.return = t; e.sibling !== null; ) e = e.sibling, n = n.sibling = Nn(e, e.pendingProps), n.return = t;
      n.sibling = null;
    }
    return t.child;
  }
  function Uh(e, t, n) {
    switch (t.tag) {
      case 3:
        zf(t), Er();
        break;
      case 5:
        df(t);
        break;
      case 1:
        dt(t.type) && ba(t);
        break;
      case 4:
        Ni(t, t.stateNode.containerInfo);
        break;
      case 10:
        var r = t.type._context, l = t.memoizedProps.value;
        Se(La, r._currentValue), r._currentValue = l;
        break;
      case 13:
        if (r = t.memoizedState, r !== null) return r.dehydrated !== null ? (Se(Re, Re.current & 1), t.flags |= 128, null) : n & t.child.childLanes ? Af(e, t, n) : (Se(Re, Re.current & 1), e = nn(e, t, n), e !== null ? e.sibling : null);
        Se(Re, Re.current & 1);
        break;
      case 19:
        if (r = (n & t.childLanes) !== 0, e.flags & 128) {
          if (r) return Uf(e, t, n);
          t.flags |= 128;
        }
        if (l = t.memoizedState, l !== null && (l.rendering = null, l.tail = null, l.lastEffect = null), Se(Re, Re.current), r) break;
        return null;
      case 22:
      case 23:
        return t.lanes = 0, Of(e, t, n);
    }
    return nn(e, t, n);
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
      e = t.stateNode, Un(Ht.current);
      var a = null;
      switch (n) {
        case "input":
          l = rs(e, l), r = rs(e, r), a = [];
          break;
        case "select":
          l = Pe({}, l, {
            value: void 0
          }), r = Pe({}, r, {
            value: void 0
          }), a = [];
          break;
        case "textarea":
          l = os(e, l), r = os(e, r), a = [];
          break;
        default:
          typeof l.onClick != "function" && typeof r.onClick == "function" && (e.onclick = Ta);
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
        else c === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, i = i ? i.__html : void 0, s != null && i !== s && (a = a || []).push(c, s)) : c === "children" ? typeof s != "string" && typeof s != "number" || (a = a || []).push(c, "" + s) : c !== "suppressContentEditableWarning" && c !== "suppressHydrationWarning" && (ul.hasOwnProperty(c) ? (s != null && c === "onScroll" && ke("scroll", e), a || i === s || (a = [])) : (a = a || []).push(c, s));
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
  function Ze(e) {
    var t = e.alternate !== null && e.alternate.child === e.child, n = 0, r = 0;
    if (t) for (var l = e.child; l !== null; ) n |= l.lanes | l.childLanes, r |= l.subtreeFlags & 14680064, r |= l.flags & 14680064, l.return = e, l = l.sibling;
    else for (l = e.child; l !== null; ) n |= l.lanes | l.childLanes, r |= l.subtreeFlags, r |= l.flags, l.return = e, l = l.sibling;
    return e.subtreeFlags |= r, e.childLanes = n, t;
  }
  function $h(e, t, n) {
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
        return dt(t.type) && Pa(), Ze(t), null;
      case 3:
        return r = t.stateNode, _r(), Ne(ct), Ne(et), Ei(), r.pendingContext && (r.context = r.pendingContext, r.pendingContext = null), (e === null || e.child === null) && (ql(t) ? t.flags |= 4 : e === null || e.memoizedState.isDehydrated && !(t.flags & 256) || (t.flags |= 1024, Ot !== null && (Vs(Ot), Ot = null))), Os(e, t), Ze(t), null;
      case 5:
        ji(t);
        var l = Un(Sl.current);
        if (n = t.type, e !== null && t.stateNode != null) Ff(e, t, n, r, l), e.ref !== t.ref && (t.flags |= 512, t.flags |= 2097152);
        else {
          if (!r) {
            if (t.stateNode === null) throw Error(b(166));
            return Ze(t), null;
          }
          if (e = Un(Ht.current), ql(t)) {
            r = t.stateNode, n = t.type;
            var a = t.memoizedProps;
            switch (r[Vt] = t, r[xl] = a, e = (t.mode & 1) !== 0, n) {
              case "dialog":
                ke("cancel", r), ke("close", r);
                break;
              case "iframe":
              case "object":
              case "embed":
                ke("load", r);
                break;
              case "video":
              case "audio":
                for (l = 0; l < Xr.length; l++) ke(Xr[l], r);
                break;
              case "source":
                ke("error", r);
                break;
              case "img":
              case "image":
              case "link":
                ke("error", r), ke("load", r);
                break;
              case "details":
                ke("toggle", r);
                break;
              case "input":
                du(r, a), ke("invalid", r);
                break;
              case "select":
                r._wrapperState = {
                  wasMultiple: !!a.multiple
                }, ke("invalid", r);
                break;
              case "textarea":
                mu(r, a), ke("invalid", r);
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
              ]) : ul.hasOwnProperty(o) && i != null && o === "onScroll" && ke("scroll", r);
            }
            switch (n) {
              case "input":
                Wl(r), fu(r, a, true);
                break;
              case "textarea":
                Wl(r), pu(r);
                break;
              case "select":
              case "option":
                break;
              default:
                typeof a.onClick == "function" && (r.onclick = Ta);
            }
            r = l, t.updateQueue = r, r !== null && (t.flags |= 4);
          } else {
            o = l.nodeType === 9 ? l : l.ownerDocument, e === "http://www.w3.org/1999/xhtml" && (e = hd(n)), e === "http://www.w3.org/1999/xhtml" ? n === "script" ? (e = o.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild)) : typeof r.is == "string" ? e = o.createElement(n, {
              is: r.is
            }) : (e = o.createElement(n), n === "select" && (o = e, r.multiple ? o.multiple = true : r.size && (o.size = r.size))) : e = o.createElementNS(e, n), e[Vt] = t, e[xl] = r, $f(e, t, false, false), t.stateNode = e;
            e: {
              switch (o = us(n, r), n) {
                case "dialog":
                  ke("cancel", e), ke("close", e), l = r;
                  break;
                case "iframe":
                case "object":
                case "embed":
                  ke("load", e), l = r;
                  break;
                case "video":
                case "audio":
                  for (l = 0; l < Xr.length; l++) ke(Xr[l], e);
                  l = r;
                  break;
                case "source":
                  ke("error", e), l = r;
                  break;
                case "img":
                case "image":
                case "link":
                  ke("error", e), ke("load", e), l = r;
                  break;
                case "details":
                  ke("toggle", e), l = r;
                  break;
                case "input":
                  du(e, r), l = rs(e, r), ke("invalid", e);
                  break;
                case "option":
                  l = r;
                  break;
                case "select":
                  e._wrapperState = {
                    wasMultiple: !!r.multiple
                  }, l = Pe({}, r, {
                    value: void 0
                  }), ke("invalid", e);
                  break;
                case "textarea":
                  mu(e, r), l = os(e, r), ke("invalid", e);
                  break;
                default:
                  l = r;
              }
              is(n, l), i = l;
              for (a in i) if (i.hasOwnProperty(a)) {
                var s = i[a];
                a === "style" ? yd(e, s) : a === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, s != null && gd(e, s)) : a === "children" ? typeof s == "string" ? (n !== "textarea" || s !== "") && cl(e, s) : typeof s == "number" && cl(e, "" + s) : a !== "suppressContentEditableWarning" && a !== "suppressHydrationWarning" && a !== "autoFocus" && (ul.hasOwnProperty(a) ? s != null && a === "onScroll" && ke("scroll", e) : s != null && ti(e, a, s, o));
              }
              switch (n) {
                case "input":
                  Wl(e), fu(e, r, false);
                  break;
                case "textarea":
                  Wl(e), pu(e);
                  break;
                case "option":
                  r.value != null && e.setAttribute("value", "" + En(r.value));
                  break;
                case "select":
                  e.multiple = !!r.multiple, a = r.value, a != null ? vr(e, !!r.multiple, a, false) : r.defaultValue != null && vr(e, !!r.multiple, r.defaultValue, true);
                  break;
                default:
                  typeof l.onClick == "function" && (e.onclick = Ta);
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
          if (n = Un(Sl.current), Un(Ht.current), ql(t)) {
            if (r = t.stateNode, n = t.memoizedProps, r[Vt] = t, (a = r.nodeValue !== n) && (e = xt, e !== null)) switch (e.tag) {
              case 3:
                Zl(r.nodeValue, n, (e.mode & 1) !== 0);
                break;
              case 5:
                e.memoizedProps.suppressHydrationWarning !== true && Zl(r.nodeValue, n, (e.mode & 1) !== 0);
            }
            a && (t.flags |= 4);
          } else r = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(r), r[Vt] = t, t.stateNode = r;
        }
        return Ze(t), null;
      case 13:
        if (Ne(Re), r = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
          if (Ce && yt !== null && t.mode & 1 && !(t.flags & 128)) af(), Er(), t.flags |= 98560, a = false;
          else if (a = ql(t), r !== null && r.dehydrated !== null) {
            if (e === null) {
              if (!a) throw Error(b(318));
              if (a = t.memoizedState, a = a !== null ? a.dehydrated : null, !a) throw Error(b(317));
              a[Vt] = t;
            } else Er(), !(t.flags & 128) && (t.memoizedState = null), t.flags |= 4;
            Ze(t), a = false;
          } else Ot !== null && (Vs(Ot), Ot = null), a = true;
          if (!a) return t.flags & 65536 ? t : null;
        }
        return t.flags & 128 ? (t.lanes = n, t) : (r = r !== null, r !== (e !== null && e.memoizedState !== null) && r && (t.child.flags |= 8192, t.mode & 1 && (e === null || Re.current & 1 ? Be === 0 && (Be = 3) : zi())), t.updateQueue !== null && (t.flags |= 4), Ze(t), null);
      case 4:
        return _r(), Os(e, t), e === null && vl(t.stateNode.containerInfo), Ze(t), null;
      case 10:
        return wi(t.type._context), Ze(t), null;
      case 17:
        return dt(t.type) && Pa(), Ze(t), null;
      case 19:
        if (Ne(Re), a = t.memoizedState, a === null) return Ze(t), null;
        if (r = (t.flags & 128) !== 0, o = a.rendering, o === null) if (r) Br(a, false);
        else {
          if (Be !== 0 || e !== null && e.flags & 128) for (e = t.child; e !== null; ) {
            if (o = za(e), o !== null) {
              for (t.flags |= 128, Br(a, false), r = o.updateQueue, r !== null && (t.updateQueue = r, t.flags |= 4), t.subtreeFlags = 0, r = n, n = t.child; n !== null; ) a = n, e = r, a.flags &= 14680066, o = a.alternate, o === null ? (a.childLanes = 0, a.lanes = e, a.child = null, a.subtreeFlags = 0, a.memoizedProps = null, a.memoizedState = null, a.updateQueue = null, a.dependencies = null, a.stateNode = null) : (a.childLanes = o.childLanes, a.lanes = o.lanes, a.child = o.child, a.subtreeFlags = 0, a.deletions = null, a.memoizedProps = o.memoizedProps, a.memoizedState = o.memoizedState, a.updateQueue = o.updateQueue, a.type = o.type, e = o.dependencies, a.dependencies = e === null ? null : {
                lanes: e.lanes,
                firstContext: e.firstContext
              }), n = n.sibling;
              return Se(Re, Re.current & 1 | 2), t.child;
            }
            e = e.sibling;
          }
          a.tail !== null && Oe() > Tr && (t.flags |= 128, r = true, Br(a, false), t.lanes = 4194304);
        }
        else {
          if (!r) if (e = za(o), e !== null) {
            if (t.flags |= 128, r = true, n = e.updateQueue, n !== null && (t.updateQueue = n, t.flags |= 4), Br(a, true), a.tail === null && a.tailMode === "hidden" && !o.alternate && !Ce) return Ze(t), null;
          } else 2 * Oe() - a.renderingStartTime > Tr && n !== 1073741824 && (t.flags |= 128, r = true, Br(a, false), t.lanes = 4194304);
          a.isBackwards ? (o.sibling = t.child, t.child = o) : (n = a.last, n !== null ? n.sibling = o : t.child = o, a.last = o);
        }
        return a.tail !== null ? (t = a.tail, a.rendering = t, a.tail = t.sibling, a.renderingStartTime = Oe(), t.sibling = null, n = Re.current, Se(Re, r ? n & 1 | 2 : n & 1), t) : (Ze(t), null);
      case 22:
      case 23:
        return Ii(), r = t.memoizedState !== null, e !== null && e.memoizedState !== null !== r && (t.flags |= 8192), r && t.mode & 1 ? gt & 1073741824 && (Ze(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : Ze(t), null;
      case 24:
        return null;
      case 25:
        return null;
    }
    throw Error(b(156, t.tag));
  }
  function Fh(e, t) {
    switch (gi(t), t.tag) {
      case 1:
        return dt(t.type) && Pa(), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 3:
        return _r(), Ne(ct), Ne(et), Ei(), e = t.flags, e & 65536 && !(e & 128) ? (t.flags = e & -65537 | 128, t) : null;
      case 5:
        return ji(t), null;
      case 13:
        if (Ne(Re), e = t.memoizedState, e !== null && e.dehydrated !== null) {
          if (t.alternate === null) throw Error(b(340));
          Er();
        }
        return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 19:
        return Ne(Re), null;
      case 4:
        return _r(), null;
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
  var na = false, qe = false, Bh = typeof WeakSet == "function" ? WeakSet : Set, z = null;
  function hr(e, t) {
    var n = e.ref;
    if (n !== null) if (typeof n == "function") try {
      n(null);
    } catch (r) {
      Le(e, t, r);
    }
    else n.current = null;
  }
  function Is(e, t, n) {
    try {
      n();
    } catch (r) {
      Le(e, t, r);
    }
  }
  var rc = false;
  function Vh(e, t) {
    if (xs = Ca, e = Qd(), pi(e)) {
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
            for (var y; d !== n || l !== 0 && d.nodeType !== 3 || (i = o + l), d !== a || r !== 0 && d.nodeType !== 3 || (s = o + r), d.nodeType === 3 && (o += d.nodeValue.length), (y = d.firstChild) !== null; ) g = d, d = y;
            for (; ; ) {
              if (d === e) break t;
              if (g === n && ++c === l && (i = o), g === a && ++m === r && (s = o), (y = d.nextSibling) !== null) break;
              d = g, g = d.parentNode;
            }
            d = y;
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
              var S = w.memoizedProps, _ = w.memoizedState, p = t.stateNode, f = p.getSnapshotBeforeUpdate(t.elementType === t.type ? S : Dt(t.type, S), _);
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
        Le(t, t.return, j);
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
  function Vf(e) {
    var t = e.alternate;
    t !== null && (e.alternate = null, Vf(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && (delete t[Vt], delete t[xl], delete t[Ns], delete t[Eh], delete t[Ch])), e.stateNode = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
  }
  function Wf(e) {
    return e.tag === 5 || e.tag === 3 || e.tag === 4;
  }
  function lc(e) {
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
  function As(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.nodeType === 8 ? n.parentNode.insertBefore(e, t) : n.insertBefore(e, t) : (n.nodeType === 8 ? (t = n.parentNode, t.insertBefore(e, n)) : (t = n, t.appendChild(e)), n = n._reactRootContainer, n != null || t.onclick !== null || (t.onclick = Ta));
    else if (r !== 4 && (e = e.child, e !== null)) for (As(e, t, n), e = e.sibling; e !== null; ) As(e, t, n), e = e.sibling;
  }
  function Us(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.insertBefore(e, t) : n.appendChild(e);
    else if (r !== 4 && (e = e.child, e !== null)) for (Us(e, t, n), e = e.sibling; e !== null; ) Us(e, t, n), e = e.sibling;
  }
  var Ke = null, Mt = false;
  function on(e, t, n) {
    for (n = n.child; n !== null; ) Hf(e, t, n), n = n.sibling;
  }
  function Hf(e, t, n) {
    if (Wt && typeof Wt.onCommitFiberUnmount == "function") try {
      Wt.onCommitFiberUnmount(Ja, n);
    } catch {
    }
    switch (n.tag) {
      case 5:
        qe || hr(n, t);
      case 6:
        var r = Ke, l = Mt;
        Ke = null, on(e, t, n), Ke = r, Mt = l, Ke !== null && (Mt ? (e = Ke, n = n.stateNode, e.nodeType === 8 ? e.parentNode.removeChild(n) : e.removeChild(n)) : Ke.removeChild(n.stateNode));
        break;
      case 18:
        Ke !== null && (Mt ? (e = Ke, n = n.stateNode, e.nodeType === 8 ? Oo(e.parentNode, n) : e.nodeType === 1 && Oo(e, n), pl(e)) : Oo(Ke, n.stateNode));
        break;
      case 4:
        r = Ke, l = Mt, Ke = n.stateNode.containerInfo, Mt = true, on(e, t, n), Ke = r, Mt = l;
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
        on(e, t, n);
        break;
      case 1:
        if (!qe && (hr(n, t), r = n.stateNode, typeof r.componentWillUnmount == "function")) try {
          r.props = n.memoizedProps, r.state = n.memoizedState, r.componentWillUnmount();
        } catch (i) {
          Le(n, t, i);
        }
        on(e, t, n);
        break;
      case 21:
        on(e, t, n);
        break;
      case 22:
        n.mode & 1 ? (qe = (r = qe) || n.memoizedState !== null, on(e, t, n), qe = r) : on(e, t, n);
        break;
      default:
        on(e, t, n);
    }
  }
  function ac(e) {
    var t = e.updateQueue;
    if (t !== null) {
      e.updateQueue = null;
      var n = e.stateNode;
      n === null && (n = e.stateNode = new Bh()), t.forEach(function(r) {
        var l = Zh.bind(null, e, r);
        n.has(r) || (n.add(r), r.then(l, l));
      });
    }
  }
  function bt(e, t) {
    var n = t.deletions;
    if (n !== null) for (var r = 0; r < n.length; r++) {
      var l = n[r];
      try {
        var a = e, o = t, i = o;
        e: for (; i !== null; ) {
          switch (i.tag) {
            case 5:
              Ke = i.stateNode, Mt = false;
              break e;
            case 3:
              Ke = i.stateNode.containerInfo, Mt = true;
              break e;
            case 4:
              Ke = i.stateNode.containerInfo, Mt = true;
              break e;
          }
          i = i.return;
        }
        if (Ke === null) throw Error(b(160));
        Hf(a, o, l), Ke = null, Mt = false;
        var s = l.alternate;
        s !== null && (s.return = null), l.return = null;
      } catch (c) {
        Le(l, t, c);
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
        if (bt(t, e), Ft(e), r & 4) {
          try {
            al(3, e, e.return), lo(3, e);
          } catch (S) {
            Le(e, e.return, S);
          }
          try {
            al(5, e, e.return);
          } catch (S) {
            Le(e, e.return, S);
          }
        }
        break;
      case 1:
        bt(t, e), Ft(e), r & 512 && n !== null && hr(n, n.return);
        break;
      case 5:
        if (bt(t, e), Ft(e), r & 512 && n !== null && hr(n, n.return), e.flags & 32) {
          var l = e.stateNode;
          try {
            cl(l, "");
          } catch (S) {
            Le(e, e.return, S);
          }
        }
        if (r & 4 && (l = e.stateNode, l != null)) {
          var a = e.memoizedProps, o = n !== null ? n.memoizedProps : a, i = e.type, s = e.updateQueue;
          if (e.updateQueue = null, s !== null) try {
            i === "input" && a.type === "radio" && a.name != null && md(l, a), us(i, o);
            var c = us(i, a);
            for (o = 0; o < s.length; o += 2) {
              var m = s[o], d = s[o + 1];
              m === "style" ? yd(l, d) : m === "dangerouslySetInnerHTML" ? gd(l, d) : m === "children" ? cl(l, d) : ti(l, m, d, c);
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
                var y = a.value;
                y != null ? vr(l, !!a.multiple, y, false) : g !== !!a.multiple && (a.defaultValue != null ? vr(l, !!a.multiple, a.defaultValue, true) : vr(l, !!a.multiple, a.multiple ? [] : "", false));
            }
            l[xl] = a;
          } catch (S) {
            Le(e, e.return, S);
          }
        }
        break;
      case 6:
        if (bt(t, e), Ft(e), r & 4) {
          if (e.stateNode === null) throw Error(b(162));
          l = e.stateNode, a = e.memoizedProps;
          try {
            l.nodeValue = a;
          } catch (S) {
            Le(e, e.return, S);
          }
        }
        break;
      case 3:
        if (bt(t, e), Ft(e), r & 4 && n !== null && n.memoizedState.isDehydrated) try {
          pl(t.containerInfo);
        } catch (S) {
          Le(e, e.return, S);
        }
        break;
      case 4:
        bt(t, e), Ft(e);
        break;
      case 13:
        bt(t, e), Ft(e), l = e.child, l.flags & 8192 && (a = l.memoizedState !== null, l.stateNode.isHidden = a, !a || l.alternate !== null && l.alternate.memoizedState !== null || (Li = Oe())), r & 4 && ac(e);
        break;
      case 22:
        if (m = n !== null && n.memoizedState !== null, e.mode & 1 ? (qe = (c = qe) || m, bt(t, e), qe = c) : bt(t, e), Ft(e), r & 8192) {
          if (c = e.memoizedState !== null, (e.stateNode.isHidden = c) && !m && e.mode & 1) for (z = e, m = e.child; m !== null; ) {
            for (d = z = m; z !== null; ) {
              switch (g = z, y = g.child, g.tag) {
                case 0:
                case 11:
                case 14:
                case 15:
                  al(4, g, g.return);
                  break;
                case 1:
                  hr(g, g.return);
                  var w = g.stateNode;
                  if (typeof w.componentWillUnmount == "function") {
                    r = g, n = g.return;
                    try {
                      t = r, w.props = t.memoizedProps, w.state = t.memoizedState, w.componentWillUnmount();
                    } catch (S) {
                      Le(r, n, S);
                    }
                  }
                  break;
                case 5:
                  hr(g, g.return);
                  break;
                case 22:
                  if (g.memoizedState !== null) {
                    sc(d);
                    continue;
                  }
              }
              y !== null ? (y.return = g, z = y) : sc(d);
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
                  Le(e, e.return, S);
                }
              }
            } else if (d.tag === 6) {
              if (m === null) try {
                d.stateNode.nodeValue = c ? "" : d.memoizedProps;
              } catch (S) {
                Le(e, e.return, S);
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
        bt(t, e), Ft(e), r & 4 && ac(e);
        break;
      case 21:
        break;
      default:
        bt(t, e), Ft(e);
    }
  }
  function Ft(e) {
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
        Le(e, e.return, s);
      }
      e.flags &= -3;
    }
    t & 4096 && (e.flags &= -4097);
  }
  function Wh(e, t, n) {
    z = e, Gf(e);
  }
  function Gf(e, t, n) {
    for (var r = (e.mode & 1) !== 0; z !== null; ) {
      var l = z, a = l.child;
      if (l.tag === 22 && r) {
        var o = l.memoizedState !== null || na;
        if (!o) {
          var i = l.alternate, s = i !== null && i.memoizedState !== null || qe;
          i = na;
          var c = qe;
          if (na = o, (qe = s) && !c) for (z = l; z !== null; ) o = z, s = o.child, o.tag === 22 && o.memoizedState !== null ? ic(l) : s !== null ? (s.return = o, z = s) : ic(l);
          for (; a !== null; ) z = a, Gf(a), a = a.sibling;
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
                var l = t.elementType === t.type ? n.memoizedProps : Dt(t.type, n.memoizedProps);
                r.componentDidUpdate(l, n.memoizedState, r.__reactInternalSnapshotBeforeUpdate);
              }
              var a = t.updateQueue;
              a !== null && Wu(t, a, r);
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
                Wu(t, o, n);
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
          Le(t, t.return, g);
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
              Le(t, n, s);
            }
            break;
          case 1:
            var r = t.stateNode;
            if (typeof r.componentDidMount == "function") {
              var l = t.return;
              try {
                r.componentDidMount();
              } catch (s) {
                Le(t, l, s);
              }
            }
            var a = t.return;
            try {
              zs(t);
            } catch (s) {
              Le(t, a, s);
            }
            break;
          case 5:
            var o = t.return;
            try {
              zs(t);
            } catch (s) {
              Le(t, o, s);
            }
        }
      } catch (s) {
        Le(t, t.return, s);
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
  var Hh = Math.ceil, $a = rn.ReactCurrentDispatcher, Di = rn.ReactCurrentOwner, Ct = rn.ReactCurrentBatchConfig, ie = 0, We = null, $e = null, Ye = 0, gt = 0, gr = Rn(0), Be = 0, El = null, Qn = 0, ao = 0, Mi = 0, ol = null, it = null, Li = 0, Tr = 1 / 0, Kt = null, Fa = false, $s = null, Sn = null, ra = false, pn = null, Ba = 0, sl = 0, Fs = null, ga = -1, va = 0;
  function rt() {
    return ie & 6 ? Oe() : ga !== -1 ? ga : ga = Oe();
  }
  function kn(e) {
    return e.mode & 1 ? ie & 2 && Ye !== 0 ? Ye & -Ye : Rh.transition !== null ? (va === 0 && (va = Pd()), va) : (e = ge, e !== 0 || (e = window.event, e = e === void 0 ? 16 : zd(e.type)), e) : 1;
  }
  function zt(e, t, n, r) {
    if (50 < sl) throw sl = 0, Fs = null, Error(b(185));
    Tl(e, n, r), (!(ie & 2) || e !== We) && (e === We && (!(ie & 2) && (ao |= n), Be === 4 && fn(e, Ye)), ft(e, r), n === 1 && ie === 0 && !(t.mode & 1) && (Tr = Oe() + 500, to && Tn()));
  }
  function ft(e, t) {
    var n = e.callbackNode;
    Rp(e, t);
    var r = Ea(e, e === We ? Ye : 0);
    if (r === 0) n !== null && vu(n), e.callbackNode = null, e.callbackPriority = 0;
    else if (t = r & -r, e.callbackPriority !== t) {
      if (n != null && vu(n), t === 1) e.tag === 0 ? _h(uc.bind(null, e)) : nf(uc.bind(null, e)), Nh(function() {
        !(ie & 6) && Tn();
      }), n = null;
      else {
        switch (bd(r)) {
          case 1:
            n = oi;
            break;
          case 4:
            n = Rd;
            break;
          case 16:
            n = ja;
            break;
          case 536870912:
            n = Td;
            break;
          default:
            n = ja;
        }
        n = tm(n, Kf.bind(null, e));
      }
      e.callbackPriority = t, e.callbackNode = n;
    }
  }
  function Kf(e, t) {
    if (ga = -1, va = 0, ie & 6) throw Error(b(327));
    var n = e.callbackNode;
    if (kr() && e.callbackNode !== n) return null;
    var r = Ea(e, e === We ? Ye : 0);
    if (r === 0) return null;
    if (r & 30 || r & e.expiredLanes || t) t = Va(e, r);
    else {
      t = r;
      var l = ie;
      ie |= 2;
      var a = Jf();
      (We !== e || Ye !== t) && (Kt = null, Tr = Oe() + 500, Fn(e, t));
      do
        try {
          Kh();
          break;
        } catch (i) {
          Yf(e, i);
        }
      while (true);
      xi(), $a.current = a, ie = l, $e !== null ? t = 0 : (We = null, Ye = 0, t = Be);
    }
    if (t !== 0) {
      if (t === 2 && (l = ps(e), l !== 0 && (r = l, t = Bs(e, l))), t === 1) throw n = El, Fn(e, 0), fn(e, r), ft(e, Oe()), n;
      if (t === 6) fn(e, r);
      else {
        if (l = e.current.alternate, !(r & 30) && !Qh(l) && (t = Va(e, r), t === 2 && (a = ps(e), a !== 0 && (r = a, t = Bs(e, a))), t === 1)) throw n = El, Fn(e, 0), fn(e, r), ft(e, Oe()), n;
        switch (e.finishedWork = l, e.finishedLanes = r, t) {
          case 0:
          case 1:
            throw Error(b(345));
          case 2:
            Ln(e, it, Kt);
            break;
          case 3:
            if (fn(e, r), (r & 130023424) === r && (t = Li + 500 - Oe(), 10 < t)) {
              if (Ea(e, 0) !== 0) break;
              if (l = e.suspendedLanes, (l & r) !== r) {
                rt(), e.pingedLanes |= e.suspendedLanes & l;
                break;
              }
              e.timeoutHandle = ks(Ln.bind(null, e, it, Kt), t);
              break;
            }
            Ln(e, it, Kt);
            break;
          case 4:
            if (fn(e, r), (r & 4194240) === r) break;
            for (t = e.eventTimes, l = -1; 0 < r; ) {
              var o = 31 - It(r);
              a = 1 << o, o = t[o], o > l && (l = o), r &= ~a;
            }
            if (r = l, r = Oe() - r, r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * Hh(r / 1960)) - r, 10 < r) {
              e.timeoutHandle = ks(Ln.bind(null, e, it, Kt), r);
              break;
            }
            Ln(e, it, Kt);
            break;
          case 5:
            Ln(e, it, Kt);
            break;
          default:
            throw Error(b(329));
        }
      }
    }
    return ft(e, Oe()), e.callbackNode === n ? Kf.bind(null, e) : null;
  }
  function Bs(e, t) {
    var n = ol;
    return e.current.memoizedState.isDehydrated && (Fn(e, t).flags |= 256), e = Va(e, t), e !== 2 && (t = it, it = n, t !== null && Vs(t)), e;
  }
  function Vs(e) {
    it === null ? it = e : it.push.apply(it, e);
  }
  function Qh(e) {
    for (var t = e; ; ) {
      if (t.flags & 16384) {
        var n = t.updateQueue;
        if (n !== null && (n = n.stores, n !== null)) for (var r = 0; r < n.length; r++) {
          var l = n[r], a = l.getSnapshot;
          l = l.value;
          try {
            if (!At(a(), l)) return false;
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
  function fn(e, t) {
    for (t &= ~Mi, t &= ~ao, e.suspendedLanes |= t, e.pingedLanes &= ~t, e = e.expirationTimes; 0 < t; ) {
      var n = 31 - It(t), r = 1 << n;
      e[n] = -1, t &= ~r;
    }
  }
  function uc(e) {
    if (ie & 6) throw Error(b(327));
    kr();
    var t = Ea(e, 0);
    if (!(t & 1)) return ft(e, Oe()), null;
    var n = Va(e, t);
    if (e.tag !== 0 && n === 2) {
      var r = ps(e);
      r !== 0 && (t = r, n = Bs(e, r));
    }
    if (n === 1) throw n = El, Fn(e, 0), fn(e, t), ft(e, Oe()), n;
    if (n === 6) throw Error(b(345));
    return e.finishedWork = e.current.alternate, e.finishedLanes = t, Ln(e, it, Kt), ft(e, Oe()), null;
  }
  function Oi(e, t) {
    var n = ie;
    ie |= 1;
    try {
      return e(t);
    } finally {
      ie = n, ie === 0 && (Tr = Oe() + 500, to && Tn());
    }
  }
  function Gn(e) {
    pn !== null && pn.tag === 0 && !(ie & 6) && kr();
    var t = ie;
    ie |= 1;
    var n = Ct.transition, r = ge;
    try {
      if (Ct.transition = null, ge = 1, e) return e();
    } finally {
      ge = r, Ct.transition = n, ie = t, !(ie & 6) && Tn();
    }
  }
  function Ii() {
    gt = gr.current, Ne(gr);
  }
  function Fn(e, t) {
    e.finishedWork = null, e.finishedLanes = 0;
    var n = e.timeoutHandle;
    if (n !== -1 && (e.timeoutHandle = -1, kh(n)), $e !== null) for (n = $e.return; n !== null; ) {
      var r = n;
      switch (gi(r), r.tag) {
        case 1:
          r = r.type.childContextTypes, r != null && Pa();
          break;
        case 3:
          _r(), Ne(ct), Ne(et), Ei();
          break;
        case 5:
          ji(r);
          break;
        case 4:
          _r();
          break;
        case 13:
          Ne(Re);
          break;
        case 19:
          Ne(Re);
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
    if (We = e, $e = e = Nn(e.current, null), Ye = gt = t, Be = 0, El = null, Mi = ao = Qn = 0, it = ol = null, An !== null) {
      for (t = 0; t < An.length; t++) if (n = An[t], r = n.interleaved, r !== null) {
        n.interleaved = null;
        var l = r.next, a = n.pending;
        if (a !== null) {
          var o = a.next;
          a.next = l, r.next = o;
        }
        n.pending = r;
      }
      An = null;
    }
    return e;
  }
  function Yf(e, t) {
    do {
      var n = $e;
      try {
        if (xi(), ma.current = Ua, Aa) {
          for (var r = Te.memoizedState; r !== null; ) {
            var l = r.queue;
            l !== null && (l.pending = null), r = r.next;
          }
          Aa = false;
        }
        if (Hn = 0, Ve = Fe = Te = null, ll = false, kl = 0, Di.current = null, n === null || n.return === null) {
          Be = 1, El = t, $e = null;
          break;
        }
        e: {
          var a = e, o = n.return, i = n, s = t;
          if (t = Ye, i.flags |= 32768, s !== null && typeof s == "object" && typeof s.then == "function") {
            var c = s, m = i, d = m.tag;
            if (!(m.mode & 1) && (d === 0 || d === 11 || d === 15)) {
              var g = m.alternate;
              g ? (m.updateQueue = g.updateQueue, m.memoizedState = g.memoizedState, m.lanes = g.lanes) : (m.updateQueue = null, m.memoizedState = null);
            }
            var y = Ju(o);
            if (y !== null) {
              y.flags &= -257, Xu(y, o, i, a, t), y.mode & 1 && Yu(a, c, t), t = y, s = c;
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
          } else if (Ce && i.mode & 1) {
            var _ = Ju(o);
            if (_ !== null) {
              !(_.flags & 65536) && (_.flags |= 256), Xu(_, o, i, a, t), vi(Rr(s, i));
              break e;
            }
          }
          a = s = Rr(s, i), Be !== 4 && (Be = 2), ol === null ? ol = [
            a
          ] : ol.push(a), a = o;
          do {
            switch (a.tag) {
              case 3:
                a.flags |= 65536, t &= -t, a.lanes |= t;
                var p = Df(a, s, t);
                Vu(a, p);
                break e;
              case 1:
                i = s;
                var f = a.type, h = a.stateNode;
                if (!(a.flags & 128) && (typeof f.getDerivedStateFromError == "function" || h !== null && typeof h.componentDidCatch == "function" && (Sn === null || !Sn.has(h)))) {
                  a.flags |= 65536, t &= -t, a.lanes |= t;
                  var j = Mf(a, i, t);
                  Vu(a, j);
                  break e;
                }
            }
            a = a.return;
          } while (a !== null);
        }
        Zf(n);
      } catch (C) {
        t = C, $e === n && n !== null && ($e = n = n.return);
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
    (Be === 0 || Be === 3 || Be === 2) && (Be = 4), We === null || !(Qn & 268435455) && !(ao & 268435455) || fn(We, Ye);
  }
  function Va(e, t) {
    var n = ie;
    ie |= 2;
    var r = Jf();
    (We !== e || Ye !== t) && (Kt = null, Fn(e, t));
    do
      try {
        Gh();
        break;
      } catch (l) {
        Yf(e, l);
      }
    while (true);
    if (xi(), ie = n, $a.current = r, $e !== null) throw Error(b(261));
    return We = null, Ye = 0, Be;
  }
  function Gh() {
    for (; $e !== null; ) Xf($e);
  }
  function Kh() {
    for (; $e !== null && !xp(); ) Xf($e);
  }
  function Xf(e) {
    var t = em(e.alternate, e, gt);
    e.memoizedProps = e.pendingProps, t === null ? Zf(e) : $e = t, Di.current = null;
  }
  function Zf(e) {
    var t = e;
    do {
      var n = t.alternate;
      if (e = t.return, t.flags & 32768) {
        if (n = Fh(n, t), n !== null) {
          n.flags &= 32767, $e = n;
          return;
        }
        if (e !== null) e.flags |= 32768, e.subtreeFlags = 0, e.deletions = null;
        else {
          Be = 6, $e = null;
          return;
        }
      } else if (n = $h(n, t, gt), n !== null) {
        $e = n;
        return;
      }
      if (t = t.sibling, t !== null) {
        $e = t;
        return;
      }
      $e = t = e;
    } while (t !== null);
    Be === 0 && (Be = 5);
  }
  function Ln(e, t, n) {
    var r = ge, l = Ct.transition;
    try {
      Ct.transition = null, ge = 1, Yh(e, t, n, r);
    } finally {
      Ct.transition = l, ge = r;
    }
    return null;
  }
  function Yh(e, t, n, r) {
    do
      kr();
    while (pn !== null);
    if (ie & 6) throw Error(b(327));
    n = e.finishedWork;
    var l = e.finishedLanes;
    if (n === null) return null;
    if (e.finishedWork = null, e.finishedLanes = 0, n === e.current) throw Error(b(177));
    e.callbackNode = null, e.callbackPriority = 0;
    var a = n.lanes | n.childLanes;
    if (Tp(e, a), e === We && ($e = We = null, Ye = 0), !(n.subtreeFlags & 2064) && !(n.flags & 2064) || ra || (ra = true, tm(ja, function() {
      return kr(), null;
    })), a = (n.flags & 15990) !== 0, n.subtreeFlags & 15990 || a) {
      a = Ct.transition, Ct.transition = null;
      var o = ge;
      ge = 1;
      var i = ie;
      ie |= 4, Di.current = null, Vh(e, n), Qf(n, e), hh(ws), Ca = !!xs, ws = xs = null, e.current = n, Wh(n), wp(), ie = i, ge = o, Ct.transition = a;
    } else e.current = n;
    if (ra && (ra = false, pn = e, Ba = l), a = e.pendingLanes, a === 0 && (Sn = null), Np(n.stateNode), ft(e, Oe()), t !== null) for (r = e.onRecoverableError, n = 0; n < t.length; n++) l = t[n], r(l.value, {
      componentStack: l.stack,
      digest: l.digest
    });
    if (Fa) throw Fa = false, e = $s, $s = null, e;
    return Ba & 1 && e.tag !== 0 && kr(), a = e.pendingLanes, a & 1 ? e === Fs ? sl++ : (sl = 0, Fs = e) : sl = 0, Tn(), null;
  }
  function kr() {
    if (pn !== null) {
      var e = bd(Ba), t = Ct.transition, n = ge;
      try {
        if (Ct.transition = null, ge = 16 > e ? 16 : e, pn === null) var r = false;
        else {
          if (e = pn, pn = null, Ba = 0, ie & 6) throw Error(b(331));
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
                      var g = m.sibling, y = m.return;
                      if (Vf(m), m === c) {
                        z = null;
                        break;
                      }
                      if (g !== null) {
                        g.return = y, z = g;
                        break;
                      }
                      z = y;
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
                Le(i, i.return, C);
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
          if (ie = l, Tn(), Wt && typeof Wt.onPostCommitFiberRoot == "function") try {
            Wt.onPostCommitFiberRoot(Ja, e);
          } catch {
          }
          r = true;
        }
        return r;
      } finally {
        ge = n, Ct.transition = t;
      }
    }
    return false;
  }
  function cc(e, t, n) {
    t = Rr(n, t), t = Df(e, t, 1), e = wn(e, t, 1), t = rt(), e !== null && (Tl(e, 1, t), ft(e, t));
  }
  function Le(e, t, n) {
    if (e.tag === 3) cc(e, e, n);
    else for (; t !== null; ) {
      if (t.tag === 3) {
        cc(t, e, n);
        break;
      } else if (t.tag === 1) {
        var r = t.stateNode;
        if (typeof t.type.getDerivedStateFromError == "function" || typeof r.componentDidCatch == "function" && (Sn === null || !Sn.has(r))) {
          e = Rr(n, e), e = Mf(t, e, 1), t = wn(t, e, 1), e = rt(), t !== null && (Tl(t, 1, e), ft(t, e));
          break;
        }
      }
      t = t.return;
    }
  }
  function Jh(e, t, n) {
    var r = e.pingCache;
    r !== null && r.delete(t), t = rt(), e.pingedLanes |= e.suspendedLanes & n, We === e && (Ye & n) === n && (Be === 4 || Be === 3 && (Ye & 130023424) === Ye && 500 > Oe() - Li ? Fn(e, 0) : Mi |= n), ft(e, t);
  }
  function qf(e, t) {
    t === 0 && (e.mode & 1 ? (t = Gl, Gl <<= 1, !(Gl & 130023424) && (Gl = 4194304)) : t = 1);
    var n = rt();
    e = tn(e, t), e !== null && (Tl(e, t, n), ft(e, n));
  }
  function Xh(e) {
    var t = e.memoizedState, n = 0;
    t !== null && (n = t.retryLane), qf(e, n);
  }
  function Zh(e, t) {
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
    if (e !== null) if (e.memoizedProps !== t.pendingProps || ct.current) ut = true;
    else {
      if (!(e.lanes & n) && !(t.flags & 128)) return ut = false, Uh(e, t, n);
      ut = !!(e.flags & 131072);
    }
    else ut = false, Ce && t.flags & 1048576 && rf(t, Ma, t.index);
    switch (t.lanes = 0, t.tag) {
      case 2:
        var r = t.type;
        ha(e, t), e = t.pendingProps;
        var l = jr(t, et.current);
        Sr(t, n), l = _i(null, t, r, e, l, n);
        var a = Ri();
        return t.flags |= 1, typeof l == "object" && l !== null && typeof l.render == "function" && l.$$typeof === void 0 ? (t.tag = 1, t.memoizedState = null, t.updateQueue = null, dt(r) ? (a = true, ba(t)) : a = false, t.memoizedState = l.state !== null && l.state !== void 0 ? l.state : null, ki(t), l.updater = ro, t.stateNode = l, l._reactInternals = t, Ts(t, r, e, n), t = Ds(null, t, r, true, a, n)) : (t.tag = 0, Ce && a && hi(t), nt(null, t, l, n), t = t.child), t;
      case 16:
        r = t.elementType;
        e: {
          switch (ha(e, t), e = t.pendingProps, l = r._init, r = l(r._payload), t.type = r, l = t.tag = eg(r), e = Dt(r, e), l) {
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
              t = qu(null, t, r, Dt(r.type, e), n);
              break e;
          }
          throw Error(b(306, r, ""));
        }
        return t;
      case 0:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), bs(e, t, r, l, n);
      case 1:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), ec(e, t, r, l, n);
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
            l = Rr(Error(b(423)), t), t = tc(e, t, r, n, l);
            break e;
          } else if (r !== l) {
            l = Rr(Error(b(424)), t), t = tc(e, t, r, n, l);
            break e;
          } else for (yt = xn(t.stateNode.containerInfo.firstChild), xt = t, Ce = true, Ot = null, n = sf(t, null, r, n), t.child = n; n; ) n.flags = n.flags & -3 | 4096, n = n.sibling;
          else {
            if (Er(), r === l) {
              t = nn(e, t, n);
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
        return Ni(t, t.stateNode.containerInfo), r = t.pendingProps, e === null ? t.child = Cr(t, null, r, n) : nt(e, t, r, n), t.child;
      case 11:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), Zu(e, t, r, l, n);
      case 7:
        return nt(e, t, t.pendingProps, n), t.child;
      case 8:
        return nt(e, t, t.pendingProps.children, n), t.child;
      case 12:
        return nt(e, t, t.pendingProps.children, n), t.child;
      case 10:
        e: {
          if (r = t.type._context, l = t.pendingProps, a = t.memoizedProps, o = l.value, Se(La, r._currentValue), r._currentValue = o, a !== null) if (At(a.value, o)) {
            if (a.children === l.children && !ct.current) {
              t = nn(e, t, n);
              break e;
            }
          } else for (a = t.child, a !== null && (a.return = t); a !== null; ) {
            var i = a.dependencies;
            if (i !== null) {
              o = a.child;
              for (var s = i.firstContext; s !== null; ) {
                if (s.context === r) {
                  if (a.tag === 1) {
                    s = Zt(-1, n & -n), s.tag = 2;
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
        return l = t.type, r = t.pendingProps.children, Sr(t, n), l = _t(l), r = r(l), t.flags |= 1, nt(e, t, r, n), t.child;
      case 14:
        return r = t.type, l = Dt(r, t.pendingProps), l = Dt(r.type, l), qu(e, t, r, l, n);
      case 15:
        return Lf(e, t, t.type, t.pendingProps, n);
      case 17:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), ha(e, t), t.tag = 1, dt(r) ? (e = true, ba(t)) : e = false, Sr(t, n), bf(t, r, l), Ts(t, r, l, n), Ds(null, t, r, true, e, n);
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
  function qh(e, t, n, r) {
    this.tag = e, this.key = n, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = r, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function Et(e, t, n, r) {
    return new qh(e, t, n, r);
  }
  function Ai(e) {
    return e = e.prototype, !(!e || !e.isReactComponent);
  }
  function eg(e) {
    if (typeof e == "function") return Ai(e) ? 1 : 0;
    if (e != null) {
      if (e = e.$$typeof, e === ri) return 11;
      if (e === li) return 14;
    }
    return 2;
  }
  function Nn(e, t) {
    var n = e.alternate;
    return n === null ? (n = Et(e.tag, t, e.key, e.mode), n.elementType = e.elementType, n.type = e.type, n.stateNode = e.stateNode, n.alternate = e, e.alternate = n) : (n.pendingProps = t, n.type = e.type, n.flags = 0, n.subtreeFlags = 0, n.deletions = null), n.flags = e.flags & 14680064, n.childLanes = e.childLanes, n.lanes = e.lanes, n.child = e.child, n.memoizedProps = e.memoizedProps, n.memoizedState = e.memoizedState, n.updateQueue = e.updateQueue, t = e.dependencies, n.dependencies = t === null ? null : {
      lanes: t.lanes,
      firstContext: t.firstContext
    }, n.sibling = e.sibling, n.index = e.index, n.ref = e.ref, n;
  }
  function ya(e, t, n, r, l, a) {
    var o = 2;
    if (r = e, typeof e == "function") Ai(e) && (o = 1);
    else if (typeof e == "string") o = 5;
    else e: switch (e) {
      case or:
        return Bn(n.children, l, a, t);
      case ni:
        o = 8, l |= 8;
        break;
      case qo:
        return e = Et(12, n, t, l | 2), e.elementType = qo, e.lanes = a, e;
      case es:
        return e = Et(13, n, t, l), e.elementType = es, e.lanes = a, e;
      case ts:
        return e = Et(19, n, t, l), e.elementType = ts, e.lanes = a, e;
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
          case un:
            o = 16, r = null;
            break e;
        }
        throw Error(b(130, e == null ? e : typeof e, ""));
    }
    return t = Et(o, n, t, l), t.elementType = e, t.type = r, t.lanes = a, t;
  }
  function Bn(e, t, n, r) {
    return e = Et(7, e, r, t), e.lanes = n, e;
  }
  function oo(e, t, n, r) {
    return e = Et(22, e, r, t), e.elementType = cd, e.lanes = n, e.stateNode = {
      isHidden: false
    }, e;
  }
  function Vo(e, t, n) {
    return e = Et(6, e, null, t), e.lanes = n, e;
  }
  function Wo(e, t, n) {
    return t = Et(4, e.children !== null ? e.children : [], e.key, t), t.lanes = n, t.stateNode = {
      containerInfo: e.containerInfo,
      pendingChildren: null,
      implementation: e.implementation
    }, t;
  }
  function tg(e, t, n, r, l) {
    this.tag = t, this.containerInfo = e, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = jo(0), this.expirationTimes = jo(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = jo(0), this.identifierPrefix = r, this.onRecoverableError = l, this.mutableSourceEagerHydrationData = null;
  }
  function Ui(e, t, n, r, l, a, o, i, s) {
    return e = new tg(e, t, n, i, s), t === 1 ? (t = 1, a === true && (t |= 8)) : t = 0, a = Et(3, null, null, t), e.current = a, a.stateNode = e, a.memoizedState = {
      element: r,
      isDehydrated: n,
      cache: null,
      transitions: null,
      pendingSuspenseBoundaries: null
    }, ki(a), e;
  }
  function ng(e, t, n) {
    var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: ar,
      key: r == null ? null : "" + r,
      children: e,
      containerInfo: t,
      implementation: n
    };
  }
  function nm(e) {
    if (!e) return Cn;
    e = e._reactInternals;
    e: {
      if (Jn(e) !== e || e.tag !== 1) throw Error(b(170));
      var t = e;
      do {
        switch (t.tag) {
          case 3:
            t = t.stateNode.context;
            break e;
          case 1:
            if (dt(t.type)) {
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
      if (dt(n)) return tf(e, n, t);
    }
    return t;
  }
  function rm(e, t, n, r, l, a, o, i, s) {
    return e = Ui(n, r, true, e, l, a, o, i, s), e.context = nm(null), n = e.current, r = rt(), l = kn(n), a = Zt(r, l), a.callback = t ?? null, wn(n, a, l), e.current.lanes = l, Tl(e, l, r), ft(e, r), e;
  }
  function so(e, t, n, r) {
    var l = t.current, a = rt(), o = kn(l);
    return n = nm(n), t.context === null ? t.context = n : t.pendingContext = n, t = Zt(a, o), t.payload = {
      element: e
    }, r = r === void 0 ? null : r, r !== null && (t.callback = r), e = wn(l, t, o), e !== null && (zt(e, l, o, a), fa(e, l, o)), o;
  }
  function Wa(e) {
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
  function rg() {
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
      Gn(function() {
        so(null, e, null, null);
      }), t[en] = null;
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
      for (var n = 0; n < dn.length && t !== 0 && t < dn[n].priority; n++) ;
      dn.splice(n, 0, e), n === 0 && Id(e);
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
  function lg(e, t, n, r, l) {
    if (l) {
      if (typeof r == "function") {
        var a = r;
        r = function() {
          var c = Wa(o);
          a.call(c);
        };
      }
      var o = rm(t, r, e, 0, null, false, false, "", fc);
      return e._reactRootContainer = o, e[en] = o.current, vl(e.nodeType === 8 ? e.parentNode : e), Gn(), o;
    }
    for (; l = e.lastChild; ) e.removeChild(l);
    if (typeof r == "function") {
      var i = r;
      r = function() {
        var c = Wa(s);
        i.call(c);
      };
    }
    var s = Ui(e, 0, false, null, null, false, false, "", fc);
    return e._reactRootContainer = s, e[en] = s.current, vl(e.nodeType === 8 ? e.parentNode : e), Gn(function() {
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
          var s = Wa(o);
          i.call(s);
        };
      }
      so(t, o, e, l);
    } else o = lg(n, t, e, l, r);
    return Wa(o);
  }
  Dd = function(e) {
    switch (e.tag) {
      case 3:
        var t = e.stateNode;
        if (t.current.memoizedState.isDehydrated) {
          var n = Jr(t.pendingLanes);
          n !== 0 && (si(t, n | 1), ft(t, Oe()), !(ie & 6) && (Tr = Oe() + 500, Tn()));
        }
        break;
      case 13:
        Gn(function() {
          var r = tn(e, 1);
          if (r !== null) {
            var l = rt();
            zt(r, e, 1, l);
          }
        }), $i(e, 1);
    }
  };
  ii = function(e) {
    if (e.tag === 13) {
      var t = tn(e, 134217728);
      if (t !== null) {
        var n = rt();
        zt(t, e, 134217728, n);
      }
      $i(e, 134217728);
    }
  };
  Md = function(e) {
    if (e.tag === 13) {
      var t = kn(e), n = tn(e, t);
      if (n !== null) {
        var r = rt();
        zt(n, e, t, r);
      }
      $i(e, t);
    }
  };
  Ld = function() {
    return ge;
  };
  Od = function(e, t) {
    var n = ge;
    try {
      return ge = e, t();
    } finally {
      ge = n;
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
        t = n.value, t != null && vr(e, !!n.multiple, t, false);
    }
  };
  Sd = Oi;
  kd = Gn;
  var ag = {
    usingClientEntryPoint: false,
    Events: [
      bl,
      cr,
      eo,
      xd,
      wd,
      Oi
    ]
  }, Vr = {
    findFiberByHostInstance: zn,
    bundleType: 0,
    version: "18.3.1",
    rendererPackageName: "react-dom"
  }, og = {
    bundleType: Vr.bundleType,
    version: Vr.version,
    rendererPackageName: Vr.rendererPackageName,
    rendererConfig: Vr.rendererConfig,
    overrideHookState: null,
    overrideHookStateDeletePath: null,
    overrideHookStateRenamePath: null,
    overrideProps: null,
    overridePropsDeletePath: null,
    overridePropsRenamePath: null,
    setErrorHandler: null,
    setSuspenseHandler: null,
    scheduleUpdate: null,
    currentDispatcherRef: rn.ReactCurrentDispatcher,
    findHostInstanceByFiber: function(e) {
      return e = Ed(e), e === null ? null : e.stateNode;
    },
    findFiberByHostInstance: Vr.findFiberByHostInstance || rg,
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
      Ja = la.inject(og), Wt = la;
    } catch {
    }
  }
  St.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = ag;
  St.createPortal = function(e, t) {
    var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!Bi(t)) throw Error(b(200));
    return ng(e, t, null, n);
  };
  St.createRoot = function(e, t) {
    if (!Bi(e)) throw Error(b(299));
    var n = false, r = "", l = lm;
    return t != null && (t.unstable_strictMode === true && (n = true), t.identifierPrefix !== void 0 && (r = t.identifierPrefix), t.onRecoverableError !== void 0 && (l = t.onRecoverableError)), t = Ui(e, 1, false, null, null, n, false, r, l), e[en] = t.current, vl(e.nodeType === 8 ? e.parentNode : e), new Fi(t);
  };
  St.findDOMNode = function(e) {
    if (e == null) return null;
    if (e.nodeType === 1) return e;
    var t = e._reactInternals;
    if (t === void 0) throw typeof e.render == "function" ? Error(b(188)) : (e = Object.keys(e).join(","), Error(b(268, e)));
    return e = Ed(t), e = e === null ? null : e.stateNode, e;
  };
  St.flushSync = function(e) {
    return Gn(e);
  };
  St.hydrate = function(e, t, n) {
    if (!uo(t)) throw Error(b(200));
    return co(null, e, t, true, n);
  };
  St.hydrateRoot = function(e, t, n) {
    if (!Bi(e)) throw Error(b(405));
    var r = n != null && n.hydratedSources || null, l = false, a = "", o = lm;
    if (n != null && (n.unstable_strictMode === true && (l = true), n.identifierPrefix !== void 0 && (a = n.identifierPrefix), n.onRecoverableError !== void 0 && (o = n.onRecoverableError)), t = rm(t, null, e, 1, n ?? null, l, false, a, o), e[en] = t.current, vl(e), r) for (e = 0; e < r.length; e++) n = r[e], l = n._getVersion, l = l(n._source), t.mutableSourceEagerHydrationData == null ? t.mutableSourceEagerHydrationData = [
      n,
      l
    ] : t.mutableSourceEagerHydrationData.push(n, l);
    return new io(t);
  };
  St.render = function(e, t, n) {
    if (!uo(t)) throw Error(b(200));
    return co(null, e, t, false, n);
  };
  St.unmountComponentAtNode = function(e) {
    if (!uo(e)) throw Error(b(40));
    return e._reactRootContainer ? (Gn(function() {
      co(null, null, e, false, function() {
        e._reactRootContainer = null, e[en] = null;
      });
    }), true) : false;
  };
  St.unstable_batchedUpdates = Oi;
  St.unstable_renderSubtreeIntoContainer = function(e, t, n, r) {
    if (!uo(n)) throw Error(b(200));
    if (e == null || e._reactInternals === void 0) throw Error(b(38));
    return co(e, t, n, false, r);
  };
  St.version = "18.3.1-next-f1338f8080-20240426";
  function am() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) try {
      __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(am);
    } catch (e) {
      console.error(e);
    }
  }
  am(), ld.exports = St;
  var Vi = ld.exports;
  const sg = Qc(Vi), ig = Hc({
    __proto__: null,
    default: sg
  }, [
    Vi
  ]);
  var mc = Vi;
  Xo.createRoot = mc.createRoot, Xo.hydrateRoot = mc.hydrateRoot;
  function Ee() {
    return Ee = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ee.apply(this, arguments);
  }
  var Ue;
  (function(e) {
    e.Pop = "POP", e.Push = "PUSH", e.Replace = "REPLACE";
  })(Ue || (Ue = {}));
  const pc = "popstate";
  function ug(e) {
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
      return typeof l == "string" ? l : Ml(l);
    }
    return dg(t, n, null, e);
  }
  function ae(e, t) {
    if (e === false || e === null || typeof e > "u") throw new Error(t);
  }
  function Kn(e, t) {
    if (!e) {
      typeof console < "u" && console.warn(t);
      try {
        throw new Error(t);
      } catch {
      }
    }
  }
  function cg() {
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
    return n === void 0 && (n = null), Ee({
      pathname: typeof e == "string" ? e : e.pathname,
      search: "",
      hash: ""
    }, typeof t == "string" ? Pn(t) : t, {
      state: n,
      key: t && t.key || r || cg()
    });
  }
  function Ml(e) {
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
  function dg(e, t, n, r) {
    r === void 0 && (r = {});
    let { window: l = document.defaultView, v5Compat: a = false } = r, o = l.history, i = Ue.Pop, s = null, c = m();
    c == null && (c = 0, o.replaceState(Ee({}, o.state, {
      idx: c
    }), ""));
    function m() {
      return (o.state || {
        idx: null
      }).idx;
    }
    function d() {
      i = Ue.Pop;
      let _ = m(), p = _ == null ? null : _ - c;
      c = _, s && s({
        action: i,
        location: S.location,
        delta: p
      });
    }
    function g(_, p) {
      i = Ue.Push;
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
    function y(_, p) {
      i = Ue.Replace;
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
      let p = l.location.origin !== "null" ? l.location.origin : l.location.href, f = typeof _ == "string" ? _ : Ml(_);
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
      replace: y,
      go(_) {
        return o.go(_);
      }
    };
    return S;
  }
  var he;
  (function(e) {
    e.data = "data", e.deferred = "deferred", e.redirect = "redirect", e.error = "error";
  })(he || (he = {}));
  const fg = /* @__PURE__ */ new Set([
    "lazy",
    "caseSensitive",
    "path",
    "id",
    "index",
    "children"
  ]);
  function mg(e) {
    return e.index === true;
  }
  function Ha(e, t, n, r) {
    return n === void 0 && (n = []), r === void 0 && (r = {}), e.map((l, a) => {
      let o = [
        ...n,
        String(a)
      ], i = typeof l.id == "string" ? l.id : o.join("-");
      if (ae(l.index !== true || !l.children, "Cannot specify children on an index route"), ae(!r[i], 'Found a route id collision on id "' + i + `".  Route id's must be globally unique within Data Router usages`), mg(l)) {
        let s = Ee({}, l, t(l), {
          id: i
        });
        return r[i] = s, s;
      } else {
        let s = Ee({}, l, t(l), {
          id: i,
          children: void 0
        });
        return r[i] = s, l.children && (s.children = Ha(l.children, t, o, r)), s;
      }
    });
  }
  function On(e, t, n) {
    return n === void 0 && (n = "/"), xa(e, t, n, false);
  }
  function xa(e, t, n, r) {
    let l = typeof t == "string" ? Pn(t) : t, a = Ll(l.pathname || "/", n);
    if (a == null) return null;
    let o = om(e);
    hg(o);
    let i = null;
    for (let s = 0; i == null && s < o.length; ++s) {
      let c = Cg(a);
      i = jg(o[s], c, r);
    }
    return i;
  }
  function pg(e, t) {
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
      let c = jn([
        r,
        s.relativePath
      ]), m = n.concat(s);
      a.children && a.children.length > 0 && (ae(a.index !== true, "Index routes must not have child routes. Please remove " + ('all child routes from route path "' + c + '".')), om(a.children, t, m, c)), !(a.path == null && !a.index) && t.push({
        path: c,
        score: kg(c, a.index),
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
  function hg(e) {
    e.sort((t, n) => t.score !== n.score ? n.score - t.score : Ng(t.routesMeta.map((r) => r.childrenIndex), n.routesMeta.map((r) => r.childrenIndex)));
  }
  const gg = /^:[\w-]+$/, vg = 3, yg = 2, xg = 1, wg = 10, Sg = -2, gc = (e) => e === "*";
  function kg(e, t) {
    let n = e.split("/"), r = n.length;
    return n.some(gc) && (r += Sg), t && (r += yg), n.filter((l) => !gc(l)).reduce((l, a) => l + (gg.test(a) ? vg : a === "" ? xg : wg), r);
  }
  function Ng(e, t) {
    return e.length === t.length && e.slice(0, -1).every((r, l) => r === t[l]) ? e[e.length - 1] - t[t.length - 1] : 0;
  }
  function jg(e, t, n) {
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
        pathname: jn([
          a,
          d.pathname
        ]),
        pathnameBase: Pg(jn([
          a,
          d.pathnameBase
        ])),
        route: g
      }), d.pathnameBase !== "/" && (a = jn([
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
    let [n, r] = Eg(e.path, e.caseSensitive, e.end), l = t.match(n);
    if (!l) return null;
    let a = l[0], o = a.replace(/(.)\/+$/, "$1"), i = l.slice(1);
    return {
      params: r.reduce((c, m, d) => {
        let { paramName: g, isOptional: y } = m;
        if (g === "*") {
          let S = i[d] || "";
          o = a.slice(0, a.length - S.length).replace(/(.)\/+$/, "$1");
        }
        const w = i[d];
        return y && !w ? c[g] = void 0 : c[g] = (w || "").replace(/%2F/g, "/"), c;
      }, {}),
      pathname: a,
      pathnameBase: o,
      pattern: e
    };
  }
  function Eg(e, t, n) {
    t === void 0 && (t = false), n === void 0 && (n = true), Kn(e === "*" || !e.endsWith("*") || e.endsWith("/*"), 'Route path "' + e + '" will be treated as if it were ' + ('"' + e.replace(/\*$/, "/*") + '" because the `*` character must ') + "always follow a `/` in the pattern. To get rid of this warning, " + ('please change the route path to "' + e.replace(/\*$/, "/*") + '".'));
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
  function Cg(e) {
    try {
      return e.split("/").map((t) => decodeURIComponent(t).replace(/\//g, "%2F")).join("/");
    } catch (t) {
      return Kn(false, 'The URL path "' + e + '" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent ' + ("encoding (" + t + ").")), e;
    }
  }
  function Ll(e, t) {
    if (t === "/") return e;
    if (!e.toLowerCase().startsWith(t.toLowerCase())) return null;
    let n = t.endsWith("/") ? t.length - 1 : t.length, r = e.charAt(n);
    return r && r !== "/" ? null : e.slice(n) || "/";
  }
  const _g = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Rg = (e) => _g.test(e);
  function Tg(e, t) {
    t === void 0 && (t = "/");
    let { pathname: n, search: r = "", hash: l = "" } = typeof e == "string" ? Pn(e) : e, a;
    if (n) if (Rg(n)) a = n;
    else {
      if (n.includes("//")) {
        let o = n;
        n = n.replace(/\/\/+/g, "/"), Kn(false, "Pathnames cannot have embedded double slashes - normalizing " + (o + " -> " + n));
      }
      n.startsWith("/") ? a = yc(n.substring(1), "/") : a = yc(n, t);
    }
    else a = t;
    return {
      pathname: a,
      search: bg(r),
      hash: Dg(l)
    };
  }
  function yc(e, t) {
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
    typeof e == "string" ? l = Pn(e) : (l = Ee({}, e), ae(!l.pathname || !l.pathname.includes("?"), Ho("?", "pathname", "search", l)), ae(!l.pathname || !l.pathname.includes("#"), Ho("#", "pathname", "hash", l)), ae(!l.search || !l.search.includes("#"), Ho("#", "search", "hash", l)));
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
  const jn = (e) => e.join("/").replace(/\/\/+/g, "/"), Pg = (e) => e.replace(/\/+$/, "").replace(/^\/*/, "/"), bg = (e) => !e || e === "?" ? "" : e.startsWith("?") ? e : "?" + e, Dg = (e) => !e || e === "#" ? "" : e.startsWith("#") ? e : "#" + e;
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
  ], Mg = new Set(dm), Lg = [
    "get",
    ...dm
  ], Og = new Set(Lg), Ig = /* @__PURE__ */ new Set([
    301,
    302,
    303,
    307,
    308
  ]), zg = /* @__PURE__ */ new Set([
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
  }, Ag = {
    state: "idle",
    data: void 0,
    formMethod: void 0,
    formAction: void 0,
    formEncType: void 0,
    formData: void 0,
    json: void 0,
    text: void 0
  }, Wr = {
    state: "unblocked",
    proceed: void 0,
    reset: void 0,
    location: void 0
  }, Wi = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Ug = (e) => ({
    hasErrorBoundary: !!e.hasErrorBoundary
  }), fm = "remix-router-transitions";
  function $g(e) {
    const t = e.window ? e.window : typeof window < "u" ? window : void 0, n = typeof t < "u" && typeof t.document < "u" && typeof t.document.createElement < "u", r = !n;
    ae(e.routes.length > 0, "You must provide a non-empty routes array to createRouter");
    let l;
    if (e.mapRouteProperties) l = e.mapRouteProperties;
    else if (e.detectErrorBoundary) {
      let x = e.detectErrorBoundary;
      l = (N) => ({
        hasErrorBoundary: x(N)
      });
    } else l = Ug;
    let a = {}, o = Ha(e.routes, l, void 0, a), i, s = e.basename || "/", c = e.dataStrategy || Wg, m = e.patchRoutesOnNavigation, d = Ee({
      v7_fetcherPersist: false,
      v7_normalizeFormMethod: false,
      v7_partialHydration: false,
      v7_prependBasename: false,
      v7_relativeSplatPath: false,
      v7_skipActionErrorRevalidation: false
    }, e.future), g = null, y = /* @__PURE__ */ new Set(), w = null, S = null, _ = null, p = e.hydrationData != null, f = On(o, e.history.location, s), h = false, j = null;
    if (f == null && !m) {
      let x = st(404, {
        pathname: e.history.location.pathname
      }), { matches: N, route: E } = Tc(o);
      f = N, j = {
        [E.id]: x
      };
    }
    f && !e.hydrationData && Al(f, o, e.history.location.pathname).active && (f = null);
    let C;
    if (f) if (f.some((x) => x.route.lazy)) C = false;
    else if (!f.some((x) => x.route.loader)) C = true;
    else if (d.v7_partialHydration) {
      let x = e.hydrationData ? e.hydrationData.loaderData : null, N = e.hydrationData ? e.hydrationData.errors : null;
      if (N) {
        let E = f.findIndex((P) => N[P.route.id] !== void 0);
        C = f.slice(0, E + 1).every((P) => !Hs(P.route, x, N));
      } else C = f.every((E) => !Hs(E.route, x, N));
    } else C = e.hydrationData != null;
    else if (C = false, f = [], d.v7_partialHydration) {
      let x = Al(null, o, e.history.location.pathname);
      x.active && x.matches && (h = true, f = x.matches);
    }
    let T, k = {
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
    }, R = Ue.Pop, $ = false, M, K = false, J = /* @__PURE__ */ new Map(), de = null, se = false, _e = false, He = [], mt = /* @__PURE__ */ new Set(), D = /* @__PURE__ */ new Map(), Q = 0, V = -1, le = /* @__PURE__ */ new Map(), ee = /* @__PURE__ */ new Set(), fe = /* @__PURE__ */ new Map(), X = /* @__PURE__ */ new Map(), ue = /* @__PURE__ */ new Set(), je = /* @__PURE__ */ new Map(), te = /* @__PURE__ */ new Map(), be;
    function ve() {
      if (g = e.history.listen((x) => {
        let { action: N, location: E, delta: P } = x;
        if (be) {
          be(), be = void 0;
          return;
        }
        Kn(te.size === 0 || P != null, "You are trying to use a blocker on a POP navigation to a location that was not created by @remix-run/router. This will fail silently in production. This can happen if you are navigating outside the router via `window.history.pushState`/`window.location.hash` instead of using router navigation APIs.  This can also happen if you are using createHashRouter and the user manually changes the URL.");
        let O = tu({
          currentLocation: k.location,
          nextLocation: E,
          historyAction: N
        });
        if (O && P != null) {
          let G = new Promise((Y) => {
            be = Y;
          });
          e.history.go(P * -1), zl(O, {
            state: "blocked",
            location: E,
            proceed() {
              zl(O, {
                state: "proceeding",
                proceed: void 0,
                reset: void 0,
                location: E
              }), G.then(() => e.history.go(P));
            },
            reset() {
              let Y = new Map(k.blockers);
              Y.set(O, Wr), Z({
                blockers: Y
              });
            }
          });
          return;
        }
        return Ie(N, E);
      }), n) {
        lv(t, J);
        let x = () => av(t, J);
        t.addEventListener("pagehide", x), de = () => t.removeEventListener("pagehide", x);
      }
      return k.initialized || Ie(Ue.Pop, k.location, {
        initialHydration: true
      }), T;
    }
    function ye() {
      g && g(), de && de(), y.clear(), M && M.abort(), k.fetchers.forEach((x, N) => Il(N)), k.blockers.forEach((x, N) => eu(N));
    }
    function B(x) {
      return y.add(x), () => y.delete(x);
    }
    function Z(x, N) {
      N === void 0 && (N = {}), k = Ee({}, k, x);
      let E = [], P = [];
      d.v7_fetcherPersist && k.fetchers.forEach((O, G) => {
        O.state === "idle" && (ue.has(G) ? P.push(G) : E.push(G));
      }), ue.forEach((O) => {
        !k.fetchers.has(O) && !D.has(O) && P.push(O);
      }), [
        ...y
      ].forEach((O) => O(k, {
        deletedFetchers: P,
        viewTransitionOpts: N.viewTransitionOpts,
        flushSync: N.flushSync === true
      })), d.v7_fetcherPersist ? (E.forEach((O) => k.fetchers.delete(O)), P.forEach((O) => Il(O))) : P.forEach((O) => ue.delete(O));
    }
    function L(x, N, E) {
      var P, O;
      let { flushSync: G } = E === void 0 ? {} : E, Y = k.actionData != null && k.navigation.formMethod != null && Lt(k.navigation.formMethod) && k.navigation.state === "loading" && ((P = x.state) == null ? void 0 : P._isRedirect) !== true, U;
      N.actionData ? Object.keys(N.actionData).length > 0 ? U = N.actionData : U = null : Y ? U = k.actionData : U = null;
      let F = N.loaderData ? _c(k.loaderData, N.loaderData, N.matches || [], N.errors) : k.loaderData, A = k.blockers;
      A.size > 0 && (A = new Map(A), A.forEach((oe, Ge) => A.set(Ge, Wr)));
      let W = $ === true || k.navigation.formMethod != null && Lt(k.navigation.formMethod) && ((O = x.state) == null ? void 0 : O._isRedirect) !== true;
      i && (o = i, i = void 0), se || R === Ue.Pop || (R === Ue.Push ? e.history.push(x, x.state) : R === Ue.Replace && e.history.replace(x, x.state));
      let q;
      if (R === Ue.Pop) {
        let oe = J.get(k.location.pathname);
        oe && oe.has(x.pathname) ? q = {
          currentLocation: k.location,
          nextLocation: x
        } : J.has(x.pathname) && (q = {
          currentLocation: x,
          nextLocation: k.location
        });
      } else if (K) {
        let oe = J.get(k.location.pathname);
        oe ? oe.add(x.pathname) : (oe = /* @__PURE__ */ new Set([
          x.pathname
        ]), J.set(k.location.pathname, oe)), q = {
          currentLocation: k.location,
          nextLocation: x
        };
      }
      Z(Ee({}, N, {
        actionData: U,
        loaderData: F,
        historyAction: R,
        location: x,
        initialized: true,
        navigation: Qo,
        revalidation: "idle",
        restoreScrollPosition: ru(x, N.matches || k.matches),
        preventScrollReset: W,
        blockers: A
      }), {
        viewTransitionOpts: q,
        flushSync: G === true
      }), R = Ue.Pop, $ = false, K = false, se = false, _e = false, He = [];
    }
    async function I(x, N) {
      if (typeof x == "number") {
        e.history.go(x);
        return;
      }
      let E = Ws(k.location, k.matches, s, d.v7_prependBasename, x, d.v7_relativeSplatPath, N == null ? void 0 : N.fromRouteId, N == null ? void 0 : N.relative), { path: P, submission: O, error: G } = xc(d.v7_normalizeFormMethod, false, E, N), Y = k.location, U = Cl(k.location, P, N && N.state);
      U = Ee({}, U, e.history.encodeLocation(U));
      let F = N && N.replace != null ? N.replace : void 0, A = Ue.Push;
      F === true ? A = Ue.Replace : F === false || O != null && Lt(O.formMethod) && O.formAction === k.location.pathname + k.location.search && (A = Ue.Replace);
      let W = N && "preventScrollReset" in N ? N.preventScrollReset === true : void 0, q = (N && N.flushSync) === true, oe = tu({
        currentLocation: Y,
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
            }), I(x, N);
          },
          reset() {
            let Ge = new Map(k.blockers);
            Ge.set(oe, Wr), Z({
              blockers: Ge
            });
          }
        });
        return;
      }
      return await Ie(A, U, {
        submission: O,
        pendingError: G,
        preventScrollReset: W,
        replace: N && N.replace,
        enableViewTransition: N && N.viewTransition,
        flushSync: q
      });
    }
    function me() {
      if (Ut(), Z({
        revalidation: "loading"
      }), k.navigation.state !== "submitting") {
        if (k.navigation.state === "idle") {
          Ie(k.historyAction, k.location, {
            startUninterruptedRevalidation: true
          });
          return;
        }
        Ie(R || k.historyAction, k.navigation.location, {
          overrideNavigation: k.navigation,
          enableViewTransition: K === true
        });
      }
    }
    async function Ie(x, N, E) {
      M && M.abort(), M = null, R = x, se = (E && E.startUninterruptedRevalidation) === true, Pm(k.location, k.matches), $ = (E && E.preventScrollReset) === true, K = (E && E.enableViewTransition) === true;
      let P = i || o, O = E && E.overrideNavigation, G = E != null && E.initialHydration && k.matches && k.matches.length > 0 && !h ? k.matches : On(P, N, s), Y = (E && E.flushSync) === true;
      if (G && k.initialized && !_e && Jg(k.location, N) && !(E && E.submission && Lt(E.submission.formMethod))) {
        L(N, {
          matches: G
        }, {
          flushSync: Y
        });
        return;
      }
      let U = Al(G, P, N.pathname);
      if (U.active && U.matches && (G = U.matches), !G) {
        let { error: we, notFoundMatches: pe, route: De } = go(N.pathname);
        L(N, {
          matches: pe,
          loaderData: {},
          errors: {
            [De.id]: we
          }
        }, {
          flushSync: Y
        });
        return;
      }
      M = new AbortController();
      let F = nr(e.history, N, M.signal, E && E.submission), A;
      if (E && E.pendingError) A = [
        In(G).route.id,
        {
          type: he.error,
          error: E.pendingError
        }
      ];
      else if (E && E.submission && Lt(E.submission.formMethod)) {
        let we = await tt(F, N, E.submission, G, U.active, {
          replace: E.replace,
          flushSync: Y
        });
        if (we.shortCircuited) return;
        if (we.pendingActionResult) {
          let [pe, De] = we.pendingActionResult;
          if (vt(De) && _l(De.error) && De.error.status === 404) {
            M = null, L(N, {
              matches: we.matches,
              loaderData: {},
              errors: {
                [pe]: De.error
              }
            });
            return;
          }
        }
        G = we.matches || G, A = we.pendingActionResult, O = Go(N, E.submission), Y = false, U.active = false, F = nr(e.history, F.url, F.signal);
      }
      let { shortCircuited: W, matches: q, loaderData: oe, errors: Ge } = await Qe(F, N, G, U.active, O, E && E.submission, E && E.fetcherSubmission, E && E.replace, E && E.initialHydration === true, Y, A);
      W || (M = null, L(N, Ee({
        matches: q || G
      }, Rc(A), {
        loaderData: oe,
        errors: Ge
      })));
    }
    async function tt(x, N, E, P, O, G) {
      G === void 0 && (G = {}), Ut();
      let Y = nv(N, E);
      if (Z({
        navigation: Y
      }, {
        flushSync: G.flushSync === true
      }), O) {
        let A = await Ul(P, N.pathname, x.signal);
        if (A.type === "aborted") return {
          shortCircuited: true
        };
        if (A.type === "error") {
          let W = In(A.partialMatches).route.id;
          return {
            matches: A.partialMatches,
            pendingActionResult: [
              W,
              {
                type: he.error,
                error: A.error
              }
            ]
          };
        } else if (A.matches) P = A.matches;
        else {
          let { notFoundMatches: W, error: q, route: oe } = go(N.pathname);
          return {
            matches: W,
            pendingActionResult: [
              oe.id,
              {
                type: he.error,
                error: q
              }
            ]
          };
        }
      }
      let U, F = Zr(P, N);
      if (!F.route.action && !F.route.lazy) U = {
        type: he.error,
        error: st(405, {
          method: x.method,
          pathname: N.pathname,
          routeId: F.route.id
        })
      };
      else if (U = (await re("action", k, x, [
        F
      ], P, null))[F.route.id], x.signal.aborted) return {
        shortCircuited: true
      };
      if ($n(U)) {
        let A;
        return G && G.replace != null ? A = G.replace : A = jc(U.response.headers.get("Location"), new URL(x.url), s, e.history) === k.location.pathname + k.location.search, await H(x, U, true, {
          submission: E,
          replace: A
        }), {
          shortCircuited: true
        };
      }
      if (hn(U)) throw st(400, {
        type: "defer-action"
      });
      if (vt(U)) {
        let A = In(P, F.route.id);
        return (G && G.replace) !== true && (R = Ue.Push), {
          matches: P,
          pendingActionResult: [
            A.route.id,
            U
          ]
        };
      }
      return {
        matches: P,
        pendingActionResult: [
          F.route.id,
          U
        ]
      };
    }
    async function Qe(x, N, E, P, O, G, Y, U, F, A, W) {
      let q = O || Go(N, G), oe = G || Y || bc(q), Ge = !se && (!d.v7_partialHydration || !F);
      if (P) {
        if (Ge) {
          let Me = ze(W);
          Z(Ee({
            navigation: q
          }, Me !== void 0 ? {
            actionData: Me
          } : {}), {
            flushSync: A
          });
        }
        let ce = await Ul(E, N.pathname, x.signal);
        if (ce.type === "aborted") return {
          shortCircuited: true
        };
        if (ce.type === "error") {
          let Me = In(ce.partialMatches).route.id;
          return {
            matches: ce.partialMatches,
            loaderData: {},
            errors: {
              [Me]: ce.error
            }
          };
        } else if (ce.matches) E = ce.matches;
        else {
          let { error: Me, notFoundMatches: qn, route: Or } = go(N.pathname);
          return {
            matches: qn,
            loaderData: {},
            errors: {
              [Or.id]: Me
            }
          };
        }
      }
      let we = i || o, [pe, De] = Sc(e.history, k, E, oe, N, d.v7_partialHydration && F === true, d.v7_skipActionErrorRevalidation, _e, He, mt, ue, fe, ee, we, s, W);
      if (vo((ce) => !(E && E.some((Me) => Me.route.id === ce)) || pe && pe.some((Me) => Me.route.id === ce)), V = ++Q, pe.length === 0 && De.length === 0) {
        let ce = Zi();
        return L(N, Ee({
          matches: E,
          loaderData: {},
          errors: W && vt(W[1]) ? {
            [W[0]]: W[1].error
          } : null
        }, Rc(W), ce ? {
          fetchers: new Map(k.fetchers)
        } : {}), {
          flushSync: A
        }), {
          shortCircuited: true
        };
      }
      if (Ge) {
        let ce = {};
        if (!P) {
          ce.navigation = q;
          let Me = ze(W);
          Me !== void 0 && (ce.actionData = Me);
        }
        De.length > 0 && (ce.fetchers = pt(De)), Z(ce, {
          flushSync: A
        });
      }
      De.forEach((ce) => {
        an(ce.key), ce.controller && D.set(ce.key, ce.controller);
      });
      let Zn = () => De.forEach((ce) => an(ce.key));
      M && M.signal.addEventListener("abort", Zn);
      let { loaderResults: Mr, fetcherResults: Gt } = await Tt(k, E, pe, De, x);
      if (x.signal.aborted) return {
        shortCircuited: true
      };
      M && M.signal.removeEventListener("abort", Zn), De.forEach((ce) => D.delete(ce.key));
      let $t = aa(Mr);
      if ($t) return await H(x, $t.result, true, {
        replace: U
      }), {
        shortCircuited: true
      };
      if ($t = aa(Gt), $t) return ee.add($t.key), await H(x, $t.result, true, {
        replace: U
      }), {
        shortCircuited: true
      };
      let { loaderData: yo, errors: Lr } = Cc(k, E, Mr, W, De, Gt, je);
      je.forEach((ce, Me) => {
        ce.subscribe((qn) => {
          (qn || ce.done) && je.delete(Me);
        });
      }), d.v7_partialHydration && F && k.errors && (Lr = Ee({}, k.errors, Lr));
      let bn = Zi(), $l = qi(V), Fl = bn || $l || De.length > 0;
      return Ee({
        matches: E,
        loaderData: yo,
        errors: Lr
      }, Fl ? {
        fetchers: new Map(k.fetchers)
      } : {});
    }
    function ze(x) {
      if (x && !vt(x[1])) return {
        [x[0]]: x[1].data
      };
      if (k.actionData) return Object.keys(k.actionData).length === 0 ? null : k.actionData;
    }
    function pt(x) {
      return x.forEach((N) => {
        let E = k.fetchers.get(N.key), P = Hr(void 0, E ? E.data : void 0);
        k.fetchers.set(N.key, P);
      }), new Map(k.fetchers);
    }
    function ot(x, N, E, P) {
      if (r) throw new Error("router.fetch() was called during the server render, but it shouldn't be. You are likely calling a useFetcher() method in the body of your component. Try moving it to a useEffect or a callback.");
      an(x);
      let O = (P && P.flushSync) === true, G = i || o, Y = Ws(k.location, k.matches, s, d.v7_prependBasename, E, d.v7_relativeSplatPath, N, P == null ? void 0 : P.relative), U = On(G, Y, s), F = Al(U, G, Y);
      if (F.active && F.matches && (U = F.matches), !U) {
        Pt(x, N, st(404, {
          pathname: Y
        }), {
          flushSync: O
        });
        return;
      }
      let { path: A, submission: W, error: q } = xc(d.v7_normalizeFormMethod, true, Y, P);
      if (q) {
        Pt(x, N, q, {
          flushSync: O
        });
        return;
      }
      let oe = Zr(U, A), Ge = (P && P.preventScrollReset) === true;
      if (W && Lt(W.formMethod)) {
        ln(x, N, A, oe, U, F.active, O, Ge, W);
        return;
      }
      fe.set(x, {
        routeId: N,
        path: A
      }), xe(x, N, A, oe, U, F.active, O, Ge, W);
    }
    async function ln(x, N, E, P, O, G, Y, U, F) {
      Ut(), fe.delete(x);
      function A(Ae) {
        if (!Ae.route.action && !Ae.route.lazy) {
          let er = st(405, {
            method: F.formMethod,
            pathname: E,
            routeId: N
          });
          return Pt(x, N, er, {
            flushSync: Y
          }), true;
        }
        return false;
      }
      if (!G && A(P)) return;
      let W = k.fetchers.get(x);
      ht(x, rv(F, W), {
        flushSync: Y
      });
      let q = new AbortController(), oe = nr(e.history, E, q.signal, F);
      if (G) {
        let Ae = await Ul(O, new URL(oe.url).pathname, oe.signal, x);
        if (Ae.type === "aborted") return;
        if (Ae.type === "error") {
          Pt(x, N, Ae.error, {
            flushSync: Y
          });
          return;
        } else if (Ae.matches) {
          if (O = Ae.matches, P = Zr(O, E), A(P)) return;
        } else {
          Pt(x, N, st(404, {
            pathname: E
          }), {
            flushSync: Y
          });
          return;
        }
      }
      D.set(x, q);
      let Ge = Q, pe = (await re("action", k, oe, [
        P
      ], O, x))[P.route.id];
      if (oe.signal.aborted) {
        D.get(x) === q && D.delete(x);
        return;
      }
      if (d.v7_fetcherPersist && ue.has(x)) {
        if ($n(pe) || vt(pe)) {
          ht(x, sn(void 0));
          return;
        }
      } else {
        if ($n(pe)) if (D.delete(x), V > Ge) {
          ht(x, sn(void 0));
          return;
        } else return ee.add(x), ht(x, Hr(F)), H(oe, pe, false, {
          fetcherSubmission: F,
          preventScrollReset: U
        });
        if (vt(pe)) {
          Pt(x, N, pe.error);
          return;
        }
      }
      if (hn(pe)) throw st(400, {
        type: "defer-action"
      });
      let De = k.navigation.location || k.location, Zn = nr(e.history, De, q.signal), Mr = i || o, Gt = k.navigation.state !== "idle" ? On(Mr, k.navigation.location, s) : k.matches;
      ae(Gt, "Didn't find any matches after fetcher action");
      let $t = ++Q;
      le.set(x, $t);
      let yo = Hr(F, pe.data);
      k.fetchers.set(x, yo);
      let [Lr, bn] = Sc(e.history, k, Gt, F, De, false, d.v7_skipActionErrorRevalidation, _e, He, mt, ue, fe, ee, Mr, s, [
        P.route.id,
        pe
      ]);
      bn.filter((Ae) => Ae.key !== x).forEach((Ae) => {
        let er = Ae.key, lu = k.fetchers.get(er), Mm = Hr(void 0, lu ? lu.data : void 0);
        k.fetchers.set(er, Mm), an(er), Ae.controller && D.set(er, Ae.controller);
      }), Z({
        fetchers: new Map(k.fetchers)
      });
      let $l = () => bn.forEach((Ae) => an(Ae.key));
      q.signal.addEventListener("abort", $l);
      let { loaderResults: Fl, fetcherResults: ce } = await Tt(k, Gt, Lr, bn, Zn);
      if (q.signal.aborted) return;
      q.signal.removeEventListener("abort", $l), le.delete(x), D.delete(x), bn.forEach((Ae) => D.delete(Ae.key));
      let Me = aa(Fl);
      if (Me) return H(Zn, Me.result, false, {
        preventScrollReset: U
      });
      if (Me = aa(ce), Me) return ee.add(Me.key), H(Zn, Me.result, false, {
        preventScrollReset: U
      });
      let { loaderData: qn, errors: Or } = Cc(k, Gt, Fl, void 0, bn, ce, je);
      if (k.fetchers.has(x)) {
        let Ae = sn(pe.data);
        k.fetchers.set(x, Ae);
      }
      qi($t), k.navigation.state === "loading" && $t > V ? (ae(R, "Expected pending action"), M && M.abort(), L(k.navigation.location, {
        matches: Gt,
        loaderData: qn,
        errors: Or,
        fetchers: new Map(k.fetchers)
      })) : (Z({
        errors: Or,
        loaderData: _c(k.loaderData, qn, Gt, Or),
        fetchers: new Map(k.fetchers)
      }), _e = false);
    }
    async function xe(x, N, E, P, O, G, Y, U, F) {
      let A = k.fetchers.get(x);
      ht(x, Hr(F, A ? A.data : void 0), {
        flushSync: Y
      });
      let W = new AbortController(), q = nr(e.history, E, W.signal);
      if (G) {
        let pe = await Ul(O, new URL(q.url).pathname, q.signal, x);
        if (pe.type === "aborted") return;
        if (pe.type === "error") {
          Pt(x, N, pe.error, {
            flushSync: Y
          });
          return;
        } else if (pe.matches) O = pe.matches, P = Zr(O, E);
        else {
          Pt(x, N, st(404, {
            pathname: E
          }), {
            flushSync: Y
          });
          return;
        }
      }
      D.set(x, W);
      let oe = Q, we = (await re("loader", k, q, [
        P
      ], O, x))[P.route.id];
      if (hn(we) && (we = await Hi(we, q.signal, true) || we), D.get(x) === W && D.delete(x), !q.signal.aborted) {
        if (ue.has(x)) {
          ht(x, sn(void 0));
          return;
        }
        if ($n(we)) if (V > oe) {
          ht(x, sn(void 0));
          return;
        } else {
          ee.add(x), await H(q, we, false, {
            preventScrollReset: U
          });
          return;
        }
        if (vt(we)) {
          Pt(x, N, we.error);
          return;
        }
        ae(!hn(we), "Unhandled fetcher deferred data"), ht(x, sn(we.data));
      }
    }
    async function H(x, N, E, P) {
      let { submission: O, fetcherSubmission: G, preventScrollReset: Y, replace: U } = P === void 0 ? {} : P;
      N.response.headers.has("X-Remix-Revalidate") && (_e = true);
      let F = N.response.headers.get("Location");
      ae(F, "Expected a Location header on the redirect Response"), F = jc(F, new URL(x.url), s, e.history);
      let A = Cl(k.location, F, {
        _isRedirect: true
      });
      if (n) {
        let pe = false;
        if (N.response.headers.has("X-Remix-Reload-Document")) pe = true;
        else if (Wi.test(F)) {
          const De = e.history.createURL(F);
          pe = De.origin !== t.location.origin || Ll(De.pathname, s) == null;
        }
        if (pe) {
          U ? t.location.replace(F) : t.location.assign(F);
          return;
        }
      }
      M = null;
      let W = U === true || N.response.headers.has("X-Remix-Replace") ? Ue.Replace : Ue.Push, { formMethod: q, formAction: oe, formEncType: Ge } = k.navigation;
      !O && !G && q && oe && Ge && (O = bc(k.navigation));
      let we = O || G;
      if (zg.has(N.response.status) && we && Lt(we.formMethod)) await Ie(W, A, {
        submission: Ee({}, we, {
          formAction: F
        }),
        preventScrollReset: Y || $,
        enableViewTransition: E ? K : void 0
      });
      else {
        let pe = Go(A, O);
        await Ie(W, A, {
          overrideNavigation: pe,
          fetcherSubmission: G,
          preventScrollReset: Y || $,
          enableViewTransition: E ? K : void 0
        });
      }
    }
    async function re(x, N, E, P, O, G) {
      let Y, U = {};
      try {
        Y = await Hg(c, x, N, E, P, O, G, a, l);
      } catch (F) {
        return P.forEach((A) => {
          U[A.route.id] = {
            type: he.error,
            error: F
          };
        }), U;
      }
      for (let [F, A] of Object.entries(Y)) if (Xg(A)) {
        let W = A.result;
        U[F] = {
          type: he.redirect,
          response: Kg(W, E, F, O, s, d.v7_relativeSplatPath)
        };
      } else U[F] = await Gg(A);
      return U;
    }
    async function Tt(x, N, E, P, O) {
      let G = x.matches, Y = re("loader", x, O, E, N, null), U = Promise.all(P.map(async (W) => {
        if (W.matches && W.match && W.controller) {
          let oe = (await re("loader", x, nr(e.history, W.path, W.controller.signal), [
            W.match
          ], W.matches, W.key))[W.match.route.id];
          return {
            [W.key]: oe
          };
        } else return Promise.resolve({
          [W.key]: {
            type: he.error,
            error: st(404, {
              pathname: W.path
            })
          }
        });
      })), F = await Y, A = (await U).reduce((W, q) => Object.assign(W, q), {});
      return await Promise.all([
        ev(N, F, O.signal, G, x.loaderData),
        tv(N, A, P)
      ]), {
        loaderResults: F,
        fetcherResults: A
      };
    }
    function Ut() {
      _e = true, He.push(...vo()), fe.forEach((x, N) => {
        D.has(N) && mt.add(N), an(N);
      });
    }
    function ht(x, N, E) {
      E === void 0 && (E = {}), k.fetchers.set(x, N), Z({
        fetchers: new Map(k.fetchers)
      }, {
        flushSync: (E && E.flushSync) === true
      });
    }
    function Pt(x, N, E, P) {
      P === void 0 && (P = {});
      let O = In(k.matches, N);
      Il(x), Z({
        errors: {
          [O.route.id]: E
        },
        fetchers: new Map(k.fetchers)
      }, {
        flushSync: (P && P.flushSync) === true
      });
    }
    function Ol(x) {
      return X.set(x, (X.get(x) || 0) + 1), ue.has(x) && ue.delete(x), k.fetchers.get(x) || Ag;
    }
    function Il(x) {
      let N = k.fetchers.get(x);
      D.has(x) && !(N && N.state === "loading" && le.has(x)) && an(x), fe.delete(x), le.delete(x), ee.delete(x), d.v7_fetcherPersist && ue.delete(x), mt.delete(x), k.fetchers.delete(x);
    }
    function _m(x) {
      let N = (X.get(x) || 0) - 1;
      N <= 0 ? (X.delete(x), ue.add(x), d.v7_fetcherPersist || Il(x)) : X.set(x, N), Z({
        fetchers: new Map(k.fetchers)
      });
    }
    function an(x) {
      let N = D.get(x);
      N && (N.abort(), D.delete(x));
    }
    function Xi(x) {
      for (let N of x) {
        let E = Ol(N), P = sn(E.data);
        k.fetchers.set(N, P);
      }
    }
    function Zi() {
      let x = [], N = false;
      for (let E of ee) {
        let P = k.fetchers.get(E);
        ae(P, "Expected fetcher: " + E), P.state === "loading" && (ee.delete(E), x.push(E), N = true);
      }
      return Xi(x), N;
    }
    function qi(x) {
      let N = [];
      for (let [E, P] of le) if (P < x) {
        let O = k.fetchers.get(E);
        ae(O, "Expected fetcher: " + E), O.state === "loading" && (an(E), le.delete(E), N.push(E));
      }
      return Xi(N), N.length > 0;
    }
    function Rm(x, N) {
      let E = k.blockers.get(x) || Wr;
      return te.get(x) !== N && te.set(x, N), E;
    }
    function eu(x) {
      k.blockers.delete(x), te.delete(x);
    }
    function zl(x, N) {
      let E = k.blockers.get(x) || Wr;
      ae(E.state === "unblocked" && N.state === "blocked" || E.state === "blocked" && N.state === "blocked" || E.state === "blocked" && N.state === "proceeding" || E.state === "blocked" && N.state === "unblocked" || E.state === "proceeding" && N.state === "unblocked", "Invalid blocker state transition: " + E.state + " -> " + N.state);
      let P = new Map(k.blockers);
      P.set(x, N), Z({
        blockers: P
      });
    }
    function tu(x) {
      let { currentLocation: N, nextLocation: E, historyAction: P } = x;
      if (te.size === 0) return;
      te.size > 1 && Kn(false, "A router only supports one blocker at a time");
      let O = Array.from(te.entries()), [G, Y] = O[O.length - 1], U = k.blockers.get(G);
      if (!(U && U.state === "proceeding") && Y({
        currentLocation: N,
        nextLocation: E,
        historyAction: P
      })) return G;
    }
    function go(x) {
      let N = st(404, {
        pathname: x
      }), E = i || o, { matches: P, route: O } = Tc(E);
      return vo(), {
        notFoundMatches: P,
        route: O,
        error: N
      };
    }
    function vo(x) {
      let N = [];
      return je.forEach((E, P) => {
        (!x || x(P)) && (E.cancel(), N.push(P), je.delete(P));
      }), N;
    }
    function Tm(x, N, E) {
      if (w = x, _ = N, S = E || null, !p && k.navigation === Qo) {
        p = true;
        let P = ru(k.location, k.matches);
        P != null && Z({
          restoreScrollPosition: P
        });
      }
      return () => {
        w = null, _ = null, S = null;
      };
    }
    function nu(x, N) {
      return S && S(x, N.map((P) => pg(P, k.loaderData))) || x.key;
    }
    function Pm(x, N) {
      if (w && _) {
        let E = nu(x, N);
        w[E] = _();
      }
    }
    function ru(x, N) {
      if (w) {
        let E = nu(x, N), P = w[E];
        if (typeof P == "number") return P;
      }
      return null;
    }
    function Al(x, N, E) {
      if (m) if (x) {
        if (Object.keys(x[0].params).length > 0) return {
          active: true,
          matches: xa(N, E, s, true)
        };
      } else return {
        active: true,
        matches: xa(N, E, s, true) || []
      };
      return {
        active: false,
        matches: null
      };
    }
    async function Ul(x, N, E, P) {
      if (!m) return {
        type: "success",
        matches: x
      };
      let O = x;
      for (; ; ) {
        let G = i == null, Y = i || o, U = a;
        try {
          await m({
            signal: E,
            path: N,
            matches: O,
            fetcherKey: P,
            patch: (W, q) => {
              E.aborted || Nc(W, q, Y, U, l);
            }
          });
        } catch (W) {
          return {
            type: "error",
            error: W,
            partialMatches: O
          };
        } finally {
          G && !E.aborted && (o = [
            ...o
          ]);
        }
        if (E.aborted) return {
          type: "aborted"
        };
        let F = On(Y, N, s);
        if (F) return {
          type: "success",
          matches: F
        };
        let A = xa(Y, N, s, true);
        if (!A || O.length === A.length && O.every((W, q) => W.route.id === A[q].route.id)) return {
          type: "success",
          matches: null
        };
        O = A;
      }
    }
    function bm(x) {
      a = {}, i = Ha(x, l, void 0, a);
    }
    function Dm(x, N) {
      let E = i == null;
      Nc(x, N, i || o, a, l), E && (o = [
        ...o
      ], Z({}));
    }
    return T = {
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
      initialize: ve,
      subscribe: B,
      enableScrollRestoration: Tm,
      navigate: I,
      fetch: ot,
      revalidate: me,
      createHref: (x) => e.history.createHref(x),
      encodeLocation: (x) => e.history.encodeLocation(x),
      getFetcher: Ol,
      deleteFetcher: _m,
      dispose: ye,
      getBlocker: Rm,
      deleteBlocker: eu,
      patchRoutes: Dm,
      _internalFetchControllers: D,
      _internalActiveDeferreds: je,
      _internalSetRoutes: bm
    }, T;
  }
  function Fg(e) {
    return e != null && ("formData" in e && e.formData != null || "body" in e && e.body !== void 0);
  }
  function Ws(e, t, n, r, l, a, o, i) {
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
        let g = new URLSearchParams(m.search), y = g.getAll("index");
        g.delete("index"), y.filter((S) => S).forEach((S) => g.append("index", S));
        let w = g.toString();
        m.search = w ? "?" + w : "";
      }
    }
    return r && n !== "/" && (m.pathname = m.pathname === "/" ? n : jn([
      n,
      m.pathname
    ])), Ml(m);
  }
  function xc(e, t, n, r) {
    if (!r || !Fg(r)) return {
      path: n
    };
    if (r.formMethod && !qg(r.formMethod)) return {
      path: n,
      error: st(405, {
        method: r.formMethod
      })
    };
    let l = () => ({
      path: n,
      error: st(400, {
        type: "invalid-body"
      })
    }), a = r.formMethod || "get", o = e ? a.toUpperCase() : a.toLowerCase(), i = hm(n);
    if (r.body !== void 0) {
      if (r.formEncType === "text/plain") {
        if (!Lt(o)) return l();
        let g = typeof r.body == "string" ? r.body : r.body instanceof FormData || r.body instanceof URLSearchParams ? Array.from(r.body.entries()).reduce((y, w) => {
          let [S, _] = w;
          return "" + y + S + "=" + _ + `
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
        if (!Lt(o)) return l();
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
    if (Lt(m.formMethod)) return {
      path: n,
      submission: m
    };
    let d = Pn(n);
    return t && d.search && Qi(d.search) && s.append("index", ""), d.search = "?" + s, {
      path: Ml(d),
      submission: m
    };
  }
  function wc(e, t, n) {
    n === void 0 && (n = false);
    let r = e.findIndex((l) => l.route.id === t);
    return r >= 0 ? e.slice(0, n ? r + 1 : r) : e;
  }
  function Sc(e, t, n, r, l, a, o, i, s, c, m, d, g, y, w, S) {
    let _ = S ? vt(S[1]) ? S[1].error : S[1].data : void 0, p = e.createURL(t.location), f = e.createURL(l), h = n;
    a && t.errors ? h = wc(n, Object.keys(t.errors)[0], true) : S && vt(S[1]) && (h = wc(n, S[0]));
    let j = S ? S[1].statusCode : void 0, C = o && j && j >= 400, T = h.filter((R, $) => {
      let { route: M } = R;
      if (M.lazy) return true;
      if (M.loader == null) return false;
      if (a) return Hs(M, t.loaderData, t.errors);
      if (Bg(t.loaderData, t.matches[$], R) || s.some((de) => de === R.route.id)) return true;
      let K = t.matches[$], J = R;
      return kc(R, Ee({
        currentUrl: p,
        currentParams: K.params,
        nextUrl: f,
        nextParams: J.params
      }, r, {
        actionResult: _,
        actionStatus: j,
        defaultShouldRevalidate: C ? false : i || p.pathname + p.search === f.pathname + f.search || p.search !== f.search || mm(K, J)
      }));
    }), k = [];
    return d.forEach((R, $) => {
      if (a || !n.some((se) => se.route.id === R.routeId) || m.has($)) return;
      let M = On(y, R.path, w);
      if (!M) {
        k.push({
          key: $,
          routeId: R.routeId,
          path: R.path,
          matches: null,
          match: null,
          controller: null
        });
        return;
      }
      let K = t.fetchers.get($), J = Zr(M, R.path), de = false;
      g.has($) ? de = false : c.has($) ? (c.delete($), de = true) : K && K.state !== "idle" && K.data === void 0 ? de = i : de = kc(J, Ee({
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
        routeId: R.routeId,
        path: R.path,
        matches: M,
        match: J,
        controller: new AbortController()
      });
    }), [
      T,
      k
    ];
  }
  function Hs(e, t, n) {
    if (e.lazy) return true;
    if (!e.loader) return false;
    let r = t != null && t[e.id] !== void 0, l = n != null && n[e.id] !== void 0;
    return !r && l ? false : typeof e.loader == "function" && e.loader.hydrate === true ? true : !r && !l;
  }
  function Bg(e, t, n) {
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
      Kn(!s, 'Route "' + l.id + '" has a static property "' + o + '" defined but its lazy function is also returning a value for this property. ' + ('The lazy route property "' + o + '" will be ignored.')), !s && !fg.has(o) && (a[o] = r[o]);
    }
    Object.assign(l, a), Object.assign(l, Ee({}, t(l), {
      lazy: void 0
    }));
  }
  async function Wg(e) {
    let { matches: t } = e, n = t.filter((l) => l.shouldLoad);
    return (await Promise.all(n.map((l) => l.resolve()))).reduce((l, a, o) => Object.assign(l, {
      [n[o].route.id]: a
    }), {});
  }
  async function Hg(e, t, n, r, l, a, o, i, s, c) {
    let m = a.map((y) => y.route.lazy ? Vg(y.route, s, i) : void 0), d = a.map((y, w) => {
      let S = m[w], _ = l.some((f) => f.route.id === y.route.id);
      return Ee({}, y, {
        shouldLoad: _,
        resolve: async (f) => (f && r.method === "GET" && (y.route.lazy || y.route.loader) && (_ = true), _ ? Qg(t, r, y, S, f, c) : Promise.resolve({
          type: he.data,
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
  async function Qg(e, t, n, r, l, a) {
    let o, i, s = (c) => {
      let m, d = new Promise((w, S) => m = S);
      i = () => m(), t.signal.addEventListener("abort", i);
      let g = (w) => typeof c != "function" ? Promise.reject(new Error("You cannot call the handler for a route which defines a boolean " + ('"' + e + '" [routeId: ' + n.route.id + "]"))) : c({
        request: t,
        params: n.params,
        context: a
      }, ...w !== void 0 ? [
        w
      ] : []), y = (async () => {
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
        y,
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
        throw st(405, {
          method: t.method,
          pathname: d,
          routeId: n.route.id
        });
      } else return {
        type: he.data,
        result: void 0
      };
      else if (c) o = await s(c);
      else {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw st(404, {
          pathname: d
        });
      }
      ae(o.result !== void 0, "You defined " + (e === "action" ? "an action" : "a loader") + " for route " + ('"' + n.route.id + "\" but didn't return anything from your `" + e + "` ") + "function. Please return a value or `null`.");
    } catch (c) {
      return {
        type: he.error,
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
          type: he.error,
          error: g
        };
      }
      return n === he.error ? {
        type: he.error,
        error: new Qa(t.status, t.statusText, d),
        statusCode: t.status,
        headers: t.headers
      } : {
        type: he.data,
        data: d,
        statusCode: t.status,
        headers: t.headers
      };
    }
    if (n === he.error) {
      if (Pc(t)) {
        var r, l;
        if (t.data instanceof Error) {
          var a, o;
          return {
            type: he.error,
            error: t.data,
            statusCode: (a = t.init) == null ? void 0 : a.status,
            headers: (o = t.init) != null && o.headers ? new Headers(t.init.headers) : void 0
          };
        }
        return {
          type: he.error,
          error: new Qa(((r = t.init) == null ? void 0 : r.status) || 500, void 0, t.data),
          statusCode: _l(t) ? t.status : void 0,
          headers: (l = t.init) != null && l.headers ? new Headers(t.init.headers) : void 0
        };
      }
      return {
        type: he.error,
        error: t,
        statusCode: _l(t) ? t.status : void 0
      };
    }
    if (Zg(t)) {
      var i, s;
      return {
        type: he.deferred,
        deferredData: t,
        statusCode: (i = t.init) == null ? void 0 : i.status,
        headers: ((s = t.init) == null ? void 0 : s.headers) && new Headers(t.init.headers)
      };
    }
    if (Pc(t)) {
      var c, m;
      return {
        type: he.data,
        data: t.data,
        statusCode: (c = t.init) == null ? void 0 : c.status,
        headers: (m = t.init) != null && m.headers ? new Headers(t.init.headers) : void 0
      };
    }
    return {
      type: he.data,
      data: t
    };
  }
  function Kg(e, t, n, r, l, a) {
    let o = e.headers.get("Location");
    if (ae(o, "Redirects returned/thrown from loaders/actions must have a Location header"), !Wi.test(o)) {
      let i = r.slice(0, r.findIndex((s) => s.route.id === n) + 1);
      o = Ws(new URL(t.url), i, l, true, o, a), e.headers.set("Location", o);
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
    if (Wi.test(e)) {
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
  function nr(e, t, n, r) {
    let l = e.createURL(hm(t)).toString(), a = {
      signal: n
    };
    if (r && Lt(r.formMethod)) {
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
  function Yg(e, t, n, r, l) {
    let a = {}, o = null, i, s = false, c = {}, m = n && vt(n[1]) ? n[1].error : void 0;
    return e.forEach((d) => {
      if (!(d.route.id in t)) return;
      let g = d.route.id, y = t[g];
      if (ae(!$n(y), "Cannot handle redirect results in processLoaderData"), vt(y)) {
        let w = y.error;
        m !== void 0 && (w = m, m = void 0), o = o || {};
        {
          let S = In(e, g);
          o[S.route.id] == null && (o[S.route.id] = w);
        }
        a[g] = void 0, s || (s = true, i = _l(y.error) ? y.error.status : 500), y.headers && (c[g] = y.headers);
      } else hn(y) ? (r.set(g, y.deferredData), a[g] = y.deferredData.data, y.statusCode != null && y.statusCode !== 200 && !s && (i = y.statusCode), y.headers && (c[g] = y.headers)) : (a[g] = y.data, y.statusCode && y.statusCode !== 200 && !s && (i = y.statusCode), y.headers && (c[g] = y.headers));
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
    let { loaderData: i, errors: s } = Yg(t, n, r, o);
    return l.forEach((c) => {
      let { key: m, match: d, controller: g } = c, y = a[m];
      if (ae(y, "Did not find corresponding fetcher result"), !(g && g.signal.aborted)) if (vt(y)) {
        let w = In(e.matches, d == null ? void 0 : d.route.id);
        s && s[w.route.id] || (s = Ee({}, s, {
          [w.route.id]: y.error
        })), e.fetchers.delete(m);
      } else if ($n(y)) ae(false, "Unhandled fetcher revalidation redirect");
      else if (hn(y)) ae(false, "Unhandled fetcher deferred data");
      else {
        let w = sn(y.data);
        e.fetchers.set(m, w);
      }
    }), {
      loaderData: i,
      errors: s
    };
  }
  function _c(e, t, n, r) {
    let l = Ee({}, t);
    for (let a of n) {
      let o = a.route.id;
      if (t.hasOwnProperty(o) ? t[o] !== void 0 && (l[o] = t[o]) : e[o] !== void 0 && a.route.loader && (l[o] = e[o]), r && r.hasOwnProperty(o)) break;
    }
    return l;
  }
  function Rc(e) {
    return e ? vt(e[1]) ? {
      actionData: {}
    } : {
      actionData: {
        [e[0]]: e[1].data
      }
    } : {};
  }
  function In(e, t) {
    return (t ? e.slice(0, e.findIndex((r) => r.route.id === t) + 1) : [
      ...e
    ]).reverse().find((r) => r.route.hasErrorBoundary === true) || e[0];
  }
  function Tc(e) {
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
  function st(e, t) {
    let { pathname: n, routeId: r, method: l, type: a, message: o } = t === void 0 ? {} : t, i = "Unknown Server Error", s = "Unknown @remix-run/router error";
    return e === 400 ? (i = "Bad Request", l && n && r ? s = "You made a " + l + ' request to "' + n + '" but ' + ('did not provide a `loader` for route "' + r + '", ') + "so there is no way to handle the request." : a === "defer-action" ? s = "defer() is not supported in actions" : a === "invalid-body" && (s = "Unable to encode submission body")) : e === 403 ? (i = "Forbidden", s = 'Route "' + r + '" does not match URL "' + n + '"') : e === 404 ? (i = "Not Found", s = 'No route matches URL "' + n + '"') : e === 405 && (i = "Method Not Allowed", l && n && r ? s = "You made a " + l.toUpperCase() + ' request to "' + n + '" but ' + ('did not provide an `action` for route "' + r + '", ') + "so there is no way to handle the request." : l && (s = 'Invalid request method "' + l.toUpperCase() + '"')), new Qa(e || 500, i, new Error(s), true);
  }
  function aa(e) {
    let t = Object.entries(e);
    for (let n = t.length - 1; n >= 0; n--) {
      let [r, l] = t[n];
      if ($n(l)) return {
        key: r,
        result: l
      };
    }
  }
  function hm(e) {
    let t = typeof e == "string" ? Pn(e) : e;
    return Ml(Ee({}, t, {
      hash: ""
    }));
  }
  function Jg(e, t) {
    return e.pathname !== t.pathname || e.search !== t.search ? false : e.hash === "" ? t.hash !== "" : e.hash === t.hash ? true : t.hash !== "";
  }
  function Xg(e) {
    return gm(e.result) && Ig.has(e.result.status);
  }
  function hn(e) {
    return e.type === he.deferred;
  }
  function vt(e) {
    return e.type === he.error;
  }
  function $n(e) {
    return (e && e.type) === he.redirect;
  }
  function Pc(e) {
    return typeof e == "object" && e != null && "type" in e && "data" in e && "init" in e && e.type === "DataWithResponseInit";
  }
  function Zg(e) {
    let t = e;
    return t && typeof t == "object" && typeof t.data == "object" && typeof t.subscribe == "function" && typeof t.cancel == "function" && typeof t.resolveData == "function";
  }
  function gm(e) {
    return e != null && typeof e.status == "number" && typeof e.statusText == "string" && typeof e.headers == "object" && typeof e.body < "u";
  }
  function qg(e) {
    return Og.has(e.toLowerCase());
  }
  function Lt(e) {
    return Mg.has(e.toLowerCase());
  }
  async function ev(e, t, n, r, l) {
    let a = Object.entries(t);
    for (let o = 0; o < a.length; o++) {
      let [i, s] = a[o], c = e.find((g) => (g == null ? void 0 : g.route.id) === i);
      if (!c) continue;
      let m = r.find((g) => g.route.id === c.route.id), d = m != null && !mm(m, c) && (l && l[c.route.id]) !== void 0;
      hn(s) && d && await Hi(s, n, false).then((g) => {
        g && (t[i] = g);
      });
    }
  }
  async function tv(e, t, n) {
    for (let r = 0; r < n.length; r++) {
      let { key: l, routeId: a, controller: o } = n[r], i = t[l];
      e.find((c) => (c == null ? void 0 : c.route.id) === a) && hn(i) && (ae(o, "Expected an AbortController for revalidating fetcher deferred result"), await Hi(i, o.signal, true).then((c) => {
        c && (t[l] = c);
      }));
    }
  }
  async function Hi(e, t, n) {
    if (n === void 0 && (n = false), !await e.deferredData.resolveData(t)) {
      if (n) try {
        return {
          type: he.data,
          data: e.deferredData.unwrappedData
        };
      } catch (l) {
        return {
          type: he.error,
          error: l
        };
      }
      return {
        type: he.data,
        data: e.deferredData.data
      };
    }
  }
  function Qi(e) {
    return new URLSearchParams(e).getAll("index").some((t) => t === "");
  }
  function Zr(e, t) {
    let n = typeof t == "string" ? Pn(t).search : t.search;
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
  function Go(e, t) {
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
  function nv(e, t) {
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
  function rv(e, t) {
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
  function sn(e) {
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
  function lv(e, t) {
    try {
      let n = e.sessionStorage.getItem(fm);
      if (n) {
        let r = JSON.parse(n);
        for (let [l, a] of Object.entries(r || {})) a && Array.isArray(a) && t.set(l, new Set(a || []));
      }
    } catch {
    }
  }
  function av(e, t) {
    if (t.size > 0) {
      let n = {};
      for (let [r, l] of t) n[r] = [
        ...l
      ];
      try {
        e.sessionStorage.setItem(fm, JSON.stringify(n));
      } catch (r) {
        Kn(false, "Failed to save applied view transitions in sessionStorage (" + r + ").");
      }
    }
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
  const fo = v.createContext(null), vm = v.createContext(null), mo = v.createContext(null), Gi = v.createContext(null), Xn = v.createContext({
    outlet: null,
    matches: [],
    isDataRoute: false
  }), ym = v.createContext(null);
  function po() {
    return v.useContext(Gi) != null;
  }
  function Ki() {
    return po() || ae(false), v.useContext(Gi).location;
  }
  function xm(e) {
    v.useContext(mo).static || v.useLayoutEffect(e);
  }
  function ho() {
    let { isDataRoute: e } = v.useContext(Xn);
    return e ? xv() : ov();
  }
  function ov() {
    po() || ae(false);
    let e = v.useContext(fo), { basename: t, future: n, navigator: r } = v.useContext(mo), { matches: l } = v.useContext(Xn), { pathname: a } = Ki(), o = JSON.stringify(um(l, n.v7_relativeSplatPath)), i = v.useRef(false);
    return xm(() => {
      i.current = true;
    }), v.useCallback(function(c, m) {
      if (m === void 0 && (m = {}), !i.current) return;
      if (typeof c == "number") {
        r.go(c);
        return;
      }
      let d = cm(c, JSON.parse(o), a, m.relative === "path");
      e == null && t !== "/" && (d.pathname = d.pathname === "/" ? t : jn([
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
  const sv = v.createContext(null);
  function iv(e) {
    let t = v.useContext(Xn).outlet;
    return t && v.createElement(sv.Provider, {
      value: e
    }, t);
  }
  function uv(e, t, n, r) {
    po() || ae(false);
    let { navigator: l } = v.useContext(mo), { matches: a } = v.useContext(Xn), o = a[a.length - 1], i = o ? o.params : {};
    o && o.pathname;
    let s = o ? o.pathnameBase : "/";
    o && o.route;
    let c = Ki(), m;
    m = c;
    let d = m.pathname || "/", g = d;
    if (s !== "/") {
      let S = s.replace(/^\//, "").split("/");
      g = "/" + d.replace(/^\//, "").split("/").slice(S.length).join("/");
    }
    let y = On(e, {
      pathname: g
    });
    return pv(y && y.map((S) => Object.assign({}, S, {
      params: Object.assign({}, i, S.params),
      pathname: jn([
        s,
        l.encodeLocation ? l.encodeLocation(S.pathname).pathname : S.pathname
      ]),
      pathnameBase: S.pathnameBase === "/" ? s : jn([
        s,
        l.encodeLocation ? l.encodeLocation(S.pathnameBase).pathname : S.pathnameBase
      ])
    })), a, n, r);
  }
  function cv() {
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
  const dv = v.createElement(cv, null);
  class fv extends v.Component {
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
      return this.state.error !== void 0 ? v.createElement(Xn.Provider, {
        value: this.props.routeContext
      }, v.createElement(ym.Provider, {
        value: this.state.error,
        children: this.props.component
      })) : this.props.children;
    }
  }
  function mv(e) {
    let { routeContext: t, match: n, children: r } = e, l = v.useContext(fo);
    return l && l.static && l.staticContext && (n.route.errorElement || n.route.ErrorBoundary) && (l.staticContext._deepestRenderedBoundaryId = n.route.id), v.createElement(Xn.Provider, {
      value: t
    }, r);
  }
  function pv(e, t, n, r) {
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
        let { loaderData: g, errors: y } = n, w = d.route.loader && g[d.route.id] === void 0 && (!y || y[d.route.id] === void 0);
        if (d.route.lazy || w) {
          s = true, c >= 0 ? o = o.slice(0, c + 1) : o = [
            o[0]
          ];
          break;
        }
      }
    }
    return o.reduceRight((m, d, g) => {
      let y, w = false, S = null, _ = null;
      n && (y = i && d.route.id ? i[d.route.id] : void 0, S = d.route.errorElement || dv, s && (c < 0 && g === 0 ? (wv("route-fallback"), w = true, _ = null) : c === g && (w = true, _ = d.route.hydrateFallbackElement || null)));
      let p = t.concat(o.slice(0, g + 1)), f = () => {
        let h;
        return y ? h = S : w ? h = _ : d.route.Component ? h = v.createElement(d.route.Component, null) : d.route.element ? h = d.route.element : h = m, v.createElement(mv, {
          match: d,
          routeContext: {
            outlet: m,
            matches: p,
            isDataRoute: n != null
          },
          children: h
        });
      };
      return n && (d.route.ErrorBoundary || d.route.errorElement || g === 0) ? v.createElement(fv, {
        location: n.location,
        revalidation: n.revalidation,
        component: S,
        error: y,
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
  function hv(e) {
    let t = v.useContext(fo);
    return t || ae(false), t;
  }
  function gv(e) {
    let t = v.useContext(vm);
    return t || ae(false), t;
  }
  function vv(e) {
    let t = v.useContext(Xn);
    return t || ae(false), t;
  }
  function km(e) {
    let t = vv(), n = t.matches[t.matches.length - 1];
    return n.route.id || ae(false), n.route.id;
  }
  function yv() {
    var e;
    let t = v.useContext(ym), n = gv(Sm.UseRouteError), r = km();
    return t !== void 0 ? t : (e = n.errors) == null ? void 0 : e[r];
  }
  function xv() {
    let { router: e } = hv(wm.UseNavigateStable), t = km(), n = v.useRef(false);
    return xm(() => {
      n.current = true;
    }), v.useCallback(function(l, a) {
      a === void 0 && (a = {}), n.current && (typeof l == "number" ? e.navigate(l) : e.navigate(l, Ga({
        fromRouteId: t
      }, a)));
    }, [
      e,
      t
    ]);
  }
  const Dc = {};
  function wv(e, t, n) {
    Dc[e] || (Dc[e] = true);
  }
  function Sv(e, t) {
    e == null ? void 0 : e.v7_startTransition, (e == null ? void 0 : e.v7_relativeSplatPath) === void 0 && (!t || t.v7_relativeSplatPath), t && (t.v7_fetcherPersist, t.v7_normalizeFormMethod, t.v7_partialHydration, t.v7_skipActionErrorRevalidation);
  }
  function kv(e) {
    return iv(e.context);
  }
  function Nv(e) {
    let { basename: t = "/", children: n = null, location: r, navigationType: l = Ue.Pop, navigator: a, static: o = false, future: i } = e;
    po() && ae(false);
    let s = t.replace(/^\/*/, "/"), c = v.useMemo(() => ({
      basename: s,
      navigator: a,
      static: o,
      future: Ga({
        v7_relativeSplatPath: false
      }, i)
    }), [
      s,
      i,
      a,
      o
    ]);
    typeof r == "string" && (r = Pn(r));
    let { pathname: m = "/", search: d = "", hash: g = "", state: y = null, key: w = "default" } = r, S = v.useMemo(() => {
      let _ = Ll(m, s);
      return _ == null ? null : {
        location: {
          pathname: _,
          search: d,
          hash: g,
          state: y,
          key: w
        },
        navigationType: l
      };
    }, [
      s,
      m,
      d,
      g,
      y,
      w,
      l
    ]);
    return S == null ? null : v.createElement(mo.Provider, {
      value: c
    }, v.createElement(Gi.Provider, {
      children: n,
      value: S
    }));
  }
  new Promise(() => {
  });
  function jv(e) {
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
  function Ka() {
    return Ka = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ka.apply(this, arguments);
  }
  const Ev = "6";
  try {
    window.__reactRouterVersion = Ev;
  } catch {
  }
  function Cv(e, t) {
    return $g({
      basename: void 0,
      future: Ka({}, void 0, {
        v7_prependBasename: true
      }),
      history: ug({
        window: void 0
      }),
      hydrationData: _v(),
      routes: e,
      mapRouteProperties: jv,
      dataStrategy: void 0,
      patchRoutesOnNavigation: void 0,
      window: void 0
    }).initialize();
  }
  function _v() {
    var e;
    let t = (e = window) == null ? void 0 : e.__staticRouterHydrationData;
    return t && t.errors && (t = Ka({}, t, {
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
  }), Pv = v.createContext(/* @__PURE__ */ new Map()), bv = "startTransition", Mc = Jm[bv], Dv = "flushSync", Lc = ig[Dv];
  function Mv(e) {
    Mc ? Mc(e) : e();
  }
  function Qr(e) {
    Lc ? Lc(e) : e();
  }
  class Lv {
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
  function Ov(e) {
    let { fallbackElement: t, router: n, future: r } = e, [l, a] = v.useState(n.state), [o, i] = v.useState(), [s, c] = v.useState({
      isTransitioning: false
    }), [m, d] = v.useState(), [g, y] = v.useState(), [w, S] = v.useState(), _ = v.useRef(/* @__PURE__ */ new Map()), { v7_startTransition: p } = r || {}, f = v.useCallback((R) => {
      p ? Mv(R) : R();
    }, [
      p
    ]), h = v.useCallback((R, $) => {
      let { deletedFetchers: M, flushSync: K, viewTransitionOpts: J } = $;
      R.fetchers.forEach((se, _e) => {
        se.data !== void 0 && _.current.set(_e, se.data);
      }), M.forEach((se) => _.current.delete(se));
      let de = n.window == null || n.window.document == null || typeof n.window.document.startViewTransition != "function";
      if (!J || de) {
        K ? Qr(() => a(R)) : f(() => a(R));
        return;
      }
      if (K) {
        Qr(() => {
          g && (m && m.resolve(), g.skipTransition()), c({
            isTransitioning: true,
            flushSync: true,
            currentLocation: J.currentLocation,
            nextLocation: J.nextLocation
          });
        });
        let se = n.window.document.startViewTransition(() => {
          Qr(() => a(R));
        });
        se.finished.finally(() => {
          Qr(() => {
            d(void 0), y(void 0), i(void 0), c({
              isTransitioning: false
            });
          });
        }), Qr(() => y(se));
        return;
      }
      g ? (m && m.resolve(), g.skipTransition(), S({
        state: R,
        currentLocation: J.currentLocation,
        nextLocation: J.nextLocation
      })) : (i(R), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: J.currentLocation,
        nextLocation: J.nextLocation
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
      s.isTransitioning && !s.flushSync && d(new Lv());
    }, [
      s
    ]), v.useEffect(() => {
      if (m && o && n.window) {
        let R = o, $ = m.promise, M = n.window.document.startViewTransition(async () => {
          f(() => a(R)), await $;
        });
        M.finished.finally(() => {
          d(void 0), y(void 0), i(void 0), c({
            isTransitioning: false
          });
        }), y(M);
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
      go: (R) => n.navigate(R),
      push: (R, $, M) => n.navigate(R, {
        state: $,
        preventScrollReset: M == null ? void 0 : M.preventScrollReset
      }),
      replace: (R, $, M) => n.navigate(R, {
        replace: true,
        state: $,
        preventScrollReset: M == null ? void 0 : M.preventScrollReset
      })
    }), [
      n
    ]), C = n.basename || "/", T = v.useMemo(() => ({
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
    return v.useEffect(() => Sv(r, n.future), [
      r,
      n.future
    ]), v.createElement(v.Fragment, null, v.createElement(fo.Provider, {
      value: T
    }, v.createElement(vm.Provider, {
      value: l
    }, v.createElement(Pv.Provider, {
      value: _.current
    }, v.createElement(Tv.Provider, {
      value: s
    }, v.createElement(Nv, {
      basename: C,
      location: l.location,
      navigationType: l.historyAction,
      navigator: j,
      future: k
    }, l.initialized || n.future.v7_partialHydration ? v.createElement(Iv, {
      routes: n.routes,
      future: n.future,
      state: l
    }) : t))))), null);
  }
  const Iv = v.memo(zv);
  function zv(e) {
    let { routes: t, future: n, state: r } = e;
    return uv(t, void 0, r, n);
  }
  var Oc;
  (function(e) {
    e.UseScrollRestoration = "useScrollRestoration", e.UseSubmit = "useSubmit", e.UseSubmitFetcher = "useSubmitFetcher", e.UseFetcher = "useFetcher", e.useViewTransitionState = "useViewTransitionState";
  })(Oc || (Oc = {}));
  var Ic;
  (function(e) {
    e.UseFetcher = "useFetcher", e.UseFetchers = "useFetchers", e.UseScrollRestoration = "useScrollRestoration";
  })(Ic || (Ic = {}));
  const Av = [
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
  ];
  function Uv() {
    const e = window.location;
    return `${e.protocol}//${e.host}`;
  }
  const $v = () => {
    const e = ho(), [t, n] = v.useState(/* @__PURE__ */ new Date()), [r, l] = v.useState(null);
    return v.useEffect(() => {
      const a = setInterval(() => n(/* @__PURE__ */ new Date()), 1e3);
      return () => clearInterval(a);
    }, []), v.useEffect(() => {
      const a = () => {
        fetch(`${Uv()}/health`).then((i) => i.json()).then(l).catch(() => l(null));
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
          children: Av.map((a) => u.jsxs("button", {
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
  const Qt = {
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
      const e = localStorage.getItem(Qt.index);
      return e ? JSON.parse(e) : [];
    } catch {
      return [];
    }
  }
  async function Fv() {
    const e = await jm("/api/conversations");
    return e && e.length > 0 ? (localStorage.setItem(Qt.index, JSON.stringify(e)), e) : Em();
  }
  function Bv(e) {
    localStorage.setItem(Qt.index, JSON.stringify(e)), Yi("/api/conversations", e);
  }
  function Gs(e) {
    try {
      const t = localStorage.getItem(Qt.messages(e));
      return t ? JSON.parse(t) : [];
    } catch {
      return [];
    }
  }
  async function Vv(e) {
    const t = await jm(`/api/conversations/${e}/messages`);
    return t && t.length > 0 ? (localStorage.setItem(Qt.messages(e), JSON.stringify(t)), t) : Gs(e);
  }
  function Cm(e, t) {
    localStorage.setItem(Qt.messages(e), JSON.stringify(t)), Yi(`/api/conversations/${e}/messages`, t);
  }
  function Wv(e) {
    localStorage.removeItem(Qt.messages(e)), Yi(`/api/conversations/${e}/delete`, {});
  }
  function Hv() {
    return localStorage.getItem(Qt.active);
  }
  function oa(e) {
    e === null ? localStorage.removeItem(Qt.active) : localStorage.setItem(Qt.active, e);
  }
  const Ks = () => typeof crypto.randomUUID == "function" ? crypto.randomUUID() : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (e) => {
    const t = Math.random() * 16 | 0;
    return (e === "x" ? t : t & 3 | 8).toString(16);
  }), Qv = /iPad|iPhone|iPod/.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  let rr = null;
  function Gv() {
    return rr && Date.now() - rr.ts < 3e5 ? Promise.resolve(rr) : navigator.geolocation ? new Promise((e) => {
      navigator.geolocation.getCurrentPosition((t) => {
        rr = {
          lat: t.coords.latitude,
          lng: t.coords.longitude,
          accuracy: t.coords.accuracy,
          ts: Date.now()
        }, e(rr);
      }, () => e(rr), {
        timeout: 5e3,
        maximumAge: 3e5
      });
    }) : Promise.resolve(null);
  }
  const Kv = `You are MAUDE \u2014 a local AI assistant running on Matt's DGX Spark, handling tasks that benefit from local execution, privacy, or when cloud access isn't available.

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
  function Ko() {
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
  function Yv(e = null) {
    const [t, n] = v.useState(() => e ? Gs(e) : []), [r, l] = v.useState(false), [a, o] = v.useState(() => localStorage.getItem("maude-model") || "claude-opus-4-20250514"), [i, s] = v.useState(() => localStorage.getItem("maude-autoroute") === "true"), c = v.useCallback((h) => {
      localStorage.setItem("maude-model", h), o(h);
    }, []), m = v.useCallback((h) => {
      localStorage.setItem("maude-autoroute", String(h)), s(h);
    }, []), d = v.useRef(a);
    d.current = a;
    const g = v.useRef(null), y = v.useRef(e), w = v.useRef(""), S = v.useRef(0);
    y.current = e, v.useEffect(() => {
      if (!e) {
        n([]);
        return;
      }
      n(Gs(e)), Vv(e).then((h) => {
        h.length > 0 && n(h);
      });
    }, [
      e
    ]), v.useEffect(() => {
      y.current && t.length > 0 && Cm(y.current, t);
    }, [
      t
    ]);
    const _ = v.useCallback(async (h, j) => {
      var _a2, _b, _c2;
      const C = j && j.length > 0;
      if (!h.trim() && !C || r) return;
      if (h.startsWith("/")) {
        const J = h.trim().toLowerCase();
        if (J === "/clear") {
          n([]);
          return;
        }
        if (J.startsWith("/model ")) {
          o(J.slice(7).trim());
          return;
        }
      }
      const T = h || (C ? "What do you see in this image?" : ""), k = {
        id: Ks(),
        role: "user",
        content: T,
        imageUrls: C ? j : void 0,
        timestamp: Date.now()
      };
      n((J) => [
        ...J,
        k
      ]), l(true);
      const R = d.current, $ = {
        id: Ks(),
        role: "assistant",
        content: "",
        model: R,
        timestamp: Date.now()
      };
      n((J) => [
        ...J,
        $
      ]);
      const M = new AbortController();
      g.current = M;
      let K = "";
      try {
        const J = t.filter((te) => te.role !== "system").slice(-20).map((te) => ({
          role: te.role,
          content: te.content
        }));
        let de = T;
        if (C) {
          const te = j.map((be) => `/home/mboard76/nvidia-workbench/terminal-llm/shared/${be.split("/").pop()}`);
          if (te.length === 1) de = `[Image attached: ${te[0]} \u2014 analyze it with view_image tool]

${T}`;
          else {
            const be = te.map((ve, ye) => `  ${ye + 1}. ${ve}`).join(`
`);
            de = `[${te.length} images attached \u2014 analyze each with view_image tool:
${be}]

${T}`;
          }
        }
        const se = await Gv(), _e = {
          model: R,
          messages: [
            {
              role: "system",
              content: Kv
            },
            ...J,
            {
              role: "user",
              content: de
            }
          ],
          stream: true,
          max_tokens: 4096,
          temperature: 0.7
        };
        if (se && (_e.location = {
          lat: se.lat,
          lng: se.lng,
          accuracy: se.accuracy
        }), Qv) {
          const te = await fetch(`${Ko()}/api/chat/create`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(_e),
            signal: M.signal
          });
          if (!te.ok) {
            const ve = await te.text();
            n((ye) => ye.map((B) => B.id === $.id ? {
              ...B,
              content: `Error: ${te.status} \u2014 ${ve}`
            } : B)), l(false);
            return;
          }
          const { sid: be } = await te.json();
          await new Promise((ve) => {
            let ye = null, B = 0, Z = false, L = "";
            const I = {
              tools: [],
              promptTokens: 0,
              completionTokens: 0,
              cacheReadTokens: 0,
              cacheCreateTokens: 0,
              elapsed: 0
            }, me = [], Ie = () => {
              if (Z) return;
              Z = true, ye == null ? void 0 : ye.close(), S.current && (cancelAnimationFrame(S.current), S.current = 0);
              const xe = {
                content: L
              };
              K && (xe.model = K), (I.promptTokens || I.tools.length) && (xe.trace = {
                ...I
              }), me.length && (xe.toolSteps = me.map((H) => ({
                ...H
              }))), n((H) => H.map((re) => re.id === $.id ? {
                ...re,
                ...xe
              } : re)), w.current = "", l(false), g.current = null, ve();
            };
            M.signal.addEventListener("abort", () => Ie());
            const tt = (xe) => {
              const H = Number(xe.lastEventId);
              B = Number.isFinite(H) ? H + 1 : B + 1;
            }, Qe = () => {
              S.current || (S.current = requestAnimationFrame(() => {
                const xe = w.current, H = {
                  ...I,
                  tools: [
                    ...I.tools
                  ]
                }, re = me.map((Tt) => ({
                  ...Tt
                }));
                n((Tt) => Tt.map((Ut) => Ut.id === $.id ? {
                  ...Ut,
                  content: xe,
                  trace: H,
                  toolSteps: re,
                  ...K && {
                    model: K
                  }
                } : Ut)), S.current = 0;
              }));
            };
            let ze = false;
            const pt = (xe) => {
              var _a3, _b2, _c3, _d2;
              if (tt(xe), xe.data === "[DONE]") {
                Ie();
                return;
              }
              try {
                const H = JSON.parse(xe.data);
                H.model && !K && (K = H.model);
                const re = (_b2 = (_a3 = H.choices) == null ? void 0 : _a3[0]) == null ? void 0 : _b2.delta;
                (re == null ? void 0 : re.reasoning_content) ? ze || (L += `*Thinking...*

`, ze = true) : (re == null ? void 0 : re.content) && (ze && (L = L.replace(`*Thinking...*

`, ""), ze = false), L += re.content), w.current = L, Qe(), ((_d2 = (_c3 = H.choices) == null ? void 0 : _c3[0]) == null ? void 0 : _d2.finish_reason) === "stop" && Ie();
              } catch {
              }
            }, ot = (xe) => {
              tt(xe);
              try {
                const H = JSON.parse(xe.data);
                if (H.type === "tool_call" && H.name) {
                  I.tools.push(H.name);
                  const re = H.args && H.args !== "{}" ? Yo(H.args) : void 0;
                  me.push({
                    name: H.name,
                    args: re,
                    status: "running"
                  }), Qe();
                } else if (H.type === "tool_result") {
                  for (let re = me.length - 1; re >= 0; re--) if (me[re].name === H.name && me[re].status === "running") {
                    me[re].result = (H.preview || "").slice(0, 60), me[re].elapsed = H.elapsed || 0, me[re].status = (me[re].result || "").startsWith("Error") ? "error" : "done";
                    break;
                  }
                  Qe();
                } else if (H.type === "keepalive" && H.name) {
                  for (let re = me.length - 1; re >= 0; re--) if (me[re].name === H.name && me[re].status === "running") {
                    me[re].elapsed = H.elapsed || 0;
                    break;
                  }
                  Qe();
                } else H.type === "llm_call" ? (I.promptTokens += H.prompt_tokens || 0, I.completionTokens += H.completion_tokens || 0, I.cacheReadTokens += H.cache_read_tokens || 0, I.cacheCreateTokens += H.cache_create_tokens || 0, I.elapsed += H.elapsed || 0) : H.type === "error" && (L += `

*Error: ${H.message || "Unknown error"}*`, w.current = L, Qe());
              } catch {
              }
            }, ln = () => {
              Z || M.signal.aborted || (ye == null ? void 0 : ye.close(), ye = new EventSource(`${Ko()}/api/chat/stream?sid=${be}&offset=${B}`), ye.onmessage = pt, ye.addEventListener("trace", ot), ye.onerror = () => {
                ye == null ? void 0 : ye.close(), !Z && !M.signal.aborted && window.setTimeout(ln, document.visibilityState === "visible" ? 1e3 : 3e3);
              });
            };
            ln();
          });
          return;
        }
        const He = await fetch(`${Ko()}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(_e),
          signal: M.signal
        });
        if (!He.ok) {
          const te = await He.text();
          n((be) => be.map((ve) => ve.id === $.id ? {
            ...ve,
            content: `Error: ${He.status} \u2014 ${te}`
          } : ve)), l(false);
          return;
        }
        const mt = (_a2 = He.body) == null ? void 0 : _a2.getReader();
        if (!mt) {
          l(false);
          return;
        }
        const D = new TextDecoder();
        let Q = "", V = "", le = "", ee = false;
        const fe = {
          tools: [],
          promptTokens: 0,
          completionTokens: 0,
          cacheReadTokens: 0,
          cacheCreateTokens: 0,
          elapsed: 0
        }, X = [], ue = () => {
          S.current || (S.current = requestAnimationFrame(() => {
            const te = w.current, be = {
              ...fe,
              tools: [
                ...fe.tools
              ]
            }, ve = X.map((ye) => ({
              ...ye
            }));
            n((ye) => ye.map((B) => B.id === $.id ? {
              ...B,
              content: te,
              trace: be,
              toolSteps: ve,
              ...K && {
                model: K
              }
            } : B)), S.current = 0;
          }));
        };
        for (; ; ) {
          const { done: te, value: be } = await mt.read();
          if (te) break;
          Q += D.decode(be, {
            stream: true
          });
          const ve = Q.split(`
`);
          Q = ve.pop() || "";
          for (const ye of ve) {
            const B = ye.trim();
            if (!B) continue;
            if (B.startsWith(": trace ")) {
              try {
                const L = JSON.parse(B.slice(8));
                if (L.type === "tool_call" && L.name) {
                  fe.tools.push(L.name);
                  const I = L.args && L.args !== "{}" ? Yo(L.args) : void 0;
                  X.push({
                    name: L.name,
                    args: I,
                    status: "running"
                  }), ue();
                } else if (L.type === "tool_result") {
                  for (let I = X.length - 1; I >= 0; I--) if (X[I].name === L.name && X[I].status === "running") {
                    const me = (L.preview || "").slice(0, 60);
                    X[I].result = me, X[I].elapsed = L.elapsed || 0, X[I].status = me.startsWith("Error") ? "error" : "done";
                    break;
                  }
                  ue();
                } else if (L.type === "keepalive" && L.name) {
                  for (let I = X.length - 1; I >= 0; I--) if (X[I].name === L.name && X[I].status === "running") {
                    X[I].elapsed = L.elapsed || 0;
                    break;
                  }
                  ue();
                } else if (L.type === "llm_call") fe.promptTokens += L.prompt_tokens || 0, fe.completionTokens += L.completion_tokens || 0, fe.cacheReadTokens += L.cache_read_tokens || 0, fe.cacheCreateTokens += L.cache_create_tokens || 0, fe.elapsed += L.elapsed || 0;
                else if (L.type === "error") {
                  const I = L.message || "Unknown error";
                  V += `

*Error: ${I}*`, w.current = V, ue();
                }
              } catch {
              }
              continue;
            }
            if (B.startsWith("event: ")) {
              le = B.slice(7);
              continue;
            }
            if (!B.startsWith("data: ")) continue;
            const Z = B.slice(6);
            if (Z !== "[DONE]") {
              if (le === "trace") {
                le = "";
                try {
                  const L = JSON.parse(Z);
                  if (L.type === "tool_call" && L.name) {
                    fe.tools.push(L.name);
                    const I = L.args && L.args !== "{}" ? Yo(L.args) : void 0;
                    X.push({
                      name: L.name,
                      args: I,
                      status: "running"
                    }), ue();
                  } else if (L.type === "tool_result") {
                    for (let I = X.length - 1; I >= 0; I--) if (X[I].name === L.name && X[I].status === "running") {
                      const me = (L.preview || "").slice(0, 60);
                      X[I].result = me, X[I].elapsed = L.elapsed || 0, X[I].status = me.startsWith("Error") ? "error" : "done";
                      break;
                    }
                    ue();
                  } else if (L.type === "keepalive" && L.name) {
                    for (let I = X.length - 1; I >= 0; I--) if (X[I].name === L.name && X[I].status === "running") {
                      X[I].elapsed = L.elapsed || 0;
                      break;
                    }
                    ue();
                  } else if (L.type === "llm_call") fe.promptTokens += L.prompt_tokens || 0, fe.completionTokens += L.completion_tokens || 0, fe.cacheReadTokens += L.cache_read_tokens || 0, fe.cacheCreateTokens += L.cache_create_tokens || 0, fe.elapsed += L.elapsed || 0;
                  else if (L.type === "error") {
                    const I = L.message || "Unknown error";
                    V += `

*Error: ${I}*`, w.current = V, ue();
                  }
                } catch {
                }
                continue;
              }
              le = "";
              try {
                const L = JSON.parse(Z);
                L.model && !K && (K = L.model);
                const I = (_c2 = (_b = L.choices) == null ? void 0 : _b[0]) == null ? void 0 : _c2.delta;
                (I == null ? void 0 : I.reasoning_content) ? ee || (V += `*Thinking...*

`, ee = true) : (I == null ? void 0 : I.content) && (ee && (V = V.replace(`*Thinking...*

`, ""), ee = false), V += I.content), ((I == null ? void 0 : I.reasoning_content) || (I == null ? void 0 : I.content)) && (w.current = V, ue());
              } catch {
              }
            }
          }
        }
        const je = {};
        K && (je.model = K), (fe.promptTokens || fe.tools.length) && (je.trace = {
          ...fe
        }), X.length && (je.toolSteps = X.map((te) => ({
          ...te
        }))), Object.keys(je).length && n((te) => te.map((be) => be.id === $.id ? {
          ...be,
          ...je
        } : be));
      } catch (J) {
        J instanceof Error && J.name !== "AbortError" && n((de) => de.map((se) => se.id === $.id ? {
          ...se,
          content: `Error: ${J.message}`
        } : se));
      } finally {
        if (S.current && (cancelAnimationFrame(S.current), S.current = 0), w.current) {
          const J = w.current, de = K || void 0;
          n((se) => se.map((_e) => _e.id === $.id ? {
            ..._e,
            content: J,
            ...de && {
              model: de
            }
          } : _e)), w.current = "";
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
  function Jv() {
    const [e, t] = v.useState(Em), [n, r] = v.useState(Hv);
    v.useEffect(() => {
      Fv().then((d) => {
        d.length > 0 && t(d);
      });
    }, []);
    const l = v.useCallback((d) => {
      const g = [
        ...d
      ].sort((y, w) => w.updatedAt - y.updatedAt);
      t(g), Bv(g);
    }, []), a = v.useCallback((d, g) => {
      const y = Ks(), w = Date.now(), _ = [
        {
          id: y,
          title: zc(d),
          createdAt: w,
          updatedAt: w,
          model: g
        },
        ...e
      ];
      return l(_), r(y), oa(y), y;
    }, [
      e,
      l
    ]), o = v.useCallback((d) => {
      r(d), oa(d);
    }, []), i = v.useCallback((d) => {
      const g = e.filter((y) => y.id !== d);
      if (l(g), Wv(d), n === d) {
        const y = g.length > 0 ? g[0].id : null;
        r(y), oa(y);
      }
    }, [
      e,
      n,
      l
    ]), s = v.useCallback((d, g) => {
      const y = e.map((w) => w.id === d ? {
        ...w,
        title: zc(g)
      } : w);
      l(y);
    }, [
      e,
      l
    ]), c = v.useCallback((d) => {
      const g = e.map((y) => y.id === d ? {
        ...y,
        updatedAt: Date.now()
      } : y);
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
  function Xv(e, t) {
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
  function Zv(e) {
    const t = `${window.location.protocol}//${window.location.host}`;
    return e.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (r, l, a) => `<img src="${a.startsWith("/") ? `${t}${a}` : a}" alt="${l}" style="max-width:100%; max-height:50vh; border-radius:8px; margin:8px 0; object-fit:contain;" loading="lazy" onerror="this.style.display='none'" />`).replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-blue-400 underline">$1</a>').replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="my-2 rounded-lg bg-[#0d1117] p-3 text-sm overflow-x-auto"><code class="text-green-300">$2</code></pre>').replace(/`([^`]+)`/g, '<code class="rounded bg-[#0d1117] px-1.5 py-0.5 text-sm text-orange-300">$1</code>').replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>").replace(/^- (.+)$/gm, '<li class="ml-4 list-disc">$1</li>').replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>').replace(/\n/g, "<br/>");
  }
  const qv = {
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
  function ey(e) {
    const t = /* @__PURE__ */ new Map();
    for (const r of e) t.set(r.name, (t.get(r.name) || 0) + 1);
    const n = [];
    for (const [r, l] of t) {
      const a = qv[r] || r.replace(/_/g, " ");
      if (l > 1) {
        const o = a.replace(/(?:a |an )(\w+)$/, `${l} $1s`);
        n.push(o === a ? `${a} x${l}` : o);
      } else n.push(a);
    }
    return n.length <= 2 ? n.join(" and ") : n.slice(0, -1).join(", ") + ", and " + n[n.length - 1];
  }
  const ty = {
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
  }, ny = ({ steps: e, streaming: t, contentStarted: n }) => {
    if (!e.length) return null;
    const r = e.some((l) => l.status === "running");
    return u.jsxs("div", {
      className: "mb-2 space-y-1",
      children: [
        e.map((l, a) => {
          const o = ty[l.name] || "\u2699\uFE0F", i = l.status === "running", s = l.status === "error", c = i ? "border-cyan-400/40" : s ? "border-red-400/40" : "border-cyan-500/20";
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
              ey(e),
              (() => {
                const l = e.reduce((a, o) => a + (o.elapsed || 0), 0);
                return l > 0 ? ` \u2014 ${l.toFixed(1)}s` : "";
              })()
            ]
          })
        })
      ]
    });
  }, ry = ({ trace: e }) => {
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
  }, ly = {
    "claude-opus-4-20250514": "Claude Opus",
    "claude-sonnet-4-20250514": "Claude Sonnet",
    "mistral-large-latest": "Mistral Large",
    "codestral-latest": "Codestral",
    "devstral-2512": "Devstral",
    "devstral-small-latest": "Devstral Small",
    "devstral-medium-latest": "Devstral Medium",
    nemotron: "Nemotron",
    "nemotron-super": "Nemotron Super",
    "gemma-4-31b": "Gemma 4",
    llava: "LLaVA"
  }, ay = ({ message: e, animate: t }) => {
    const n = e.role === "user", r = Xv(e.content, !!t), l = !n && e.toolSteps && e.toolSteps.length > 0, a = !e.content && !n && !l;
    return u.jsx("div", {
      className: `flex ${n ? "justify-end" : "justify-start"} mb-3`,
      children: u.jsxs("div", {
        className: `max-w-[85%] rounded-2xl px-4 py-3 ${n ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`,
        children: [
          e.model && !n && u.jsx("div", {
            className: "mb-1 text-[10px] font-medium tracking-wider text-maude-muted",
            children: ly[e.model] || e.model
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
          l && u.jsx(ny, {
            steps: e.toolSteps,
            streaming: !!t,
            contentStarted: !!e.content
          }),
          r && u.jsx("div", {
            className: "break-words text-sm leading-relaxed",
            dangerouslySetInnerHTML: {
              __html: Zv(r)
            }
          }),
          !n && e.trace && u.jsx(ry, {
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
  const oy = ({ onSend: e, isStreaming: t, onStop: n }) => {
    const [r, l] = v.useState(""), [a, o] = v.useState([]), [i, s] = v.useState(false), c = v.useRef(null), m = v.useRef(null), d = v.useRef(null);
    v.useEffect(() => {
      var _a2;
      (_a2 = c.current) == null ? void 0 : _a2.focus();
    }, []);
    const g = () => {
      (a.length > 0 || r.trim()) && (e(r.trim(), a.length > 0 ? a : void 0), l(""), o([]), c.current && (c.current.style.height = "44px"));
    }, y = (f) => {
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
            const T = `camera_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.jpg`;
            (await fetch(`${Ac()}/share/${encodeURIComponent(T)}`, {
              method: "POST",
              body: C
            })).ok && j.push(`/download/${T}`);
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
              onKeyDown: y,
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
      desc: "Local 120B"
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
  ], sy = ({ currentModel: e, onSelect: t, autoRoute: n, onToggleAutoRoute: r }) => {
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
  function iy(e) {
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
  const uy = ({ open: e, onClose: t, conversations: n, activeId: r, onSelect: l, onDelete: a, onNewChat: o }) => {
    const i = iy(n), [s, c] = v.useState(false);
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
  }, cy = ({ conversationId: e, onFirstMessage: t, onMessageSent: n, onOpenDrawer: r, onNewChat: l }) => {
    const a = ho(), o = v.useRef(null), i = v.useRef(e), { messages: s, isStreaming: c, currentModel: m, setCurrentModel: d, autoRoute: g, setAutoRoute: y, sendMessage: w, stopStreaming: S } = Yv(e);
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
            u.jsx(sy, {
              currentModel: m,
              onSelect: d,
              autoRoute: g,
              onToggleAutoRoute: y
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
                  children: "Your local AI assistant. Ask me anything."
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
            s.map((p, f) => u.jsx(ay, {
              message: p,
              animate: c && f === s.length - 1
            }, p.id))
          ]
        }),
        u.jsx(oy, {
          onSend: (p, f) => _(p, f),
          isStreaming: c,
          onStop: S
        })
      ]
    });
  }, dy = () => {
    const [e, t] = v.useState(false), { conversations: n, activeId: r, createConversation: l, switchConversation: a, deleteConversation: o, touchConversation: i, startNewChat: s } = Jv(), c = v.useCallback((d, g) => l(d, g), [
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
        u.jsx(cy, {
          conversationId: r,
          onFirstMessage: c,
          onMessageSent: m,
          onOpenDrawer: () => t(true),
          onNewChat: s
        }, r || "new"),
        u.jsx(uy, {
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
  }, fy = "modulepreload", my = function(e) {
    return "/" + e;
  }, Uc = {}, il = function(t, n, r) {
    let l = Promise.resolve();
    if (n && n.length > 0) {
      document.getElementsByTagName("link");
      const o = document.querySelector("meta[property=csp-nonce]"), i = (o == null ? void 0 : o.nonce) || (o == null ? void 0 : o.getAttribute("nonce"));
      l = Promise.allSettled(n.map((s) => {
        if (s = my(s), s in Uc) return;
        Uc[s] = true;
        const c = s.endsWith(".css"), m = c ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${s}"]${m}`)) return;
        const d = document.createElement("link");
        if (d.rel = c ? "stylesheet" : fy, c || (d.as = "script"), d.crossOrigin = "", d.href = s, i && d.setAttribute("nonce", i), document.head.appendChild(d), c) return new Promise((g, y) => {
          d.addEventListener("load", g), d.addEventListener("error", () => y(new Error(`Unable to preload CSS for ${s}`)));
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
  }, py = {
    0: 0
  }, hy = {
    0: 0
  }, gy = {
    start: 0,
    endTurn: 1,
    pause: 2,
    restart: 3
  }, vy = (e) => {
    switch (e.type) {
      case "handshake":
        return new Uint8Array([
          0,
          py[e.version],
          hy[e.model]
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
          gy[e.action]
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
  }, yy = "You are MAUDE, a capable AI assistant with a warm Scottish accent. You are direct, competent, and quietly confident. Keep responses concise and natural for voice conversation. You run locally on Matt\u2019s DGX Spark workstation.", xy = "NATF2.pt";
  function wy() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  function Sy(e) {
    const n = `wss://${window.location.host}`;
    let r = yy;
    e && (r += `

--- Image Context ---
` + e);
    const l = new URLSearchParams({
      text_prompt: r
    });
    return `${n}/api/chat?${l}`;
  }
  const ky = `
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
  async function Ny(e, t) {
    const n = new Blob([
      ky
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
  }, jy = () => {
    const e = ho(), [t, n] = v.useState("disconnected"), [r, l] = v.useState(""), [a, o] = v.useState(false), [i, s] = v.useState(0), [c, m] = v.useState(""), [d, g] = v.useState(""), [y, w] = v.useState(null), [S, _] = v.useState(null), [p, f] = v.useState(false), [h, j] = v.useState(false), C = v.useRef(null), T = v.useRef(null), k = v.useRef(null), R = v.useRef(null), $ = v.useRef(null), M = v.useRef(null), K = v.useRef(null), J = v.useRef(null), de = v.useRef(null), se = v.useRef(null), _e = v.useRef(0), He = v.useRef(0), mt = v.useRef(0), D = v.useRef(0), Q = v.useRef(0), V = v.useRef(0), le = v.useRef(0), ee = v.useCallback(async () => {
      m(""), l(""), s(0), _e.current = 0;
      try {
        $.current || ($.current = new AudioContext({
          sampleRate: 48e3
        }));
        const B = $.current;
        await B.resume();
        const Z = B.createBuffer(1, 1, B.sampleRate), L = B.createBufferSource();
        L.buffer = Z, L.connect(B.destination), L.start(), mt.current = 0, D.current = 0, g(`ctx: ${B.state} ${B.sampleRate}Hz`), M.current || (M.current = await Ny(B, (pt, ot) => {
          ot.underruns != null && (Q.current = ot.underruns), ot.avail != null && (V.current = ot.avail);
        }), M.current.connect(B.destination)), M.current.reset(), Q.current = 0;
        const I = B.createAnalyser();
        M.current.connect(I), J.current = I;
        const me = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1
          }
        });
        se.current = me;
        const Ie = B.createAnalyser();
        B.createMediaStreamSource(me).connect(Ie), de.current = Ie;
        const Qe = Sy(R.current ?? void 0);
        console.log("Connecting to voice server:", Qe);
        const ze = new WebSocket(Qe);
        ze.binaryType = "arraybuffer", C.current = ze, n("connecting"), ze.onopen = () => {
          console.log("voice server WS open, waiting for handshake");
        }, ze.onmessage = (pt) => {
          var _a2;
          try {
            const ot = new Uint8Array(pt.data), ln = ot[0];
            if (ln === 0) console.log("voice server handshake received"), n("connected"), fe(ze, me, B), He.current = window.setInterval(() => {
              var _a3;
              _e.current += 1, s(_e.current);
              const xe = ((_a3 = $.current) == null ? void 0 : _a3.state) ?? "?", H = Math.round(V.current / 48);
              g(`dec:${D.current} buf:${H}ms ur:${Q.current}`);
            }, 1e3);
            else if (ln === 2) {
              const xe = new TextDecoder().decode(ot.slice(1));
              xe.includes("[Searching...]") ? o(true) : (xe.includes("[Tool result:]") || xe.includes("[Error:")) && o(false), l((H) => H + xe);
            } else if (ln === 3) {
              D.current++;
              const xe = ot.slice(1), H = new Float32Array(xe.buffer, xe.byteOffset, xe.byteLength / 4), re = new Float32Array(H.length * 2);
              for (let Tt = 0; Tt < re.length; Tt++) {
                const Ut = Tt * 0.5, ht = Ut | 0, Pt = Math.min(ht + 1, H.length - 1), Ol = Ut - ht;
                re[Tt] = H[ht] + (H[Pt] - H[ht]) * Ol;
              }
              (_a2 = M.current) == null ? void 0 : _a2.feedAudio(re);
            }
          } catch (ot) {
            console.error("Message decode error:", ot);
          }
        }, ze.onclose = (pt) => {
          console.log("voice server WS closed:", pt.code, pt.reason), n("disconnected"), X(), clearInterval(He.current);
        }, ze.onerror = (pt) => {
          console.error("voice server WS error:", pt), m("WebSocket connection failed. Is voice server running?"), n("disconnected");
        };
      } catch (B) {
        const Z = B instanceof Error ? B.message : "Connection failed";
        console.error("Voice connect error:", Z), m(Z), n("disconnected");
      }
    }, []), fe = v.useCallback(async (B, Z, L) => {
      try {
        const I = (await il(async () => {
          const { default: Qe } = await import("./recorder.min-C-ucyZOc.js").then((ze) => ze.r);
          return {
            default: Qe
          };
        }, [])).default, me = (await il(async () => {
          const { default: Qe } = await import("./encoderWorker.min-De-nC0Q0.js");
          return {
            default: Qe
          };
        }, [])).default, Ie = L.createMediaStreamSource(Z), tt = new I({
          encoderPath: me,
          bufferLength: Math.round(960 * L.sampleRate / 24e3),
          encoderFrameSize: 20,
          encoderSampleRate: 24e3,
          maxFramesPerPage: 2,
          numberOfChannels: 1,
          recordingGain: 1,
          resampleQuality: 3,
          encoderComplexity: 3,
          encoderApplication: 2049,
          streamPages: true,
          sourceNode: Ie
        });
        tt.ondataavailable = (Qe) => {
          B.readyState === WebSocket.OPEN && B.send(vy({
            type: "audio",
            data: Qe
          }));
        }, tt.onstart = () => {
          console.log("Opus recorder started");
        }, tt.start(), K.current = tt;
      } catch (I) {
        console.error("Recorder start error:", I), m("Failed to start microphone recording");
      }
    }, []), X = v.useCallback(() => {
      if (K.current) {
        try {
          K.current.stop();
        } catch {
        }
        K.current = null;
      }
      se.current && (se.current.getTracks().forEach((B) => B.stop()), se.current = null);
    }, []), ue = v.useCallback(() => {
      X(), clearInterval(He.current), clearInterval(le.current), C.current && (C.current.close(), C.current = null), n("disconnected");
    }, [
      X
    ]), je = v.useCallback(async (B) => {
      var _a2;
      const Z = (_a2 = B.target.files) == null ? void 0 : _a2[0];
      if (!Z) return;
      B.target.value = "";
      const L = `voice_camera_${Date.now()}.jpg`, I = wy(), me = URL.createObjectURL(Z);
      w(me), _(null), j(true);
      try {
        if (!(await fetch(`${I}/share/${L}`, {
          method: "POST",
          body: Z
        })).ok) throw new Error("Upload failed");
        j(false), f(true);
        const tt = await fetch(`${I}/api/analyze-image`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            filename: L,
            question: "Describe this image in detail. What do you see?"
          })
        });
        if (!tt.ok) throw new Error("Analysis failed");
        const ze = (await tt.json()).analysis || "No analysis returned.";
        _(ze), f(false), R.current = `The user shared an image (${L}). Analysis: ${ze}`, C.current && C.current.readyState === WebSocket.OPEN && (ue(), await new Promise((pt) => setTimeout(pt, 300)), ee());
      } catch (Ie) {
        const tt = Ie instanceof Error ? Ie.message : "Image processing failed";
        m(tt), j(false), f(false);
      }
    }, [
      ee,
      ue
    ]), te = v.useCallback(async () => {
      R.current = null, w(null), _(null), C.current && C.current.readyState === WebSocket.OPEN && (ue(), await new Promise((B) => setTimeout(B, 300)), ee());
    }, [
      ee,
      ue
    ]);
    v.useEffect(() => () => {
      ue();
    }, []);
    const be = (B) => {
      const Z = Math.floor(B / 60), L = B % 60;
      return `${Z}:${L.toString().padStart(2, "0")}`;
    }, ve = t === "connected", ye = t === "connecting";
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
                  className: `h-32 w-32 rounded-full border-4 ${ve ? "animate-pulse border-maude-accent shadow-[0_0_30px_rgba(255,69,0,0.3)]" : ye ? "animate-spin border-maude-muted" : "border-maude-border"} flex items-center justify-center`,
                  children: u.jsx("span", {
                    className: "text-4xl",
                    children: ve ? "\u{1F399}\uFE0F" : ye ? "\u23F3" : "\u{1F399}\uFE0F"
                  })
                }),
                u.jsx("span", {
                  className: "text-sm text-maude-muted",
                  children: ve ? `Connected \u2022 ${be(i)}` : ye ? "Connecting to MAUDE Voice..." : "Tap to start voice chat"
                })
              ]
            }),
            ve && u.jsxs("div", {
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
                        analyser: J.current,
                        active: ve,
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
                        active: ve,
                        color: "#888"
                      })
                    })
                  ]
                })
              ]
            }),
            ve && u.jsxs("div", {
              className: "flex gap-3",
              children: [
                u.jsxs("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = T.current) == null ? void 0 : _a2.click();
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
              ref: T,
              type: "file",
              accept: "image/*",
              capture: "environment",
              onChange: je,
              className: "hidden"
            }),
            u.jsx("input", {
              ref: k,
              type: "file",
              accept: "image/*",
              onChange: je,
              className: "hidden"
            }),
            y && u.jsxs("div", {
              className: "w-full max-w-xs rounded-xl bg-maude-surface p-3",
              children: [
                u.jsx("span", {
                  className: "mb-2 block text-[10px] uppercase tracking-wider text-maude-muted",
                  children: "Image Context"
                }),
                u.jsx("img", {
                  src: y,
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
                  onClick: te,
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
`).map((B, Z) => B.includes("[Searching...]") ? u.jsx("p", {
                    className: "my-1 text-xs italic text-maude-accent",
                    children: B
                  }, Z) : B.includes("[Tool result:]") ? u.jsx("p", {
                    className: "mt-2 mb-1 text-[10px] font-bold uppercase tracking-wider text-maude-accent",
                    children: B
                  }, Z) : B.includes("[Error:") ? u.jsx("p", {
                    className: "my-1 text-xs text-red-400",
                    children: B
                  }, Z) : u.jsxs("span", {
                    children: [
                      B,
                      Z < r.split(`
`).length - 1 ? `
` : ""
                    ]
                  }, Z))
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
              onClick: ve || ye ? ue : ee,
              className: `min-w-[200px] rounded-2xl px-8 py-4 text-base font-semibold text-white transition-all ${ve ? "bg-red-600 hover:bg-red-700" : ye ? "bg-maude-muted" : "fire-bg hover:opacity-90"}`,
              disabled: ye,
              children: ve ? "End Call" : ye ? "Connecting..." : "Start Voice Chat"
            }),
            u.jsxs("div", {
              className: "text-center text-[10px] text-maude-muted",
              children: [
                "Voice: ",
                (localStorage.getItem("maude-default-voice") || xy).replace(".pt", ""),
                " \u2022 ",
                "MAUDE Voice"
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
  }, Ey = /iPad|iPhone|iPod/.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1, Cy = () => {
    const e = v.useRef(null), t = v.useRef(null), n = v.useRef(null), r = v.useRef(null), l = v.useRef(null), a = v.useRef(null), o = v.useRef(null), [i, s] = v.useState("disconnected");
    return v.useEffect(() => {
      let c, m;
      return (async () => {
        const { Terminal: g } = await il(async () => {
          const { Terminal: _ } = await import("./xterm-PglAAeey.js").then((p) => p.x);
          return {
            Terminal: _
          };
        }, []), { FitAddon: y } = await il(async () => {
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
        const S = new y();
        if (c.loadAddon(S), c.loadAddon(new w()), r.current = c, l.current = S, e.current && (c.open(e.current), S.fit()), s("connecting"), Ey) try {
          const _ = await fetch("/api/terminal/create", {
            method: "POST"
          }), { sid: p } = await _.json();
          o.current = p;
          const f = new EventSource(`/api/terminal/stream?sid=${p}`);
          a.current = f, f.onopen = () => {
            s("connected");
            const T = S.proposeDimensions();
            T && fetch("/api/terminal/resize", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: p,
                cols: T.cols,
                rows: T.rows
              })
            });
          }, f.onmessage = (T) => {
            const k = Uint8Array.from(atob(T.data), (R) => R.charCodeAt(0));
            c.write(k);
          }, f.onerror = () => {
            s("disconnected"), c.write(`\r
\x1B[33m[Connection closed]\x1B[0m\r
`), f.close();
          };
          const h = (T) => {
            fetch("/api/terminal/input", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: p,
                data: T
              })
            });
          };
          n.current = h, c.onData(h);
          const j = () => {
            S.fit();
            const T = S.proposeDimensions();
            T && fetch("/api/terminal/resize", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: p,
                cols: T.cols,
                rows: T.rows
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
  function _y() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  const Ry = [
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
  ], Ty = () => {
    const [e, t] = v.useState(""), [n, r] = v.useState(""), [l, a] = v.useState(""), [o, i] = v.useState(false), [s, c] = v.useState(""), m = v.useRef(null), [d, g] = v.useState("proxy"), [y, w] = v.useState([]), [S, _] = v.useState(-1), p = v.useCallback(async (j) => {
      if (!j.trim()) return;
      let C = j.trim();
      if (!C.startsWith("http://") && !C.startsWith("https://") && (C = "https://" + C), r(C), c(""), d === "iframe") {
        t(C), w((T) => [
          ...T.slice(0, S + 1),
          C
        ]), _((T) => T + 1);
        return;
      }
      i(true);
      try {
        const T = await fetch(`${_y()}/proxy?url=${encodeURIComponent(C)}`);
        if (!T.ok) {
          c(`Failed: ${T.status}`), i(false);
          return;
        }
        if ((T.headers.get("content-type") || "").includes("application/json")) {
          const R = await T.json();
          if (R.redirect) {
            i(false), p(R.redirect);
            return;
          }
          c(R.error || "Unknown error");
        } else a(await T.text());
        w((R) => [
          ...R.slice(0, S + 1),
          C
        ]), _((R) => R + 1);
      } catch (T) {
        c(T instanceof Error ? T.message : "Failed");
      }
      i(false);
    }, [
      d,
      S
    ]), f = () => {
      S > 0 && (_(S - 1), p(y[S - 1]));
    }, h = () => {
      S < y.length - 1 && (_(S + 1), p(y[S + 1]));
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
                      disabled: S >= y.length - 1,
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
              children: Ry.map((j) => u.jsx("button", {
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
  function Py() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  const by = () => {
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
        const c = await fetch(`${Py()}/v1/chat/completions`, {
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
  function Gr() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  function Dy(e) {
    return e < 1024 ? e + " B" : e < 1048576 ? (e / 1024).toFixed(1) + " KB" : (e / 1048576).toFixed(1) + " MB";
  }
  function My(e) {
    return new Date(e * 1e3).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }
  const Ly = () => {
    const [e, t] = v.useState([]), [n, r] = v.useState(""), [l, a] = v.useState(false), [o, i] = v.useState(""), [s, c] = v.useState("shared"), m = v.useRef(null), d = v.useCallback(async (w) => {
      a(true), i("");
      try {
        const S = s === "transfers" ? `${Gr()}/transfers` : w ? `${Gr()}/list?path=${encodeURIComponent(w)}` : `${Gr()}/list`, p = await (await fetch(S)).json();
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
      window.open(`${Gr()}/${s === "transfers" ? "download-transfer" : "download"}/${encodeURIComponent(w)}`);
    }, y = async (w) => {
      var _a2;
      const S = (_a2 = w.target.files) == null ? void 0 : _a2[0];
      if (S) {
        a(true);
        try {
          const p = await (await fetch(`${Gr()}/upload/${encodeURIComponent(S.name)}`, {
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
                  onChange: y,
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
                        w.is_dir ? "Directory" : Dy(w.size),
                        " \xB7 ",
                        My(w.modified)
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
  const Oy = [
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
  function Iy(e) {
    document.documentElement.setAttribute("data-theme", e), localStorage.setItem("maude-theme", e);
  }
  const zy = () => {
    var _a2, _b;
    const [e, t] = v.useState(null), [n, r] = v.useState([]), [l, a] = v.useState(() => localStorage.getItem("maude-default-model") || "mistral-large-latest"), [o, i] = v.useState(() => localStorage.getItem("maude-default-voice") || "NATF2.pt"), [s, c] = v.useState(() => localStorage.getItem("maude-theme") || "dark"), m = e !== null, d = (e == null ? void 0 : e.gateway_port) ?? 3e4, g = (_a2 = e == null ? void 0 : e.services) == null ? void 0 : _a2.llama_server, y = (_b = e == null ? void 0 : e.services) == null ? void 0 : _b.voice_server;
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
    }, p = _(g), f = _(y);
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
                          children: "Voice Server"
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
                  children: Oy.map((h) => u.jsxs("button", {
                    onClick: () => {
                      c(h.id), Iy(h.id);
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
                  children: "Voice"
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
                            "MAUDE Voice (",
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
  function Ay(e = 1e4) {
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
  function Uy() {
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
  function $y() {
    if (Bc) return;
    Bc = true;
    const e = Uy(), t = `${e.clientType}-${Math.random().toString(36).slice(2, 8)}`, n = () => {
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
  const Fy = {
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
  }, By = ({ entry: e, now: t }) => u.jsxs("div", {
    className: "flex items-center gap-3 rounded-xl bg-maude-surface p-3",
    children: [
      u.jsx("div", {
        className: "flex h-10 w-10 items-center justify-center rounded-full bg-maude-card text-lg",
        children: Vc[e.client_type] || Vc.unknown
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
  }), Wc = {
    chat: "\u{1F4AC}",
    task_dispatched: "\u{1F680}",
    project_created: "\u{1F4C1}",
    custom: "\u2022"
  }, Vy = ({ event: e, now: t }) => u.jsxs("div", {
    className: "flex items-start gap-2 py-1.5",
    children: [
      u.jsx("span", {
        className: "mt-0.5 text-sm",
        children: Wc[e.type] || Wc.custom
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
  }), Wy = ({ project: e }) => u.jsxs("div", {
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
  }), Hy = ({ task: e, now: t }) => u.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      u.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          u.jsx("span", {
            className: `inline-block h-2 w-2 rounded-full ${Fy[e.status] || "bg-gray-500"}`
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
  }), Qy = () => {
    const { status: e, loading: t } = Ay(), [n, r] = v.useState("presence");
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
              }) : e.presence.map((o) => u.jsx(By, {
                entry: o,
                now: l
              }, o.client_id))
            }),
            n === "activity" && u.jsx("div", {
              className: "flex flex-col divide-y divide-maude-border",
              children: e.activity.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No recent activity"
              }) : e.activity.map((o) => u.jsx(Vy, {
                event: o,
                now: l
              }, o.id))
            }),
            n === "projects" && u.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.projects.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No projects yet"
              }) : e.projects.map((o) => u.jsx(Wy, {
                project: o
              }, o.id))
            }),
            n === "tasks" && u.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.tasks.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No tasks dispatched"
              }) : e.tasks.map((o) => u.jsx(Hy, {
                task: o,
                now: l
              }, o.id))
            })
          ]
        })
      ]
    });
  };
  function Gy() {
    return `${window.location.protocol}//${window.location.host}`;
  }
  async function lr(e) {
    try {
      const t = await fetch(`${Gy()}/api/command-center/${e}`);
      return t.ok ? await t.json() : null;
    } catch {
      return null;
    }
  }
  function Ky(e = 1e4) {
    const [t, n] = v.useState(null), [r, l] = v.useState(null), [a, o] = v.useState([]), [i, s] = v.useState([]), [c, m] = v.useState(null), [d, g] = v.useState([]), [y, w] = v.useState(true), S = v.useCallback(async () => {
      const [_, p, f, h, j, C] = await Promise.all([
        lr("system"),
        lr("gpu-processes"),
        lr("sessions?limit=10"),
        lr("activity?limit=15"),
        lr("scheduler"),
        lr("nodes")
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
      loading: y,
      refresh: S
    };
  }
  const Dn = ({ label: e, value: t, sub: n, color: r = "text-maude-accent" }) => u.jsxs("div", {
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
  }), Yy = ({ processes: e }) => {
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
  }, Jy = ({ node: e }) => u.jsxs("div", {
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
  }), Xy = ({ task: e }) => u.jsxs("div", {
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
  }), Zy = ({ item: e }) => u.jsxs("div", {
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
  }), qy = ({ session: e }) => u.jsxs("div", {
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
  }), ex = () => {
    var _a2, _b, _c2, _d2, _e, _f2, _g2, _h2, _i2;
    const { system: e, gpuProcesses: t, sessions: n, activity: r, scheduler: l, nodes: a, loading: o, refresh: i } = Ky(), [s, c] = v.useState("overview");
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
          children: m.map((y) => u.jsx("button", {
            onClick: () => c(y.key),
            className: `rounded-full px-3 py-1 text-xs font-medium transition-colors ${s === y.key ? "bg-maude-accent text-white" : "bg-maude-surface text-maude-muted"}`,
            children: y.label
          }, y.key))
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
                    u.jsx(Dn, {
                      label: "CPU",
                      value: `${(e == null ? void 0 : e.cpu_percent) ?? 0}%`,
                      sub: `${((_b = e == null ? void 0 : e.ram) == null ? void 0 : _b.used_gb) ?? 0}/${((_c2 = e == null ? void 0 : e.ram) == null ? void 0 : _c2.total_gb) ?? 0}GB RAM`
                    }),
                    u.jsx(Dn, {
                      label: "GPU Temp",
                      value: `${d}\xB0C`,
                      sub: ((_d2 = e == null ? void 0 : e.gpu) == null ? void 0 : _d2.name) || "N/A",
                      color: g
                    }),
                    u.jsx(Dn, {
                      label: "Disk",
                      value: `${((_e = e == null ? void 0 : e.disk) == null ? void 0 : _e.percent) ?? 0}%`,
                      sub: `${((_f2 = e == null ? void 0 : e.disk) == null ? void 0 : _f2.used_gb) ?? 0}/${((_g2 = e == null ? void 0 : e.disk) == null ? void 0 : _g2.total_gb) ?? 0}GB`
                    }),
                    u.jsx(Dn, {
                      label: "Sessions",
                      value: n.length,
                      sub: `${((_h2 = l == null ? void 0 : l.stats) == null ? void 0 : _h2.active) ?? 0} scheduled tasks`
                    })
                  ]
                }),
                t && u.jsx(Yy, {
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
                      children: n.slice(0, 5).map((y) => u.jsx(qy, {
                        session: y
                      }, y.session_id + y.channel))
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
              }) : a.map((y, w) => u.jsx(Jy, {
                node: y
              }, y.name + w))
            }),
            s === "activity" && u.jsx("div", {
              className: "divide-y divide-maude-border",
              children: r.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No recent activity"
              }) : r.map((y, w) => u.jsx(Zy, {
                item: y
              }, w))
            }),
            s === "scheduler" && u.jsxs("div", {
              className: "space-y-2",
              children: [
                (l == null ? void 0 : l.stats) && u.jsxs("div", {
                  className: "grid grid-cols-3 gap-2",
                  children: [
                    u.jsx(Dn, {
                      label: "Total",
                      value: l.stats.total
                    }),
                    u.jsx(Dn, {
                      label: "Active",
                      value: l.stats.active,
                      color: "text-green-400"
                    }),
                    u.jsx(Dn, {
                      label: "Runs",
                      value: l.stats.total_runs
                    })
                  ]
                }),
                ((_i2 = l == null ? void 0 : l.tasks) == null ? void 0 : _i2.length) ? l.tasks.map((y) => u.jsx(Xy, {
                  task: y
                }, y.id)) : u.jsx("p", {
                  className: "py-8 text-center text-sm text-maude-muted",
                  children: "No scheduled tasks"
                })
              ]
            })
          ]
        })
      ]
    });
  }, tx = [
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
  ], nx = () => {
    const e = Ki(), t = ho();
    return u.jsx("nav", {
      className: "safe-bottom flex shrink-0 items-center justify-around border-t border-maude-border bg-maude-surface px-1 pb-1 pt-1",
      children: tx.map((n) => {
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
  $y();
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
  function rx() {
    return u.jsxs("div", {
      className: "flex h-[100dvh] flex-col bg-maude-bg safe-top",
      children: [
        u.jsx("div", {
          className: "min-h-0 flex-1 overflow-hidden",
          children: u.jsx(kv, {})
        }),
        u.jsx(nx, {})
      ]
    });
  }
  const lx = Cv([
    {
      element: u.jsx(rx, {}),
      children: [
        {
          path: "/",
          element: u.jsx($v, {})
        },
        {
          path: "/maude",
          element: u.jsx(dy, {})
        },
        {
          path: "/maude/voice",
          element: u.jsx(jy, {})
        },
        {
          path: "/terminal",
          element: u.jsx(Cy, {})
        },
        {
          path: "/browser",
          element: u.jsx(Ty, {})
        },
        {
          path: "/messages",
          element: u.jsx(by, {})
        },
        {
          path: "/files",
          element: u.jsx(Ly, {})
        },
        {
          path: "/collab",
          element: u.jsx(Qy, {})
        },
        {
          path: "/command-center",
          element: u.jsx(ex, {})
        },
        {
          path: "/settings",
          element: u.jsx(zy, {})
        }
      ]
    }
  ]);
  Xo.createRoot(document.getElementById("root")).render(u.jsx(Ov, {
    router: lx
  }));
})();
export {
  __tla,
  ax as c,
  Qc as g
};
