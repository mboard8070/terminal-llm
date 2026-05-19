let ay, Wc;
let __tla = (async () => {
  function Vc(e, t) {
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
  Wc = function(e) {
    return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
  };
  var Hc = {
    exports: {}
  }, Xa = {}, Qc = {
    exports: {}
  }, q = {};
  var Rl = Symbol.for("react.element"), Mm = Symbol.for("react.portal"), Dm = Symbol.for("react.fragment"), Lm = Symbol.for("react.strict_mode"), Om = Symbol.for("react.profiler"), Am = Symbol.for("react.provider"), Im = Symbol.for("react.context"), zm = Symbol.for("react.forward_ref"), Um = Symbol.for("react.suspense"), Fm = Symbol.for("react.memo"), $m = Symbol.for("react.lazy"), au = Symbol.iterator;
  function Bm(e) {
    return e === null || typeof e != "object" ? null : (e = au && e[au] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var Kc = {
    isMounted: function() {
      return false;
    },
    enqueueForceUpdate: function() {
    },
    enqueueReplaceState: function() {
    },
    enqueueSetState: function() {
    }
  }, Yc = Object.assign, Gc = {};
  function Pr(e, t, n) {
    this.props = e, this.context = t, this.refs = Gc, this.updater = n || Kc;
  }
  Pr.prototype.isReactComponent = {};
  Pr.prototype.setState = function(e, t) {
    if (typeof e != "object" && typeof e != "function" && e != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
    this.updater.enqueueSetState(this, e, t, "setState");
  };
  Pr.prototype.forceUpdate = function(e) {
    this.updater.enqueueForceUpdate(this, e, "forceUpdate");
  };
  function Xc() {
  }
  Xc.prototype = Pr.prototype;
  function Gs(e, t, n) {
    this.props = e, this.context = t, this.refs = Gc, this.updater = n || Kc;
  }
  var Xs = Gs.prototype = new Xc();
  Xs.constructor = Gs;
  Yc(Xs, Pr.prototype);
  Xs.isPureReactComponent = true;
  var ou = Array.isArray, Jc = Object.prototype.hasOwnProperty, Js = {
    current: null
  }, Zc = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function qc(e, t, n) {
    var r, l = {}, a = null, o = null;
    if (t != null) for (r in t.ref !== void 0 && (o = t.ref), t.key !== void 0 && (a = "" + t.key), t) Jc.call(t, r) && !Zc.hasOwnProperty(r) && (l[r] = t[r]);
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
      _owner: Js.current
    };
  }
  function Vm(e, t) {
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
  function Wm(e) {
    var t = {
      "=": "=0",
      ":": "=2"
    };
    return "$" + e.replace(/[=:]/g, function(n) {
      return t[n];
    });
  }
  var su = /\/+/g;
  function wo(e, t) {
    return typeof e == "object" && e !== null && e.key != null ? Wm("" + e.key) : t.toString(36);
  }
  function ia(e, t, n, r, l) {
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
          case Mm:
            o = true;
        }
    }
    if (o) return o = e, l = l(o), e = r === "" ? "." + wo(o, 0) : r, ou(l) ? (n = "", e != null && (n = e.replace(su, "$&/") + "/"), ia(l, t, n, "", function(c) {
      return c;
    })) : l != null && (Zs(l) && (l = Vm(l, n + (!l.key || o && o.key === l.key ? "" : ("" + l.key).replace(su, "$&/") + "/") + e)), t.push(l)), 1;
    if (o = 0, r = r === "" ? "." : r + ":", ou(e)) for (var i = 0; i < e.length; i++) {
      a = e[i];
      var s = r + wo(a, i);
      o += ia(a, t, n, s, l);
    }
    else if (s = Bm(e), typeof s == "function") for (e = s.call(e), i = 0; !(a = e.next()).done; ) a = a.value, s = r + wo(a, i++), o += ia(a, t, n, s, l);
    else if (a === "object") throw t = String(e), Error("Objects are not valid as a React child (found: " + (t === "[object Object]" ? "object with keys {" + Object.keys(e).join(", ") + "}" : t) + "). If you meant to render a collection of children, use an array instead.");
    return o;
  }
  function Bl(e, t, n) {
    if (e == null) return e;
    var r = [], l = 0;
    return ia(e, r, "", "", function(a) {
      return t.call(n, a, l++);
    }), r;
  }
  function Hm(e) {
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
  }, ua = {
    transition: null
  }, Qm = {
    ReactCurrentDispatcher: lt,
    ReactCurrentBatchConfig: ua,
    ReactCurrentOwner: Js
  };
  function ed() {
    throw Error("act(...) is not supported in production builds of React.");
  }
  q.Children = {
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
  q.Component = Pr;
  q.Fragment = Dm;
  q.Profiler = Om;
  q.PureComponent = Gs;
  q.StrictMode = Lm;
  q.Suspense = Um;
  q.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Qm;
  q.act = ed;
  q.cloneElement = function(e, t, n) {
    if (e == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + e + ".");
    var r = Yc({}, e.props), l = e.key, a = e.ref, o = e._owner;
    if (t != null) {
      if (t.ref !== void 0 && (a = t.ref, o = Js.current), t.key !== void 0 && (l = "" + t.key), e.type && e.type.defaultProps) var i = e.type.defaultProps;
      for (s in t) Jc.call(t, s) && !Zc.hasOwnProperty(s) && (r[s] = t[s] === void 0 && i !== void 0 ? i[s] : t[s]);
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
  q.createContext = function(e) {
    return e = {
      $$typeof: Im,
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
  q.createElement = qc;
  q.createFactory = function(e) {
    var t = qc.bind(null, e);
    return t.type = e, t;
  };
  q.createRef = function() {
    return {
      current: null
    };
  };
  q.forwardRef = function(e) {
    return {
      $$typeof: zm,
      render: e
    };
  };
  q.isValidElement = Zs;
  q.lazy = function(e) {
    return {
      $$typeof: $m,
      _payload: {
        _status: -1,
        _result: e
      },
      _init: Hm
    };
  };
  q.memo = function(e, t) {
    return {
      $$typeof: Fm,
      type: e,
      compare: t === void 0 ? null : t
    };
  };
  q.startTransition = function(e) {
    var t = ua.transition;
    ua.transition = {};
    try {
      e();
    } finally {
      ua.transition = t;
    }
  };
  q.unstable_act = ed;
  q.useCallback = function(e, t) {
    return lt.current.useCallback(e, t);
  };
  q.useContext = function(e) {
    return lt.current.useContext(e);
  };
  q.useDebugValue = function() {
  };
  q.useDeferredValue = function(e) {
    return lt.current.useDeferredValue(e);
  };
  q.useEffect = function(e, t) {
    return lt.current.useEffect(e, t);
  };
  q.useId = function() {
    return lt.current.useId();
  };
  q.useImperativeHandle = function(e, t, n) {
    return lt.current.useImperativeHandle(e, t, n);
  };
  q.useInsertionEffect = function(e, t) {
    return lt.current.useInsertionEffect(e, t);
  };
  q.useLayoutEffect = function(e, t) {
    return lt.current.useLayoutEffect(e, t);
  };
  q.useMemo = function(e, t) {
    return lt.current.useMemo(e, t);
  };
  q.useReducer = function(e, t, n) {
    return lt.current.useReducer(e, t, n);
  };
  q.useRef = function(e) {
    return lt.current.useRef(e);
  };
  q.useState = function(e) {
    return lt.current.useState(e);
  };
  q.useSyncExternalStore = function(e, t, n) {
    return lt.current.useSyncExternalStore(e, t, n);
  };
  q.useTransition = function() {
    return lt.current.useTransition();
  };
  q.version = "18.3.1";
  Qc.exports = q;
  var g = Qc.exports;
  const Km = Wc(g), Ym = Vc({
    __proto__: null,
    default: Km
  }, [
    g
  ]);
  var Gm = g, Xm = Symbol.for("react.element"), Jm = Symbol.for("react.fragment"), Zm = Object.prototype.hasOwnProperty, qm = Gm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, ep = {
    key: true,
    ref: true,
    __self: true,
    __source: true
  };
  function td(e, t, n) {
    var r, l = {}, a = null, o = null;
    n !== void 0 && (a = "" + n), t.key !== void 0 && (a = "" + t.key), t.ref !== void 0 && (o = t.ref);
    for (r in t) Zm.call(t, r) && !ep.hasOwnProperty(r) && (l[r] = t[r]);
    if (e && e.defaultProps) for (r in t = e.defaultProps, t) l[r] === void 0 && (l[r] = t[r]);
    return {
      $$typeof: Xm,
      type: e,
      key: a,
      ref: o,
      props: l,
      _owner: qm.current
    };
  }
  Xa.Fragment = Jm;
  Xa.jsx = td;
  Xa.jsxs = td;
  Hc.exports = Xa;
  var u = Hc.exports, Jo = {}, nd = {
    exports: {}
  }, Nt = {}, rd = {
    exports: {}
  }, ld = {};
  (function(e) {
    function t(M, V) {
      var F = M.length;
      M.push(V);
      e: for (; 0 < F; ) {
        var ee = F - 1 >>> 1, X = M[ee];
        if (0 < l(X, V)) M[ee] = V, M[F] = X, F = ee;
        else break e;
      }
    }
    function n(M) {
      return M.length === 0 ? null : M[0];
    }
    function r(M) {
      if (M.length === 0) return null;
      var V = M[0], F = M.pop();
      if (F !== V) {
        M[0] = F;
        e: for (var ee = 0, X = M.length, be = X >>> 1; ee < be; ) {
          var Ee = 2 * (ee + 1) - 1, ge = M[Ee], Se = Ee + 1, J = M[Se];
          if (0 > l(ge, F)) Se < X && 0 > l(J, ge) ? (M[ee] = J, M[Se] = F, ee = Se) : (M[ee] = ge, M[Ee] = F, ee = Ee);
          else if (Se < X && 0 > l(J, F)) M[ee] = J, M[Se] = F, ee = Se;
          else break e;
        }
      }
      return V;
    }
    function l(M, V) {
      var F = M.sortIndex - V.sortIndex;
      return F !== 0 ? F : M.id - V.id;
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
    var s = [], c = [], m = 1, d = null, p = 3, x = false, w = false, k = false, R = typeof setTimeout == "function" ? setTimeout : null, h = typeof clearTimeout == "function" ? clearTimeout : null, f = typeof setImmediate < "u" ? setImmediate : null;
    typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
    function v(M) {
      for (var V = n(c); V !== null; ) {
        if (V.callback === null) r(c);
        else if (V.startTime <= M) r(c), V.sortIndex = V.expirationTime, t(s, V);
        else break;
        V = n(c);
      }
    }
    function E(M) {
      if (k = false, v(M), !w) if (n(s) !== null) w = true, Qe(_);
      else {
        var V = n(c);
        V !== null && pt(E, V.startTime - M);
      }
    }
    function _(M, V) {
      w = false, k && (k = false, h(j), j = -1), x = true;
      var F = p;
      try {
        for (v(V), d = n(s); d !== null && (!(d.expirationTime > V) || M && !H()); ) {
          var ee = d.callback;
          if (typeof ee == "function") {
            d.callback = null, p = d.priorityLevel;
            var X = ee(d.expirationTime <= V);
            V = e.unstable_now(), typeof X == "function" ? d.callback = X : d === n(s) && r(s), v(V);
          } else r(s);
          d = n(s);
        }
        if (d !== null) var be = true;
        else {
          var Ee = n(c);
          Ee !== null && pt(E, Ee.startTime - V), be = false;
        }
        return be;
      } finally {
        d = null, p = F, x = false;
      }
    }
    var b = false, S = null, j = -1, z = 5, D = -1;
    function H() {
      return !(e.unstable_now() - D < z);
    }
    function K() {
      if (S !== null) {
        var M = e.unstable_now();
        D = M;
        var V = true;
        try {
          V = S(true, M);
        } finally {
          V ? se() : (b = false, S = null);
        }
      } else b = false;
    }
    var se;
    if (typeof f == "function") se = function() {
      f(K);
    };
    else if (typeof MessageChannel < "u") {
      var le = new MessageChannel(), je = le.port2;
      le.port1.onmessage = K, se = function() {
        je.postMessage(null);
      };
    } else se = function() {
      R(K, 0);
    };
    function Qe(M) {
      S = M, b || (b = true, se());
    }
    function pt(M, V) {
      j = R(function() {
        M(e.unstable_now());
      }, V);
    }
    e.unstable_IdlePriority = 5, e.unstable_ImmediatePriority = 1, e.unstable_LowPriority = 4, e.unstable_NormalPriority = 3, e.unstable_Profiling = null, e.unstable_UserBlockingPriority = 2, e.unstable_cancelCallback = function(M) {
      M.callback = null;
    }, e.unstable_continueExecution = function() {
      w || x || (w = true, Qe(_));
    }, e.unstable_forceFrameRate = function(M) {
      0 > M || 125 < M ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : z = 0 < M ? Math.floor(1e3 / M) : 5;
    }, e.unstable_getCurrentPriorityLevel = function() {
      return p;
    }, e.unstable_getFirstCallbackNode = function() {
      return n(s);
    }, e.unstable_next = function(M) {
      switch (p) {
        case 1:
        case 2:
        case 3:
          var V = 3;
          break;
        default:
          V = p;
      }
      var F = p;
      p = V;
      try {
        return M();
      } finally {
        p = F;
      }
    }, e.unstable_pauseExecution = function() {
    }, e.unstable_requestPaint = function() {
    }, e.unstable_runWithPriority = function(M, V) {
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
      var F = p;
      p = M;
      try {
        return V();
      } finally {
        p = F;
      }
    }, e.unstable_scheduleCallback = function(M, V, F) {
      var ee = e.unstable_now();
      switch (typeof F == "object" && F !== null ? (F = F.delay, F = typeof F == "number" && 0 < F ? ee + F : ee) : F = ee, M) {
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
      return X = F + X, M = {
        id: m++,
        callback: V,
        priorityLevel: M,
        startTime: F,
        expirationTime: X,
        sortIndex: -1
      }, F > ee ? (M.sortIndex = F, t(c, M), n(s) === null && M === n(c) && (k ? (h(j), j = -1) : k = true, pt(E, F - ee))) : (M.sortIndex = X, t(s, M), w || x || (w = true, Qe(_))), M;
    }, e.unstable_shouldYield = H, e.unstable_wrapCallback = function(M) {
      var V = p;
      return function() {
        var F = p;
        p = V;
        try {
          return M.apply(this, arguments);
        } finally {
          p = F;
        }
      };
    };
  })(ld);
  rd.exports = ld;
  var tp = rd.exports;
  var np = g, kt = tp;
  function P(e) {
    for (var t = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, n = 1; n < arguments.length; n++) t += "&args[]=" + encodeURIComponent(arguments[n]);
    return "Minified React error #" + e + "; visit " + t + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  var ad = /* @__PURE__ */ new Set(), ul = {};
  function Xn(e, t) {
    jr(e, t), jr(e + "Capture", t);
  }
  function jr(e, t) {
    for (ul[e] = t, e = 0; e < t.length; e++) ad.add(t[e]);
  }
  var en = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), Zo = Object.prototype.hasOwnProperty, rp = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, iu = {}, uu = {};
  function lp(e) {
    return Zo.call(uu, e) ? true : Zo.call(iu, e) ? false : rp.test(e) ? uu[e] = true : (iu[e] = true, false);
  }
  function ap(e, t, n, r) {
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
  function op(e, t, n, r) {
    if (t === null || typeof t > "u" || ap(e, t, n, r)) return true;
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
  var Xe = {};
  "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e) {
    Xe[e] = new at(e, 0, false, e, null, false, false);
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
    Xe[t] = new at(t, 1, false, e[1], null, false, false);
  });
  [
    "contentEditable",
    "draggable",
    "spellCheck",
    "value"
  ].forEach(function(e) {
    Xe[e] = new at(e, 2, false, e.toLowerCase(), null, false, false);
  });
  [
    "autoReverse",
    "externalResourcesRequired",
    "focusable",
    "preserveAlpha"
  ].forEach(function(e) {
    Xe[e] = new at(e, 2, false, e, null, false, false);
  });
  "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e) {
    Xe[e] = new at(e, 3, false, e.toLowerCase(), null, false, false);
  });
  [
    "checked",
    "multiple",
    "muted",
    "selected"
  ].forEach(function(e) {
    Xe[e] = new at(e, 3, true, e, null, false, false);
  });
  [
    "capture",
    "download"
  ].forEach(function(e) {
    Xe[e] = new at(e, 4, false, e, null, false, false);
  });
  [
    "cols",
    "rows",
    "size",
    "span"
  ].forEach(function(e) {
    Xe[e] = new at(e, 6, false, e, null, false, false);
  });
  [
    "rowSpan",
    "start"
  ].forEach(function(e) {
    Xe[e] = new at(e, 5, false, e.toLowerCase(), null, false, false);
  });
  var qs = /[\-:]([a-z])/g;
  function ei(e) {
    return e[1].toUpperCase();
  }
  "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e) {
    var t = e.replace(qs, ei);
    Xe[t] = new at(t, 1, false, e, null, false, false);
  });
  "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e) {
    var t = e.replace(qs, ei);
    Xe[t] = new at(t, 1, false, e, "http://www.w3.org/1999/xlink", false, false);
  });
  [
    "xml:base",
    "xml:lang",
    "xml:space"
  ].forEach(function(e) {
    var t = e.replace(qs, ei);
    Xe[t] = new at(t, 1, false, e, "http://www.w3.org/XML/1998/namespace", false, false);
  });
  [
    "tabIndex",
    "crossOrigin"
  ].forEach(function(e) {
    Xe[e] = new at(e, 1, false, e.toLowerCase(), null, false, false);
  });
  Xe.xlinkHref = new at("xlinkHref", 1, false, "xlink:href", "http://www.w3.org/1999/xlink", true, false);
  [
    "src",
    "href",
    "action",
    "formAction"
  ].forEach(function(e) {
    Xe[e] = new at(e, 1, false, e.toLowerCase(), null, true, true);
  });
  function ti(e, t, n, r) {
    var l = Xe.hasOwnProperty(t) ? Xe[t] : null;
    (l !== null ? l.type !== 0 : r || !(2 < t.length) || t[0] !== "o" && t[0] !== "O" || t[1] !== "n" && t[1] !== "N") && (op(t, n, l, r) && (n = null), r || l === null ? lp(t) && (n === null ? e.removeAttribute(t) : e.setAttribute(t, "" + n)) : l.mustUseProperty ? e[l.propertyName] = n === null ? l.type === 3 ? false : "" : n : (t = l.attributeName, r = l.attributeNamespace, n === null ? e.removeAttribute(t) : (l = l.type, n = l === 3 || l === 4 && n === true ? "" : "" + n, r ? e.setAttributeNS(r, t, n) : e.setAttribute(t, n))));
  }
  var ln = np.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, Vl = Symbol.for("react.element"), or = Symbol.for("react.portal"), sr = Symbol.for("react.fragment"), ni = Symbol.for("react.strict_mode"), qo = Symbol.for("react.profiler"), od = Symbol.for("react.provider"), sd = Symbol.for("react.context"), ri = Symbol.for("react.forward_ref"), es = Symbol.for("react.suspense"), ts = Symbol.for("react.suspense_list"), li = Symbol.for("react.memo"), cn = Symbol.for("react.lazy"), id = Symbol.for("react.offscreen"), cu = Symbol.iterator;
  function Ir(e) {
    return e === null || typeof e != "object" ? null : (e = cu && e[cu] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var Re = Object.assign, So;
  function Yr(e) {
    if (So === void 0) try {
      throw Error();
    } catch (n) {
      var t = n.stack.trim().match(/\n( *(at )?)/);
      So = t && t[1] || "";
    }
    return `
` + So + e;
  }
  var ko = false;
  function No(e, t) {
    if (!e || ko) return "";
    ko = true;
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
      ko = false, Error.prepareStackTrace = n;
    }
    return (e = e ? e.displayName || e.name : "") ? Yr(e) : "";
  }
  function sp(e) {
    switch (e.tag) {
      case 5:
        return Yr(e.type);
      case 16:
        return Yr("Lazy");
      case 13:
        return Yr("Suspense");
      case 19:
        return Yr("SuspenseList");
      case 0:
      case 2:
      case 15:
        return e = No(e.type, false), e;
      case 11:
        return e = No(e.type.render, false), e;
      case 1:
        return e = No(e.type, true), e;
      default:
        return "";
    }
  }
  function ns(e) {
    if (e == null) return null;
    if (typeof e == "function") return e.displayName || e.name || null;
    if (typeof e == "string") return e;
    switch (e) {
      case sr:
        return "Fragment";
      case or:
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
      case sd:
        return (e.displayName || "Context") + ".Consumer";
      case od:
        return (e._context.displayName || "Context") + ".Provider";
      case ri:
        var t = e.render;
        return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
      case li:
        return t = e.displayName || null, t !== null ? t : ns(e.type) || "Memo";
      case cn:
        t = e._payload, e = e._init;
        try {
          return ns(e(t));
        } catch {
        }
    }
    return null;
  }
  function ip(e) {
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
  function ud(e) {
    var t = e.type;
    return (e = e.nodeName) && e.toLowerCase() === "input" && (t === "checkbox" || t === "radio");
  }
  function up(e) {
    var t = ud(e) ? "checked" : "value", n = Object.getOwnPropertyDescriptor(e.constructor.prototype, t), r = "" + e[t];
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
    e._valueTracker || (e._valueTracker = up(e));
  }
  function cd(e) {
    if (!e) return false;
    var t = e._valueTracker;
    if (!t) return true;
    var n = t.getValue(), r = "";
    return e && (r = ud(e) ? e.checked ? "true" : "false" : e.value), e = r, e !== n ? (t.setValue(e), true) : false;
  }
  function ka(e) {
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
    n = Cn(t.value != null ? t.value : n), e._wrapperState = {
      initialChecked: r,
      initialValue: n,
      controlled: t.type === "checkbox" || t.type === "radio" ? t.checked != null : t.value != null
    };
  }
  function dd(e, t) {
    t = t.checked, t != null && ti(e, "checked", t, false);
  }
  function ls(e, t) {
    dd(e, t);
    var n = Cn(t.value), r = t.type;
    if (n != null) r === "number" ? (n === 0 && e.value === "" || e.value != n) && (e.value = "" + n) : e.value !== "" + n && (e.value = "" + n);
    else if (r === "submit" || r === "reset") {
      e.removeAttribute("value");
      return;
    }
    t.hasOwnProperty("value") ? as(e, t.type, n) : t.hasOwnProperty("defaultValue") && as(e, t.type, Cn(t.defaultValue)), t.checked == null && t.defaultChecked != null && (e.defaultChecked = !!t.defaultChecked);
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
    (t !== "number" || ka(e.ownerDocument) !== e) && (n == null ? e.defaultValue = "" + e._wrapperState.initialValue : e.defaultValue !== "" + n && (e.defaultValue = "" + n));
  }
  var Gr = Array.isArray;
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
  function os(e, t) {
    if (t.dangerouslySetInnerHTML != null) throw Error(P(91));
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
        if (t != null) throw Error(P(92));
        if (Gr(n)) {
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
  function fd(e, t) {
    var n = Cn(t.value), r = Cn(t.defaultValue);
    n != null && (n = "" + n, n !== e.value && (e.value = n), t.defaultValue == null && e.defaultValue !== n && (e.defaultValue = n)), r != null && (e.defaultValue = "" + r);
  }
  function pu(e) {
    var t = e.textContent;
    t === e._wrapperState.initialValue && t !== "" && t !== null && (e.value = t);
  }
  function md(e) {
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
    return e == null || e === "http://www.w3.org/1999/xhtml" ? md(t) : e === "http://www.w3.org/2000/svg" && t === "foreignObject" ? "http://www.w3.org/1999/xhtml" : e;
  }
  var Hl, pd = function(e) {
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
  }, cp = [
    "Webkit",
    "ms",
    "Moz",
    "O"
  ];
  Object.keys(qr).forEach(function(e) {
    cp.forEach(function(t) {
      t = t + e.charAt(0).toUpperCase() + e.substring(1), qr[t] = qr[e];
    });
  });
  function hd(e, t, n) {
    return t == null || typeof t == "boolean" || t === "" ? "" : n || typeof t != "number" || t === 0 || qr.hasOwnProperty(e) && qr[e] ? ("" + t).trim() : t + "px";
  }
  function gd(e, t) {
    e = e.style;
    for (var n in t) if (t.hasOwnProperty(n)) {
      var r = n.indexOf("--") === 0, l = hd(n, t[n], r);
      n === "float" && (n = "cssFloat"), r ? e.setProperty(n, l) : e[n] = l;
    }
  }
  var dp = Re({
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
      if (dp[e] && (t.children != null || t.dangerouslySetInnerHTML != null)) throw Error(P(137, e));
      if (t.dangerouslySetInnerHTML != null) {
        if (t.children != null) throw Error(P(60));
        if (typeof t.dangerouslySetInnerHTML != "object" || !("__html" in t.dangerouslySetInnerHTML)) throw Error(P(61));
      }
      if (t.style != null && typeof t.style != "object") throw Error(P(62));
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
  var ds = null, yr = null, wr = null;
  function hu(e) {
    if (e = Pl(e)) {
      if (typeof ds != "function") throw Error(P(280));
      var t = e.stateNode;
      t && (t = to(t), ds(e.stateNode, e.type, t));
    }
  }
  function vd(e) {
    yr ? wr ? wr.push(e) : wr = [
      e
    ] : yr = e;
  }
  function xd() {
    if (yr) {
      var e = yr, t = wr;
      if (wr = yr = null, hu(e), t) for (e = 0; e < t.length; e++) hu(t[e]);
    }
  }
  function yd(e, t) {
    return e(t);
  }
  function wd() {
  }
  var jo = false;
  function Sd(e, t, n) {
    if (jo) return e(t, n);
    jo = true;
    try {
      return yd(e, t, n);
    } finally {
      jo = false, (yr !== null || wr !== null) && (wd(), xd());
    }
  }
  function dl(e, t) {
    var n = e.stateNode;
    if (n === null) return null;
    var r = to(n);
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
  var fs = false;
  if (en) try {
    var zr = {};
    Object.defineProperty(zr, "passive", {
      get: function() {
        fs = true;
      }
    }), window.addEventListener("test", zr, zr), window.removeEventListener("test", zr, zr);
  } catch {
    fs = false;
  }
  function fp(e, t, n, r, l, a, o, i, s) {
    var c = Array.prototype.slice.call(arguments, 3);
    try {
      t.apply(n, c);
    } catch (m) {
      this.onError(m);
    }
  }
  var el = false, Na = null, ja = false, ms = null, mp = {
    onError: function(e) {
      el = true, Na = e;
    }
  };
  function pp(e, t, n, r, l, a, o, i, s) {
    el = false, Na = null, fp.apply(mp, arguments);
  }
  function hp(e, t, n, r, l, a, o, i, s) {
    if (pp.apply(this, arguments), el) {
      if (el) {
        var c = Na;
        el = false, Na = null;
      } else throw Error(P(198));
      ja || (ja = true, ms = c);
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
  function kd(e) {
    if (e.tag === 13) {
      var t = e.memoizedState;
      if (t === null && (e = e.alternate, e !== null && (t = e.memoizedState)), t !== null) return t.dehydrated;
    }
    return null;
  }
  function gu(e) {
    if (Jn(e) !== e) throw Error(P(188));
  }
  function gp(e) {
    var t = e.alternate;
    if (!t) {
      if (t = Jn(e), t === null) throw Error(P(188));
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
        throw Error(P(188));
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
          if (!o) throw Error(P(189));
        }
      }
      if (n.alternate !== r) throw Error(P(190));
    }
    if (n.tag !== 3) throw Error(P(188));
    return n.stateNode.current === n ? e : t;
  }
  function Nd(e) {
    return e = gp(e), e !== null ? jd(e) : null;
  }
  function jd(e) {
    if (e.tag === 5 || e.tag === 6) return e;
    for (e = e.child; e !== null; ) {
      var t = jd(e);
      if (t !== null) return t;
      e = e.sibling;
    }
    return null;
  }
  var Ed = kt.unstable_scheduleCallback, vu = kt.unstable_cancelCallback, vp = kt.unstable_shouldYield, xp = kt.unstable_requestPaint, Le = kt.unstable_now, yp = kt.unstable_getCurrentPriorityLevel, oi = kt.unstable_ImmediatePriority, Cd = kt.unstable_UserBlockingPriority, Ea = kt.unstable_NormalPriority, wp = kt.unstable_LowPriority, _d = kt.unstable_IdlePriority, Ja = null, Ht = null;
  function Sp(e) {
    if (Ht && typeof Ht.onCommitFiberRoot == "function") try {
      Ht.onCommitFiberRoot(Ja, e, void 0, (e.current.flags & 128) === 128);
    } catch {
    }
  }
  var It = Math.clz32 ? Math.clz32 : jp, kp = Math.log, Np = Math.LN2;
  function jp(e) {
    return e >>>= 0, e === 0 ? 32 : 31 - (kp(e) / Np | 0) | 0;
  }
  var Ql = 64, Kl = 4194304;
  function Xr(e) {
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
  function Ca(e, t) {
    var n = e.pendingLanes;
    if (n === 0) return 0;
    var r = 0, l = e.suspendedLanes, a = e.pingedLanes, o = n & 268435455;
    if (o !== 0) {
      var i = o & ~l;
      i !== 0 ? r = Xr(i) : (a &= o, a !== 0 && (r = Xr(a)));
    } else o = n & ~l, o !== 0 ? r = Xr(o) : a !== 0 && (r = Xr(a));
    if (r === 0) return 0;
    if (t !== 0 && t !== r && !(t & l) && (l = r & -r, a = t & -t, l >= a || l === 16 && (a & 4194240) !== 0)) return t;
    if (r & 4 && (r |= n & 16), t = e.entangledLanes, t !== 0) for (e = e.entanglements, t &= r; 0 < t; ) n = 31 - It(t), l = 1 << n, r |= e[n], t &= ~l;
    return r;
  }
  function Ep(e, t) {
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
  function Cp(e, t) {
    for (var n = e.suspendedLanes, r = e.pingedLanes, l = e.expirationTimes, a = e.pendingLanes; 0 < a; ) {
      var o = 31 - It(a), i = 1 << o, s = l[o];
      s === -1 ? (!(i & n) || i & r) && (l[o] = Ep(i, t)) : s <= t && (e.expiredLanes |= i), a &= ~i;
    }
  }
  function ps(e) {
    return e = e.pendingLanes & -1073741825, e !== 0 ? e : e & 1073741824 ? 1073741824 : 0;
  }
  function Rd() {
    var e = Ql;
    return Ql <<= 1, !(Ql & 4194240) && (Ql = 64), e;
  }
  function Eo(e) {
    for (var t = [], n = 0; 31 > n; n++) t.push(e);
    return t;
  }
  function bl(e, t, n) {
    e.pendingLanes |= t, t !== 536870912 && (e.suspendedLanes = 0, e.pingedLanes = 0), e = e.eventTimes, t = 31 - It(t), e[t] = n;
  }
  function _p(e, t) {
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
  var de = 0;
  function bd(e) {
    return e &= -e, 1 < e ? 4 < e ? e & 268435455 ? 16 : 536870912 : 4 : 1;
  }
  var Td, ii, Pd, Md, Dd, hs = false, Yl = [], vn = null, xn = null, yn = null, fl = /* @__PURE__ */ new Map(), ml = /* @__PURE__ */ new Map(), fn = [], Rp = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
  function xu(e, t) {
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
    }, t !== null && (t = Pl(t), t !== null && ii(t)), e) : (e.eventSystemFlags |= r, t = e.targetContainers, l !== null && t.indexOf(l) === -1 && t.push(l), e);
  }
  function bp(e, t, n, r, l) {
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
  function Ld(e) {
    var t = zn(e.target);
    if (t !== null) {
      var n = Jn(t);
      if (n !== null) {
        if (t = n.tag, t === 13) {
          if (t = kd(n), t !== null) {
            e.blockedOn = t, Dd(e.priority, function() {
              Pd(n);
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
  function ca(e) {
    if (e.blockedOn !== null) return false;
    for (var t = e.targetContainers; 0 < t.length; ) {
      var n = gs(e.domEventName, e.eventSystemFlags, t[0], e.nativeEvent);
      if (n === null) {
        n = e.nativeEvent;
        var r = new n.constructor(n.type, n);
        cs = r, n.target.dispatchEvent(r), cs = null;
      } else return t = Pl(n), t !== null && ii(t), e.blockedOn = n, false;
      t.shift();
    }
    return true;
  }
  function yu(e, t, n) {
    ca(e) && n.delete(t);
  }
  function Tp() {
    hs = false, vn !== null && ca(vn) && (vn = null), xn !== null && ca(xn) && (xn = null), yn !== null && ca(yn) && (yn = null), fl.forEach(yu), ml.forEach(yu);
  }
  function Fr(e, t) {
    e.blockedOn === t && (e.blockedOn = null, hs || (hs = true, kt.unstable_scheduleCallback(kt.unstable_NormalPriority, Tp)));
  }
  function pl(e) {
    function t(l) {
      return Fr(l, e);
    }
    if (0 < Yl.length) {
      Fr(Yl[0], e);
      for (var n = 1; n < Yl.length; n++) {
        var r = Yl[n];
        r.blockedOn === e && (r.blockedOn = null);
      }
    }
    for (vn !== null && Fr(vn, e), xn !== null && Fr(xn, e), yn !== null && Fr(yn, e), fl.forEach(t), ml.forEach(t), n = 0; n < fn.length; n++) r = fn[n], r.blockedOn === e && (r.blockedOn = null);
    for (; 0 < fn.length && (n = fn[0], n.blockedOn === null); ) Ld(n), n.blockedOn === null && fn.shift();
  }
  var Sr = ln.ReactCurrentBatchConfig, _a = true;
  function Pp(e, t, n, r) {
    var l = de, a = Sr.transition;
    Sr.transition = null;
    try {
      de = 1, ui(e, t, n, r);
    } finally {
      de = l, Sr.transition = a;
    }
  }
  function Mp(e, t, n, r) {
    var l = de, a = Sr.transition;
    Sr.transition = null;
    try {
      de = 4, ui(e, t, n, r);
    } finally {
      de = l, Sr.transition = a;
    }
  }
  function ui(e, t, n, r) {
    if (_a) {
      var l = gs(e, t, n, r);
      if (l === null) Oo(e, t, r, Ra, n), xu(e, r);
      else if (bp(l, e, t, n, r)) r.stopPropagation();
      else if (xu(e, r), t & 4 && -1 < Rp.indexOf(e)) {
        for (; l !== null; ) {
          var a = Pl(l);
          if (a !== null && Td(a), a = gs(e, t, n, r), a === null && Oo(e, t, r, Ra, n), a === l) break;
          l = a;
        }
        l !== null && r.stopPropagation();
      } else Oo(e, t, r, null, n);
    }
  }
  var Ra = null;
  function gs(e, t, n, r) {
    if (Ra = null, e = ai(r), e = zn(e), e !== null) if (t = Jn(e), t === null) e = null;
    else if (n = t.tag, n === 13) {
      if (e = kd(t), e !== null) return e;
      e = null;
    } else if (n === 3) {
      if (t.stateNode.current.memoizedState.isDehydrated) return t.tag === 3 ? t.stateNode.containerInfo : null;
      e = null;
    } else t !== e && (e = null);
    return Ra = e, null;
  }
  function Od(e) {
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
        switch (yp()) {
          case oi:
            return 1;
          case Cd:
            return 4;
          case Ea:
          case wp:
            return 16;
          case _d:
            return 536870912;
          default:
            return 16;
        }
      default:
        return 16;
    }
  }
  var pn = null, ci = null, da = null;
  function Ad() {
    if (da) return da;
    var e, t = ci, n = t.length, r, l = "value" in pn ? pn.value : pn.textContent, a = l.length;
    for (e = 0; e < n && t[e] === l[e]; e++) ;
    var o = n - e;
    for (r = 1; r <= o && t[n - r] === l[a - r]; r++) ;
    return da = l.slice(e, 1 < r ? 1 - r : void 0);
  }
  function fa(e) {
    var t = e.keyCode;
    return "charCode" in e ? (e = e.charCode, e === 0 && t === 13 && (e = 13)) : e = t, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
  }
  function Gl() {
    return true;
  }
  function wu() {
    return false;
  }
  function jt(e) {
    function t(n, r, l, a, o) {
      this._reactName = n, this._targetInst = l, this.type = r, this.nativeEvent = a, this.target = o, this.currentTarget = null;
      for (var i in e) e.hasOwnProperty(i) && (n = e[i], this[i] = n ? n(a) : a[i]);
      return this.isDefaultPrevented = (a.defaultPrevented != null ? a.defaultPrevented : a.returnValue === false) ? Gl : wu, this.isPropagationStopped = wu, this;
    }
    return Re(t.prototype, {
      preventDefault: function() {
        this.defaultPrevented = true;
        var n = this.nativeEvent;
        n && (n.preventDefault ? n.preventDefault() : typeof n.returnValue != "unknown" && (n.returnValue = false), this.isDefaultPrevented = Gl);
      },
      stopPropagation: function() {
        var n = this.nativeEvent;
        n && (n.stopPropagation ? n.stopPropagation() : typeof n.cancelBubble != "unknown" && (n.cancelBubble = true), this.isPropagationStopped = Gl);
      },
      persist: function() {
      },
      isPersistent: Gl
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
  }, di = jt(Mr), Tl = Re({}, Mr, {
    view: 0,
    detail: 0
  }), Dp = jt(Tl), Co, _o, $r, Za = Re({}, Tl, {
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
      return "movementX" in e ? e.movementX : (e !== $r && ($r && e.type === "mousemove" ? (Co = e.screenX - $r.screenX, _o = e.screenY - $r.screenY) : _o = Co = 0, $r = e), Co);
    },
    movementY: function(e) {
      return "movementY" in e ? e.movementY : _o;
    }
  }), Su = jt(Za), Lp = Re({}, Za, {
    dataTransfer: 0
  }), Op = jt(Lp), Ap = Re({}, Tl, {
    relatedTarget: 0
  }), Ro = jt(Ap), Ip = Re({}, Mr, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), zp = jt(Ip), Up = Re({}, Mr, {
    clipboardData: function(e) {
      return "clipboardData" in e ? e.clipboardData : window.clipboardData;
    }
  }), Fp = jt(Up), $p = Re({}, Mr, {
    data: 0
  }), ku = jt($p), Bp = {
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
  }, Vp = {
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
  }, Wp = {
    Alt: "altKey",
    Control: "ctrlKey",
    Meta: "metaKey",
    Shift: "shiftKey"
  };
  function Hp(e) {
    var t = this.nativeEvent;
    return t.getModifierState ? t.getModifierState(e) : (e = Wp[e]) ? !!t[e] : false;
  }
  function fi() {
    return Hp;
  }
  var Qp = Re({}, Tl, {
    key: function(e) {
      if (e.key) {
        var t = Bp[e.key] || e.key;
        if (t !== "Unidentified") return t;
      }
      return e.type === "keypress" ? (e = fa(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? Vp[e.keyCode] || "Unidentified" : "";
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
      return e.type === "keypress" ? fa(e) : 0;
    },
    keyCode: function(e) {
      return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
    },
    which: function(e) {
      return e.type === "keypress" ? fa(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
    }
  }), Kp = jt(Qp), Yp = Re({}, Za, {
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
  }), Nu = jt(Yp), Gp = Re({}, Tl, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: fi
  }), Xp = jt(Gp), Jp = Re({}, Mr, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Zp = jt(Jp), qp = Re({}, Za, {
    deltaX: function(e) {
      return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
    },
    deltaY: function(e) {
      return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), eh = jt(qp), th = [
    9,
    13,
    27,
    32
  ], mi = en && "CompositionEvent" in window, tl = null;
  en && "documentMode" in document && (tl = document.documentMode);
  var nh = en && "TextEvent" in window && !tl, Id = en && (!mi || tl && 8 < tl && 11 >= tl), ju = " ", Eu = false;
  function zd(e, t) {
    switch (e) {
      case "keyup":
        return th.indexOf(t.keyCode) !== -1;
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
  function Ud(e) {
    return e = e.detail, typeof e == "object" && "data" in e ? e.data : null;
  }
  var ir = false;
  function rh(e, t) {
    switch (e) {
      case "compositionend":
        return Ud(t);
      case "keypress":
        return t.which !== 32 ? null : (Eu = true, ju);
      case "textInput":
        return e = t.data, e === ju && Eu ? null : e;
      default:
        return null;
    }
  }
  function lh(e, t) {
    if (ir) return e === "compositionend" || !mi && zd(e, t) ? (e = Ad(), da = ci = pn = null, ir = false, e) : null;
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
        return Id && t.locale !== "ko" ? null : t.data;
      default:
        return null;
    }
  }
  var ah = {
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
    return t === "input" ? !!ah[e.type] : t === "textarea";
  }
  function Fd(e, t, n, r) {
    vd(r), t = ba(t, "onChange"), 0 < t.length && (n = new di("onChange", "change", null, n, r), e.push({
      event: n,
      listeners: t
    }));
  }
  var nl = null, hl = null;
  function oh(e) {
    Jd(e, 0);
  }
  function qa(e) {
    var t = dr(e);
    if (cd(t)) return e;
  }
  function sh(e, t) {
    if (e === "change") return t;
  }
  var $d = false;
  if (en) {
    var bo;
    if (en) {
      var To = "oninput" in document;
      if (!To) {
        var _u = document.createElement("div");
        _u.setAttribute("oninput", "return;"), To = typeof _u.oninput == "function";
      }
      bo = To;
    } else bo = false;
    $d = bo && (!document.documentMode || 9 < document.documentMode);
  }
  function Ru() {
    nl && (nl.detachEvent("onpropertychange", Bd), hl = nl = null);
  }
  function Bd(e) {
    if (e.propertyName === "value" && qa(hl)) {
      var t = [];
      Fd(t, hl, e, ai(e)), Sd(oh, t);
    }
  }
  function ih(e, t, n) {
    e === "focusin" ? (Ru(), nl = t, hl = n, nl.attachEvent("onpropertychange", Bd)) : e === "focusout" && Ru();
  }
  function uh(e) {
    if (e === "selectionchange" || e === "keyup" || e === "keydown") return qa(hl);
  }
  function ch(e, t) {
    if (e === "click") return qa(t);
  }
  function dh(e, t) {
    if (e === "input" || e === "change") return qa(t);
  }
  function fh(e, t) {
    return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
  }
  var Ut = typeof Object.is == "function" ? Object.is : fh;
  function gl(e, t) {
    if (Ut(e, t)) return true;
    if (typeof e != "object" || e === null || typeof t != "object" || t === null) return false;
    var n = Object.keys(e), r = Object.keys(t);
    if (n.length !== r.length) return false;
    for (r = 0; r < n.length; r++) {
      var l = n[r];
      if (!Zo.call(t, l) || !Ut(e[l], t[l])) return false;
    }
    return true;
  }
  function bu(e) {
    for (; e && e.firstChild; ) e = e.firstChild;
    return e;
  }
  function Tu(e, t) {
    var n = bu(e);
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
      n = bu(n);
    }
  }
  function Vd(e, t) {
    return e && t ? e === t ? true : e && e.nodeType === 3 ? false : t && t.nodeType === 3 ? Vd(e, t.parentNode) : "contains" in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : false : false;
  }
  function Wd() {
    for (var e = window, t = ka(); t instanceof e.HTMLIFrameElement; ) {
      try {
        var n = typeof t.contentWindow.location.href == "string";
      } catch {
        n = false;
      }
      if (n) e = t.contentWindow;
      else break;
      t = ka(e.document);
    }
    return t;
  }
  function pi(e) {
    var t = e && e.nodeName && e.nodeName.toLowerCase();
    return t && (t === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || t === "textarea" || e.contentEditable === "true");
  }
  function mh(e) {
    var t = Wd(), n = e.focusedElem, r = e.selectionRange;
    if (t !== n && n && n.ownerDocument && Vd(n.ownerDocument.documentElement, n)) {
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
  var ph = en && "documentMode" in document && 11 >= document.documentMode, ur = null, vs = null, rl = null, xs = false;
  function Pu(e, t, n) {
    var r = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
    xs || ur == null || ur !== ka(r) || (r = ur, "selectionStart" in r && pi(r) ? r = {
      start: r.selectionStart,
      end: r.selectionEnd
    } : (r = (r.ownerDocument && r.ownerDocument.defaultView || window).getSelection(), r = {
      anchorNode: r.anchorNode,
      anchorOffset: r.anchorOffset,
      focusNode: r.focusNode,
      focusOffset: r.focusOffset
    }), rl && gl(rl, r) || (rl = r, r = ba(vs, "onSelect"), 0 < r.length && (t = new di("onSelect", "select", null, t, n), e.push({
      event: t,
      listeners: r
    }), t.target = ur)));
  }
  function Xl(e, t) {
    var n = {};
    return n[e.toLowerCase()] = t.toLowerCase(), n["Webkit" + e] = "webkit" + t, n["Moz" + e] = "moz" + t, n;
  }
  var cr = {
    animationend: Xl("Animation", "AnimationEnd"),
    animationiteration: Xl("Animation", "AnimationIteration"),
    animationstart: Xl("Animation", "AnimationStart"),
    transitionend: Xl("Transition", "TransitionEnd")
  }, Po = {}, Hd = {};
  en && (Hd = document.createElement("div").style, "AnimationEvent" in window || (delete cr.animationend.animation, delete cr.animationiteration.animation, delete cr.animationstart.animation), "TransitionEvent" in window || delete cr.transitionend.transition);
  function eo(e) {
    if (Po[e]) return Po[e];
    if (!cr[e]) return e;
    var t = cr[e], n;
    for (n in t) if (t.hasOwnProperty(n) && n in Hd) return Po[e] = t[n];
    return e;
  }
  var Qd = eo("animationend"), Kd = eo("animationiteration"), Yd = eo("animationstart"), Gd = eo("transitionend"), Xd = /* @__PURE__ */ new Map(), Mu = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
  function Rn(e, t) {
    Xd.set(e, t), Xn(t, [
      e
    ]);
  }
  for (var Mo = 0; Mo < Mu.length; Mo++) {
    var Do = Mu[Mo], hh = Do.toLowerCase(), gh = Do[0].toUpperCase() + Do.slice(1);
    Rn(hh, "on" + gh);
  }
  Rn(Qd, "onAnimationEnd");
  Rn(Kd, "onAnimationIteration");
  Rn(Yd, "onAnimationStart");
  Rn("dblclick", "onDoubleClick");
  Rn("focusin", "onFocus");
  Rn("focusout", "onBlur");
  Rn(Gd, "onTransitionEnd");
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
  var Jr = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), vh = new Set("cancel close invalid load scroll toggle".split(" ").concat(Jr));
  function Du(e, t, n) {
    var r = e.type || "unknown-event";
    e.currentTarget = n, hp(r, t, void 0, e), e.currentTarget = null;
  }
  function Jd(e, t) {
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
    if (ja) throw e = ms, ja = false, ms = null, e;
  }
  function ye(e, t) {
    var n = t[Ns];
    n === void 0 && (n = t[Ns] = /* @__PURE__ */ new Set());
    var r = e + "__bubble";
    n.has(r) || (Zd(t, e, 2, false), n.add(r));
  }
  function Lo(e, t, n) {
    var r = 0;
    t && (r |= 4), Zd(n, e, r, t);
  }
  var Jl = "_reactListening" + Math.random().toString(36).slice(2);
  function vl(e) {
    if (!e[Jl]) {
      e[Jl] = true, ad.forEach(function(n) {
        n !== "selectionchange" && (vh.has(n) || Lo(n, false, e), Lo(n, true, e));
      });
      var t = e.nodeType === 9 ? e : e.ownerDocument;
      t === null || t[Jl] || (t[Jl] = true, Lo("selectionchange", false, t));
    }
  }
  function Zd(e, t, n, r) {
    switch (Od(t)) {
      case 1:
        var l = Pp;
        break;
      case 4:
        l = Mp;
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
  function Oo(e, t, n, r, l) {
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
    Sd(function() {
      var c = a, m = ai(n), d = [];
      e: {
        var p = Xd.get(e);
        if (p !== void 0) {
          var x = di, w = e;
          switch (e) {
            case "keypress":
              if (fa(n) === 0) break e;
            case "keydown":
            case "keyup":
              x = Kp;
              break;
            case "focusin":
              w = "focus", x = Ro;
              break;
            case "focusout":
              w = "blur", x = Ro;
              break;
            case "beforeblur":
            case "afterblur":
              x = Ro;
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
              x = Op;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              x = Xp;
              break;
            case Qd:
            case Kd:
            case Yd:
              x = zp;
              break;
            case Gd:
              x = Zp;
              break;
            case "scroll":
              x = Dp;
              break;
            case "wheel":
              x = eh;
              break;
            case "copy":
            case "cut":
            case "paste":
              x = Fp;
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
          var k = (t & 4) !== 0, R = !k && e === "scroll", h = k ? p !== null ? p + "Capture" : null : p;
          k = [];
          for (var f = c, v; f !== null; ) {
            v = f;
            var E = v.stateNode;
            if (v.tag === 5 && E !== null && (v = E, h !== null && (E = dl(f, h), E != null && k.push(xl(f, E, v)))), R) break;
            f = f.return;
          }
          0 < k.length && (p = new x(p, w, null, n, m), d.push({
            event: p,
            listeners: k
          }));
        }
      }
      if (!(t & 7)) {
        e: {
          if (p = e === "mouseover" || e === "pointerover", x = e === "mouseout" || e === "pointerout", p && n !== cs && (w = n.relatedTarget || n.fromElement) && (zn(w) || w[tn])) break e;
          if ((x || p) && (p = m.window === m ? m : (p = m.ownerDocument) ? p.defaultView || p.parentWindow : window, x ? (w = n.relatedTarget || n.toElement, x = c, w = w ? zn(w) : null, w !== null && (R = Jn(w), w !== R || w.tag !== 5 && w.tag !== 6) && (w = null)) : (x = null, w = c), x !== w)) {
            if (k = Su, E = "onMouseLeave", h = "onMouseEnter", f = "mouse", (e === "pointerout" || e === "pointerover") && (k = Nu, E = "onPointerLeave", h = "onPointerEnter", f = "pointer"), R = x == null ? p : dr(x), v = w == null ? p : dr(w), p = new k(E, f + "leave", x, n, m), p.target = R, p.relatedTarget = v, E = null, zn(m) === c && (k = new k(h, f + "enter", w, n, m), k.target = v, k.relatedTarget = R, E = k), R = E, x && w) t: {
              for (k = x, h = w, f = 0, v = k; v; v = nr(v)) f++;
              for (v = 0, E = h; E; E = nr(E)) v++;
              for (; 0 < f - v; ) k = nr(k), f--;
              for (; 0 < v - f; ) h = nr(h), v--;
              for (; f--; ) {
                if (k === h || h !== null && k === h.alternate) break t;
                k = nr(k), h = nr(h);
              }
              k = null;
            }
            else k = null;
            x !== null && Lu(d, p, x, k, false), w !== null && R !== null && Lu(d, R, w, k, true);
          }
        }
        e: {
          if (p = c ? dr(c) : window, x = p.nodeName && p.nodeName.toLowerCase(), x === "select" || x === "input" && p.type === "file") var _ = sh;
          else if (Cu(p)) if ($d) _ = dh;
          else {
            _ = uh;
            var b = ih;
          }
          else (x = p.nodeName) && x.toLowerCase() === "input" && (p.type === "checkbox" || p.type === "radio") && (_ = ch);
          if (_ && (_ = _(e, c))) {
            Fd(d, _, n, m);
            break e;
          }
          b && b(e, p, c), e === "focusout" && (b = p._wrapperState) && b.controlled && p.type === "number" && as(p, "number", p.value);
        }
        switch (b = c ? dr(c) : window, e) {
          case "focusin":
            (Cu(b) || b.contentEditable === "true") && (ur = b, vs = c, rl = null);
            break;
          case "focusout":
            rl = vs = ur = null;
            break;
          case "mousedown":
            xs = true;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            xs = false, Pu(d, n, m);
            break;
          case "selectionchange":
            if (ph) break;
          case "keydown":
          case "keyup":
            Pu(d, n, m);
        }
        var S;
        if (mi) e: {
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
        else ir ? zd(e, n) && (j = "onCompositionEnd") : e === "keydown" && n.keyCode === 229 && (j = "onCompositionStart");
        j && (Id && n.locale !== "ko" && (ir || j !== "onCompositionStart" ? j === "onCompositionEnd" && ir && (S = Ad()) : (pn = m, ci = "value" in pn ? pn.value : pn.textContent, ir = true)), b = ba(c, j), 0 < b.length && (j = new ku(j, e, null, n, m), d.push({
          event: j,
          listeners: b
        }), S ? j.data = S : (S = Ud(n), S !== null && (j.data = S)))), (S = nh ? rh(e, n) : lh(e, n)) && (c = ba(c, "onBeforeInput"), 0 < c.length && (m = new ku("onBeforeInput", "beforeinput", null, n, m), d.push({
          event: m,
          listeners: c
        }), m.data = S));
      }
      Jd(d, t);
    });
  }
  function xl(e, t, n) {
    return {
      instance: e,
      listener: t,
      currentTarget: n
    };
  }
  function ba(e, t) {
    for (var n = t + "Capture", r = []; e !== null; ) {
      var l = e, a = l.stateNode;
      l.tag === 5 && a !== null && (l = a, a = dl(e, n), a != null && r.unshift(xl(e, a, l)), a = dl(e, t), a != null && r.push(xl(e, a, l))), e = e.return;
    }
    return r;
  }
  function nr(e) {
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
  var xh = /\r\n?/g, yh = /\u0000|\uFFFD/g;
  function Ou(e) {
    return (typeof e == "string" ? e : "" + e).replace(xh, `
`).replace(yh, "");
  }
  function Zl(e, t, n) {
    if (t = Ou(t), Ou(e) !== t && n) throw Error(P(425));
  }
  function Ta() {
  }
  var ys = null, ws = null;
  function Ss(e, t) {
    return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
  }
  var ks = typeof setTimeout == "function" ? setTimeout : void 0, wh = typeof clearTimeout == "function" ? clearTimeout : void 0, Au = typeof Promise == "function" ? Promise : void 0, Sh = typeof queueMicrotask == "function" ? queueMicrotask : typeof Au < "u" ? function(e) {
    return Au.resolve(null).then(e).catch(kh);
  } : ks;
  function kh(e) {
    setTimeout(function() {
      throw e;
    });
  }
  function Ao(e, t) {
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
  function Iu(e) {
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
  var Dr = Math.random().toString(36).slice(2), Wt = "__reactFiber$" + Dr, yl = "__reactProps$" + Dr, tn = "__reactContainer$" + Dr, Ns = "__reactEvents$" + Dr, Nh = "__reactListeners$" + Dr, jh = "__reactHandles$" + Dr;
  function zn(e) {
    var t = e[Wt];
    if (t) return t;
    for (var n = e.parentNode; n; ) {
      if (t = n[tn] || n[Wt]) {
        if (n = t.alternate, t.child !== null || n !== null && n.child !== null) for (e = Iu(e); e !== null; ) {
          if (n = e[Wt]) return n;
          e = Iu(e);
        }
        return t;
      }
      e = n, n = e.parentNode;
    }
    return null;
  }
  function Pl(e) {
    return e = e[Wt] || e[tn], !e || e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3 ? null : e;
  }
  function dr(e) {
    if (e.tag === 5 || e.tag === 6) return e.stateNode;
    throw Error(P(33));
  }
  function to(e) {
    return e[yl] || null;
  }
  var js = [], fr = -1;
  function bn(e) {
    return {
      current: e
    };
  }
  function we(e) {
    0 > fr || (e.current = js[fr], js[fr] = null, fr--);
  }
  function xe(e, t) {
    fr++, js[fr] = e.current, e.current = t;
  }
  var _n = {}, et = bn(_n), dt = bn(false), Wn = _n;
  function Er(e, t) {
    var n = e.type.contextTypes;
    if (!n) return _n;
    var r = e.stateNode;
    if (r && r.__reactInternalMemoizedUnmaskedChildContext === t) return r.__reactInternalMemoizedMaskedChildContext;
    var l = {}, a;
    for (a in n) l[a] = t[a];
    return r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = t, e.__reactInternalMemoizedMaskedChildContext = l), l;
  }
  function ft(e) {
    return e = e.childContextTypes, e != null;
  }
  function Pa() {
    we(dt), we(et);
  }
  function zu(e, t, n) {
    if (et.current !== _n) throw Error(P(168));
    xe(et, t), xe(dt, n);
  }
  function qd(e, t, n) {
    var r = e.stateNode;
    if (t = t.childContextTypes, typeof r.getChildContext != "function") return n;
    r = r.getChildContext();
    for (var l in r) if (!(l in t)) throw Error(P(108, ip(e) || "Unknown", l));
    return Re({}, n, r);
  }
  function Ma(e) {
    return e = (e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext || _n, Wn = et.current, xe(et, e), xe(dt, dt.current), true;
  }
  function Uu(e, t, n) {
    var r = e.stateNode;
    if (!r) throw Error(P(169));
    n ? (e = qd(e, t, Wn), r.__reactInternalMemoizedMergedChildContext = e, we(dt), we(et), xe(et, e)) : we(dt), xe(dt, n);
  }
  var Xt = null, no = false, Io = false;
  function ef(e) {
    Xt === null ? Xt = [
      e
    ] : Xt.push(e);
  }
  function Eh(e) {
    no = true, ef(e);
  }
  function Tn() {
    if (!Io && Xt !== null) {
      Io = true;
      var e = 0, t = de;
      try {
        var n = Xt;
        for (de = 1; e < n.length; e++) {
          var r = n[e];
          do
            r = r(true);
          while (r !== null);
        }
        Xt = null, no = false;
      } catch (l) {
        throw Xt !== null && (Xt = Xt.slice(e + 1)), Ed(oi, Tn), l;
      } finally {
        de = t, Io = false;
      }
    }
    return null;
  }
  var mr = [], pr = 0, Da = null, La = 0, Et = [], Ct = 0, Hn = null, Jt = 1, Zt = "";
  function Ln(e, t) {
    mr[pr++] = La, mr[pr++] = Da, Da = e, La = t;
  }
  function tf(e, t, n) {
    Et[Ct++] = Jt, Et[Ct++] = Zt, Et[Ct++] = Hn, Hn = e;
    var r = Jt;
    e = Zt;
    var l = 32 - It(r) - 1;
    r &= ~(1 << l), n += 1;
    var a = 32 - It(t) + l;
    if (30 < a) {
      var o = l - l % 5;
      a = (r & (1 << o) - 1).toString(32), r >>= o, l -= o, Jt = 1 << 32 - It(t) + l | n << l | r, Zt = a + e;
    } else Jt = 1 << a | n << l | r, Zt = e;
  }
  function hi(e) {
    e.return !== null && (Ln(e, 1), tf(e, 1, 0));
  }
  function gi(e) {
    for (; e === Da; ) Da = mr[--pr], mr[pr] = null, La = mr[--pr], mr[pr] = null;
    for (; e === Hn; ) Hn = Et[--Ct], Et[Ct] = null, Zt = Et[--Ct], Et[Ct] = null, Jt = Et[--Ct], Et[Ct] = null;
  }
  var St = null, wt = null, Ne = false, At = null;
  function nf(e, t) {
    var n = _t(5, null, null, 0);
    n.elementType = "DELETED", n.stateNode = t, n.return = e, t = e.deletions, t === null ? (e.deletions = [
      n
    ], e.flags |= 16) : t.push(n);
  }
  function Fu(e, t) {
    switch (e.tag) {
      case 5:
        var n = e.type;
        return t = t.nodeType !== 1 || n.toLowerCase() !== t.nodeName.toLowerCase() ? null : t, t !== null ? (e.stateNode = t, St = e, wt = wn(t.firstChild), true) : false;
      case 6:
        return t = e.pendingProps === "" || t.nodeType !== 3 ? null : t, t !== null ? (e.stateNode = t, St = e, wt = null, true) : false;
      case 13:
        return t = t.nodeType !== 8 ? null : t, t !== null ? (n = Hn !== null ? {
          id: Jt,
          overflow: Zt
        } : null, e.memoizedState = {
          dehydrated: t,
          treeContext: n,
          retryLane: 1073741824
        }, n = _t(18, null, null, 0), n.stateNode = t, n.return = e, e.child = n, St = e, wt = null, true) : false;
      default:
        return false;
    }
  }
  function Es(e) {
    return (e.mode & 1) !== 0 && (e.flags & 128) === 0;
  }
  function Cs(e) {
    if (Ne) {
      var t = wt;
      if (t) {
        var n = t;
        if (!Fu(e, t)) {
          if (Es(e)) throw Error(P(418));
          t = wn(n.nextSibling);
          var r = St;
          t && Fu(e, t) ? nf(r, n) : (e.flags = e.flags & -4097 | 2, Ne = false, St = e);
        }
      } else {
        if (Es(e)) throw Error(P(418));
        e.flags = e.flags & -4097 | 2, Ne = false, St = e;
      }
    }
  }
  function $u(e) {
    for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13; ) e = e.return;
    St = e;
  }
  function ql(e) {
    if (e !== St) return false;
    if (!Ne) return $u(e), Ne = true, false;
    var t;
    if ((t = e.tag !== 3) && !(t = e.tag !== 5) && (t = e.type, t = t !== "head" && t !== "body" && !Ss(e.type, e.memoizedProps)), t && (t = wt)) {
      if (Es(e)) throw rf(), Error(P(418));
      for (; t; ) nf(e, t), t = wn(t.nextSibling);
    }
    if ($u(e), e.tag === 13) {
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
  function rf() {
    for (var e = wt; e; ) e = wn(e.nextSibling);
  }
  function Cr() {
    wt = St = null, Ne = false;
  }
  function vi(e) {
    At === null ? At = [
      e
    ] : At.push(e);
  }
  var Ch = ln.ReactCurrentBatchConfig;
  function Br(e, t, n) {
    if (e = n.ref, e !== null && typeof e != "function" && typeof e != "object") {
      if (n._owner) {
        if (n = n._owner, n) {
          if (n.tag !== 1) throw Error(P(309));
          var r = n.stateNode;
        }
        if (!r) throw Error(P(147, e));
        var l = r, a = "" + e;
        return t !== null && t.ref !== null && typeof t.ref == "function" && t.ref._stringRef === a ? t.ref : (t = function(o) {
          var i = l.refs;
          o === null ? delete i[a] : i[a] = o;
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
  function Bu(e) {
    var t = e._init;
    return t(e._payload);
  }
  function lf(e) {
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
    function o(h) {
      return e && h.alternate === null && (h.flags |= 2), h;
    }
    function i(h, f, v, E) {
      return f === null || f.tag !== 6 ? (f = Wo(v, h.mode, E), f.return = h, f) : (f = l(f, v), f.return = h, f);
    }
    function s(h, f, v, E) {
      var _ = v.type;
      return _ === sr ? m(h, f, v.props.children, E, v.key) : f !== null && (f.elementType === _ || typeof _ == "object" && _ !== null && _.$$typeof === cn && Bu(_) === f.type) ? (E = l(f, v.props), E.ref = Br(h, f, v), E.return = h, E) : (E = ya(v.type, v.key, v.props, null, h.mode, E), E.ref = Br(h, f, v), E.return = h, E);
    }
    function c(h, f, v, E) {
      return f === null || f.tag !== 4 || f.stateNode.containerInfo !== v.containerInfo || f.stateNode.implementation !== v.implementation ? (f = Ho(v, h.mode, E), f.return = h, f) : (f = l(f, v.children || []), f.return = h, f);
    }
    function m(h, f, v, E, _) {
      return f === null || f.tag !== 7 ? (f = Vn(v, h.mode, E, _), f.return = h, f) : (f = l(f, v), f.return = h, f);
    }
    function d(h, f, v) {
      if (typeof f == "string" && f !== "" || typeof f == "number") return f = Wo("" + f, h.mode, v), f.return = h, f;
      if (typeof f == "object" && f !== null) {
        switch (f.$$typeof) {
          case Vl:
            return v = ya(f.type, f.key, f.props, null, h.mode, v), v.ref = Br(h, null, f), v.return = h, v;
          case or:
            return f = Ho(f, h.mode, v), f.return = h, f;
          case cn:
            var E = f._init;
            return d(h, E(f._payload), v);
        }
        if (Gr(f) || Ir(f)) return f = Vn(f, h.mode, v, null), f.return = h, f;
        ea(h, f);
      }
      return null;
    }
    function p(h, f, v, E) {
      var _ = f !== null ? f.key : null;
      if (typeof v == "string" && v !== "" || typeof v == "number") return _ !== null ? null : i(h, f, "" + v, E);
      if (typeof v == "object" && v !== null) {
        switch (v.$$typeof) {
          case Vl:
            return v.key === _ ? s(h, f, v, E) : null;
          case or:
            return v.key === _ ? c(h, f, v, E) : null;
          case cn:
            return _ = v._init, p(h, f, _(v._payload), E);
        }
        if (Gr(v) || Ir(v)) return _ !== null ? null : m(h, f, v, E, null);
        ea(h, v);
      }
      return null;
    }
    function x(h, f, v, E, _) {
      if (typeof E == "string" && E !== "" || typeof E == "number") return h = h.get(v) || null, i(f, h, "" + E, _);
      if (typeof E == "object" && E !== null) {
        switch (E.$$typeof) {
          case Vl:
            return h = h.get(E.key === null ? v : E.key) || null, s(f, h, E, _);
          case or:
            return h = h.get(E.key === null ? v : E.key) || null, c(f, h, E, _);
          case cn:
            var b = E._init;
            return x(h, f, v, b(E._payload), _);
        }
        if (Gr(E) || Ir(E)) return h = h.get(v) || null, m(f, h, E, _, null);
        ea(f, E);
      }
      return null;
    }
    function w(h, f, v, E) {
      for (var _ = null, b = null, S = f, j = f = 0, z = null; S !== null && j < v.length; j++) {
        S.index > j ? (z = S, S = null) : z = S.sibling;
        var D = p(h, S, v[j], E);
        if (D === null) {
          S === null && (S = z);
          break;
        }
        e && S && D.alternate === null && t(h, S), f = a(D, f, j), b === null ? _ = D : b.sibling = D, b = D, S = z;
      }
      if (j === v.length) return n(h, S), Ne && Ln(h, j), _;
      if (S === null) {
        for (; j < v.length; j++) S = d(h, v[j], E), S !== null && (f = a(S, f, j), b === null ? _ = S : b.sibling = S, b = S);
        return Ne && Ln(h, j), _;
      }
      for (S = r(h, S); j < v.length; j++) z = x(S, h, j, v[j], E), z !== null && (e && z.alternate !== null && S.delete(z.key === null ? j : z.key), f = a(z, f, j), b === null ? _ = z : b.sibling = z, b = z);
      return e && S.forEach(function(H) {
        return t(h, H);
      }), Ne && Ln(h, j), _;
    }
    function k(h, f, v, E) {
      var _ = Ir(v);
      if (typeof _ != "function") throw Error(P(150));
      if (v = _.call(v), v == null) throw Error(P(151));
      for (var b = _ = null, S = f, j = f = 0, z = null, D = v.next(); S !== null && !D.done; j++, D = v.next()) {
        S.index > j ? (z = S, S = null) : z = S.sibling;
        var H = p(h, S, D.value, E);
        if (H === null) {
          S === null && (S = z);
          break;
        }
        e && S && H.alternate === null && t(h, S), f = a(H, f, j), b === null ? _ = H : b.sibling = H, b = H, S = z;
      }
      if (D.done) return n(h, S), Ne && Ln(h, j), _;
      if (S === null) {
        for (; !D.done; j++, D = v.next()) D = d(h, D.value, E), D !== null && (f = a(D, f, j), b === null ? _ = D : b.sibling = D, b = D);
        return Ne && Ln(h, j), _;
      }
      for (S = r(h, S); !D.done; j++, D = v.next()) D = x(S, h, j, D.value, E), D !== null && (e && D.alternate !== null && S.delete(D.key === null ? j : D.key), f = a(D, f, j), b === null ? _ = D : b.sibling = D, b = D);
      return e && S.forEach(function(K) {
        return t(h, K);
      }), Ne && Ln(h, j), _;
    }
    function R(h, f, v, E) {
      if (typeof v == "object" && v !== null && v.type === sr && v.key === null && (v = v.props.children), typeof v == "object" && v !== null) {
        switch (v.$$typeof) {
          case Vl:
            e: {
              for (var _ = v.key, b = f; b !== null; ) {
                if (b.key === _) {
                  if (_ = v.type, _ === sr) {
                    if (b.tag === 7) {
                      n(h, b.sibling), f = l(b, v.props.children), f.return = h, h = f;
                      break e;
                    }
                  } else if (b.elementType === _ || typeof _ == "object" && _ !== null && _.$$typeof === cn && Bu(_) === b.type) {
                    n(h, b.sibling), f = l(b, v.props), f.ref = Br(h, b, v), f.return = h, h = f;
                    break e;
                  }
                  n(h, b);
                  break;
                } else t(h, b);
                b = b.sibling;
              }
              v.type === sr ? (f = Vn(v.props.children, h.mode, E, v.key), f.return = h, h = f) : (E = ya(v.type, v.key, v.props, null, h.mode, E), E.ref = Br(h, f, v), E.return = h, h = E);
            }
            return o(h);
          case or:
            e: {
              for (b = v.key; f !== null; ) {
                if (f.key === b) if (f.tag === 4 && f.stateNode.containerInfo === v.containerInfo && f.stateNode.implementation === v.implementation) {
                  n(h, f.sibling), f = l(f, v.children || []), f.return = h, h = f;
                  break e;
                } else {
                  n(h, f);
                  break;
                }
                else t(h, f);
                f = f.sibling;
              }
              f = Ho(v, h.mode, E), f.return = h, h = f;
            }
            return o(h);
          case cn:
            return b = v._init, R(h, f, b(v._payload), E);
        }
        if (Gr(v)) return w(h, f, v, E);
        if (Ir(v)) return k(h, f, v, E);
        ea(h, v);
      }
      return typeof v == "string" && v !== "" || typeof v == "number" ? (v = "" + v, f !== null && f.tag === 6 ? (n(h, f.sibling), f = l(f, v), f.return = h, h = f) : (n(h, f), f = Wo(v, h.mode, E), f.return = h, h = f), o(h)) : n(h, f);
    }
    return R;
  }
  var _r = lf(true), af = lf(false), Oa = bn(null), Aa = null, hr = null, xi = null;
  function yi() {
    xi = hr = Aa = null;
  }
  function wi(e) {
    var t = Oa.current;
    we(Oa), e._currentValue = t;
  }
  function _s(e, t, n) {
    for (; e !== null; ) {
      var r = e.alternate;
      if ((e.childLanes & t) !== t ? (e.childLanes |= t, r !== null && (r.childLanes |= t)) : r !== null && (r.childLanes & t) !== t && (r.childLanes |= t), e === n) break;
      e = e.return;
    }
  }
  function kr(e, t) {
    Aa = e, xi = hr = null, e = e.dependencies, e !== null && e.firstContext !== null && (e.lanes & t && (ct = true), e.firstContext = null);
  }
  function bt(e) {
    var t = e._currentValue;
    if (xi !== e) if (e = {
      context: e,
      memoizedValue: t,
      next: null
    }, hr === null) {
      if (Aa === null) throw Error(P(308));
      hr = e, Aa.dependencies = {
        lanes: 0,
        firstContext: e
      };
    } else hr = hr.next = e;
    return t;
  }
  var Un = null;
  function Si(e) {
    Un === null ? Un = [
      e
    ] : Un.push(e);
  }
  function of(e, t, n, r) {
    var l = t.interleaved;
    return l === null ? (n.next = n, Si(t)) : (n.next = l.next, l.next = n), t.interleaved = n, nn(e, r);
  }
  function nn(e, t) {
    e.lanes |= t;
    var n = e.alternate;
    for (n !== null && (n.lanes |= t), n = e, e = e.return; e !== null; ) e.childLanes |= t, n = e.alternate, n !== null && (n.childLanes |= t), n = e, e = e.return;
    return n.tag === 3 ? n.stateNode : null;
  }
  var dn = false;
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
  function sf(e, t) {
    e = e.updateQueue, t.updateQueue === e && (t.updateQueue = {
      baseState: e.baseState,
      firstBaseUpdate: e.firstBaseUpdate,
      lastBaseUpdate: e.lastBaseUpdate,
      shared: e.shared,
      effects: e.effects
    });
  }
  function qt(e, t) {
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
    if (r = r.shared, ae & 2) {
      var l = r.pending;
      return l === null ? t.next = t : (t.next = l.next, l.next = t), r.pending = t, nn(e, n);
    }
    return l = r.interleaved, l === null ? (t.next = t, Si(r)) : (t.next = l.next, l.next = t), r.interleaved = t, nn(e, n);
  }
  function ma(e, t, n) {
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
    dn = false;
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
        var p = i.lane, x = i.eventTime;
        if ((r & p) === p) {
          m !== null && (m = m.next = {
            eventTime: x,
            lane: 0,
            tag: i.tag,
            payload: i.payload,
            callback: i.callback,
            next: null
          });
          e: {
            var w = e, k = i;
            switch (p = t, x = n, k.tag) {
              case 1:
                if (w = k.payload, typeof w == "function") {
                  d = w.call(x, d, p);
                  break e;
                }
                d = w;
                break e;
              case 3:
                w.flags = w.flags & -65537 | 128;
              case 0:
                if (w = k.payload, p = typeof w == "function" ? w.call(x, d, p) : w, p == null) break e;
                d = Re({}, d, p);
                break e;
              case 2:
                dn = true;
            }
          }
          i.callback !== null && i.lane !== 0 && (e.flags |= 64, p = l.effects, p === null ? l.effects = [
            i
          ] : p.push(i));
        } else x = {
          eventTime: x,
          lane: p,
          tag: i.tag,
          payload: i.payload,
          callback: i.callback,
          next: null
        }, m === null ? (c = m = x, s = d) : m = m.next = x, o |= p;
        if (i = i.next, i === null) {
          if (i = l.shared.pending, i === null) break;
          p = i, i = p.next, p.next = null, l.lastBaseUpdate = p, l.shared.pending = null;
        }
      } while (true);
      if (m === null && (s = d), l.baseState = s, l.firstBaseUpdate = c, l.lastBaseUpdate = m, t = l.shared.interleaved, t !== null) {
        l = t;
        do
          o |= l.lane, l = l.next;
        while (l !== t);
      } else a === null && (l.shared.lanes = 0);
      Kn |= o, e.lanes = o, e.memoizedState = d;
    }
  }
  function Wu(e, t, n) {
    if (e = t.effects, t.effects = null, e !== null) for (t = 0; t < e.length; t++) {
      var r = e[t], l = r.callback;
      if (l !== null) {
        if (r.callback = null, r = n, typeof l != "function") throw Error(P(191, l));
        l.call(r);
      }
    }
  }
  var Ml = {}, Qt = bn(Ml), wl = bn(Ml), Sl = bn(Ml);
  function Fn(e) {
    if (e === Ml) throw Error(P(174));
    return e;
  }
  function Ni(e, t) {
    switch (xe(Sl, t), xe(wl, e), xe(Qt, Ml), e = t.nodeType, e) {
      case 9:
      case 11:
        t = (t = t.documentElement) ? t.namespaceURI : ss(null, "");
        break;
      default:
        e = e === 8 ? t.parentNode : t, t = e.namespaceURI || null, e = e.tagName, t = ss(t, e);
    }
    we(Qt), xe(Qt, t);
  }
  function Rr() {
    we(Qt), we(wl), we(Sl);
  }
  function uf(e) {
    Fn(Sl.current);
    var t = Fn(Qt.current), n = ss(t, e.type);
    t !== n && (xe(wl, e), xe(Qt, n));
  }
  function ji(e) {
    wl.current === e && (we(Qt), we(wl));
  }
  var Ce = bn(0);
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
  var pa = ln.ReactCurrentDispatcher, Uo = ln.ReactCurrentBatchConfig, Qn = 0, _e = null, $e = null, We = null, Ua = false, ll = false, kl = 0, _h = 0;
  function Je() {
    throw Error(P(321));
  }
  function Ci(e, t) {
    if (t === null) return false;
    for (var n = 0; n < t.length && n < e.length; n++) if (!Ut(e[n], t[n])) return false;
    return true;
  }
  function _i(e, t, n, r, l, a) {
    if (Qn = a, _e = t, t.memoizedState = null, t.updateQueue = null, t.lanes = 0, pa.current = e === null || e.memoizedState === null ? Ph : Mh, e = n(r, l), ll) {
      a = 0;
      do {
        if (ll = false, kl = 0, 25 <= a) throw Error(P(301));
        a += 1, We = $e = null, t.updateQueue = null, pa.current = Dh, e = n(r, l);
      } while (ll);
    }
    if (pa.current = Fa, t = $e !== null && $e.next !== null, Qn = 0, We = $e = _e = null, Ua = false, t) throw Error(P(300));
    return e;
  }
  function Ri() {
    var e = kl !== 0;
    return kl = 0, e;
  }
  function Vt() {
    var e = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return We === null ? _e.memoizedState = We = e : We = We.next = e, We;
  }
  function Tt() {
    if ($e === null) {
      var e = _e.alternate;
      e = e !== null ? e.memoizedState : null;
    } else e = $e.next;
    var t = We === null ? _e.memoizedState : We.next;
    if (t !== null) We = t, $e = e;
    else {
      if (e === null) throw Error(P(310));
      $e = e, e = {
        memoizedState: $e.memoizedState,
        baseState: $e.baseState,
        baseQueue: $e.baseQueue,
        queue: $e.queue,
        next: null
      }, We === null ? _e.memoizedState = We = e : We = We.next = e;
    }
    return We;
  }
  function Nl(e, t) {
    return typeof t == "function" ? t(e) : t;
  }
  function Fo(e) {
    var t = Tt(), n = t.queue;
    if (n === null) throw Error(P(311));
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
        if ((Qn & m) === m) s !== null && (s = s.next = {
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
          s === null ? (i = s = d, o = r) : s = s.next = d, _e.lanes |= m, Kn |= m;
        }
        c = c.next;
      } while (c !== null && c !== a);
      s === null ? o = r : s.next = i, Ut(r, t.memoizedState) || (ct = true), t.memoizedState = r, t.baseState = o, t.baseQueue = s, n.lastRenderedState = r;
    }
    if (e = n.interleaved, e !== null) {
      l = e;
      do
        a = l.lane, _e.lanes |= a, Kn |= a, l = l.next;
      while (l !== e);
    } else l === null && (n.lanes = 0);
    return [
      t.memoizedState,
      n.dispatch
    ];
  }
  function $o(e) {
    var t = Tt(), n = t.queue;
    if (n === null) throw Error(P(311));
    n.lastRenderedReducer = e;
    var r = n.dispatch, l = n.pending, a = t.memoizedState;
    if (l !== null) {
      n.pending = null;
      var o = l = l.next;
      do
        a = e(a, o.action), o = o.next;
      while (o !== l);
      Ut(a, t.memoizedState) || (ct = true), t.memoizedState = a, t.baseQueue === null && (t.baseState = a), n.lastRenderedState = a;
    }
    return [
      a,
      r
    ];
  }
  function cf() {
  }
  function df(e, t) {
    var n = _e, r = Tt(), l = t(), a = !Ut(r.memoizedState, l);
    if (a && (r.memoizedState = l, ct = true), r = r.queue, bi(pf.bind(null, n, r, e), [
      e
    ]), r.getSnapshot !== t || a || We !== null && We.memoizedState.tag & 1) {
      if (n.flags |= 2048, jl(9, mf.bind(null, n, r, l, t), void 0, null), He === null) throw Error(P(349));
      Qn & 30 || ff(n, t, l);
    }
    return l;
  }
  function ff(e, t, n) {
    e.flags |= 16384, e = {
      getSnapshot: t,
      value: n
    }, t = _e.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, _e.updateQueue = t, t.stores = [
      e
    ]) : (n = t.stores, n === null ? t.stores = [
      e
    ] : n.push(e));
  }
  function mf(e, t, n, r) {
    t.value = n, t.getSnapshot = r, hf(t) && gf(e);
  }
  function pf(e, t, n) {
    return n(function() {
      hf(t) && gf(e);
    });
  }
  function hf(e) {
    var t = e.getSnapshot;
    e = e.value;
    try {
      var n = t();
      return !Ut(e, n);
    } catch {
      return true;
    }
  }
  function gf(e) {
    var t = nn(e, 1);
    t !== null && zt(t, e, 1, -1);
  }
  function Hu(e) {
    var t = Vt();
    return typeof e == "function" && (e = e()), t.memoizedState = t.baseState = e, e = {
      pending: null,
      interleaved: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Nl,
      lastRenderedState: e
    }, t.queue = e, e = e.dispatch = Th.bind(null, _e, e), [
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
    }, t = _e.updateQueue, t === null ? (t = {
      lastEffect: null,
      stores: null
    }, _e.updateQueue = t, t.lastEffect = e.next = e) : (n = t.lastEffect, n === null ? t.lastEffect = e.next = e : (r = n.next, n.next = e, e.next = r, t.lastEffect = e)), e;
  }
  function vf() {
    return Tt().memoizedState;
  }
  function ha(e, t, n, r) {
    var l = Vt();
    _e.flags |= e, l.memoizedState = jl(1 | t, n, void 0, r === void 0 ? null : r);
  }
  function ro(e, t, n, r) {
    var l = Tt();
    r = r === void 0 ? null : r;
    var a = void 0;
    if ($e !== null) {
      var o = $e.memoizedState;
      if (a = o.destroy, r !== null && Ci(r, o.deps)) {
        l.memoizedState = jl(t, n, a, r);
        return;
      }
    }
    _e.flags |= e, l.memoizedState = jl(1 | t, n, a, r);
  }
  function Qu(e, t) {
    return ha(8390656, 8, e, t);
  }
  function bi(e, t) {
    return ro(2048, 8, e, t);
  }
  function xf(e, t) {
    return ro(4, 2, e, t);
  }
  function yf(e, t) {
    return ro(4, 4, e, t);
  }
  function wf(e, t) {
    if (typeof t == "function") return e = e(), t(e), function() {
      t(null);
    };
    if (t != null) return e = e(), t.current = e, function() {
      t.current = null;
    };
  }
  function Sf(e, t, n) {
    return n = n != null ? n.concat([
      e
    ]) : null, ro(4, 4, wf.bind(null, t, e), n);
  }
  function Ti() {
  }
  function kf(e, t) {
    var n = Tt();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Ci(t, r[1]) ? r[0] : (n.memoizedState = [
      e,
      t
    ], e);
  }
  function Nf(e, t) {
    var n = Tt();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Ci(t, r[1]) ? r[0] : (e = e(), n.memoizedState = [
      e,
      t
    ], e);
  }
  function jf(e, t, n) {
    return Qn & 21 ? (Ut(n, t) || (n = Rd(), _e.lanes |= n, Kn |= n, e.baseState = true), t) : (e.baseState && (e.baseState = false, ct = true), e.memoizedState = n);
  }
  function Rh(e, t) {
    var n = de;
    de = n !== 0 && 4 > n ? n : 4, e(true);
    var r = Uo.transition;
    Uo.transition = {};
    try {
      e(false), t();
    } finally {
      de = n, Uo.transition = r;
    }
  }
  function Ef() {
    return Tt().memoizedState;
  }
  function bh(e, t, n) {
    var r = Nn(e);
    if (n = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    }, Cf(e)) _f(t, n);
    else if (n = of(e, t, n, r), n !== null) {
      var l = rt();
      zt(n, e, r, l), Rf(n, t, r);
    }
  }
  function Th(e, t, n) {
    var r = Nn(e), l = {
      lane: r,
      action: n,
      hasEagerState: false,
      eagerState: null,
      next: null
    };
    if (Cf(e)) _f(t, l);
    else {
      var a = e.alternate;
      if (e.lanes === 0 && (a === null || a.lanes === 0) && (a = t.lastRenderedReducer, a !== null)) try {
        var o = t.lastRenderedState, i = a(o, n);
        if (l.hasEagerState = true, l.eagerState = i, Ut(i, o)) {
          var s = t.interleaved;
          s === null ? (l.next = l, Si(t)) : (l.next = s.next, s.next = l), t.interleaved = l;
          return;
        }
      } catch {
      } finally {
      }
      n = of(e, t, l, r), n !== null && (l = rt(), zt(n, e, r, l), Rf(n, t, r));
    }
  }
  function Cf(e) {
    var t = e.alternate;
    return e === _e || t !== null && t === _e;
  }
  function _f(e, t) {
    ll = Ua = true;
    var n = e.pending;
    n === null ? t.next = t : (t.next = n.next, n.next = t), e.pending = t;
  }
  function Rf(e, t, n) {
    if (n & 4194240) {
      var r = t.lanes;
      r &= e.pendingLanes, n |= r, t.lanes = n, si(e, n);
    }
  }
  var Fa = {
    readContext: bt,
    useCallback: Je,
    useContext: Je,
    useEffect: Je,
    useImperativeHandle: Je,
    useInsertionEffect: Je,
    useLayoutEffect: Je,
    useMemo: Je,
    useReducer: Je,
    useRef: Je,
    useState: Je,
    useDebugValue: Je,
    useDeferredValue: Je,
    useTransition: Je,
    useMutableSource: Je,
    useSyncExternalStore: Je,
    useId: Je,
    unstable_isNewReconciler: false
  }, Ph = {
    readContext: bt,
    useCallback: function(e, t) {
      return Vt().memoizedState = [
        e,
        t === void 0 ? null : t
      ], e;
    },
    useContext: bt,
    useEffect: Qu,
    useImperativeHandle: function(e, t, n) {
      return n = n != null ? n.concat([
        e
      ]) : null, ha(4194308, 4, wf.bind(null, t, e), n);
    },
    useLayoutEffect: function(e, t) {
      return ha(4194308, 4, e, t);
    },
    useInsertionEffect: function(e, t) {
      return ha(4, 2, e, t);
    },
    useMemo: function(e, t) {
      var n = Vt();
      return t = t === void 0 ? null : t, e = e(), n.memoizedState = [
        e,
        t
      ], e;
    },
    useReducer: function(e, t, n) {
      var r = Vt();
      return t = n !== void 0 ? n(t) : t, r.memoizedState = r.baseState = t, e = {
        pending: null,
        interleaved: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: e,
        lastRenderedState: t
      }, r.queue = e, e = e.dispatch = bh.bind(null, _e, e), [
        r.memoizedState,
        e
      ];
    },
    useRef: function(e) {
      var t = Vt();
      return e = {
        current: e
      }, t.memoizedState = e;
    },
    useState: Hu,
    useDebugValue: Ti,
    useDeferredValue: function(e) {
      return Vt().memoizedState = e;
    },
    useTransition: function() {
      var e = Hu(false), t = e[0];
      return e = Rh.bind(null, e[1]), Vt().memoizedState = e, [
        t,
        e
      ];
    },
    useMutableSource: function() {
    },
    useSyncExternalStore: function(e, t, n) {
      var r = _e, l = Vt();
      if (Ne) {
        if (n === void 0) throw Error(P(407));
        n = n();
      } else {
        if (n = t(), He === null) throw Error(P(349));
        Qn & 30 || ff(r, t, n);
      }
      l.memoizedState = n;
      var a = {
        value: n,
        getSnapshot: t
      };
      return l.queue = a, Qu(pf.bind(null, r, a, e), [
        e
      ]), r.flags |= 2048, jl(9, mf.bind(null, r, a, n, t), void 0, null), n;
    },
    useId: function() {
      var e = Vt(), t = He.identifierPrefix;
      if (Ne) {
        var n = Zt, r = Jt;
        n = (r & ~(1 << 32 - It(r) - 1)).toString(32) + n, t = ":" + t + "R" + n, n = kl++, 0 < n && (t += "H" + n.toString(32)), t += ":";
      } else n = _h++, t = ":" + t + "r" + n.toString(32) + ":";
      return e.memoizedState = t;
    },
    unstable_isNewReconciler: false
  }, Mh = {
    readContext: bt,
    useCallback: kf,
    useContext: bt,
    useEffect: bi,
    useImperativeHandle: Sf,
    useInsertionEffect: xf,
    useLayoutEffect: yf,
    useMemo: Nf,
    useReducer: Fo,
    useRef: vf,
    useState: function() {
      return Fo(Nl);
    },
    useDebugValue: Ti,
    useDeferredValue: function(e) {
      var t = Tt();
      return jf(t, $e.memoizedState, e);
    },
    useTransition: function() {
      var e = Fo(Nl)[0], t = Tt().memoizedState;
      return [
        e,
        t
      ];
    },
    useMutableSource: cf,
    useSyncExternalStore: df,
    useId: Ef,
    unstable_isNewReconciler: false
  }, Dh = {
    readContext: bt,
    useCallback: kf,
    useContext: bt,
    useEffect: bi,
    useImperativeHandle: Sf,
    useInsertionEffect: xf,
    useLayoutEffect: yf,
    useMemo: Nf,
    useReducer: $o,
    useRef: vf,
    useState: function() {
      return $o(Nl);
    },
    useDebugValue: Ti,
    useDeferredValue: function(e) {
      var t = Tt();
      return $e === null ? t.memoizedState = e : jf(t, $e.memoizedState, e);
    },
    useTransition: function() {
      var e = $o(Nl)[0], t = Tt().memoizedState;
      return [
        e,
        t
      ];
    },
    useMutableSource: cf,
    useSyncExternalStore: df,
    useId: Ef,
    unstable_isNewReconciler: false
  };
  function Dt(e, t) {
    if (e && e.defaultProps) {
      t = Re({}, t), e = e.defaultProps;
      for (var n in e) t[n] === void 0 && (t[n] = e[n]);
      return t;
    }
    return t;
  }
  function Rs(e, t, n, r) {
    t = e.memoizedState, n = n(r, t), n = n == null ? t : Re({}, t, n), e.memoizedState = n, e.lanes === 0 && (e.updateQueue.baseState = n);
  }
  var lo = {
    isMounted: function(e) {
      return (e = e._reactInternals) ? Jn(e) === e : false;
    },
    enqueueSetState: function(e, t, n) {
      e = e._reactInternals;
      var r = rt(), l = Nn(e), a = qt(r, l);
      a.payload = t, n != null && (a.callback = n), t = Sn(e, a, l), t !== null && (zt(t, e, l, r), ma(t, e, l));
    },
    enqueueReplaceState: function(e, t, n) {
      e = e._reactInternals;
      var r = rt(), l = Nn(e), a = qt(r, l);
      a.tag = 1, a.payload = t, n != null && (a.callback = n), t = Sn(e, a, l), t !== null && (zt(t, e, l, r), ma(t, e, l));
    },
    enqueueForceUpdate: function(e, t) {
      e = e._reactInternals;
      var n = rt(), r = Nn(e), l = qt(n, r);
      l.tag = 2, t != null && (l.callback = t), t = Sn(e, l, r), t !== null && (zt(t, e, r, n), ma(t, e, r));
    }
  };
  function Ku(e, t, n, r, l, a, o) {
    return e = e.stateNode, typeof e.shouldComponentUpdate == "function" ? e.shouldComponentUpdate(r, a, o) : t.prototype && t.prototype.isPureReactComponent ? !gl(n, r) || !gl(l, a) : true;
  }
  function bf(e, t, n) {
    var r = false, l = _n, a = t.contextType;
    return typeof a == "object" && a !== null ? a = bt(a) : (l = ft(t) ? Wn : et.current, r = t.contextTypes, a = (r = r != null) ? Er(e, l) : _n), t = new t(n, a), e.memoizedState = t.state !== null && t.state !== void 0 ? t.state : null, t.updater = lo, e.stateNode = t, t._reactInternals = e, r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = l, e.__reactInternalMemoizedMaskedChildContext = a), t;
  }
  function Yu(e, t, n, r) {
    e = t.state, typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(n, r), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(n, r), t.state !== e && lo.enqueueReplaceState(t, t.state, null);
  }
  function bs(e, t, n, r) {
    var l = e.stateNode;
    l.props = n, l.state = e.memoizedState, l.refs = {}, ki(e);
    var a = t.contextType;
    typeof a == "object" && a !== null ? l.context = bt(a) : (a = ft(t) ? Wn : et.current, l.context = Er(e, a)), l.state = e.memoizedState, a = t.getDerivedStateFromProps, typeof a == "function" && (Rs(e, t, a, n), l.state = e.memoizedState), typeof t.getDerivedStateFromProps == "function" || typeof l.getSnapshotBeforeUpdate == "function" || typeof l.UNSAFE_componentWillMount != "function" && typeof l.componentWillMount != "function" || (t = l.state, typeof l.componentWillMount == "function" && l.componentWillMount(), typeof l.UNSAFE_componentWillMount == "function" && l.UNSAFE_componentWillMount(), t !== l.state && lo.enqueueReplaceState(l, l.state, null), Ia(e, n, l, r), l.state = e.memoizedState), typeof l.componentDidMount == "function" && (e.flags |= 4194308);
  }
  function br(e, t) {
    try {
      var n = "", r = t;
      do
        n += sp(r), r = r.return;
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
  function Bo(e, t, n) {
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
  var Lh = typeof WeakMap == "function" ? WeakMap : Map;
  function Tf(e, t, n) {
    n = qt(-1, n), n.tag = 3, n.payload = {
      element: null
    };
    var r = t.value;
    return n.callback = function() {
      Ba || (Ba = true, Fs = r), Ts(e, t);
    }, n;
  }
  function Pf(e, t, n) {
    n = qt(-1, n), n.tag = 3;
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
      Ts(e, t), typeof r != "function" && (kn === null ? kn = /* @__PURE__ */ new Set([
        this
      ]) : kn.add(this));
      var o = t.stack;
      this.componentDidCatch(t.value, {
        componentStack: o !== null ? o : ""
      });
    }), n;
  }
  function Gu(e, t, n) {
    var r = e.pingCache;
    if (r === null) {
      r = e.pingCache = new Lh();
      var l = /* @__PURE__ */ new Set();
      r.set(t, l);
    } else l = r.get(t), l === void 0 && (l = /* @__PURE__ */ new Set(), r.set(t, l));
    l.has(n) || (l.add(n), e = Yh.bind(null, e, t, n), t.then(e, e));
  }
  function Xu(e) {
    do {
      var t;
      if ((t = e.tag === 13) && (t = e.memoizedState, t = t !== null ? t.dehydrated !== null : true), t) return e;
      e = e.return;
    } while (e !== null);
    return null;
  }
  function Ju(e, t, n, r, l) {
    return e.mode & 1 ? (e.flags |= 65536, e.lanes = l, e) : (e === t ? e.flags |= 65536 : (e.flags |= 128, n.flags |= 131072, n.flags &= -52805, n.tag === 1 && (n.alternate === null ? n.tag = 17 : (t = qt(-1, 1), t.tag = 2, Sn(n, t, 1))), n.lanes |= 1), e);
  }
  var Oh = ln.ReactCurrentOwner, ct = false;
  function nt(e, t, n, r) {
    t.child = e === null ? af(t, null, n, r) : _r(t, e.child, n, r);
  }
  function Zu(e, t, n, r, l) {
    n = n.render;
    var a = t.ref;
    return kr(t, l), r = _i(e, t, n, r, a, l), n = Ri(), e !== null && !ct ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, rn(e, t, l)) : (Ne && n && hi(t), t.flags |= 1, nt(e, t, r, l), t.child);
  }
  function qu(e, t, n, r, l) {
    if (e === null) {
      var a = n.type;
      return typeof a == "function" && !zi(a) && a.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0 ? (t.tag = 15, t.type = a, Mf(e, t, a, r, l)) : (e = ya(n.type, null, r, t, t.mode, l), e.ref = t.ref, e.return = t, t.child = e);
    }
    if (a = e.child, !(e.lanes & l)) {
      var o = a.memoizedProps;
      if (n = n.compare, n = n !== null ? n : gl, n(o, r) && e.ref === t.ref) return rn(e, t, l);
    }
    return t.flags |= 1, e = jn(a, r), e.ref = t.ref, e.return = t, t.child = e;
  }
  function Mf(e, t, n, r, l) {
    if (e !== null) {
      var a = e.memoizedProps;
      if (gl(a, r) && e.ref === t.ref) if (ct = false, t.pendingProps = r = a, (e.lanes & l) !== 0) e.flags & 131072 && (ct = true);
      else return t.lanes = e.lanes, rn(e, t, l);
    }
    return Ps(e, t, n, r, l);
  }
  function Df(e, t, n) {
    var r = t.pendingProps, l = r.children, a = e !== null ? e.memoizedState : null;
    if (r.mode === "hidden") if (!(t.mode & 1)) t.memoizedState = {
      baseLanes: 0,
      cachePool: null,
      transitions: null
    }, xe(vr, xt), xt |= n;
    else {
      if (!(n & 1073741824)) return e = a !== null ? a.baseLanes | n : n, t.lanes = t.childLanes = 1073741824, t.memoizedState = {
        baseLanes: e,
        cachePool: null,
        transitions: null
      }, t.updateQueue = null, xe(vr, xt), xt |= e, null;
      t.memoizedState = {
        baseLanes: 0,
        cachePool: null,
        transitions: null
      }, r = a !== null ? a.baseLanes : n, xe(vr, xt), xt |= r;
    }
    else a !== null ? (r = a.baseLanes | n, t.memoizedState = null) : r = n, xe(vr, xt), xt |= r;
    return nt(e, t, l, n), t.child;
  }
  function Lf(e, t) {
    var n = t.ref;
    (e === null && n !== null || e !== null && e.ref !== n) && (t.flags |= 512, t.flags |= 2097152);
  }
  function Ps(e, t, n, r, l) {
    var a = ft(n) ? Wn : et.current;
    return a = Er(t, a), kr(t, l), n = _i(e, t, n, r, a, l), r = Ri(), e !== null && !ct ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~l, rn(e, t, l)) : (Ne && r && hi(t), t.flags |= 1, nt(e, t, n, l), t.child);
  }
  function ec(e, t, n, r, l) {
    if (ft(n)) {
      var a = true;
      Ma(t);
    } else a = false;
    if (kr(t, l), t.stateNode === null) ga(e, t), bf(t, n, r), bs(t, n, r, l), r = true;
    else if (e === null) {
      var o = t.stateNode, i = t.memoizedProps;
      o.props = i;
      var s = o.context, c = n.contextType;
      typeof c == "object" && c !== null ? c = bt(c) : (c = ft(n) ? Wn : et.current, c = Er(t, c));
      var m = n.getDerivedStateFromProps, d = typeof m == "function" || typeof o.getSnapshotBeforeUpdate == "function";
      d || typeof o.UNSAFE_componentWillReceiveProps != "function" && typeof o.componentWillReceiveProps != "function" || (i !== r || s !== c) && Yu(t, o, r, c), dn = false;
      var p = t.memoizedState;
      o.state = p, Ia(t, r, o, l), s = t.memoizedState, i !== r || p !== s || dt.current || dn ? (typeof m == "function" && (Rs(t, n, m, r), s = t.memoizedState), (i = dn || Ku(t, n, i, r, p, s, c)) ? (d || typeof o.UNSAFE_componentWillMount != "function" && typeof o.componentWillMount != "function" || (typeof o.componentWillMount == "function" && o.componentWillMount(), typeof o.UNSAFE_componentWillMount == "function" && o.UNSAFE_componentWillMount()), typeof o.componentDidMount == "function" && (t.flags |= 4194308)) : (typeof o.componentDidMount == "function" && (t.flags |= 4194308), t.memoizedProps = r, t.memoizedState = s), o.props = r, o.state = s, o.context = c, r = i) : (typeof o.componentDidMount == "function" && (t.flags |= 4194308), r = false);
    } else {
      o = t.stateNode, sf(e, t), i = t.memoizedProps, c = t.type === t.elementType ? i : Dt(t.type, i), o.props = c, d = t.pendingProps, p = o.context, s = n.contextType, typeof s == "object" && s !== null ? s = bt(s) : (s = ft(n) ? Wn : et.current, s = Er(t, s));
      var x = n.getDerivedStateFromProps;
      (m = typeof x == "function" || typeof o.getSnapshotBeforeUpdate == "function") || typeof o.UNSAFE_componentWillReceiveProps != "function" && typeof o.componentWillReceiveProps != "function" || (i !== d || p !== s) && Yu(t, o, r, s), dn = false, p = t.memoizedState, o.state = p, Ia(t, r, o, l);
      var w = t.memoizedState;
      i !== d || p !== w || dt.current || dn ? (typeof x == "function" && (Rs(t, n, x, r), w = t.memoizedState), (c = dn || Ku(t, n, c, r, p, w, s) || false) ? (m || typeof o.UNSAFE_componentWillUpdate != "function" && typeof o.componentWillUpdate != "function" || (typeof o.componentWillUpdate == "function" && o.componentWillUpdate(r, w, s), typeof o.UNSAFE_componentWillUpdate == "function" && o.UNSAFE_componentWillUpdate(r, w, s)), typeof o.componentDidUpdate == "function" && (t.flags |= 4), typeof o.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof o.componentDidUpdate != "function" || i === e.memoizedProps && p === e.memoizedState || (t.flags |= 4), typeof o.getSnapshotBeforeUpdate != "function" || i === e.memoizedProps && p === e.memoizedState || (t.flags |= 1024), t.memoizedProps = r, t.memoizedState = w), o.props = r, o.state = w, o.context = s, r = c) : (typeof o.componentDidUpdate != "function" || i === e.memoizedProps && p === e.memoizedState || (t.flags |= 4), typeof o.getSnapshotBeforeUpdate != "function" || i === e.memoizedProps && p === e.memoizedState || (t.flags |= 1024), r = false);
    }
    return Ms(e, t, n, r, a, l);
  }
  function Ms(e, t, n, r, l, a) {
    Lf(e, t);
    var o = (t.flags & 128) !== 0;
    if (!r && !o) return l && Uu(t, n, false), rn(e, t, a);
    r = t.stateNode, Oh.current = t;
    var i = o && typeof n.getDerivedStateFromError != "function" ? null : r.render();
    return t.flags |= 1, e !== null && o ? (t.child = _r(t, e.child, null, a), t.child = _r(t, null, i, a)) : nt(e, t, i, a), t.memoizedState = r.state, l && Uu(t, n, true), t.child;
  }
  function Of(e) {
    var t = e.stateNode;
    t.pendingContext ? zu(e, t.pendingContext, t.pendingContext !== t.context) : t.context && zu(e, t.context, false), Ni(e, t.containerInfo);
  }
  function tc(e, t, n, r, l) {
    return Cr(), vi(l), t.flags |= 256, nt(e, t, n, r), t.child;
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
    var r = t.pendingProps, l = Ce.current, a = false, o = (t.flags & 128) !== 0, i;
    if ((i = o) || (i = e !== null && e.memoizedState === null ? false : (l & 2) !== 0), i ? (a = true, t.flags &= -129) : (e === null || e.memoizedState !== null) && (l |= 1), xe(Ce, l & 1), e === null) return Cs(t), e = t.memoizedState, e !== null && (e = e.dehydrated, e !== null) ? (t.mode & 1 ? e.data === "$!" ? t.lanes = 8 : t.lanes = 1073741824 : t.lanes = 1, null) : (o = r.children, e = r.fallback, a ? (r = t.mode, a = t.child, o = {
      mode: "hidden",
      children: o
    }, !(r & 1) && a !== null ? (a.childLanes = 0, a.pendingProps = o) : a = so(o, r, 0, null), e = Vn(e, r, n, null), a.return = t, e.return = t, a.sibling = e, t.child = a, t.child.memoizedState = Ls(n), t.memoizedState = Ds, e) : Pi(t, o));
    if (l = e.memoizedState, l !== null && (i = l.dehydrated, i !== null)) return Ah(e, t, o, r, i, l, n);
    if (a) {
      a = r.fallback, o = t.mode, l = e.child, i = l.sibling;
      var s = {
        mode: "hidden",
        children: r.children
      };
      return !(o & 1) && t.child !== l ? (r = t.child, r.childLanes = 0, r.pendingProps = s, t.deletions = null) : (r = jn(l, s), r.subtreeFlags = l.subtreeFlags & 14680064), i !== null ? a = jn(i, a) : (a = Vn(a, o, n, null), a.flags |= 2), a.return = t, r.return = t, r.sibling = a, t.child = r, r = a, a = t.child, o = e.child.memoizedState, o = o === null ? Ls(n) : {
        baseLanes: o.baseLanes | n,
        cachePool: null,
        transitions: o.transitions
      }, a.memoizedState = o, a.childLanes = e.childLanes & ~n, t.memoizedState = Ds, r;
    }
    return a = e.child, e = a.sibling, r = jn(a, {
      mode: "visible",
      children: r.children
    }), !(t.mode & 1) && (r.lanes = n), r.return = t, r.sibling = null, e !== null && (n = t.deletions, n === null ? (t.deletions = [
      e
    ], t.flags |= 16) : n.push(e)), t.child = r, t.memoizedState = null, r;
  }
  function Pi(e, t) {
    return t = so({
      mode: "visible",
      children: t
    }, e.mode, 0, null), t.return = e, e.child = t;
  }
  function ta(e, t, n, r) {
    return r !== null && vi(r), _r(t, e.child, null, n), e = Pi(t, t.pendingProps.children), e.flags |= 2, t.memoizedState = null, e;
  }
  function Ah(e, t, n, r, l, a, o) {
    if (n) return t.flags & 256 ? (t.flags &= -257, r = Bo(Error(P(422))), ta(e, t, o, r)) : t.memoizedState !== null ? (t.child = e.child, t.flags |= 128, null) : (a = r.fallback, l = t.mode, r = so({
      mode: "visible",
      children: r.children
    }, l, 0, null), a = Vn(a, l, o, null), a.flags |= 2, r.return = t, a.return = t, r.sibling = a, t.child = r, t.mode & 1 && _r(t, e.child, null, o), t.child.memoizedState = Ls(o), t.memoizedState = Ds, a);
    if (!(t.mode & 1)) return ta(e, t, o, null);
    if (l.data === "$!") {
      if (r = l.nextSibling && l.nextSibling.dataset, r) var i = r.dgst;
      return r = i, a = Error(P(419)), r = Bo(a, r, void 0), ta(e, t, o, r);
    }
    if (i = (o & e.childLanes) !== 0, ct || i) {
      if (r = He, r !== null) {
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
        l = l & (r.suspendedLanes | o) ? 0 : l, l !== 0 && l !== a.retryLane && (a.retryLane = l, nn(e, l), zt(r, e, l, -1));
      }
      return Ii(), r = Bo(Error(P(421))), ta(e, t, o, r);
    }
    return l.data === "$?" ? (t.flags |= 128, t.child = e.child, t = Gh.bind(null, e), l._reactRetry = t, null) : (e = a.treeContext, wt = wn(l.nextSibling), St = t, Ne = true, At = null, e !== null && (Et[Ct++] = Jt, Et[Ct++] = Zt, Et[Ct++] = Hn, Jt = e.id, Zt = e.overflow, Hn = t), t = Pi(t, r.children), t.flags |= 4096, t);
  }
  function nc(e, t, n) {
    e.lanes |= t;
    var r = e.alternate;
    r !== null && (r.lanes |= t), _s(e.return, t, n);
  }
  function Vo(e, t, n, r, l) {
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
  function If(e, t, n) {
    var r = t.pendingProps, l = r.revealOrder, a = r.tail;
    if (nt(e, t, r.children, n), r = Ce.current, r & 2) r = r & 1 | 2, t.flags |= 128;
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
    if (xe(Ce, r), !(t.mode & 1)) t.memoizedState = null;
    else switch (l) {
      case "forwards":
        for (n = t.child, l = null; n !== null; ) e = n.alternate, e !== null && za(e) === null && (l = n), n = n.sibling;
        n = l, n === null ? (l = t.child, t.child = null) : (l = n.sibling, n.sibling = null), Vo(t, false, l, n, a);
        break;
      case "backwards":
        for (n = null, l = t.child, t.child = null; l !== null; ) {
          if (e = l.alternate, e !== null && za(e) === null) {
            t.child = l;
            break;
          }
          e = l.sibling, l.sibling = n, n = l, l = e;
        }
        Vo(t, true, n, null, a);
        break;
      case "together":
        Vo(t, false, null, null, void 0);
        break;
      default:
        t.memoizedState = null;
    }
    return t.child;
  }
  function ga(e, t) {
    !(t.mode & 1) && e !== null && (e.alternate = null, t.alternate = null, t.flags |= 2);
  }
  function rn(e, t, n) {
    if (e !== null && (t.dependencies = e.dependencies), Kn |= t.lanes, !(n & t.childLanes)) return null;
    if (e !== null && t.child !== e.child) throw Error(P(153));
    if (t.child !== null) {
      for (e = t.child, n = jn(e, e.pendingProps), t.child = n, n.return = t; e.sibling !== null; ) e = e.sibling, n = n.sibling = jn(e, e.pendingProps), n.return = t;
      n.sibling = null;
    }
    return t.child;
  }
  function Ih(e, t, n) {
    switch (t.tag) {
      case 3:
        Of(t), Cr();
        break;
      case 5:
        uf(t);
        break;
      case 1:
        ft(t.type) && Ma(t);
        break;
      case 4:
        Ni(t, t.stateNode.containerInfo);
        break;
      case 10:
        var r = t.type._context, l = t.memoizedProps.value;
        xe(Oa, r._currentValue), r._currentValue = l;
        break;
      case 13:
        if (r = t.memoizedState, r !== null) return r.dehydrated !== null ? (xe(Ce, Ce.current & 1), t.flags |= 128, null) : n & t.child.childLanes ? Af(e, t, n) : (xe(Ce, Ce.current & 1), e = rn(e, t, n), e !== null ? e.sibling : null);
        xe(Ce, Ce.current & 1);
        break;
      case 19:
        if (r = (n & t.childLanes) !== 0, e.flags & 128) {
          if (r) return If(e, t, n);
          t.flags |= 128;
        }
        if (l = t.memoizedState, l !== null && (l.rendering = null, l.tail = null, l.lastEffect = null), xe(Ce, Ce.current), r) break;
        return null;
      case 22:
      case 23:
        return t.lanes = 0, Df(e, t, n);
    }
    return rn(e, t, n);
  }
  var zf, Os, Uf, Ff;
  zf = function(e, t) {
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
  Uf = function(e, t, n, r) {
    var l = e.memoizedProps;
    if (l !== r) {
      e = t.stateNode, Fn(Qt.current);
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
        else c === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, i = i ? i.__html : void 0, s != null && i !== s && (a = a || []).push(c, s)) : c === "children" ? typeof s != "string" && typeof s != "number" || (a = a || []).push(c, "" + s) : c !== "suppressContentEditableWarning" && c !== "suppressHydrationWarning" && (ul.hasOwnProperty(c) ? (s != null && c === "onScroll" && ye("scroll", e), a || i === s || (a = [])) : (a = a || []).push(c, s));
      }
      n && (a = a || []).push("style", n);
      var c = a;
      (t.updateQueue = c) && (t.flags |= 4);
    }
  };
  Ff = function(e, t, n, r) {
    n !== r && (t.flags |= 4);
  };
  function Vr(e, t) {
    if (!Ne) switch (e.tailMode) {
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
  function zh(e, t, n) {
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
        return ft(t.type) && Pa(), Ze(t), null;
      case 3:
        return r = t.stateNode, Rr(), we(dt), we(et), Ei(), r.pendingContext && (r.context = r.pendingContext, r.pendingContext = null), (e === null || e.child === null) && (ql(t) ? t.flags |= 4 : e === null || e.memoizedState.isDehydrated && !(t.flags & 256) || (t.flags |= 1024, At !== null && (Vs(At), At = null))), Os(e, t), Ze(t), null;
      case 5:
        ji(t);
        var l = Fn(Sl.current);
        if (n = t.type, e !== null && t.stateNode != null) Uf(e, t, n, r, l), e.ref !== t.ref && (t.flags |= 512, t.flags |= 2097152);
        else {
          if (!r) {
            if (t.stateNode === null) throw Error(P(166));
            return Ze(t), null;
          }
          if (e = Fn(Qt.current), ql(t)) {
            r = t.stateNode, n = t.type;
            var a = t.memoizedProps;
            switch (r[Wt] = t, r[yl] = a, e = (t.mode & 1) !== 0, n) {
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
                for (l = 0; l < Jr.length; l++) ye(Jr[l], r);
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
            o = l.nodeType === 9 ? l : l.ownerDocument, e === "http://www.w3.org/1999/xhtml" && (e = md(n)), e === "http://www.w3.org/1999/xhtml" ? n === "script" ? (e = o.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild)) : typeof r.is == "string" ? e = o.createElement(n, {
              is: r.is
            }) : (e = o.createElement(n), n === "select" && (o = e, r.multiple ? o.multiple = true : r.size && (o.size = r.size))) : e = o.createElementNS(e, n), e[Wt] = t, e[yl] = r, zf(e, t, false, false), t.stateNode = e;
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
                  for (l = 0; l < Jr.length; l++) ye(Jr[l], e);
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
                a === "style" ? gd(e, s) : a === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, s != null && pd(e, s)) : a === "children" ? typeof s == "string" ? (n !== "textarea" || s !== "") && cl(e, s) : typeof s == "number" && cl(e, "" + s) : a !== "suppressContentEditableWarning" && a !== "suppressHydrationWarning" && a !== "autoFocus" && (ul.hasOwnProperty(a) ? s != null && a === "onScroll" && ye("scroll", e) : s != null && ti(e, a, s, o));
              }
              switch (n) {
                case "input":
                  Wl(e), fu(e, r, false);
                  break;
                case "textarea":
                  Wl(e), pu(e);
                  break;
                case "option":
                  r.value != null && e.setAttribute("value", "" + Cn(r.value));
                  break;
                case "select":
                  e.multiple = !!r.multiple, a = r.value, a != null ? xr(e, !!r.multiple, a, false) : r.defaultValue != null && xr(e, !!r.multiple, r.defaultValue, true);
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
        if (e && t.stateNode != null) Ff(e, t, e.memoizedProps, r);
        else {
          if (typeof r != "string" && t.stateNode === null) throw Error(P(166));
          if (n = Fn(Sl.current), Fn(Qt.current), ql(t)) {
            if (r = t.stateNode, n = t.memoizedProps, r[Wt] = t, (a = r.nodeValue !== n) && (e = St, e !== null)) switch (e.tag) {
              case 3:
                Zl(r.nodeValue, n, (e.mode & 1) !== 0);
                break;
              case 5:
                e.memoizedProps.suppressHydrationWarning !== true && Zl(r.nodeValue, n, (e.mode & 1) !== 0);
            }
            a && (t.flags |= 4);
          } else r = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(r), r[Wt] = t, t.stateNode = r;
        }
        return Ze(t), null;
      case 13:
        if (we(Ce), r = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
          if (Ne && wt !== null && t.mode & 1 && !(t.flags & 128)) rf(), Cr(), t.flags |= 98560, a = false;
          else if (a = ql(t), r !== null && r.dehydrated !== null) {
            if (e === null) {
              if (!a) throw Error(P(318));
              if (a = t.memoizedState, a = a !== null ? a.dehydrated : null, !a) throw Error(P(317));
              a[Wt] = t;
            } else Cr(), !(t.flags & 128) && (t.memoizedState = null), t.flags |= 4;
            Ze(t), a = false;
          } else At !== null && (Vs(At), At = null), a = true;
          if (!a) return t.flags & 65536 ? t : null;
        }
        return t.flags & 128 ? (t.lanes = n, t) : (r = r !== null, r !== (e !== null && e.memoizedState !== null) && r && (t.child.flags |= 8192, t.mode & 1 && (e === null || Ce.current & 1 ? Be === 0 && (Be = 3) : Ii())), t.updateQueue !== null && (t.flags |= 4), Ze(t), null);
      case 4:
        return Rr(), Os(e, t), e === null && vl(t.stateNode.containerInfo), Ze(t), null;
      case 10:
        return wi(t.type._context), Ze(t), null;
      case 17:
        return ft(t.type) && Pa(), Ze(t), null;
      case 19:
        if (we(Ce), a = t.memoizedState, a === null) return Ze(t), null;
        if (r = (t.flags & 128) !== 0, o = a.rendering, o === null) if (r) Vr(a, false);
        else {
          if (Be !== 0 || e !== null && e.flags & 128) for (e = t.child; e !== null; ) {
            if (o = za(e), o !== null) {
              for (t.flags |= 128, Vr(a, false), r = o.updateQueue, r !== null && (t.updateQueue = r, t.flags |= 4), t.subtreeFlags = 0, r = n, n = t.child; n !== null; ) a = n, e = r, a.flags &= 14680066, o = a.alternate, o === null ? (a.childLanes = 0, a.lanes = e, a.child = null, a.subtreeFlags = 0, a.memoizedProps = null, a.memoizedState = null, a.updateQueue = null, a.dependencies = null, a.stateNode = null) : (a.childLanes = o.childLanes, a.lanes = o.lanes, a.child = o.child, a.subtreeFlags = 0, a.deletions = null, a.memoizedProps = o.memoizedProps, a.memoizedState = o.memoizedState, a.updateQueue = o.updateQueue, a.type = o.type, e = o.dependencies, a.dependencies = e === null ? null : {
                lanes: e.lanes,
                firstContext: e.firstContext
              }), n = n.sibling;
              return xe(Ce, Ce.current & 1 | 2), t.child;
            }
            e = e.sibling;
          }
          a.tail !== null && Le() > Tr && (t.flags |= 128, r = true, Vr(a, false), t.lanes = 4194304);
        }
        else {
          if (!r) if (e = za(o), e !== null) {
            if (t.flags |= 128, r = true, n = e.updateQueue, n !== null && (t.updateQueue = n, t.flags |= 4), Vr(a, true), a.tail === null && a.tailMode === "hidden" && !o.alternate && !Ne) return Ze(t), null;
          } else 2 * Le() - a.renderingStartTime > Tr && n !== 1073741824 && (t.flags |= 128, r = true, Vr(a, false), t.lanes = 4194304);
          a.isBackwards ? (o.sibling = t.child, t.child = o) : (n = a.last, n !== null ? n.sibling = o : t.child = o, a.last = o);
        }
        return a.tail !== null ? (t = a.tail, a.rendering = t, a.tail = t.sibling, a.renderingStartTime = Le(), t.sibling = null, n = Ce.current, xe(Ce, r ? n & 1 | 2 : n & 1), t) : (Ze(t), null);
      case 22:
      case 23:
        return Ai(), r = t.memoizedState !== null, e !== null && e.memoizedState !== null !== r && (t.flags |= 8192), r && t.mode & 1 ? xt & 1073741824 && (Ze(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : Ze(t), null;
      case 24:
        return null;
      case 25:
        return null;
    }
    throw Error(P(156, t.tag));
  }
  function Uh(e, t) {
    switch (gi(t), t.tag) {
      case 1:
        return ft(t.type) && Pa(), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 3:
        return Rr(), we(dt), we(et), Ei(), e = t.flags, e & 65536 && !(e & 128) ? (t.flags = e & -65537 | 128, t) : null;
      case 5:
        return ji(t), null;
      case 13:
        if (we(Ce), e = t.memoizedState, e !== null && e.dehydrated !== null) {
          if (t.alternate === null) throw Error(P(340));
          Cr();
        }
        return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
      case 19:
        return we(Ce), null;
      case 4:
        return Rr(), null;
      case 10:
        return wi(t.type._context), null;
      case 22:
      case 23:
        return Ai(), null;
      case 24:
        return null;
      default:
        return null;
    }
  }
  var na = false, qe = false, Fh = typeof WeakSet == "function" ? WeakSet : Set, O = null;
  function gr(e, t) {
    var n = e.ref;
    if (n !== null) if (typeof n == "function") try {
      n(null);
    } catch (r) {
      De(e, t, r);
    }
    else n.current = null;
  }
  function As(e, t, n) {
    try {
      n();
    } catch (r) {
      De(e, t, r);
    }
  }
  var rc = false;
  function $h(e, t) {
    if (ys = _a, e = Wd(), pi(e)) {
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
          var o = 0, i = -1, s = -1, c = 0, m = 0, d = e, p = null;
          t: for (; ; ) {
            for (var x; d !== n || l !== 0 && d.nodeType !== 3 || (i = o + l), d !== a || r !== 0 && d.nodeType !== 3 || (s = o + r), d.nodeType === 3 && (o += d.nodeValue.length), (x = d.firstChild) !== null; ) p = d, d = x;
            for (; ; ) {
              if (d === e) break t;
              if (p === n && ++c === l && (i = o), p === a && ++m === r && (s = o), (x = d.nextSibling) !== null) break;
              d = p, p = d.parentNode;
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
    }, _a = false, O = t; O !== null; ) if (t = O, e = t.child, (t.subtreeFlags & 1028) !== 0 && e !== null) e.return = t, O = e;
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
              var k = w.memoizedProps, R = w.memoizedState, h = t.stateNode, f = h.getSnapshotBeforeUpdate(t.elementType === t.type ? k : Dt(t.type, k), R);
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
    return w = rc, rc = false, w;
  }
  function al(e, t, n) {
    var r = t.updateQueue;
    if (r = r !== null ? r.lastEffect : null, r !== null) {
      var l = r = r.next;
      do {
        if ((l.tag & e) === e) {
          var a = l.destroy;
          l.destroy = void 0, a !== void 0 && As(t, n, a);
        }
        l = l.next;
      } while (l !== r);
    }
  }
  function ao(e, t) {
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
  function Is(e) {
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
  function $f(e) {
    var t = e.alternate;
    t !== null && (e.alternate = null, $f(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && (delete t[Wt], delete t[yl], delete t[Ns], delete t[Nh], delete t[jh])), e.stateNode = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
  }
  function Bf(e) {
    return e.tag === 5 || e.tag === 3 || e.tag === 4;
  }
  function lc(e) {
    e: for (; ; ) {
      for (; e.sibling === null; ) {
        if (e.return === null || Bf(e.return)) return null;
        e = e.return;
      }
      for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
        if (e.flags & 2 || e.child === null || e.tag === 4) continue e;
        e.child.return = e, e = e.child;
      }
      if (!(e.flags & 2)) return e.stateNode;
    }
  }
  function zs(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.nodeType === 8 ? n.parentNode.insertBefore(e, t) : n.insertBefore(e, t) : (n.nodeType === 8 ? (t = n.parentNode, t.insertBefore(e, n)) : (t = n, t.appendChild(e)), n = n._reactRootContainer, n != null || t.onclick !== null || (t.onclick = Ta));
    else if (r !== 4 && (e = e.child, e !== null)) for (zs(e, t, n), e = e.sibling; e !== null; ) zs(e, t, n), e = e.sibling;
  }
  function Us(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.insertBefore(e, t) : n.appendChild(e);
    else if (r !== 4 && (e = e.child, e !== null)) for (Us(e, t, n), e = e.sibling; e !== null; ) Us(e, t, n), e = e.sibling;
  }
  var Ye = null, Lt = false;
  function sn(e, t, n) {
    for (n = n.child; n !== null; ) Vf(e, t, n), n = n.sibling;
  }
  function Vf(e, t, n) {
    if (Ht && typeof Ht.onCommitFiberUnmount == "function") try {
      Ht.onCommitFiberUnmount(Ja, n);
    } catch {
    }
    switch (n.tag) {
      case 5:
        qe || gr(n, t);
      case 6:
        var r = Ye, l = Lt;
        Ye = null, sn(e, t, n), Ye = r, Lt = l, Ye !== null && (Lt ? (e = Ye, n = n.stateNode, e.nodeType === 8 ? e.parentNode.removeChild(n) : e.removeChild(n)) : Ye.removeChild(n.stateNode));
        break;
      case 18:
        Ye !== null && (Lt ? (e = Ye, n = n.stateNode, e.nodeType === 8 ? Ao(e.parentNode, n) : e.nodeType === 1 && Ao(e, n), pl(e)) : Ao(Ye, n.stateNode));
        break;
      case 4:
        r = Ye, l = Lt, Ye = n.stateNode.containerInfo, Lt = true, sn(e, t, n), Ye = r, Lt = l;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        if (!qe && (r = n.updateQueue, r !== null && (r = r.lastEffect, r !== null))) {
          l = r = r.next;
          do {
            var a = l, o = a.destroy;
            a = a.tag, o !== void 0 && (a & 2 || a & 4) && As(n, t, o), l = l.next;
          } while (l !== r);
        }
        sn(e, t, n);
        break;
      case 1:
        if (!qe && (gr(n, t), r = n.stateNode, typeof r.componentWillUnmount == "function")) try {
          r.props = n.memoizedProps, r.state = n.memoizedState, r.componentWillUnmount();
        } catch (i) {
          De(n, t, i);
        }
        sn(e, t, n);
        break;
      case 21:
        sn(e, t, n);
        break;
      case 22:
        n.mode & 1 ? (qe = (r = qe) || n.memoizedState !== null, sn(e, t, n), qe = r) : sn(e, t, n);
        break;
      default:
        sn(e, t, n);
    }
  }
  function ac(e) {
    var t = e.updateQueue;
    if (t !== null) {
      e.updateQueue = null;
      var n = e.stateNode;
      n === null && (n = e.stateNode = new Fh()), t.forEach(function(r) {
        var l = Xh.bind(null, e, r);
        n.has(r) || (n.add(r), r.then(l, l));
      });
    }
  }
  function Mt(e, t) {
    var n = t.deletions;
    if (n !== null) for (var r = 0; r < n.length; r++) {
      var l = n[r];
      try {
        var a = e, o = t, i = o;
        e: for (; i !== null; ) {
          switch (i.tag) {
            case 5:
              Ye = i.stateNode, Lt = false;
              break e;
            case 3:
              Ye = i.stateNode.containerInfo, Lt = true;
              break e;
            case 4:
              Ye = i.stateNode.containerInfo, Lt = true;
              break e;
          }
          i = i.return;
        }
        if (Ye === null) throw Error(P(160));
        Vf(a, o, l), Ye = null, Lt = false;
        var s = l.alternate;
        s !== null && (s.return = null), l.return = null;
      } catch (c) {
        De(l, t, c);
      }
    }
    if (t.subtreeFlags & 12854) for (t = t.child; t !== null; ) Wf(t, e), t = t.sibling;
  }
  function Wf(e, t) {
    var n = e.alternate, r = e.flags;
    switch (e.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        if (Mt(t, e), Bt(e), r & 4) {
          try {
            al(3, e, e.return), ao(3, e);
          } catch (k) {
            De(e, e.return, k);
          }
          try {
            al(5, e, e.return);
          } catch (k) {
            De(e, e.return, k);
          }
        }
        break;
      case 1:
        Mt(t, e), Bt(e), r & 512 && n !== null && gr(n, n.return);
        break;
      case 5:
        if (Mt(t, e), Bt(e), r & 512 && n !== null && gr(n, n.return), e.flags & 32) {
          var l = e.stateNode;
          try {
            cl(l, "");
          } catch (k) {
            De(e, e.return, k);
          }
        }
        if (r & 4 && (l = e.stateNode, l != null)) {
          var a = e.memoizedProps, o = n !== null ? n.memoizedProps : a, i = e.type, s = e.updateQueue;
          if (e.updateQueue = null, s !== null) try {
            i === "input" && a.type === "radio" && a.name != null && dd(l, a), us(i, o);
            var c = us(i, a);
            for (o = 0; o < s.length; o += 2) {
              var m = s[o], d = s[o + 1];
              m === "style" ? gd(l, d) : m === "dangerouslySetInnerHTML" ? pd(l, d) : m === "children" ? cl(l, d) : ti(l, m, d, c);
            }
            switch (i) {
              case "input":
                ls(l, a);
                break;
              case "textarea":
                fd(l, a);
                break;
              case "select":
                var p = l._wrapperState.wasMultiple;
                l._wrapperState.wasMultiple = !!a.multiple;
                var x = a.value;
                x != null ? xr(l, !!a.multiple, x, false) : p !== !!a.multiple && (a.defaultValue != null ? xr(l, !!a.multiple, a.defaultValue, true) : xr(l, !!a.multiple, a.multiple ? [] : "", false));
            }
            l[yl] = a;
          } catch (k) {
            De(e, e.return, k);
          }
        }
        break;
      case 6:
        if (Mt(t, e), Bt(e), r & 4) {
          if (e.stateNode === null) throw Error(P(162));
          l = e.stateNode, a = e.memoizedProps;
          try {
            l.nodeValue = a;
          } catch (k) {
            De(e, e.return, k);
          }
        }
        break;
      case 3:
        if (Mt(t, e), Bt(e), r & 4 && n !== null && n.memoizedState.isDehydrated) try {
          pl(t.containerInfo);
        } catch (k) {
          De(e, e.return, k);
        }
        break;
      case 4:
        Mt(t, e), Bt(e);
        break;
      case 13:
        Mt(t, e), Bt(e), l = e.child, l.flags & 8192 && (a = l.memoizedState !== null, l.stateNode.isHidden = a, !a || l.alternate !== null && l.alternate.memoizedState !== null || (Li = Le())), r & 4 && ac(e);
        break;
      case 22:
        if (m = n !== null && n.memoizedState !== null, e.mode & 1 ? (qe = (c = qe) || m, Mt(t, e), qe = c) : Mt(t, e), Bt(e), r & 8192) {
          if (c = e.memoizedState !== null, (e.stateNode.isHidden = c) && !m && e.mode & 1) for (O = e, m = e.child; m !== null; ) {
            for (d = O = m; O !== null; ) {
              switch (p = O, x = p.child, p.tag) {
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
                    } catch (k) {
                      De(r, n, k);
                    }
                  }
                  break;
                case 5:
                  gr(p, p.return);
                  break;
                case 22:
                  if (p.memoizedState !== null) {
                    sc(d);
                    continue;
                  }
              }
              x !== null ? (x.return = p, O = x) : sc(d);
            }
            m = m.sibling;
          }
          e: for (m = null, d = e; ; ) {
            if (d.tag === 5) {
              if (m === null) {
                m = d;
                try {
                  l = d.stateNode, c ? (a = l.style, typeof a.setProperty == "function" ? a.setProperty("display", "none", "important") : a.display = "none") : (i = d.stateNode, s = d.memoizedProps.style, o = s != null && s.hasOwnProperty("display") ? s.display : null, i.style.display = hd("display", o));
                } catch (k) {
                  De(e, e.return, k);
                }
              }
            } else if (d.tag === 6) {
              if (m === null) try {
                d.stateNode.nodeValue = c ? "" : d.memoizedProps;
              } catch (k) {
                De(e, e.return, k);
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
        Mt(t, e), Bt(e), r & 4 && ac(e);
        break;
      case 21:
        break;
      default:
        Mt(t, e), Bt(e);
    }
  }
  function Bt(e) {
    var t = e.flags;
    if (t & 2) {
      try {
        e: {
          for (var n = e.return; n !== null; ) {
            if (Bf(n)) {
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
            var a = lc(e);
            Us(e, a, l);
            break;
          case 3:
          case 4:
            var o = r.stateNode.containerInfo, i = lc(e);
            zs(e, i, o);
            break;
          default:
            throw Error(P(161));
        }
      } catch (s) {
        De(e, e.return, s);
      }
      e.flags &= -3;
    }
    t & 4096 && (e.flags &= -4097);
  }
  function Bh(e, t, n) {
    O = e, Hf(e);
  }
  function Hf(e, t, n) {
    for (var r = (e.mode & 1) !== 0; O !== null; ) {
      var l = O, a = l.child;
      if (l.tag === 22 && r) {
        var o = l.memoizedState !== null || na;
        if (!o) {
          var i = l.alternate, s = i !== null && i.memoizedState !== null || qe;
          i = na;
          var c = qe;
          if (na = o, (qe = s) && !c) for (O = l; O !== null; ) o = O, s = o.child, o.tag === 22 && o.memoizedState !== null ? ic(l) : s !== null ? (s.return = o, O = s) : ic(l);
          for (; a !== null; ) O = a, Hf(a), a = a.sibling;
          O = l, na = i, qe = c;
        }
        oc(e);
      } else l.subtreeFlags & 8772 && a !== null ? (a.return = l, O = a) : oc(e);
    }
  }
  function oc(e) {
    for (; O !== null; ) {
      var t = O;
      if (t.flags & 8772) {
        var n = t.alternate;
        try {
          if (t.flags & 8772) switch (t.tag) {
            case 0:
            case 11:
            case 15:
              qe || ao(5, t);
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
              throw Error(P(163));
          }
          qe || t.flags & 512 && Is(t);
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
  function sc(e) {
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
  function ic(e) {
    for (; O !== null; ) {
      var t = O;
      try {
        switch (t.tag) {
          case 0:
          case 11:
          case 15:
            var n = t.return;
            try {
              ao(4, t);
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
              Is(t);
            } catch (s) {
              De(t, a, s);
            }
            break;
          case 5:
            var o = t.return;
            try {
              Is(t);
            } catch (s) {
              De(t, o, s);
            }
        }
      } catch (s) {
        De(t, t.return, s);
      }
      if (t === e) {
        O = null;
        break;
      }
      var i = t.sibling;
      if (i !== null) {
        i.return = t.return, O = i;
        break;
      }
      O = t.return;
    }
  }
  var Vh = Math.ceil, $a = ln.ReactCurrentDispatcher, Mi = ln.ReactCurrentOwner, Rt = ln.ReactCurrentBatchConfig, ae = 0, He = null, Fe = null, Ge = 0, xt = 0, vr = bn(0), Be = 0, El = null, Kn = 0, oo = 0, Di = 0, ol = null, ut = null, Li = 0, Tr = 1 / 0, Gt = null, Ba = false, Fs = null, kn = null, ra = false, hn = null, Va = 0, sl = 0, $s = null, va = -1, xa = 0;
  function rt() {
    return ae & 6 ? Le() : va !== -1 ? va : va = Le();
  }
  function Nn(e) {
    return e.mode & 1 ? ae & 2 && Ge !== 0 ? Ge & -Ge : Ch.transition !== null ? (xa === 0 && (xa = Rd()), xa) : (e = de, e !== 0 || (e = window.event, e = e === void 0 ? 16 : Od(e.type)), e) : 1;
  }
  function zt(e, t, n, r) {
    if (50 < sl) throw sl = 0, $s = null, Error(P(185));
    bl(e, n, r), (!(ae & 2) || e !== He) && (e === He && (!(ae & 2) && (oo |= n), Be === 4 && mn(e, Ge)), mt(e, r), n === 1 && ae === 0 && !(t.mode & 1) && (Tr = Le() + 500, no && Tn()));
  }
  function mt(e, t) {
    var n = e.callbackNode;
    Cp(e, t);
    var r = Ca(e, e === He ? Ge : 0);
    if (r === 0) n !== null && vu(n), e.callbackNode = null, e.callbackPriority = 0;
    else if (t = r & -r, e.callbackPriority !== t) {
      if (n != null && vu(n), t === 1) e.tag === 0 ? Eh(uc.bind(null, e)) : ef(uc.bind(null, e)), Sh(function() {
        !(ae & 6) && Tn();
      }), n = null;
      else {
        switch (bd(r)) {
          case 1:
            n = oi;
            break;
          case 4:
            n = Cd;
            break;
          case 16:
            n = Ea;
            break;
          case 536870912:
            n = _d;
            break;
          default:
            n = Ea;
        }
        n = qf(n, Qf.bind(null, e));
      }
      e.callbackPriority = t, e.callbackNode = n;
    }
  }
  function Qf(e, t) {
    if (va = -1, xa = 0, ae & 6) throw Error(P(327));
    var n = e.callbackNode;
    if (Nr() && e.callbackNode !== n) return null;
    var r = Ca(e, e === He ? Ge : 0);
    if (r === 0) return null;
    if (r & 30 || r & e.expiredLanes || t) t = Wa(e, r);
    else {
      t = r;
      var l = ae;
      ae |= 2;
      var a = Yf();
      (He !== e || Ge !== t) && (Gt = null, Tr = Le() + 500, Bn(e, t));
      do
        try {
          Qh();
          break;
        } catch (i) {
          Kf(e, i);
        }
      while (true);
      yi(), $a.current = a, ae = l, Fe !== null ? t = 0 : (He = null, Ge = 0, t = Be);
    }
    if (t !== 0) {
      if (t === 2 && (l = ps(e), l !== 0 && (r = l, t = Bs(e, l))), t === 1) throw n = El, Bn(e, 0), mn(e, r), mt(e, Le()), n;
      if (t === 6) mn(e, r);
      else {
        if (l = e.current.alternate, !(r & 30) && !Wh(l) && (t = Wa(e, r), t === 2 && (a = ps(e), a !== 0 && (r = a, t = Bs(e, a))), t === 1)) throw n = El, Bn(e, 0), mn(e, r), mt(e, Le()), n;
        switch (e.finishedWork = l, e.finishedLanes = r, t) {
          case 0:
          case 1:
            throw Error(P(345));
          case 2:
            On(e, ut, Gt);
            break;
          case 3:
            if (mn(e, r), (r & 130023424) === r && (t = Li + 500 - Le(), 10 < t)) {
              if (Ca(e, 0) !== 0) break;
              if (l = e.suspendedLanes, (l & r) !== r) {
                rt(), e.pingedLanes |= e.suspendedLanes & l;
                break;
              }
              e.timeoutHandle = ks(On.bind(null, e, ut, Gt), t);
              break;
            }
            On(e, ut, Gt);
            break;
          case 4:
            if (mn(e, r), (r & 4194240) === r) break;
            for (t = e.eventTimes, l = -1; 0 < r; ) {
              var o = 31 - It(r);
              a = 1 << o, o = t[o], o > l && (l = o), r &= ~a;
            }
            if (r = l, r = Le() - r, r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * Vh(r / 1960)) - r, 10 < r) {
              e.timeoutHandle = ks(On.bind(null, e, ut, Gt), r);
              break;
            }
            On(e, ut, Gt);
            break;
          case 5:
            On(e, ut, Gt);
            break;
          default:
            throw Error(P(329));
        }
      }
    }
    return mt(e, Le()), e.callbackNode === n ? Qf.bind(null, e) : null;
  }
  function Bs(e, t) {
    var n = ol;
    return e.current.memoizedState.isDehydrated && (Bn(e, t).flags |= 256), e = Wa(e, t), e !== 2 && (t = ut, ut = n, t !== null && Vs(t)), e;
  }
  function Vs(e) {
    ut === null ? ut = e : ut.push.apply(ut, e);
  }
  function Wh(e) {
    for (var t = e; ; ) {
      if (t.flags & 16384) {
        var n = t.updateQueue;
        if (n !== null && (n = n.stores, n !== null)) for (var r = 0; r < n.length; r++) {
          var l = n[r], a = l.getSnapshot;
          l = l.value;
          try {
            if (!Ut(a(), l)) return false;
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
    for (t &= ~Di, t &= ~oo, e.suspendedLanes |= t, e.pingedLanes &= ~t, e = e.expirationTimes; 0 < t; ) {
      var n = 31 - It(t), r = 1 << n;
      e[n] = -1, t &= ~r;
    }
  }
  function uc(e) {
    if (ae & 6) throw Error(P(327));
    Nr();
    var t = Ca(e, 0);
    if (!(t & 1)) return mt(e, Le()), null;
    var n = Wa(e, t);
    if (e.tag !== 0 && n === 2) {
      var r = ps(e);
      r !== 0 && (t = r, n = Bs(e, r));
    }
    if (n === 1) throw n = El, Bn(e, 0), mn(e, t), mt(e, Le()), n;
    if (n === 6) throw Error(P(345));
    return e.finishedWork = e.current.alternate, e.finishedLanes = t, On(e, ut, Gt), mt(e, Le()), null;
  }
  function Oi(e, t) {
    var n = ae;
    ae |= 1;
    try {
      return e(t);
    } finally {
      ae = n, ae === 0 && (Tr = Le() + 500, no && Tn());
    }
  }
  function Yn(e) {
    hn !== null && hn.tag === 0 && !(ae & 6) && Nr();
    var t = ae;
    ae |= 1;
    var n = Rt.transition, r = de;
    try {
      if (Rt.transition = null, de = 1, e) return e();
    } finally {
      de = r, Rt.transition = n, ae = t, !(ae & 6) && Tn();
    }
  }
  function Ai() {
    xt = vr.current, we(vr);
  }
  function Bn(e, t) {
    e.finishedWork = null, e.finishedLanes = 0;
    var n = e.timeoutHandle;
    if (n !== -1 && (e.timeoutHandle = -1, wh(n)), Fe !== null) for (n = Fe.return; n !== null; ) {
      var r = n;
      switch (gi(r), r.tag) {
        case 1:
          r = r.type.childContextTypes, r != null && Pa();
          break;
        case 3:
          Rr(), we(dt), we(et), Ei();
          break;
        case 5:
          ji(r);
          break;
        case 4:
          Rr();
          break;
        case 13:
          we(Ce);
          break;
        case 19:
          we(Ce);
          break;
        case 10:
          wi(r.type._context);
          break;
        case 22:
        case 23:
          Ai();
      }
      n = n.return;
    }
    if (He = e, Fe = e = jn(e.current, null), Ge = xt = t, Be = 0, El = null, Di = oo = Kn = 0, ut = ol = null, Un !== null) {
      for (t = 0; t < Un.length; t++) if (n = Un[t], r = n.interleaved, r !== null) {
        n.interleaved = null;
        var l = r.next, a = n.pending;
        if (a !== null) {
          var o = a.next;
          a.next = l, r.next = o;
        }
        n.pending = r;
      }
      Un = null;
    }
    return e;
  }
  function Kf(e, t) {
    do {
      var n = Fe;
      try {
        if (yi(), pa.current = Fa, Ua) {
          for (var r = _e.memoizedState; r !== null; ) {
            var l = r.queue;
            l !== null && (l.pending = null), r = r.next;
          }
          Ua = false;
        }
        if (Qn = 0, We = $e = _e = null, ll = false, kl = 0, Mi.current = null, n === null || n.return === null) {
          Be = 1, El = t, Fe = null;
          break;
        }
        e: {
          var a = e, o = n.return, i = n, s = t;
          if (t = Ge, i.flags |= 32768, s !== null && typeof s == "object" && typeof s.then == "function") {
            var c = s, m = i, d = m.tag;
            if (!(m.mode & 1) && (d === 0 || d === 11 || d === 15)) {
              var p = m.alternate;
              p ? (m.updateQueue = p.updateQueue, m.memoizedState = p.memoizedState, m.lanes = p.lanes) : (m.updateQueue = null, m.memoizedState = null);
            }
            var x = Xu(o);
            if (x !== null) {
              x.flags &= -257, Ju(x, o, i, a, t), x.mode & 1 && Gu(a, c, t), t = x, s = c;
              var w = t.updateQueue;
              if (w === null) {
                var k = /* @__PURE__ */ new Set();
                k.add(s), t.updateQueue = k;
              } else w.add(s);
              break e;
            } else {
              if (!(t & 1)) {
                Gu(a, c, t), Ii();
                break e;
              }
              s = Error(P(426));
            }
          } else if (Ne && i.mode & 1) {
            var R = Xu(o);
            if (R !== null) {
              !(R.flags & 65536) && (R.flags |= 256), Ju(R, o, i, a, t), vi(br(s, i));
              break e;
            }
          }
          a = s = br(s, i), Be !== 4 && (Be = 2), ol === null ? ol = [
            a
          ] : ol.push(a), a = o;
          do {
            switch (a.tag) {
              case 3:
                a.flags |= 65536, t &= -t, a.lanes |= t;
                var h = Tf(a, s, t);
                Vu(a, h);
                break e;
              case 1:
                i = s;
                var f = a.type, v = a.stateNode;
                if (!(a.flags & 128) && (typeof f.getDerivedStateFromError == "function" || v !== null && typeof v.componentDidCatch == "function" && (kn === null || !kn.has(v)))) {
                  a.flags |= 65536, t &= -t, a.lanes |= t;
                  var E = Pf(a, i, t);
                  Vu(a, E);
                  break e;
                }
            }
            a = a.return;
          } while (a !== null);
        }
        Xf(n);
      } catch (_) {
        t = _, Fe === n && n !== null && (Fe = n = n.return);
        continue;
      }
      break;
    } while (true);
  }
  function Yf() {
    var e = $a.current;
    return $a.current = Fa, e === null ? Fa : e;
  }
  function Ii() {
    (Be === 0 || Be === 3 || Be === 2) && (Be = 4), He === null || !(Kn & 268435455) && !(oo & 268435455) || mn(He, Ge);
  }
  function Wa(e, t) {
    var n = ae;
    ae |= 2;
    var r = Yf();
    (He !== e || Ge !== t) && (Gt = null, Bn(e, t));
    do
      try {
        Hh();
        break;
      } catch (l) {
        Kf(e, l);
      }
    while (true);
    if (yi(), ae = n, $a.current = r, Fe !== null) throw Error(P(261));
    return He = null, Ge = 0, Be;
  }
  function Hh() {
    for (; Fe !== null; ) Gf(Fe);
  }
  function Qh() {
    for (; Fe !== null && !vp(); ) Gf(Fe);
  }
  function Gf(e) {
    var t = Zf(e.alternate, e, xt);
    e.memoizedProps = e.pendingProps, t === null ? Xf(e) : Fe = t, Mi.current = null;
  }
  function Xf(e) {
    var t = e;
    do {
      var n = t.alternate;
      if (e = t.return, t.flags & 32768) {
        if (n = Uh(n, t), n !== null) {
          n.flags &= 32767, Fe = n;
          return;
        }
        if (e !== null) e.flags |= 32768, e.subtreeFlags = 0, e.deletions = null;
        else {
          Be = 6, Fe = null;
          return;
        }
      } else if (n = zh(n, t, xt), n !== null) {
        Fe = n;
        return;
      }
      if (t = t.sibling, t !== null) {
        Fe = t;
        return;
      }
      Fe = t = e;
    } while (t !== null);
    Be === 0 && (Be = 5);
  }
  function On(e, t, n) {
    var r = de, l = Rt.transition;
    try {
      Rt.transition = null, de = 1, Kh(e, t, n, r);
    } finally {
      Rt.transition = l, de = r;
    }
    return null;
  }
  function Kh(e, t, n, r) {
    do
      Nr();
    while (hn !== null);
    if (ae & 6) throw Error(P(327));
    n = e.finishedWork;
    var l = e.finishedLanes;
    if (n === null) return null;
    if (e.finishedWork = null, e.finishedLanes = 0, n === e.current) throw Error(P(177));
    e.callbackNode = null, e.callbackPriority = 0;
    var a = n.lanes | n.childLanes;
    if (_p(e, a), e === He && (Fe = He = null, Ge = 0), !(n.subtreeFlags & 2064) && !(n.flags & 2064) || ra || (ra = true, qf(Ea, function() {
      return Nr(), null;
    })), a = (n.flags & 15990) !== 0, n.subtreeFlags & 15990 || a) {
      a = Rt.transition, Rt.transition = null;
      var o = de;
      de = 1;
      var i = ae;
      ae |= 4, Mi.current = null, $h(e, n), Wf(n, e), mh(ws), _a = !!ys, ws = ys = null, e.current = n, Bh(n), xp(), ae = i, de = o, Rt.transition = a;
    } else e.current = n;
    if (ra && (ra = false, hn = e, Va = l), a = e.pendingLanes, a === 0 && (kn = null), Sp(n.stateNode), mt(e, Le()), t !== null) for (r = e.onRecoverableError, n = 0; n < t.length; n++) l = t[n], r(l.value, {
      componentStack: l.stack,
      digest: l.digest
    });
    if (Ba) throw Ba = false, e = Fs, Fs = null, e;
    return Va & 1 && e.tag !== 0 && Nr(), a = e.pendingLanes, a & 1 ? e === $s ? sl++ : (sl = 0, $s = e) : sl = 0, Tn(), null;
  }
  function Nr() {
    if (hn !== null) {
      var e = bd(Va), t = Rt.transition, n = de;
      try {
        if (Rt.transition = null, de = 16 > e ? 16 : e, hn === null) var r = false;
        else {
          if (e = hn, hn = null, Va = 0, ae & 6) throw Error(P(331));
          var l = ae;
          for (ae |= 4, O = e.current; O !== null; ) {
            var a = O, o = a.child;
            if (O.flags & 16) {
              var i = a.deletions;
              if (i !== null) {
                for (var s = 0; s < i.length; s++) {
                  var c = i[s];
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
                      var p = m.sibling, x = m.return;
                      if ($f(m), m === c) {
                        O = null;
                        break;
                      }
                      if (p !== null) {
                        p.return = x, O = p;
                        break;
                      }
                      O = x;
                    }
                  }
                }
                var w = a.alternate;
                if (w !== null) {
                  var k = w.child;
                  if (k !== null) {
                    w.child = null;
                    do {
                      var R = k.sibling;
                      k.sibling = null, k = R;
                    } while (k !== null);
                  }
                }
                O = a;
              }
            }
            if (a.subtreeFlags & 2064 && o !== null) o.return = a, O = o;
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
            o = O;
            var v = o.child;
            if (o.subtreeFlags & 2064 && v !== null) v.return = o, O = v;
            else e: for (o = f; O !== null; ) {
              if (i = O, i.flags & 2048) try {
                switch (i.tag) {
                  case 0:
                  case 11:
                  case 15:
                    ao(9, i);
                }
              } catch (_) {
                De(i, i.return, _);
              }
              if (i === o) {
                O = null;
                break e;
              }
              var E = i.sibling;
              if (E !== null) {
                E.return = i.return, O = E;
                break e;
              }
              O = i.return;
            }
          }
          if (ae = l, Tn(), Ht && typeof Ht.onPostCommitFiberRoot == "function") try {
            Ht.onPostCommitFiberRoot(Ja, e);
          } catch {
          }
          r = true;
        }
        return r;
      } finally {
        de = n, Rt.transition = t;
      }
    }
    return false;
  }
  function cc(e, t, n) {
    t = br(n, t), t = Tf(e, t, 1), e = Sn(e, t, 1), t = rt(), e !== null && (bl(e, 1, t), mt(e, t));
  }
  function De(e, t, n) {
    if (e.tag === 3) cc(e, e, n);
    else for (; t !== null; ) {
      if (t.tag === 3) {
        cc(t, e, n);
        break;
      } else if (t.tag === 1) {
        var r = t.stateNode;
        if (typeof t.type.getDerivedStateFromError == "function" || typeof r.componentDidCatch == "function" && (kn === null || !kn.has(r))) {
          e = br(n, e), e = Pf(t, e, 1), t = Sn(t, e, 1), e = rt(), t !== null && (bl(t, 1, e), mt(t, e));
          break;
        }
      }
      t = t.return;
    }
  }
  function Yh(e, t, n) {
    var r = e.pingCache;
    r !== null && r.delete(t), t = rt(), e.pingedLanes |= e.suspendedLanes & n, He === e && (Ge & n) === n && (Be === 4 || Be === 3 && (Ge & 130023424) === Ge && 500 > Le() - Li ? Bn(e, 0) : Di |= n), mt(e, t);
  }
  function Jf(e, t) {
    t === 0 && (e.mode & 1 ? (t = Kl, Kl <<= 1, !(Kl & 130023424) && (Kl = 4194304)) : t = 1);
    var n = rt();
    e = nn(e, t), e !== null && (bl(e, t, n), mt(e, n));
  }
  function Gh(e) {
    var t = e.memoizedState, n = 0;
    t !== null && (n = t.retryLane), Jf(e, n);
  }
  function Xh(e, t) {
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
    r !== null && r.delete(t), Jf(e, n);
  }
  var Zf;
  Zf = function(e, t, n) {
    if (e !== null) if (e.memoizedProps !== t.pendingProps || dt.current) ct = true;
    else {
      if (!(e.lanes & n) && !(t.flags & 128)) return ct = false, Ih(e, t, n);
      ct = !!(e.flags & 131072);
    }
    else ct = false, Ne && t.flags & 1048576 && tf(t, La, t.index);
    switch (t.lanes = 0, t.tag) {
      case 2:
        var r = t.type;
        ga(e, t), e = t.pendingProps;
        var l = Er(t, et.current);
        kr(t, n), l = _i(null, t, r, e, l, n);
        var a = Ri();
        return t.flags |= 1, typeof l == "object" && l !== null && typeof l.render == "function" && l.$$typeof === void 0 ? (t.tag = 1, t.memoizedState = null, t.updateQueue = null, ft(r) ? (a = true, Ma(t)) : a = false, t.memoizedState = l.state !== null && l.state !== void 0 ? l.state : null, ki(t), l.updater = lo, t.stateNode = l, l._reactInternals = t, bs(t, r, e, n), t = Ms(null, t, r, true, a, n)) : (t.tag = 0, Ne && a && hi(t), nt(null, t, l, n), t = t.child), t;
      case 16:
        r = t.elementType;
        e: {
          switch (ga(e, t), e = t.pendingProps, l = r._init, r = l(r._payload), t.type = r, l = t.tag = Zh(r), e = Dt(r, e), l) {
            case 0:
              t = Ps(null, t, r, e, n);
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
          throw Error(P(306, r, ""));
        }
        return t;
      case 0:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), Ps(e, t, r, l, n);
      case 1:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), ec(e, t, r, l, n);
      case 3:
        e: {
          if (Of(t), e === null) throw Error(P(387));
          r = t.pendingProps, a = t.memoizedState, l = a.element, sf(e, t), Ia(t, r, null, n);
          var o = t.memoizedState;
          if (r = o.element, a.isDehydrated) if (a = {
            element: r,
            isDehydrated: false,
            cache: o.cache,
            pendingSuspenseBoundaries: o.pendingSuspenseBoundaries,
            transitions: o.transitions
          }, t.updateQueue.baseState = a, t.memoizedState = a, t.flags & 256) {
            l = br(Error(P(423)), t), t = tc(e, t, r, n, l);
            break e;
          } else if (r !== l) {
            l = br(Error(P(424)), t), t = tc(e, t, r, n, l);
            break e;
          } else for (wt = wn(t.stateNode.containerInfo.firstChild), St = t, Ne = true, At = null, n = af(t, null, r, n), t.child = n; n; ) n.flags = n.flags & -3 | 4096, n = n.sibling;
          else {
            if (Cr(), r === l) {
              t = rn(e, t, n);
              break e;
            }
            nt(e, t, r, n);
          }
          t = t.child;
        }
        return t;
      case 5:
        return uf(t), e === null && Cs(t), r = t.type, l = t.pendingProps, a = e !== null ? e.memoizedProps : null, o = l.children, Ss(r, l) ? o = null : a !== null && Ss(r, a) && (t.flags |= 32), Lf(e, t), nt(e, t, o, n), t.child;
      case 6:
        return e === null && Cs(t), null;
      case 13:
        return Af(e, t, n);
      case 4:
        return Ni(t, t.stateNode.containerInfo), r = t.pendingProps, e === null ? t.child = _r(t, null, r, n) : nt(e, t, r, n), t.child;
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
          if (r = t.type._context, l = t.pendingProps, a = t.memoizedProps, o = l.value, xe(Oa, r._currentValue), r._currentValue = o, a !== null) if (Ut(a.value, o)) {
            if (a.children === l.children && !dt.current) {
              t = rn(e, t, n);
              break e;
            }
          } else for (a = t.child, a !== null && (a.return = t); a !== null; ) {
            var i = a.dependencies;
            if (i !== null) {
              o = a.child;
              for (var s = i.firstContext; s !== null; ) {
                if (s.context === r) {
                  if (a.tag === 1) {
                    s = qt(-1, n & -n), s.tag = 2;
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
              if (o = a.return, o === null) throw Error(P(341));
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
        return l = t.type, r = t.pendingProps.children, kr(t, n), l = bt(l), r = r(l), t.flags |= 1, nt(e, t, r, n), t.child;
      case 14:
        return r = t.type, l = Dt(r, t.pendingProps), l = Dt(r.type, l), qu(e, t, r, l, n);
      case 15:
        return Mf(e, t, t.type, t.pendingProps, n);
      case 17:
        return r = t.type, l = t.pendingProps, l = t.elementType === r ? l : Dt(r, l), ga(e, t), t.tag = 1, ft(r) ? (e = true, Ma(t)) : e = false, kr(t, n), bf(t, r, l), bs(t, r, l, n), Ms(null, t, r, true, e, n);
      case 19:
        return If(e, t, n);
      case 22:
        return Df(e, t, n);
    }
    throw Error(P(156, t.tag));
  };
  function qf(e, t) {
    return Ed(e, t);
  }
  function Jh(e, t, n, r) {
    this.tag = e, this.key = n, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = r, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function _t(e, t, n, r) {
    return new Jh(e, t, n, r);
  }
  function zi(e) {
    return e = e.prototype, !(!e || !e.isReactComponent);
  }
  function Zh(e) {
    if (typeof e == "function") return zi(e) ? 1 : 0;
    if (e != null) {
      if (e = e.$$typeof, e === ri) return 11;
      if (e === li) return 14;
    }
    return 2;
  }
  function jn(e, t) {
    var n = e.alternate;
    return n === null ? (n = _t(e.tag, t, e.key, e.mode), n.elementType = e.elementType, n.type = e.type, n.stateNode = e.stateNode, n.alternate = e, e.alternate = n) : (n.pendingProps = t, n.type = e.type, n.flags = 0, n.subtreeFlags = 0, n.deletions = null), n.flags = e.flags & 14680064, n.childLanes = e.childLanes, n.lanes = e.lanes, n.child = e.child, n.memoizedProps = e.memoizedProps, n.memoizedState = e.memoizedState, n.updateQueue = e.updateQueue, t = e.dependencies, n.dependencies = t === null ? null : {
      lanes: t.lanes,
      firstContext: t.firstContext
    }, n.sibling = e.sibling, n.index = e.index, n.ref = e.ref, n;
  }
  function ya(e, t, n, r, l, a) {
    var o = 2;
    if (r = e, typeof e == "function") zi(e) && (o = 1);
    else if (typeof e == "string") o = 5;
    else e: switch (e) {
      case sr:
        return Vn(n.children, l, a, t);
      case ni:
        o = 8, l |= 8;
        break;
      case qo:
        return e = _t(12, n, t, l | 2), e.elementType = qo, e.lanes = a, e;
      case es:
        return e = _t(13, n, t, l), e.elementType = es, e.lanes = a, e;
      case ts:
        return e = _t(19, n, t, l), e.elementType = ts, e.lanes = a, e;
      case id:
        return so(n, l, a, t);
      default:
        if (typeof e == "object" && e !== null) switch (e.$$typeof) {
          case od:
            o = 10;
            break e;
          case sd:
            o = 9;
            break e;
          case ri:
            o = 11;
            break e;
          case li:
            o = 14;
            break e;
          case cn:
            o = 16, r = null;
            break e;
        }
        throw Error(P(130, e == null ? e : typeof e, ""));
    }
    return t = _t(o, n, t, l), t.elementType = e, t.type = r, t.lanes = a, t;
  }
  function Vn(e, t, n, r) {
    return e = _t(7, e, r, t), e.lanes = n, e;
  }
  function so(e, t, n, r) {
    return e = _t(22, e, r, t), e.elementType = id, e.lanes = n, e.stateNode = {
      isHidden: false
    }, e;
  }
  function Wo(e, t, n) {
    return e = _t(6, e, null, t), e.lanes = n, e;
  }
  function Ho(e, t, n) {
    return t = _t(4, e.children !== null ? e.children : [], e.key, t), t.lanes = n, t.stateNode = {
      containerInfo: e.containerInfo,
      pendingChildren: null,
      implementation: e.implementation
    }, t;
  }
  function qh(e, t, n, r, l) {
    this.tag = t, this.containerInfo = e, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = Eo(0), this.expirationTimes = Eo(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = Eo(0), this.identifierPrefix = r, this.onRecoverableError = l, this.mutableSourceEagerHydrationData = null;
  }
  function Ui(e, t, n, r, l, a, o, i, s) {
    return e = new qh(e, t, n, i, s), t === 1 ? (t = 1, a === true && (t |= 8)) : t = 0, a = _t(3, null, null, t), e.current = a, a.stateNode = e, a.memoizedState = {
      element: r,
      isDehydrated: n,
      cache: null,
      transitions: null,
      pendingSuspenseBoundaries: null
    }, ki(a), e;
  }
  function eg(e, t, n) {
    var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: or,
      key: r == null ? null : "" + r,
      children: e,
      containerInfo: t,
      implementation: n
    };
  }
  function em(e) {
    if (!e) return _n;
    e = e._reactInternals;
    e: {
      if (Jn(e) !== e || e.tag !== 1) throw Error(P(170));
      var t = e;
      do {
        switch (t.tag) {
          case 3:
            t = t.stateNode.context;
            break e;
          case 1:
            if (ft(t.type)) {
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
      if (ft(n)) return qd(e, n, t);
    }
    return t;
  }
  function tm(e, t, n, r, l, a, o, i, s) {
    return e = Ui(n, r, true, e, l, a, o, i, s), e.context = em(null), n = e.current, r = rt(), l = Nn(n), a = qt(r, l), a.callback = t ?? null, Sn(n, a, l), e.current.lanes = l, bl(e, l, r), mt(e, r), e;
  }
  function io(e, t, n, r) {
    var l = t.current, a = rt(), o = Nn(l);
    return n = em(n), t.context === null ? t.context = n : t.pendingContext = n, t = qt(a, o), t.payload = {
      element: e
    }, r = r === void 0 ? null : r, r !== null && (t.callback = r), e = Sn(l, t, o), e !== null && (zt(e, l, o, a), ma(e, l, o)), o;
  }
  function Ha(e) {
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
  function Fi(e, t) {
    dc(e, t), (e = e.alternate) && dc(e, t);
  }
  function tg() {
    return null;
  }
  var nm = typeof reportError == "function" ? reportError : function(e) {
    console.error(e);
  };
  function $i(e) {
    this._internalRoot = e;
  }
  uo.prototype.render = $i.prototype.render = function(e) {
    var t = this._internalRoot;
    if (t === null) throw Error(P(409));
    io(e, t, null, null);
  };
  uo.prototype.unmount = $i.prototype.unmount = function() {
    var e = this._internalRoot;
    if (e !== null) {
      this._internalRoot = null;
      var t = e.containerInfo;
      Yn(function() {
        io(null, e, null, null);
      }), t[tn] = null;
    }
  };
  function uo(e) {
    this._internalRoot = e;
  }
  uo.prototype.unstable_scheduleHydration = function(e) {
    if (e) {
      var t = Md();
      e = {
        blockedOn: null,
        target: e,
        priority: t
      };
      for (var n = 0; n < fn.length && t !== 0 && t < fn[n].priority; n++) ;
      fn.splice(n, 0, e), n === 0 && Ld(e);
    }
  };
  function Bi(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
  }
  function co(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11 && (e.nodeType !== 8 || e.nodeValue !== " react-mount-point-unstable "));
  }
  function fc() {
  }
  function ng(e, t, n, r, l) {
    if (l) {
      if (typeof r == "function") {
        var a = r;
        r = function() {
          var c = Ha(o);
          a.call(c);
        };
      }
      var o = tm(t, r, e, 0, null, false, false, "", fc);
      return e._reactRootContainer = o, e[tn] = o.current, vl(e.nodeType === 8 ? e.parentNode : e), Yn(), o;
    }
    for (; l = e.lastChild; ) e.removeChild(l);
    if (typeof r == "function") {
      var i = r;
      r = function() {
        var c = Ha(s);
        i.call(c);
      };
    }
    var s = Ui(e, 0, false, null, null, false, false, "", fc);
    return e._reactRootContainer = s, e[tn] = s.current, vl(e.nodeType === 8 ? e.parentNode : e), Yn(function() {
      io(t, s, n, r);
    }), s;
  }
  function fo(e, t, n, r, l) {
    var a = n._reactRootContainer;
    if (a) {
      var o = a;
      if (typeof l == "function") {
        var i = l;
        l = function() {
          var s = Ha(o);
          i.call(s);
        };
      }
      io(t, o, e, l);
    } else o = ng(n, t, e, l, r);
    return Ha(o);
  }
  Td = function(e) {
    switch (e.tag) {
      case 3:
        var t = e.stateNode;
        if (t.current.memoizedState.isDehydrated) {
          var n = Xr(t.pendingLanes);
          n !== 0 && (si(t, n | 1), mt(t, Le()), !(ae & 6) && (Tr = Le() + 500, Tn()));
        }
        break;
      case 13:
        Yn(function() {
          var r = nn(e, 1);
          if (r !== null) {
            var l = rt();
            zt(r, e, 1, l);
          }
        }), Fi(e, 1);
    }
  };
  ii = function(e) {
    if (e.tag === 13) {
      var t = nn(e, 134217728);
      if (t !== null) {
        var n = rt();
        zt(t, e, 134217728, n);
      }
      Fi(e, 134217728);
    }
  };
  Pd = function(e) {
    if (e.tag === 13) {
      var t = Nn(e), n = nn(e, t);
      if (n !== null) {
        var r = rt();
        zt(n, e, t, r);
      }
      Fi(e, t);
    }
  };
  Md = function() {
    return de;
  };
  Dd = function(e, t) {
    var n = de;
    try {
      return de = e, t();
    } finally {
      de = n;
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
              var l = to(r);
              if (!l) throw Error(P(90));
              cd(r), ls(r, l);
            }
          }
        }
        break;
      case "textarea":
        fd(e, n);
        break;
      case "select":
        t = n.value, t != null && xr(e, !!n.multiple, t, false);
    }
  };
  yd = Oi;
  wd = Yn;
  var rg = {
    usingClientEntryPoint: false,
    Events: [
      Pl,
      dr,
      to,
      vd,
      xd,
      Oi
    ]
  }, Wr = {
    findFiberByHostInstance: zn,
    bundleType: 0,
    version: "18.3.1",
    rendererPackageName: "react-dom"
  }, lg = {
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
    currentDispatcherRef: ln.ReactCurrentDispatcher,
    findHostInstanceByFiber: function(e) {
      return e = Nd(e), e === null ? null : e.stateNode;
    },
    findFiberByHostInstance: Wr.findFiberByHostInstance || tg,
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
      Ja = la.inject(lg), Ht = la;
    } catch {
    }
  }
  Nt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = rg;
  Nt.createPortal = function(e, t) {
    var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!Bi(t)) throw Error(P(200));
    return eg(e, t, null, n);
  };
  Nt.createRoot = function(e, t) {
    if (!Bi(e)) throw Error(P(299));
    var n = false, r = "", l = nm;
    return t != null && (t.unstable_strictMode === true && (n = true), t.identifierPrefix !== void 0 && (r = t.identifierPrefix), t.onRecoverableError !== void 0 && (l = t.onRecoverableError)), t = Ui(e, 1, false, null, null, n, false, r, l), e[tn] = t.current, vl(e.nodeType === 8 ? e.parentNode : e), new $i(t);
  };
  Nt.findDOMNode = function(e) {
    if (e == null) return null;
    if (e.nodeType === 1) return e;
    var t = e._reactInternals;
    if (t === void 0) throw typeof e.render == "function" ? Error(P(188)) : (e = Object.keys(e).join(","), Error(P(268, e)));
    return e = Nd(t), e = e === null ? null : e.stateNode, e;
  };
  Nt.flushSync = function(e) {
    return Yn(e);
  };
  Nt.hydrate = function(e, t, n) {
    if (!co(t)) throw Error(P(200));
    return fo(null, e, t, true, n);
  };
  Nt.hydrateRoot = function(e, t, n) {
    if (!Bi(e)) throw Error(P(405));
    var r = n != null && n.hydratedSources || null, l = false, a = "", o = nm;
    if (n != null && (n.unstable_strictMode === true && (l = true), n.identifierPrefix !== void 0 && (a = n.identifierPrefix), n.onRecoverableError !== void 0 && (o = n.onRecoverableError)), t = tm(t, null, e, 1, n ?? null, l, false, a, o), e[tn] = t.current, vl(e), r) for (e = 0; e < r.length; e++) n = r[e], l = n._getVersion, l = l(n._source), t.mutableSourceEagerHydrationData == null ? t.mutableSourceEagerHydrationData = [
      n,
      l
    ] : t.mutableSourceEagerHydrationData.push(n, l);
    return new uo(t);
  };
  Nt.render = function(e, t, n) {
    if (!co(t)) throw Error(P(200));
    return fo(null, e, t, false, n);
  };
  Nt.unmountComponentAtNode = function(e) {
    if (!co(e)) throw Error(P(40));
    return e._reactRootContainer ? (Yn(function() {
      fo(null, null, e, false, function() {
        e._reactRootContainer = null, e[tn] = null;
      });
    }), true) : false;
  };
  Nt.unstable_batchedUpdates = Oi;
  Nt.unstable_renderSubtreeIntoContainer = function(e, t, n, r) {
    if (!co(n)) throw Error(P(200));
    if (e == null || e._reactInternals === void 0) throw Error(P(38));
    return fo(e, t, n, false, r);
  };
  Nt.version = "18.3.1-next-f1338f8080-20240426";
  function rm() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) try {
      __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(rm);
    } catch (e) {
      console.error(e);
    }
  }
  rm(), nd.exports = Nt;
  var Vi = nd.exports;
  const ag = Wc(Vi), og = Vc({
    __proto__: null,
    default: ag
  }, [
    Vi
  ]);
  var mc = Vi;
  Jo.createRoot = mc.createRoot, Jo.hydrateRoot = mc.hydrateRoot;
  function ke() {
    return ke = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, ke.apply(this, arguments);
  }
  var Ue;
  (function(e) {
    e.Pop = "POP", e.Push = "PUSH", e.Replace = "REPLACE";
  })(Ue || (Ue = {}));
  const pc = "popstate";
  function sg(e) {
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
    return ug(t, n, null, e);
  }
  function te(e, t) {
    if (e === false || e === null || typeof e > "u") throw new Error(t);
  }
  function Gn(e, t) {
    if (!e) {
      typeof console < "u" && console.warn(t);
      try {
        throw new Error(t);
      } catch {
      }
    }
  }
  function ig() {
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
    return n === void 0 && (n = null), ke({
      pathname: typeof e == "string" ? e : e.pathname,
      search: "",
      hash: ""
    }, typeof t == "string" ? Pn(t) : t, {
      state: n,
      key: t && t.key || r || ig()
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
  function ug(e, t, n, r) {
    r === void 0 && (r = {});
    let { window: l = document.defaultView, v5Compat: a = false } = r, o = l.history, i = Ue.Pop, s = null, c = m();
    c == null && (c = 0, o.replaceState(ke({}, o.state, {
      idx: c
    }), ""));
    function m() {
      return (o.state || {
        idx: null
      }).idx;
    }
    function d() {
      i = Ue.Pop;
      let R = m(), h = R == null ? null : R - c;
      c = R, s && s({
        action: i,
        location: k.location,
        delta: h
      });
    }
    function p(R, h) {
      i = Ue.Push;
      let f = Cl(k.location, R, h);
      c = m() + 1;
      let v = hc(f, c), E = k.createHref(f);
      try {
        o.pushState(v, "", E);
      } catch (_) {
        if (_ instanceof DOMException && _.name === "DataCloneError") throw _;
        l.location.assign(E);
      }
      a && s && s({
        action: i,
        location: k.location,
        delta: 1
      });
    }
    function x(R, h) {
      i = Ue.Replace;
      let f = Cl(k.location, R, h);
      c = m();
      let v = hc(f, c), E = k.createHref(f);
      o.replaceState(v, "", E), a && s && s({
        action: i,
        location: k.location,
        delta: 0
      });
    }
    function w(R) {
      let h = l.location.origin !== "null" ? l.location.origin : l.location.href, f = typeof R == "string" ? R : Dl(R);
      return f = f.replace(/ $/, "%20"), te(h, "No window.location.(origin|href) available to create URL for href: " + f), new URL(f, h);
    }
    let k = {
      get action() {
        return i;
      },
      get location() {
        return e(l, o);
      },
      listen(R) {
        if (s) throw new Error("A history only accepts one active listener");
        return l.addEventListener(pc, d), s = R, () => {
          l.removeEventListener(pc, d), s = null;
        };
      },
      createHref(R) {
        return t(l, R);
      },
      createURL: w,
      encodeLocation(R) {
        let h = w(R);
        return {
          pathname: h.pathname,
          search: h.search,
          hash: h.hash
        };
      },
      push: p,
      replace: x,
      go(R) {
        return o.go(R);
      }
    };
    return k;
  }
  var ue;
  (function(e) {
    e.data = "data", e.deferred = "deferred", e.redirect = "redirect", e.error = "error";
  })(ue || (ue = {}));
  const cg = /* @__PURE__ */ new Set([
    "lazy",
    "caseSensitive",
    "path",
    "id",
    "index",
    "children"
  ]);
  function dg(e) {
    return e.index === true;
  }
  function Qa(e, t, n, r) {
    return n === void 0 && (n = []), r === void 0 && (r = {}), e.map((l, a) => {
      let o = [
        ...n,
        String(a)
      ], i = typeof l.id == "string" ? l.id : o.join("-");
      if (te(l.index !== true || !l.children, "Cannot specify children on an index route"), te(!r[i], 'Found a route id collision on id "' + i + `".  Route id's must be globally unique within Data Router usages`), dg(l)) {
        let s = ke({}, l, t(l), {
          id: i
        });
        return r[i] = s, s;
      } else {
        let s = ke({}, l, t(l), {
          id: i,
          children: void 0
        });
        return r[i] = s, l.children && (s.children = Qa(l.children, t, o, r)), s;
      }
    });
  }
  function An(e, t, n) {
    return n === void 0 && (n = "/"), wa(e, t, n, false);
  }
  function wa(e, t, n, r) {
    let l = typeof t == "string" ? Pn(t) : t, a = Ll(l.pathname || "/", n);
    if (a == null) return null;
    let o = lm(e);
    mg(o);
    let i = null;
    for (let s = 0; i == null && s < o.length; ++s) {
      let c = jg(a);
      i = kg(o[s], c, r);
    }
    return i;
  }
  function fg(e, t) {
    let { route: n, pathname: r, params: l } = e;
    return {
      id: n.id,
      pathname: r,
      params: l,
      data: t[n.id],
      handle: n.handle
    };
  }
  function lm(e, t, n, r) {
    t === void 0 && (t = []), n === void 0 && (n = []), r === void 0 && (r = "");
    let l = (a, o, i) => {
      let s = {
        relativePath: i === void 0 ? a.path || "" : i,
        caseSensitive: a.caseSensitive === true,
        childrenIndex: o,
        route: a
      };
      s.relativePath.startsWith("/") && (te(s.relativePath.startsWith(r), 'Absolute route path "' + s.relativePath + '" nested under path ' + ('"' + r + '" is not valid. An absolute child route path ') + "must start with the combined path of all its parent routes."), s.relativePath = s.relativePath.slice(r.length));
      let c = En([
        r,
        s.relativePath
      ]), m = n.concat(s);
      a.children && a.children.length > 0 && (te(a.index !== true, "Index routes must not have child routes. Please remove " + ('all child routes from route path "' + c + '".')), lm(a.children, t, m, c)), !(a.path == null && !a.index) && t.push({
        path: c,
        score: wg(c, a.index),
        routesMeta: m
      });
    };
    return e.forEach((a, o) => {
      var i;
      if (a.path === "" || !((i = a.path) != null && i.includes("?"))) l(a, o);
      else for (let s of am(a.path)) l(a, o, s);
    }), t;
  }
  function am(e) {
    let t = e.split("/");
    if (t.length === 0) return [];
    let [n, ...r] = t, l = n.endsWith("?"), a = n.replace(/\?$/, "");
    if (r.length === 0) return l ? [
      a,
      ""
    ] : [
      a
    ];
    let o = am(r.join("/")), i = [];
    return i.push(...o.map((s) => s === "" ? a : [
      a,
      s
    ].join("/"))), l && i.push(...o), i.map((s) => e.startsWith("/") && s === "" ? "/" : s);
  }
  function mg(e) {
    e.sort((t, n) => t.score !== n.score ? n.score - t.score : Sg(t.routesMeta.map((r) => r.childrenIndex), n.routesMeta.map((r) => r.childrenIndex)));
  }
  const pg = /^:[\w-]+$/, hg = 3, gg = 2, vg = 1, xg = 10, yg = -2, gc = (e) => e === "*";
  function wg(e, t) {
    let n = e.split("/"), r = n.length;
    return n.some(gc) && (r += yg), t && (r += gg), n.filter((l) => !gc(l)).reduce((l, a) => l + (pg.test(a) ? hg : a === "" ? vg : xg), r);
  }
  function Sg(e, t) {
    return e.length === t.length && e.slice(0, -1).every((r, l) => r === t[l]) ? e[e.length - 1] - t[t.length - 1] : 0;
  }
  function kg(e, t, n) {
    n === void 0 && (n = false);
    let { routesMeta: r } = e, l = {}, a = "/", o = [];
    for (let i = 0; i < r.length; ++i) {
      let s = r[i], c = i === r.length - 1, m = a === "/" ? t : t.slice(a.length) || "/", d = vc({
        path: s.relativePath,
        caseSensitive: s.caseSensitive,
        end: c
      }, m), p = s.route;
      if (!d && c && n && !r[r.length - 1].route.index && (d = vc({
        path: s.relativePath,
        caseSensitive: s.caseSensitive,
        end: false
      }, m)), !d) return null;
      Object.assign(l, d.params), o.push({
        params: l,
        pathname: En([
          a,
          d.pathname
        ]),
        pathnameBase: Rg(En([
          a,
          d.pathnameBase
        ])),
        route: p
      }), d.pathnameBase !== "/" && (a = En([
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
    let [n, r] = Ng(e.path, e.caseSensitive, e.end), l = t.match(n);
    if (!l) return null;
    let a = l[0], o = a.replace(/(.)\/+$/, "$1"), i = l.slice(1);
    return {
      params: r.reduce((c, m, d) => {
        let { paramName: p, isOptional: x } = m;
        if (p === "*") {
          let k = i[d] || "";
          o = a.slice(0, a.length - k.length).replace(/(.)\/+$/, "$1");
        }
        const w = i[d];
        return x && !w ? c[p] = void 0 : c[p] = (w || "").replace(/%2F/g, "/"), c;
      }, {}),
      pathname: a,
      pathnameBase: o,
      pattern: e
    };
  }
  function Ng(e, t, n) {
    t === void 0 && (t = false), n === void 0 && (n = true), Gn(e === "*" || !e.endsWith("*") || e.endsWith("/*"), 'Route path "' + e + '" will be treated as if it were ' + ('"' + e.replace(/\*$/, "/*") + '" because the `*` character must ') + "always follow a `/` in the pattern. To get rid of this warning, " + ('please change the route path to "' + e.replace(/\*$/, "/*") + '".'));
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
  function jg(e) {
    try {
      return e.split("/").map((t) => decodeURIComponent(t).replace(/\//g, "%2F")).join("/");
    } catch (t) {
      return Gn(false, 'The URL path "' + e + '" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent ' + ("encoding (" + t + ").")), e;
    }
  }
  function Ll(e, t) {
    if (t === "/") return e;
    if (!e.toLowerCase().startsWith(t.toLowerCase())) return null;
    let n = t.endsWith("/") ? t.length - 1 : t.length, r = e.charAt(n);
    return r && r !== "/" ? null : e.slice(n) || "/";
  }
  const Eg = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Cg = (e) => Eg.test(e);
  function _g(e, t) {
    t === void 0 && (t = "/");
    let { pathname: n, search: r = "", hash: l = "" } = typeof e == "string" ? Pn(e) : e, a;
    if (n) if (Cg(n)) a = n;
    else {
      if (n.includes("//")) {
        let o = n;
        n = n.replace(/\/\/+/g, "/"), Gn(false, "Pathnames cannot have embedded double slashes - normalizing " + (o + " -> " + n));
      }
      n.startsWith("/") ? a = xc(n.substring(1), "/") : a = xc(n, t);
    }
    else a = t;
    return {
      pathname: a,
      search: bg(r),
      hash: Tg(l)
    };
  }
  function xc(e, t) {
    let n = t.replace(/\/+$/, "").split("/");
    return e.split("/").forEach((l) => {
      l === ".." ? n.length > 1 && n.pop() : l !== "." && n.push(l);
    }), n.length > 1 ? n.join("/") : "/";
  }
  function Qo(e, t, n, r) {
    return "Cannot include a '" + e + "' character in a manually specified " + ("`to." + t + "` field [" + JSON.stringify(r) + "].  Please separate it out to the ") + ("`to." + n + "` field. Alternatively you may provide the full path as ") + 'a string in <Link to="..."> and the router will parse it for you.';
  }
  function om(e) {
    return e.filter((t, n) => n === 0 || t.route.path && t.route.path.length > 0);
  }
  function sm(e, t) {
    let n = om(e);
    return t ? n.map((r, l) => l === n.length - 1 ? r.pathname : r.pathnameBase) : n.map((r) => r.pathnameBase);
  }
  function im(e, t, n, r) {
    r === void 0 && (r = false);
    let l;
    typeof e == "string" ? l = Pn(e) : (l = ke({}, e), te(!l.pathname || !l.pathname.includes("?"), Qo("?", "pathname", "search", l)), te(!l.pathname || !l.pathname.includes("#"), Qo("#", "pathname", "hash", l)), te(!l.search || !l.search.includes("#"), Qo("#", "search", "hash", l)));
    let a = e === "" || l.pathname === "", o = a ? "/" : l.pathname, i;
    if (o == null) i = n;
    else {
      let d = t.length - 1;
      if (!r && o.startsWith("..")) {
        let p = o.split("/");
        for (; p[0] === ".."; ) p.shift(), d -= 1;
        l.pathname = p.join("/");
      }
      i = d >= 0 ? t[d] : "/";
    }
    let s = _g(l, i), c = o && o !== "/" && o.endsWith("/"), m = (a || o === ".") && n.endsWith("/");
    return !s.pathname.endsWith("/") && (c || m) && (s.pathname += "/"), s;
  }
  const En = (e) => e.join("/").replace(/\/\/+/g, "/"), Rg = (e) => e.replace(/\/+$/, "").replace(/^\/*/, "/"), bg = (e) => !e || e === "?" ? "" : e.startsWith("?") ? e : "?" + e, Tg = (e) => !e || e === "#" ? "" : e.startsWith("#") ? e : "#" + e;
  class Ka {
    constructor(t, n, r, l) {
      l === void 0 && (l = false), this.status = t, this.statusText = n || "", this.internal = l, r instanceof Error ? (this.data = r.toString(), this.error = r) : this.data = r;
    }
  }
  function _l(e) {
    return e != null && typeof e.status == "number" && typeof e.statusText == "string" && typeof e.internal == "boolean" && "data" in e;
  }
  const um = [
    "post",
    "put",
    "patch",
    "delete"
  ], Pg = new Set(um), Mg = [
    "get",
    ...um
  ], Dg = new Set(Mg), Lg = /* @__PURE__ */ new Set([
    301,
    302,
    303,
    307,
    308
  ]), Og = /* @__PURE__ */ new Set([
    307,
    308
  ]), Ko = {
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
  }, Hr = {
    state: "unblocked",
    proceed: void 0,
    reset: void 0,
    location: void 0
  }, Wi = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i, Ig = (e) => ({
    hasErrorBoundary: !!e.hasErrorBoundary
  }), cm = "remix-router-transitions";
  function zg(e) {
    const t = e.window ? e.window : typeof window < "u" ? window : void 0, n = typeof t < "u" && typeof t.document < "u" && typeof t.document.createElement < "u", r = !n;
    te(e.routes.length > 0, "You must provide a non-empty routes array to createRouter");
    let l;
    if (e.mapRouteProperties) l = e.mapRouteProperties;
    else if (e.detectErrorBoundary) {
      let y = e.detectErrorBoundary;
      l = (N) => ({
        hasErrorBoundary: y(N)
      });
    } else l = Ig;
    let a = {}, o = Qa(e.routes, l, void 0, a), i, s = e.basename || "/", c = e.dataStrategy || Bg, m = e.patchRoutesOnNavigation, d = ke({
      v7_fetcherPersist: false,
      v7_normalizeFormMethod: false,
      v7_partialHydration: false,
      v7_prependBasename: false,
      v7_relativeSplatPath: false,
      v7_skipActionErrorRevalidation: false
    }, e.future), p = null, x = /* @__PURE__ */ new Set(), w = null, k = null, R = null, h = e.hydrationData != null, f = An(o, e.history.location, s), v = false, E = null;
    if (f == null && !m) {
      let y = it(404, {
        pathname: e.history.location.pathname
      }), { matches: N, route: C } = bc(o);
      f = N, E = {
        [C.id]: y
      };
    }
    f && !e.hydrationData && zl(f, o, e.history.location.pathname).active && (f = null);
    let _;
    if (f) if (f.some((y) => y.route.lazy)) _ = false;
    else if (!f.some((y) => y.route.loader)) _ = true;
    else if (d.v7_partialHydration) {
      let y = e.hydrationData ? e.hydrationData.loaderData : null, N = e.hydrationData ? e.hydrationData.errors : null;
      if (N) {
        let C = f.findIndex((T) => N[T.route.id] !== void 0);
        _ = f.slice(0, C + 1).every((T) => !Hs(T.route, y, N));
      } else _ = f.every((C) => !Hs(C.route, y, N));
    } else _ = e.hydrationData != null;
    else if (_ = false, f = [], d.v7_partialHydration) {
      let y = zl(null, o, e.history.location.pathname);
      y.active && y.matches && (v = true, f = y.matches);
    }
    let b, S = {
      historyAction: e.history.action,
      location: e.history.location,
      matches: f,
      initialized: _,
      navigation: Ko,
      restoreScrollPosition: e.hydrationData != null ? false : null,
      preventScrollReset: false,
      revalidation: "idle",
      loaderData: e.hydrationData && e.hydrationData.loaderData || {},
      actionData: e.hydrationData && e.hydrationData.actionData || null,
      errors: e.hydrationData && e.hydrationData.errors || E,
      fetchers: /* @__PURE__ */ new Map(),
      blockers: /* @__PURE__ */ new Map()
    }, j = Ue.Pop, z = false, D, H = false, K = /* @__PURE__ */ new Map(), se = null, le = false, je = false, Qe = [], pt = /* @__PURE__ */ new Set(), M = /* @__PURE__ */ new Map(), V = 0, F = -1, ee = /* @__PURE__ */ new Map(), X = /* @__PURE__ */ new Set(), be = /* @__PURE__ */ new Map(), Ee = /* @__PURE__ */ new Map(), ge = /* @__PURE__ */ new Set(), Se = /* @__PURE__ */ new Map(), J = /* @__PURE__ */ new Map(), Te;
    function me() {
      if (p = e.history.listen((y) => {
        let { action: N, location: C, delta: T } = y;
        if (Te) {
          Te(), Te = void 0;
          return;
        }
        Gn(J.size === 0 || T != null, "You are trying to use a blocker on a POP navigation to a location that was not created by @remix-run/router. This will fail silently in production. This can happen if you are navigating outside the router via `window.history.pushState`/`window.location.hash` instead of using router navigation APIs.  This can also happen if you are using createHashRouter and the user manually changes the URL.");
        let L = tu({
          currentLocation: S.location,
          nextLocation: C,
          historyAction: N
        });
        if (L && T != null) {
          let W = new Promise((Q) => {
            Te = Q;
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
              }), W.then(() => e.history.go(T));
            },
            reset() {
              let Q = new Map(S.blockers);
              Q.set(L, Hr), Y({
                blockers: Q
              });
            }
          });
          return;
        }
        return Oe(N, C);
      }), n) {
        nv(t, K);
        let y = () => rv(t, K);
        t.addEventListener("pagehide", y), se = () => t.removeEventListener("pagehide", y);
      }
      return S.initialized || Oe(Ue.Pop, S.location, {
        initialHydration: true
      }), b;
    }
    function pe() {
      p && p(), se && se(), x.clear(), D && D.abort(), S.fetchers.forEach((y, N) => Al(N)), S.blockers.forEach((y, N) => eu(N));
    }
    function $(y) {
      return x.add(y), () => x.delete(y);
    }
    function Y(y, N) {
      N === void 0 && (N = {}), S = ke({}, S, y);
      let C = [], T = [];
      d.v7_fetcherPersist && S.fetchers.forEach((L, W) => {
        L.state === "idle" && (ge.has(W) ? T.push(W) : C.push(W));
      }), ge.forEach((L) => {
        !S.fetchers.has(L) && !M.has(L) && T.push(L);
      }), [
        ...x
      ].forEach((L) => L(S, {
        deletedFetchers: T,
        viewTransitionOpts: N.viewTransitionOpts,
        flushSync: N.flushSync === true
      })), d.v7_fetcherPersist ? (C.forEach((L) => S.fetchers.delete(L)), T.forEach((L) => Al(L))) : T.forEach((L) => ge.delete(L));
    }
    function Z(y, N, C) {
      var T, L;
      let { flushSync: W } = C === void 0 ? {} : C, Q = S.actionData != null && S.navigation.formMethod != null && Ot(S.navigation.formMethod) && S.navigation.state === "loading" && ((T = y.state) == null ? void 0 : T._isRedirect) !== true, I;
      N.actionData ? Object.keys(N.actionData).length > 0 ? I = N.actionData : I = null : Q ? I = S.actionData : I = null;
      let U = N.loaderData ? _c(S.loaderData, N.loaderData, N.matches || [], N.errors) : S.loaderData, A = S.blockers;
      A.size > 0 && (A = new Map(A), A.forEach((re, Ke) => A.set(Ke, Hr)));
      let B = z === true || S.navigation.formMethod != null && Ot(S.navigation.formMethod) && ((L = y.state) == null ? void 0 : L._isRedirect) !== true;
      i && (o = i, i = void 0), le || j === Ue.Pop || (j === Ue.Push ? e.history.push(y, y.state) : j === Ue.Replace && e.history.replace(y, y.state));
      let G;
      if (j === Ue.Pop) {
        let re = K.get(S.location.pathname);
        re && re.has(y.pathname) ? G = {
          currentLocation: S.location,
          nextLocation: y
        } : K.has(y.pathname) && (G = {
          currentLocation: y,
          nextLocation: S.location
        });
      } else if (H) {
        let re = K.get(S.location.pathname);
        re ? re.add(y.pathname) : (re = /* @__PURE__ */ new Set([
          y.pathname
        ]), K.set(S.location.pathname, re)), G = {
          currentLocation: S.location,
          nextLocation: y
        };
      }
      Y(ke({}, N, {
        actionData: I,
        loaderData: U,
        historyAction: j,
        location: y,
        initialized: true,
        navigation: Ko,
        revalidation: "idle",
        restoreScrollPosition: ru(y, N.matches || S.matches),
        preventScrollReset: B,
        blockers: A
      }), {
        viewTransitionOpts: G,
        flushSync: W === true
      }), j = Ue.Pop, z = false, H = false, le = false, je = false, Qe = [];
    }
    async function fe(y, N) {
      if (typeof y == "number") {
        e.history.go(y);
        return;
      }
      let C = Ws(S.location, S.matches, s, d.v7_prependBasename, y, d.v7_relativeSplatPath, N == null ? void 0 : N.fromRouteId, N == null ? void 0 : N.relative), { path: T, submission: L, error: W } = yc(d.v7_normalizeFormMethod, false, C, N), Q = S.location, I = Cl(S.location, T, N && N.state);
      I = ke({}, I, e.history.encodeLocation(I));
      let U = N && N.replace != null ? N.replace : void 0, A = Ue.Push;
      U === true ? A = Ue.Replace : U === false || L != null && Ot(L.formMethod) && L.formAction === S.location.pathname + S.location.search && (A = Ue.Replace);
      let B = N && "preventScrollReset" in N ? N.preventScrollReset === true : void 0, G = (N && N.flushSync) === true, re = tu({
        currentLocation: Q,
        nextLocation: I,
        historyAction: A
      });
      if (re) {
        Il(re, {
          state: "blocked",
          location: I,
          proceed() {
            Il(re, {
              state: "proceeding",
              proceed: void 0,
              reset: void 0,
              location: I
            }), fe(y, N);
          },
          reset() {
            let Ke = new Map(S.blockers);
            Ke.set(re, Hr), Y({
              blockers: Ke
            });
          }
        });
        return;
      }
      return await Oe(A, I, {
        submission: L,
        pendingError: W,
        preventScrollReset: B,
        replace: N && N.replace,
        enableViewTransition: N && N.viewTransition,
        flushSync: G
      });
    }
    function Ve() {
      if (Ft(), Y({
        revalidation: "loading"
      }), S.navigation.state !== "submitting") {
        if (S.navigation.state === "idle") {
          Oe(S.historyAction, S.location, {
            startUninterruptedRevalidation: true
          });
          return;
        }
        Oe(j || S.historyAction, S.navigation.location, {
          overrideNavigation: S.navigation,
          enableViewTransition: H === true
        });
      }
    }
    async function Oe(y, N, C) {
      D && D.abort(), D = null, j = y, le = (C && C.startUninterruptedRevalidation) === true, Rm(S.location, S.matches), z = (C && C.preventScrollReset) === true, H = (C && C.enableViewTransition) === true;
      let T = i || o, L = C && C.overrideNavigation, W = C != null && C.initialHydration && S.matches && S.matches.length > 0 && !v ? S.matches : An(T, N, s), Q = (C && C.flushSync) === true;
      if (W && S.initialized && !je && Yg(S.location, N) && !(C && C.submission && Ot(C.submission.formMethod))) {
        Z(N, {
          matches: W
        }, {
          flushSync: Q
        });
        return;
      }
      let I = zl(W, T, N.pathname);
      if (I.active && I.matches && (W = I.matches), !W) {
        let { error: ve, notFoundMatches: ie, route: Pe } = vo(N.pathname);
        Z(N, {
          matches: ie,
          loaderData: {},
          errors: {
            [Pe.id]: ve
          }
        }, {
          flushSync: Q
        });
        return;
      }
      D = new AbortController();
      let U = rr(e.history, N, D.signal, C && C.submission), A;
      if (C && C.pendingError) A = [
        In(W).route.id,
        {
          type: ue.error,
          error: C.pendingError
        }
      ];
      else if (C && C.submission && Ot(C.submission.formMethod)) {
        let ve = await tt(U, N, C.submission, W, I.active, {
          replace: C.replace,
          flushSync: Q
        });
        if (ve.shortCircuited) return;
        if (ve.pendingActionResult) {
          let [ie, Pe] = ve.pendingActionResult;
          if (yt(Pe) && _l(Pe.error) && Pe.error.status === 404) {
            D = null, Z(N, {
              matches: ve.matches,
              loaderData: {},
              errors: {
                [ie]: Pe.error
              }
            });
            return;
          }
        }
        W = ve.matches || W, A = ve.pendingActionResult, L = Yo(N, C.submission), Q = false, I.active = false, U = rr(e.history, U.url, U.signal);
      }
      let { shortCircuited: B, matches: G, loaderData: re, errors: Ke } = await ot(U, N, W, I.active, L, C && C.submission, C && C.fetcherSubmission, C && C.replace, C && C.initialHydration === true, Q, A);
      B || (D = null, Z(N, ke({
        matches: G || W
      }, Rc(A), {
        loaderData: re,
        errors: Ke
      })));
    }
    async function tt(y, N, C, T, L, W) {
      W === void 0 && (W = {}), Ft();
      let Q = ev(N, C);
      if (Y({
        navigation: Q
      }, {
        flushSync: W.flushSync === true
      }), L) {
        let A = await Ul(T, N.pathname, y.signal);
        if (A.type === "aborted") return {
          shortCircuited: true
        };
        if (A.type === "error") {
          let B = In(A.partialMatches).route.id;
          return {
            matches: A.partialMatches,
            pendingActionResult: [
              B,
              {
                type: ue.error,
                error: A.error
              }
            ]
          };
        } else if (A.matches) T = A.matches;
        else {
          let { notFoundMatches: B, error: G, route: re } = vo(N.pathname);
          return {
            matches: B,
            pendingActionResult: [
              re.id,
              {
                type: ue.error,
                error: G
              }
            ]
          };
        }
      }
      let I, U = Zr(T, N);
      if (!U.route.action && !U.route.lazy) I = {
        type: ue.error,
        error: it(405, {
          method: y.method,
          pathname: N.pathname,
          routeId: U.route.id
        })
      };
      else if (I = (await Ie("action", S, y, [
        U
      ], T, null))[U.route.id], y.signal.aborted) return {
        shortCircuited: true
      };
      if ($n(I)) {
        let A;
        return W && W.replace != null ? A = W.replace : A = jc(I.response.headers.get("Location"), new URL(y.url), s, e.history) === S.location.pathname + S.location.search, await ne(y, I, true, {
          submission: C,
          replace: A
        }), {
          shortCircuited: true
        };
      }
      if (gn(I)) throw it(400, {
        type: "defer-action"
      });
      if (yt(I)) {
        let A = In(T, U.route.id);
        return (W && W.replace) !== true && (j = Ue.Push), {
          matches: T,
          pendingActionResult: [
            A.route.id,
            I
          ]
        };
      }
      return {
        matches: T,
        pendingActionResult: [
          U.route.id,
          I
        ]
      };
    }
    async function ot(y, N, C, T, L, W, Q, I, U, A, B) {
      let G = L || Yo(N, W), re = W || Q || Pc(G), Ke = !le && (!d.v7_partialHydration || !U);
      if (T) {
        if (Ke) {
          let Me = Ae(B);
          Y(ke({
            navigation: G
          }, Me !== void 0 ? {
            actionData: Me
          } : {}), {
            flushSync: A
          });
        }
        let oe = await Ul(C, N.pathname, y.signal);
        if (oe.type === "aborted") return {
          shortCircuited: true
        };
        if (oe.type === "error") {
          let Me = In(oe.partialMatches).route.id;
          return {
            matches: oe.partialMatches,
            loaderData: {},
            errors: {
              [Me]: oe.error
            }
          };
        } else if (oe.matches) C = oe.matches;
        else {
          let { error: Me, notFoundMatches: er, route: Ar } = vo(N.pathname);
          return {
            matches: er,
            loaderData: {},
            errors: {
              [Ar.id]: Me
            }
          };
        }
      }
      let ve = i || o, [ie, Pe] = Sc(e.history, S, C, re, N, d.v7_partialHydration && U === true, d.v7_skipActionErrorRevalidation, je, Qe, pt, ge, be, X, ve, s, B);
      if (xo((oe) => !(C && C.some((Me) => Me.route.id === oe)) || ie && ie.some((Me) => Me.route.id === oe)), F = ++V, ie.length === 0 && Pe.length === 0) {
        let oe = Zi();
        return Z(N, ke({
          matches: C,
          loaderData: {},
          errors: B && yt(B[1]) ? {
            [B[0]]: B[1].error
          } : null
        }, Rc(B), oe ? {
          fetchers: new Map(S.fetchers)
        } : {}), {
          flushSync: A
        }), {
          shortCircuited: true
        };
      }
      if (Ke) {
        let oe = {};
        if (!T) {
          oe.navigation = G;
          let Me = Ae(B);
          Me !== void 0 && (oe.actionData = Me);
        }
        Pe.length > 0 && (oe.fetchers = ht(Pe)), Y(oe, {
          flushSync: A
        });
      }
      Pe.forEach((oe) => {
        on(oe.key), oe.controller && M.set(oe.key, oe.controller);
      });
      let qn = () => Pe.forEach((oe) => on(oe.key));
      D && D.signal.addEventListener("abort", qn);
      let { loaderResults: Lr, fetcherResults: Yt } = await gt(S, C, ie, Pe, y);
      if (y.signal.aborted) return {
        shortCircuited: true
      };
      D && D.signal.removeEventListener("abort", qn), Pe.forEach((oe) => M.delete(oe.key));
      let $t = aa(Lr);
      if ($t) return await ne(y, $t.result, true, {
        replace: I
      }), {
        shortCircuited: true
      };
      if ($t = aa(Yt), $t) return X.add($t.key), await ne(y, $t.result, true, {
        replace: I
      }), {
        shortCircuited: true
      };
      let { loaderData: yo, errors: Or } = Cc(S, C, Lr, B, Pe, Yt, Se);
      Se.forEach((oe, Me) => {
        oe.subscribe((er) => {
          (er || oe.done) && Se.delete(Me);
        });
      }), d.v7_partialHydration && U && S.errors && (Or = ke({}, S.errors, Or));
      let Mn = Zi(), Fl = qi(F), $l = Mn || Fl || Pe.length > 0;
      return ke({
        matches: C,
        loaderData: yo,
        errors: Or
      }, $l ? {
        fetchers: new Map(S.fetchers)
      } : {});
    }
    function Ae(y) {
      if (y && !yt(y[1])) return {
        [y[0]]: y[1].data
      };
      if (S.actionData) return Object.keys(S.actionData).length === 0 ? null : S.actionData;
    }
    function ht(y) {
      return y.forEach((N) => {
        let C = S.fetchers.get(N.key), T = Qr(void 0, C ? C.data : void 0);
        S.fetchers.set(N.key, T);
      }), new Map(S.fetchers);
    }
    function st(y, N, C, T) {
      if (r) throw new Error("router.fetch() was called during the server render, but it shouldn't be. You are likely calling a useFetcher() method in the body of your component. Try moving it to a useEffect or a callback.");
      on(y);
      let L = (T && T.flushSync) === true, W = i || o, Q = Ws(S.location, S.matches, s, d.v7_prependBasename, C, d.v7_relativeSplatPath, N, T == null ? void 0 : T.relative), I = An(W, Q, s), U = zl(I, W, Q);
      if (U.active && U.matches && (I = U.matches), !I) {
        Pt(y, N, it(404, {
          pathname: Q
        }), {
          flushSync: L
        });
        return;
      }
      let { path: A, submission: B, error: G } = yc(d.v7_normalizeFormMethod, true, Q, T);
      if (G) {
        Pt(y, N, G, {
          flushSync: L
        });
        return;
      }
      let re = Zr(I, A), Ke = (T && T.preventScrollReset) === true;
      if (B && Ot(B.formMethod)) {
        an(y, N, A, re, I, U.active, L, Ke, B);
        return;
      }
      be.set(y, {
        routeId: N,
        path: A
      }), he(y, N, A, re, I, U.active, L, Ke, B);
    }
    async function an(y, N, C, T, L, W, Q, I, U) {
      Ft(), be.delete(y);
      function A(ze) {
        if (!ze.route.action && !ze.route.lazy) {
          let tr = it(405, {
            method: U.formMethod,
            pathname: C,
            routeId: N
          });
          return Pt(y, N, tr, {
            flushSync: Q
          }), true;
        }
        return false;
      }
      if (!W && A(T)) return;
      let B = S.fetchers.get(y);
      vt(y, tv(U, B), {
        flushSync: Q
      });
      let G = new AbortController(), re = rr(e.history, C, G.signal, U);
      if (W) {
        let ze = await Ul(L, new URL(re.url).pathname, re.signal, y);
        if (ze.type === "aborted") return;
        if (ze.type === "error") {
          Pt(y, N, ze.error, {
            flushSync: Q
          });
          return;
        } else if (ze.matches) {
          if (L = ze.matches, T = Zr(L, C), A(T)) return;
        } else {
          Pt(y, N, it(404, {
            pathname: C
          }), {
            flushSync: Q
          });
          return;
        }
      }
      M.set(y, G);
      let Ke = V, ie = (await Ie("action", S, re, [
        T
      ], L, y))[T.route.id];
      if (re.signal.aborted) {
        M.get(y) === G && M.delete(y);
        return;
      }
      if (d.v7_fetcherPersist && ge.has(y)) {
        if ($n(ie) || yt(ie)) {
          vt(y, un(void 0));
          return;
        }
      } else {
        if ($n(ie)) if (M.delete(y), F > Ke) {
          vt(y, un(void 0));
          return;
        } else return X.add(y), vt(y, Qr(U)), ne(re, ie, false, {
          fetcherSubmission: U,
          preventScrollReset: I
        });
        if (yt(ie)) {
          Pt(y, N, ie.error);
          return;
        }
      }
      if (gn(ie)) throw it(400, {
        type: "defer-action"
      });
      let Pe = S.navigation.location || S.location, qn = rr(e.history, Pe, G.signal), Lr = i || o, Yt = S.navigation.state !== "idle" ? An(Lr, S.navigation.location, s) : S.matches;
      te(Yt, "Didn't find any matches after fetcher action");
      let $t = ++V;
      ee.set(y, $t);
      let yo = Qr(U, ie.data);
      S.fetchers.set(y, yo);
      let [Or, Mn] = Sc(e.history, S, Yt, U, Pe, false, d.v7_skipActionErrorRevalidation, je, Qe, pt, ge, be, X, Lr, s, [
        T.route.id,
        ie
      ]);
      Mn.filter((ze) => ze.key !== y).forEach((ze) => {
        let tr = ze.key, lu = S.fetchers.get(tr), Pm = Qr(void 0, lu ? lu.data : void 0);
        S.fetchers.set(tr, Pm), on(tr), ze.controller && M.set(tr, ze.controller);
      }), Y({
        fetchers: new Map(S.fetchers)
      });
      let Fl = () => Mn.forEach((ze) => on(ze.key));
      G.signal.addEventListener("abort", Fl);
      let { loaderResults: $l, fetcherResults: oe } = await gt(S, Yt, Or, Mn, qn);
      if (G.signal.aborted) return;
      G.signal.removeEventListener("abort", Fl), ee.delete(y), M.delete(y), Mn.forEach((ze) => M.delete(ze.key));
      let Me = aa($l);
      if (Me) return ne(qn, Me.result, false, {
        preventScrollReset: I
      });
      if (Me = aa(oe), Me) return X.add(Me.key), ne(qn, Me.result, false, {
        preventScrollReset: I
      });
      let { loaderData: er, errors: Ar } = Cc(S, Yt, $l, void 0, Mn, oe, Se);
      if (S.fetchers.has(y)) {
        let ze = un(ie.data);
        S.fetchers.set(y, ze);
      }
      qi($t), S.navigation.state === "loading" && $t > F ? (te(j, "Expected pending action"), D && D.abort(), Z(S.navigation.location, {
        matches: Yt,
        loaderData: er,
        errors: Ar,
        fetchers: new Map(S.fetchers)
      })) : (Y({
        errors: Ar,
        loaderData: _c(S.loaderData, er, Yt, Ar),
        fetchers: new Map(S.fetchers)
      }), je = false);
    }
    async function he(y, N, C, T, L, W, Q, I, U) {
      let A = S.fetchers.get(y);
      vt(y, Qr(U, A ? A.data : void 0), {
        flushSync: Q
      });
      let B = new AbortController(), G = rr(e.history, C, B.signal);
      if (W) {
        let ie = await Ul(L, new URL(G.url).pathname, G.signal, y);
        if (ie.type === "aborted") return;
        if (ie.type === "error") {
          Pt(y, N, ie.error, {
            flushSync: Q
          });
          return;
        } else if (ie.matches) L = ie.matches, T = Zr(L, C);
        else {
          Pt(y, N, it(404, {
            pathname: C
          }), {
            flushSync: Q
          });
          return;
        }
      }
      M.set(y, B);
      let re = V, ve = (await Ie("loader", S, G, [
        T
      ], L, y))[T.route.id];
      if (gn(ve) && (ve = await Hi(ve, G.signal, true) || ve), M.get(y) === B && M.delete(y), !G.signal.aborted) {
        if (ge.has(y)) {
          vt(y, un(void 0));
          return;
        }
        if ($n(ve)) if (F > re) {
          vt(y, un(void 0));
          return;
        } else {
          X.add(y), await ne(G, ve, false, {
            preventScrollReset: I
          });
          return;
        }
        if (yt(ve)) {
          Pt(y, N, ve.error);
          return;
        }
        te(!gn(ve), "Unhandled fetcher deferred data"), vt(y, un(ve.data));
      }
    }
    async function ne(y, N, C, T) {
      let { submission: L, fetcherSubmission: W, preventScrollReset: Q, replace: I } = T === void 0 ? {} : T;
      N.response.headers.has("X-Remix-Revalidate") && (je = true);
      let U = N.response.headers.get("Location");
      te(U, "Expected a Location header on the redirect Response"), U = jc(U, new URL(y.url), s, e.history);
      let A = Cl(S.location, U, {
        _isRedirect: true
      });
      if (n) {
        let ie = false;
        if (N.response.headers.has("X-Remix-Reload-Document")) ie = true;
        else if (Wi.test(U)) {
          const Pe = e.history.createURL(U);
          ie = Pe.origin !== t.location.origin || Ll(Pe.pathname, s) == null;
        }
        if (ie) {
          I ? t.location.replace(U) : t.location.assign(U);
          return;
        }
      }
      D = null;
      let B = I === true || N.response.headers.has("X-Remix-Replace") ? Ue.Replace : Ue.Push, { formMethod: G, formAction: re, formEncType: Ke } = S.navigation;
      !L && !W && G && re && Ke && (L = Pc(S.navigation));
      let ve = L || W;
      if (Og.has(N.response.status) && ve && Ot(ve.formMethod)) await Oe(B, A, {
        submission: ke({}, ve, {
          formAction: U
        }),
        preventScrollReset: Q || z,
        enableViewTransition: C ? H : void 0
      });
      else {
        let ie = Yo(A, L);
        await Oe(B, A, {
          overrideNavigation: ie,
          fetcherSubmission: W,
          preventScrollReset: Q || z,
          enableViewTransition: C ? H : void 0
        });
      }
    }
    async function Ie(y, N, C, T, L, W) {
      let Q, I = {};
      try {
        Q = await Vg(c, y, N, C, T, L, W, a, l);
      } catch (U) {
        return T.forEach((A) => {
          I[A.route.id] = {
            type: ue.error,
            error: U
          };
        }), I;
      }
      for (let [U, A] of Object.entries(Q)) if (Gg(A)) {
        let B = A.result;
        I[U] = {
          type: ue.redirect,
          response: Qg(B, C, U, L, s, d.v7_relativeSplatPath)
        };
      } else I[U] = await Hg(A);
      return I;
    }
    async function gt(y, N, C, T, L) {
      let W = y.matches, Q = Ie("loader", y, L, C, N, null), I = Promise.all(T.map(async (B) => {
        if (B.matches && B.match && B.controller) {
          let re = (await Ie("loader", y, rr(e.history, B.path, B.controller.signal), [
            B.match
          ], B.matches, B.key))[B.match.route.id];
          return {
            [B.key]: re
          };
        } else return Promise.resolve({
          [B.key]: {
            type: ue.error,
            error: it(404, {
              pathname: B.path
            })
          }
        });
      })), U = await Q, A = (await I).reduce((B, G) => Object.assign(B, G), {});
      return await Promise.all([
        Zg(N, U, L.signal, W, y.loaderData),
        qg(N, A, T)
      ]), {
        loaderResults: U,
        fetcherResults: A
      };
    }
    function Ft() {
      je = true, Qe.push(...xo()), be.forEach((y, N) => {
        M.has(N) && pt.add(N), on(N);
      });
    }
    function vt(y, N, C) {
      C === void 0 && (C = {}), S.fetchers.set(y, N), Y({
        fetchers: new Map(S.fetchers)
      }, {
        flushSync: (C && C.flushSync) === true
      });
    }
    function Pt(y, N, C, T) {
      T === void 0 && (T = {});
      let L = In(S.matches, N);
      Al(y), Y({
        errors: {
          [L.route.id]: C
        },
        fetchers: new Map(S.fetchers)
      }, {
        flushSync: (T && T.flushSync) === true
      });
    }
    function Ol(y) {
      return Ee.set(y, (Ee.get(y) || 0) + 1), ge.has(y) && ge.delete(y), S.fetchers.get(y) || Ag;
    }
    function Al(y) {
      let N = S.fetchers.get(y);
      M.has(y) && !(N && N.state === "loading" && ee.has(y)) && on(y), be.delete(y), ee.delete(y), X.delete(y), d.v7_fetcherPersist && ge.delete(y), pt.delete(y), S.fetchers.delete(y);
    }
    function Em(y) {
      let N = (Ee.get(y) || 0) - 1;
      N <= 0 ? (Ee.delete(y), ge.add(y), d.v7_fetcherPersist || Al(y)) : Ee.set(y, N), Y({
        fetchers: new Map(S.fetchers)
      });
    }
    function on(y) {
      let N = M.get(y);
      N && (N.abort(), M.delete(y));
    }
    function Ji(y) {
      for (let N of y) {
        let C = Ol(N), T = un(C.data);
        S.fetchers.set(N, T);
      }
    }
    function Zi() {
      let y = [], N = false;
      for (let C of X) {
        let T = S.fetchers.get(C);
        te(T, "Expected fetcher: " + C), T.state === "loading" && (X.delete(C), y.push(C), N = true);
      }
      return Ji(y), N;
    }
    function qi(y) {
      let N = [];
      for (let [C, T] of ee) if (T < y) {
        let L = S.fetchers.get(C);
        te(L, "Expected fetcher: " + C), L.state === "loading" && (on(C), ee.delete(C), N.push(C));
      }
      return Ji(N), N.length > 0;
    }
    function Cm(y, N) {
      let C = S.blockers.get(y) || Hr;
      return J.get(y) !== N && J.set(y, N), C;
    }
    function eu(y) {
      S.blockers.delete(y), J.delete(y);
    }
    function Il(y, N) {
      let C = S.blockers.get(y) || Hr;
      te(C.state === "unblocked" && N.state === "blocked" || C.state === "blocked" && N.state === "blocked" || C.state === "blocked" && N.state === "proceeding" || C.state === "blocked" && N.state === "unblocked" || C.state === "proceeding" && N.state === "unblocked", "Invalid blocker state transition: " + C.state + " -> " + N.state);
      let T = new Map(S.blockers);
      T.set(y, N), Y({
        blockers: T
      });
    }
    function tu(y) {
      let { currentLocation: N, nextLocation: C, historyAction: T } = y;
      if (J.size === 0) return;
      J.size > 1 && Gn(false, "A router only supports one blocker at a time");
      let L = Array.from(J.entries()), [W, Q] = L[L.length - 1], I = S.blockers.get(W);
      if (!(I && I.state === "proceeding") && Q({
        currentLocation: N,
        nextLocation: C,
        historyAction: T
      })) return W;
    }
    function vo(y) {
      let N = it(404, {
        pathname: y
      }), C = i || o, { matches: T, route: L } = bc(C);
      return xo(), {
        notFoundMatches: T,
        route: L,
        error: N
      };
    }
    function xo(y) {
      let N = [];
      return Se.forEach((C, T) => {
        (!y || y(T)) && (C.cancel(), N.push(T), Se.delete(T));
      }), N;
    }
    function _m(y, N, C) {
      if (w = y, R = N, k = C || null, !h && S.navigation === Ko) {
        h = true;
        let T = ru(S.location, S.matches);
        T != null && Y({
          restoreScrollPosition: T
        });
      }
      return () => {
        w = null, R = null, k = null;
      };
    }
    function nu(y, N) {
      return k && k(y, N.map((T) => fg(T, S.loaderData))) || y.key;
    }
    function Rm(y, N) {
      if (w && R) {
        let C = nu(y, N);
        w[C] = R();
      }
    }
    function ru(y, N) {
      if (w) {
        let C = nu(y, N), T = w[C];
        if (typeof T == "number") return T;
      }
      return null;
    }
    function zl(y, N, C) {
      if (m) if (y) {
        if (Object.keys(y[0].params).length > 0) return {
          active: true,
          matches: wa(N, C, s, true)
        };
      } else return {
        active: true,
        matches: wa(N, C, s, true) || []
      };
      return {
        active: false,
        matches: null
      };
    }
    async function Ul(y, N, C, T) {
      if (!m) return {
        type: "success",
        matches: y
      };
      let L = y;
      for (; ; ) {
        let W = i == null, Q = i || o, I = a;
        try {
          await m({
            signal: C,
            path: N,
            matches: L,
            fetcherKey: T,
            patch: (B, G) => {
              C.aborted || Nc(B, G, Q, I, l);
            }
          });
        } catch (B) {
          return {
            type: "error",
            error: B,
            partialMatches: L
          };
        } finally {
          W && !C.aborted && (o = [
            ...o
          ]);
        }
        if (C.aborted) return {
          type: "aborted"
        };
        let U = An(Q, N, s);
        if (U) return {
          type: "success",
          matches: U
        };
        let A = wa(Q, N, s, true);
        if (!A || L.length === A.length && L.every((B, G) => B.route.id === A[G].route.id)) return {
          type: "success",
          matches: null
        };
        L = A;
      }
    }
    function bm(y) {
      a = {}, i = Qa(y, l, void 0, a);
    }
    function Tm(y, N) {
      let C = i == null;
      Nc(y, N, i || o, a, l), C && (o = [
        ...o
      ], Y({}));
    }
    return b = {
      get basename() {
        return s;
      },
      get future() {
        return d;
      },
      get state() {
        return S;
      },
      get routes() {
        return o;
      },
      get window() {
        return t;
      },
      initialize: me,
      subscribe: $,
      enableScrollRestoration: _m,
      navigate: fe,
      fetch: st,
      revalidate: Ve,
      createHref: (y) => e.history.createHref(y),
      encodeLocation: (y) => e.history.encodeLocation(y),
      getFetcher: Ol,
      deleteFetcher: Em,
      dispose: pe,
      getBlocker: Cm,
      deleteBlocker: eu,
      patchRoutes: Tm,
      _internalFetchControllers: M,
      _internalActiveDeferreds: Se,
      _internalSetRoutes: bm
    }, b;
  }
  function Ug(e) {
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
    let m = im(l || ".", sm(s, a), Ll(e.pathname, n) || e.pathname, i === "path");
    if (l == null && (m.search = e.search, m.hash = e.hash), (l == null || l === "" || l === ".") && c) {
      let d = Qi(m.search);
      if (c.route.index && !d) m.search = m.search ? m.search.replace(/^\?/, "?index&") : "?index";
      else if (!c.route.index && d) {
        let p = new URLSearchParams(m.search), x = p.getAll("index");
        p.delete("index"), x.filter((k) => k).forEach((k) => p.append("index", k));
        let w = p.toString();
        m.search = w ? "?" + w : "";
      }
    }
    return r && n !== "/" && (m.pathname = m.pathname === "/" ? n : En([
      n,
      m.pathname
    ])), Dl(m);
  }
  function yc(e, t, n, r) {
    if (!r || !Ug(r)) return {
      path: n
    };
    if (r.formMethod && !Jg(r.formMethod)) return {
      path: n,
      error: it(405, {
        method: r.formMethod
      })
    };
    let l = () => ({
      path: n,
      error: it(400, {
        type: "invalid-body"
      })
    }), a = r.formMethod || "get", o = e ? a.toUpperCase() : a.toLowerCase(), i = mm(n);
    if (r.body !== void 0) {
      if (r.formEncType === "text/plain") {
        if (!Ot(o)) return l();
        let p = typeof r.body == "string" ? r.body : r.body instanceof FormData || r.body instanceof URLSearchParams ? Array.from(r.body.entries()).reduce((x, w) => {
          let [k, R] = w;
          return "" + x + k + "=" + R + `
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
            text: p
          }
        };
      } else if (r.formEncType === "application/json") {
        if (!Ot(o)) return l();
        try {
          let p = typeof r.body == "string" ? JSON.parse(r.body) : r.body;
          return {
            path: n,
            submission: {
              formMethod: o,
              formAction: i,
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
    te(typeof FormData == "function", "FormData is not available in this environment");
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
    if (Ot(m.formMethod)) return {
      path: n,
      submission: m
    };
    let d = Pn(n);
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
  function Sc(e, t, n, r, l, a, o, i, s, c, m, d, p, x, w, k) {
    let R = k ? yt(k[1]) ? k[1].error : k[1].data : void 0, h = e.createURL(t.location), f = e.createURL(l), v = n;
    a && t.errors ? v = wc(n, Object.keys(t.errors)[0], true) : k && yt(k[1]) && (v = wc(n, k[0]));
    let E = k ? k[1].statusCode : void 0, _ = o && E && E >= 400, b = v.filter((j, z) => {
      let { route: D } = j;
      if (D.lazy) return true;
      if (D.loader == null) return false;
      if (a) return Hs(D, t.loaderData, t.errors);
      if (Fg(t.loaderData, t.matches[z], j) || s.some((se) => se === j.route.id)) return true;
      let H = t.matches[z], K = j;
      return kc(j, ke({
        currentUrl: h,
        currentParams: H.params,
        nextUrl: f,
        nextParams: K.params
      }, r, {
        actionResult: R,
        actionStatus: E,
        defaultShouldRevalidate: _ ? false : i || h.pathname + h.search === f.pathname + f.search || h.search !== f.search || dm(H, K)
      }));
    }), S = [];
    return d.forEach((j, z) => {
      if (a || !n.some((le) => le.route.id === j.routeId) || m.has(z)) return;
      let D = An(x, j.path, w);
      if (!D) {
        S.push({
          key: z,
          routeId: j.routeId,
          path: j.path,
          matches: null,
          match: null,
          controller: null
        });
        return;
      }
      let H = t.fetchers.get(z), K = Zr(D, j.path), se = false;
      p.has(z) ? se = false : c.has(z) ? (c.delete(z), se = true) : H && H.state !== "idle" && H.data === void 0 ? se = i : se = kc(K, ke({
        currentUrl: h,
        currentParams: t.matches[t.matches.length - 1].params,
        nextUrl: f,
        nextParams: n[n.length - 1].params
      }, r, {
        actionResult: R,
        actionStatus: E,
        defaultShouldRevalidate: _ ? false : i
      })), se && S.push({
        key: z,
        routeId: j.routeId,
        path: j.path,
        matches: D,
        match: K,
        controller: new AbortController()
      });
    }), [
      b,
      S
    ];
  }
  function Hs(e, t, n) {
    if (e.lazy) return true;
    if (!e.loader) return false;
    let r = t != null && t[e.id] !== void 0, l = n != null && n[e.id] !== void 0;
    return !r && l ? false : typeof e.loader == "function" && e.loader.hydrate === true ? true : !r && !l;
  }
  function Fg(e, t, n) {
    let r = !t || n.route.id !== t.route.id, l = e[n.route.id] === void 0;
    return r || l;
  }
  function dm(e, t) {
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
      te(c, "No route found to patch children into: routeId = " + e), c.children || (c.children = []), o = c.children;
    } else o = n;
    let i = t.filter((c) => !o.some((m) => fm(c, m))), s = Qa(i, l, [
      e || "_",
      "patch",
      String(((a = o) == null ? void 0 : a.length) || "0")
    ], r);
    o.push(...s);
  }
  function fm(e, t) {
    return "id" in e && "id" in t && e.id === t.id ? true : e.index === t.index && e.path === t.path && e.caseSensitive === t.caseSensitive ? (!e.children || e.children.length === 0) && (!t.children || t.children.length === 0) ? true : e.children.every((n, r) => {
      var l;
      return (l = t.children) == null ? void 0 : l.some((a) => fm(n, a));
    }) : false;
  }
  async function $g(e, t, n) {
    if (!e.lazy) return;
    let r = await e.lazy();
    if (!e.lazy) return;
    let l = n[e.id];
    te(l, "No route found in manifest");
    let a = {};
    for (let o in r) {
      let s = l[o] !== void 0 && o !== "hasErrorBoundary";
      Gn(!s, 'Route "' + l.id + '" has a static property "' + o + '" defined but its lazy function is also returning a value for this property. ' + ('The lazy route property "' + o + '" will be ignored.')), !s && !cg.has(o) && (a[o] = r[o]);
    }
    Object.assign(l, a), Object.assign(l, ke({}, t(l), {
      lazy: void 0
    }));
  }
  async function Bg(e) {
    let { matches: t } = e, n = t.filter((l) => l.shouldLoad);
    return (await Promise.all(n.map((l) => l.resolve()))).reduce((l, a, o) => Object.assign(l, {
      [n[o].route.id]: a
    }), {});
  }
  async function Vg(e, t, n, r, l, a, o, i, s, c) {
    let m = a.map((x) => x.route.lazy ? $g(x.route, s, i) : void 0), d = a.map((x, w) => {
      let k = m[w], R = l.some((f) => f.route.id === x.route.id);
      return ke({}, x, {
        shouldLoad: R,
        resolve: async (f) => (f && r.method === "GET" && (x.route.lazy || x.route.loader) && (R = true), R ? Wg(t, r, x, k, f, c) : Promise.resolve({
          type: ue.data,
          result: void 0
        }))
      });
    }), p = await e({
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
    return p;
  }
  async function Wg(e, t, n, r, l, a) {
    let o, i, s = (c) => {
      let m, d = new Promise((w, k) => m = k);
      i = () => m(), t.signal.addEventListener("abort", i);
      let p = (w) => typeof c != "function" ? Promise.reject(new Error("You cannot call the handler for a route which defines a boolean " + ('"' + e + '" [routeId: ' + n.route.id + "]"))) : c({
        request: t,
        params: n.params,
        context: a
      }, ...w !== void 0 ? [
        w
      ] : []), x = (async () => {
        try {
          return {
            type: "data",
            result: await (l ? l((k) => p(k)) : p())
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
          s(c).catch((p) => {
            m = p;
          }),
          r
        ]);
        if (m !== void 0) throw m;
        o = d;
      } else if (await r, c = n.route[e], c) o = await s(c);
      else if (e === "action") {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw it(405, {
          method: t.method,
          pathname: d,
          routeId: n.route.id
        });
      } else return {
        type: ue.data,
        result: void 0
      };
      else if (c) o = await s(c);
      else {
        let m = new URL(t.url), d = m.pathname + m.search;
        throw it(404, {
          pathname: d
        });
      }
      te(o.result !== void 0, "You defined " + (e === "action" ? "an action" : "a loader") + " for route " + ('"' + n.route.id + "\" but didn't return anything from your `" + e + "` ") + "function. Please return a value or `null`.");
    } catch (c) {
      return {
        type: ue.error,
        result: c
      };
    } finally {
      i && t.signal.removeEventListener("abort", i);
    }
    return o;
  }
  async function Hg(e) {
    let { result: t, type: n } = e;
    if (pm(t)) {
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
        error: new Ka(t.status, t.statusText, d),
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
      if (Tc(t)) {
        var r, l;
        if (t.data instanceof Error) {
          var a, o;
          return {
            type: ue.error,
            error: t.data,
            statusCode: (a = t.init) == null ? void 0 : a.status,
            headers: (o = t.init) != null && o.headers ? new Headers(t.init.headers) : void 0
          };
        }
        return {
          type: ue.error,
          error: new Ka(((r = t.init) == null ? void 0 : r.status) || 500, void 0, t.data),
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
    if (Xg(t)) {
      var i, s;
      return {
        type: ue.deferred,
        deferredData: t,
        statusCode: (i = t.init) == null ? void 0 : i.status,
        headers: ((s = t.init) == null ? void 0 : s.headers) && new Headers(t.init.headers)
      };
    }
    if (Tc(t)) {
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
  function Qg(e, t, n, r, l, a) {
    let o = e.headers.get("Location");
    if (te(o, "Redirects returned/thrown from loaders/actions must have a Location header"), !Wi.test(o)) {
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
  function rr(e, t, n, r) {
    let l = e.createURL(mm(t)).toString(), a = {
      signal: n
    };
    if (r && Ot(r.formMethod)) {
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
  function Kg(e, t, n, r, l) {
    let a = {}, o = null, i, s = false, c = {}, m = n && yt(n[1]) ? n[1].error : void 0;
    return e.forEach((d) => {
      if (!(d.route.id in t)) return;
      let p = d.route.id, x = t[p];
      if (te(!$n(x), "Cannot handle redirect results in processLoaderData"), yt(x)) {
        let w = x.error;
        m !== void 0 && (w = m, m = void 0), o = o || {};
        {
          let k = In(e, p);
          o[k.route.id] == null && (o[k.route.id] = w);
        }
        a[p] = void 0, s || (s = true, i = _l(x.error) ? x.error.status : 500), x.headers && (c[p] = x.headers);
      } else gn(x) ? (r.set(p, x.deferredData), a[p] = x.deferredData.data, x.statusCode != null && x.statusCode !== 200 && !s && (i = x.statusCode), x.headers && (c[p] = x.headers)) : (a[p] = x.data, x.statusCode && x.statusCode !== 200 && !s && (i = x.statusCode), x.headers && (c[p] = x.headers));
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
    let { loaderData: i, errors: s } = Kg(t, n, r, o);
    return l.forEach((c) => {
      let { key: m, match: d, controller: p } = c, x = a[m];
      if (te(x, "Did not find corresponding fetcher result"), !(p && p.signal.aborted)) if (yt(x)) {
        let w = In(e.matches, d == null ? void 0 : d.route.id);
        s && s[w.route.id] || (s = ke({}, s, {
          [w.route.id]: x.error
        })), e.fetchers.delete(m);
      } else if ($n(x)) te(false, "Unhandled fetcher revalidation redirect");
      else if (gn(x)) te(false, "Unhandled fetcher deferred data");
      else {
        let w = un(x.data);
        e.fetchers.set(m, w);
      }
    }), {
      loaderData: i,
      errors: s
    };
  }
  function _c(e, t, n, r) {
    let l = ke({}, t);
    for (let a of n) {
      let o = a.route.id;
      if (t.hasOwnProperty(o) ? t[o] !== void 0 && (l[o] = t[o]) : e[o] !== void 0 && a.route.loader && (l[o] = e[o]), r && r.hasOwnProperty(o)) break;
    }
    return l;
  }
  function Rc(e) {
    return e ? yt(e[1]) ? {
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
  function bc(e) {
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
  function it(e, t) {
    let { pathname: n, routeId: r, method: l, type: a, message: o } = t === void 0 ? {} : t, i = "Unknown Server Error", s = "Unknown @remix-run/router error";
    return e === 400 ? (i = "Bad Request", l && n && r ? s = "You made a " + l + ' request to "' + n + '" but ' + ('did not provide a `loader` for route "' + r + '", ') + "so there is no way to handle the request." : a === "defer-action" ? s = "defer() is not supported in actions" : a === "invalid-body" && (s = "Unable to encode submission body")) : e === 403 ? (i = "Forbidden", s = 'Route "' + r + '" does not match URL "' + n + '"') : e === 404 ? (i = "Not Found", s = 'No route matches URL "' + n + '"') : e === 405 && (i = "Method Not Allowed", l && n && r ? s = "You made a " + l.toUpperCase() + ' request to "' + n + '" but ' + ('did not provide an `action` for route "' + r + '", ') + "so there is no way to handle the request." : l && (s = 'Invalid request method "' + l.toUpperCase() + '"')), new Ka(e || 500, i, new Error(s), true);
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
  function mm(e) {
    let t = typeof e == "string" ? Pn(e) : e;
    return Dl(ke({}, t, {
      hash: ""
    }));
  }
  function Yg(e, t) {
    return e.pathname !== t.pathname || e.search !== t.search ? false : e.hash === "" ? t.hash !== "" : e.hash === t.hash ? true : t.hash !== "";
  }
  function Gg(e) {
    return pm(e.result) && Lg.has(e.result.status);
  }
  function gn(e) {
    return e.type === ue.deferred;
  }
  function yt(e) {
    return e.type === ue.error;
  }
  function $n(e) {
    return (e && e.type) === ue.redirect;
  }
  function Tc(e) {
    return typeof e == "object" && e != null && "type" in e && "data" in e && "init" in e && e.type === "DataWithResponseInit";
  }
  function Xg(e) {
    let t = e;
    return t && typeof t == "object" && typeof t.data == "object" && typeof t.subscribe == "function" && typeof t.cancel == "function" && typeof t.resolveData == "function";
  }
  function pm(e) {
    return e != null && typeof e.status == "number" && typeof e.statusText == "string" && typeof e.headers == "object" && typeof e.body < "u";
  }
  function Jg(e) {
    return Dg.has(e.toLowerCase());
  }
  function Ot(e) {
    return Pg.has(e.toLowerCase());
  }
  async function Zg(e, t, n, r, l) {
    let a = Object.entries(t);
    for (let o = 0; o < a.length; o++) {
      let [i, s] = a[o], c = e.find((p) => (p == null ? void 0 : p.route.id) === i);
      if (!c) continue;
      let m = r.find((p) => p.route.id === c.route.id), d = m != null && !dm(m, c) && (l && l[c.route.id]) !== void 0;
      gn(s) && d && await Hi(s, n, false).then((p) => {
        p && (t[i] = p);
      });
    }
  }
  async function qg(e, t, n) {
    for (let r = 0; r < n.length; r++) {
      let { key: l, routeId: a, controller: o } = n[r], i = t[l];
      e.find((c) => (c == null ? void 0 : c.route.id) === a) && gn(i) && (te(o, "Expected an AbortController for revalidating fetcher deferred result"), await Hi(i, o.signal, true).then((c) => {
        c && (t[l] = c);
      }));
    }
  }
  async function Hi(e, t, n) {
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
  function Qi(e) {
    return new URLSearchParams(e).getAll("index").some((t) => t === "");
  }
  function Zr(e, t) {
    let n = typeof t == "string" ? Pn(t).search : t.search;
    if (e[e.length - 1].route.index && Qi(n || "")) return e[e.length - 1];
    let r = om(e);
    return r[r.length - 1];
  }
  function Pc(e) {
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
  function Yo(e, t) {
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
  function ev(e, t) {
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
  function tv(e, t) {
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
  function nv(e, t) {
    try {
      let n = e.sessionStorage.getItem(cm);
      if (n) {
        let r = JSON.parse(n);
        for (let [l, a] of Object.entries(r || {})) a && Array.isArray(a) && t.set(l, new Set(a || []));
      }
    } catch {
    }
  }
  function rv(e, t) {
    if (t.size > 0) {
      let n = {};
      for (let [r, l] of t) n[r] = [
        ...l
      ];
      try {
        e.sessionStorage.setItem(cm, JSON.stringify(n));
      } catch (r) {
        Gn(false, "Failed to save applied view transitions in sessionStorage (" + r + ").");
      }
    }
  }
  function Ya() {
    return Ya = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ya.apply(this, arguments);
  }
  const mo = g.createContext(null), hm = g.createContext(null), po = g.createContext(null), Ki = g.createContext(null), Zn = g.createContext({
    outlet: null,
    matches: [],
    isDataRoute: false
  }), gm = g.createContext(null);
  function ho() {
    return g.useContext(Ki) != null;
  }
  function Yi() {
    return ho() || te(false), g.useContext(Ki).location;
  }
  function vm(e) {
    g.useContext(po).static || g.useLayoutEffect(e);
  }
  function go() {
    let { isDataRoute: e } = g.useContext(Zn);
    return e ? vv() : lv();
  }
  function lv() {
    ho() || te(false);
    let e = g.useContext(mo), { basename: t, future: n, navigator: r } = g.useContext(po), { matches: l } = g.useContext(Zn), { pathname: a } = Yi(), o = JSON.stringify(sm(l, n.v7_relativeSplatPath)), i = g.useRef(false);
    return vm(() => {
      i.current = true;
    }), g.useCallback(function(c, m) {
      if (m === void 0 && (m = {}), !i.current) return;
      if (typeof c == "number") {
        r.go(c);
        return;
      }
      let d = im(c, JSON.parse(o), a, m.relative === "path");
      e == null && t !== "/" && (d.pathname = d.pathname === "/" ? t : En([
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
  const av = g.createContext(null);
  function ov(e) {
    let t = g.useContext(Zn).outlet;
    return t && g.createElement(av.Provider, {
      value: e
    }, t);
  }
  function sv(e, t, n, r) {
    ho() || te(false);
    let { navigator: l } = g.useContext(po), { matches: a } = g.useContext(Zn), o = a[a.length - 1], i = o ? o.params : {};
    o && o.pathname;
    let s = o ? o.pathnameBase : "/";
    o && o.route;
    let c = Yi(), m;
    m = c;
    let d = m.pathname || "/", p = d;
    if (s !== "/") {
      let k = s.replace(/^\//, "").split("/");
      p = "/" + d.replace(/^\//, "").split("/").slice(k.length).join("/");
    }
    let x = An(e, {
      pathname: p
    });
    return fv(x && x.map((k) => Object.assign({}, k, {
      params: Object.assign({}, i, k.params),
      pathname: En([
        s,
        l.encodeLocation ? l.encodeLocation(k.pathname).pathname : k.pathname
      ]),
      pathnameBase: k.pathnameBase === "/" ? s : En([
        s,
        l.encodeLocation ? l.encodeLocation(k.pathnameBase).pathname : k.pathnameBase
      ])
    })), a, n, r);
  }
  function iv() {
    let e = gv(), t = _l(e) ? e.status + " " + e.statusText : e instanceof Error ? e.message : JSON.stringify(e), n = e instanceof Error ? e.stack : null, l = {
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
  const uv = g.createElement(iv, null);
  class cv extends g.Component {
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
      return this.state.error !== void 0 ? g.createElement(Zn.Provider, {
        value: this.props.routeContext
      }, g.createElement(gm.Provider, {
        value: this.state.error,
        children: this.props.component
      })) : this.props.children;
    }
  }
  function dv(e) {
    let { routeContext: t, match: n, children: r } = e, l = g.useContext(mo);
    return l && l.static && l.staticContext && (n.route.errorElement || n.route.ErrorBoundary) && (l.staticContext._deepestRenderedBoundaryId = n.route.id), g.createElement(Zn.Provider, {
      value: t
    }, r);
  }
  function fv(e, t, n, r) {
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
      m >= 0 || te(false), o = o.slice(0, Math.min(o.length, m + 1));
    }
    let s = false, c = -1;
    if (n && r && r.v7_partialHydration) for (let m = 0; m < o.length; m++) {
      let d = o[m];
      if ((d.route.HydrateFallback || d.route.hydrateFallbackElement) && (c = m), d.route.id) {
        let { loaderData: p, errors: x } = n, w = d.route.loader && p[d.route.id] === void 0 && (!x || x[d.route.id] === void 0);
        if (d.route.lazy || w) {
          s = true, c >= 0 ? o = o.slice(0, c + 1) : o = [
            o[0]
          ];
          break;
        }
      }
    }
    return o.reduceRight((m, d, p) => {
      let x, w = false, k = null, R = null;
      n && (x = i && d.route.id ? i[d.route.id] : void 0, k = d.route.errorElement || uv, s && (c < 0 && p === 0 ? (xv("route-fallback"), w = true, R = null) : c === p && (w = true, R = d.route.hydrateFallbackElement || null)));
      let h = t.concat(o.slice(0, p + 1)), f = () => {
        let v;
        return x ? v = k : w ? v = R : d.route.Component ? v = g.createElement(d.route.Component, null) : d.route.element ? v = d.route.element : v = m, g.createElement(dv, {
          match: d,
          routeContext: {
            outlet: m,
            matches: h,
            isDataRoute: n != null
          },
          children: v
        });
      };
      return n && (d.route.ErrorBoundary || d.route.errorElement || p === 0) ? g.createElement(cv, {
        location: n.location,
        revalidation: n.revalidation,
        component: k,
        error: x,
        children: f(),
        routeContext: {
          outlet: null,
          matches: h,
          isDataRoute: true
        }
      }) : f();
    }, null);
  }
  var xm = function(e) {
    return e.UseBlocker = "useBlocker", e.UseRevalidator = "useRevalidator", e.UseNavigateStable = "useNavigate", e;
  }(xm || {}), ym = function(e) {
    return e.UseBlocker = "useBlocker", e.UseLoaderData = "useLoaderData", e.UseActionData = "useActionData", e.UseRouteError = "useRouteError", e.UseNavigation = "useNavigation", e.UseRouteLoaderData = "useRouteLoaderData", e.UseMatches = "useMatches", e.UseRevalidator = "useRevalidator", e.UseNavigateStable = "useNavigate", e.UseRouteId = "useRouteId", e;
  }(ym || {});
  function mv(e) {
    let t = g.useContext(mo);
    return t || te(false), t;
  }
  function pv(e) {
    let t = g.useContext(hm);
    return t || te(false), t;
  }
  function hv(e) {
    let t = g.useContext(Zn);
    return t || te(false), t;
  }
  function wm(e) {
    let t = hv(), n = t.matches[t.matches.length - 1];
    return n.route.id || te(false), n.route.id;
  }
  function gv() {
    var e;
    let t = g.useContext(gm), n = pv(ym.UseRouteError), r = wm();
    return t !== void 0 ? t : (e = n.errors) == null ? void 0 : e[r];
  }
  function vv() {
    let { router: e } = mv(xm.UseNavigateStable), t = wm(), n = g.useRef(false);
    return vm(() => {
      n.current = true;
    }), g.useCallback(function(l, a) {
      a === void 0 && (a = {}), n.current && (typeof l == "number" ? e.navigate(l) : e.navigate(l, Ya({
        fromRouteId: t
      }, a)));
    }, [
      e,
      t
    ]);
  }
  const Mc = {};
  function xv(e, t, n) {
    Mc[e] || (Mc[e] = true);
  }
  function yv(e, t) {
    e == null ? void 0 : e.v7_startTransition, (e == null ? void 0 : e.v7_relativeSplatPath) === void 0 && (!t || t.v7_relativeSplatPath), t && (t.v7_fetcherPersist, t.v7_normalizeFormMethod, t.v7_partialHydration, t.v7_skipActionErrorRevalidation);
  }
  function wv(e) {
    return ov(e.context);
  }
  function Sv(e) {
    let { basename: t = "/", children: n = null, location: r, navigationType: l = Ue.Pop, navigator: a, static: o = false, future: i } = e;
    ho() && te(false);
    let s = t.replace(/^\/*/, "/"), c = g.useMemo(() => ({
      basename: s,
      navigator: a,
      static: o,
      future: Ya({
        v7_relativeSplatPath: false
      }, i)
    }), [
      s,
      i,
      a,
      o
    ]);
    typeof r == "string" && (r = Pn(r));
    let { pathname: m = "/", search: d = "", hash: p = "", state: x = null, key: w = "default" } = r, k = g.useMemo(() => {
      let R = Ll(m, s);
      return R == null ? null : {
        location: {
          pathname: R,
          search: d,
          hash: p,
          state: x,
          key: w
        },
        navigationType: l
      };
    }, [
      s,
      m,
      d,
      p,
      x,
      w,
      l
    ]);
    return k == null ? null : g.createElement(po.Provider, {
      value: c
    }, g.createElement(Ki.Provider, {
      children: n,
      value: k
    }));
  }
  new Promise(() => {
  });
  function kv(e) {
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
  function Ga() {
    return Ga = Object.assign ? Object.assign.bind() : function(e) {
      for (var t = 1; t < arguments.length; t++) {
        var n = arguments[t];
        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
      }
      return e;
    }, Ga.apply(this, arguments);
  }
  const Nv = "6";
  try {
    window.__reactRouterVersion = Nv;
  } catch {
  }
  function jv(e, t) {
    return zg({
      basename: void 0,
      future: Ga({}, void 0, {
        v7_prependBasename: true
      }),
      history: sg({
        window: void 0
      }),
      hydrationData: Ev(),
      routes: e,
      mapRouteProperties: kv,
      dataStrategy: void 0,
      patchRoutesOnNavigation: void 0,
      window: void 0
    }).initialize();
  }
  function Ev() {
    var e;
    let t = (e = window) == null ? void 0 : e.__staticRouterHydrationData;
    return t && t.errors && (t = Ga({}, t, {
      errors: Cv(t.errors)
    })), t;
  }
  function Cv(e) {
    if (!e) return null;
    let t = Object.entries(e), n = {};
    for (let [r, l] of t) if (l && l.__type === "RouteErrorResponse") n[r] = new Ka(l.status, l.statusText, l.data, l.internal === true);
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
  const _v = g.createContext({
    isTransitioning: false
  }), Rv = g.createContext(/* @__PURE__ */ new Map()), bv = "startTransition", Dc = Ym[bv], Tv = "flushSync", Lc = og[Tv];
  function Pv(e) {
    Dc ? Dc(e) : e();
  }
  function Kr(e) {
    Lc ? Lc(e) : e();
  }
  class Mv {
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
  function Dv(e) {
    let { fallbackElement: t, router: n, future: r } = e, [l, a] = g.useState(n.state), [o, i] = g.useState(), [s, c] = g.useState({
      isTransitioning: false
    }), [m, d] = g.useState(), [p, x] = g.useState(), [w, k] = g.useState(), R = g.useRef(/* @__PURE__ */ new Map()), { v7_startTransition: h } = r || {}, f = g.useCallback((j) => {
      h ? Pv(j) : j();
    }, [
      h
    ]), v = g.useCallback((j, z) => {
      let { deletedFetchers: D, flushSync: H, viewTransitionOpts: K } = z;
      j.fetchers.forEach((le, je) => {
        le.data !== void 0 && R.current.set(je, le.data);
      }), D.forEach((le) => R.current.delete(le));
      let se = n.window == null || n.window.document == null || typeof n.window.document.startViewTransition != "function";
      if (!K || se) {
        H ? Kr(() => a(j)) : f(() => a(j));
        return;
      }
      if (H) {
        Kr(() => {
          p && (m && m.resolve(), p.skipTransition()), c({
            isTransitioning: true,
            flushSync: true,
            currentLocation: K.currentLocation,
            nextLocation: K.nextLocation
          });
        });
        let le = n.window.document.startViewTransition(() => {
          Kr(() => a(j));
        });
        le.finished.finally(() => {
          Kr(() => {
            d(void 0), x(void 0), i(void 0), c({
              isTransitioning: false
            });
          });
        }), Kr(() => x(le));
        return;
      }
      p ? (m && m.resolve(), p.skipTransition(), k({
        state: j,
        currentLocation: K.currentLocation,
        nextLocation: K.nextLocation
      })) : (i(j), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: K.currentLocation,
        nextLocation: K.nextLocation
      }));
    }, [
      n.window,
      p,
      m,
      R,
      f
    ]);
    g.useLayoutEffect(() => n.subscribe(v), [
      n,
      v
    ]), g.useEffect(() => {
      s.isTransitioning && !s.flushSync && d(new Mv());
    }, [
      s
    ]), g.useEffect(() => {
      if (m && o && n.window) {
        let j = o, z = m.promise, D = n.window.document.startViewTransition(async () => {
          f(() => a(j)), await z;
        });
        D.finished.finally(() => {
          d(void 0), x(void 0), i(void 0), c({
            isTransitioning: false
          });
        }), x(D);
      }
    }, [
      f,
      o,
      m,
      n.window
    ]), g.useEffect(() => {
      m && o && l.location.key === o.location.key && m.resolve();
    }, [
      m,
      p,
      l.location,
      o
    ]), g.useEffect(() => {
      !s.isTransitioning && w && (i(w.state), c({
        isTransitioning: true,
        flushSync: false,
        currentLocation: w.currentLocation,
        nextLocation: w.nextLocation
      }), k(void 0));
    }, [
      s.isTransitioning,
      w
    ]), g.useEffect(() => {
    }, []);
    let E = g.useMemo(() => ({
      createHref: n.createHref,
      encodeLocation: n.encodeLocation,
      go: (j) => n.navigate(j),
      push: (j, z, D) => n.navigate(j, {
        state: z,
        preventScrollReset: D == null ? void 0 : D.preventScrollReset
      }),
      replace: (j, z, D) => n.navigate(j, {
        replace: true,
        state: z,
        preventScrollReset: D == null ? void 0 : D.preventScrollReset
      })
    }), [
      n
    ]), _ = n.basename || "/", b = g.useMemo(() => ({
      router: n,
      navigator: E,
      static: false,
      basename: _
    }), [
      n,
      E,
      _
    ]), S = g.useMemo(() => ({
      v7_relativeSplatPath: n.future.v7_relativeSplatPath
    }), [
      n.future.v7_relativeSplatPath
    ]);
    return g.useEffect(() => yv(r, n.future), [
      r,
      n.future
    ]), g.createElement(g.Fragment, null, g.createElement(mo.Provider, {
      value: b
    }, g.createElement(hm.Provider, {
      value: l
    }, g.createElement(Rv.Provider, {
      value: R.current
    }, g.createElement(_v.Provider, {
      value: s
    }, g.createElement(Sv, {
      basename: _,
      location: l.location,
      navigationType: l.historyAction,
      navigator: E,
      future: S
    }, l.initialized || n.future.v7_partialHydration ? g.createElement(Lv, {
      routes: n.routes,
      future: n.future,
      state: l
    }) : t))))), null);
  }
  const Lv = g.memo(Ov);
  function Ov(e) {
    let { routes: t, future: n, state: r } = e;
    return sv(t, void 0, r, n);
  }
  var Oc;
  (function(e) {
    e.UseScrollRestoration = "useScrollRestoration", e.UseSubmit = "useSubmit", e.UseSubmitFetcher = "useSubmitFetcher", e.UseFetcher = "useFetcher", e.useViewTransitionState = "useViewTransitionState";
  })(Oc || (Oc = {}));
  var Ac;
  (function(e) {
    e.UseFetcher = "useFetcher", e.UseFetchers = "useFetchers", e.UseScrollRestoration = "useScrollRestoration";
  })(Ac || (Ac = {}));
  const Av = "https://100.107.132.16:30000";
  function Iv() {
    const e = window.location;
    return e.protocol === "file:" || e.protocol === "capacitor:" || e.protocol === "ionic:" || e.hostname === "localhost" || e.hostname === "127.0.0.1" || e.hostname === "";
  }
  function ce() {
    const e = window.location;
    return !Iv() && (e.protocol === "http:" || e.protocol === "https:") ? `${e.protocol}//${e.host}` : Av;
  }
  function zv(e) {
    const t = new URL(ce());
    return t.protocol = t.protocol === "https:" ? "wss:" : "ws:", `${t.protocol}//${t.host}${e}`;
  }
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
  ], Fv = () => {
    const e = go(), [t, n] = g.useState(/* @__PURE__ */ new Date()), [r, l] = g.useState(null);
    return g.useEffect(() => {
      const a = setInterval(() => n(/* @__PURE__ */ new Date()), 1e3);
      return () => clearInterval(a);
    }, []), g.useEffect(() => {
      const a = () => {
        fetch(`${ce()}/health`).then((i) => i.json()).then(l).catch(() => l(null));
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
  function Sm() {
    return ce();
  }
  const Kt = {
    index: "maude-conversations",
    messages: (e) => `maude-conv-msgs:${e}`,
    active: "maude-active-conv"
  };
  async function km(e) {
    try {
      const t = await fetch(`${Sm()}${e}`);
      return t.ok ? await t.json() : null;
    } catch {
      return null;
    }
  }
  function Gi(e, t) {
    fetch(`${Sm()}${e}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(t)
    }).catch(() => {
    });
  }
  function Nm() {
    try {
      const e = localStorage.getItem(Kt.index);
      return e ? JSON.parse(e) : [];
    } catch {
      return [];
    }
  }
  async function $v() {
    const e = await km("/api/conversations");
    return e && e.length > 0 ? (localStorage.setItem(Kt.index, JSON.stringify(e)), e) : Nm();
  }
  function Bv(e) {
    localStorage.setItem(Kt.index, JSON.stringify(e)), Gi("/api/conversations", e);
  }
  function Ks(e) {
    try {
      const t = localStorage.getItem(Kt.messages(e));
      return t ? JSON.parse(t) : [];
    } catch {
      return [];
    }
  }
  async function Vv(e) {
    const t = await km(`/api/conversations/${e}/messages`);
    return t && t.length > 0 ? (localStorage.setItem(Kt.messages(e), JSON.stringify(t)), t) : Ks(e);
  }
  function jm(e, t) {
    localStorage.setItem(Kt.messages(e), JSON.stringify(t)), Gi(`/api/conversations/${e}/messages`, t);
  }
  function Wv(e) {
    localStorage.removeItem(Kt.messages(e)), Gi(`/api/conversations/${e}/delete`, {});
  }
  function Hv() {
    return localStorage.getItem(Kt.active);
  }
  function oa(e) {
    e === null ? localStorage.removeItem(Kt.active) : localStorage.setItem(Kt.active, e);
  }
  const Ys = () => typeof crypto.randomUUID == "function" ? crypto.randomUUID() : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (e) => {
    const t = Math.random() * 16 | 0;
    return (e === "x" ? t : t & 3 | 8).toString(16);
  }), Qv = /iPad|iPhone|iPod/.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1, Kv = {
    "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
    "nvidia/nemotron-3-nano-30b-a3b": "nemotron-a3b",
    "nemotron-nano": "nemotron-a3b",
    a3b: "nemotron-a3b",
    "codex-cli": "codex"
  };
  function sa(e) {
    const t = (e || "").trim();
    return !t || t === "claude-opus-4-20250514" ? "nemotron-super" : Kv[t] || t;
  }
  let lr = null;
  function Yv() {
    return lr && Date.now() - lr.ts < 3e5 ? Promise.resolve(lr) : navigator.geolocation ? new Promise((e) => {
      navigator.geolocation.getCurrentPosition((t) => {
        lr = {
          lat: t.coords.latitude,
          lng: t.coords.longitude,
          accuracy: t.coords.accuracy,
          ts: Date.now()
        }, e(lr);
      }, () => e(lr), {
        timeout: 5e3,
        maximumAge: 3e5
      });
    }) : Promise.resolve(null);
  }
  const Gv = `You are MAUDE \u2014 a local AI assistant running on Matt's DGX Spark, handling tasks that benefit from local execution, privacy, or when cloud access isn't available.

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
  function Xv(e) {
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
  function Go(e, t, n, r) {
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
      const l = e.args && e.args !== "{}" ? Xv(e.args) : void 0;
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
  function Jv(e = null) {
    const [t, n] = g.useState(() => e ? Ks(e) : []), [r, l] = g.useState(false), [a, o] = g.useState(() => {
      const v = localStorage.getItem("maude-model"), E = sa(v);
      return v !== E && localStorage.setItem("maude-model", E), E;
    }), [i, s] = g.useState(() => localStorage.getItem("maude-autoroute") === "true"), c = g.useCallback((v) => {
      const E = sa(v);
      localStorage.setItem("maude-model", E), o(E);
    }, []), m = g.useCallback((v) => {
      localStorage.setItem("maude-autoroute", String(v)), s(v);
    }, []), d = g.useRef(a);
    d.current = a;
    const p = g.useRef(null), x = g.useRef(e), w = g.useRef(""), k = g.useRef(0);
    x.current = e, g.useEffect(() => {
      if (!e) {
        n([]);
        return;
      }
      n(Ks(e)), Vv(e).then((v) => {
        v.length > 0 && n(v);
      });
    }, [
      e
    ]), g.useEffect(() => {
      x.current && t.length > 0 && jm(x.current, t);
    }, [
      t
    ]);
    const R = g.useCallback(async (v, E) => {
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
      const b = v || (_ ? "What do you see in this image?" : ""), S = {
        id: Ys(),
        role: "user",
        content: b,
        imageUrls: _ ? E : void 0,
        timestamp: Date.now()
      };
      n((K) => [
        ...K,
        S
      ]), l(true);
      const j = d.current, z = {
        id: Ys(),
        role: "assistant",
        content: "",
        model: j,
        timestamp: Date.now()
      };
      n((K) => [
        ...K,
        z
      ]);
      const D = new AbortController();
      p.current = D;
      let H = "";
      try {
        const K = t.filter((J) => J.role !== "system").slice(-20).map((J) => ({
          role: J.role,
          content: J.content
        }));
        let se = b;
        if (_) {
          const J = E.map((Te) => `/home/mboard76/nvidia-workbench/terminal-llm/shared/${Te.split("/").pop()}`);
          if (J.length === 1) se = `[Image attached: ${J[0]} \u2014 analyze it with view_image tool]

${b}`;
          else {
            const Te = J.map((me, pe) => `  ${pe + 1}. ${me}`).join(`
`);
            se = `[${J.length} images attached \u2014 analyze each with view_image tool:
${Te}]

${b}`;
          }
        }
        const le = await Yv(), je = {
          model: j,
          messages: [
            {
              role: "system",
              content: Gv
            },
            ...K,
            {
              role: "user",
              content: se
            }
          ],
          stream: true,
          max_tokens: 4096,
          temperature: 0.7
        };
        if (le && (je.location = {
          lat: le.lat,
          lng: le.lng,
          accuracy: le.accuracy
        }), Qv) {
          const J = await fetch(`${ce()}/api/chat/create`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(je),
            signal: D.signal
          });
          if (!J.ok) {
            const me = await J.text();
            n((pe) => pe.map(($) => $.id === z.id ? {
              ...$,
              content: `Error: ${J.status} \u2014 ${me}`
            } : $)), l(false);
            return;
          }
          const { sid: Te } = await J.json();
          await new Promise((me) => {
            let pe = null, $ = 0, Y = false, Z = "";
            const fe = {
              tools: [],
              promptTokens: 0,
              completionTokens: 0,
              cacheReadTokens: 0,
              cacheCreateTokens: 0,
              elapsed: 0
            }, Ve = [], Oe = () => {
              if (Y) return;
              Y = true, pe == null ? void 0 : pe.close(), k.current && (cancelAnimationFrame(k.current), k.current = 0);
              const he = {
                content: Z
              };
              H && (he.model = H), (fe.promptTokens || fe.tools.length || fe.route) && (he.trace = {
                ...fe
              }), Ve.length && (he.toolSteps = Ve.map((ne) => ({
                ...ne
              }))), n((ne) => ne.map((Ie) => Ie.id === z.id ? {
                ...Ie,
                ...he
              } : Ie)), w.current = "", l(false), p.current = null, me();
            };
            D.signal.addEventListener("abort", () => Oe());
            const tt = (he) => {
              const ne = Number(he.lastEventId);
              $ = Number.isFinite(ne) ? ne + 1 : $ + 1;
            }, ot = () => {
              k.current || (k.current = requestAnimationFrame(() => {
                const he = w.current, ne = {
                  ...fe,
                  tools: [
                    ...fe.tools
                  ]
                }, Ie = Ve.map((gt) => ({
                  ...gt
                }));
                n((gt) => gt.map((Ft) => Ft.id === z.id ? {
                  ...Ft,
                  content: he,
                  trace: ne,
                  toolSteps: Ie,
                  ...H && {
                    model: H
                  }
                } : Ft)), k.current = 0;
              }));
            };
            let Ae = false;
            const ht = (he) => {
              var _a3, _b2, _c3, _d2;
              if (tt(he), he.data === "[DONE]") {
                Oe();
                return;
              }
              try {
                const ne = JSON.parse(he.data);
                ne.model && !H && (H = sa(ne.model));
                const Ie = (_b2 = (_a3 = ne.choices) == null ? void 0 : _a3[0]) == null ? void 0 : _b2.delta;
                (Ie == null ? void 0 : Ie.reasoning_content) ? Ae || (Z += `*Thinking...*

`, Ae = true) : (Ie == null ? void 0 : Ie.content) && (Ae && (Z = Z.replace(`*Thinking...*

`, ""), Ae = false), Z += Ie.content), w.current = Z, ot(), ((_d2 = (_c3 = ne.choices) == null ? void 0 : _c3[0]) == null ? void 0 : _d2.finish_reason) === "stop" && Oe();
              } catch {
              }
            }, st = (he) => {
              tt(he);
              try {
                const ne = JSON.parse(he.data);
                Go(ne, fe, Ve, (gt) => {
                  Z += `

*Error: ${gt}*`, w.current = Z;
                }) && (ne.type !== "error" && (w.current = Z), ot());
              } catch {
              }
            }, an = () => {
              Y || D.signal.aborted || (pe == null ? void 0 : pe.close(), pe = new EventSource(`${ce()}/api/chat/stream?sid=${Te}&offset=${$}`), pe.onmessage = ht, pe.addEventListener("trace", st), pe.onerror = () => {
                pe == null ? void 0 : pe.close(), !Y && !D.signal.aborted && window.setTimeout(an, document.visibilityState === "visible" ? 1e3 : 3e3);
              });
            };
            an();
          });
          return;
        }
        const Qe = await fetch(`${ce()}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(je),
          signal: D.signal
        });
        if (!Qe.ok) {
          const J = await Qe.text();
          n((Te) => Te.map((me) => me.id === z.id ? {
            ...me,
            content: `Error: ${Qe.status} \u2014 ${J}`
          } : me)), l(false);
          return;
        }
        const pt = (_a2 = Qe.body) == null ? void 0 : _a2.getReader();
        if (!pt) {
          l(false);
          return;
        }
        const M = new TextDecoder();
        let V = "", F = "", ee = "", X = false;
        const be = {
          tools: [],
          promptTokens: 0,
          completionTokens: 0,
          cacheReadTokens: 0,
          cacheCreateTokens: 0,
          elapsed: 0
        }, Ee = [], ge = () => {
          k.current || (k.current = requestAnimationFrame(() => {
            const J = w.current, Te = {
              ...be,
              tools: [
                ...be.tools
              ]
            }, me = Ee.map((pe) => ({
              ...pe
            }));
            n((pe) => pe.map(($) => $.id === z.id ? {
              ...$,
              content: J,
              trace: Te,
              toolSteps: me,
              ...H && {
                model: H
              }
            } : $)), k.current = 0;
          }));
        };
        for (; ; ) {
          const { done: J, value: Te } = await pt.read();
          if (J) break;
          V += M.decode(Te, {
            stream: true
          });
          const me = V.split(`
`);
          V = me.pop() || "";
          for (const pe of me) {
            const $ = pe.trim();
            if (!$) continue;
            if ($.startsWith(": trace ")) {
              try {
                const Z = JSON.parse($.slice(8));
                Go(Z, be, Ee, (Ve) => {
                  F += `

*Error: ${Ve}*`, w.current = F;
                }) && (Z.type !== "error" && (w.current = F), ge());
              } catch {
              }
              continue;
            }
            if ($.startsWith("event: ")) {
              ee = $.slice(7);
              continue;
            }
            if (!$.startsWith("data: ")) continue;
            const Y = $.slice(6);
            if (Y !== "[DONE]") {
              if (ee === "trace") {
                ee = "";
                try {
                  const Z = JSON.parse(Y);
                  Go(Z, be, Ee, (Ve) => {
                    F += `

*Error: ${Ve}*`, w.current = F;
                  }) && (Z.type !== "error" && (w.current = F), ge());
                } catch {
                }
                continue;
              }
              ee = "";
              try {
                const Z = JSON.parse(Y);
                Z.model && !H && (H = sa(Z.model));
                const fe = (_c2 = (_b = Z.choices) == null ? void 0 : _b[0]) == null ? void 0 : _c2.delta;
                (fe == null ? void 0 : fe.reasoning_content) ? X || (F += `*Thinking...*

`, X = true) : (fe == null ? void 0 : fe.content) && (X && (F = F.replace(`*Thinking...*

`, ""), X = false), F += fe.content), ((fe == null ? void 0 : fe.reasoning_content) || (fe == null ? void 0 : fe.content)) && (w.current = F, ge());
              } catch {
              }
            }
          }
        }
        const Se = {};
        H && (Se.model = H), (be.promptTokens || be.tools.length || be.route) && (Se.trace = {
          ...be
        }), Ee.length && (Se.toolSteps = Ee.map((J) => ({
          ...J
        }))), Object.keys(Se).length && n((J) => J.map((Te) => Te.id === z.id ? {
          ...Te,
          ...Se
        } : Te));
      } catch (K) {
        K instanceof Error && K.name !== "AbortError" && n((se) => se.map((le) => le.id === z.id ? {
          ...le,
          content: `Error: ${K.message}`
        } : le));
      } finally {
        if (k.current && (cancelAnimationFrame(k.current), k.current = 0), w.current) {
          const K = w.current, se = H || void 0;
          n((le) => le.map((je) => je.id === z.id ? {
            ...je,
            content: K,
            ...se && {
              model: se
            }
          } : je)), w.current = "";
        }
        l(false), p.current = null;
      }
    }, [
      t,
      r,
      a,
      i,
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
      autoRoute: i,
      setAutoRoute: m,
      sendMessage: R,
      stopStreaming: h,
      clearMessages: f
    };
  }
  function Ic(e) {
    const t = e.trim().replace(/\s+/g, " ");
    return t.length <= 40 ? t : t.slice(0, 37) + "...";
  }
  function Zv() {
    const [e, t] = g.useState(Nm), [n, r] = g.useState(Hv);
    g.useEffect(() => {
      $v().then((d) => {
        d.length > 0 && t(d);
      });
    }, []);
    const l = g.useCallback((d) => {
      const p = [
        ...d
      ].sort((x, w) => w.updatedAt - x.updatedAt);
      t(p), Bv(p);
    }, []), a = g.useCallback((d, p) => {
      const x = Ys(), w = Date.now(), R = [
        {
          id: x,
          title: Ic(d),
          createdAt: w,
          updatedAt: w,
          model: p
        },
        ...e
      ];
      return l(R), r(x), oa(x), x;
    }, [
      e,
      l
    ]), o = g.useCallback((d) => {
      r(d), oa(d);
    }, []), i = g.useCallback((d) => {
      const p = e.filter((x) => x.id !== d);
      if (l(p), Wv(d), n === d) {
        const x = p.length > 0 ? p[0].id : null;
        r(x), oa(x);
      }
    }, [
      e,
      n,
      l
    ]), s = g.useCallback((d, p) => {
      const x = e.map((w) => w.id === d ? {
        ...w,
        title: Ic(p)
      } : w);
      l(x);
    }, [
      e,
      l
    ]), c = g.useCallback((d) => {
      const p = e.map((x) => x.id === d ? {
        ...x,
        updatedAt: Date.now()
      } : x);
      l(p);
    }, [
      e,
      l
    ]), m = g.useCallback(() => {
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
  function qv(e, t) {
    const [n, r] = g.useState(0), l = g.useRef(0), a = g.useRef(false);
    return t && (a.current = true), g.useEffect(() => {
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
  function ex(e) {
    const t = ce();
    return e.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (r, l, a) => `<img src="${a.startsWith("/") ? `${t}${a}` : a}" alt="${l}" style="max-width:100%; max-height:50vh; border-radius:8px; margin:8px 0; object-fit:contain;" loading="lazy" onerror="this.style.display='none'" />`).replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-blue-400 underline">$1</a>').replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="my-2 rounded-lg bg-[#0d1117] p-3 text-sm overflow-x-auto"><code class="text-green-300">$2</code></pre>').replace(/`([^`]+)`/g, '<code class="rounded bg-[#0d1117] px-1.5 py-0.5 text-sm text-orange-300">$1</code>').replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>").replace(/^- (.+)$/gm, '<li class="ml-4 list-disc">$1</li>').replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>').replace(/\n/g, "<br/>");
  }
  const tx = {
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
  function nx(e) {
    const t = /* @__PURE__ */ new Map();
    for (const r of e.filter((l) => !l.kind || l.kind === "tool")) t.set(r.name, (t.get(r.name) || 0) + 1);
    const n = [];
    for (const [r, l] of t) {
      const a = tx[r] || r.replace(/_/g, " ");
      if (l > 1) {
        const o = a.replace(/(?:a |an )(\w+)$/, `${l} $1s`);
        n.push(o === a ? `${a} x${l}` : o);
      } else n.push(a);
    }
    return n.length <= 2 ? n.join(" and ") : n.slice(0, -1).join(", ") + ", and " + n[n.length - 1];
  }
  const rx = {
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
  }, lx = ({ steps: e, streaming: t, contentStarted: n }) => {
    if (!e.length) return null;
    const r = e.some((a) => a.status === "running"), l = nx(e);
    return u.jsxs("div", {
      className: "mb-2 space-y-1",
      children: [
        e.map((a, o) => {
          const i = rx[a.name] || "\u2699\uFE0F", s = a.status === "running", c = a.status === "error", m = s ? "border-cyan-400/40" : c ? "border-red-400/40" : "border-cyan-500/20";
          return u.jsxs("div", {
            className: `border-l-2 ${m} pl-2.5 py-0.5 transition-all duration-300`,
            style: {
              animation: t ? "fadeSlideIn 0.3s ease-out" : "none"
            },
            children: [
              u.jsxs("div", {
                className: "flex items-center gap-1.5",
                children: [
                  s && t ? u.jsx("span", {
                    className: "inline-block h-3 w-3 animate-spin rounded-full border-2 border-cyan-300/30 border-t-cyan-300"
                  }) : u.jsx("span", {
                    className: "text-[11px]",
                    children: i
                  }),
                  u.jsx("span", {
                    className: "text-[11px] font-semibold text-cyan-300",
                    children: a.task || a.name
                  }),
                  s && u.jsx("span", {
                    className: "animate-pulse text-[10px] font-medium text-cyan-300",
                    children: "still working"
                  }),
                  a.elapsed !== void 0 && u.jsxs("span", {
                    className: "ml-auto font-mono text-[10px] text-maude-muted",
                    children: [
                      a.elapsed.toFixed(1),
                      "s"
                    ]
                  })
                ]
              }),
              a.task && (!a.kind || a.kind === "tool") && u.jsx("div", {
                className: "truncate font-mono text-[10px] leading-tight text-maude-muted",
                children: a.name
              }),
              a.args && u.jsx("div", {
                className: "truncate font-mono text-[10px] leading-tight text-maude-muted",
                children: a.args
              }),
              a.result && u.jsxs("div", {
                className: `truncate font-mono text-[10px] leading-tight ${c ? "text-red-400" : "text-green-400/80"}`,
                children: [
                  c ? "\u2717 " : "\u2713 ",
                  a.result
                ]
              })
            ]
          }, `${a.name}-${o}`);
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
        !t && l && u.jsx("div", {
          className: "mt-1 border-l-2 border-green-400/30 py-0.5 pl-2.5",
          children: u.jsxs("span", {
            className: "text-[10px] text-green-400/70",
            children: [
              "\u2713 ",
              l,
              (() => {
                const a = e.reduce((o, i) => o + (i.elapsed || 0), 0);
                return a > 0 ? ` \u2014 ${a.toFixed(1)}s` : "";
              })()
            ]
          })
        })
      ]
    });
  }, ax = ({ trace: e }) => {
    const t = e.promptTokens + e.cacheReadTokens + e.cacheCreateTokens;
    if (!t && !e.tools.length && !e.route) return null;
    const n = t > 0 ? Math.round(e.cacheReadTokens / t * 100) : 0;
    return u.jsxs("div", {
      className: "mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-maude-muted",
      children: [
        e.route && u.jsx("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5 text-cyan-300",
          children: e.route.requestedModel && e.route.requestedModel !== e.route.resolvedModel ? `${e.route.requestedModel} -> ${e.route.resolvedModel}` : e.route.resolvedModel || e.route.requestedModel
        }),
        e.tools.length > 0 && u.jsxs("span", {
          className: "rounded bg-maude-bg px-1.5 py-0.5",
          children: [
            e.tools.length,
            " tool",
            e.tools.length > 1 ? "s" : ""
          ]
        }),
        t + e.completionTokens > 0 && u.jsxs("span", {
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
  }, ox = {
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
  }, sx = ({ message: e, animate: t }) => {
    const n = e.role === "user", r = qv(e.content, !!t), l = !n && e.toolSteps && e.toolSteps.length > 0, a = !e.content && !n && !l;
    return u.jsx("div", {
      className: `flex ${n ? "justify-end" : "justify-start"} mb-3`,
      children: u.jsxs("div", {
        className: `max-w-[85%] rounded-2xl px-4 py-3 ${n ? "fire-bg text-white" : "bg-maude-surface text-maude-text"}`,
        children: [
          e.model && !n && u.jsx("div", {
            className: "mb-1 text-[10px] font-medium tracking-wider text-maude-muted",
            children: ox[e.model] || e.model
          }),
          (() => {
            const o = e.imageUrls || (e.imageUrl ? [
              e.imageUrl
            ] : []);
            if (!o.length) return null;
            const i = ce();
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
          l && u.jsx(lx, {
            steps: e.toolSteps,
            streaming: !!t,
            contentStarted: !!e.content
          }),
          r && u.jsx("div", {
            className: "break-words text-sm leading-relaxed",
            dangerouslySetInnerHTML: {
              __html: ex(r)
            }
          }),
          !n && e.trace && u.jsx(ax, {
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
  }, ix = ({ onSend: e, isStreaming: t, onStop: n }) => {
    const [r, l] = g.useState(""), [a, o] = g.useState([]), [i, s] = g.useState(false), c = g.useRef(null), m = g.useRef(null), d = g.useRef(null);
    g.useEffect(() => {
      var _a2;
      (_a2 = c.current) == null ? void 0 : _a2.focus();
    }, []);
    const p = () => {
      (a.length > 0 || r.trim()) && (e(r.trim(), a.length > 0 ? a : void 0), l(""), o([]), c.current && (c.current.style.height = "44px"));
    }, x = (f) => {
      f.key === "Enter" && !f.shiftKey && (f.preventDefault(), p());
    }, w = () => {
      c.current && (c.current.style.height = "44px", c.current.style.height = Math.min(c.current.scrollHeight, 120) + "px");
    }, k = async (f) => {
      const v = f.target.files;
      if (!(!v || v.length === 0)) {
        s(true);
        try {
          const E = [];
          for (const _ of Array.from(v)) {
            const b = `camera_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.jpg`;
            (await fetch(`${ce()}/share/${encodeURIComponent(b)}`, {
              method: "POST",
              body: _
            })).ok && E.push(`/download/${b}`);
          }
          E.length > 0 && o((_) => [
            ..._,
            ...E
          ]);
        } catch {
        } finally {
          s(false), m.current && (m.current.value = ""), d.current && (d.current.value = "");
        }
      }
    }, R = (f) => {
      o((v) => v.filter((E, _) => _ !== f));
    }, h = a.length > 0 || r.trim();
    return u.jsxs("div", {
      className: "border-t border-maude-border bg-maude-surface p-3",
      children: [
        a.length > 0 && u.jsx("div", {
          className: "mb-2 flex gap-2 overflow-x-auto",
          children: a.map((f, v) => u.jsxs("div", {
            className: "relative shrink-0",
            children: [
              u.jsx("img", {
                src: `${ce()}${f}`,
                alt: `Pending upload ${v + 1}`,
                className: "h-20 w-20 rounded-lg object-cover"
              }),
              u.jsx("button", {
                onClick: () => R(v),
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
              onChange: k,
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
              onChange: k,
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
              onClick: p,
              disabled: !h,
              className: "flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-xl fire-bg text-white disabled:opacity-30",
              children: "\u2191"
            })
          ]
        })
      ]
    });
  }, Xo = [
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
  ], ux = {
    "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
    "nvidia/nemotron-3-nano-30b-a3b": "nemotron-a3b",
    "nemotron-nano": "nemotron-a3b",
    a3b: "nemotron-a3b",
    "codex-cli": "codex"
  }, cx = ({ currentModel: e, onSelect: t, autoRoute: n, onToggleAutoRoute: r }) => {
    const [l, a] = g.useState(false), o = ux[e] || e, i = Xo.find((s) => s.id === o) || Xo[0];
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
            i.label,
            n && u.jsx("span", {
              className: "text-[10px] text-maude-accent",
              children: "AUTO"
            })
          ]
        }),
        l && u.jsxs("div", {
          className: "absolute right-0 top-full z-50 mt-1 w-56 rounded-xl border border-maude-border bg-maude-surface p-2 shadow-xl",
          children: [
            Xo.map((s) => u.jsxs("button", {
              onClick: () => {
                t(s.id), a(false);
              },
              className: `flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${s.id === o ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
              children: [
                u.jsx("span", {
                  children: s.label
                }),
                u.jsx("span", {
                  className: "text-xs text-maude-muted",
                  children: s.desc
                })
              ]
            }, s.id)),
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
  function dx(e) {
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
  const fx = ({ open: e, onClose: t, conversations: n, activeId: r, onSelect: l, onDelete: a, onNewChat: o }) => {
    const i = dx(n), [s, c] = g.useState(false);
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
                          onClick: (p) => {
                            p.stopPropagation(), a(d.id);
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
  }, mx = ({ conversationId: e, onFirstMessage: t, onMessageSent: n, onOpenDrawer: r, onNewChat: l }) => {
    const a = go(), o = g.useRef(null), i = g.useRef(e), { messages: s, isStreaming: c, currentModel: m, setCurrentModel: d, autoRoute: p, setAutoRoute: x, sendMessage: w, stopStreaming: k } = Jv(e);
    g.useEffect(() => {
      o.current && (o.current.scrollTop = o.current.scrollHeight);
    }, [
      s
    ]), g.useEffect(() => {
      if (!c || !o.current) return;
      const h = setInterval(() => {
        o.current && (o.current.scrollTop = o.current.scrollHeight);
      }, 200);
      return () => clearInterval(h);
    }, [
      c
    ]);
    const R = g.useCallback((h, f) => {
      if (!i.current) {
        const v = h || ((f == null ? void 0 : f.length) ? "Image conversation" : "New chat"), E = t(v, m);
        i.current = E;
      }
      w(h, f), n();
    }, [
      w,
      t,
      n,
      m
    ]);
    return g.useEffect(() => {
      i.current && s.length > 0 && jm(i.current, s);
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
            u.jsx(cx, {
              currentModel: m,
              onSelect: d,
              autoRoute: p,
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
                  children: "Your local AI assistant. Ask me anything."
                }),
                u.jsx("div", {
                  className: "mt-4 flex flex-wrap justify-center gap-2",
                  children: [
                    "What can you do?",
                    "Write a Python script",
                    "Explain this code",
                    "System status"
                  ].map((h) => u.jsx("button", {
                    onClick: () => R(h),
                    className: "rounded-full border border-maude-border px-3 py-1.5 text-xs text-maude-muted transition-colors hover:border-maude-accent hover:text-maude-text",
                    children: h
                  }, h))
                })
              ]
            }),
            s.map((h, f) => u.jsx(sx, {
              message: h,
              animate: c && f === s.length - 1
            }, h.id))
          ]
        }),
        u.jsx(ix, {
          onSend: (h, f) => R(h, f),
          isStreaming: c,
          onStop: k
        })
      ]
    });
  }, px = () => {
    const [e, t] = g.useState(false), { conversations: n, activeId: r, createConversation: l, switchConversation: a, deleteConversation: o, touchConversation: i, startNewChat: s } = Zv(), c = g.useCallback((d, p) => l(d, p), [
      l
    ]), m = g.useCallback(() => {
      r && i(r);
    }, [
      r,
      i
    ]);
    return u.jsxs("div", {
      className: "flex h-full flex-col",
      children: [
        u.jsx(mx, {
          conversationId: r,
          onFirstMessage: c,
          onMessageSent: m,
          onOpenDrawer: () => t(true),
          onNewChat: s
        }, r || "new"),
        u.jsx(fx, {
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
  }, hx = "modulepreload", gx = function(e) {
    return "/" + e;
  }, zc = {}, il = function(t, n, r) {
    let l = Promise.resolve();
    if (n && n.length > 0) {
      document.getElementsByTagName("link");
      const o = document.querySelector("meta[property=csp-nonce]"), i = (o == null ? void 0 : o.nonce) || (o == null ? void 0 : o.getAttribute("nonce"));
      l = Promise.allSettled(n.map((s) => {
        if (s = gx(s), s in zc) return;
        zc[s] = true;
        const c = s.endsWith(".css"), m = c ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${s}"]${m}`)) return;
        const d = document.createElement("link");
        if (d.rel = c ? "stylesheet" : hx, c || (d.as = "script"), d.crossOrigin = "", d.href = s, i && d.setAttribute("nonce", i), document.head.appendChild(d), c) return new Promise((p, x) => {
          d.addEventListener("load", p), d.addEventListener("error", () => x(new Error(`Unable to preload CSS for ${s}`)));
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
  }, vx = {
    0: 0
  }, xx = {
    0: 0
  }, yx = {
    start: 0,
    endTurn: 1,
    pause: 2,
    restart: 3
  }, wx = (e) => {
    switch (e.type) {
      case "handshake":
        return new Uint8Array([
          0,
          vx[e.version],
          xx[e.model]
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
          yx[e.action]
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
  }, Sx = "You are MAUDE, a capable AI assistant with a warm Scottish accent. You are direct, competent, and quietly confident. Keep responses concise and natural for voice conversation. You run locally on Matt\u2019s DGX Spark workstation.", kx = "NATF2.pt";
  function Nx(e) {
    const t = zv("");
    let n = Sx;
    e && (n += `

--- Image Context ---
` + e);
    const r = new URLSearchParams({
      text_prompt: n
    });
    return `${t}/api/chat?${r}`;
  }
  const jx = `
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
  async function Ex(e, t) {
    const n = new Blob([
      jx
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
  const Uc = ({ analyser: e, active: t, color: n }) => {
    const r = g.useRef(null), l = g.useRef(0);
    return g.useEffect(() => {
      if (!e || !t || !r.current) return;
      const a = r.current, o = a.getContext("2d"), i = e.frequencyBinCount, s = new Uint8Array(i), c = () => {
        l.current = requestAnimationFrame(c), e.getByteTimeDomainData(s), o.clearRect(0, 0, a.width, a.height), o.lineWidth = 2, o.strokeStyle = n, o.beginPath();
        const m = a.width / i;
        let d = 0;
        for (let p = 0; p < i; p++) {
          const w = s[p] / 128 * a.height / 2;
          p === 0 ? o.moveTo(d, w) : o.lineTo(d, w), d += m;
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
  }, Cx = () => {
    const e = go(), [t, n] = g.useState("disconnected"), [r, l] = g.useState(""), [a, o] = g.useState(false), [i, s] = g.useState(0), [c, m] = g.useState(""), [d, p] = g.useState(""), [x, w] = g.useState(null), [k, R] = g.useState(null), [h, f] = g.useState(false), [v, E] = g.useState(false), _ = g.useRef(null), b = g.useRef(null), S = g.useRef(null), j = g.useRef(null), z = g.useRef(null), D = g.useRef(null), H = g.useRef(null), K = g.useRef(null), se = g.useRef(null), le = g.useRef(null), je = g.useRef(0), Qe = g.useRef(0), pt = g.useRef(0), M = g.useRef(0), V = g.useRef(0), F = g.useRef(0), ee = g.useRef(0), X = g.useCallback(async () => {
      m(""), l(""), s(0), je.current = 0;
      try {
        z.current || (z.current = new AudioContext({
          sampleRate: 48e3
        }));
        const $ = z.current;
        await $.resume();
        const Y = $.createBuffer(1, 1, $.sampleRate), Z = $.createBufferSource();
        Z.buffer = Y, Z.connect($.destination), Z.start(), pt.current = 0, M.current = 0, p(`ctx: ${$.state} ${$.sampleRate}Hz`), D.current || (D.current = await Ex($, (ht, st) => {
          st.underruns != null && (V.current = st.underruns), st.avail != null && (F.current = st.avail);
        }), D.current.connect($.destination)), D.current.reset(), V.current = 0;
        const fe = $.createAnalyser();
        D.current.connect(fe), K.current = fe;
        const Ve = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1
          }
        });
        le.current = Ve;
        const Oe = $.createAnalyser();
        $.createMediaStreamSource(Ve).connect(Oe), se.current = Oe;
        const ot = Nx(j.current ?? void 0);
        console.log("Connecting to voice server:", ot);
        const Ae = new WebSocket(ot);
        Ae.binaryType = "arraybuffer", _.current = Ae, n("connecting"), Ae.onopen = () => {
          console.log("voice server WS open, waiting for handshake");
        }, Ae.onmessage = (ht) => {
          var _a2;
          try {
            const st = new Uint8Array(ht.data), an = st[0];
            if (an === 0) console.log("voice server handshake received"), n("connected"), be(Ae, Ve, $), Qe.current = window.setInterval(() => {
              var _a3;
              je.current += 1, s(je.current);
              const he = ((_a3 = z.current) == null ? void 0 : _a3.state) ?? "?", ne = Math.round(F.current / 48);
              p(`dec:${M.current} buf:${ne}ms ur:${V.current}`);
            }, 1e3);
            else if (an === 2) {
              const he = new TextDecoder().decode(st.slice(1));
              he.includes("[Searching...]") ? o(true) : (he.includes("[Tool result:]") || he.includes("[Error:")) && o(false), l((ne) => ne + he);
            } else if (an === 3) {
              M.current++;
              const he = st.slice(1), ne = new Float32Array(he.buffer, he.byteOffset, he.byteLength / 4), Ie = new Float32Array(ne.length * 2);
              for (let gt = 0; gt < Ie.length; gt++) {
                const Ft = gt * 0.5, vt = Ft | 0, Pt = Math.min(vt + 1, ne.length - 1), Ol = Ft - vt;
                Ie[gt] = ne[vt] + (ne[Pt] - ne[vt]) * Ol;
              }
              (_a2 = D.current) == null ? void 0 : _a2.feedAudio(Ie);
            }
          } catch (st) {
            console.error("Message decode error:", st);
          }
        }, Ae.onclose = (ht) => {
          console.log("voice server WS closed:", ht.code, ht.reason), n("disconnected"), Ee(), clearInterval(Qe.current);
        }, Ae.onerror = (ht) => {
          console.error("voice server WS error:", ht), m("WebSocket connection failed. Is voice server running?"), n("disconnected");
        };
      } catch ($) {
        const Y = $ instanceof Error ? $.message : "Connection failed";
        console.error("Voice connect error:", Y), m(Y), n("disconnected");
      }
    }, []), be = g.useCallback(async ($, Y, Z) => {
      try {
        const fe = (await il(async () => {
          const { default: ot } = await import("./recorder.min-T7j-wk65.js").then((Ae) => Ae.r);
          return {
            default: ot
          };
        }, [])).default, Ve = (await il(async () => {
          const { default: ot } = await import("./encoderWorker.min-De-nC0Q0.js");
          return {
            default: ot
          };
        }, [])).default, Oe = Z.createMediaStreamSource(Y), tt = new fe({
          encoderPath: Ve,
          bufferLength: Math.round(960 * Z.sampleRate / 24e3),
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
        tt.ondataavailable = (ot) => {
          $.readyState === WebSocket.OPEN && $.send(wx({
            type: "audio",
            data: ot
          }));
        }, tt.onstart = () => {
          console.log("Opus recorder started");
        }, tt.start(), H.current = tt;
      } catch (fe) {
        console.error("Recorder start error:", fe), m("Failed to start microphone recording");
      }
    }, []), Ee = g.useCallback(() => {
      if (H.current) {
        try {
          H.current.stop();
        } catch {
        }
        H.current = null;
      }
      le.current && (le.current.getTracks().forEach(($) => $.stop()), le.current = null);
    }, []), ge = g.useCallback(() => {
      Ee(), clearInterval(Qe.current), clearInterval(ee.current), _.current && (_.current.close(), _.current = null), n("disconnected");
    }, [
      Ee
    ]), Se = g.useCallback(async ($) => {
      var _a2;
      const Y = (_a2 = $.target.files) == null ? void 0 : _a2[0];
      if (!Y) return;
      $.target.value = "";
      const Z = `voice_camera_${Date.now()}.jpg`, fe = ce(), Ve = URL.createObjectURL(Y);
      w(Ve), R(null), E(true);
      try {
        if (!(await fetch(`${fe}/share/${Z}`, {
          method: "POST",
          body: Y
        })).ok) throw new Error("Upload failed");
        E(false), f(true);
        const tt = await fetch(`${fe}/api/analyze-image`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            filename: Z,
            question: "Describe this image in detail. What do you see?"
          })
        });
        if (!tt.ok) throw new Error("Analysis failed");
        const Ae = (await tt.json()).analysis || "No analysis returned.";
        R(Ae), f(false), j.current = `The user shared an image (${Z}). Analysis: ${Ae}`, _.current && _.current.readyState === WebSocket.OPEN && (ge(), await new Promise((ht) => setTimeout(ht, 300)), X());
      } catch (Oe) {
        const tt = Oe instanceof Error ? Oe.message : "Image processing failed";
        m(tt), E(false), f(false);
      }
    }, [
      X,
      ge
    ]), J = g.useCallback(async () => {
      j.current = null, w(null), R(null), _.current && _.current.readyState === WebSocket.OPEN && (ge(), await new Promise(($) => setTimeout($, 300)), X());
    }, [
      X,
      ge
    ]);
    g.useEffect(() => () => {
      ge();
    }, []);
    const Te = ($) => {
      const Y = Math.floor($ / 60), Z = $ % 60;
      return `${Y}:${Z.toString().padStart(2, "0")}`;
    }, me = t === "connected", pe = t === "connecting";
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
                  className: `h-32 w-32 rounded-full border-4 ${me ? "animate-pulse border-maude-accent shadow-[0_0_30px_rgba(255,69,0,0.3)]" : pe ? "animate-spin border-maude-muted" : "border-maude-border"} flex items-center justify-center`,
                  children: u.jsx("span", {
                    className: "text-4xl",
                    children: me ? "\u{1F399}\uFE0F" : pe ? "\u23F3" : "\u{1F399}\uFE0F"
                  })
                }),
                u.jsx("span", {
                  className: "text-sm text-maude-muted",
                  children: me ? `Connected \u2022 ${Te(i)}` : pe ? "Connecting to MAUDE Voice..." : "Tap to start voice chat"
                })
              ]
            }),
            me && u.jsxs("div", {
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
                      children: u.jsx(Uc, {
                        analyser: K.current,
                        active: me,
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
                      children: u.jsx(Uc, {
                        analyser: se.current,
                        active: me,
                        color: "#888"
                      })
                    })
                  ]
                })
              ]
            }),
            me && u.jsxs("div", {
              className: "flex gap-3",
              children: [
                u.jsxs("button", {
                  onClick: () => {
                    var _a2;
                    return (_a2 = b.current) == null ? void 0 : _a2.click();
                  },
                  disabled: h || v,
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
                    return (_a2 = S.current) == null ? void 0 : _a2.click();
                  },
                  disabled: h || v,
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
              ref: b,
              type: "file",
              accept: "image/*",
              capture: "environment",
              onChange: Se,
              className: "hidden"
            }),
            u.jsx("input", {
              ref: S,
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
                v && u.jsx("p", {
                  className: "text-xs text-maude-muted",
                  children: "Uploading..."
                }),
                h && u.jsxs("div", {
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
                k && u.jsx("p", {
                  className: "text-xs leading-relaxed text-maude-text",
                  children: k
                }),
                k && u.jsx("button", {
                  onClick: J,
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
`).map(($, Y) => $.includes("[Searching...]") ? u.jsx("p", {
                    className: "my-1 text-xs italic text-maude-accent",
                    children: $
                  }, Y) : $.includes("[Tool result:]") ? u.jsx("p", {
                    className: "mt-2 mb-1 text-[10px] font-bold uppercase tracking-wider text-maude-accent",
                    children: $
                  }, Y) : $.includes("[Error:") ? u.jsx("p", {
                    className: "my-1 text-xs text-red-400",
                    children: $
                  }, Y) : u.jsxs("span", {
                    children: [
                      $,
                      Y < r.split(`
`).length - 1 ? `
` : ""
                    ]
                  }, Y))
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
              onClick: me || pe ? ge : X,
              className: `min-w-[200px] rounded-2xl px-8 py-4 text-base font-semibold text-white transition-all ${me ? "bg-red-600 hover:bg-red-700" : pe ? "bg-maude-muted" : "fire-bg hover:opacity-90"}`,
              disabled: pe,
              children: me ? "End Call" : pe ? "Connecting..." : "Start Voice Chat"
            }),
            u.jsxs("div", {
              className: "text-center text-[10px] text-maude-muted",
              children: [
                "Voice: ",
                (localStorage.getItem("maude-default-voice") || kx).replace(".pt", ""),
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
  }, _x = () => {
    const e = g.useRef(null), t = g.useRef(null), n = g.useRef(null), r = g.useRef(null), l = g.useRef(null), a = g.useRef(null), o = g.useRef(null), [i, s] = g.useState("disconnected");
    return g.useEffect(() => {
      let c, m;
      return (async () => {
        const { Terminal: p } = await il(async () => {
          const { Terminal: R } = await import("./xterm-PglAAeey.js").then((h) => h.x);
          return {
            Terminal: R
          };
        }, []), { FitAddon: x } = await il(async () => {
          const { FitAddon: R } = await import("./addon-fit-CyyJcX4C.js").then((h) => h.a);
          return {
            FitAddon: R
          };
        }, []), { WebLinksAddon: w } = await il(async () => {
          const { WebLinksAddon: R } = await import("./addon-web-links-B1M8nFkN.js").then((h) => h.a);
          return {
            WebLinksAddon: R
          };
        }, []);
        if (!document.querySelector('link[href*="xterm"]')) {
          const R = document.createElement("link");
          R.rel = "stylesheet", R.href = "https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/css/xterm.min.css", document.head.appendChild(R);
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
        const k = new x();
        c.loadAddon(k), c.loadAddon(new w()), r.current = c, l.current = k, e.current && (c.open(e.current), k.fit()), s("connecting");
        try {
          const R = await fetch(`${ce()}/api/terminal/create`, {
            method: "POST"
          }), { sid: h } = await R.json();
          o.current = h;
          const f = new EventSource(`${ce()}/api/terminal/stream?sid=${h}`);
          a.current = f, f.onopen = () => {
            s("connected");
            const b = k.proposeDimensions();
            b && fetch(`${ce()}/api/terminal/resize`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: h,
                cols: b.cols,
                rows: b.rows
              })
            });
          }, f.onmessage = (b) => {
            const S = Uint8Array.from(atob(b.data), (j) => j.charCodeAt(0));
            c.write(S);
          }, f.onerror = () => {
            s("disconnected"), c.write(`\r
\x1B[33m[Connection closed]\x1B[0m\r
`), f.close();
          };
          const v = (b) => {
            fetch(`${ce()}/api/terminal/input`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: h,
                data: b
              })
            });
          };
          n.current = v, c.onData(v);
          const E = () => {
            k.fit();
            const b = k.proposeDimensions();
            b && fetch(`${ce()}/api/terminal/resize`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                sid: h,
                cols: b.cols,
                rows: b.rows
              })
            });
          }, _ = new ResizeObserver(E);
          e.current && _.observe(e.current), m = () => _.disconnect();
        } catch {
          s("disconnected"), c.write(`\x1B[31m[Failed to connect]\x1B[0m\r
`);
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
  }, Rx = [
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
  ], bx = () => {
    const [e, t] = g.useState(""), [n, r] = g.useState(""), [l, a] = g.useState(""), [o, i] = g.useState(false), [s, c] = g.useState(""), m = g.useRef(null), [d, p] = g.useState("proxy"), [x, w] = g.useState([]), [k, R] = g.useState(-1), h = g.useCallback(async (E) => {
      if (!E.trim()) return;
      let _ = E.trim();
      if (!_.startsWith("http://") && !_.startsWith("https://") && (_ = "https://" + _), r(_), c(""), d === "iframe") {
        t(_), w((b) => [
          ...b.slice(0, k + 1),
          _
        ]), R((b) => b + 1);
        return;
      }
      i(true);
      try {
        const b = await fetch(`${ce()}/proxy?url=${encodeURIComponent(_)}`);
        if (!b.ok) {
          c(`Failed: ${b.status}`), i(false);
          return;
        }
        if ((b.headers.get("content-type") || "").includes("application/json")) {
          const j = await b.json();
          if (j.redirect) {
            i(false), h(j.redirect);
            return;
          }
          c(j.error || "Unknown error");
        } else a(await b.text());
        w((j) => [
          ...j.slice(0, k + 1),
          _
        ]), R((j) => j + 1);
      } catch (b) {
        c(b instanceof Error ? b.message : "Failed");
      }
      i(false);
    }, [
      d,
      k
    ]), f = () => {
      k > 0 && (R(k - 1), h(x[k - 1]));
    }, v = () => {
      k < x.length - 1 && (R(k + 1), h(x[k + 1]));
    };
    return u.jsxs("div", {
      className: "flex h-full flex-col bg-maude-bg",
      children: [
        u.jsxs("div", {
          className: "flex shrink-0 flex-col border-b border-maude-border bg-maude-surface",
          children: [
            u.jsxs("form", {
              onSubmit: (E) => {
                E.preventDefault(), h(n);
              },
              className: "flex items-center gap-2 px-3 py-2",
              children: [
                u.jsxs("div", {
                  className: "flex gap-1",
                  children: [
                    u.jsx("button", {
                      type: "button",
                      onClick: f,
                      disabled: k <= 0,
                      className: "rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30",
                      children: "\u25C0"
                    }),
                    u.jsx("button", {
                      type: "button",
                      onClick: v,
                      disabled: k >= x.length - 1,
                      className: "rounded px-2 py-1 text-sm text-maude-muted disabled:opacity-30",
                      children: "\u25B6"
                    }),
                    u.jsx("button", {
                      type: "button",
                      onClick: () => h(n),
                      className: "rounded px-2 py-1 text-sm text-maude-muted",
                      children: "\u21BB"
                    })
                  ]
                }),
                u.jsx("input", {
                  type: "text",
                  value: n,
                  onChange: (E) => r(E.target.value),
                  placeholder: "Enter URL...",
                  className: "flex-1 rounded-lg bg-maude-bg px-3 py-2 text-sm text-maude-text placeholder-maude-muted outline-none focus:ring-1 focus:ring-maude-accent"
                }),
                u.jsx("button", {
                  type: "button",
                  onClick: () => p(d === "proxy" ? "iframe" : "proxy"),
                  className: "rounded-lg bg-maude-bg px-2 py-1 text-[10px] text-maude-muted",
                  children: d === "proxy" ? "PROXY" : "IFRAME"
                })
              ]
            }),
            u.jsx("div", {
              className: "flex gap-1 overflow-x-auto px-3 pb-2 no-scrollbar",
              children: Rx.map((E) => u.jsx("button", {
                onClick: () => {
                  r(E.url), h(E.url);
                },
                className: "shrink-0 rounded-full bg-maude-bg px-3 py-1 text-xs text-maude-muted hover:text-maude-text",
                children: E.label
              }, E.url))
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
  }, Tx = () => {
    const [e, t] = g.useState([]), [n, r] = g.useState(""), [l, a] = g.useState(false), o = g.useRef(null);
    g.useEffect(() => {
      o.current && (o.current.scrollTop = o.current.scrollHeight);
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
        const c = await fetch(`${ce()}/v1/chat/completions`, {
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
                content: s
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
  function Px(e) {
    return e < 1024 ? e + " B" : e < 1048576 ? (e / 1024).toFixed(1) + " KB" : (e / 1048576).toFixed(1) + " MB";
  }
  function Mx(e) {
    return new Date(e * 1e3).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }
  const Dx = () => {
    const [e, t] = g.useState([]), [n, r] = g.useState(""), [l, a] = g.useState(false), [o, i] = g.useState(""), [s, c] = g.useState("shared"), m = g.useRef(null), d = g.useCallback(async (w) => {
      a(true), i("");
      try {
        const k = s === "transfers" ? `${ce()}/transfers` : w ? `${ce()}/list?path=${encodeURIComponent(w)}` : `${ce()}/list`, h = await (await fetch(k)).json();
        h.error ? i(h.error) : (t(h.files || []), r(h.path || ""));
      } catch (k) {
        i(k instanceof Error ? k.message : "Failed");
      }
      a(false);
    }, [
      s
    ]);
    g.useEffect(() => {
      d();
    }, [
      d
    ]);
    const p = (w) => {
      window.open(`${ce()}/${s === "transfers" ? "download-transfer" : "download"}/${encodeURIComponent(w)}`);
    }, x = async (w) => {
      var _a2;
      const k = (_a2 = w.target.files) == null ? void 0 : _a2[0];
      if (k) {
        a(true);
        try {
          const h = await (await fetch(`${ce()}/upload/${encodeURIComponent(k.name)}`, {
            method: "POST",
            body: k
          })).json();
          h.error ? i(h.error) : d();
        } catch (R) {
          i(R instanceof Error ? R.message : "Upload failed");
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
              onClick: () => w.is_dir ? d(n ? `${n}/${w.name}` : w.name) : p(w.name),
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
                        w.is_dir ? "Directory" : Px(w.size),
                        " \xB7 ",
                        Mx(w.modified)
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
  async function Lx() {
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
  async function Ox() {
    await Lx(), window.location.replace(`/?fresh=${Date.now()}`);
  }
  const Ax = [
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
    const [e, t] = g.useState(null), [n, r] = g.useState([]), [l, a] = g.useState(() => {
      const j = localStorage.getItem("maude-default-model");
      return !j || j === "mistral-large-latest" ? "nemotron-super" : j;
    }), [o, i] = g.useState(() => localStorage.getItem("maude-default-voice") || "NATF2.pt"), [s, c] = g.useState(() => localStorage.getItem("maude-theme") || "dark"), [m, d] = g.useState(false), [p, x] = g.useState(""), w = e !== null, k = (e == null ? void 0 : e.gateway_port) ?? (new URL(ce()).port || "30080"), R = (_a2 = e == null ? void 0 : e.services) == null ? void 0 : _a2.llama_server, h = (_b = e == null ? void 0 : e.services) == null ? void 0 : _b.voice_server;
    g.useEffect(() => {
      fetch(`${ce()}/health`).then((j) => j.json()).then(t).catch(() => t(null)), fetch(`${ce()}/models`).then((j) => j.json()).then((j) => r(j.models || [])).catch(() => r([]));
    }, []);
    const f = (j) => {
      a(j), localStorage.setItem("maude-default-model", j);
    }, v = (j) => {
      i(j), localStorage.setItem("maude-default-voice", j);
    }, E = async () => {
      d(true), x("");
      try {
        await Ox();
      } catch (j) {
        x(j instanceof Error ? j.message : "Reset failed"), d(false);
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
    }, b = _(R), S = _(h);
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
                          className: `flex items-center gap-1.5 text-sm ${w ? "text-green-400" : "text-red-400"}`,
                          children: [
                            u.jsx("span", {
                              className: `h-2 w-2 rounded-full ${w ? "bg-green-400" : "bg-red-400"}`
                            }),
                            w ? "Connected" : "Offline"
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
                          className: `font-mono text-sm ${w ? "text-green-400" : "text-maude-muted"}`,
                          children: w ? `${k} (up)` : "\u2014"
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
                          className: `font-mono text-sm ${b.color}`,
                          children: b.text
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
                          className: `font-mono text-sm ${S.color}`,
                          children: S.text
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
                          children: ce().replace(/^https?:\/\//, "")
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
                  children: Ax.map((j) => u.jsxs("button", {
                    onClick: () => {
                      c(j.id), Ix(j.id);
                    },
                    className: `flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${j.id === s ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                    children: [
                      u.jsx("span", {
                        children: j.label
                      }),
                      u.jsx("span", {
                        className: "text-xs text-maude-muted",
                        children: j.desc
                      })
                    ]
                  }, j.id))
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
                    n.map((j) => u.jsxs("button", {
                      onClick: () => f(j.id),
                      className: `flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${j.id === l ? "bg-maude-bg text-maude-accent" : "text-maude-text hover:bg-maude-bg"}`,
                      children: [
                        u.jsxs("div", {
                          className: "flex items-center gap-2",
                          children: [
                            u.jsx("span", {
                              className: `h-2 w-2 rounded-full ${j.available ? "bg-green-400" : "bg-red-400"}`
                            }),
                            j.id
                          ]
                        }),
                        u.jsx("span", {
                          className: "text-xs text-maude-muted",
                          children: j.provider
                        })
                      ]
                    }, j.id)),
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
                    ].map((j) => u.jsxs("option", {
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
            u.jsxs("section", {
              children: [
                u.jsx("h2", {
                  className: "mb-3 text-xs font-semibold uppercase tracking-wider text-maude-muted",
                  children: "Network"
                }),
                u.jsxs("div", {
                  className: "space-y-3 rounded-xl bg-maude-surface p-4",
                  children: [
                    u.jsx("p", {
                      className: "text-sm text-maude-muted",
                      children: "Network settings are managed via Tailscale and your device's system settings."
                    }),
                    u.jsx("button", {
                      onClick: E,
                      disabled: m,
                      className: "w-full rounded-lg bg-maude-bg px-3 py-2.5 text-sm font-medium text-maude-text transition-colors hover:text-maude-accent disabled:opacity-50",
                      children: m ? "Resetting..." : "Reset App Cache"
                    }),
                    p && u.jsx("p", {
                      className: "text-xs text-red-400",
                      children: p
                    })
                  ]
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
                          children: "1.0.0"
                        })
                      ]
                    }),
                    u.jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        u.jsx("span", {
                          className: "text-sm text-maude-text",
                          children: "Build"
                        }),
                        u.jsx("span", {
                          className: "text-right font-mono text-[11px] text-maude-muted",
                          children: (/* @__PURE__ */ new Date("2026-05-19T15:42:03.253Z")).toLocaleString()
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
  function Sa() {
    return ce();
  }
  function Ux(e = 1e4) {
    const [t, n] = g.useState(null), [r, l] = g.useState(true), a = g.useCallback(async () => {
      try {
        const s = await fetch(`${Sa()}/api/collab/status`);
        s.ok && n(await s.json());
      } catch {
      } finally {
        l(false);
      }
    }, []);
    g.useEffect(() => {
      a();
      const s = setInterval(a, e);
      return () => clearInterval(s);
    }, [
      a,
      e
    ]);
    const o = g.useCallback(async (s, c = "", m = []) => {
      const d = await fetch(`${Sa()}/api/collab/projects`, {
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
    ]), i = g.useCallback(async (s, c = "", m = "SHELL") => {
      const d = await fetch(`${Sa()}/api/collab/tasks`, {
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
  function Fx() {
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
  let Fc = false;
  function $x() {
    if (Fc) return;
    Fc = true;
    const e = Fx(), t = `${e.clientType}-${Math.random().toString(36).slice(2, 8)}`, n = () => {
      fetch(`${Sa()}/api/collab/presence`, {
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
  function Xi(e, t) {
    const n = Math.max(0, Math.floor(t - e));
    return n < 60 ? `${n}s ago` : n < 3600 ? `${Math.floor(n / 60)}m ago` : n < 86400 ? `${Math.floor(n / 3600)}h ago` : `${Math.floor(n / 86400)}d ago`;
  }
  const Bx = {
    pending: "bg-yellow-500",
    running: "bg-blue-500",
    completed: "bg-green-500",
    failed: "bg-red-500"
  }, $c = {
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
  }, Vx = ({ entry: e, now: t }) => u.jsxs("div", {
    className: "flex items-center gap-3 rounded-xl bg-maude-surface p-3",
    children: [
      u.jsx("div", {
        className: "flex h-10 w-10 items-center justify-center rounded-full bg-maude-card text-lg",
        children: $c[e.client_type] || $c.unknown
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
              Xi(e.last_seen, t)
            ]
          })
        ]
      })
    ]
  }), Bc = {
    chat: "\u{1F4AC}",
    task_dispatched: "\u{1F680}",
    project_created: "\u{1F4C1}",
    custom: "\u2022"
  }, Wx = ({ event: e, now: t }) => u.jsxs("div", {
    className: "flex items-start gap-2 py-1.5",
    children: [
      u.jsx("span", {
        className: "mt-0.5 text-sm",
        children: Bc[e.type] || Bc.custom
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
              Xi(e.ts, t)
            ]
          })
        ]
      })
    ]
  }), Hx = ({ project: e }) => u.jsxs("div", {
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
  }), Qx = ({ task: e, now: t }) => u.jsxs("div", {
    className: "rounded-xl bg-maude-surface p-3",
    children: [
      u.jsxs("div", {
        className: "flex items-center gap-2",
        children: [
          u.jsx("span", {
            className: `inline-block h-2 w-2 rounded-full ${Bx[e.status] || "bg-gray-500"}`
          }),
          u.jsx("span", {
            className: "text-[10px] font-medium uppercase text-maude-muted",
            children: e.status
          }),
          u.jsx("span", {
            className: "ml-auto text-[10px] text-maude-muted",
            children: Xi(e.created_at, t)
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
  }), Kx = () => {
    const { status: e, loading: t } = Ux(), [n, r] = g.useState("presence");
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
              }) : e.presence.map((o) => u.jsx(Vx, {
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
              }) : e.projects.map((o) => u.jsx(Hx, {
                project: o
              }, o.id))
            }),
            n === "tasks" && u.jsx("div", {
              className: "flex flex-col gap-2",
              children: e.tasks.length === 0 ? u.jsx("p", {
                className: "py-8 text-center text-sm text-maude-muted",
                children: "No tasks dispatched"
              }) : e.tasks.map((o) => u.jsx(Qx, {
                task: o,
                now: l
              }, o.id))
            })
          ]
        })
      ]
    });
  };
  async function ar(e) {
    try {
      const t = await fetch(`${ce()}/api/command-center/${e}`);
      return t.ok ? await t.json() : null;
    } catch {
      return null;
    }
  }
  function Yx(e = 1e4) {
    const [t, n] = g.useState(null), [r, l] = g.useState(null), [a, o] = g.useState([]), [i, s] = g.useState([]), [c, m] = g.useState(null), [d, p] = g.useState([]), [x, w] = g.useState(true), k = g.useCallback(async () => {
      const [R, h, f, v, E, _] = await Promise.all([
        ar("system"),
        ar("gpu-processes"),
        ar("sessions?limit=10"),
        ar("activity?limit=15"),
        ar("scheduler"),
        ar("nodes")
      ]);
      n(R), l(h && Array.isArray(h.processes) ? h : null), o((f == null ? void 0 : f.sessions) || []), s((v == null ? void 0 : v.activities) || []), m(E), p((_ == null ? void 0 : _.nodes) || []), w(false);
    }, []);
    return g.useEffect(() => {
      k();
      const R = setInterval(k, e);
      return () => clearInterval(R);
    }, [
      k,
      e
    ]), {
      system: t,
      gpuProcesses: r,
      sessions: a,
      activity: i,
      scheduler: c,
      nodes: d,
      loading: x,
      refresh: k
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
  }), Gx = ({ processes: e }) => {
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
  }, Xx = ({ node: e }) => u.jsxs("div", {
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
  }), Jx = ({ task: e }) => u.jsxs("div", {
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
    const { system: e, gpuProcesses: t, sessions: n, activity: r, scheduler: l, nodes: a, loading: o, refresh: i } = Yx(), [s, c] = g.useState("overview");
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
    ], d = typeof ((_a2 = e == null ? void 0 : e.gpu) == null ? void 0 : _a2.temperature_c) == "number" ? e.gpu.temperature_c : 0, p = d > 80 ? "text-red-400" : d > 60 ? "text-yellow-400" : "text-green-400";
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
                    u.jsx(Dn, {
                      label: "CPU",
                      value: `${(e == null ? void 0 : e.cpu_percent) ?? 0}%`,
                      sub: `${((_b = e == null ? void 0 : e.ram) == null ? void 0 : _b.used_gb) ?? 0}/${((_c2 = e == null ? void 0 : e.ram) == null ? void 0 : _c2.total_gb) ?? 0}GB RAM`
                    }),
                    u.jsx(Dn, {
                      label: "GPU Temp",
                      value: `${d}\xB0C`,
                      sub: ((_d2 = e == null ? void 0 : e.gpu) == null ? void 0 : _d2.name) || "N/A",
                      color: p
                    }),
                    u.jsx(Dn, {
                      label: "Disk",
                      value: `${((_e2 = e == null ? void 0 : e.disk) == null ? void 0 : _e2.percent) ?? 0}%`,
                      sub: `${((_f2 = e == null ? void 0 : e.disk) == null ? void 0 : _f2.used_gb) ?? 0}/${((_g2 = e == null ? void 0 : e.disk) == null ? void 0 : _g2.total_gb) ?? 0}GB`
                    }),
                    u.jsx(Dn, {
                      label: "Sessions",
                      value: n.length,
                      sub: `${((_h2 = l == null ? void 0 : l.stats) == null ? void 0 : _h2.active) ?? 0} scheduled tasks`
                    })
                  ]
                }),
                t && u.jsx(Gx, {
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
              }) : a.map((x, w) => u.jsx(Xx, {
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
                ((_i2 = l == null ? void 0 : l.tasks) == null ? void 0 : _i2.length) ? l.tasks.map((x) => u.jsx(Jx, {
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
    const e = Yi(), t = go();
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
          children: u.jsx(wv, {})
        }),
        u.jsx(ny, {})
      ]
    });
  }
  const ly = jv([
    {
      element: u.jsx(ry, {}),
      children: [
        {
          path: "/",
          element: u.jsx(Fv, {})
        },
        {
          path: "/maude",
          element: u.jsx(px, {})
        },
        {
          path: "/maude/voice",
          element: u.jsx(Cx, {})
        },
        {
          path: "/terminal",
          element: u.jsx(_x, {})
        },
        {
          path: "/browser",
          element: u.jsx(bx, {})
        },
        {
          path: "/messages",
          element: u.jsx(Tx, {})
        },
        {
          path: "/files",
          element: u.jsx(Dx, {})
        },
        {
          path: "/collab",
          element: u.jsx(Kx, {})
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
  Jo.createRoot(document.getElementById("root")).render(u.jsx(Dv, {
    router: ly
  }));
})();
export {
  __tla,
  ay as c,
  Wc as g
};
