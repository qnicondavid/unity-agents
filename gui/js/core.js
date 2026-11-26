/* Shared helpers: data access, formatting, DOM building, charts, URL state. */
(function () {
  "use strict";
  var D = window.UA_DATA || {};
  var UA = (window.UA = { views: [], data: D });

  UA.FEATURES = ["batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers", "beta", "epsilon", "num_epoch"];
  UA.LABELS = {
    batch_size: "batch size", buffer_size: "buffer size", learning_rate: "learning rate",
    hidden_units: "hidden units", num_layers: "layers", beta: "beta (entropy)", epsilon: "epsilon (clip)",
    num_epoch: "epochs per update", lambd: "lambda", time_horizon: "time horizon",
    learning_rate_schedule: "LR schedule", normalize: "normalize", max_steps: "budget"
  };
  UA.LOGGED = { batch_size: 1, buffer_size: 1, learning_rate: 1, beta: 1 };
  UA.MODEL_NAMES = { gpr: "Gaussian process", mlp: "MLP", forest: "Regression forest", ridge: "Ridge (linear)", median: "Median baseline", scikit_learn: "scikit-learn version (check)", boosting: "Gradient boosting", knn: "Nearest neighbours" };
  // A model name inside a sentence: "the MLP", "the Gaussian process", "the ridge (linear)".
  UA.inText = function (m) { return m === "mlp" || m === "gpr" ? UA.MODEL_NAMES[m] : UA.MODEL_NAMES[m].charAt(0).toLowerCase() + UA.MODEL_NAMES[m].slice(1); };
  // Shape as well as colour tells the balls apart, for colour-blind viewers and grey projectors.
  UA.ballSymbol = function (ball) { return ball === "random" ? "diamond" : "circle"; };
  UA.MACHINE_NAMES = { m01: "m01 · Intel Core Ultra 9 285H", m02: "m02 · Intel i7-10750H", m03: "m03 · Apple M4", m05: "m05 · Intel i7-1370P" };

  /* Data --------------------------------------------------------------------------------------- */

  function objects(t) {
    if (!t) return [];
    return t.rows.map(function (r) { var o = {}; for (var i = 0; i < t.columns.length; i++) o[t.columns[i]] = r[i]; return o; });
  }
  UA.objects = objects;

  UA.ready = !!(D.runs && D.curves && D.results);
  if (UA.ready) {
    UA.g10 = objects(D.runs.group10).map(function (r) {
      r.id = r.RunID;
      r.reward = D.curves.reward[r.id] || null;
      r.steps = D.curves.other_steps[r.id] || D.curves.steps;
      r.duration = D.curves.duration[r.id];
      r.reached = r.reached_95 === true || r.reached_95 === "True";
      r.key = UA.FEATURES.map(function (f) { return r[f]; }).join("|");
      return r;
    });
    UA.byId = {};
    UA.g10.forEach(function (r) { UA.byId[r.id] = r; });
    UA.g15 = objects(D.runs.group15).map(function (r) {
      r.reached = r.reached_95 === true || r.reached_95 === "True";
      r.unknown = r.reached_95 === null;
      return r;
    });
    UA.values = {};
    UA.FEATURES.forEach(function (f) {
      var s = {};
      UA.g10.forEach(function (r) { s[r[f]] = r[f]; });
      UA.values[f] = Object.keys(s).map(function (k) { return s[k]; }).sort(function (a, b) { return a - b; });
    });
    UA.configs = {};
    UA.g10.forEach(function (r) { (UA.configs[r.key] = UA.configs[r.key] || []).push(r); });
  }

  // Seed 2 and seed 3 of the same config, ball and batch, when both reached 95.
  // With d = log(steps seed 2) - log(steps seed 3), each run sits sigma = sd(d)/sqrt(2) from its
  // config's typical value. A model that knew that value exactly would still be off by about
  // mean|d|/sqrt(2) on average (equal to sigma*sqrt(2/pi) for normal noise, but less swayed by a
  // few extreme pairs). That is the noise floor for the mean absolute error.
  UA.seedPairs = function (ball) {
    var groups = {};
    UA.g10.forEach(function (r) {
      if (ball && r.ball !== ball) return;
      var k = r.key + "|" + r.ball + "|" + r.batch;
      (groups[k] = groups[k] || {})[r.seed] = r;
    });
    var pairs = [];
    Object.keys(groups).forEach(function (k) {
      var g = groups[k];
      if (g[2] && g[3]) pairs.push({ a: g[2], b: g[3], both: g[2].reached && g[3].reached });
    });
    return pairs;
  };
  UA.noiseFloor = function (ball) {
    var d = UA.seedPairs(ball).filter(function (p) { return p.both; }).map(function (p) { return Math.log(p.a.steps_to_95) - Math.log(p.b.steps_to_95); });
    var sigma = UA.stats.sd(d) / Math.SQRT2;
    var meanAbs = UA.stats.mean(d.map(Math.abs));
    return { pairs: d.length, sigma: sigma, meanAbs: meanAbs, floor: meanAbs / Math.SQRT2, diffs: d };
  };

  // Share of the Gaussian process's error that is seed noise: noise floor / its error.
  // Every view that quotes this number calls this, so they always agree.
  UA.noiseShare = function (ball) {
    var mae = UA.models ? UA.models.balls[ball].scores.gpr.mae
      : (UA.result("gpr/group10_steps_to_95/tuned_best_mean/summary.csv").filter(function (r) { return r.ball === ball; })[0] || {}).gpr_mean_absolute_error;
    var nf = UA.noiseFloor(ball);
    return { share: nf.floor / mae, floor: nf.floor, mae: mae, nf: nf };
  };

  // Short names for table headers, so tables fit without scrolling sideways.
  UA.SHORT = { batch_size: "batch", buffer_size: "buffer", learning_rate: "lr", hidden_units: "units", num_layers: "layers", beta: "beta", epsilon: "epsilon", num_epoch: "epochs" };

  UA.result = function (path) { return objects(D.results && D.results.files[path]); };
  UA.config = function (path) { return D.results && D.results.configs[path]; };
  UA.models = D.models || null;

  UA.gridIndex = function (cfg) {
    var m = UA.models, idx = 0;
    for (var i = 0; i < m.features.length; i++) {
      var vals = m.values[m.features[i]];
      var j = vals.indexOf(Number(cfg[m.features[i]]));
      if (j < 0) return -1;
      idx = idx * vals.length + j;
    }
    return idx;
  };
  UA.gridConfig = function (idx) {
    var m = UA.models, cfg = {};
    for (var i = m.features.length - 1; i >= 0; i--) {
      var vals = m.values[m.features[i]];
      cfg[m.features[i]] = vals[idx % vals.length];
      idx = Math.floor(idx / vals.length);
    }
    return cfg;
  };

  var loaded = {};
  UA.loadScript = function (src) {
    if (!loaded[src]) loaded[src] = new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src; s.onload = resolve; s.onerror = function () { reject(new Error("could not load " + src)); };
      document.head.appendChild(s);
    });
    return loaded[src];
  };

  /* Stats -------------------------------------------------------------------------------------- */

  var S = (UA.stats = {});
  S.sorted = function (a) { return a.slice().sort(function (x, y) { return x - y; }); };
  S.quantile = function (sorted, q) {
    if (!sorted.length) return NaN;
    var pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  S.median = function (a) { return S.quantile(S.sorted(a), 0.5); };
  S.mean = function (a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
  S.sd = function (a) { var m = S.mean(a), s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return Math.sqrt(s / (a.length - 1)); };
  S.se = function (a) { return S.sd(a) / Math.sqrt(a.length); };

  /* Formatting --------------------------------------------------------------------------------- */

  var F = (UA.fmt = {});
  F.int = function (n) { return n == null || isNaN(n) ? "–" : Math.round(n).toLocaleString("en-GB"); };
  F.steps = function (n) {
    if (n == null || isNaN(n)) return "–";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 2).replace(/\.?0+$/, "") + "M";
    if (n >= 1e3) return Math.round(n / 1e3) + "k";
    return String(Math.round(n));
  };
  F.num = function (x, d) { return x == null || isNaN(x) ? "–" : Number(x).toFixed(d == null ? 3 : d); };
  F.pct = function (x, d) { return x == null || isNaN(x) ? "–" : (x * 100).toFixed(d == null ? 0 : d) + "%"; };
  F.setting = function (f, v) {
    if (v == null) return "–";
    if (f === "learning_rate" || f === "beta") return String(Number(v));
    if (f === "max_steps") return F.steps(v);
    return String(v);
  };
  // A mean absolute error in natural-log units, read as "a typical prediction is x% off".
  F.offBy = function (mae) { return Math.round((Math.exp(mae) - 1) * 100) + "%"; };
  F.duration = function (s) {
    if (s == null) return "–";
    var m = Math.round(s / 60);
    return m < 60 ? m + " min" : Math.floor(m / 60) + " h " + (m % 60) + " min";
  };
  UA.cap = function (s) { return s.charAt(0).toUpperCase() + s.slice(1); };
  F.esc = function (s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); };
  F.config = function (cfg) {
    return UA.FEATURES.map(function (f) { return UA.LABELS[f] + " " + F.setting(f, cfg[f]); }).join(" · ");
  };

  /* DOM ---------------------------------------------------------------------------------------- */

  UA.h = function (tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      var v = attrs[k];
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k === "text") el.textContent = v;
      else if (k.slice(0, 2) === "on") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c == null || c === false) continue;
      if (Array.isArray(c)) c.forEach(function (x) { if (x != null) el.appendChild(typeof x === "string" ? document.createTextNode(x) : x); });
      else el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return el;
  };
  var h = UA.h;

  UA.head = function (parent, o) {
    parent.appendChild(h("header", { class: "view-head" },
      h("h1", { text: o.title }),
      o.lede ? h("p", { class: "lede", html: o.lede }) : null));
  };

  UA.section = function (parent, o) {
    var s = h("section", { class: "section", id: o.id || null });
    var head = h("div", { class: "section-head" },
      h("div", null, h("h2", { text: o.title }), o.text ? h("div", { class: "prose", html: "<p>" + o.text + "</p>" }) : null));
    s.appendChild(head);
    parent.appendChild(s);
    s.head = head;
    return s;
  };

  // A segmented control. options: [{value, label}] or plain values.
  UA.seg = function (o) {
    var wrap = h("div", { class: "seg" + (o.mono ? " mono" : "") + (o.small ? " seg-small" : ""), role: "group", "aria-label": o.label || null });
    var value = o.value;
    var opts = o.options.map(function (x) { return typeof x === "object" ? x : { value: x, label: String(x) }; });
    function sync() {
      Array.prototype.forEach.call(wrap.children, function (b, i) { b.setAttribute("aria-pressed", String(opts[i].value) === String(value)); });
    }
    opts.forEach(function (x) {
      wrap.appendChild(h("button", { type: "button", title: x.title || null, onclick: function () { value = x.value; sync(); o.onChange(x.value); } }, x.label));
    });
    sync();
    wrap.set = function (v) { value = v; sync(); };
    return wrap;
  };

  UA.control = function (label, el) {
    return h("div", { class: "control" }, h("span", { class: "label", text: label }), el);
  };

  UA.select = function (o) {
    var s = h("select", { "aria-label": o.label || null, onchange: function () { o.onChange(s.value); } });
    o.options.forEach(function (x) {
      var opt = typeof x === "object" ? x : { value: x, label: String(x) };
      var el = h("option", { value: opt.value }, opt.label);
      if (String(opt.value) === String(o.value)) el.selected = true;
      s.appendChild(el);
    });
    return s;
  };

  // A sortable, paged table. columns: [{key, label, num, fmt, title}]
  UA.table = function (parent, o) {
    var state = { sort: o.sort || null, dir: o.dir || 1, page: 0 };
    var size = o.pageSize || 25;
    var wrap = h("div", { class: "table-wrap" });
    var pager = h("div", { class: "pager" });
    parent.appendChild(wrap);
    parent.appendChild(pager);
    var rows = o.rows;
    function render() {
      var data = rows.slice();
      if (state.sort) {
        var k = state.sort;
        data.sort(function (a, b) {
          var x = a[k], y = b[k];
          if (x == null && y == null) return 0;
          if (x == null) return 1;
          if (y == null) return -1;
          return (x < y ? -1 : x > y ? 1 : 0) * state.dir;
        });
      }
      var pages = Math.max(1, Math.ceil(data.length / size));
      state.page = Math.min(state.page, pages - 1);
      var view = data.slice(state.page * size, state.page * size + size);
      var thead = h("tr", null, o.columns.map(function (c) {
        var th = h("th", { class: (c.num ? "n " : "") + "sortable", scope: "col", title: c.title || null, tabindex: "0" }, c.label);
        if (state.sort === c.key) th.setAttribute("aria-sort", state.dir > 0 ? "ascending" : "descending");
        function act() { if (state.sort === c.key) state.dir = -state.dir; else { state.sort = c.key; state.dir = c.num ? 1 : 1; } render(); }
        th.addEventListener("click", act);
        th.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } });
        return th;
      }));
      var tbody = h("tbody", null, view.map(function (r) {
        var tr = h("tr", { class: (o.onRow ? "clickable" : "") + (o.rowClass ? " " + o.rowClass(r) : "") }, o.columns.map(function (c) {
          var v = c.fmt ? c.fmt(r[c.key], r) : r[c.key];
          var td = h("td", { class: c.num ? "n" : null });
          if (v instanceof Node) td.appendChild(v); else td.textContent = v == null ? "–" : v;
          return td;
        }));
        if (o.onRow) {
          tr.tabIndex = 0;
          tr.addEventListener("click", function () { o.onRow(r); });
          tr.addEventListener("keydown", function (e) { if (e.key === "Enter") o.onRow(r); });
        }
        return tr;
      }));
      wrap.innerHTML = "";
      wrap.appendChild(h("table", null, h("thead", null, thead), tbody));
      pager.innerHTML = "";
      if (pages <= 1) return;
      pager.appendChild(h("span", { class: "num", text: data.length ? (state.page * size + 1) + "–" + (state.page * size + view.length) + " of " + F.int(data.length) : "No rows match" }));
      if (pages > 1) pager.appendChild(h("span", null,
        h("button", { class: "btn quiet", type: "button", disabled: state.page === 0 || null, onclick: function () { state.page--; render(); } }, "Previous"),
        h("button", { class: "btn quiet", type: "button", disabled: state.page >= pages - 1 || null, onclick: function () { state.page++; render(); } }, "Next")));
    }
    render();
    return { update: function (r) { rows = r; state.page = 0; render(); } };
  };

  /* Charts ------------------------------------------------------------------------------------- */

  UA.color = function (name) { return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim(); };
  UA.rgba = function (hex, a) {
    var m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
    return m ? "rgba(" + parseInt(m[1], 16) + "," + parseInt(m[2], 16) + "," + parseInt(m[3], 16) + "," + a + ")" : hex;
  };
  UA.ballColor = function (ball) { return UA.color(ball === "random" ? "random" : "fixed"); };
  // Projectors wash out pale fills, so shaded bands get stronger in projector mode.
  UA.ballSoft = function (ball) {
    if (document.documentElement.classList.contains("projector")) return UA.rgba(UA.ballColor(ball), 0.3);
    return UA.color(ball === "random" ? "random-soft" : "fixed-soft");
  };
  UA.BALLS = ["fixed", "random"];
  UA.BALL_LABEL = { fixed: "Fixed ball", random: "Random ball" };

  function merge(base, over) {
    var out = {};
    for (var k in base) out[k] = base[k];
    for (var j in over) {
      if (over[j] && typeof over[j] === "object" && !Array.isArray(over[j]) && base[j] && typeof base[j] === "object") out[j] = merge(base[j], over[j]);
      else out[j] = over[j];
    }
    return out;
  }
  UA.merge = merge;

  UA.layout = function (over) {
    var big = document.documentElement.classList.contains("projector");
    var fs = big ? 16 : 13;
    var axis = {
      gridcolor: UA.color("rule"), linecolor: UA.color("rule-strong"), zeroline: false,
      ticks: "outside", tickcolor: UA.color("rule-strong"), ticklen: 4, automargin: true,
      tickfont: { family: "Atkinson Next, Segoe UI, sans-serif", size: fs, color: UA.color("ink-2") },
      title: { font: { size: fs, color: UA.color("ink-2") }, standoff: 8 }
    };
    var base = {
      font: { family: "Atkinson Next, Segoe UI, sans-serif", size: fs, color: UA.color("ink-2") },
      paper_bgcolor: UA.color("paper"), plot_bgcolor: UA.color("paper"),
      margin: { l: 8, r: 12, t: 8, b: 8 }, autosize: true,
      hoverlabel: { bgcolor: UA.color("surface"), bordercolor: UA.color("rule-strong"), font: { family: "Atkinson Next, Segoe UI, sans-serif", size: fs, color: UA.color("ink") }, align: "left" },
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: fs, color: UA.color("ink-2") }, bgcolor: "rgba(0,0,0,0)" },
      showlegend: false, hovermode: "closest", dragmode: false
    };
    var out = merge(base, over || {});
    Object.keys(out).concat(["xaxis", "yaxis"]).forEach(function (k) {
      if (/^[xy]axis\d*$/.test(k)) {
        out[k] = merge(axis, out[k] || {});
        // Category labels are words, so they get the text colour and size of the axis titles.
        if (out[k].type === "category") out[k].tickfont = merge(out[k].tickfont, { family: "Atkinson Next, Segoe UI, sans-serif", size: fs, color: UA.color("ink-2") });
        if (out[k].tickfont && out[k].tickfont.size < fs) out[k].tickfont.size = fs;
      }
    });
    // Annotations follow the same size as the axes, so projector mode scales them too.
    if (out.annotations) out.annotations = out.annotations.map(function (a) {
      return merge(a, { font: merge({ family: "Atkinson Next, Segoe UI, sans-serif", color: UA.color("ink-2") }, merge(a.font || {}, { size: fs })) });
    });
    return out;
  };

  UA.plot = function (el, traces, layout, config) {
    var cfg = merge({ displayModeBar: false, responsive: true, displaylogo: false, scrollZoom: false }, config || {});
    return Plotly.react(el, traces, UA.layout(layout), cfg);
  };

  // A figure block: title, caption, chart, source line and a PNG button.
  UA.fig = function (parent, o) {
    var chart = h("div", { class: "fig-chart", style: "height:" + (o.height || 320) + "px", role: "img", "aria-label": o.alt || o.title || "chart" });
    var name = (o.file || o.title || "chart").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    var png = h("button", { class: "png-btn", type: "button", title: "Save this chart as a PNG", onclick: function () {
      Plotly.downloadImage(chart, { format: "png", filename: name, scale: 2 });
    } }, "PNG");
    var fig = h("figure", { class: "fig" + (o.small ? " small-multiple" : ""), style: "margin:0" },
      o.title || o.caption || !o.noPng ? h("div", { class: "fig-head" + (o.title || o.caption ? "" : " fig-head-bare") },
        h("div", null, o.title ? h("p", { class: "fig-title", text: o.title }) : null, o.caption ? h("p", { class: "fig-cap", html: o.caption }) : null),
        o.noPng ? null : png) : null,
      chart,
      o.source ? h("p", { class: "fig-src", text: "Source: " + o.source }) : null);
    parent.appendChild(fig);
    fig.chart = chart;
    return chart;
  };

  /* URL state ---------------------------------------------------------------------------------- */
  // The hash holds the view and its settings, e.g. #/curves?runs=a,b&metric=reward

  UA.parseHash = function () {
    var raw = location.hash.replace(/^#\/?/, "");
    var q = raw.indexOf("?");
    var view = q < 0 ? raw : raw.slice(0, q);
    var params = {};
    if (q >= 0) raw.slice(q + 1).split("&").forEach(function (p) {
      if (!p) return;
      var i = p.indexOf("=");
      params[decodeURIComponent(i < 0 ? p : p.slice(0, i))] = i < 0 ? "" : decodeURIComponent(p.slice(i + 1));
    });
    return { view: view, params: params };
  };
  UA.hashFor = function (view, params) {
    var q = Object.keys(params || {}).filter(function (k) { return params[k] != null && params[k] !== ""; })
      .map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(params[k]).replace(/%2C/g, ","); }).join("&");
    return "#/" + view + (q ? "?" + q : "");
  };
  UA.params = {};
  // Update the current view's settings in the URL without re-rendering.
  UA.setParams = function (patch) {
    for (var k in patch) UA.params[k] = patch[k];
    history.replaceState(null, "", UA.hashFor(UA.current, UA.params));
  };
  UA.go = function (view, params) { location.hash = UA.hashFor(view, params || {}); };

  UA.register = function (view) { UA.views.push(view); };
})();
