(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt, S = UA.stats;

  var METRICS = [
    { value: "reward", label: "Mean reward" },
    { value: "episode_length", label: "Episode length" },
    { value: "entropy", label: "Policy entropy" },
    { value: "policy_loss", label: "Policy loss" },
    { value: "value_loss", label: "Value loss" },
    { value: "value_estimate", label: "Value estimate" },
    { value: "learning_rate", label: "Learning rate (schedule)" }
  ];

  function series(run, metric) {
    if (metric === "reward") return run.reward;
    var m = UA.data.metrics && UA.data.metrics.runs[run.id];
    return m ? m[metric] : null;
  }

  UA.register({
    id: "curves", num: "02", nav: "Training curves",
    render: function (main, params) {
      var mode = params.mode === "all" ? "all" : "runs";
      var metric = params.metric || "reward";
      var ids = (params.runs || "").split(",").filter(function (id) { return UA.byId[id]; });
      if (mode === "runs" && !ids.length) {
        var first = UA.g10.filter(function (r) { return r.reward && r.reached; })[0];
        ids = [first.id];
        var other = UA.g10.filter(function (r) { return r.key === first.key && r.ball === first.ball && r.batch === first.batch && r.id !== first.id; })[0];
        if (other) ids.push(other.id);
      }

      UA.head(main, {
        num: "02", kicker: "Training curves",
        title: mode === "all" ? "Every run, side by side" : "Read a run's training curve",
        lede: mode === "all"
          ? "The middle half of runs, the middle 80%, and the median, at every logged step. The lower chart counts how many runs have reached 95 so far."
          : "Pick runs to overlay. The dot marks each run's target: the first logged step with mean reward 95 or more."
      });

      var controls = h("div", { class: "controls" });
      main.appendChild(controls);
      controls.appendChild(UA.control("Show", UA.seg({ label: "Show", value: mode, options: [{ value: "runs", label: "Chosen runs" }, { value: "all", label: "All runs" }],
        onChange: function (v) { UA.go("curves", { mode: v === "all" ? "all" : null, metric: metric === "reward" ? null : metric, runs: v === "runs" ? ids.join(",") : null }); } })));
      controls.appendChild(UA.control("Metric", UA.select({ label: "Metric", value: metric, options: METRICS, onChange: function (v) { UA.setParams({ metric: v === "reward" ? null : v }); metric = v; draw(); } })));

      var body = h("div");
      main.appendChild(body);

      if (mode === "runs") {
        var list = h("datalist", { id: "run-ids" }, UA.g10.filter(function (r) { return r.reward; }).map(function (r) { return h("option", { value: r.id }); }));
        var input = h("input", { type: "search", list: "run-ids", placeholder: "Run id, e.g. 3DBall_b45_c12_s2", "aria-label": "Add a run by id" });
        function addFromInput() {
          var id = input.value.trim();
          if (UA.byId[id] && ids.indexOf(id) < 0) { ids.push(id); input.value = ""; sync(); }
        }
        input.addEventListener("keydown", function (e) { if (e.key === "Enter") addFromInput(); });
        input.addEventListener("change", addFromInput);
        controls.appendChild(UA.control("Add a run", input));
        controls.appendChild(h("div", { class: "control" }, h("span", { class: "label", text: "Quick picks" }), h("div", { style: "display:flex;gap:6px;flex-wrap:wrap" },
          h("button", { class: "btn", type: "button", onclick: function () { addRandom(); }, title: "Add a random run that has a curve" }, "Random run"),
          h("button", { class: "btn", type: "button", onclick: function () { addRelated("seed"); }, title: "Add the other seed of the last chosen run" }, "Other seed"),
          h("button", { class: "btn", type: "button", onclick: function () { addRelated("ball"); }, title: "Add the same config and seed with the other ball" }, "Other ball"),
          h("button", { class: "btn", type: "button", onclick: function () { ids = []; sync(); } }, "Clear all"))));
      }

      var chips = h("div", { class: "chips", style: "margin-bottom:16px" });
      if (mode === "runs") body.appendChild(chips);
      var chart = null, panels = {}, chart2 = null, tableHost = h("div", { style: "margin-top:32px" });
      if (mode === "all") {
        var pair = h("div", { class: "grid two" });
        body.appendChild(pair);
        UA.BALLS.forEach(function (b) {
          panels[b] = UA.fig(pair, { title: UA.BALL_LABEL[b], caption: "Line: median run. Darker shading: middle 50% of runs. Lighter shading, edged with dots: middle 80%.", height: 340, file: "all-curves-" + b });
        });
        var shareWrap = h("div", { style: "margin-top:40px" });
        body.appendChild(shareWrap);
        chart2 = UA.fig(shareWrap, { title: "Share of runs that have reached 95", caption: "Runs that never reach 95 within 500k steps keep the curve below 100%.", height: 260, file: "share-reached-95" });
      } else {
        chart = UA.fig(body, { title: null, height: 400, file: "training-curves", alt: "Training curves" });
        body.appendChild(tableHost);
      }

      function addRandom() {
        var pool = UA.g10.filter(function (r) { return r.reward && ids.indexOf(r.id) < 0; });
        ids.push(pool[Math.floor(Math.random() * pool.length)].id); sync();
      }
      function addRelated(kind) {
        var last = UA.byId[ids[ids.length - 1]];
        if (!last) return;
        var match = UA.g10.filter(function (r) {
          if (r.key !== last.key || r.batch !== last.batch) return false;
          return kind === "seed" ? r.ball === last.ball && r.seed !== last.seed : r.ball !== last.ball && r.seed === last.seed;
        })[0];
        if (match && ids.indexOf(match.id) < 0) { ids.push(match.id); sync(); }
      }
      function sync() { UA.setParams({ runs: ids.join(",") }); draw(); }

      function draw() {
        if (metric !== "reward" && !UA.data.metrics) {
          var target = chart || panels.fixed;
          target.innerHTML = "<div class='empty'>Loading the other metrics…</div>";
          UA.loadScript("data/metrics.js").then(function () { target.innerHTML = ""; draw(); }, function () { target.innerHTML = "<div class='empty'>gui/data/metrics.js is missing. Run python -m gui.export_data.</div>"; });
          return;
        }
        var label = METRICS.filter(function (m) { return m.value === metric; })[0].label;
        if (mode === "all") drawAll(label); else drawRuns(label);
      }

      function drawRuns(label) {
        chips.innerHTML = "";
        var traces = [], shapes = [], labels = [];
        ids.forEach(function (id, i) {
          var r = UA.byId[id];
          var y = series(r, metric);
          chips.appendChild(h("span", { class: "chip", title: F.config(r) },
            h("span", { class: "stroke" + (r.seed === 3 ? " dash" : ""), title: "seed " + r.seed, style: "color:" + UA.ballColor(r.ball) }),
            id + " · " + (r.reached ? F.steps(r.steps_to_95) : "never reached 95"),
            h("button", { type: "button", "aria-label": "Remove " + id, onclick: function () { ids.splice(ids.indexOf(id), 1); sync(); } }, "×")));
          if (!y) return;
          var dash = r.seed === 3 ? "dash" : "solid";
          labels.push({ x: r.steps[r.steps.length - 1], y: y[y.length - 1], xanchor: "left", xshift: 4, yshift: (i - (ids.length - 1) / 2) * -26, showarrow: false, text: "s" + r.seed + (ids.length > 2 ? " · " + r.id.replace(/^3DBall(Randomize)?_/, "") : ""), font: { size: 12, color: UA.ballColor(r.ball), family: "Atkinson Next, Segoe UI, sans-serif" } });
          traces.push({ x: r.steps, y: y, mode: "lines", name: id, line: { color: UA.ballColor(r.ball), width: 2, dash: dash },
            opacity: ids.length > 4 ? 0.8 : 1,
            hovertemplate: "<b>" + id + "</b><br>step %{x:,}<br>" + label.toLowerCase() + " %{y:.4~g}<extra></extra>" });
          if (metric === "reward" && r.reached) {
            var j = r.steps.indexOf(r.steps_to_95);
            traces.push({ x: [r.steps_to_95], y: [y[j]], mode: "markers", marker: { color: UA.ballColor(r.ball), size: 9, line: { color: UA.color("paper"), width: 2 } }, hoverinfo: "skip", showlegend: false });
          }
        });
        if (metric === "reward") shapes.push({ type: "line", x0: 0, x1: 500000, y0: 95, y1: 95, line: { color: UA.color("ink-3"), width: 1, dash: "dash" } });
        UA.plot(chart, traces, {
          showlegend: false,
          xaxis: { title: { text: "training step" }, range: [0, 505000], tickformat: "~s" },
          yaxis: { title: { text: label.toLowerCase() }, range: metric === "reward" ? [-2, 105] : null, type: metric === "learning_rate" ? "log" : "linear" },
          shapes: shapes,
          margin: { r: ids.length > 2 ? 90 : 30 },
          annotations: traces.length ? labels : [{ text: "Add a run to see its curve", showarrow: false, x: 0.5, y: 0.5, xref: "paper", yref: "paper", font: { color: UA.color("ink-3") } }]
        });
        tableHost.innerHTML = "";
        if (!ids.length) return;
        tableHost.appendChild(h("p", { class: "fig-title", style: "margin-bottom:8px", text: "Chosen runs" }));
        tableHost.appendChild(h("p", { class: "fig-cap", style: "margin-bottom:12px", text: "Solid lines are seed 2, dashed lines seed 3." + (ids.some(function (id) { return UA.byId[id].ball === "fixed"; }) && ids.some(function (id) { return UA.byId[id].ball === "random"; }) ? " Blue is the fixed ball, orange the random ball." : "") }));
        UA.table(tableHost, {
          rows: ids.map(function (id) { return UA.byId[id]; }), pageSize: 20,
          columns: [{ key: "id", label: "run" }, { key: "steps_to_95", label: "steps to 95", num: true, fmt: function (v) { return v ? F.int(v) : "not reached"; } },
            { key: "ball", label: "ball" }, { key: "seed", label: "seed", num: true }]
            .concat(UA.FEATURES.map(function (f) { return { key: f, label: UA.SHORT[f], title: UA.LABELS[f], num: true, fmt: function (v) { return F.setting(f, v); } }; }))
            .concat([{ key: "machine", label: "machine" }, { key: "duration", label: "wall time", num: true, fmt: F.duration }])
        });
      }

      function drawAll(label) {
        var steps = UA.data.curves.steps;
        var all = {}, lo = Infinity, hi = -Infinity;
        UA.BALLS.forEach(function (ball) {
          var traces = all[ball] = [];
          var runs = UA.g10.filter(function (r) { return r.ball === ball && r.reward && r.steps === steps; });
          var q = [[], [], [], [], []];
          steps.forEach(function (_, j) {
            var col = [];
            runs.forEach(function (r) { var y = series(r, metric); if (y && y[j] != null) col.push(y[j]); });
            col = S.sorted(col);
            [0.1, 0.25, 0.5, 0.75, 0.9].forEach(function (p, i) { q[i].push(S.quantile(col, p)); });
          });
          var c = UA.ballColor(ball), soft = UA.ballSoft(ball);
          traces.push({ x: steps, y: q[0], mode: "lines", line: { width: 1, color: c, dash: "dot" }, opacity: 0.7, hoverinfo: "skip", showlegend: false });
          traces.push({ x: steps, y: q[4], mode: "lines", line: { width: 1, color: c, dash: "dot" }, fill: "tonexty", fillcolor: soft, hoverinfo: "skip", showlegend: false });
          traces.push({ x: steps, y: q[1], mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false });
          traces.push({ x: steps, y: q[3], mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: soft, hoverinfo: "skip", showlegend: false });
          traces.push({ x: steps, y: q[2], mode: "lines", name: UA.BALL_LABEL[ball] + " (" + F.int(runs.length) + " runs)", line: { color: c, width: 2.5 },
            customdata: q[0].map(function (_, i) { return [q[0][i], q[4][i]]; }),
            hovertemplate: "step %{x:,}<br>median %{y:.4~g}<br>middle 80%: %{customdata[0]:.3~g} to %{customdata[1]:.3~g}<extra></extra>" });
          q[0].concat(q[4]).forEach(function (v) { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
        });
        var logY = metric === "learning_rate";
        var pad = (hi - lo) * 0.04;
        UA.BALLS.forEach(function (ball) {
          UA.plot(panels[ball], all[ball], {
            hovermode: "x",
            xaxis: { title: { text: "training step" }, range: [0, 505000], tickformat: "~s" },
            yaxis: { title: { text: label.toLowerCase() }, type: logY ? "log" : "linear", range: logY ? null : [lo - pad, hi + pad] },
            shapes: metric === "reward" ? [{ type: "line", x0: 0, x1: 500000, y0: 95, y1: 95, line: { color: UA.color("ink-3"), width: 1, dash: "dash" } }] : []
          });
        });

        var share = UA.BALLS.map(function (ball) {
          var runs = UA.g10.filter(function (r) { return r.ball === ball && r.reached_95 != null; });
          var y = steps.map(function (s) { return runs.filter(function (r) { return r.reached && r.steps_to_95 <= s; }).length / runs.length; });
          var half = steps[y.findIndex(function (v) { return v >= 0.5; })];
          return { x: steps, y: y, mode: "lines", name: UA.BALL_LABEL[ball], line: { color: UA.ballColor(ball), width: 2, shape: "hv" },
            hovertemplate: UA.BALL_LABEL[ball] + "<br>by step %{x:,}: %{y:.0%}<extra></extra>", half: half };
        });
        UA.plot(chart2, share, {
          showlegend: true, hovermode: "x",
          xaxis: { title: { text: "training step" }, range: [0, 505000], tickformat: "~s" },
          yaxis: { title: { text: "runs at 95 or above" }, tickformat: ".0%", range: [0, 1.02] },
          annotations: share.map(function (t, i) {
            return { x: t.half, y: 0.5, text: "half by " + F.steps(t.half), showarrow: true, arrowhead: 0, arrowwidth: 1, arrowcolor: UA.ballColor(UA.BALLS[i]), ax: i ? 40 : -40, ay: i ? 30 : -30, font: { size: 11, color: UA.ballColor(UA.BALLS[i]) } };
          })
        });
      }

      draw();
    }
  });
})();
