(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt, S = UA.stats;

  var G15_FEATURES = ["batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers", "beta", "epsilon", "lambd", "num_epoch", "time_horizon", "learning_rate_schedule", "normalize"];
  var G15_LOG = { batch_size: 1, buffer_size: 1, learning_rate: 1, hidden_units: 1, beta: 1, time_horizon: 1 };
  var BUDGETS = [500000, 1000000, 2000000];

  function budgetColor(i) {
    // One hue, darker for a bigger budget: the budgets are ordered, so they get a ramp.
    var dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.getAttribute("data-theme") && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
    // Grey to black, not blue: blue already means the fixed ball everywhere else.
    var light = ["#7a828b", "#454c55", "#15171a"], night = ["#6b727b", "#a7adb5", "#eef0f2"];
    return (dark ? night : light)[i];
  }

  UA.register({
    id: "explorer", num: "03", nav: "Dataset explorer",
    render: function (main, params) {
      var data = params.data === "g15" ? "g15" : "g10";
      var ball = params.ball === "fixed" || params.ball === "random" ? params.ball : "both";

      UA.head(main, {
        num: "03", kicker: "Dataset explorer", tag: "SQ1",
        title: data === "g10" ? "Each setting against steps to 95" : "Group 15's runs, setting by setting",
        lede: data === "g10"
          ? "Before any model: how steps to 95 spreads out for each value of each setting. Only runs that reached 95 have a value; the share that never did is listed under each chart."
          : "Group 15 drew their settings from wider ranges than ours, some from a list of values and some continuous, and used three budgets. We reused their " + F.int(UA.g15.length) + " runs for the forest classifier and an MLP."
      });

      var controls = h("div", { class: "controls" });
      main.appendChild(controls);
      controls.appendChild(UA.control("Dataset", UA.seg({ label: "Dataset", value: data,
        options: [{ value: "g10", label: "Our runs (Group 10)" }, { value: "g15", label: "Group 15" }],
        onChange: function (v) { UA.go("explorer", { data: v === "g15" ? "g15" : null }); } })));
      if (data === "g10") controls.appendChild(UA.control("Ball", UA.seg({ label: "Ball", value: ball,
        options: [{ value: "both", label: "Both" }, { value: "fixed", label: "Fixed" }, { value: "random", label: "Random" }],
        onChange: function (v) { UA.go("explorer", { ball: v === "both" ? null : v }); } })));

      if (data === "g10") renderG10(main, ball); else renderG15(main);
    }
  });

  function renderG10(main, ballChoice) {
    var balls = ballChoice === "both" ? UA.BALLS : [ballChoice];
    var runs = UA.g10.filter(function (r) { return balls.indexOf(r.ball) >= 0 && r.reached_95 != null; });

    var s1 = UA.section(main, { title: "How long runs take" });
    var g = h("div", { class: "grid side-right" });
    s1.appendChild(g);
    var hist = UA.fig(g, { title: "Steps to 95", caption: "Runs are logged every 12,000 steps, so the target takes only those values.", height: 300, source: "data/group10.csv" });
    var side = h("div", { class: "side-text" });
    g.appendChild(side);
    balls.forEach(function (b) {
      var part = runs.filter(function (r) { return r.ball === b; });
      var reached = part.filter(function (r) { return r.reached; });
      var med = S.median(reached.map(function (r) { return r.steps_to_95; }));
      side.appendChild(h("p", { html: "With the <span class='" + (b === "random" ? "random-c" : "fixed-c") + "'><b>" + b + " ball</b></span>, the median run reached 95 after <b>" + F.steps(med) + "</b> steps. " +
        F.pct(reached.length / part.length, 1) + " of its " + F.int(part.length) + " runs got there within the 500k budget." }));
    });
    // Outlines rather than filled bars, so the two balls can overlap without muddying each other.
    var bins = [];
    for (var s = 12000; s <= 504000; s += 12000) bins.push(s);
    UA.plot(hist, balls.map(function (b) {
      var counts = bins.map(function () { return 0; });
      runs.forEach(function (r) { if (r.ball === b && r.reached) { var i = bins.indexOf(r.steps_to_95); if (i >= 0) counts[i]++; } });
      return { x: bins, y: counts, type: "scatter", mode: "lines", line: { shape: "hvh", color: UA.ballColor(b), width: 2 },
        fill: balls.length > 1 ? "none" : "tozeroy", fillcolor: UA.ballSoft(b), name: UA.BALL_LABEL[b],
        hovertemplate: UA.BALL_LABEL[b] + "<br>reached 95 at step %{x:,}<br>%{y} runs<extra></extra>" };
    }), { showlegend: balls.length > 1, hovermode: "x",
      xaxis: { title: { text: "steps to 95" }, tickformat: "~s", range: [0, 510000] }, yaxis: { title: { text: "runs" }, rangemode: "tozero" } });

    var s2 = UA.section(main, { title: "Setting by setting", text: "Each box spans the middle half of runs with that value; the line inside is the median. The y-axis is logarithmic, so equal gaps are equal ratios." });
    var grid = h("div", { class: "grid four" });
    s2.appendChild(grid);
    UA.FEATURES.forEach(function (f) {
      var vals = UA.values[f];
      var never = vals.map(function (v) {
        var part = runs.filter(function (r) { return r[f] === v; });
        return F.pct(part.filter(function (r) { return !r.reached; }).length / part.length);
      });
      var el = UA.fig(grid, { title: UA.LABELS[f], small: true, height: 230, file: "steps-by-" + f,
        caption: "never reached 95, by value: " + never.join(", ") });
      UA.plot(el, balls.map(function (b) {
        var part = runs.filter(function (r) { return r.ball === b && r.reached; });
        return { type: "box", name: UA.BALL_LABEL[b], x: part.map(function (r) { return F.setting(f, r[f]); }), y: part.map(function (r) { return r.steps_to_95; }),
          marker: { color: UA.ballColor(b) }, line: { width: 1.2 }, fillcolor: UA.ballSoft(b), boxpoints: false,
          hovertemplate: UA.BALL_LABEL[b] + "<extra></extra>" };
      }), { boxmode: "group", boxgap: 0.35, boxgroupgap: 0.12,
        xaxis: { type: "category", categoryorder: "array", categoryarray: vals.map(function (v) { return F.setting(f, v); }) },
        yaxis: { type: "log", tickvals: [20000, 50000, 100000, 200000, 500000], ticktext: ["20k", "50k", "100k", "200k", "500k"], range: [Math.log10(10000), Math.log10(520000)] } });
    });

    var s3 = UA.section(main, { title: "All runs", text: "Click a run to open its training curve." });
    var search = h("input", { type: "search", placeholder: "Filter, e.g. b45, random, m03", "aria-label": "Filter runs" });
    s3.appendChild(h("div", { class: "controls" }, UA.control("Filter", search)));
    // The outcome sits next to the run id, so it never scrolls out of sight.
    var columns = [{ key: "id", label: "run" }, { key: "steps_to_95", label: "steps to 95", num: true, fmt: function (v, r) { return r.reached ? F.int(v) : "not reached"; } },
      { key: "ball", label: "ball" }, { key: "seed", label: "seed", num: true }]
      .concat(UA.FEATURES.map(function (f) { return { key: f, label: UA.SHORT[f], title: UA.LABELS[f], num: true, fmt: function (v) { return F.setting(f, v); } }; }))
      .concat([{ key: "machine", label: "machine" }]);
    var t = UA.table(s3, { rows: runs, columns: columns, sort: "steps_to_95", pageSize: 15, onRow: function (r) { UA.go("curves", { runs: r.id }); } });
    search.addEventListener("input", function () {
      var q = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      t.update(runs.filter(function (r) {
        var text = (r.id + " " + r.ball + " " + r.machine + " b" + r.batch).toLowerCase();
        return q.every(function (w) { return text.indexOf(w) >= 0; });
      }));
    });
  }

  function renderG15(main) {
    var runs = UA.g15;
    var s1 = UA.section(main, { title: "Budgets and outcomes" });
    var g = h("div", { class: "grid two" });
    s1.appendChild(g);
    var bars = UA.fig(g, { title: "Runs that reached 95, by budget", height: 260, source: "data/group15.csv" });
    var hist = UA.fig(g, { title: "Steps to 95, by budget", height: 260 });
    UA.plot(bars, [
      { type: "bar", name: "reached 95", x: BUDGETS.map(F.steps), y: BUDGETS.map(function (b) { return runs.filter(function (r) { return r.max_steps === b && r.reached; }).length; }), marker: { color: UA.color("ink") }, hovertemplate: "%{x}: %{y} reached<extra></extra>" },
      { type: "bar", name: "did not", x: BUDGETS.map(F.steps), y: BUDGETS.map(function (b) { return runs.filter(function (r) { return r.max_steps === b && !r.reached && !r.unknown; }).length; }), marker: { color: UA.color("paper"), line: { color: UA.color("ink-3"), width: 1.5 } }, hovertemplate: "%{x}: %{y} did not<extra></extra>" }
    ], { barmode: "stack", showlegend: true, bargap: 0.45, xaxis: { title: { text: "budget (max steps)" } }, yaxis: { title: { text: "runs" } } });
    var width = 50000, edges = [];
    for (var e = 0; e < 2000000; e += width) edges.push(e);
    UA.plot(hist, BUDGETS.map(function (b, i) {
      var counts = edges.map(function () { return 0; });
      runs.forEach(function (r) { if (r.max_steps === b && r.reached) counts[Math.min(edges.length - 1, Math.floor(r.steps_to_95 / width))]++; });
      return { type: "scatter", mode: "lines", name: F.steps(b), x: edges.map(function (x) { return x + width / 2; }), y: counts,
        line: { shape: "hvh", color: budgetColor(i), width: 2 }, hovertemplate: "%{x:,.0f} ± 25k<br>%{y} runs<extra>" + F.steps(b) + " budget</extra>" };
    }), { showlegend: true, hovermode: "x", xaxis: { title: { text: "steps to 95" }, tickformat: "~s", range: [0, 2000000] }, yaxis: { title: { text: "runs" }, rangemode: "tozero" } });

    var s2 = UA.section(main, { title: "Setting by setting", text: "Runs that reached 95, coloured by budget: light grey 500k, dark grey 1M, black 2M. A setting with a few values gets a box per value; a continuous one gets a dot per run. Steps are on a log scale." });
    var grid = h("div", { class: "grid four" });
    s2.appendChild(grid);
    G15_FEATURES.forEach(function (f) {
      var distinct = {};
      runs.forEach(function (r) { distinct[r[f]] = 1; });
      var only = Object.keys(distinct);
      // A setting with a handful of values gets a box per value; a continuous one gets dots.
      var cat = only.length <= 8;
      var order = only.slice().sort(function (a, b) { var x = Number(a), y = Number(b); return isNaN(x) || isNaN(y) ? (a < b ? -1 : 1) : x - y; });
      if (only.length < 2) {
        // A setting Group 15 never varied has nothing to plot.
        grid.appendChild(h("figure", { class: "fig small-multiple", style: "margin:0" },
          h("div", { class: "fig-head" }, h("div", null, h("p", { class: "fig-title", text: UA.LABELS[f] || f }), h("p", { class: "fig-cap", text: "" }))),
          h("div", { class: "constant", style: "height:220px" }, "Same for every Group 15 run", h("span", { class: "num", text: only[0] }))));
        return;
      }
      // Short category labels (10240 -> 10k) so they fit without rotating; plain decimals on
      // log axes instead of Plotly's SI prefixes (100µ).
      function short(v) { var n = Number(v); return !isNaN(n) && n >= 1000 ? F.steps(n) : String(v); }
      var xaxis = { type: "category", categoryorder: "array", categoryarray: order.map(short), tickangle: 0 };
      if (!cat) {
        xaxis = { type: G15_LOG[f] ? "log" : "linear", tickangle: 0 };
        if (G15_LOG[f]) {
          var lo = Math.min.apply(null, runs.map(function (r) { return r[f]; })), hi = Math.max.apply(null, runs.map(function (r) { return r[f]; }));
          var ticks = [];
          for (var p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) [1, 3].forEach(function (m) { var t = m * Math.pow(10, p); if (t >= lo * 0.8 && t <= hi * 1.25) ticks.push(+t.toPrecision(1)); });
          xaxis.tickvals = ticks;
          xaxis.ticktext = ticks.map(function (t) { return t >= 1 ? String(t) : t.toExponential(0).replace("e-", "e−"); });
        }
      }
      var el = UA.fig(grid, { title: UA.LABELS[f] || f, small: true, height: 220, file: "g15-" + f });
      UA.plot(el, BUDGETS.map(function (b, i) {
        var part = runs.filter(function (r) { return r.max_steps === b && r.reached; });
        return cat
          ? { type: "box", name: F.steps(b), x: part.map(function (r) { return short(r[f]); }), y: part.map(function (r) { return r.steps_to_95; }), marker: { color: budgetColor(i) }, boxpoints: false, line: { width: 1.2 } }
          : { type: "scatter", mode: "markers", name: F.steps(b), x: part.map(function (r) { return r[f]; }), y: part.map(function (r) { return r.steps_to_95; }),
            marker: { color: budgetColor(i), size: 4, opacity: 0.7 }, hovertemplate: (UA.LABELS[f] || f) + " %{x:.4~g}<br>steps %{y:,}<extra>" + F.steps(b) + "</extra>" };
      }), { boxmode: "group", xaxis: xaxis, yaxis: { type: "log", tickvals: [20000, 100000, 500000, 2000000], ticktext: ["20k", "100k", "500k", "2M"] } });
    });

    var s3 = UA.section(main, { title: "All of Group 15's runs" });
    var cols = [{ key: "RunID", label: "run" }, { key: "max_steps", label: "budget", num: true, fmt: F.steps }]
      .concat(G15_FEATURES.map(function (f) { return { key: f, label: UA.LABELS[f] || f, num: f !== "learning_rate_schedule" && f !== "normalize", fmt: function (v) { return typeof v === "number" ? String(+v.toPrecision(4)) : String(v); } }; }))
      .concat([{ key: "steps_to_95", label: "steps to 95", num: true, fmt: function (v, r) { return r.reached ? F.int(v) : r.unknown ? "stopped early" : "not reached"; } }, { key: "cpu", label: "cpu" }]);
    UA.table(s3, { rows: runs, columns: cols, sort: "steps_to_95", pageSize: 15 });
    s3.appendChild(h("p", { class: "note", html: "Data: Group 15 (AY 2025–2026), from the course's shared data repository, Apache License 2.0." }));
  }
})();
