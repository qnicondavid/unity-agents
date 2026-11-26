(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt;
  var MODELS = ["gpr", "mlp", "forest", "ridge"];

  UA.register({
    id: "predictor", num: "06", nav: "Predictor",
    render: function (main, params) {
      UA.head(main, {
        num: "06", kicker: "Predictor",
        title: "Choose settings, read the prediction",
        lede: "Every one of the 15,552 combinations in our search grid, predicted ahead of time for both balls by the four models. The Gaussian process also says how far a single run is likely to land from its prediction."
      });
      var M = UA.models;
      if (!M) { main.appendChild(h("div", { class: "empty", text: "gui/data/models.js is missing. Run python -m gui.export_data (about 15 minutes) to compute the predictions." })); return; }

      var feats = M.features, values = M.values;
      var ball = params.ball === "random" ? "random" : "fixed";
      var pick = (params.c || "").split("").map(Number);
      if (pick.length !== feats.length || pick.some(function (v, i) { return isNaN(v) || v >= values[feats[i]].length; })) pick = feats.map(function (f) { return Math.floor((values[f].length - 1) / 2); });

      function index(p) { var idx = 0; feats.forEach(function (f, i) { idx = idx * values[f].length + p[i]; }); return idx; }
      function cfgOf(p) { var c = {}; feats.forEach(function (f, i) { c[f] = values[f][p[i]]; }); return c; }
      function keyOf(p) { return feats.map(function (f, i) { return values[f][p[i]]; }).join("|"); }
      function pickOf(cfg) { return feats.map(function (f) { return values[f].indexOf(Number(cfg[f])); }); }
      function save() { UA.setParams({ c: pick.join(""), ball: ball === "fixed" ? null : ball }); }

      var layout = h("div", { class: "grid two", style: "align-items:start" });
      main.appendChild(layout);
      var left = h("div"), right = h("div");
      layout.appendChild(left); layout.appendChild(right);

      // Settings
      left.appendChild(h("div", { class: "setting-row" }, h("span", { class: "label", text: "ball" }),
        UA.seg({ label: "Ball", value: ball, options: [{ value: "fixed", label: "Fixed" }, { value: "random", label: "Random" }], onChange: function (v) { ball = v; update(); } })));
      var segs = feats.map(function (f, i) {
        var s = UA.seg({ label: UA.LABELS[f], mono: true, value: pick[i], options: values[f].map(function (v, j) { return { value: j, label: F.setting(f, v) }; }),
          onChange: function (j) { pick[i] = Number(j); update(); } });
        left.appendChild(h("div", { class: "setting-row" }, h("span", { class: "label", text: UA.LABELS[f] }), s));
        return s;
      });
      var runSelect = h("select", { "aria-label": "Load the settings of a real run" });
      runSelect.appendChild(h("option", { value: "" }, "Load a real run's settings…"));
      UA.g10.filter(function (r) { return r.ball === "fixed" && r.seed === 2; }).forEach(function (r) { runSelect.appendChild(h("option", { value: r.id }, r.id.replace(/_s2$/, ""))); });
      runSelect.addEventListener("change", function () { var r = UA.byId[runSelect.value]; if (r) { setPick(pickOf(r)); } runSelect.value = ""; });
      left.appendChild(h("div", { style: "display:flex;gap:8px;flex-wrap:wrap;margin-top:16px" },
        h("button", { class: "btn", type: "button", onclick: function () { setPick(feats.map(function (f) { return Math.floor(Math.random() * values[f].length); })); } }, "Random settings"),
        h("button", { class: "btn", type: "button", onclick: function () { setPick(fastest(1)[0].pick); } }, "Fastest predicted"),
        runSelect));
      function setPick(p) { pick = p; segs.forEach(function (s, i) { s.set(pick[i]); }); update(); }

      // Readout
      var readout = h("div", { class: "readout" });
      right.appendChild(readout);
      var realHost = h("div", { style: "margin-top:24px" });
      right.appendChild(realHost);

      var s2 = UA.section(main, { title: "Change one setting at a time", text: "The Gaussian process's prediction for each value of a setting, with the other seven held where they are. Bars span ±1 sd for a single run. Click a value to switch to it." });
      var g2 = h("div", { class: "grid four" });
      s2.appendChild(g2);
      var smalls = feats.map(function (f) { return UA.fig(g2, { title: UA.LABELS[f], small: true, height: 190, file: "what-if-" + f }); });

      var s3 = UA.section(main, { title: "Fastest configs, by prediction", text: "The ten configs with the lowest predicted steps to 95 for this ball. Where we ran a config, its real results are listed too. Click a row to load it." });
      var fastHost = h("div");
      s3.appendChild(fastHost);
      main.appendChild(h("p", { class: "note", html: "These models were trained on all of our runs, so for a config we ran they have seen its results. " +
        "For the honest error on unseen configs, see <a href='#/models'>Model comparison</a>." }));

      // "seed 2: 144k, seed 3: 156k", with the batch named when a config was drawn in two batches.
      function seedsText(runs) {
        var byBatch = {};
        runs.forEach(function (x) { (byBatch[x.batch] = byBatch[x.batch] || []).push(x); });
        var batches = Object.keys(byBatch);
        return batches.map(function (b) {
          var txt = byBatch[b].sort(function (a, c) { return a.seed - c.seed; }).map(function (x) { return "seed " + x.seed + ": " + (x.reached ? F.steps(x.steps_to_95) : "not reached"); }).join(", ");
          return batches.length > 1 ? (b === "first" ? "first batch" : "batch " + b) + ": " + txt : txt;
        }).join("; ");
      }

      function fastest(n) {
        var g = M.balls[ball].grid.gpr, idx = [];
        for (var i = 0; i < g.length; i++) idx.push(i);
        idx.sort(function (a, b) { return g[a] - g[b]; });
        return idx.slice(0, n).map(function (i) { var c = UA.gridConfig(i); return { i: i, cfg: c, pick: pickOf(c) }; });
      }

      function update() {
        save();
        var i = index(pick), B = M.balls[ball];
        var mean = B.grid.gpr[i], sd = B.grid.gpr_std[i];
        readout.innerHTML = "";
        readout.appendChild(h("p", { class: "readout-line", html: "With the " + ball + " ball, the Gaussian process expects these settings to reach 95 after about <b>" + F.int(Math.round(Math.exp(mean) / 1000) * 1000) + "</b> steps." }));
        readout.appendChild(h("p", { class: "readout-range", html:
          "A single run would land between <span class='num'>" + F.steps(Math.exp(mean - sd)) + "</span> and <span class='num'>" + F.steps(Math.exp(mean + sd)) + "</span> about two times in three, " +
          "and between <span class='num'>" + F.steps(Math.exp(mean - 1.96 * sd)) + "</span> and <span class='num'>" + F.steps(Math.exp(mean + 1.96 * sd)) + "</span> 95% of the time." +
          (Math.exp(mean + 1.96 * sd) > 500000 ? " The top of that range is past our 500k budget." : "") }));
        var preds = MODELS.map(function (m) { return { m: m, v: Math.exp(B.grid[m][i]) }; });
        var max = Math.max.apply(null, preds.map(function (p) { return p.v; }));
        var list = h("ul", { class: "model-list" });
        preds.forEach(function (p) {
          list.appendChild(h("li", null, h("span", { text: UA.MODEL_NAMES[p.m] }),
            h("span", null, h("span", { class: "bar", style: "width:" + Math.round(p.v / max * 120) + "px;" + (p.m === "gpr" ? "background:var(--ink)" : "") })),
            h("span", { class: "num", text: F.steps(p.v) })));
        });
        readout.appendChild(list);

        // Real runs with these exact settings
        var real = (UA.configs[keyOf(pick)] || []).slice().sort(function (a, b) { return (a.ball > b.ball ? 1 : -1) || a.seed - b.seed; });
        Array.prototype.forEach.call(realHost.querySelectorAll(".js-plotly-plot"), function (el) { Plotly.purge(el); });
        realHost.innerHTML = "";
        if (real.length) {
          realHost.appendChild(h("p", { class: "fig-title", text: "We ran these settings " + real.length + " times" }));
          var mine = real.filter(function (r) { return r.ball === ball && r.reward; });
          var realChart = UA.fig(realHost, { title: null, height: 200, file: "real-runs", caption: mine.length ? "Their " + UA.BALL_LABEL[ball].toLowerCase() + " curves. The shaded band is the predicted ±1 sd range." : "None with the " + ball + " ball." });
          var lo = Math.exp(mean - sd), hi = Math.exp(mean + sd);
          UA.plot(realChart, mine.map(function (r) {
            return { x: r.steps, y: r.reward, mode: "lines", line: { color: UA.ballColor(ball), width: 2, dash: r.seed === 3 ? "dash" : "solid" }, name: r.id,
              hovertemplate: r.id + "<br>step %{x:,}<br>reward %{y:.1f}<extra></extra>" };
          }), { xaxis: { range: [0, 505000], tickformat: "~s" }, yaxis: { range: [-2, 105] },
            shapes: [{ type: "rect", x0: lo, x1: Math.min(hi, 505000), y0: 0, y1: 1, yref: "paper", fillcolor: UA.ballSoft(ball), line: { width: 0 }, layer: "below" },
              { type: "line", x0: Math.exp(mean), x1: Math.exp(mean), y0: 0, y1: 1, yref: "paper", line: { color: UA.ballColor(ball), width: 1 } },
              { type: "line", x0: 0, x1: 505000, y0: 95, y1: 95, line: { color: UA.color("ink-3"), width: 1, dash: "dash" } }] });
          var tbl = h("div", { style: "margin-top:8px" });
          realHost.appendChild(tbl);
          UA.table(tbl, { rows: real, pageSize: 8, onRow: function (r) { UA.go("curves", { runs: r.id }); }, columns: [
            { key: "id", label: "run" }, { key: "ball", label: "ball" }, { key: "seed", label: "seed", num: true },
            { key: "steps_to_95", label: "actual steps to 95", num: true, fmt: function (v, r) { return r.reached ? F.int(v) : "not reached"; } }] });
        } else {
          realHost.appendChild(h("p", { class: "fig-title", text: "We never ran these exact settings" }));
          realHost.appendChild(h("p", { class: "fig-cap", text: "So this is a true prediction. Closest configs we did run:" }));
          var near = Object.keys(UA.configs).map(function (k) {
            var parts = k.split("|").map(Number), d = 0;
            feats.forEach(function (f, j) { if (parts[j] !== values[f][pick[j]]) d++; });
            return { k: k, d: d, runs: UA.configs[k] };
          }).sort(function (a, b) { return a.d - b.d; }).slice(0, 4);
          var ul = h("ul", { class: "model-list near" });
          ul.appendChild(h("li", { class: "label", style: "grid-template-columns:1fr" }, h("span", { text: "Settings that differ from yours, and steps to 95 of our " + ball + "-ball runs" })));
          near.forEach(function (n) {
            var r = n.runs.filter(function (x) { return x.ball === ball; });
            var parts = n.k.split("|").map(Number);
            var diff = feats.filter(function (f, j) { return parts[j] !== values[f][pick[j]]; }).map(function (f, _) { return UA.LABELS[f] + " " + F.setting(f, parts[feats.indexOf(f)]); }).join(", ");
            ul.appendChild(h("li", { style: "grid-template-columns:1fr auto;gap:2px 12px", tabindex: "0", title: "Load these settings", onkeydown: function (e) { if (e.key === "Enter") setPick(pickOf(n.runs[0])); }, onclick: function () { setPick(pickOf(n.runs[0])); } },
              h("span", { text: diff }), h("span", { class: "load", text: "Load" }),
              h("span", { class: "muted small", text: seedsText(r) })));
          });
          realHost.appendChild(ul);
        }

        // One setting at a time
        var allLo = Infinity, allHi = 0, rows = [];
        feats.forEach(function (f, fi) {
          var ys = [], sds = [];
          values[f].forEach(function (_, j) { var p = pick.slice(); p[fi] = j; var k = index(p); ys.push(B.grid.gpr[k]); sds.push(B.grid.gpr_std[k]); });
          rows.push({ ys: ys, sds: sds });
          ys.forEach(function (y, j) { allLo = Math.min(allLo, y - sds[j]); allHi = Math.max(allHi, y + sds[j]); });
        });
        feats.forEach(function (f, fi) {
          var r = rows[fi], c = UA.ballColor(ball), ink = UA.color("ink");
          UA.plot(smalls[fi], [{
            type: "scatter", mode: "lines+markers", x: values[f].map(function (v) { return F.setting(f, v); }), y: r.ys.map(Math.exp),
            error_y: { type: "data", symmetric: false, array: r.ys.map(function (y, j) { return Math.exp(y + r.sds[j]) - Math.exp(y); }), arrayminus: r.ys.map(function (y, j) { return Math.exp(y) - Math.exp(y - r.sds[j]); }), color: c, thickness: 1, width: 0 },
            line: { color: c, width: 1.5 }, marker: { size: values[f].map(function (_, j) { return j === pick[fi] ? 12 : 7; }), color: values[f].map(function (_, j) { return j === pick[fi] ? ink : c; }) },
            hovertemplate: UA.LABELS[f] + " %{x}<br>%{y:,.0f} steps<extra>click to choose</extra>"
          }], { xaxis: { type: "category" }, yaxis: { type: "log", range: [allLo / Math.LN10, allHi / Math.LN10], tickformat: "~s" } });
          if (!smalls[fi]._bound) {
            smalls[fi]._bound = true;
            smalls[fi].on("plotly_click", function (ev) { var p = ev.points && ev.points[0]; if (p) { pick[fi] = p.pointIndex; segs[fi].set(pick[fi]); update(); } });
          }
        });

        // Fastest configs
        fastHost.innerHTML = "";
        var top = fastest(10).map(function (t) {
          var real = (UA.configs[keyOf(t.pick)] || []).filter(function (r) { return r.ball === ball; });
          var row = { pick: t.pick, pred: Math.exp(B.grid.gpr[t.i]), sd: B.grid.gpr_std[t.i], ran: real.length ? real.map(function (r) { return r.reached ? F.steps(r.steps_to_95) : "not reached"; }).join(" / ") : "not run" };
          feats.forEach(function (f) { row[f] = t.cfg[f]; });
          return row;
        });
        var anyRan = top.some(function (r) { return r.ran !== "not run"; });
        if (!anyRan) fastHost.appendChild(h("p", { class: "fig-cap", style: "margin-bottom:12px", text: "We never ran any of these ten, so every number here is a prediction." }));
        UA.table(fastHost, { rows: top, pageSize: 10, onRow: function (r) { setPick(r.pick); window.scrollTo({ top: 0, behavior: "smooth" }); },
          rowClass: function (r) { return r.pick.join("") === pick.join("") ? "chosen" : ""; },
          columns: feats.map(function (f) { return { key: f, label: UA.SHORT[f], title: UA.LABELS[f], num: true, fmt: function (v) { return F.setting(f, v); } }; })
            .concat([{ key: "pred", label: "predicted", num: true, fmt: F.steps }])
            .concat(anyRan ? [{ key: "ran", label: "our runs (" + ball + ")", num: true }] : [])
            .concat([{ key: "load", label: "", fmt: function () { return h("span", { class: "load-link", text: "Load" }); } }]) });
      }
      update();
    }
  });
})();
