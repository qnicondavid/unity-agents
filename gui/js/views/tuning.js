(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt;

  // One entry per folder in analysis/results; a new model folder shows up under its own name.
  var MODEL_LABELS = { gpr: "Gaussian process", mlp: "MLP", boosting: "Gradient boosting", knn: "Nearest neighbours" };

  function tunings() {
    return Object.keys(UA.data.results.files).filter(function (k) { return /candidates\.csv$/.test(k); }).map(function (k) {
      var parts = k.split("/");
      var model = parts[0], config = parts[1], kind = parts[2];
      var label = model === "forest" ? (config.indexOf("reached_95") === 0 ? "Forest classifier" : "Regression forest") : (MODEL_LABELS[model] || model);
      return { path: k, dir: parts.slice(0, 3).join("/"), model: model, config: config, kind: kind, label: label,
        data: config.indexOf("group10") === 0 ? "our runs" : "Group 15" };
    });
  }

  // How many runs a result was scored on, from its folds.csv, so results from older data are marked.
  UA.scoredRuns = function (dir) {
    var folds = UA.result(dir + "/folds.csv");
    var n = 0;
    folds.forEach(function (r) { n += r.test_runs || 0; });
    return n;
  };
  UA.currentScored = function () { return UA.g10.filter(function (r) { return r.reached; }).length; };
  UA.versionText = function (dir) {
    var n = UA.scoredRuns(dir), cur = UA.currentScored();
    if (!n) return null;
    return n === cur ? "Scored on the current data: all " + F.int(n) + " runs that reached 95."
      : "Scored on an older version of the data, " + F.int(n) + " runs that reached 95 instead of today's " + F.int(cur) + ", so it does not compare directly with current results.";
  };

  UA.register({
    id: "tuning", num: "08", nav: "Tuning",
    render: function (main, params) {
      var all = tunings();
      UA.head(main, {
        num: "08", kicker: "Tuning",
        title: "Every setting we tried for every model",
        lede: "Each tuning tries every combination in a grid, scores it on one set of folds, picks the best mean, then reports it on fresh folds that played no part in the choice."
      });

      var labels = ["Gaussian process", "MLP", "Gradient boosting", "Nearest neighbours", "Regression forest", "Forest classifier"]
        .filter(function (l) { return all.some(function (t) { return t.label === l; }); });
      all.forEach(function (t) { if (labels.indexOf(t.label) < 0) labels.push(t.label); });
      var label = labels.indexOf(params.model) >= 0 ? params.model : "MLP";
      var options = all.filter(function (t) { return t.label === label; });
      var chosen = options.filter(function (t) { return t.path === params.run; })[0] || options[0];

      var ctl = h("div", { class: "controls" });
      main.appendChild(ctl);
      ctl.appendChild(UA.control("Model", UA.seg({ label: "Model", value: label, options: labels, onChange: function (v) { UA.go("tuning", { model: v }); } })));
      ctl.appendChild(UA.control("Experiment", UA.select({ label: "Experiment", value: chosen.path, options: options.map(function (t) {
        return { value: t.path, label: t.config + (t.kind === "tuned_two_se" ? " (two-SE rule)" : "") + " · " + t.data };
      }), onChange: function (v) { UA.go("tuning", { model: label, run: v }); } })));

      var rows = UA.result(chosen.path);
      var cols = UA.data.results.files[chosen.path].columns;
      var splitCol = cols[0];
      var scoreCol = cols.filter(function (c) { return /^tuning_/.test(c) && c !== "tuning_se"; })[0];
      var keys = cols.slice(1, cols.indexOf(scoreCol)).filter(function (c) { return c !== "tied_candidates"; });
      var lowerBetter = /error/.test(scoreCol);
      var splits = [];
      rows.forEach(function (r) { if (splits.indexOf(r[splitCol]) < 0) splits.push(r[splitCol]); });
      var split = splits.map(String).indexOf(params.split) >= 0 ? splits[splits.map(String).indexOf(params.split)] : splits[0];
      if (splits.length > 1) ctl.appendChild(UA.control(splitCol === "ball" ? "Ball" : "Budget", UA.seg({ label: "Split", mono: splitCol !== "ball", value: split,
        options: splits.map(function (s) { return { value: s, label: splitCol === "max_steps" ? F.steps(s) : UA.cap(String(s)) }; }),
        onChange: function (v) { UA.go("tuning", { model: label, run: chosen.path, split: v }); } })));

      var part = rows.filter(function (r) { return r[splitCol] === split; }).sort(function (a, b) { return a.rank - b.rank; });
      var pick = part.filter(function (r) { return r.chosen === true || r.chosen === "True"; })[0];
      var summary = UA.result(chosen.dir + "/summary.csv").filter(function (r) { return r[splitCol] === split; })[0] || {};
      var metricName = lowerBetter ? "mean absolute error" : "balanced accuracy";

      var s1 = UA.section(main, { title: F.int(part.length) + " combinations, ranked", text: chosen.data === "our runs" ? UA.versionText(chosen.dir) : null });
      var g = h("div", { class: "grid side-right" });
      s1.appendChild(g);
      var el = UA.fig(g, { title: null, height: 320, file: "tuning-ranked", caption: "Tuning score of each combination with ± one standard error, best first. The black dot is the choice.", source: "analysis/results/" + chosen.path });
      var scored = part.filter(function (r) { return r[scoreCol] != null; });
      UA.plot(el, [
        { type: "scatter", mode: "lines", x: scored.map(function (r) { return r.rank; }), y: scored.map(function (r) { return r[scoreCol] + r.tuning_se; }), line: { width: 0 }, hoverinfo: "skip" },
        { type: "scatter", mode: "lines", x: scored.map(function (r) { return r.rank; }), y: scored.map(function (r) { return r[scoreCol] - r.tuning_se; }), line: { width: 0 }, fill: "tonexty", fillcolor: UA.color("sunken"), hoverinfo: "skip" },
        { type: "scatter", mode: "markers", x: scored.map(function (r) { return r.rank; }), y: scored.map(function (r) { return r[scoreCol]; }),
          text: scored.map(function (r) { return keys.map(function (k) { return k + " " + r[k]; }).join("<br>"); }),
          marker: { color: UA.color("ink-3"), size: 4 }, hovertemplate: "rank %{x}<br>" + metricName + " %{y:.4f}<br>%{text}<extra></extra>" },
        pick ? { type: "scatter", mode: "markers", x: [pick.rank], y: [pick[scoreCol]], marker: { color: UA.color("ink"), size: 11, line: { color: UA.color("paper"), width: 2 } }, hoverinfo: "skip" } : {}
      ], { xaxis: part.length > 60
          ? { title: { text: "rank (log scale)" }, type: "log", tickvals: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000], ticktext: ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k"] }
          : { title: { text: "rank" } },
        yaxis: { title: { text: "tuning " + metricName } } });

      var side = h("div", { class: "prose small" });
      g.appendChild(side);
      var report = Object.keys(summary).filter(function (k) { return /_(mean_absolute_error|balanced_accuracy)$/.test(k) && !/^tuning_/.test(k); });
      side.innerHTML = "<p class='fig-title'>Best of this grid</p>" + (label === "MLP" && chosen.data === "our runs" ? "<p class='muted small'>The MLP in Model comparison uses the edges grid's choice, the last of three tunings below.</p>" : "") + keys.map(function (k) {
        return "<div style='display:flex;justify-content:space-between;border-bottom:1px solid var(--rule);padding:4px 0'><span>" + F.esc(k.replace(/_/g, " ")) + "</span><span class='num'>" + F.esc(pick ? pick[k] : "–") + "</span></div>";
      }).join("") + "<p class='fig-title' style='margin-top:16px'>Error on fresh folds</p>" + report.map(function (k) {
        var key = k.replace(/_(mean_absolute_error|balanced_accuracy)$/, "");
        var name = key === "forest" ? (lowerBetter ? "Regression forest" : "Our forest") : UA.MODEL_NAMES[key] || key.replace(/_/g, " ");
        return "<div style='display:flex;justify-content:space-between;border-bottom:1px solid var(--rule);padding:4px 0'><span>" + F.esc(name) + "</span><span class='num'>" + F.num(summary[k]) + "</span></div>";
      }).join("");

      // For each tuned setting: how often each value appears among the best tenth of combinations,
      // against how often it appears in the whole grid. A value the good combinations favour
      // has a dark bar longer than its grey one.
      var varied = keys.filter(function (k) {
        var s = {}; scored.forEach(function (r) { s[String(r[k])] = 1; });
        return Object.keys(s).length > 1;
      });
      var maxCats = Math.max.apply(null, varied.map(function (k) { var s = {}; scored.forEach(function (r) { s[String(r[k])] = 1; }); return Object.keys(s).length; }).concat([1]));
      if (varied.length && scored.length >= 10) {
        var topN = Math.max(1, Math.round(scored.length / 10));
        var top = scored.slice().sort(function (a, b) { return a.rank - b.rank; }).slice(0, topN);
        var s2 = UA.section(main, { title: "What the best tenth have in common",
          text: "The best " + topN + " of " + F.int(scored.length) + " combinations. Dark bars: how often each value appears among them. Outlined bars: how often it appears in the whole grid, which is what chance alone would give." });
        var g2 = h("div", { class: "grid " + (varied.length > 3 ? "four" : varied.length === 3 ? "three" : "two") });
        s2.appendChild(g2);
        varied.forEach(function (k) {
          var cats = [];
          scored.forEach(function (r) { if (cats.indexOf(String(r[k])) < 0) cats.push(String(r[k])); });
          cats.sort(function (a, b) {
            var x = Number(a), y = Number(b);
            if (!isNaN(x) && !isNaN(y)) return x - y;
            if (a === "none") return 1;
            if (b === "none") return -1;
            var la = (a.match(/\d+/g) || []).length, lb = (b.match(/\d+/g) || []).length;
            return la - lb || parseFloat(a.replace(/[^\d.]/g, "")) - parseFloat(b.replace(/[^\d.]/g, ""));
          });
          function share(rows, c) { return rows.filter(function (r) { return String(r[k]) === c; }).length / rows.length; }
          var chart = UA.fig(g2, { title: k.replace(/_/g, " "), small: true, height: 60 + maxCats * 34, file: "best-tenth-" + k });
          UA.plot(chart, [
            { type: "bar", orientation: "h", name: "whole grid", y: cats, x: cats.map(function (c) { return share(scored, c); }), marker: { color: UA.color("paper"), line: { color: UA.color("ink-3"), width: 1.5 } }, hovertemplate: "%{y}: %{x:.0%} of the grid<extra></extra>" },
            { type: "bar", orientation: "h", name: "best tenth", y: cats, x: cats.map(function (c) { return share(top, c); }), marker: { color: UA.color("ink") }, hovertemplate: "%{y}: %{x:.0%} of the best tenth<extra></extra>" }
          ], { barmode: "group", bargap: 0.25, bargroupgap: 0.05, xaxis: { tickformat: ".0%", range: [0, 1] }, yaxis: { type: "category", autorange: "reversed" } });
        });
      }

      // The grid that was searched.
      var cfg = UA.config(chosen.model + "/" + chosen.config + ".toml");
      if (cfg && cfg.tuning) {
        var s3 = UA.section(main, { title: "The grid", text: "From analysis/configs/" + chosen.model + "/" + chosen.config + ".toml." });
        var gridRows = Object.keys(cfg.tuning).filter(function (k) { return Array.isArray(cfg.tuning[k]); }).map(function (k) { return { key: k.replace(/_/g, " "), values: cfg.tuning[k].map(function (v) { return Array.isArray(v) ? "[" + v.join(", ") + "]" : String(v); }).join("   ") }; });
        var t = h("div");
        s3.appendChild(t);
        UA.table(t, { rows: gridRows, pageSize: 20, columns: [{ key: "key", label: "setting" }, { key: "values", label: "values tried" }] });
        s3.appendChild(h("p", { class: "fig-src", text: "Tuning folds: " + cfg.tuning.folds + " (seed " + cfg.tuning.seed + "). Report folds: " + cfg.tuning.report_folds + " (seed " + cfg.tuning.report_seed + ")." }));
      }

      if (label === "MLP" && chosen.data === "our runs") mlpStory(main);
    }
  });

  function mlpStory(main) {
    var steps = [
      { config: "group10_steps_to_95", name: "first grid" },
      { config: "group10_steps_to_95_extended", name: "extended grid" },
      { config: "group10_steps_to_95_edges", name: "edges grid" }
    ];
    var s = UA.section(main, { title: "How the MLP was tuned",
      text: "The first grid chose values at its own edges on both balls, such as the smallest network. The extended grid tried one value past each edge and added dropout: 1,125 combinations per ball. " +
        "Its choices sat at new edges, so the edges grid tried 54 combinations around and past them. Those are the final MLP settings." });
    var rows = [];
    var current = UA.currentScored();
    steps.forEach(function (st) {
      var dir = "mlp/" + st.config + "/tuned_best_mean";
      var n = UA.scoredRuns(dir);
      UA.result(dir + "/summary.csv").forEach(function (r) {
        rows.push({ step: st.name, ball: r.ball, net: r.hidden_sizes, lr: r.learning_rate, batch: r.batch_size, epochs: r.epochs, dropout: r.dropout, mae: r.mlp_mean_absolute_error, runs: n });
      });
    });
    var t = h("div");
    s.appendChild(t);
    UA.table(t, { rows: rows, pageSize: 10, columns: [
      { key: "step", label: "grid" }, { key: "ball", label: "ball" }, { key: "net", label: "hidden layers" }, { key: "lr", label: "lr", title: "learning rate", num: true },
      { key: "batch", label: "batch", num: true }, { key: "epochs", label: "epochs", num: true }, { key: "dropout", label: "dropout", num: true },
      { key: "mae", label: "report error", num: true, fmt: function (v) { return F.num(v); } },
      { key: "runs", label: "runs scored", num: true, fmt: function (v) { return F.int(v) + (v === current ? "" : " (older data)"); } }] });
    var older = steps.filter(function (st) { var n = UA.scoredRuns("mlp/" + st.config + "/tuned_best_mean"); return n && n !== current; }).map(function (st) { return st.name; });
    if (older.length) s.appendChild(h("p", { class: "note", text: "The " + older.join(" and ") + " " + (older.length > 1 ? "were" : "was") + " scored on an older version of the data, with fewer runs, so " +
      (older.length > 1 ? "their errors do" : "its errors do") + " not compare directly with the others. Model comparison retrains the final settings on all current runs." }));
  }
})();
