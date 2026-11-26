(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt, S = UA.stats;
  var ORDER = ["gpr", "mlp", "forest", "ridge", "median"];
  var BUDGETS = [500000, 1000000, 2000000];

  function maeTicks(lo, hi) {
    var vals = [], t;
    for (t = Math.ceil(lo * 20) / 20; t <= hi + 1e-9; t += 0.05) vals.push(+t.toFixed(2));
    return { tickvals: vals, ticktext: vals.map(function (v) { return v.toFixed(2) + "<br><span style='font-size:0.9em'>" + F.offBy(v) + " off</span>"; }) };
  }
  function truthy(v) { return v === true || v === "True" || v === "true"; }

  UA.register({
    id: "models", num: "04", nav: "Model comparison",
    render: function (main, params) {
      var M = UA.models;
      UA.head(main, {
        num: "04", kicker: "Model comparison", tag: "SQ2",
        title: "Prediction error by model",
        lede: "Every score here comes from grouped cross-validation: the runs of a config are tested by a model that never saw that config. " +
          "Errors are in natural-log steps: the best model's 0.206 means a typical prediction is about 23% off."
      });

      /* A. Regression scores -------------------------------------------------------------------- */
      var data = params.data === "g15" ? "g15" : "g10";
      var sA = UA.section(main, { id: "scores", title: "Mean absolute error on unseen configs" });
      var ctl = h("div", { class: "controls" });
      sA.appendChild(ctl);
      ctl.appendChild(UA.control("Runs", UA.seg({ label: "Runs", value: data, options: [{ value: "g10", label: "Our runs" }, { value: "g15", label: "Group 15" }],
        onChange: function (v) { UA.go("models", { data: v === "g15" ? "g15" : null, section: "scores" }); } })));
      var holder = h("div");
      sA.appendChild(holder);
      if (data === "g10") scoresG10(holder, M); else scoresG15(holder, ctl, params);

      /* B. Fold by fold ------------------------------------------------------------------------ */
      var sB = UA.section(main, { id: "folds", title: "Fold by fold", text: "Each dot is one of the 20 test folds (10 per ball). Below the diagonal, the Gaussian process had the lower error on that fold." });
      foldByFold(sB, M, params);

      /* C. Predicted against actual -------------------------------------------------------------- */
      var sC = UA.section(main, { id: "oof", title: "Predicted against actual", text: "Every run, predicted while its config was held out. Dots on the diagonal are perfect; the dashed lines mark the model's own typical error." });
      if (M) predictedVsActual(sC, M, params);
      else sC.appendChild(h("div", { class: "empty", text: "gui/data/models.js is missing. Run python -m gui.export_data to compute the predictions." }));

      /* D. The forest classifier ---------------------------------------------------------------- */
      var sD = UA.section(main, { id: "forest", title: "Reaching 95 within the budget", text: "Our ID3 forest predicts reached_95 for Group 15's runs, which had budgets of 500k, 1M and 2M steps. The score is balanced accuracy: always guessing the more common outcome scores 0.5." });
      forestClassifier(sD, params);
    }
  });

  function scoresG10(holder, M) {
    var rows = [];
    if (M) {
      ORDER.forEach(function (m) {
        UA.BALLS.forEach(function (b) { var s = M.balls[b].scores[m]; rows.push({ model: m, ball: b, mae: s.mae, se: s.se }); });
      });
    } else {
      var g = UA.result("gpr/group10_steps_to_95/tuned_best_mean/summary.csv");
      var m = UA.result("mlp/group10_steps_to_95/tuned_best_mean/summary.csv");
      UA.BALLS.forEach(function (b) {
        var gr = g.filter(function (r) { return r.ball === b; })[0], mr = m.filter(function (r) { return r.ball === b; })[0];
        rows.push({ model: "gpr", ball: b, mae: gr.gpr_mean_absolute_error, se: gr.gpr_se });
        ["mlp", "ridge", "median"].forEach(function (k) { rows.push({ model: k, ball: b, mae: mr[k + "_mean_absolute_error"], se: mr[k + "_se"] }); });
      });
    }
    var grid = h("div", { class: "grid side-right" });
    holder.appendChild(grid);
    var el = UA.fig(grid, { title: "Error by model", caption: "Dots are the mean over 10 test folds; bars are ± one standard error. Lower is better.", height: 330, source: M ? "gui/data/models.js (recomputed on the report folds, seed 11)" : "analysis/results/*/group10_steps_to_95/tuned_best_mean/summary.csv" });
    var models = ORDER.filter(function (m) { return rows.some(function (r) { return r.model === m; }); });
    var ticks = maeTicks(0.15, 0.45);
    UA.plot(el, UA.BALLS.map(function (b, i) {
      var part = models.map(function (m) { return rows.filter(function (r) { return r.model === m && r.ball === b; })[0]; });
      return { type: "scatter", mode: "markers", name: UA.BALL_LABEL[b],
        y: models.map(function (m) { return UA.MODEL_NAMES[m]; }), x: part.map(function (r) { return r.mae; }),
        error_x: { type: "data", array: part.map(function (r) { return r.se; }), color: UA.ballColor(b), thickness: 1.5, width: 0 },
        marker: { color: UA.ballColor(b), size: 10, symbol: UA.ballSymbol(b) },
        offsetgroup: b, customdata: part.map(function (r) { return [r.se, F.offBy(r.mae)]; }),
        hovertemplate: "%{y} · " + UA.BALL_LABEL[b].toLowerCase() + "<br>MAE %{x:.3f} ± %{customdata[0]:.3f}<br>typically %{customdata[1]} off<extra></extra>" };
    }), { showlegend: true, scattermode: "group", scattergap: 0.5,
      xaxis: { title: { text: "mean absolute error (log steps), and how far off that typically is" }, range: [0.15, 0.45], tickvals: ticks.tickvals, ticktext: ticks.ticktext },
      yaxis: { autorange: "reversed", type: "category" } });

    var side = h("div", { class: "prose small" });
    grid.appendChild(side);
    var best = M ? M.balls.fixed.scores.gpr.mae : rows[0].mae;
    side.innerHTML =
      "<p>The median baseline ignores the settings and predicts the median of the training runs. " +
      "Ridge is a straight line through the logged settings. The gap between them is what the settings explain linearly; the gap from ridge to the Gaussian process is what needs a curve.</p>" +
      (M ? "<p>Each model uses its tuned settings: Gaussian process " + M.balls.fixed.settings.gpr.kind + " with one length scale per setting; " +
        "MLP one hidden layer of " + M.balls.fixed.settings.mlp.hidden_sizes.join("×") + ", averaged over 3 seeds; forest " + M.balls.fixed.settings.forest.trees + " trees (fixed ball). " +
        "All five were scored on the same 2,400 runs and the same folds. The Gaussian process, ridge and median scores match <span class='mono'>analysis/results</span> exactly; the MLP and forest were retrained here on all 2,400 runs.</p>" : "") +
      "<p>Best fixed-ball error: <span class='num'>" + F.num(best) + "</span>, about " + F.offBy(best) + " off.</p>";
  }

  function scoresG15(holder, ctl, params) {
    var feats = params.feats === "5" ? "5" : "12";
    ctl.appendChild(UA.control("Settings used", UA.seg({ label: "Settings used", value: feats, options: [{ value: "5", label: "5 settings" }, { value: "12", label: "All 12" }],
      onChange: function (v) { UA.go("models", { data: "g15", feats: v === "5" ? "5" : null, section: "scores" }); } })));
    var split = UA.result("mlp/steps_to_95_" + feats + "_settings/tuned_best_mean/summary.csv");
    var pooled = UA.result("mlp/steps_to_95_" + feats + "_settings_pooled/tuned_best_mean/summary.csv")[0];
    var models = ["mlp", "scikit_learn", "ridge", "median"];
    var names = { mlp: "MLP", scikit_learn: "scikit-learn MLP", ridge: "Ridge (linear)", median: "Median baseline" };
    var el = UA.fig(holder, { title: "Error by model and budget (one model per budget)", caption: "Group 15's runs that reached 95: 805 of 1,555. The pooled MLP, trained on all budgets at once with the budget as a feature, scores " + F.num(pooled.mlp_mean_absolute_error) + " overall.", height: 320, source: "analysis/results/mlp/steps_to_95_" + feats + "_settings/tuned_best_mean/summary.csv" });
    var colors = ["#7a828b", "#454c55", "#15171a"]; // budgets in greys: blue means the fixed ball
    if (document.documentElement.getAttribute("data-theme") === "dark" || (!document.documentElement.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches)) colors = ["#6b727b", "#a7adb5", "#eef0f2"];
    var ticks = maeTicks(0.1, 0.65);
    UA.plot(el, BUDGETS.map(function (b, i) {
      var row = split.filter(function (r) { return r.max_steps === b; })[0];
      return { type: "scatter", mode: "markers", name: F.steps(b) + " budget", y: models.map(function (m) { return names[m]; }),
        x: models.map(function (m) { return row[m + "_mean_absolute_error"]; }),
        error_x: { type: "data", array: models.map(function (m) { return row[m + "_se"]; }), color: colors[i], thickness: 1.5, width: 0 },
        marker: { color: colors[i], size: 10 }, hovertemplate: "%{y}<br>MAE %{x:.3f}<extra>" + F.steps(b) + "</extra>" };
    }), { showlegend: true, scattermode: "group", scattergap: 0.4,
      xaxis: { title: { text: "mean absolute error (log steps), and how far off that typically is" }, range: [0.1, 0.65], tickvals: ticks.tickvals, ticktext: ticks.ticktext },
      yaxis: { autorange: "reversed", type: "category" } });
  }

  function foldByFold(section, M, params) {
    var other = ["mlp", "forest", "ridge"].indexOf(params.vs) >= 0 ? params.vs : "mlp";
    var ctl = h("div", { class: "controls" });
    section.appendChild(ctl);
    var points = [];
    if (M) {
      ctl.appendChild(UA.control("Compare with", UA.seg({ label: "Compare with", value: other, options: ["mlp", "forest", "ridge"].map(function (m) { return { value: m, label: UA.MODEL_NAMES[m] }; }),
        onChange: function (v) { UA.setParams({ vs: v === "mlp" ? null : v }); other = v; draw(); } })));
    }
    var grid = h("div", { class: "grid side-right" });
    section.appendChild(grid);
    var el = UA.fig(grid, { title: null, height: 360, file: "folds-gpr-vs-other", source: M ? "gui/data/models.js" : "analysis/results/{gpr,mlp}/group10_steps_to_95/tuned_best_mean/folds.csv" });
    var side = h("div");
    grid.appendChild(side);
    function draw() {
      points = [];
      if (M) {
        UA.BALLS.forEach(function (b) {
          var g = M.balls[b].scores.gpr.folds, o = M.balls[b].scores[other].folds;
          g.forEach(function (v, i) { points.push({ ball: b, fold: i + 1, gpr: v, other: o[i] }); });
        });
      } else {
        var gf = UA.result("gpr/group10_steps_to_95/tuned_best_mean/folds.csv"), mf = UA.result("mlp/group10_steps_to_95/tuned_best_mean/folds.csv");
        gf.forEach(function (r) {
          var o = mf.filter(function (x) { return x.ball === r.ball && x.fold === r.fold; })[0];
          if (o) points.push({ ball: r.ball, fold: r.fold, gpr: r.gpr_mean_absolute_error, other: o.mlp_mean_absolute_error });
        });
      }
      var all = points.map(function (p) { return p.gpr; }).concat(points.map(function (p) { return p.other; }));
      var lo = Math.min.apply(null, all) - 0.01, hi = Math.max.apply(null, all) + 0.01;
      var wins = points.filter(function (p) { return p.gpr < p.other; }).length;
      UA.plot(el, UA.BALLS.map(function (b) {
        var part = points.filter(function (p) { return p.ball === b; });
        return { type: "scatter", mode: "markers", name: UA.BALL_LABEL[b], x: part.map(function (p) { return p.other; }), y: part.map(function (p) { return p.gpr; }),
          marker: { color: UA.ballColor(b), size: 10, symbol: UA.ballSymbol(b), line: { color: UA.color("paper"), width: 1 } }, text: part.map(function (p) { return "fold " + p.fold; }),
          hovertemplate: "%{text} · " + UA.BALL_LABEL[b].toLowerCase() + "<br>" + UA.MODEL_NAMES[other] + " %{x:.3f}<br>Gaussian process %{y:.3f}<extra></extra>" };
      }), { showlegend: true,
        xaxis: { title: { text: UA.MODEL_NAMES[other] + " error on the fold" }, range: [lo, hi], tickformat: ".2f" },
        yaxis: { title: { text: "Gaussian process error on the fold" }, range: [lo, hi], tickformat: ".2f" },
        shapes: [{ type: "line", x0: lo, y0: lo, x1: hi, y1: hi, line: { color: UA.color("ink-3"), width: 1, dash: "dot" } }] });
      var diffs = points.map(function (p) { return p.other - p.gpr; });
      var name = UA.inText(other);
      side.innerHTML = "";
      side.appendChild(h("div", { class: "side-text", html:
        "<p>The Gaussian process has the lower error on <b>" + wins + " of " + points.length + "</b> folds against the " + F.esc(name) + ".</p>" +
        "<p>On average it is <b>" + F.num(S.mean(diffs)) + "</b> lower, give or take " + F.num(S.se(diffs)) + " (one standard error).</p>" +
        "<p class='muted small'>Both models are tested on exactly the same folds, so comparing them fold by fold is stricter than comparing two averages.</p>" }));
    }
    draw();
  }

  function predictedVsActual(section, M, params) {
    var model = ORDER.indexOf(params.model) >= 0 ? params.model : "gpr";
    var ball = params.ball === "random" ? "random" : "fixed";
    var ctl = h("div", { class: "controls" });
    section.appendChild(ctl);
    ctl.appendChild(UA.control("Model", UA.seg({ label: "Model", value: model, options: ORDER.map(function (m) { return { value: m, label: UA.MODEL_NAMES[m] }; }),
      onChange: function (v) { model = v; UA.setParams({ model: v === "gpr" ? null : v }); draw(); } })));
    ctl.appendChild(UA.control("Ball", UA.seg({ label: "Ball", value: ball, options: [{ value: "fixed", label: "Fixed" }, { value: "random", label: "Random" }],
      onChange: function (v) { ball = v; UA.setParams({ ball: v === "fixed" ? null : v }); draw(); } })));
    var grid = h("div", { class: "grid side-right" });
    section.appendChild(grid);
    var el = UA.fig(grid, { title: null, height: 440, file: "predicted-vs-actual", source: "gui/data/models.js, out-of-fold predictions" });
    var sideWrap = h("div"), side = h("div");
    sideWrap.appendChild(side);
    grid.appendChild(sideWrap);
    var resid = UA.fig(sideWrap, { title: "Error of each prediction", height: 170, file: "residuals" });
    resid.parentNode.style.marginTop = "24px";

    function draw() {
      var B = M.balls[ball], pred = B.oof[model];
      var xs = [], ys = [], ids = [], errs = [];
      B.ids.forEach(function (id, i) {
        var r = UA.byId[id];
        xs.push(r.steps_to_95); ys.push(Math.exp(pred[i])); ids.push(id); errs.push(pred[i] - Math.log(r.steps_to_95));
      });
      var mae = S.mean(errs.map(Math.abs)), k = Math.exp(mae);
      var lo = 8000, hi = 600000;
      UA.plot(el, [{
        type: "scatter", mode: "markers", x: xs, y: ys, text: ids,
        marker: { color: UA.ballColor(ball), size: 5, opacity: 0.45, symbol: UA.ballSymbol(ball) },
        hovertemplate: "<b>%{text}</b><br>actual %{x:,}<br>predicted %{y:,.0f}<extra>click to open</extra>"
      }], {
        xaxis: { title: { text: "actual steps to 95" }, type: "log", range: [Math.log10(lo), Math.log10(hi)], tickvals: [10000, 20000, 50000, 100000, 200000, 500000], ticktext: ["10k", "20k", "50k", "100k", "200k", "500k"] },
        yaxis: { title: { text: "predicted steps to 95" }, type: "log", range: [Math.log10(lo), Math.log10(hi)], tickvals: [10000, 20000, 50000, 100000, 200000, 500000], ticktext: ["10k", "20k", "50k", "100k", "200k", "500k"] },
        shapes: [
          { type: "line", x0: lo, y0: lo, x1: hi, y1: hi, line: { color: UA.color("ink"), width: 1 } },
          { type: "line", x0: lo, y0: lo * k, x1: hi, y1: hi * k, line: { color: UA.color("ink-3"), width: 1, dash: "dash" } },
          { type: "line", x0: lo, y0: lo / k, x1: hi, y1: hi / k, line: { color: UA.color("ink-3"), width: 1, dash: "dash" } }
        ]
      });
      el.removeAllListeners && el.removeAllListeners("plotly_click");
      el.on("plotly_click", function (ev) { var p = ev.points && ev.points[0]; if (p) UA.go("curves", { runs: p.text }); });

      var over = errs.filter(function (e) { return e > 0; }).length / errs.length;
      var html = "<p>Over " + F.int(xs.length) + " " + ball + "-ball runs, the " + F.esc(UA.inText(model)) + " is off by <b>" + F.num(mae) + "</b> on average, so a typical prediction is " + F.offBy(mae) + " off.</p>" +
        "<p>" + F.pct(over) + " of its predictions are too high. Runs that finish very early or very late get pulled toward the middle, because nothing in the settings says which seed got lucky.</p>";
      if (model === "gpr") {
        var std = B.oof.gpr_std;
        var in1 = 0, in2 = 0;
        errs.forEach(function (e, i) { if (Math.abs(e) <= std[i]) in1++; if (Math.abs(e) <= 1.96 * std[i]) in2++; });
        html += "<p>Its own uncertainty can be checked the same way: the ±1 sd range around each prediction holds <b>" + F.pct(in1 / errs.length) + "</b> of the actual results (it should hold 68%), and the 95% range holds <b>" + F.pct(in2 / errs.length) + "</b>.</p>";
      }
      side.innerHTML = "";
      side.appendChild(h("div", { class: "side-text", html: html }));
      UA.plot(resid, [{ type: "histogram", x: errs, xbins: { size: 0.05 }, marker: { color: UA.ballColor(ball) }, hovertemplate: "%{x}<br>%{y} runs<extra></extra>" }], {
        bargap: 0.05, xaxis: { title: { text: "log(predicted ÷ actual)" }, range: [-1.5, 1.5], zeroline: true, zerolinecolor: UA.color("ink-3") }, yaxis: { title: { text: "runs" } } });
    }
    draw();
  }

  function forestClassifier(section, params) {
    var feats = params.ffeats === "5" ? "5" : "12";
    var how = params.fhow === "pooled" ? "pooled" : params.fhow === "two_se" ? "two_se" : "split";
    var budget = BUDGETS.indexOf(Number(params.fbudget)) >= 0 ? Number(params.fbudget) : 500000;
    var ctl = h("div", { class: "controls" });
    section.appendChild(ctl);
    function go(p) { UA.go("models", UA.merge(UA.params, UA.merge(p, { section: "forest" }))); }
    ctl.appendChild(UA.control("Settings used", UA.seg({ label: "Settings used", value: feats, options: [{ value: "5", label: "5 settings" }, { value: "12", label: "All 12" }], onChange: function (v) { go({ ffeats: v === "5" ? "5" : null }); } })));
    ctl.appendChild(UA.control("Forest", UA.seg({ label: "Forest", value: how, options: [
      { value: "split", label: "One per budget", title: "Tuned for the best mean score" },
      { value: "two_se", label: "One per budget, two-SE rule", title: "The first tuning, which picked the simplest forest within two standard errors" },
      { value: "pooled", label: "Pooled", title: "One forest over all budgets, with the budget as a feature" }], onChange: function (v) { go({ fhow: v === "split" ? null : v }); } })));

    var base = "forest/reached_95_" + feats + "_settings" + (how === "pooled" ? "_pooled" : "") + "/" + (how === "two_se" ? "tuned_two_se" : "tuned_best_mean") + "/";
    var rows = [];
    if (how === "pooled") {
      var folds = UA.result(base + "folds_by_max_steps.csv");
      BUDGETS.forEach(function (b) {
        var part = folds.filter(function (r) { return r.max_steps === b; });
        var f = part.map(function (r) { return r.forest_balanced_accuracy; }), s = part.map(function (r) { return r.scikit_learn_balanced_accuracy; });
        rows.push({ budget: b, forest: S.mean(f), forest_se: S.se(f), sk: S.mean(s), sk_se: S.se(s) });
      });
    } else {
      UA.result(base + "summary.csv").forEach(function (r) {
        rows.push({ budget: r.max_steps, forest: r.forest_balanced_accuracy, forest_se: r.forest_se, sk: r.scikit_learn_balanced_accuracy, sk_se: r.scikit_learn_se, settings: r });
      });
    }

    var grid = h("div", { class: "grid two" });
    section.appendChild(grid);
    var el = UA.fig(grid, { title: "Balanced accuracy by budget", caption: "Our ID3 forest against scikit-learn's forest on the same folds, ± one standard error.", height: 300, source: "analysis/results/" + base + (how === "pooled" ? "folds_by_max_steps.csv" : "summary.csv") });
    var ink = UA.color("ink"), ink3 = UA.color("ink-3");
    UA.plot(el, [
      { type: "scatter", mode: "markers", name: "Our forest", x: rows.map(function (r) { return F.steps(r.budget); }), y: rows.map(function (r) { return r.forest; }),
        error_y: { type: "data", array: rows.map(function (r) { return r.forest_se; }), color: ink, thickness: 1.5, width: 0 }, marker: { color: ink, size: 11 }, hovertemplate: "our forest %{y:.3f}<extra>%{x}</extra>" },
      { type: "scatter", mode: "markers", name: "scikit-learn check", x: rows.map(function (r) { return F.steps(r.budget); }), y: rows.map(function (r) { return r.sk; }),
        error_y: { type: "data", array: rows.map(function (r) { return r.sk_se; }), color: ink3, thickness: 1.5, width: 0 }, marker: { color: UA.color("paper"), size: 10, line: { color: ink3, width: 2 } }, hovertemplate: "scikit-learn %{y:.3f}<extra>%{x}</extra>" }
    ], { showlegend: true, scattermode: "group", scattergap: 0.6,
      xaxis: { type: "category", title: { text: "budget" } },
      yaxis: { title: { text: "balanced accuracy" }, range: [0.45, 1], tickformat: ".2f" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: ink3, width: 1, dash: "dot" } }],
      annotations: [{ xref: "paper", x: 1, y: 0.5, xanchor: "right", yanchor: "bottom", text: "guessing the common outcome", showarrow: false, font: { size: 11, color: ink3 } }] });

    var right = h("div");
    grid.appendChild(right);
    right.appendChild(h("div", { class: "controls", style: "margin-bottom:8px" }, UA.control("Confusion matrix for budget", UA.seg({ label: "Budget", mono: true, value: budget,
      options: BUDGETS.map(function (b) { return { value: b, label: F.steps(b) }; }), onChange: function (v) { go({ fbudget: v === 500000 ? null : v }); } }))));
    var conf = UA.result(base + (how === "pooled" ? "confusion_by_max_steps.csv" : "confusion.csv")).filter(function (r) { return r.max_steps === budget; });
    var labels = ["False", "True"], names = ["not reached", "reached"];
    var z = labels.map(function (t) { return labels.map(function (p) {
      var row = conf.filter(function (r) { return String(truthy(r.true)) === t.toLowerCase() && String(truthy(r.predicted)) === p.toLowerCase(); })[0];
      return row ? row.count : 0;
    }); });
    var rowShare = z.map(function (row) { var s = row[0] + row[1]; return row.map(function (v) { return s ? v / s : 0; }); });
    var cm = UA.fig(right, { title: null, height: 240, file: "confusion-" + budget, source: "analysis/results/" + base + (how === "pooled" ? "confusion_by_max_steps.csv" : "confusion.csv") });
    UA.plot(cm, [{ type: "heatmap", z: rowShare, x: names.map(function (n) { return "predicted " + n; }), y: names.map(function (n) { return "actually " + n; }),
      colorscale: [[0, UA.color("surface")], [1, UA.color("ink-2")]], zmin: 0, zmax: 1, showscale: false, xgap: 2, ygap: 2,
      text: z.map(function (row, i) { return row.map(function (v, j) { return v + " runs (" + F.pct(rowShare[i][j]) + ")"; }); }),
      texttemplate: "%{text}", textfont: { family: "Atkinson Next, Segoe UI, sans-serif", size: 13 }, hovertemplate: "%{y}, %{x}: %{text}<extra></extra>" }],
      { margin: { l: 8, r: 8, t: 8, b: 8 }, xaxis: { side: "top", ticks: "", showgrid: false, tickfont: { family: "Atkinson Next, Segoe UI, sans-serif", size: 12 } }, yaxis: { autorange: "reversed", ticks: "", showgrid: false, tickfont: { family: "Atkinson Next, Segoe UI, sans-serif", size: 12 } } });
    var acc = rows.filter(function (r) { return r.budget === budget; })[0];
    var st = acc && acc.settings;
    right.appendChild(h("p", { class: "prose small", html: "Balanced accuracy at " + F.steps(budget) + ": <span class='num'>" + F.num(acc.forest) + "</span>." +
      (st ? " Chosen forest: " + st.trees + " trees, " + st.attributes_per_tree + " settings per tree, depth " + st.max_depth + "." : "") }));
    if (how === "two_se") section.appendChild(h("p", { class: "note", html: "The first tuning picked the simplest forest within two standard errors of the best. On this data that chose forests too shallow to score well, so the final tuning picks the best mean instead." }));
  }
})();
