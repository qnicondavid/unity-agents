(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt, S = UA.stats;

  // Settings ordered by relevance (1 / length scale), averaged over both balls.
  UA.relevanceOrder = function () {
    var rows = UA.result("gpr/group10_steps_to_95/tuned_best_mean/length_scales.csv");
    function rel(f) { return UA.stats.mean(rows.map(function (r) { return 1 / r[f]; })); }
    return UA.FEATURES.slice().sort(function (a, b) { return rel(b) - rel(a); });
  };

  function foldWins() {
    // Same source as Model comparison: the final MLP, retrained on all current runs.
    if (UA.models) {
      var w = 0, k = 0;
      UA.BALLS.forEach(function (b) {
        var s = UA.models.balls[b].scores;
        s.gpr.folds.forEach(function (v, i) { k++; if (v < s.mlp.folds[i]) w++; });
      });
      return { wins: w, n: k };
    }
    var g = UA.result("gpr/group10_steps_to_95/tuned_best_mean/folds.csv");
    var m = UA.result("mlp/group10_steps_to_95/tuned_best_mean/folds.csv");
    var wins = 0, n = 0;
    g.forEach(function (r) {
      var o = m.filter(function (x) { return x.ball === r.ball && x.fold === r.fold; })[0];
      if (!o) return;
      n++;
      if (r.gpr_mean_absolute_error < o.mlp_mean_absolute_error) wins++;
    });
    return { wins: wins, n: n };
  }

  UA.register({
    id: "overview", num: "01", nav: "Overview",
    render: function (main) {
      var g10 = UA.g10;
      var configs = Object.keys(UA.configs).length;
      var batches = {}, machines = {};
      g10.forEach(function (r) { batches[r.batch] = 1; machines[r.machine] = 1; });
      var withCurve = g10.filter(function (r) { return r.reward; }).length;

      UA.head(main, {
        num: "01", kicker: "Overview",
        title: "Can PPO settings predict how long training takes?",
        lede: "We trained Unity's 3DBall agent " + F.int(g10.length) + " times with PPO, drawing eight hyperparameters at random, " +
          "and recorded how many steps each run needed to reach a mean reward of 95. Then we asked how well models can predict that number from the settings alone."
      });

      var gif = h("figure", { class: "clip" },
        h("div", { class: "frame" }, h("img", { src: "../docs/assets/hero.gif", alt: "3DBall agents tilting their platforms to keep a ball on top", width: "678", height: "381" })),
        h("figcaption", { text: "3DBall agents tilt their platform to keep a ball on top. Reward 100 means a full episode without dropping it." }));
      var right = h("div");
      var hero = h("div", { class: "hero" }, gif, right);

      right.appendChild(h("p", { class: "facts", text:
        "We logged " + F.int(g10.length) + " runs of our own: " + F.int(configs) + " distinct configurations, drawn in " + Object.keys(batches).length + " batches and trained on " +
        Object.keys(machines).length + " laptops. For the forest classifier we also reused " + F.int(UA.g15.length) + " runs from Group 15." }));
      right.appendChild(h("p", { class: "prose small", style: "margin-top:16px", html:
        "Each configuration ran with seeds 2 and 3, once with the standard ball and once with a ball that changes mass and size every episode. " +
        "Budget: 500,000 steps per run." + (withCurve < g10.length ? " " + F.int(withCurve) + " runs have a full training curve." : " Every run has its full training curve.") }));

      // The three sub-questions, each answered from the data on disk.
      var gpr = UA.result("gpr/group10_steps_to_95/tuned_best_mean/summary.csv");
      var gf = gpr.filter(function (r) { return r.ball === "fixed"; })[0] || {};
      var gr = gpr.filter(function (r) { return r.ball === "random"; })[0] || {};
      var rank = UA.relevanceOrder();
      var wins = foldWins();

      var nsF = UA.noiseShare("fixed"), nsR = UA.noiseShare("random");
      var s = UA.section(main, { title: "What we found", text: "Our three sub-questions, answered from the runs." });
      var found = h("ol", { class: "found" });
      found.appendChild(answer("Which hyperparameters have the greatest influence?",
        UA.cap(UA.LABELS[rank[0]]) + " and " + UA.LABELS[rank[1]] + " change the prediction fastest; " + UA.LABELS[rank[rank.length - 1]] + " barely changes it.",
        "From the length scales the Gaussian process learned, averaged over both balls.", "importance", "What matters"));
      found.appendChild(answer("How accurately can models predict the steps?",
        "The Gaussian process does best. A typical prediction is about " + F.offBy(gf.gpr_mean_absolute_error) + " off, and it beats the MLP on " + wins.wins + " of " + wins.n + " test folds.",
        "Mean absolute error " + F.num(gf.gpr_mean_absolute_error) + " (fixed ball) and " + F.num(gr.gpr_mean_absolute_error) + " (random ball), in log steps, on configs the model never saw.", "models", "Model comparison"));
      found.appendChild(answer("How reliable is that, given the variation between runs?",
        "About " + F.pct(nsF.share) + " of the fixed ball's error and " + F.pct(nsR.share) + " of the random ball's is the gap between two seeds of the same settings, which no model of the settings can remove.",
        "A perfect model of the settings would still be off by " + F.num(nsF.floor) + " and " + F.num(nsR.floor) + ".", "noise", "Noise and machines"));
      s.appendChild(found);

      var sx = UA.section(main, { title: "The experiment" });
      sx.appendChild(hero);

      // How one run becomes one data point.
      var s2 = UA.section(main, { title: "How a run becomes a data point" });
      s2.appendChild(h("div", { class: "prose" }, h("p", { text:
        "A batch seed picks 8 settings from a grid of 15,552 combinations. PPO trains on them for 500,000 steps, and ML-Agents logs the mean reward every 12,000 steps. " +
        "The first logged step with a mean reward of 95 or more is the run's target. The models then learn that number from the 8 settings, and are always tested on configs they never trained on." })));

      var example = g10.filter(function (r) { return r.reached && r.reward && r.ball === "fixed"; });
      var sortedSteps = S.sorted(example.map(function (r) { return r.steps_to_95; }));
      var med = S.quantile(sortedSteps, 0.5);
      var pick = example.filter(function (r) { return r.steps_to_95 === med; })[0] || example[0];
      var wrap = h("div", { class: "grid side-right", style: "margin-top:32px" });
      s2.appendChild(wrap);
      var chart = UA.fig(wrap, { title: "One run: " + pick.id, caption: "A median fixed-ball run. It first logs 95 or more at step " + F.int(pick.steps_to_95) + ", so that is its target.", height: 280, source: "data/curves/" + pick.id + ".csv" });
      var aside = h("div", { class: "prose small" });
      aside.innerHTML = "<p class='fig-title'>Settings of this run</p>" + UA.FEATURES.map(function (f) {
        return "<div style='display:flex;justify-content:space-between;border-bottom:1px solid var(--rule);padding:4px 0'><span>" + UA.LABELS[f] + "</span><span class='num'>" + F.setting(f, pick[f]) + "</span></div>";
      }).join("") + "<p style='margin-top:12px'><a href='#/curves?runs=" + pick.id + "'>Open in training curves →</a></p>";
      wrap.appendChild(aside);
      var ink = UA.color("ink");
      UA.plot(chart, [
        { x: pick.steps, y: pick.reward, mode: "lines", line: { color: UA.ballColor("fixed"), width: 2 }, hovertemplate: "step %{x:,}<br>reward %{y:.1f}<extra></extra>" },
        { x: [pick.steps_to_95], y: [pick.reward[pick.steps.indexOf(pick.steps_to_95)]], mode: "markers", marker: { color: ink, size: 9 }, hoverinfo: "skip" }
      ], {
        margin: { l: 8, r: 12, t: 8, b: 8 },
        xaxis: { title: { text: "training step" }, range: [0, 500000], tickformat: "~s" },
        yaxis: { title: { text: "mean reward" }, range: [0, 105] },
        shapes: [
          { type: "line", x0: 0, x1: 500000, y0: 95, y1: 95, line: { color: UA.color("ink-3"), width: 1, dash: "dash" } },
          { type: "line", x0: pick.steps_to_95, x1: pick.steps_to_95, y0: 0, y1: 95, line: { color: ink, width: 1, dash: "dot" } }
        ],
        annotations: [
          { x: 500000, y: 95, xanchor: "right", yanchor: "top", yshift: -4, text: "threshold 95", showarrow: false, font: { size: 11, color: UA.color("ink-3") } },
          { x: pick.steps_to_95, y: 6, xanchor: "left", text: "  steps to 95 = " + F.steps(pick.steps_to_95), showarrow: false, font: { size: 12, color: ink } }
        ]
      });

      var s3 = UA.section(main, { title: "Where to look" });
      s3.appendChild(h("div", { class: "prose" }, h("p", { html:
        "Use the menu on the left, or press <kbd>P</kbd> to step through the main results in order. " +
        "Hover over a chart to save it as a PNG, and the address bar keeps each view's settings, so a bookmark brings it back as it was." })));
    }
  });

  function answer(question, text, sub, view, viewName) {
    return h("li", null, h("em", { text: question }), " " + text,
      h("span", { class: "sub" }, sub + " See ", h("a", { href: "#/" + view, text: viewName }), "."));
  }
})();
