(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt, S = UA.stats;

  // Mean of the logged GPR prediction over all grid configs, for each value of one setting.
  UA.marginal = function (ball, model, feature) {
    var M = UA.models, pred = M.balls[ball].grid[model], vals = M.values[feature];
    var sums = vals.map(function () { return 0; }), counts = vals.map(function () { return 0; });
    var stride = 1, fi = M.features.indexOf(feature);
    for (var i = M.features.length - 1; i > fi; i--) stride *= M.values[M.features[i]].length;
    for (var k = 0; k < pred.length; k++) {
      var j = Math.floor(k / stride) % vals.length;
      sums[j] += pred[k]; counts[j]++;
    }
    return sums.map(function (s, j) { return s / counts[j]; });
  };

  UA.register({
    id: "importance", num: "05", nav: "What matters",
    render: function (main) {
      var ls = UA.result("gpr/group10_steps_to_95/tuned_best_mean/length_scales.csv");
      var byBall = {};
      ls.forEach(function (r) { byBall[r.ball] = r; });
      var order = UA.relevanceOrder();
      function top2(b) { return UA.FEATURES.slice().sort(function (x, y) { return byBall[b][x] - byBall[b][y]; }).slice(0, 2); }
      var tf = top2("fixed"), tr = top2("random");

      UA.head(main, {
        num: "05", kicker: "What matters", tag: "SQ1",
        title: UA.cap(UA.LABELS[order[0]]) + " and " + UA.LABELS[order[1]] + " move the prediction fastest",
        lede: "The Gaussian process learns one length scale per setting: how far that setting must move, in standard deviations, before the prediction changes much. " +
          "A short length scale means the setting matters. Below, <strong>relevance is one over the length scale</strong>."
      });

      var s1 = UA.section(main, { title: "Relevance of each setting" });
      var grid = h("div", { class: "grid side-right" });
      s1.appendChild(grid);
      var el = UA.fig(grid, { title: null, height: 360, file: "setting-relevance", source: "analysis/results/gpr/group10_steps_to_95/tuned_best_mean/length_scales.csv" });
      UA.plot(el, UA.BALLS.map(function (b) {
        return { type: "bar", orientation: "h", name: UA.BALL_LABEL[b], y: order.map(function (f) { return UA.LABELS[f]; }), x: order.map(function (f) { return 1 / byBall[b][f]; }),
          marker: { color: UA.ballColor(b) }, customdata: order.map(function (f) { return byBall[b][f]; }),
          hovertemplate: "%{y} · " + UA.BALL_LABEL[b].toLowerCase() + "<br>length scale %{customdata:.4~g}<br>relevance %{x:.3f}<extra></extra>" };
      }), { barmode: "group", bargap: 0.3, bargroupgap: 0.08, showlegend: true,
        xaxis: { title: { text: "relevance (1 ÷ length scale)" } }, yaxis: { autorange: "reversed", type: "category" } });
      var side = h("div", { class: "prose small" });
      grid.appendChild(side);
      var bf = byBall.fixed, br = byBall.random;
      side.innerHTML =
        "<p>Averaged over both balls, " + UA.LABELS[order[0]] + " (length scale " + F.num(bf[order[0]], 1) + " fixed, " + F.num(br[order[0]], 1) + " random) and " +
        UA.LABELS[order[1]] + " (" + F.num(bf[order[1]], 1) + ", " + F.num(br[order[1]], 1) + ") lead. " +
        (tf.sort().join() === tr.sort().join() ? "Both balls have the same top two." :
          "On the fixed ball alone the top two are " + UA.LABELS[tf[0]] + " and " + UA.LABELS[tf[1]] + "; on the random ball, " + UA.LABELS[tr[0]] + " and " + UA.LABELS[tr[1]] + ".") +
        " Beta's length scale is " + F.int(bf.beta) + " on the fixed ball and " + F.int(br.beta) + " on the random ball, so in our range it changes almost nothing.</p>" +
        "<p>The model's noise is <span class='num'>" + F.num(bf.noise_sd) + "</span> (fixed) and <span class='num'>" + F.num(br.noise_sd) + "</span> (random) in log steps, against a signal of <span class='num'>" + F.num(bf.signal_sd, 2) + "</span>. " +
        "That noise is the spread between runs with identical settings; see <a href='#/noise'>Noise and machines</a>.</p>" +
        "<p class='muted'>Length scales are measured on the logged settings for batch size, buffer size, learning rate and beta, after scaling each setting to standard deviation 1.</p>";

      if (!UA.models) {
        main.appendChild(h("div", { class: "empty", text: "Run python -m gui.export_data to add the effect of each setting." }));
        return;
      }
      var s2 = UA.section(main, { title: "The effect of each setting", text: "The Gaussian process's prediction for each value, averaged over every combination of the other seven settings (15,552 configs per ball), in the same order as the chart above. The ratio is the biggest value against the smallest." });
      var effects = UA.FEATURES.map(function (f) {
        var ratio = {};
        UA.BALLS.forEach(function (b) { var m = UA.marginal(b, "gpr", f); ratio[b] = Math.exp(Math.max.apply(null, m) - Math.min.apply(null, m)); });
        return { f: f, ratio: ratio };
      }).sort(function (a, b) { return b.ratio.fixed - a.ratio.fixed; });
      var g2 = h("div", { class: "grid four" });
      s2.appendChild(g2);
      // Same order as the relevance chart above, so the two read side by side.
      order.map(function (f) { return effects.filter(function (x) { return x.f === f; })[0]; }).forEach(function (e) {
        var f = e.f, vals = UA.models.values[f];
        var chart = UA.fig(g2, { title: UA.LABELS[f], small: true, height: 210, file: "effect-" + f,
          caption: "×" + e.ratio.fixed.toFixed(2) + " fixed · ×" + e.ratio.random.toFixed(2) + " random" });
        UA.plot(chart, UA.BALLS.map(function (b) {
          var m = UA.marginal(b, "gpr", f);
          return { type: "scatter", mode: "lines+markers", name: UA.BALL_LABEL[b], x: vals.map(function (v) { return F.setting(f, v); }), y: m.map(Math.exp),
            line: { color: UA.ballColor(b), width: 2 }, marker: { color: UA.ballColor(b), size: 6 },
            hovertemplate: UA.LABELS[f] + " %{x}<br>predicted %{y:,.0f} steps<extra>" + UA.BALL_LABEL[b] + "</extra>" };
        }), { xaxis: { type: "category" }, yaxis: { type: "log", range: [Math.log10(40000), Math.log10(260000)], tickvals: [50000, 100000, 200000], ticktext: ["50k", "100k", "200k"] } });
      });
      var topLs = order.slice(0, 2).sort().join(), topEff = effects.slice(0, 2).map(function (e) { return e.f; }).sort().join();
      // Say it in the lede too, so the headline and the averaged-effect charts never seem to disagree.
      if (topLs !== topEff) main.querySelector(".lede").insertAdjacentHTML("beforeend", " Measured by total effect across our range instead, " + UA.LABELS[effects[0].f] + " and " + UA.LABELS[effects[1].f] + " move it most; both views are below.");
      s2.appendChild(h("p", { class: "note", html: "Length scales say how quickly a setting changes the prediction; the averaged effect says by how much across our range. " +
        (topLs === topEff ? "Both put " + UA.LABELS[order[0]] + " and " + UA.LABELS[order[1]] + " in the top two."
          : "By averaged effect the top two are " + UA.LABELS[effects[0].f] + " and " + UA.LABELS[effects[1].f] + ": a setting can matter a lot locally but span a narrow range, or the reverse.") }));
    }
  });

  UA.cap = function (s) { return s.charAt(0).toUpperCase() + s.slice(1); };
})();
