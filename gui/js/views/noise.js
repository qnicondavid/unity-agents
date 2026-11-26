(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt, S = UA.stats;
  var TICKS = { tickvals: [10000, 20000, 50000, 100000, 200000, 500000], ticktext: ["10k", "20k", "50k", "100k", "200k", "500k"] };

  function bestMae(ball) {
    if (UA.models) return UA.models.balls[ball].scores.gpr.mae;
    var r = UA.result("gpr/group10_steps_to_95/tuned_best_mean/summary.csv").filter(function (x) { return x.ball === ball; })[0];
    return r ? r.gpr_mean_absolute_error : NaN;
  }

  UA.register({
    id: "noise", num: "07", nav: "Noise and machines",
    render: function (main, params) {
      var nf = { fixed: UA.noiseFloor("fixed"), random: UA.noiseFloor("random") };
      var share = { fixed: bestMae("fixed") / nf.fixed.floor, random: bestMae("random") / nf.random.floor };
      UA.head(main, {
        num: "07", kicker: "Noise and machines", tag: "SQ3",
        title: "Seed noise and machines",
        lede: "Run the same settings with another seed and steps to 95 changes. No model of the settings can predict that part. " +
          "Comparing seed 2 with seed 3 of every config tells us the smallest error any model could reach."
      });

      /* Seeds --------------------------------------------------------------------------------- */
      var ball = params.ball === "random" ? "random" : "fixed";
      var s1 = UA.section(main, { title: "Seed 2 against seed 3" });
      var ctl = h("div", { class: "controls" });
      s1.appendChild(ctl);
      ctl.appendChild(UA.control("Ball", UA.seg({ label: "Ball", value: ball, options: [{ value: "fixed", label: "Fixed" }, { value: "random", label: "Random" }],
        onChange: function (v) { ball = v; UA.setParams({ ball: v === "fixed" ? null : v }); drawSeeds(); } })));
      var grid = h("div", { class: "grid side-right" });
      s1.appendChild(grid);
      var el = UA.fig(grid, { title: null, height: 420, file: "seed-2-vs-seed-3", source: "data/group10.csv" });
      var side = h("div");
      grid.appendChild(side);

      function drawSeeds() {
        var pairs = UA.seedPairs(ball), both = pairs.filter(function (p) { return p.both; });
        var n = nf[ball];
        UA.plot(el, [{
          type: "scatter", mode: "markers", x: both.map(function (p) { return p.a.steps_to_95; }), y: both.map(function (p) { return p.b.steps_to_95; }),
          text: both.map(function (p) { return p.a.id + "," + p.b.id; }),
          marker: { color: UA.ballColor(ball), size: 6, opacity: 0.5, symbol: UA.ballSymbol(ball) },
          hovertemplate: "seed 2: %{x:,}<br>seed 3: %{y:,}<extra>click to compare curves</extra>"
        }], {
          xaxis: UA.merge({ title: { text: "seed 2: steps to 95" }, type: "log", range: [4, Math.log10(550000)] }, TICKS),
          yaxis: UA.merge({ title: { text: "seed 3: steps to 95" }, type: "log", range: [4, Math.log10(550000)] }, TICKS),
          shapes: [{ type: "line", x0: 10000, y0: 10000, x1: 550000, y1: 550000, line: { color: UA.color("ink"), width: 1 } }]
        });
        el.removeAllListeners && el.removeAllListeners("plotly_click");
        el.on("plotly_click", function (ev) { var p = ev.points && ev.points[0]; if (p) UA.go("curves", { runs: p.text }); });
        var mae = bestMae(ball);
        side.innerHTML = "";
        var ns = UA.noiseShare(ball);
        var oneMissing = pairs.length - both.length;
        side.appendChild(h("div", { class: "side-text", html:
          "<p>Across " + F.int(n.pairs) + " configs where both seeds reached 95, the two seeds land <b>" + F.num(n.meanAbs) + "</b> apart in log steps on average, about " + F.offBy(n.meanAbs) + ".</p>" +
          "<p>So even a model that knew each config's typical result exactly would be off by about that divided by √2: <b>" + F.num(ns.floor) + "</b>. " +
          "The Gaussian process is off by <b>" + F.num(ns.mae) + "</b>, which means about <b>" + F.pct(ns.share) + "</b> of its error is this seed noise.</p>" +
          "<p>The rest, " + F.num(ns.mae - ns.floor) + ", is the most that a better model or more runs could still win back.</p>" +
          "<p class='muted small'>" + oneMissing + " more configs are left out because at least one seed never reached 95. Dots on the diagonal are configs whose seeds reached 95 at the same logged step.</p>" }));
      }
      drawSeeds();

      /* Ball --------------------------------------------------------------------------------- */
      var s2 = UA.section(main, { title: "Fixed ball against random ball", text: "The same config and seed, trained once with each ball." });
      var pairsBall = [];
      var groups = {};
      UA.g10.forEach(function (r) { var k = r.key + "|" + r.seed + "|" + r.batch; (groups[k] = groups[k] || {})[r.ball] = r; });
      Object.keys(groups).forEach(function (k) { var g = groups[k]; if (g.fixed && g.random && g.fixed.reached && g.random.reached) pairsBall.push(g); });
      var ratios = pairsBall.map(function (g) { return g.random.steps_to_95 / g.fixed.steps_to_95; });
      var slower = ratios.filter(function (x) { return x > 1; }).length;
      var g2 = h("div", { class: "grid side-right" });
      s2.appendChild(g2);
      var el2 = UA.fig(g2, { title: null, height: 360, file: "fixed-vs-random-ball", source: "data/group10.csv" });
      UA.plot(el2, [{ type: "scatter", mode: "markers", x: pairsBall.map(function (g) { return g.fixed.steps_to_95; }), y: pairsBall.map(function (g) { return g.random.steps_to_95; }),
        marker: { color: UA.color("ink-2"), size: 5, opacity: 0.45 }, text: pairsBall.map(function (g) { return g.fixed.id + "," + g.random.id; }),
        hovertemplate: "fixed %{x:,}<br>random %{y:,}<extra>click to compare</extra>" }], {
        xaxis: UA.merge({ title: { text: "fixed ball: steps to 95" }, type: "log", range: [4, Math.log10(550000)] }, TICKS),
        yaxis: UA.merge({ title: { text: "random ball: steps to 95" }, type: "log", range: [4, Math.log10(550000)] }, TICKS),
        shapes: [{ type: "line", x0: 10000, y0: 10000, x1: 550000, y1: 550000, line: { color: UA.color("ink"), width: 1 } }] });
      el2.on("plotly_click", function (ev) { var p = ev.points && ev.points[0]; if (p) UA.go("curves", { runs: p.text }); });
      var med = S.median(ratios);
      g2.appendChild(h("div", { class: "side-text", html:
        "<p>Over " + F.int(pairsBall.length) + " pairs, the random ball needed <b>" + med.toFixed(2) + " times</b> as many steps as the fixed ball (median), and took longer in " + F.pct(slower / pairsBall.length) + " of them.</p>" +
        "<p>That is why every model is trained and scored on each ball separately.</p>" +
        "<p class='muted small'>Each dot is one pair of runs, one with each ball, so it has no ball colour.</p>" }));

      /* Machines -------------------------------------------------------------------------------- */
      var same = UA.objects(UA.data.runs.same_config);
      same.forEach(function (r) {
        r.ball = /Randomize/.test(r.run_id) ? "random" : "fixed";
        r.cfg = UA.FEATURES.map(function (f) { return F.setting(f, r[f]); }).join(" · ");
      });
      var sameGroups = {};
      same.forEach(function (r) { sameGroups[r.cfg + "|" + r.seed] = 1; });
      var s3 = UA.section(main, { title: "Same config, different machine", text: F.int(same.length) + " runs in our log share their settings and seed with a run on another laptop (" + Object.keys(sameGroups).length + " groups). " +
        "Between our Intel laptops such repeats gave identical curves; runs on the Apple M4 (m03) came out differently. The outcome columns use the log's own threshold of 80." });
      same.sort(function (a, b) { return a.cfg < b.cfg ? -1 : a.cfg > b.cfg ? 1 : (a.ball < b.ball ? -1 : a.ball > b.ball ? 1 : a.seed - b.seed); });
      var t = h("div");
      s3.appendChild(t);
      UA.table(t, { rows: same, pageSize: 12, columns: [
        { key: "run_id", label: "run" }, { key: "machine", label: "machine" }, { key: "seed", label: "seed", num: true },
        { key: "cfg", label: "settings (batch · buffer · lr · units · layers · beta · eps · epochs)" },
        { key: "steps_to_threshold", label: "steps to 80 (log threshold)", num: true, fmt: function (v, r) { return r.reached_threshold === true || r.reached_threshold === "True" ? F.int(v) : "not reached"; } },
        { key: "final_mean_reward", label: "final reward", num: true, fmt: function (v) { return F.num(v, 1); } }] });
      s3.appendChild(h("p", { class: "note", html: "Runs of the same config always share a cross-validation fold, so repeated configs cannot leak from training into testing." }));

      var s4 = UA.section(main, { title: "Wall-clock time per run", text: "From the first to the last logged step. Bigger buffers and more epochs per update cost time on every machine." });
      var g4 = h("div", { class: "grid side-right" });
      s4.appendChild(g4);
      var el4 = UA.fig(g4, { title: null, height: 300, file: "wall-time-by-machine", source: "data/curves/*.csv (wall_time)" });
      var machines = ["m01", "m02", "m03", "m05"];
      UA.plot(el4, UA.BALLS.map(function (b) {
        var part = UA.g10.filter(function (r) { return r.ball === b && r.duration != null; });
        return { type: "box", name: UA.BALL_LABEL[b], x: part.map(function (r) { return r.machine; }), y: part.map(function (r) { return r.duration / 60; }),
          marker: { color: UA.ballColor(b) }, fillcolor: UA.ballSoft(b), line: { width: 1.2 }, boxpoints: false, hovertemplate: "%{x}<extra>" + UA.BALL_LABEL[b] + "</extra>" };
      }), { boxmode: "group", showlegend: true, xaxis: { type: "category", categoryorder: "array", categoryarray: machines }, yaxis: { title: { text: "minutes per run" }, rangemode: "tozero" } });
      var hw = UA.objects(UA.data.runs.machines);
      var counts = {};
      UA.g10.forEach(function (r) { counts[r.machine] = (counts[r.machine] || 0) + 1; });
      var hwHost = h("div");
      g4.appendChild(hwHost);
      UA.table(hwHost, { rows: hw.map(function (m) { m.runs = counts[m.machine_id] || 0; return m; }), pageSize: 6, columns: [
        { key: "machine_id", label: "id" }, { key: "cpu_model", label: "cpu", fmt: function (v) { return String(v).replace(/\(R\)|\(TM\)|CPU|@.*$/g, "").replace(/\s+/g, " ").trim(); } },
        { key: "cpu_cores_physical", label: "cores", num: true }, { key: "runs", label: "runs", num: true }] });
    }
  });
})();
