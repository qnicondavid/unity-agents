(function () {
  "use strict";
  var UA = window.UA, h = UA.h, F = UA.fmt;

  UA.register({
    id: "sources", num: "09", nav: "Data sources",
    render: function (main) {
      var D = UA.data.runs;
      var log = UA.objects(D.log);
      var curves = Object.keys(UA.data.curves.reward).length;
      var survey = UA.objects(D.survey_files);
      var groups = {};
      survey.forEach(function (r) { groups[r.group] = 1; });

      UA.head(main, {
        num: "09", kicker: "Data sources",
        title: "Everything this explorer reads",
        lede: "All numbers on these pages come from files in the repository. <span class='mono'>python -m gui.export_data</span> turns them into the files in <span class='mono'>gui/data/</span>."
      });

      var s1 = UA.section(main, { title: "Files" });
      var files = [
        { file: "data/group10.csv", rows: UA.g10.length, what: "One row per 3DBall run with a 500k budget: ball, 12 PPO settings, seed, machine and three outcomes.", used: "every view on our runs" },
        { file: "data/group10_2026.csv", rows: log.length, what: "Our raw run log, written by tools/collect_run.py. Includes 4 early test runs with a 100k budget and threshold 80.", used: "noise and machines" },
        { file: "data/curves/*.csv", rows: curves, what: "TensorBoard scalars of each run, every 12,000 steps.", used: "training curves, wall time" },
        { file: "data/group15.csv", rows: UA.g15.length, what: "Group 15's runs (AY 2025–2026), rebuilt from the course's shared data. Budgets 500k, 1M and 2M.", used: "forest classifier, Group 15 MLP" },
        { file: "data/same_config_diff_machines.csv", rows: D.same_config.rows.length, what: "Runs that share settings and seed with a run on another laptop.", used: "noise and machines" },
        { file: "docs/hardware/machines.csv", rows: D.machines.rows.length, what: "The four laptops that ran our batches.", used: "noise and machines" },
        { file: "docs/shared-data/survey-output/", rows: survey.length, what: "A survey of every group's shared CSVs: " + Object.keys(groups).length + " groups, which columns each one logged.", used: "this page" },
        { file: "analysis/results/", rows: Object.keys(UA.data.results.files).length, what: "What each experiment wrote: folds, summaries, candidates, confusion matrices, length scales.", used: "model comparison, what matters, tuning" },
        { file: "gui/data/models.js", rows: UA.models ? UA.models.balls.fixed.grid.gpr.length * 2 : 0, what: "Made by the export: out-of-fold predictions for every run, and predictions for all 15,552 grid configs per ball.", used: "predictor, predicted against actual" }
      ];
      var t = h("div");
      s1.appendChild(t);
      UA.table(t, { rows: files, pageSize: 20, columns: [{ key: "file", label: "file", fmt: function (v) { return h("span", { class: "mono", text: v }); } }, { key: "rows", label: "rows", num: true, fmt: F.int }, { key: "what", label: "what it holds" }, { key: "used", label: "used in" }] });
      Array.prototype.forEach.call(t.querySelectorAll("td:nth-child(3)"), function (td) { td.style.whiteSpace = "normal"; td.style.minWidth = "320px"; });

      var s2 = UA.section(main, { title: "Our batches", text: "Each batch draws 50 configs from a batch seed and runs each one with two seeds and two balls: 200 runs." });
      var batches = {};
      UA.g10.forEach(function (r) { var b = batches[r.batch] = batches[r.batch] || { n: 0, machines: {} }; b.n++; b.machines[r.machine] = 1; });
      var names = Object.keys(batches).sort(function (a, b) { return (a === "first" ? -1 : +a) - (b === "first" ? -1 : +b); });
      var bt = h("div");
      s2.appendChild(bt);
      UA.table(bt, { pageSize: 20, rows: names.map(function (n) {
        var runs = UA.g10.filter(function (r) { return r.batch === n; });
        return { batch: n === "first" ? "first (no seed in the run ids)" : "b" + n, runs: batches[n].n,
          machines: Object.keys(batches[n].machines).sort().map(function (m) { return UA.MACHINE_NAMES[m] || m; }).join(", "),
          reached: runs.filter(function (r) { return r.reached; }).length / runs.length };
      }), columns: [{ key: "batch", label: "batch (batch seed)" }, { key: "runs", label: "runs", num: true },
        { key: "reached", label: "reached 95", num: true, fmt: function (v) { return F.pct(v); } }, { key: "machines", label: "machine" }] });
      s2.appendChild(h("p", { class: "fig-src", text: "Source: data/group10.csv (batch from the run id, machine from the CPU)" }));

      var tests = log.filter(function (r) { return r.max_steps !== 500000; });
      if (tests.length) {
        s2.appendChild(h("p", { class: "note", html: "Left out of the dataset: " + tests.map(function (r) { return "<span class='mono'>" + F.esc(r.run_id) + "</span>"; }).join(", ") + ", the first test runs, with a 100k budget." }));
      }

      var s3 = UA.section(main, { title: "What the other groups logged", text: "The course's shared data repository, surveyed by tools/survey_shared_data.py. Rows are column names used by at least two groups." });
      var matrix = UA.objects(D.survey_matrix).filter(function (r) { return r.n_groups >= 2; }).sort(function (a, b) { return b.n_groups - a.n_groups; });
      var gcols = D.survey_matrix.columns.filter(function (c) { return /^Group /.test(c); });
      var hm = UA.fig(s3, { title: null, height: Math.max(320, matrix.length * 17 + 60), file: "shared-data-columns", source: "docs/shared-data/survey-output/matrix.csv" });
      UA.plot(hm, [{ type: "heatmap", x: gcols, y: matrix.map(function (r) { return r.column; }), z: matrix.map(function (r) { return gcols.map(function (g) { return r[g] === "x" ? 1 : 0; }); }),
        colorscale: [[0, UA.color("surface")], [1, UA.color("ink-3")]], showscale: false, xgap: 2, ygap: 2,
        customdata: matrix.map(function (r) { return gcols.map(function (g) { return r[g] === "x" ? "logged" : "not logged"; }); }),
        hovertemplate: "%{x}: %{y} %{customdata}<extra></extra>" }],
        { margin: { l: 230, r: 8, t: 8, b: 8 }, xaxis: { side: "top", ticks: "", showgrid: false }, yaxis: { autorange: "reversed", ticks: "", showgrid: false, tickfont: { family: "Atkinson Next, Segoe UI, sans-serif", size: 11 } } });
      var ft = h("div", { style: "margin-top:24px" });
      s3.appendChild(ft);
      UA.table(ft, { rows: survey, pageSize: 12, columns: [{ key: "group", label: "group" }, { key: "file", label: "file" }, { key: "rows", label: "rows", num: true, fmt: F.int }, { key: "columns", label: "columns", num: true }, { key: "granularity", label: "granularity" }] });
    }
  });
})();
