/* Navigation, theme, projector text and presentation mode. */
(function () {
  "use strict";
  var UA = window.UA, h = UA.h;
  var main = document.getElementById("main");
  var nav = document.getElementById("nav");

  if (!UA.ready) { document.getElementById("missing").hidden = false; return; }

  var GROUPS = [
    { label: "The data", ids: ["overview", "curves", "explorer"] },
    { label: "The models", ids: ["models", "importance", "predictor"] },
    { label: "The checks", ids: ["noise", "tuning", "sources"] }
  ];
  var byId = {};
  UA.views.forEach(function (v) { byId[v.id] = v; });

  GROUPS.forEach(function (g) {
    nav.appendChild(h("div", { class: "nav-group", text: g.label }));
    g.ids.forEach(function (id) {
      var v = byId[id];
      nav.appendChild(h("a", { href: "#/" + id, "data-view": id }, v.nav));
    });
  });

  /* Rendering ------------------------------------------------------------------------------------ */

  var lastView = null;
  function render(keepScroll) {
    var route = UA.parseHash();
    var view = byId[route.view] || byId.overview;
    UA.current = view.id;
    UA.params = route.params;
    Array.prototype.forEach.call(main.querySelectorAll(".js-plotly-plot"), function (el) { Plotly.purge(el); });
    main.innerHTML = "";
    Array.prototype.forEach.call(nav.querySelectorAll("a"), function (a) {
      if (a.getAttribute("data-view") === view.id) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    // On a narrow screen the nav scrolls sideways: keep the current view's link in sight.
    var active = nav.querySelector('[aria-current="page"]');
    if (active && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = active.offsetLeft - nav.clientWidth / 2 + active.offsetWidth / 2;
    document.title = view.nav + " · unity-agents";
    try { view.render(main, route.params); }
    catch (err) {
      main.appendChild(h("div", { class: "empty" }, "This view failed to draw: " + err.message));
      console.error(err);
    }
    if (route.params.section) {
      var target = document.getElementById(route.params.section);
      if (target) { target.scrollIntoView({ block: "start" }); return; }
    }
    if (!keepScroll && view.id !== lastView) window.scrollTo(0, 0);
    if (view.id !== lastView) main.focus({ preventScroll: true });
    lastView = view.id;
  }
  UA.rerender = function () { var y = window.scrollY; render(true); window.scrollTo(0, y); };
  window.addEventListener("hashchange", function () { render(false); });

  /* Theme and projector text ------------------------------------------------------------------ */

  var root = document.documentElement;
  function savedTheme() { try { return localStorage.getItem("ua-theme") || "auto"; } catch (e) { return "auto"; } }
  function syncThemeButtons() {
    var t = savedTheme();
    Array.prototype.forEach.call(document.querySelectorAll("[data-theme-choice]"), function (b) {
      b.setAttribute("aria-checked", String(b.getAttribute("data-theme-choice") === t));
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll("[data-theme-choice]"), function (b) {
    b.addEventListener("click", function () {
      var t = b.getAttribute("data-theme-choice");
      try { localStorage.setItem("ua-theme", t); } catch (e) {}
      if (t === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", t);
      syncThemeButtons();
      UA.rerender();
    });
  });
  syncThemeButtons();
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if (savedTheme() === "auto") UA.rerender();
  });

  var projector = document.getElementById("projector-toggle");
  projector.checked = root.classList.contains("projector");
  projector.addEventListener("change", function () {
    root.classList.toggle("projector", projector.checked);
    try { localStorage.setItem("ua-projector", projector.checked ? "1" : "0"); } catch (e) {}
    UA.rerender();
  });

  /* Presentation mode ----------------------------------------------------------------------------- */
  // A fixed sequence of views with prepared settings. Arrow keys step through it.

  function seedPair() {
    // A config whose two seeds both reached 95 but far apart: the clearest picture of seed noise.
    var best = null, gap = 0;
    Object.keys(UA.configs).forEach(function (k) {
      var runs = UA.configs[k].filter(function (r) { return r.ball === "fixed" && r.reached && r.reward; });
      if (runs.length !== 2) return;
      var g = Math.abs(Math.log(runs[0].steps_to_95 / runs[1].steps_to_95));
      if (g > gap && g < 1.2) { gap = g; best = runs; }
    });
    return best ? best.map(function (r) { return r.id; }).join(",") : "";
  }

  var TOUR = [
    { view: "overview", note: "The question: can eight PPO settings predict how many steps a 3DBall run needs to reach a mean reward of 95?" },
    { view: "curves", params: { mode: "all" }, note: "Every training curve we logged, both balls. The target is the first step at or above 95." },
    { view: "curves", params: function () { return { runs: seedPair() }; }, note: "The same config with two seeds: identical settings, different steps to 95. That gap is noise no model can predict." },
    { view: "explorer", note: "Each setting against steps to 95, run by run." },
    { view: "importance", note: "Sub-question 1: what the Gaussian process learned about each setting." },
    { view: "models", note: "Sub-question 2: how far off each model is, on runs it never saw." },
    { view: "models", params: { section: "oof" }, note: "Predicted against actual for every run, each predicted by a model that never saw its config." },
    { view: "models", params: { section: "forest" }, note: "The forest classifier on Group 15's runs: does a run reach 95 within its budget at all?" },
    { view: "noise", note: "Sub-question 3: how close the models are to the noise between seeds." },
    { view: "predictor", note: "Live demo: choose settings and read off the prediction." }
  ];
  var tour = document.getElementById("tour"), step = -1;
  function showStep(i) {
    step = Math.max(0, Math.min(TOUR.length - 1, i));
    var s = TOUR[step];
    var params = typeof s.params === "function" ? s.params() : (s.params || {});
    var hash = UA.hashFor(s.view, params);
    if (location.hash === hash) render(false); else location.hash = hash;
    document.getElementById("tour-count").textContent = (step + 1) + " / " + TOUR.length;
    document.getElementById("tour-note").textContent = s.note;
    document.getElementById("tour-fill").style.width = ((step + 1) / TOUR.length * 100) + "%";
  }
  // Presentation mode turns on projector-sized text, and turns it off again afterwards
  // unless it was already on.
  var addedProjector = false;
  function startTour() {
    document.body.classList.add("presenting");
    if (!root.classList.contains("projector")) { root.classList.add("projector"); addedProjector = true; }
    tour.hidden = false;
    try { if (root.requestFullscreen && !document.fullscreenElement) root.requestFullscreen().catch(function () {}); } catch (e) {}
    showStep(0);
  }
  function stopTour() {
    document.body.classList.remove("presenting");
    tour.hidden = true; step = -1;
    if (addedProjector) { root.classList.remove("projector"); addedProjector = false; }
    try { if (document.fullscreenElement) document.exitFullscreen(); } catch (e) {}
    UA.rerender();
  }
  UA.presenting = function () { return step >= 0; };
  document.getElementById("present-btn").addEventListener("click", startTour);
  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (step < 0) {
      if (e.key === "p" || e.key === "P") { e.preventDefault(); startTour(); }
      return;
    }
    if (e.key === "ArrowRight" || e.key === "PageDown" || (e.key === " " && t === document.body)) { e.preventDefault(); showStep(step + 1); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); showStep(step - 1); }
    else if (e.key === "Escape") { e.preventDefault(); stopTour(); }
  });
  document.addEventListener("fullscreenchange", function () { if (!document.fullscreenElement && step >= 0) { /* keep the tour; Esc already handled */ } });

  render(false);
})();
