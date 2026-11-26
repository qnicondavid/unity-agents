# GUI

A static explorer for the project's data, models and results, for the presentation and the demo. Plain HTML, CSS and JavaScript, with Plotly for the charts. It needs no server and no internet connection.

## Opening it

Double-click `gui/index.html`, or open it in any browser.

Keys: <kbd>P</kbd> starts the presentation, which steps through the main results with <kbd>←</kbd> and <kbd>→</kbd>. <kbd>Esc</kbd> leaves it. The sidebar has a light/dark switch and a larger-text option for projectors. Each view keeps its settings in the address bar, so a bookmark brings a view back exactly as it was, and every chart has a PNG button.

## Updating the data

The GUI reads the files in `gui/data/`. They are made from `data/`, `analysis/results/` and `docs/` by an export script. From the repository root, with `.venv` active:

```powershell
python -m gui.export_data
```

This also retrains the tuned models (Gaussian process, MLP, regression forest, ridge) to compute:

- out-of-fold predictions for every run, on the report folds (10 folds, seed 11)
- predictions for all 15,552 configs of the search grid, for each ball

That step took about 15 minutes on m05. After a change that does not touch the models, for example new results files, skip it:

```powershell
python -m gui.export_data --skip-models
```

The Gaussian process, ridge and median scores the export computes match `analysis/results/` exactly. The MLP and the regression forest use their final tuned settings, retrained on all current runs, so their scores can differ from the older results files.

## Layout

```
index.html          the page
css/style.css       colours, type and layout, light and dark
js/core.js          shared helpers: data, formatting, charts, URL state
js/app.js           navigation, theme, presentation mode
js/views/*.js       one file per view
data/*.js           made by export_data.py
vendor/plotly.min.js, fonts/   local copies, so it works offline
export_data.py      builds data/
```

Plotly is under the MIT License. IBM Plex Sans, IBM Plex Mono and Source Serif 4 are under the SIL Open Font License.
