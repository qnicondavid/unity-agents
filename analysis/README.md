# Analysis

This folder has the code that builds our datasets and runs the experiments on them, plus their configs and results. For the project and the training runs, see the [main README](../README.md).

## Contents

- [Layout](#layout)
- [Environment](#environment)
- [Data](#data)
  - [Group 15](#group-15)
- [Experiments](#experiments)
- [Forest](#forest)
  - [Running the forest](#running-the-forest)
  - [Tuning the forest](#tuning-the-forest)
- [MLP](#mlp)
  - [Running the MLP](#running-the-mlp)
  - [Tuning the MLP](#tuning-the-mlp)
- [Checks](#checks)

## Layout

```
core/
  cv.py          folds, cross-validation and metrics
  experiment.py  configs, data loading, evaluation, results and tuning
datasets/
  group15.py     builds data/group15.csv
forest/
  forest.py      our ID3 forest
  run.py         cross-validates the forest
  tune.py        tunes the forest
mlp/
  network.py     our MLP
  run.py         cross-validates the network
  tune.py        tunes the network
configs/         experiment settings, one TOML file per experiment
results/         what each experiment wrote
requirements.txt the pinned packages
```

Each model has its own folder with the model and two scripts, `run.py` and `tune.py`. The code they share is in `core/`. The scripts run as modules from the repository root, for example `python -m analysis.forest.run`.

## Environment

1) Create the environment in the repository root:

   ```powershell
   C:\path\to\python314\python.exe -m venv .venv
   ```

2) Activate it (needed in every new terminal):

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   .\.venv\Scripts\Activate.ps1
   ```

3) Install the pinned packages:

   ```powershell
   python -m pip install -r analysis\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## Data

Each dataset has a script in `datasets/` that writes a CSV to `data/`. A config names the CSV it uses.

### Group 15

`analysis/datasets/group15.py` reads `Group15_data.7z` from the data submodule and writes `data/group15.csv`: one row per PPO run on 3DBall with a budget of 500k, 1M or 2M steps (1,555 runs).

From the repository root, with `.venv` active:

```powershell
python -m analysis.datasets.group15
```

Each row has the run's 12 PPO settings, its budget (`max_steps`), the machine it ran on and three outcome columns:

- `last_logged_step`: the last step the run logged
- `reached_95`: whether the mean reward reached 95 within the budget; empty if the run stopped early without reaching it
- `steps_to_95`: the first logged step with a mean reward of 95 or more; empty otherwise

The CSV is committed. Rerunning the script should leave `git diff` empty.

Data: Group 15 (AY 2025-2026), from the course's shared data repository (Apache License 2.0).

## Experiments

An experiment is one TOML config in `configs/<model>/`. To run a new experiment, add a config file. A model's `run.py` cross-validates it on the dataset the config names, and its `tune.py` tunes it.

A config sets:

- `data`: the CSV, relative to the repository root
- `target`: the column to predict
- `features`: the columns the model learns from
- `split_by` (optional): the column whose values each get their own cross-validation (`max_steps` in our split configs, so one per budget); without it, one cross-validation covers all runs
- `report_by` (optional): a column to break the results down by (`max_steps` in our pooled configs); the cross-validation stays the same
- `group_by`: the columns that make up a configuration; runs with the same values always share a fold
- `[cross_validation]`: `folds` (at most the number of configurations) and `seed`
- a table named after the model, such as `[forest]`, with the model's settings

The results go to `analysis/results/<model>/<config name>/`. `folds.csv` has each model's score on each fold, for each value of `split_by`. Without `split_by`, the first column is `split`, with the value `all`. With `report_by`, `folds_by_<column>.csv` has the same scores for each of that column's values, so a pooled config's results compare directly with a split config's.

`tune.py` tries every combination of the values listed in the config's `[tuning]` table, separately for each value of `split_by` if the config has one. It picks the combination with the best mean score on a grouped cross-validation, then reports that choice on fresh folds that were not used for choosing. It uses every CPU core; `--workers` sets a lower number.

The `[tuning]` table sets:

- `folds` and `seed`: the cross-validation used for choosing
- `report_folds` and `report_seed`: the fresh cross-validation used for reporting
- the values to try for each of the model's settings

The tuning results go to `analysis/results/<model>/<config name>/tuned_best_mean/`: `candidates.csv` has every combination's tuning score and rank, `summary.csv` the chosen setting with its tuning and report scores, and the other files the report run.

## Forest

`analysis/forest/` predicts `reached_95` with our ID3 forest (`analysis/forest/forest.py`). It also runs scikit-learn's forest on the same folds, as a check. The score is balanced accuracy. Always predicting the more common class gives a balanced accuracy of 0.5.

The four configs in `analysis/configs/forest/` predict `reached_95` from the plan's 5 settings and from all 12. The split configs give each budget its own cross-validation. The pooled ones, ending in `_pooled`, use one cross-validation over all budgets, with `max_steps` as one more feature.

### Running the forest

From the repository root, with `.venv` active:

```powershell
python -m analysis.forest.run --config analysis\configs\forest\reached_95_12_settings.toml
```

The `[forest]` table sets `trees`, `attributes_per_tree` (at most the number of features), `seed`, and optionally `max_depth` (no limit if it is left out).

`confusion.csv` has our forest's confusion matrix for each value of `split_by`. With `report_by`, `confusion_by_<column>.csv` has it for each of that column's values.

### Tuning the forest

From the repository root, with `.venv` active:

```powershell
python -m analysis.forest.tune --config analysis\configs\forest\reached_95_12_settings.toml
```

The `[tuning]` table lists the values to try for `trees`, `attributes_per_tree` and `max_depth`, with `"none"` for no depth limit. For the split configs, `tuned_two_se/` holds the first tuning run, which used a tie rule (within two standard errors of the best, the simplest wins) on a 5-fold cross-validation; on our data that rule chose forests that were too shallow.

## MLP

`analysis/mlp/` predicts `steps_to_95` with our network (`analysis/mlp/network.py`): fully connected hidden layers with batch norm and ReLU, trained with plain SGD on the mean squared error. It also scores three other models on the same folds:

- scikit-learn's `MLPRegressor` with the same settings, as a check on our network
- ridge regression on the same inputs, a linear model
- the median of the training runs, the baseline that ignores the settings

The score is the mean absolute error, averaged over the seeds for the two networks.

The four configs in `analysis/configs/mlp/` predict `steps_to_95` from the plan's 5 settings and from all 12, split and pooled like the forest's. Only runs that reached 95 have a `steps_to_95`, so only they are used: 805 of the 1,555. The configs take the natural log of the target and of the settings that span a factor of 8 or more (and of `max_steps` in the pooled configs), so an error of 0.1 means a typical prediction is about 10% off.

### Running the MLP

From the repository root, with `.venv` active:

```powershell
python -m analysis.mlp.run --config analysis\configs\mlp\steps_to_95_12_settings.toml
```

The `[mlp]` table sets `hidden_sizes` (one number per hidden layer), `learning_rate`, `batch_size`, `epochs`, `seeds` (each network is trained once per seed), and optionally `dropout`. Three more keys are for the MLP only:

- `log_target`: `true` takes the natural log of the target
- `log_features`: the features to take the natural log of
- `median_by` (optional): the baseline takes its median separately for each value of this feature (`max_steps` in our pooled configs, so one per budget); without it, one median covers each split

### Tuning the MLP

From the repository root, with `.venv` active:

```powershell
python -m analysis.mlp.tune --config analysis\configs\mlp\steps_to_95_12_settings.toml
```

The `[tuning]` table lists the values to try for `hidden_sizes`, `learning_rate`, `batch_size` and `epochs`. Only our network is tuned. The lowest mean error wins, and ties go to fewer layers, then fewer neurons, fewer epochs, larger batches and finally the lower learning rate. The report run then scores all four models, with scikit-learn's network using the chosen settings.

## Checks

Ideas for later, once we have our own runs:

- shuffled targets: train on targets shuffled at random; the score should fall to the baseline, which shows that nothing leaks from the test folds into training
- permutation importance: shuffle one feature at a time in the test folds and measure how much the score drops, which shows the settings a model relies on
- learning curve: train on growing parts of the training folds and score on the same test folds; if the error is still falling with all the runs, more runs would help
