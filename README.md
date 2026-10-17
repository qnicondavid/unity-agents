# unity-agents

Can PPO hyperparameters predict how many training steps a Unity ML-Agents run needs to reach a mean reward threshold? Group 10, BCS2720 Project 2-1 (AI and Machine Learning), Maastricht University. A work in progress.

## Contents

- [Layout](#layout)
- [Getting the code](#getting-the-code)
- [Training runs](#training-runs)
  - [Setup](#setup)
  - [Running a training run](#running-a-training-run)
  - [Viewing results in TensorBoard](#viewing-results-in-tensorboard)
  - [Logging a training run](#logging-a-training-run)
- [Analysis](#analysis)
  - [Environment](#environment)
  - [Building the Group 15 dataset](#building-the-group-15-dataset)
  - [Running the forest](#running-the-forest)

## Layout

```
docs/
  reading/       notes on the papers
  runs/          training runs
  hardware/      machine register
  shared-data/   analysis of last year's group datasets
  resources/     official materials
tools/           scripts
analysis/        ML code and its pinned requirements
  configs/       experiment settings, one TOML file per experiment
  results/       what each experiment wrote
data/            datasets built by the analysis scripts
external/
  ml-agents/     our ML-Agents fork
  BCS2720-project-data-collection/   the course's shared group data
```

## Getting the code

Clone with the submodules:

```powershell
git clone --recurse-submodules https://github.com/qnicondavid/unity-agents.git
```

If you cloned before the data submodule was added, fetch it with:

```powershell
git submodule update --init
```

## Training runs

### Setup

1) Create the virtual environment (Python 3.10) inside the ML-Agents submodule:

   ```powershell
   cd external\ml-agents
   C:\path\to\python310\python.exe -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install setuptools==81.0.0
   pip install -e .\ml-agents-envs
   pip install -e .\ml-agents
   ```

2) Unity: open `external\ml-agents\Project` with editor **2022.3.4f1**.

### Running a training run

```powershell
cd external\ml-agents
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\Activate.ps1
mlagents-learn config/ppo/3DBall.yaml --run-id=<name>
```

Wait for `Start training by pressing the Play button`, then press Play in Unity.

### Viewing results in TensorBoard

```powershell
tensorboard --logdir results
```

### Logging a training run

1. First-time setup: install the required dependencies.

   ```powershell
   python -m pip install psutil PyYAML
   ```

2. From the `unity-agents` repository, run:

   ```powershell
   python tools/run_logger.py --config <path-to-config.yaml> --run-id <unique-run-id>
   ```

   Replace `<path-to-config.yaml>` with the path to the configuration file and `<unique-run-id>` with a unique identifier for the experiment run you want to log.

## Analysis

### Environment

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

### Building the Group 15 dataset

`analysis/group15.py` reads `Group15_data.7z` from the data submodule and writes `data/group15.csv`: one row per PPO run on 3DBall with a budget of 500k, 1M or 2M steps (1,555 runs).

From the repository root, with `.venv` active:

```powershell
python analysis\group15.py
```

Each row has the run's 12 PPO settings, its budget (`max_steps`), the machine it ran on and three outcome columns:

- `last_logged_step`: the last step the run logged
- `reached_95`: whether the mean reward reached 95 within the budget; empty if the run stopped early without reaching it
- `steps_to_95`: the first logged step with a mean reward of 95 or more; empty otherwise

The CSV is committed. Rerunning the script should leave `git diff` empty.

Data: Group 15 (AY 2025-2026), from the course's shared data repository (Apache License 2.0).

### Running the forest

`analysis/run_forest.py` cross-validates our ID3 forest (`analysis/forest.py`) on the dataset a config file names. It also runs scikit-learn's forest on the same folds, as a check.

From the repository root, with `.venv` active:

```powershell
python analysis\run_forest.py --config analysis\configs\forest\reached_95_12_settings.toml
```

A config sets:

- `data`: the CSV, relative to the repository root
- `class_attribute`: the column to predict
- `features`: the columns the forest learns from
- `split_by`: the column whose values each get their own cross-validation (`max_steps` in our configs, so one per budget)
- `group_by`: the columns that make up a configuration; runs with the same values always share a fold
- `[forest]`: `trees`, `attributes_per_tree` (at most the number of features), `seed`, and optionally `max_depth` (no limit if it is left out)
- `[cross_validation]`: `folds` (at most the number of configurations) and `seed`

To run a new experiment, add a config file. The two in `analysis/configs/forest/` predict `reached_95` from the plan's 5 settings and from all 12.

The results go to `analysis/results/forest/<config name>/`: `folds.csv` has both forests' balanced accuracy on each fold, and `confusion.csv` has our forest's confusion matrix for each value of `split_by`. Always predicting the more common class gives a balanced accuracy of 0.5.
