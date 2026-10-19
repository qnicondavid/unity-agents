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
- [License](#license)

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
  core/          code shared by the experiments
  datasets/      scripts that build data/
  forest/        our ID3 forest, its runner and its tuner
  mlp/           our MLP, its runner and its tuner
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

The analysis code, its configs and its results are in `analysis/`. [`analysis/README.md`](analysis/README.md) explains how to set it up and run it.

## License

Our code and docs are under the MIT License (`LICENSE`). The two submodules, `external/ml-agents` (Unity Technologies) and `external/BCS2720-project-data-collection` (the course's shared data), keep their own Apache License 2.0. `data/group15.csv` is built from Group 15's data, so it stays under the Apache License 2.0 too.
