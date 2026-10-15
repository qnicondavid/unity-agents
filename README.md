# unity-agents

M2-1: AI and Machine Learning (2026-2027-002-BCS2720). A work in progress.


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
external/
  ml-agents/     our ML-Agents fork
```

## Setup

1) Clone with the submodule:

   ```powershell
   git clone --recurse-submodules https://github.com/qnicondavid/unity-agents.git
   ```

2) Create the virtual environment (Python 3.10) inside the submodule:

   ```powershell
   cd external\ml-agents
   C:\path\to\python310\python.exe -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install setuptools==81.0.0
   pip install -e .\ml-agents-envs
   pip install -e .\ml-agents
   ```

3) Unity: open `external\ml-agents\Project` with editor **2022.3.4f1**.

## Running a training run

```powershell
cd external\ml-agents
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\Activate.ps1
mlagents-learn config/ppo/3DBall.yaml --run-id=<name>
```

Wait for `Start training by pressing the Play button`, then press Play in Unity.

## Results

```powershell
tensorboard --logdir results
```
## Logging Training Run**

1. First-time setup: install the required dependencies.

   ```powershell
   python -m pip install psutil PyYAML
   ```

2. From the `unity-agents` repository, run:

   ```powershell
   python tools/run_logger.py --config <path-to-config.yaml> --run-id <unique-run-id>
   ```

   Replace `<path-to-config.yaml>` with the path to the configuration file and `<unique-run-id>` with a unique identifier for the experiment run you want to log.

## Analysis environment

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

