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
