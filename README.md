# GTLRL: Generalized Transfer, Meta, and Continual Learning (Modular RL Project)

This is a cleaned-up, modular version of your original coursework codebase.  
It’s organized as a real Python package (`src/gtlrl`) with three tracks:

- **Transfer Learning (PPO / Stable-Baselines3)** on **Pong**, **Snake**, **PuckWorld**
- **Meta-Learning (first-order MAML/Reptile-style)** across **Snake**, **PuckWorld**, **Pong**
- **Continual Learning** (Transfer baselines + MAML + EWC) on **sine-wave regression tasks**

Because humans love chaos, the original repo mixed `gym`, `gymnasium`, duplicated env code, and had at least one
file with corrupted characters. This version fixes those things and puts everything in one coherent layout.

---

## Project layout

```
gtlrl_modular_project/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
└─ src/
   └─ gtlrl/
      ├─ __init__.py
      ├─ common/
      │  ├─ seed.py
      │  └─ io.py
      ├─ envs/
      │  ├─ snake.py
      │  ├─ puckworld.py
      │  └─ pong.py
      ├─ continual/
      │  ├─ config.py
      │  ├─ models.py
      │  ├─ data.py
      │  ├─ transfer.py
      │  ├─ maml_sine.py
      │  ├─ ewc.py
      │  ├─ plots.py
      │  └─ run.py
      ├─ meta/
      │  ├─ config.py
      │  ├─ policy.py
      │  ├─ reptile.py
      │  ├─ utils.py
      │  └─ train.py
      └─ transfer/
         ├─ config.py
         ├─ extractor.py
         ├─ callbacks.py
         ├─ env_factory.py
         └─ train.py
```

---

## Setup

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Atari ROMs (Pong)

This project uses **Gymnasium Atari**. You have two common options:

**Option A (recommended):**
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

**Option B (AutoROM):**
```bash
pip install autorom[accept-rom-license]
AutoROM --accept-license
```

Quick sanity check:
```bash
python -c "import gymnasium as gym; env=gym.make('ALE/Pong-v5'); env.reset(); print('Pong OK')"
```

---

## Transfer learning (PPO)

Train PPO on a single environment using the same CNN feature extractor for all games:

```bash
python -m gtlrl.transfer.train --env pong --timesteps 1000000
python -m gtlrl.transfer.train --env snake --timesteps 500000
python -m gtlrl.transfer.train --env puckworld --timesteps 500000
```

Outputs go to `outputs/transfer/<env>/<timestamp>/` (models, plots, TensorBoard logs).

---

## Meta-learning (first-order MAML/Reptile-style)

This module implements a *first-order* meta-learning loop that learns a good initialization per-environment
(you can call it “Reptile” or “FOMAML-ish” depending on how charitable you’re feeling).

```bash
python -m gtlrl.meta.train --iterations 200 --k_shots 5
```

Outputs go to `outputs/meta/<timestamp>/`.

---

## Continual learning (sine tasks)

Runs:
- transfer baselines (scratch/freeze/finetune) across sine tasks
- sine-wave MAML
- EWC forgetting matrix

```bash
python -m gtlrl.continual.run --all
```

Outputs go to `outputs/continual/<timestamp>/`.

---

## Notes / Differences vs your original code

- Fixed broken imports and corrupted text in `meta_learning/maml.py`.
- Removed duplicated environment definitions and centralized them under `gtlrl/envs`.
- Standardized to **gymnasium** APIs (`reset -> (obs, info)`, `step -> (obs, reward, terminated, truncated, info)`).
- Added clean CLI entrypoints via `python -m ...` modules.
- Added consistent output directories and simple plotting.
- The “meta-learning RL” portion in the original repo was not actually sharing a single initialization across tasks
  (because architectures and action spaces differed). This version makes that explicit and uses a first-order update per env.

---

## License

Educational / research use. If you publish, cite the libraries you used and don’t commit ROMs to GitHub unless
you enjoy getting emails from lawyers.
