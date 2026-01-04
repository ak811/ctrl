# Bridging Tasks: Cross-Model Knowledge Sharing (Deep RL Project)

We study **cross-task knowledge reuse** in deep reinforcement learning using three complementary paradigms:

- **Transfer Learning**: pretrain on a source task, then adapt to a target task
- **Meta-Learning**: learn an initialization that adapts quickly (few-shot) to a task
- **Continual Learning**: learn tasks sequentially while reducing catastrophic forgetting

Environments used:
- **Snake** (custom)
- **PuckWorld** (custom)
- **Pong** (Atari: `ALE/Pong-v5`)

Plus a controlled **sine-wave regression benchmark** (for EWC + toy MAML sanity checks).

---

## Setup

### 1) Create and activate a venv
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Install Pong ROMs
Gymnasium Atari typically needs ROM install/acceptance.

Recommended:
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

Or AutoROM:
```bash
pip install "autorom[accept-rom-license]"
AutoROM --accept-license
```

Sanity check:
```bash
python -c "import gymnasium as gym; env=gym.make('ALE/Pong-v5'); env.reset(); print('Pong OK')"
```

---

## Transfer learning (PPO, Stable-Baselines3)

Train PPO on one environment:

```bash
python -m bridging_tasks.transfer.train --env pong --timesteps 1000000
python -m bridging_tasks.transfer.train --env snake --timesteps 500000
python -m bridging_tasks.transfer.train --env puckworld --timesteps 500000
```

Outputs:
- `outputs/transfer/<env>/<timestamp>/`
  - saved SB3 model (`.zip`)
  - reward curve plot
  - tensorboard logs

---

## Meta-learning (first-order RL meta loop)

The original project report frames meta-learning as MAML-style adaptation.
In practice, implementing full second-order MAML in RL is expensive and brittle, so this repo includes a
**first-order** meta-learning loop (Reptile-style) that still learns an initialization that adapts quickly.

```bash
python -m bridging_tasks.meta.train --iterations 200 --k_shots 5
```

Outputs:
- `outputs/meta/<timestamp>/` (plots + init checkpoints)

---

## Continual learning (EWC on sine benchmark)

Runs:
- sine transfer baselines (scratch / freeze / finetune)
- toy sine MAML
- EWC forgetting matrix

```bash
python -m bridging_tasks.continual.run --all
```

Outputs:
- `outputs/continual/<timestamp>/`

---

## License

MIT.
