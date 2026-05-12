# Skin RL

Reinforcement learning experiments for skin cancer classification using DQN and Dueling DQN models.

The project was refactored from a notebook into a small Python codebase:

- `dqn.py` contains the standard DQN network.
- `duelingdqn.py` contains the Dueling DQN network.
- `main.py` handles data loading, environment setup, training, evaluation, and optional plotting.
- `requirements.txt` lists the Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## Data

By default, `main.py` uses the original Kaggle paths from the notebook:

```text
/kaggle/input/datasets/abhisheksairam278/skincancer/nmed_rn34_ham10k_vectors.npy
/kaggle/input/datasets/abhisheksairam278/skincancer/vectorDB.csv
```

For local data, pass custom paths:

```bash
python main.py \
  --feature-path /path/to/nmed_rn34_ham10k_vectors.npy \
  --csv-path /path/to/vectorDB.csv
```

## Train

Train the Dueling DQN model:

```bash
python main.py --model dueling --epochs 30
```

Train the standard DQN model:

```bash
python main.py --model dqn --epochs 30
```

Show training plots:

```bash
python main.py --model dueling --epochs 30 --plot
```

Use reward-only bandit-style targets:

```bash
python main.py --model dueling --epochs 30 --bandit
```

## Outputs

After training, the script prints:

- epoch reward
- epoch accuracy
- confusion matrix
- classification report

## Notes

The class label `scc` is merged into `akiec`, matching the original notebook logic.
