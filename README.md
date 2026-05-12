# Skin RL

Reinforcement learning experiments for skin cancer classification using DQN and Dueling DQN models.

The project was refactored from a notebook into a small Python codebase:

- `dqn.py` contains the standard DQN network.
- `duelingdqn.py` contains the Dueling DQN network.
- `main.py` handles data loading, environment setup, training, evaluation, and optional plotting.
- `paper_demos/` contains the paper-style TensorFlow/Gym demo scripts for diagnosis, lesion-level management, and patient-level management.
- `data/` contains the included HAM10000 vector data used by the refactored DQN workflow.
- `management/` contains management comparison plotting scripts and the saved training log.
- `requirements.txt` lists the Python dependencies.

This project also references the original paper codebase:
[catarina-barata/Skin_RL](https://github.com/catarina-barata/Skin_RL).

## Setup

```bash
pip install -r requirements.txt
```

## Data

The repo includes:

```text
data/nmed_rn34_ham10k_vectors.npy
data/vectorDB.csv
```

By default, `main.py` uses the original Kaggle paths from the notebook:

```text
/kaggle/input/datasets/abhisheksairam278/skincancer/nmed_rn34_ham10k_vectors.npy
/kaggle/input/datasets/abhisheksairam278/skincancer/vectorDB.csv
```

For local data, pass custom paths:

```bash
python main.py \
  --feature-path data/nmed_rn34_ham10k_vectors.npy \
  --csv-path data/vectorDB.csv
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

## Paper Demos

The `paper_demos/` folder contains the three demo scripts from the paper workflow:

- `RL_Skin_Cancer_Demo_Diagnosis.py`
- `RL_Skin_Cancer_Demo_Management.py`
- `Skin_Cancer_RL_Demo_Patient_Management.py`

Example commands:

```bash
python paper_demos/RL_Skin_Cancer_Demo_Diagnosis.py --n_patients 100 --n_episodes 150 --use_unknown False
python paper_demos/RL_Skin_Cancer_Demo_Management.py --n_patients 100 --n_episodes 150 --n_actions 2
python paper_demos/Skin_Cancer_RL_Demo_Patient_Management.py --n_patients 1 --n_episodes 130 --n_actions 3
```

These scripts expect the paper demo data files under a `data/` folder, following the original repository format.

## Management Plots

The `management/` folder contains plotting helpers for the lesion-level management replication:

```bash
cd management
python plot_management_comparison.py
python plot_management_separate.py
```

The scripts read `training_log.txt` and expect baseline result files in the same working directory when generating comparison plots.

## Outputs

After training, the script prints:

- epoch reward
- epoch accuracy
- confusion matrix
- classification report

## Notes

The class label `scc` is merged into `akiec`, matching the original notebook logic.
