import argparse
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dqn import QNetwork
from duelingdqn import DuelingQNetwork


REWARD_MATRIX = np.array(
    [
        [2, -2, -3, -3, -2, -3, -3],
        [-2, 3, -4, -4, -2, -4, -4],
        [-2, -2, 1, -2, -3, -2, -2],
        [-2, -2, -2, 1, -3, -2, -2],
        [-8, -7, -10, -10, 5, -12, -10],
        [-2, -2, -2, -2, -3, 1, -2],
        [-2, -2, -2, -2, -3, -2, 1],
    ],
    dtype=np.float32,
)


class SkinCancerEnv:
    def __init__(self, states, labels, reward_matrix=REWARD_MATRIX):
        self.states = states
        self.labels = labels
        self.reward_matrix = reward_matrix
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self.states[self.idx]

    def step(self, action):
        true_label = self.labels[self.idx]
        reward = self.reward_matrix[true_label][action]

        self.idx += 1
        done = self.idx >= len(self.states)
        next_state = None if done else self.states[self.idx]

        return next_state, reward, done, true_label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(feature_path, csv_path, normalize=True):
    cnn_features = np.load(feature_path)
    if normalize:
        cnn_features = StandardScaler().fit_transform(cnn_features)

    df = pd.read_csv(csv_path)
    df["dx"] = df["dx"].replace("scc", "akiec")

    labels = df["dx"].values
    probabilities = df.drop(columns=["dx"]).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    states = np.concatenate([cnn_features, probabilities], axis=1).astype(np.float32)

    return states, y, label_encoder


def create_model(model_name, input_dim, feature_dim, prob_dim, num_actions, device):
    if model_name == "dqn":
        model = QNetwork(
            feature_dim=feature_dim,
            prob_dim=prob_dim,
            num_actions=num_actions,
        )
    elif model_name == "dueling":
        model = DuelingQNetwork(input_dim=input_dim, num_actions=num_actions)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def choose_action(model, state, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.argmax(model(state_t)).item()


def train(
    model,
    target_model,
    env,
    device,
    *,
    epochs=30,
    gamma=0.99,
    epsilon=0.2,
    lr=2.5e-4,
    batch_size=64,
    max_memory=10000,
    bandit=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    memory = deque(maxlen=max_memory)
    input_dim = env.states.shape[1]

    reward_history = []
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        state = env.reset()
        total_reward = 0

        while True:
            action = choose_action(model, state, epsilon, env.reward_matrix.shape[1], device)
            next_state, reward, done, _ = env.step(action)

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                loss = train_step(
                    model,
                    target_model,
                    optimizer,
                    loss_fn,
                    memory,
                    batch_size,
                    input_dim,
                    gamma,
                    device,
                    bandit,
                )
                loss_history.append(loss)

            if done:
                break

        target_model.load_state_dict(model.state_dict())
        accuracy = evaluate_accuracy(model, env, device)

        reward_history.append(total_reward)
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch + 1} | Reward: {total_reward:.2f} | Acc: {accuracy:.4f}")

    return reward_history, loss_history, accuracy_history


def train_step(
    model,
    target_model,
    optimizer,
    loss_fn,
    memory,
    batch_size,
    input_dim,
    gamma,
    device,
    bandit,
):
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    next_states = torch.tensor(
        np.array([s if s is not None else np.zeros(input_dim) for s in next_states]),
        dtype=torch.float32,
    ).to(device)

    q_values = model(states)
    q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        if bandit:
            target = rewards
        else:
            next_q = target_model(next_states).max(1)[0]
            target = rewards + gamma * next_q * (1 - dones)

    loss = loss_fn(q_action, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, env, device, class_names):
    preds = []
    true_labels = []
    state = env.reset()

    while True:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_t)).item()

        next_state, _, done, true_label = env.step(action)
        preds.append(action)
        true_labels.append(true_label)

        if done:
            break
        state = next_state

    print("\nConfusion Matrix:\n", confusion_matrix(true_labels, preds))
    print("\nReport:\n", classification_report(true_labels, preds, target_names=class_names))


def evaluate_accuracy(model, env, device):
    correct = 0
    total = 0
    state = env.reset()

    while True:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_t)).item()

        next_state, _, done, true_label = env.step(action)
        correct += int(action == true_label)
        total += 1

        if done:
            break
        state = next_state

    return correct / total


def smooth(values, window=5):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(reward_history, loss_history, accuracy_history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(smooth(reward_history))
    plt.title("Reward Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")

    plt.subplot(1, 3, 2)
    plt.plot(smooth(loss_history))
    plt.title("Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(smooth(accuracy_history))
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN models for skin cancer classification.")
    parser.add_argument("--model", choices=["dqn", "dueling"], default="dueling")
    parser.add_argument(
        "--feature-path",
        default="/kaggle/input/datasets/abhisheksairam278/skincancer/nmed_rn34_ham10k_vectors.npy",
        help="Path to .npy CNN feature vectors.",
    )
    parser.add_argument(
        "--csv-path",
        default="/kaggle/input/datasets/abhisheksairam278/skincancer/vectorDB.csv",
        help="Path to vectorDB.csv.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-memory", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--bandit", action="store_true", help="Use reward-only target without bootstrapping.")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    states, labels, label_encoder = load_data(
        args.feature_path,
        args.csv_path,
        normalize=not args.no_normalize,
    )
    print("Classes:", label_encoder.classes_)
    print("State shape:", states.shape)

    input_dim = states.shape[1]
    feature_dim = 512
    prob_dim = input_dim - feature_dim
    num_actions = len(label_encoder.classes_)

    model = create_model(args.model, input_dim, feature_dim, prob_dim, num_actions, device)
    target_model = create_model(args.model, input_dim, feature_dim, prob_dim, num_actions, device)
    target_model.load_state_dict(model.state_dict())

    env = SkinCancerEnv(states, labels)
    reward_history, loss_history, accuracy_history = train(
        model,
        target_model,
        env,
        device,
        epochs=args.epochs,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr,
        batch_size=args.batch_size,
        max_memory=args.max_memory,
        bandit=args.bandit,
    )

    evaluate(model, env, device, label_encoder.classes_)

    if args.plot:
        plot_training(reward_history, loss_history, accuracy_history)


if __name__ == "__main__":
    main()
