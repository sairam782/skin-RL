import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import argparse


CLASS_LABELS_7 = ['MEL', 'BCC', 'AKIEC', 'BKL', 'NV', 'DF', 'VASC']
CLASS_LABELS_8 = ['MEL', 'BCC', 'AKIEC', 'BKL', 'NV', 'DF', 'VASC', 'Refer']


def parse_confusion_matrix(text, section='reward'):
    """Extract the confusion matrix from a results file."""
    if section == 'bacc':
        marker = 'Scores for Best Validation BAcc:'
    else:
        marker = 'Scores for Best Validation Reward:'

    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Section '{marker}' not found in results file")

    sub = text[idx:]
    cm_start = sub.find('Confusion Matrix:\n')
    if cm_start == -1:
        raise ValueError("Confusion Matrix not found in section")

    cm_text = sub[cm_start + len('Confusion Matrix:\n'):]

    rows = []
    for line in cm_text.strip().split('\n'):
        line = line.strip()
        if not line or not line.startswith('['):
            break
        nums = re.findall(r'-?\d+', line)
        rows.append([int(n) for n in nums])

    return np.array(rows)


def parse_use_unknown(text):
    """Check if the results file used --use_unknown."""
    match = re.search(r'use_unknown:\s*(True|False)', text)
    if match:
        return match.group(1) == 'True'
    return False


def get_latest_results_file(results_dir):
    """Get the most recently modified results file."""
    files = glob.glob(os.path.join(results_dir, 'diagnosis_results_*.txt'))
    if not files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    return max(files, key=os.path.getmtime)


def plot_confusion_matrix(cm, labels, title='', save_path=None):
    """Plot a normalized confusion matrix as a blue heatmap."""
    n_rows, n_cols = cm.shape

    # Normalize rows (only non-empty rows)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    # Drop empty rows (e.g. the unkn/refer ground truth row is always 0)
    non_empty = cm.sum(axis=1) > 0
    cm_plot = cm_norm[non_empty]
    row_labels = [labels[i] for i in range(n_rows) if non_empty[i]]
    col_labels = labels[:n_cols]

    n_r = len(row_labels)
    n_c = len(col_labels)

    fig, ax = plt.subplots(figsize=(n_c + 2, n_r + 2))

    cmap = plt.cm.Reds

    im = ax.imshow(cm_plot, cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')

    # Axis ticks and labels
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight='bold')
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_labels, fontsize=11, fontweight='bold')

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Predicted Action', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')

    # Cell text
    for i in range(n_r):
        for j in range(n_c):
            val = cm_plot[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall', fontsize=11, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix from results file')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to results file (default: latest in results/)')
    parser.add_argument('--section', type=str, default='reward', choices=['bacc', 'reward'],
                        help='Which section to plot: bacc or reward (default: reward)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')

    if args.file:
        filepath = args.file
    else:
        filepath = get_latest_results_file(results_dir)

    print(f"Reading: {filepath}")

    with open(filepath, 'r') as f:
        text = f.read()

    cm = parse_confusion_matrix(text, section=args.section)
    use_unknown = parse_use_unknown(text)

    if use_unknown:
        labels = CLASS_LABELS_8
        title_suffix = '8-Action Confusion Matrix\n(Includes Referral Column)'
    else:
        labels = CLASS_LABELS_7
        cm = cm[:7, :7]
        title_suffix = '7-Action Confusion Matrix'

    section_label = 'Best BAcc' if args.section == 'bacc' else 'Best Reward'
    title = f'Complete {title_suffix}'

    basename = os.path.splitext(os.path.basename(filepath))[0]
    save_name = f"{basename}_{args.section}_cm.png"
    save_path = os.path.join(script_dir, save_name)

    plot_confusion_matrix(cm, labels, title=title, save_path=save_path)


if __name__ == '__main__':
    main()
