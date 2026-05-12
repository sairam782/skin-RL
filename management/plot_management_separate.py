"""
Render the three management confusion matrices as SEPARATE high-res PNGs,
sized for individual placement in slides.

Generates:
  plots/mgmt_naive_sl.png    + .pdf
  plots/mgmt_threshold_sl.png + .pdf
  plots/mgmt_rl_dqn.png      + .pdf
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from plot_management_comparison import (
    CLASS_ORDER_ALPHA, PAPER_ROW_ORDER, ACTION_NAMES, CODE_TO_PAPER_COL,
    PAPER_NAIVE, PAPER_THRESHOLD, PAPER_RL,
    reorder_to_paper, parse_best_actions_table,
)


def single_heatmap(ours, paper, title, out_path):
    """Render a single 7x3 heatmap as its own figure, slide-friendly proportions."""
    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.get_cmap('Oranges')
    im = ax.imshow(ours, vmin=0, vmax=1, cmap=cmap, aspect='auto')

    ax.set_xticks(range(3))
    ax.set_xticklabels(ACTION_NAMES, fontsize=13)
    ax.set_yticks(range(7))
    ax.set_yticklabels([c.upper() for c in PAPER_ROW_ORDER], fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Action', fontsize=13, labelpad=8)
    ax.set_ylabel('Ground truth', fontsize=13, labelpad=8)

    for i in range(ours.shape[0]):
        for j in range(ours.shape[1]):
            ours_v = ours[i, j]
            paper_v = paper[i, j]
            text_color = 'white' if ours_v > 0.55 else 'black'
            ax.text(j, i, f'{ours_v:.2f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)
            ax.text(j, i + 0.30, f'(paper {paper_v:.2f})', ha='center', va='center',
                    fontsize=9, color=text_color, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label('Proportion', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path + '.png', dpi=180, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}.{{png,pdf}}')


def main():
    os.makedirs('plots', exist_ok=True)

    with open('sl_baseline_results.json') as f:
        sl = json.load(f)
    naive_alpha = np.array(sl['naive_proportions'])
    thr_alpha = np.array(sl['threshold_proportions'])
    naive_paper_layout = reorder_to_paper(naive_alpha)
    thr_paper_layout = reorder_to_paper(thr_alpha)

    rl_counts_alpha = parse_best_actions_table('training_log.txt')
    rl_props_alpha = rl_counts_alpha / np.maximum(
        rl_counts_alpha.sum(axis=1, keepdims=True), 1)
    rl_paper_layout = reorder_to_paper(rl_props_alpha)

    single_heatmap(naive_paper_layout, PAPER_NAIVE,
                   'Naive SL  —  Fig 2f replica',
                   'plots/mgmt_naive_sl')
    single_heatmap(thr_paper_layout, PAPER_THRESHOLD,
                   'Threshold-adjusted SL  —  Fig 2g replica',
                   'plots/mgmt_threshold_sl')
    single_heatmap(rl_paper_layout, PAPER_RL,
                   'RL DQN  —  Fig 2h replica',
                   'plots/mgmt_rl_dqn')


if __name__ == '__main__':
    main()
