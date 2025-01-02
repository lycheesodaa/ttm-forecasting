from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = Path('figures/carbon/')

pred_lens = [i for i in range (2,19)]
context_length = [i for i in range(15,7,-1)]
feat_pct = [8, 17, 26]  # 0.1, 0.2, 0.3 features

# calculate mape
output = []
for pct in feat_pct:
    ft_map = {}
    for ctx in context_length:
        ft_map[f'ctx_{ctx}'] = {}
        for pl in pred_lens:
            print(f'Running {pct}-feats, CTX-{ctx} and pl-{pl}...')
            df = pd.read_csv(f'results/carbon/TTMs_feat{pct}_ctx{ctx}_pl{pl}_fullshot.csv')

            mape = np.mean(np.abs((df['Price'] - df['true']) / df['true'])) * 100

            ft_map[f'ctx_{ctx}'][f'pl_{pl}'] = mape

    output.append(pd.DataFrame(ft_map))

for idx, mape_df in enumerate(output):
    mape_df.index = mape_df.index.str.replace('pl_', '')
    mape_df.columns = mape_df.columns.str.replace('ctx_', '')
    fig = plt.figure(figsize=(12, 12))
    ax = sns.heatmap(mape_df, annot=True, fmt=".2f")
    ax.set_title(f'Heatmap of MAPE across Prediction Lengths and Context Lengths ({feat_pct[idx]} features)')
    ax.set_xlabel('Context length')
    ax.set_ylabel('Prediction length')
    fig.savefig(output_dir / f'heatmap_feat{feat_pct[idx]}.png', bbox_inches='tight')

    # Create subplot figure
    n_rows, n_cols = 5, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,12), sharey=True)
    fig.suptitle('MAPE vs Context Length', fontsize=16, y=0.95)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Plot each dataset
    for i, (index, row) in enumerate(mape_df.iterrows()):
        if i >= 17:  # Only plot first 17 datasets
            break

        # Customize plot
        ax = axes[i]
        ax.plot(row, 'b-', marker='o', markersize=4)
        ax.set_title(f'PL-{index}', fontsize=10)
        ax.set_xticks(row.index)
        ax.set_xlabel('Context Length')
        ax.set_ylabel('MAPE (%)')
        ax.grid(True, linestyle='--', alpha=0.7)

    # Remove empty subplots
    for j in range(17, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output_dir / f'subplots_feat{feat_pct[idx]}.png', bbox_inches='tight')
