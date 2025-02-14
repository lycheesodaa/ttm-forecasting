from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

data_dir = Path('results/carbon_daily/')
output_dir = Path('figures/carbon_daily/')
output_dir.mkdir(parents=True, exist_ok=True)

pred_lens = [1, 2, 3, 4, 5, 7, 14, 21, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]
feat_pct = [0, 4, 13]  # 0, 0.3, 0.8 features

# # calculate mape
output = {}
for pct in feat_pct:
    if pct == 0: output['zeroshot'] = {}
    output[f'feat_{pct}'] = {}

    for pl in pred_lens:
        if pct == 0:
            zs = pd.read_csv(data_dir / f'TTMs_feat{pct}_ctx512_pl{pl}_zeroshot.csv')
            output['zeroshot'][f'pl_{pl}'] = np.mean(np.abs((zs['Price'] - zs['true']) / zs['true'])) * 100

        df = pd.read_csv(data_dir / f'TTMs_feat{pct}_ctx512_pl{pl}_fullshot.csv')

        mape = np.mean(np.abs((df['Price'] - df['true']) / df['true'])) * 100

        output[f'feat_{pct}'][f'pl_{pl}'] = mape

mape_df = pd.DataFrame(output)

print(mape_df)
mape_df.to_csv(output_dir / 'carbon_daily_results.csv')

# plot diagrams
mape_df.index = mape_df.index.str.replace('pl_', '')
mape_df.columns = mape_df.columns.str.replace('feat_', '')

# plot heatmap
fig = plt.figure(figsize=(10, 15))
ax = sns.heatmap(mape_df, annot=True, fmt=".2f")
ax.set_title(f'Heatmap of MAPE across Prediction Lengths and Number of features')
ax.set_xlabel('# Features')
ax.set_ylabel('Prediction length')
fig.savefig(output_dir / f'heatmap.png', bbox_inches='tight')

# subplot figure
n_rows, n_cols = 5, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12), sharey=True)
fig.suptitle('MAPE vs Number of features', fontsize=16)

# flatten axes for easier iteration
axes = axes.flatten()

# plot function
for i, (index, row) in enumerate(mape_df.iterrows()):
    # Customize plot
    ax = axes[i]
    ax.plot(row, 'b-', marker='o', markersize=4)
    ax.set_title(f'Pred. len. {index}', fontsize=10)
    ax.set_xticks(row.index)
    ax.set_xlabel('# Features')
    ax.set_ylabel('MAPE (%)')
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove empty subplots
for j in range(17, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
fig.savefig(output_dir / f'subplots.png', bbox_inches='tight')


# ALT subplot figure
n_rows, n_cols = 4, 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 15), sharey=True)
fig.suptitle('MAPE vs Forecast steps', fontsize=16)

# flatten axes for easier iteration
axes = axes.flatten()

# plot function
for i, (label, col) in enumerate(mape_df.items()):
    # Customize plot
    ax = axes[i]
    ax.plot(col.index.astype(int), col, 'b-', marker='o', markersize=4)
    ax.set_title(f'Selected {label} features', fontsize=10)
    if i == 0:
        ax.set_title('Zeroshot', fontsize=10)
    ax.set_xlabel('Forecast steps')
    ax.set_ylabel('MAPE (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('linear')

fig.tight_layout()
fig.savefig(output_dir / f'subplots_horizon.png', bbox_inches='tight')
exit()

# downsample to monthly by taking the mean of every month window
monthly_data = pd.read_csv('datasets/carbon/res_daily/merged_data_imputed.csv', index_col=0, parse_dates=True)
monthly_data = monthly_data[['Price']]
monthly_data.rename(columns={'Price': 'Actual'}, inplace=True)

def agg_monthly(df, pl, window_size):
    monthly_agg = []
    dates = []

    def find_first_month_start(df, month):
        mask = (df.index.day) == 1 & (df.index.month == month)
        matching = df.index[mask]

        return matching[0] if len(matching) > 0 else None

    print(df)
    first_window_end = df.index[0] + relativedelta(months=+2)
    print(first_window_end)
    curr_end_date = find_first_month_start(df, first_window_end.month)
    print(curr_end_date)
    exit()


# calculate mape
output = {}

for pct in feat_pct:
    if pct == 0: output['zeroshot'] = {}
    output[f'feat_{pct}'] = {}

    for pl in pred_lens:
        month_window = pl % 30 + 1  # e.g. 30-day predictions can span across 2 months, so month_window = 2

        if pct == 0:
            zs = pd.read_csv(data_dir / f'TTMs_feat{pct}_ctx512_pl{pl}_zeroshot.csv', parse_dates=True, index_col=1)

            agg_monthly(zs, pl, month_window)



        df = pd.read_csv(data_dir / f'TTMs_feat{pct}_ctx512_pl{pl}_fullshot.csv', parse_dates=True, index_col=1)



        output[f'feat_{pct}'][f'pl_{pl}'] = mape

mape_df = pd.DataFrame(output)

print(mape_df)

# plot diagrams
mape_df.index = mape_df.index.str.replace('pl_', '')
mape_df.columns = mape_df.columns.str.replace('feat_', '')

# plot heatmap
fig = plt.figure(figsize=(10, 8))
ax = sns.heatmap(mape_df, annot=True, fmt=".2f")
ax.set_title(f'Heatmap of MAPE across Prediction Lengths and Number of features')
ax.set_xlabel('# Features')
ax.set_ylabel('Prediction length')
fig.savefig(output_dir / f'heatmap.png', bbox_inches='tight')

# subplot figure
n_rows, n_cols = 1, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6), sharey=True)
fig.suptitle('MAPE vs Context Length', fontsize=16, y=0.95)

# flatten axes for easier iteration
axes = axes.flatten()

# plot function
for i, (index, row) in enumerate(mape_df.iterrows()):
    # Customize plot
    ax = axes[i]
    ax.plot(row, 'b-', marker='o', markersize=4)
    ax.set_title(f'PL-{index}', fontsize=10)
    ax.set_xticks(row.index)
    ax.set_xlabel('# Features')
    ax.set_ylabel('MAPE (%)')
    ax.grid(True, linestyle='--', alpha=0.7)

# Remove empty subplots
for j in range(17, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
