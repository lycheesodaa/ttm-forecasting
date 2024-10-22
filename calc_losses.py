import os
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


def calculate_mape(y_true: list, y_pred: list) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Args:
    y_true: Array of true values
    y_pred: Array of predicted values

    Returns:
    float: The calculated MAPE

    Raises:
    ValueError: If the input arrays have different shapes
    ZeroDivisionError: If any true value is zero
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted arrays must have the same shape")

    if np.any(y_true == 0):
        raise ZeroDivisionError("MAPE is undefined when true values contain zeros")

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


pred_lens = [1, 3, 7, 14]
results_dir = 'results/stocks_fyp/'
runs = os.listdir(Path(results_dir))
runs = [run for run in runs if '.' not in run]
runs.sort()

results = {
    f'pl{pred_len}': {} for pred_len in pred_lens
}

for run_dir in runs:
    full_run_dir = Path(results_dir) / run_dir
    run_name = run_dir.replace('stocks_', '')

    for pred_len in pred_lens:
        # zero shot
        temp_df = pd.read_csv(full_run_dir / f'TTMs_pl{pred_len}_zeroshot.csv')
        temp_df.dropna(inplace=True)

        results[f'pl{pred_len}'][run_name] = calculate_mape(temp_df['true'], temp_df['close'])

        # TODO fine-tuned

pd.DataFrame(results).to_csv('mape_losses.csv')