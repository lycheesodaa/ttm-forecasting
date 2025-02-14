import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import os

from tqdm import tqdm

def calculate_mse(y_true: list, y_pred: list) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Args:
    y_true: Array of true values
    y_pred: Array of predicted values

    Returns:
    float: The calculated MSE

    Raises:
    ValueError: If the input arrays have different shapes
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted arrays must have the same shape")

    return np.mean((y_true - y_pred) ** 2)


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


def log_into_csv(
    results_df: pd.DataFrame,
    name: str,
    stage: str,
    ch_mix: bool = None,
    seq_len: int = 512,
    pred_len: int = 96,
    pred_filter_len: int | None = 96,
    lr: float = None,
    bsz: int = 16,
    log_file_name: str = 'demand',
    pred_col_name: str = 'actual',
):
    log_file = f'results/{log_file_name}_runs.csv'

    # Create sample first line in records
    if not os.path.exists(log_file):
        df = pd.DataFrame({
            'timestamp': datetime.datetime.now(),
            'name': 'sample',
            'stage': 'finetuned',
            'model': 'TTM',
            'ch_mix': True,
            'seq_len': 512,
            'pred_len': 96,
            'pred_filter_len': 96,
            'lr': 0.01,
            'bsz': 16,
            'score_type': 'mape',
            'score': 1.23,
        }, index=[0])
        df.to_csv(log_file)

    curr_run = pd.DataFrame({
        'timestamp': datetime.datetime.now(),
        'name': name,
        'stage': stage,
        'model': 'TTM',
        'ch_mix': ch_mix,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'pred_filter_len': pred_filter_len,
        'lr': lr,
        'bsz': bsz,
        'score_type': 'mape',
        'score': calculate_mape(results_df['true'], results_df[pred_col_name])
    }, index=[0])

    df = pd.read_csv(log_file, index_col=0)
    assert len(df.columns) == len(curr_run.columns)

    df = pd.concat([df, curr_run]).reset_index(drop=True)
    df.to_csv(log_file)
