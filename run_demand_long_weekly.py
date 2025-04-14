import math
import os

from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPredictionOutput
from tsfm_public.toolkit.time_series_preprocessor import create_timestamps
from utils import log_into_csv

os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import yaml
import glob
import logging
from importlib import resources
from pathlib import Path
import seaborn as sns

import tempfile
import pandas as pd

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TinyTimeMixerForPrediction, TrackingCallback, count_parameters, load_dataset, \
    TinyTimeMixerConfig, TimeSeriesForecastingPipeline
from tsfm_public.toolkit.visualization import plot_predictions

# Set seed for reproducibility
SEED = 1234
set_seed(SEED)

CH_MIX=True

# DATA ROOT PATH
DATA_ROOT_PATH = "datasets/"

# DATASET CONFIG FILE (CHANGE THIS ACCORDING TO THE DATASET)
target_dataset = "demand_sg_long_weekly"

# csv output dir
output_dir = f'results/{target_dataset}/ch_mix/'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Results dir
OUT_DIR = "ttm_finetuned_models/"

DATASET_FREQ = 'W'

# TTM model branch
# Use main for 512-96 model
# Use "1024_96_v1" for 1024-96 model
# TTM_MODEL_REVISION = "main"

# global param setting
BSZ = 16
grad_acc = 1

def zeroshot_eval(dataset_name, batch_size, context_length=512, forecast_length=336):
    model_revision_name = f'{context_length}-{forecast_length}-r2'

    # Get data
    data = pd.read_csv(
        os.path.join(DATA_ROOT_PATH, 'demand/demand_full_weekly.csv'),
        parse_dates=["datetime"],
    )

    # Reformat data for model processing
    model_input = data.iloc[-512:, 1:2].values.reshape(1, 512, 1)

    # Load model
    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        revision=model_revision_name,
    )

    output: TinyTimeMixerForPredictionOutput = zeroshot_model(torch.Tensor(model_input))
    output_df = pd.DataFrame(output.prediction_outputs.reshape(-1, 1).detach().cpu().numpy(), columns=['system_demand_actual'])

    output_df['datetime'] = create_timestamps(last_timestamp=data['datetime'].iloc[-1], freq='W', periods=forecast_length)

    # dset_test should be the same size as reshaped
    output_df.to_csv(output_dir + f'TTMs_ctx_{context_length}_pl{forecast_length}_zeroshot.csv')

    # plot and export figure
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data, x='datetime', y='system_demand_actual', label='true')
    sns.lineplot(output_df, x='datetime', y='system_demand_actual', label='pred')

    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=10, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.xlabel('Date')
    plt.ylabel('Demand (KwH)')
    plt.title('5-year ahead weekly demand forecasts')
    plt.legend()
    plt.grid()
    plt.show()
    plt.tight_layout()
    fig.savefig(output_dir + 'ttm_5year.png', bbox_inches='tight')

err_log = []
context_lengths = [512]
for CTX in context_lengths:
    try:
        zeroshot_eval(
            dataset_name=target_dataset,
            batch_size=BSZ * grad_acc,
            context_length=CTX
        )
    except Exception as e:
        print(e.with_traceback())
        err_log.append(e)

    torch.cuda.empty_cache()

print(err_log)
