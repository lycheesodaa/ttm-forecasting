import math
import os
import argparse

from utils import log_into_csv, create_revision_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import yaml
import glob
import logging
from importlib import resources
from pathlib import Path

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

CH_MIX = True

# DATA ROOT PATH
DATA_ROOT_PATH = "datasets/"

parser = argparse.ArgumentParser(description='TTM-demand-price')

# Configuration: choose config file
parser.add_argument('--config', type=str, required=False, default=None, help='Path to config yaml')
parser.add_argument('--target_dataset', type=str, required=True, help='dataset config yaml name')
parser.add_argument('--dataset_freq', type=str, required=True, default='1h', help='freq of dataset, [1h, D]')

args = parser.parse_args()

assert args.target_dataset in ['demand_price', 'demand_price_aus']

# csv output dir
output_dir = f'results/{args.target_dataset}/ch_mix/'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Results dir
OUT_DIR = "ttm_finetuned_models/"

# global param setting
BSZ = 8
grad_acc = 2

PRED_LEN = 48  # Fixed to horizon 48 due to task requirements


def zeroshot_eval(dataset_name, batch_size, context_length=512, forecast_length=PRED_LEN):
    # Get data
    _, _, dset_test, tsp, cfg = load_dataset(
        dataset_name=dataset_name,
        context_length=context_length,
        forecast_length=forecast_length,
        fewshot_fraction=1.0,
        dataset_root_path=DATA_ROOT_PATH,
        conditional_columns=[]
    )

    # Load model
    model_revision = create_revision_name(context_length, forecast_length)
    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        revision=model_revision,
        prediction_filter_length=forecast_length,
    )

    temp_dir = tempfile.mkdtemp()
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            seed=SEED,
        ),
    )
    print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.predict(dset_test)
    print(zeroshot_output.metrics)
    print("+" * 60)

    col_list = cfg['target_columns'] + cfg['conditional_columns']
    reshaped = zeroshot_output.predictions[0][:, :, 0].flatten()

    timestamps = []
    true = []
    for item in dset_test:
        end_timestamp = pd.Timestamp(item['timestamp'])
        future_dates = pd.date_range(
            start=end_timestamp,
            periods=forecast_length + 1,
            freq=args.dataset_freq
        )
        timestamps.extend(future_dates[1:])
        true.extend(item['future_values'][:forecast_length, 0].flatten().tolist())

    to_export = pd.DataFrame({
        'date': timestamps,
        'true': true,
        col_list[0]: reshaped
    })
    to_export[col_list[0]] = tsp.inverse_scale_targets(to_export[[col_list[0]]])
    to_export['true'] = tsp.inverse_scale_targets(to_export[['true']].rename(columns={'true': col_list[0]}))

    to_export.to_csv(output_dir + f'TTMs_pl{forecast_length}_zeroshot.csv')
    log_into_csv(to_export, dataset_name, 'zeroshot', bsz=batch_size, pred_filter_len=forecast_length,
                 pred_col_name=col_list[0])


def finetune_eval(
    dataset_name,
    batch_size,
    gradient_accumulation_steps=1,
    learning_rate=0.001,
    context_length=512,
    forecast_length=PRED_LEN,
    fewshot_percent=5,
    freeze_backbone=True,
    num_epochs=50,
    save_dir=OUT_DIR,
):
    out_dir = os.path.join(save_dir, dataset_name)

    if fewshot_percent == 100:
        print("-" * 20, f"Running full-shot", "-" * 20)
    else:
        print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)

    dset_train, dset_val, dset_test, tsp, cfg = load_dataset(
        dataset_name,
        context_length,
        forecast_length,
        fewshot_fraction=fewshot_percent / 100,
        dataset_root_path=DATA_ROOT_PATH,
    )

    model_revision = create_revision_name(context_length, forecast_length)
    if CH_MIX:
        finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",
            revision=model_revision,
            prediction_filter_length=forecast_length,
            num_input_channels=tsp.num_input_channels,
            decoder_mode="mix_channel",
            prediction_channel_indices=tsp.prediction_channel_indices,
        )
    else:
        finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r1",
            revision=model_revision,
            prediction_filter_length=forecast_length,
        )

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=finetune_forecast_model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), TrackingCallback()],
    )

    trainer.train()
    trainer.save_model(out_dir)
    print("Model saved to", out_dir)

    # Evaluate
    print("Evaluating on test set...")
    test_output = trainer.predict(dset_test)
    print(test_output.metrics)

    col_list = cfg['target_columns'] + cfg['conditional_columns']
    reshaped = test_output.predictions[0][:, :, 0].flatten()
    timestamps = []
    true = []
    for item in dset_test:
        end_timestamp = pd.Timestamp(item['timestamp'])
        future_dates = pd.date_range(
            start=end_timestamp,
            periods=forecast_length + 1,
            freq=args.dataset_freq
        )
        timestamps.extend(future_dates[1:])
        true.extend(item['future_values'][:forecast_length, 0].flatten().tolist())
    to_export = pd.DataFrame({
        'date': timestamps,
        'true': true,
        col_list[0]: reshaped
    })
    to_export[col_list[0]] = tsp.inverse_scale_targets(to_export[[col_list[0]]])
    to_export['true'] = tsp.inverse_scale_targets(to_export[['true']].rename(columns={'true': col_list[0]}))
    to_export.to_csv(output_dir + f'TTMs_pl{forecast_length}_finetune.csv')
    log_into_csv(to_export, dataset_name, 'finetune', bsz=batch_size, pred_filter_len=forecast_length,
                 pred_col_name=col_list[0])


if __name__ == "__main__":
    err_log = []
    try:
        zeroshot_eval(args.target_dataset, BSZ, context_length=512, forecast_length=PRED_LEN)
    except Exception as e:
        logging.error(f"Zeroshot eval failed: {e}")
        err_log.append(str(e))
    try:
        finetune_eval(args.target_dataset, BSZ, gradient_accumulation_steps=grad_acc, context_length=512, forecast_length=PRED_LEN)
    except Exception as e:
        logging.error(f"Finetune eval failed: {e}")
        err_log.append(str(e))
    if err_log:
        print("Errors encountered:", err_log)
