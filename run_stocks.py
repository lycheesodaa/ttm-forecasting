import math
import os

import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='TTMs_stocks')
parser.add_argument('--filepath', type=str)
parser.add_argument('--news_type', type=str, default='headlines',
                    choices=['headlines_sentiment', 'content_sentiment',
                             'headlines_emotion', 'content_emotion',
                             'headlines_historical', 'content_historical'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu_id', type=int, default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=f'{args.gpu_id}'
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

# DATA ROOT PATH
DATA_ROOT_PATH = "datasets/"
DATA_REL_PATH = "stocks/candle_w_emotion/day_average_content/"
# DATA_REL_PATH = "stocks/candle_w_emotion/day_average_headlines/"

target_dataset = f"stocks_{args.news_type}"

# csv output dir
output_dir = f'results/stocks_fyp/{target_dataset}/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Results dir
OUT_DIR = "ttm_finetuned_models/"

DATASET_FREQ = 'B'

# TTM model branch
# Use main for 512-96 model
# Use "1024_96_v1" for 1024-96 model
TTM_MODEL_REVISION = "main"

# global param setting
BSZ = args.batch_size

def zeroshot_eval(dataset_name, batch_size, context_length=512, forecast_length=96, prediction_filter_length=None):
    dataframes = []
    filepath = args.filepath if args.filepath is not None else Path(DATA_ROOT_PATH) / DATA_REL_PATH
    for file in tqdm(os.listdir(filepath)):
        # Get data
        try:
            _, _, dset_test, tsp, cfg = load_dataset(
                dataset_name=dataset_name,
                context_length=context_length,
                forecast_length=forecast_length,
                fewshot_fraction=1.0,
                # dataset_root_path=DATA_ROOT_PATH,
                dataset_path=Path(filepath) / file,
            )
        except IndexError as e:
            # a lot fewer predictions, since the context and forecast lengths are fixed
            print(e, f'Skipping {file}')
            continue

        # model_cfg = TinyTimeMixerConfig(
        #     context_length=context_length,
        #     prediction_length=forecast_length
        # )

        # Load model
        if prediction_filter_length is None:
            zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-v1",
                revision=TTM_MODEL_REVISION,
            )
        else:
            if prediction_filter_length <= forecast_length:
                zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
                    "ibm-granite/granite-timeseries-ttm-v1",
                    revision=TTM_MODEL_REVISION,
                    prediction_filter_length=prediction_filter_length,
                )
            else:
                raise ValueError("`prediction_filter_length` should be <= `forecast_length")
        temp_dir = tempfile.mkdtemp()
        # zeroshot_trainer
        zeroshot_trainer = Trainer(
            model=zeroshot_model,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size,
                seed=SEED,
            ),
        )
        # evaluate = zero-shot performance
        print("+" * 20, "Test MSE zero-shot", "+" * 20)
        # zeroshot_output = zeroshot_trainer.evaluate(dset_test)
        zeroshot_output = zeroshot_trainer.predict(dset_test)
        print(zeroshot_output.metrics)
        print("+" * 60)

        # extract only the predicted target column values and flatten
        col_list = cfg['target_columns'] + cfg['conditional_columns']
        # reshaped = zeroshot_output.predictions[0].reshape(-1, len(col_list))
        reshaped = zeroshot_output.predictions[0][:, :, 0].flatten()

        # extract timestamps and generate future dates
        timestamps = []
        true = []
        for item in dset_test:
            # dset dict objects are defined in tsfm_public.toolkit.dataset.ForecastDFDataset
            end_timestamp = pd.Timestamp(item['timestamp'])
            future_dates = pd.date_range(
                start=end_timestamp,
                periods=prediction_filter_length + 1,
                freq=DATASET_FREQ
            )
            timestamps.extend(future_dates[1:])
            true.extend(item['future_values'][:prediction_filter_length, 0].flatten().tolist())

        # create DataFrame with predictions and dates
        to_export = pd.DataFrame({
            'date': timestamps,
            'true': true,
            'stock': (Path(filepath) / file).stem,
            col_list[0]: reshaped
        })
        to_export[col_list[0]] = tsp.inverse_scale_targets(to_export[[col_list[0]]])
        to_export['true'] = tsp.inverse_scale_targets(to_export[['true']].rename(columns={'true':'close'}))

        dataframes.append(to_export)

    # concat all dataframes
    result = pd.concat(dataframes)

    # dset_test should be the same size as reshaped
    result.to_csv(output_dir + f'TTMs_pl{prediction_filter_length}_zeroshot.csv')

        # plot
        # plot_predictions(
        #     model=zeroshot_trainer.model,
        #     dset=dset_test,
        #     plot_dir=os.path.join(OUT_DIR, dataset_name),
        #     plot_prefix="test_zeroshot",
        #     channel=0,
        # )cfg['target_columns'] + cfg['conditional_columns']


def finetune_eval(
    dataset_name,
    batch_size,
    learning_rate=0.001,
    context_length=512,
    forecast_length=96,
    fewshot_percent=5,
    freeze_backbone=True,
    num_epochs=50,
    save_dir=OUT_DIR,
    prediction_filter_length=None,
):
    out_dir = os.path.join(save_dir, dataset_name)

    if fewshot_percent == 100:
        print("-" * 20, f"Running full-shot", "-" * 20)
    else:
        print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)

    # Data prep: Get dataset
    dset_train, dset_val, dset_test, tsp, cfg = load_dataset(
        dataset_name,
        context_length,
        forecast_length,
        fewshot_fraction=fewshot_percent / 100,
        dataset_root_path=DATA_ROOT_PATH,
    )

    # change head dropout to 0.7 for ett datasets
    if "ett" in dataset_name:
        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-v1", revision=TTM_MODEL_REVISION, head_dropout=0.7
            )
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-v1",
                revision=TTM_MODEL_REVISION,
                head_dropout=0.7,
                prediction_filter_length=prediction_filter_length,
            )
        else:
            raise ValueError("`prediction_filter_length` should be <= `forecast_length")
    else:
        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-v1",
                revision=TTM_MODEL_REVISION,
            )
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-v1",
                revision=TTM_MODEL_REVISION,
                prediction_filter_length=prediction_filter_length,
            )
        else:
            raise ValueError("`prediction_filter_length` should be <= `forecast_length")

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        seed=SEED,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    # Evaluation
    if fewshot_percent == 100:
        print("+" * 20, f"Test MSE after full-shot {fewshot_percent}% fine-tuning", "+" * 20)
    else:
        print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
    # fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
    fewshot_output = finetune_forecast_trainer.predict(dset_test)
    print(fewshot_output.metrics)
    print("+" * 60)

    col_list = cfg['target_columns'] + cfg['conditional_columns']
    # reshaped = fewshot_output.predictions[0].reshape(-1, len(col_list)) # no point in keeping other columns
    # to_export = pd.DataFrame(reshaped, columns=col_list)

    reshaped = fewshot_output.predictions[0][:, :, 0].flatten()

    # Extract timestamps and generate future dates
    timestamps = []
    true = []
    for item in dset_test:
        end_timestamp = pd.Timestamp(item['timestamp'])
        future_dates = pd.date_range(
            start=end_timestamp,
            periods=prediction_filter_length + 1,
            freq=DATASET_FREQ
        )
        timestamps.extend(future_dates[1:])
        true.extend(item['future_values'][:prediction_filter_length, 0].flatten().tolist())

    # Create DataFrame with predictions and dates
    to_export = pd.DataFrame({
        'date': timestamps,
        'true': true,
        col_list[0]: reshaped
    })
    to_export[col_list[0]] = tsp.inverse_scale_targets(to_export[[col_list[0]]])
    to_export['true'] = tsp.inverse_scale_targets(to_export[['true']].rename(columns={'true':'close'}))

    if fewshot_percent == 100:
        to_export.to_csv(f'results/TTMs_pl{prediction_filter_length}_predictions_fullshot.csv')
    else:
        to_export.to_csv(f'results/TTMs_pl{prediction_filter_length}_predictions_fewshot{fewshot_percent}.csv')

    # plot
    # plot_predictions(
    #     model=finetune_forecast_trainer.model,
    #     dset=dset_test,
    #     plot_dir=os.path.join(OUT_DIR, dataset_name),
    #     plot_prefix="test_fewshot",
    #     channel=0,
    # )

pred_lens = [1, 3, 7, 14] # stocks
# pred_lens = [1] # stocks
for pl in pred_lens:
    zeroshot_eval(
        dataset_name=target_dataset,
        batch_size=BSZ,
        prediction_filter_length=pl
    )

    # finetune_eval(
    #     dataset_name=target_dataset,
    #     batch_size=BSZ,
    #     prediction_filter_length=pl,
    #     fewshot_percent=100
    # )
