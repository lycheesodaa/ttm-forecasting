import math
import os

from utils import log_into_csv, create_revision_name

os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

# DATA ROOT PATH
DATA_ROOT_PATH = "datasets/"

target_dataset = "carbon_daily"

# csv output dir
output_dir = f'results/{target_dataset}/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Results dir
OUT_DIR = "ttm_finetuned_models/"

DATASET_FREQ = 'B'

# global param setting
BSZ = 4
grad_acc = 4

def zeroshot_eval(dataset_name, batch_size, context_length, forecast_length):
    # Get data - zeroshot models are channel independent, and thus should not require conditional columns
    _, _, dset_test, tsp, cfg = load_dataset(
        dataset_name=dataset_name,
        context_length=context_length,
        forecast_length=forecast_length,
        fewshot_fraction=1.0,
        dataset_root_path=DATA_ROOT_PATH,
        conditional_columns=[]
    )

    # Load the most optimal model
    model_revision = create_revision_name(context_length, forecast_length)
    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        revision=model_revision,
        prediction_filter_length=forecast_length,
    )

    # zeroshot_trainer
    temp_dir = tempfile.mkdtemp()
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

    col_list = cfg['target_columns'] + cfg['conditional_columns']
    # reshaped = zeroshot_output.predictions[0].reshape(-1, len(col_list))
    reshaped = zeroshot_output.predictions[0][:, :, 0].flatten()

    # Extract timestamps and generate future dates
    timestamps = []
    true = []
    for item in dset_test:
        end_timestamp = pd.Timestamp(item['timestamp'])
        future_dates = pd.date_range(
            start=end_timestamp,
            periods=forecast_length + 1,
            freq=DATASET_FREQ
        )
        timestamps.extend(future_dates[1:])
        true.extend(item['future_values'][:forecast_length, 0].flatten().tolist())

    # Create DataFrame with predictions and dates
    to_export = pd.DataFrame({
        'date': timestamps,
        'true': true,
        col_list[0]: reshaped
    })
    to_export[col_list[0]] = tsp.inverse_scale_targets(to_export[[col_list[0]]])
    to_export['true'] = tsp.inverse_scale_targets(to_export[['true']].rename(columns={'true':tsp.target_columns[0]}))

    # dset_test should be the same size as reshaped
    to_export.to_csv(output_dir + f'TTMs_feat0_ctx{context_length}_pl{forecast_length}_zeroshot.csv')

    log_into_csv(to_export, dataset_name, 'zeroshot', model=f'TTM-{model_revision}',
                 ch_mix=True, seq_len=context_length, pred_len=forecast_length,
                 pred_filter_len=forecast_length, bsz=batch_size,
                 log_file_name = 'carbon', pred_col_name='Price')


def finetune_eval(
    dataset_name,
    batch_size,
    context_length,
    forecast_length,
    gradient_accumulation_steps=1,
    learning_rate=0.001,
    fewshot_percent=5,
    freeze_backbone=True,
    num_epochs=50,
    save_dir=OUT_DIR,
    conditional_columns=None,
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
        conditional_columns=conditional_columns,
    )

    ''' 
    Load the most optimal model configurations.
    
     `prediction_filter_length` is used in the model to shorten its prediction head output size. According to
     https://github.com/ibm-granite/granite-tsfm/issues/102, the loss will also be calculated based on this shortened
     prediction length.
    '''
    model_revision = create_revision_name(context_length, forecast_length)
    finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        revision=model_revision,
        prediction_filter_length=forecast_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
        prediction_channel_indices=tsp.prediction_channel_indices,
    )

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
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        steps_per_epoch=math.ceil(len(dset_train) / batch_size),
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

    torch.cuda.empty_cache()

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
            periods=forecast_length + 1,
            freq=DATASET_FREQ
        )
        timestamps.extend(future_dates[1:])
        true.extend(item['future_values'][:forecast_length, 0].flatten().tolist())

    # Create DataFrame with predictions and dates
    to_export = pd.DataFrame({
        'date': timestamps,
        'true': true,
        col_list[0]: reshaped
    })
    to_export[col_list[0]] = tsp.inverse_scale_targets(to_export[[col_list[0]]])
    to_export['true'] = tsp.inverse_scale_targets(to_export[['true']].rename(columns={'true':col_list[0]}))

    if fewshot_percent == 100:
        to_export.to_csv(output_dir + f'TTMs_feat{len(conditional_columns)}_ctx{context_length}_pl{forecast_length}_fullshot.csv')
    else:
        to_export.to_csv(output_dir + f'TTMs_feat{len(conditional_columns)}_ctx{context_length}_pl{forecast_length}_fewshot{fewshot_percent}.csv')

    log_into_csv(to_export, dataset_name, 'fullshot', model=f'TTM-{model_revision}',
                 ch_mix=True, seq_len=context_length, pred_len=forecast_length,
                 pred_filter_len=forecast_length, lr=learning_rate, bsz=batch_size,
                 log_file_name = 'carbon', pred_col_name='Price')


err_log = []
pred_lens = [1, 2, 3, 4, 5, 7, 14, 21, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 180, 365, 545]
feat_pct = [0, 0.3, 0.5, 0.75]
CTX = 512

for pct in feat_pct:
    # Load in selected features based on the spearman correlation analysis
    sel_features_df0 = pd.read_excel("datasets/carbon/res_daily/ranked_abs_features_daily.xlsx")
    sel_feature_len = int(pct * len(sel_features_df0))
    sel_features_df0.sort_values(by="Correlation", ascending=False, inplace=True)
    sel_feature_names = sel_features_df0["Factor"][0:sel_feature_len].tolist()
    sel_feature_names = [val for val in sel_feature_names if 'Historical Price' not in val]

    for pl in pred_lens:
        print(f'Running {pct}-feats and pl-{pl}...')

        try:
            if pct == 0:
                zeroshot_eval(
                    dataset_name=target_dataset,
                    batch_size=BSZ * grad_acc,
                    context_length=CTX,
                    forecast_length=pl,  # this is for data formatting purposes, e.g. in the TimeSeriesPreprocessor
                )

            finetune_eval(
                dataset_name=target_dataset,
                batch_size=BSZ,
                context_length=CTX,
                forecast_length=pl,  # this is for data formatting purposes, e.g. in the TimeSeriesPreprocessor
                gradient_accumulation_steps=grad_acc,
                fewshot_percent=100,
                conditional_columns=sel_feature_names,
            )
        except Exception as e:
            print(e.with_traceback())
            err_log.append(e)

        torch.cuda.empty_cache()

print(err_log)
