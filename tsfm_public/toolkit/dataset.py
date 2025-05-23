# Copyright contributors to the TSFM project
#
"""Tools for building torch datasets"""

import copy
from itertools import starmap
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .util import join_list_without_repeat


def pad_sequence_to_length(sequence: torch.Tensor, target_length: int, pad_value: float = 0.0,
                           left_pad: bool = True) -> torch.Tensor:
    """
    Pads a sequence tensor to a target length, with option for left or right padding.

    Args:
        sequence: Input tensor of shape (seq_len, features)
        target_length: Desired length of the sequence
        pad_value: Value to use for padding
        left_pad: If True, add padding at the beginning of sequence; if False, add at the end

    Returns:
        Padded tensor of shape (target_length, features)
    """
    current_length = sequence.shape[0]

    if current_length > target_length:
        # If sequence is longer, truncate it from the left to keep the most recent values
        return sequence[-target_length:]

    if current_length < target_length:
        # If sequence is shorter, pad it
        padding_length = target_length - current_length

        # Create padding tensor
        if len(sequence.shape) == 2:
            # For 2D tensors (sequence, features)
            num_features = sequence.shape[1]
            padding = torch.full((padding_length, num_features), pad_value, dtype=sequence.dtype)
        else:
            # For 1D tensors
            padding = torch.full((padding_length,), pad_value, dtype=sequence.dtype)

        # Concatenate the padding with the original sequence
        if left_pad:
            return torch.cat([padding, sequence], dim=0)
        else:
            return torch.cat([sequence, padding], dim=0)

    # If lengths match, return original sequence
    return sequence


class BaseDFDataset(torch.utils.data.Dataset):
    """Base dataset for time series models built upon a pandas dataframe

    Args:
        data_df (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        group_id (Optional[Union[List[int], List[str]]], optional): _description_. Defaults to None.
        x_cols (list, optional): Columns to treat as inputs. If an empty list ([]) all the columns in the data_df are taken, except the timestamp column. Defaults to [].
        y_cols (list, optional): Columns to treat as outputs. Defaults to [].
        drop_cols (list, optional): List of columns that are dropped to form the X matrix (input). Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        prediction_length (int, optional): Length of prediction (future values). Defaults to 0.
        zero_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        group_id: Optional[Union[List[int], List[str]]] = None,
        x_cols: list = [],
        y_cols: list = [],
        drop_cols: list = [],
        context_length: int = 1,
        prediction_length: int = 0,
        zero_padding: bool = True,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
    ):
        super().__init__()
        if not isinstance(x_cols, list):
            x_cols = [x_cols]
        if not isinstance(y_cols, list):
            y_cols = [y_cols]

        if len(x_cols) > 0:
            assert is_cols_in_df(data_df, x_cols), f"one or more {x_cols} is not in the list of data_df columns"

        if len(y_cols) > 0:
            assert is_cols_in_df(data_df, y_cols), f"one or more {y_cols} is not in the list of data_df columns"

        if timestamp_column:
            assert timestamp_column in list(
                data_df.columns
            ), f"{timestamp_column} is not in the list of data_df columns"
            assert timestamp_column not in x_cols, f"{timestamp_column} should not be in the list of x_cols"

        self.data_df = data_df
        self.datetime_col = timestamp_column
        self.id_columns = id_columns
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.drop_cols = drop_cols
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.zero_padding = zero_padding
        self.fill_value = fill_value
        self.timestamps = None
        self.group_id = group_id
        self.stride = stride

        # sort the data by datetime
        if timestamp_column in list(data_df.columns):
            if not isinstance(data_df[timestamp_column].iloc[0], pd.Timestamp):
                data_df[timestamp_column] = pd.to_datetime(data_df[timestamp_column])
            data_df = data_df.sort_values(timestamp_column, ignore_index=True)

        # pad zero to the data_df if the len is shorter than seq_len+pred_len
        if zero_padding:
            data_df = self.pad_zero(data_df)

        if timestamp_column in list(data_df.columns):
            self.timestamps = data_df[timestamp_column].to_list()  # .values coerces timestamps
        # get the input data
        if len(x_cols) > 0:
            self.X = data_df[x_cols]
        else:
            drop_cols = self.drop_cols + y_cols
            if timestamp_column:
                drop_cols += [timestamp_column]
            self.X = data_df.drop(drop_cols, axis=1) if len(drop_cols) > 0 else data_df
            self.x_cols = list(self.X.columns)

        # get target data
        if len(y_cols) > 0:
            self.y = data_df[y_cols]
        else:
            self.y = None

        # get number of X variables
        self.n_vars = self.X.shape[1]
        # get number of target
        self.n_targets = len(y_cols) if len(y_cols) > 0 else 0

    def pad_zero(self, data_df):
        # return zero_padding_to_df(data_df, self.seq_len + self.pred_len)
        return ts_padding(
            data_df,
            timestamp_column=self.datetime_col,
            id_columns=self.id_columns,
            context_length=self.context_length + self.prediction_length,
        )

    def __len__(self):
        return (len(self.X) - self.context_length - self.prediction_length) // self.stride + 1

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


class BaseConcatDFDataset(torch.utils.data.ConcatDataset):
    """A dataset consisting of a concatenation of other datasets, based on torch ConcatDataset.

    Args:
        data_df (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        prediction_length (int, optional): Length of prediction (future values). Defaults to 0.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        cls (_type_, optional): The dataset class used to create the underlying datasets. Defaults to BaseDFDataset.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        fill_value: Union[float, int] = 0.0,
        cls=BaseDFDataset,
        stride: int = 1,
        **kwargs,
    ):
        if len(id_columns) > 0:
            assert is_cols_in_df(data_df, id_columns), f"{id_columns} is not in the data_df columns"

        self.timestamp_column = timestamp_column
        self.id_columns = id_columns
        # self.x_cols = x_cols
        # self.y_cols = y_cols
        self.context_length = context_length
        self.num_workers = num_workers
        self.cls = cls
        self.prediction_length = prediction_length
        self.stride = stride
        self.extra_kwargs = kwargs
        self.fill_value = fill_value
        self.cls = cls

        # create groupby object
        if len(id_columns) == 1:
            self.group_df = data_df.groupby(by=self.id_columns[0])
        elif len(id_columns) > 1:
            self.group_df = data_df.groupby(by=self.id_columns)
        else:
            data_df["group"] = 0  # create a artificial group
            self.group_df = data_df.groupby(by="group")

        # add group_ids to the drop_cols
        self.drop_cols = id_columns if len(id_columns) > 0 else ["group"]

        self.group_names = list(self.group_df.groups.keys())
        datasets = self.concat_dataset()
        super().__init__(datasets)
        self.n_vars = self.datasets[0].n_vars
        self.n_targets = self.datasets[0].n_targets

    def concat_dataset(self):
        """Create a list of Datasets

        Returns:
            List of datasets
        """
        group_df = self.group_df
        # print(f'group_df: {group_df}')
        # pool = mp.Pool(self.num_workers)
        # pool.starmap(
        list_dset = starmap(
            get_group_data,
            [
                (
                    self.cls,
                    group,
                    group_id,
                    self.id_columns,
                    self.timestamp_column,
                    self.context_length,
                    self.prediction_length,
                    self.drop_cols,
                    self.stride,
                    self.fill_value,
                    self.extra_kwargs,
                )
                for group_id, group in group_df
            ],
        )

        # pool.close()
        # del group_df
        return list_dset


def get_group_data(
    cls,
    group,
    group_id,
    id_columns: List[str] = [],
    timestamp_column: Optional[str] = None,
    context_length: int = 1,
    prediction_length: int = 1,
    drop_cols: Optional[List[str]] = None,
    stride: int = 1,
    fill_value: Union[float, int] = 0.0,
    extra_kwargs: Dict[str, Any] = {},
):
    return cls(
        data_df=group,
        group_id=group_id if isinstance(group_id, tuple) else (group_id,),
        id_columns=id_columns,
        timestamp_column=timestamp_column,
        context_length=context_length,
        prediction_length=prediction_length,
        drop_cols=drop_cols,
        stride=stride,
        fill_value=fill_value,
        **extra_kwargs,
    )


class PretrainDFDataset(BaseConcatDFDataset):
    """A dataset used for masked pre-training.

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.


    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length
        past_observed_mask: tensor indicating which values are observed in the past values tensor
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
    ):
        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=0,
            cls=self.BasePretrainDFDataset,
            target_columns=target_columns,
            stride=stride,
            fill_value=fill_value,
        )
        self.n_inp = 1

    class BasePretrainDFDataset(BaseDFDataset):
        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 0,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
        ):
            self.target_columns = target_columns

            x_cols = target_columns
            y_cols = []

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
                stride=stride,
                fill_value=fill_value,
            )

        def __getitem__(self, index):
            time_id = index * self.stride
            seq_x = self.X[time_id : time_id + self.context_length].values
            ret = {
                "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]
            if self.group_id:
                ret["id"] = self.group_id

            return ret


class ForecastDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        observable_columns (List[str], optional): List of column names which identify the observable channels in the input.
            Observable channels are channels which we have knowledge about in the past and future. For example, weather
            conditions such as temperature or precipitation may be known or estimated in the future, but cannot be
            changed. Defaults to [].
        control_columns (List[str], optional): List of column names which identify the control channels in the input. Control
            channels are similar to observable channels, except that future values may be controlled. For example, discount
            percentage of a particular product is known and controllable in the future. Defaults to [].
        conditional_columns (List[str], optional): List of column names which identify the conditional channels in the input.
            Conditional channels are channels which we know in the past, but do not know in the future. Defaults to [].
        static_categorical_columns (List[str], optional): List of column names which identify categorical-valued channels in the
            input which are fixed over time. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        prediction_length (int, optional): Length of the future forecast. Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        frequency_token (Optional[int], optional): An integer representing the frequency of the data. Please see for an example of
            frequency token mappings. Defaults to None.
        autoregressive_modeling (bool, optional): (Experimental) If False, any target values in the context window are masked and
            replaced by 0. If True, the context window contains all the historical target information. Defaults to True.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.

    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x number of features)
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x number of features)
        future_values: tensor of future values of the target columns of length equal to prediction length (prediction_length x number of features)
        future_observed_mask: tensor indicating which values are observed in the future values tensor (prediction_length x number of features)
        freq_token: tensor containing the frequency token (scalar)
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment

        where number of features is the total number of columns specified in target_columns, observable_columns, control_columns,
            conditional_columns
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        observable_columns: List[str] = [],
        control_columns: List[str] = [],
        conditional_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        frequency_token: Optional[int] = None,
        autoregressive_modeling: bool = True,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        pad_past_length: Optional[int] = None,  # New parameter
        pad_future_length: Optional[int] = None,  # New parameter
    ):
        # output_columns_tmp = input_columns if output_columns == [] else output_columns

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=prediction_length,
            fill_value=fill_value,
            cls=self.BaseForecastDFDataset,
            stride=stride,
            # extra_args
            target_columns=target_columns,
            observable_columns=observable_columns,
            control_columns=control_columns,
            conditional_columns=conditional_columns,
            static_categorical_columns=static_categorical_columns,
            frequency_token=frequency_token,
            autoregressive_modeling=autoregressive_modeling,
            pad_past_length=pad_past_length,  # Pass through to BaseForecastDFDataset
            pad_future_length=pad_future_length,  # Pass through to BaseForecastDFDataset
        )
        self.n_inp = 2
        # for forecasting, the number of targets is the same as number of X variables
        self.n_targets = self.n_vars


    class BaseForecastDFDataset(BaseDFDataset):
        """
        X_{t+1,..., t+p} = f(X_{:t})
        """

        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 1,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
            observable_columns: List[str] = [],
            control_columns: List[str] = [],
            conditional_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
            frequency_token: Optional[int] = None,
            autoregressive_modeling: bool = True,
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
            pad_past_length: Optional[int] = None,  # New parameter
            pad_future_length: Optional[int] = None,  # New parameter
        ):
            self.frequency_token = frequency_token
            self.target_columns = target_columns
            self.observable_columns = observable_columns
            self.control_columns = control_columns
            self.conditional_columns = conditional_columns
            self.static_categorical_columns = static_categorical_columns
            self.autoregressive_modeling = autoregressive_modeling
            self.pad_past_length = pad_past_length
            self.pad_future_length = pad_future_length

            x_cols = join_list_without_repeat(
                target_columns,
                observable_columns,
                control_columns,
                conditional_columns,
            )
            y_cols = copy.copy(x_cols)

            # check non-autoregressive case
            if len(target_columns) == len(x_cols) and not self.autoregressive_modeling:
                raise ValueError(
                    "Non-autoregressive modeling was chosen, but there are no input columns for prediction."
                )

            # masking for conditional values which are not observed during future period
            self.y_mask_conditional = np.array([(c in conditional_columns) for c in y_cols])

            # create a mask of x which masks targets
            self.x_mask_targets = np.array([(c in target_columns) for c in x_cols])

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
                stride=stride,
                fill_value=fill_value,
            )

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols

            time_id = index * self.stride

            seq_x = self.X[time_id : time_id + self.context_length].values
            if not self.autoregressive_modeling:
                seq_x[:, self.x_mask_targets] = 0

            # seq_y: batch_size x pred_len x num_x_cols
            seq_y = self.y[
                time_id + self.context_length : time_id + self.context_length + self.prediction_length
            ].values

            seq_y[:, self.y_mask_conditional] = 0

            # New code - applying padding to data that is shorter than TTM requirements
            # Convert to torch tensors first
            past_values = np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value))
            future_values = np_to_torch(np.nan_to_num(seq_y, nan=self.fill_value))
            past_observed_mask = np_to_torch(~np.isnan(seq_x))
            future_observed_mask = np_to_torch(~np.isnan(seq_y))

            # Apply padding if specified
            if self.pad_past_length is not None:
                past_values = pad_sequence_to_length(past_values, self.pad_past_length, self.fill_value)
                past_observed_mask = pad_sequence_to_length(past_observed_mask, self.pad_past_length, False)

            if self.pad_future_length is not None:
                future_values = pad_sequence_to_length(future_values, self.pad_future_length, self.fill_value, left_pad=False)
                future_observed_mask = pad_sequence_to_length(future_observed_mask, self.pad_future_length, False, left_pad=False)

            ret = {
                "past_values": past_values,
                "future_values": future_values,
                "past_observed_mask": past_observed_mask,
                "future_observed_mask": future_observed_mask,
            }

            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.frequency_token is not None:
                ret["freq_token"] = torch.tensor(self.frequency_token, dtype=torch.int)

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret

        def __len__(self):
            return (len(self.X) - self.context_length - self.prediction_length) // self.stride + 1


class RegressionDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        input_columns (List[str], optional): List of columns to use as inputs to the regression
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.

    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x len(input_columns))
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x len(input_columns))
        target_values: tensor of future values of the target columns of length equal to prediction length (prediction_length x len(target_columns))
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        input_columns: List[str] = [],
        target_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
    ):
        # self.y_cols = y_cols

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=0,
            cls=self.BaseRegressionDFDataset,
            input_columns=input_columns,
            target_columns=target_columns,
            static_categorical_columns=static_categorical_columns,
            stride=stride,
            fill_value=fill_value,
        )

        self.n_inp = 2

    class BaseRegressionDFDataset(BaseDFDataset):
        """
        y_{t} = f(X_{:t})
        """

        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 0,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
            input_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
        ):
            self.target_columns = target_columns
            self.input_columns = input_columns
            self.static_categorical_columns = static_categorical_columns

            x_cols = input_columns
            y_cols = target_columns

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
                stride=stride,
                fill_value=fill_value,
            )

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols

            time_id = index * self.stride
            seq_x = self.X[time_id : time_id + self.context_length].values
            seq_y = self.y[time_id + self.context_length - 1 : time_id + self.context_length].values.ravel()
            # return _torch(seq_x, seq_y)

            ret = {
                "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
                "target_values": np_to_torch(np.nan_to_num(seq_y, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret


class ClassificationDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        input_columns (List[str], optional): List of columns to use as inputs to the regression
        label_column (str, optional): List of column names which identify the label of the time series. Defaults to "label".
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.

    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x len(input_columns))
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x len(input_columns))
        target_values: tensor containing the label (scalar)
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        input_columns: List[str] = [],
        label_column: str = "label",
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
    ):
        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=0,
            cls=self.BaseClassificationDFDataset,
            input_columns=input_columns,
            label_column=label_column,
            static_categorical_columns=static_categorical_columns,
            stride=stride,
            fill_value=fill_value,
        )

        self.n_inp = 2

    class BaseClassificationDFDataset(BaseDFDataset):
        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 0,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            label_column: str = "label",
            input_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
        ):
            self.label_column = label_column
            self.input_columns = input_columns
            self.static_categorical_columns = static_categorical_columns

            x_cols = input_columns
            y_cols = label_column

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
                stride=stride,
                fill_value=fill_value,
            )

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols

            time_id = index * self.stride
            seq_x = self.X[time_id : time_id + self.context_length].values
            # seq_y = self.y[time_id + self.context_length - 1 : time_id + self.context_length].values.ravel()
            seq_y = self.y.iloc[time_id + self.context_length - 1].values[0]

            ret = {
                "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
                "target_values": torch.tensor(np.nan_to_num(seq_y, nan=self.fill_value), dtype=torch.int64),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret


def np_to_torch(data: np.array, float_type=np.float32):
    if data.dtype == "float":
        return torch.from_numpy(data.astype(float_type))
    elif data.dtype == "int":
        return torch.from_numpy(data)
    elif data.dtype == "bool":
        return torch.from_numpy(data)
    return torch.from_numpy(data)


def _torch(*nps):
    return tuple(np_to_torch(x) for x in nps)


def zero_padding_to_df(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """
    check if df has length > seq_len.
    If not, then fill in zero
    Args:
        df (_type_): data frame
        seq_len (int): sequence length
    Returns:
        data frame
    """
    if len(df) >= seq_len:
        return df
    fill_len = seq_len - len(df) + 1
    # add zeros dataframe
    zeros_df = pd.DataFrame(np.zeros([fill_len, df.shape[1]]), columns=df.columns)
    # combine the data
    new_df = pd.concat([zeros_df, df])
    return new_df


def ts_padding(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    timestamp_column: Optional[str] = None,
    context_length: int = 1,
) -> pd.DataFrame:
    """
    Pad a dataframe, which is aware of time series conventions.

    Check if df has length >= context_length.
    If not, then fill (prepending) while preserving types and properly handling IDs and dates/timestamps. When
    prepending dates, the sampling interval will be estimated, to create proper preceeding dates.

    The assumption is the provided data contains only one id across the provided ID columns, the value will be
    replicated in the prepended rows.

    Args:
        df (_type_): data frame
        id_columns: List of strings representing columns containing ID information.
        timestamp_column: str for column name containing timestamps.
        context_length (int): required length

    Returns:
        Padded data frame
    """
    l = len(df)
    if l >= context_length:
        return df
    fill_length = context_length - l  # why did we previously have + 1 here?

    # create dataframe
    pad_df = pd.DataFrame(np.zeros([fill_length, df.shape[1]]), columns=df.columns)

    for c in df.columns:
        if (id_columns and c in id_columns) or (c == timestamp_column):
            continue
        pad_df[c] = pad_df[c].astype(df.dtypes[c], copy=False)

    if timestamp_column:
        if (df[timestamp_column].dtype.type == np.datetime64) or (df[timestamp_column].dtype == int):
            last_timestamp = df.iloc[0][timestamp_column]
            period = df.iloc[1][timestamp_column] - df.iloc[0][timestamp_column]
            prepended_timestamps = [last_timestamp + offset * period for offset in range(-fill_length, 0)]
            pad_df[timestamp_column] = prepended_timestamps
        else:
            pad_df[timestamp_column] = None
        # Ensure same type
        pad_df[timestamp_column] = pad_df[timestamp_column].astype(df[timestamp_column].dtype)

    if id_columns:
        id_values = df.iloc[0][id_columns].to_list()
        for id_column_name, id_column_value in zip(id_columns, id_values):
            pad_df[id_column_name] = id_column_value

    # combine the data
    new_df = pd.concat([pad_df, df])
    return new_df


def is_cols_in_df(df: pd.DataFrame, cols: List[str]) -> bool:
    """
    Args:
        df:
        cols:

    Returns:
        bool
    """
    for col in cols:
        if col not in list(df.columns):
            return False
    return True


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8],
            "B": [4, 5, 6, 7, 8, 9, 10, 11],
            "C": [7, 8, 9, 10, 11, 12, 13, 14],
            "g1": [0, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    print(df)

    d6 = PretrainDFDataset(data_df=df, x_cols=["A", "B"], group_ids=["g1"], seq_len=2)
    print(f"d6: {d6}")

    d7 = ForecastDFDataset(data_df=df, x_cols=["A", "B"], group_ids=["g1"], seq_len=2, pred_len=2)
