# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
from yaml import safe_load
from sklearn.model_selection import train_test_split
from entities import SplittingParams


def get_config(path):
    with open(path, 'r') as stream:
        config = safe_load(stream)
    return config


def get_dataset(
        path: str, params: SplittingParams, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :rtype: object
    """
    df = pd.read_csv(path)
    target_col = df[target]
    df.drop(target, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df, target_col, test_size=params.test_size,
        random_state=params.random_state, shuffle=params.shuffle)
    return X_train, X_test, y_train, y_test


def get_eval_data(
        path: str,
) -> pd.DataFrame:

    return pd.read_csv(path)
