import os
import sys

PROJECT_PATH = os.sep + os.path.join(*__file__.split(os.sep)[:-2])
os.chdir(PROJECT_PATH)
sys.path.append(PROJECT_PATH)

import pandas as pd

from models.svd_model import get_model, eval
from utils.prepare_data import (
    create_custom_surprise_dataset,
    process_rating_table,
    split_train_test,
)


def evaluate(path_to_model, k=5, threshold=3.5, verbose=True):
    """Evaluate the model on several test groups and collect metrics"""

    model = get_model(path_to_model)

    path_to_test_data = os.path.join(PROJECT_PATH, "benchmark", "data")
    files = [
        os.path.join(path_to_test_data, f)
        for f in os.listdir(path_to_test_data)
        if os.path.isfile(os.path.join(path_to_test_data, f))
    ]
    dataframes = [process_rating_table(f) for f in files]
    datasets = [create_custom_surprise_dataset(df) for df in dataframes]
    tests = []
    for d in datasets:
        _, t = split_train_test(d, 1.0, shuffle=False, random_state=0)
        tests.append(t)
    results = []
    for i, dataset in enumerate(tests):
        rmse, precision, recall = eval(model, dataset, k, threshold)
        results.append((rmse, precision, recall))
        if verbose:
            print(f"{i+1} test group:")
            print(f"RMSE = {rmse}")
            print(f"precision@{k} = {precision}")
            print(f"recall@{k} = {recall}")
            print("########")
    return results
