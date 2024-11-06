import pandas as pd
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path, sep=',', low_memory=False)

    data.replace('?', np.nan, inplace=True)

    return data


def clean_data(data):
    for column in data.columns:
        if column not in ['Date', 'Time']:
            data[column] = pd.to_numeric(data[column], errors='coerce')

    data = data.dropna(how='all', subset=data.columns[1:])

    return data


def fill_missing_values(data):
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if not data[column].isnull().all():
            data[column] = data[column].fillna(data[column].mean())

    return data
