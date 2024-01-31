import torch.utils.data as data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from helpers import *


def make_loaders(samples: int = n_samples):
    df_train = pd.read_json(data_dir + 'train.json', lines=True)


    # df_train = filter_signal_noise(df_train)
    df_train = preprocess(df_train, feature_cols)

    if samples:
        df_train = df_train.iloc[:samples]

    print(df_train['signal_to_noise'].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(to_np_array(df_train, feature_cols), to_np_array(df_train, target_cols), test_size=.1, stratify=df_train["signal_to_noise"])


    x_train, x_test, y_train, y_test = convert_transpose(x_train), convert_transpose(x_test), convert_transpose(y_train), convert_transpose(y_test)

    # print(x_train.shape)
    # print(x_test.shape)

    dataset_train = data.TensorDataset(x_train, y_train)
    train_loader = data.DataLoader(dataset_train, batch_size, shuffle = True)


    dataset_test = data.TensorDataset(x_test, y_test)
    # test_loader = data.DataLoader(dataset_test, batch_size, shuffle = True)

    return train_loader, dataset_test

# make_loaders(None)