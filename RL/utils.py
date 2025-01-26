import numpy as np

def time_split(df_dict, train_ratio=0.7, val_ratio=0.2):
    train_dict = {}
    val_dict = {}
    test_dict = {}

    for key, df in df_dict.items():
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        df_train = df.iloc[:train_end].reset_index(drop=True)
        df_val = df.iloc[train_end:val_end].reset_index(drop=True)
        df_test = df.iloc[val_end:].reset_index(drop=True)

        train_dict[key] = df_train
        val_dict[key] = df_val
        test_dict[key] = df_test

    return train_dict, val_dict, test_dict


def prepare_data(data_dict) -> dict:

    for i in data_dict:
        data_dict[i] = data_dict[i].sort_values(by=['begin'], ascending=True).reset_index(drop=True)

    return data_dict
