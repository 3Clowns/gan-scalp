import sys
import apimoex
import pandas as pd
import requests
import datetime
import os
import pickle

monday = datetime.datetime(year=2024, month=12, day=16)
tuesday = datetime.datetime(year=2024, month=12, day=17)
wednesday = datetime.datetime(year=2024, month=12, day=18)
thursday = datetime.datetime(year=2024, month=12, day=19)
friday = datetime.datetime(year=2024, month=12, day=20)
week = [monday, tuesday, wednesday, thursday, friday]

# global_tickers = ["SBER", "VTBR", "ROSN", "GOLD", "LKOH"]
global_tickers = ["VTBR", "LKOH"]
extra_tickers = ["MOEXFN", "MOEXCN", "TRUR", ]


def create_dataset(tickers=global_tickers, num_weeks=36):
    print("Fetching candles data started...")
    seven = datetime.timedelta(days=7)
    data = {}

    with requests.Session() as session:
        for i in tickers:
            for weeks in range(num_weeks):
                dt = friday - weeks * seven
                candles = apimoex.get_market_candles(session, i, 1, str(dt - datetime.timedelta(days=5))[:-9],
                                                     str(dt)[:-9])
                dc = pd.DataFrame(candles)
                if i not in data:
                    data[i] = [dc]
                else:
                    data[i].append(dc)

    for key in data:
        data[key] = pd.concat(data[key], ignore_index=True)

    return data


def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def save_dataset(data, filename):
    drive_path = "./"
    save_data_as_pickle(data, os.path.join(drive_path, filename))
    print(f"Successfully saved {filename}")


def load_dataset(filename):
    if os.path.exists(filename):
        data = load_data_from_pickle(os.path.join("./", filename))
    else:
        assert False, f"Файл {filename} не найден."

    print(f"Downloaded tickers: {data.keys()}")
    return data
