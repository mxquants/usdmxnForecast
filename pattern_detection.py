# -*- coding: utf-8 -*-
""" Pattern detection in financial time-series data.
import os
os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")
os.chdir("C://Users//danie//Documents//tcForecast")
@author: Rodrigo Hernández-Mota
"""
import numpy as np
import pandas as pd
import quanta as mx
import datetime as dt
import matplotlib.pyplot as plt
from metallic_blue_lizard.neural_net import competitive_neurons


def lagMatrix(df, lag=5):
    """Return lag matrix for a given time series (dataframe)."""
    n = len(df)
    input_data = [df.iloc[i:(n - lag + i + 1)].values for i in range(lag)]
    input_data = pd.DataFrame(np.concatenate(input_data, 1))
    return input_data


def string2datetime(x):
    """."""
    return dt.datetime.strptime(x, "%d/%m/%Y")


def numericDf(df):
    """."""
    df["timestamp"] = df["timestamp"].apply(string2datetime).values
    df["values"] = df["values"].apply(np.float).values
    return df


# Test
df = mx.data.getBanxicoSeries("usdmxn_fix")
df = numericDf(df)
df_lags = lagMatrix(df[["values"]], lag=5)

# Detect patterns
cn = competitive_neurons(neurons=15, x_data=df_lags)
cn.train(max_iter=5000, eta=0.2)
cn.evaluate()

neurons = np.unique(cn.y)
print('Neurons that found a cluster: {}'.format(neurons))
cn.cost
for i in cn.w.columns:
    temp = cn.w[i]
    plt.plot(temp)
plt.show()
