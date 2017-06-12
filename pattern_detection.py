#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern detection in financial time-series data.

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


def cutDf(df, refence_date):
    """."""
    date = string2datetime(refence_date)
    reference_index = df["timestamp"] > date
    return df[reference_index]


# Test
df = mx.data.getBanxicoSeries("usdmxn_fix")
df = numericDf(df)
df = cutDf(df, "01/01/2014")
rend = np.log(df["values"].iloc[1:].values/df["values"].iloc[:-1].values)
rend_df = pd.DataFrame({"rends": rend})
df_lags = lagMatrix(rend_df, lag=5)

# k-means
dataset = mx.dataHandler.Dataset(input_data=df_lags, output_data=df_lags,
                                 normalize=None)
kmeans = mx.unsupervised.Kmeans(n_clusters=8)
kmeans.train(dataset, epochs=5000)
kmeans.c


# Detect patterns
cn = competitive_neurons(neurons=10, x_data=df_lags)
cn.train(max_iter=5000, eta=0.005)
cn.evaluate()

neurons = np.unique(cn.y)
print('Neurons that found a cluster: {}'.format(neurons))
cn.cost
for i in neurons:
    temp = cn.w[i]
    plt.plot(temp)
plt.show()
for i in kmeans.c:
    temp = kmeans.c[i]
    plt.plot(temp)
plt.show()
cn.w[neurons].to_pickle("competitive_neurons.pkl")
pd.DataFrame(kmeans.c).to_pickle("kmeans.pkl")
kmeans.nearest
