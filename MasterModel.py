# -*- coding: utf-8 -*-
""" Master Model
import os
os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")
os.chdir("C://Users//danie//Documents//tcForecast")
"""

import numpy as np
import pandas as pd
import quanta as mx
import datetime as dt
import matplotlib.pyplot as plt


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


def separatFeatures(df, lag=5):
    """Separate into input and output."""
    # get length of dataframe (rows)
    n = len(df)

    # extract output data
    output_data = df.iloc[lag:]
    output_data.index = np.arange(len(output_data))

    # extract input data
    input_data = [df.iloc[i:(n - lag + i)].values for i in range(lag)]
    input_data = pd.DataFrame(np.concatenate(input_data, 1))

    return input_data, output_data


# Download prices
df = mx.data.getBanxicoSeries("usdmxn_fix")
df = numericDf(df)
df = cutDf(df, "01/01/2010")
rend = np.log(df["values"].iloc[1:].values/df["values"].iloc[:-1].values)
rend_df = pd.DataFrame({"rends": rend})
df_lags = lagMatrix(rend_df, lag=5)


# Feed with prices
input_data, output_data = separatFeatures(df)
input_data = input_data.iloc[1:]
output_data = output_data.iloc[1:]

# Add returns, NOTE: check index
input_data = np.concatenate([input_data, rend_df], axis=1)



error_data = test_data.apply(lambda x: x[0]-x[1],1).values

def generateKDE(datapoints,_plot=True):
    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel='gaussian',bandwidth=0.3).fit(
        datapoints.reshape(-1, 1))

    _min, _max = np.min(datapoints),np.max(datapoints)
    x_plot = np.arange(_min,_max+_max/2,(_max-_min)/10000)
    log_dens = kde.score_samples(x_plot.reshape(-1, 1))

    if _plot:
        plt.plot(x_plot,np.exp(log_dens))
        plt.title("Kernel Density Estimator: Errors")

    return pd.DataFrame({'x_data':x_plot,'density':np.exp(log_dens)}),kde


distribution,kde = generateKDE(error_data)
