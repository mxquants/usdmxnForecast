# -*- coding: utf-8 -*-
"""
Pattern detection in financial time-series data.
@author: Rodrigo Hernández-Mota
"""

import os
os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")
import numpy as np
import pandas as pd
import quanta as mx
import matplotlib.pyplot as plt


def lagMatrix(df, lag=5):
    """Return lag matrix for a given time series (dataframe)."""
    n = len(df)
    input_data = [df.iloc[i:(n - lag + i + 1)].values for i in range(lag)]
    input_data = pd.DataFrame(np.concatenate(input_data, 1))
    return input_data


# Test
df = mx.data.getBanxicoSeries("usdmxn_fix")
lags = lagMatrix(df)
