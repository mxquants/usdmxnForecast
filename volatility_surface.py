# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:01:37 2017
import os
os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")
os.chdir("C://Users//danie//Documents//tcForecast")
@author: danie
"""

# %% Change directory (windows)

import os
os.chdir('C://Users//danie//Documents//tcForecast')
# %% Imports

import quanta as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
# %%

mx.data.getAvailableBanxicoSeries()

# %%

temp = mx.data.getBanxicoSeries("tiie28")
temp.head()

# %%

temp = mx.data.getBanxicoSeries("usdmxn_fix")
temp.head()

# %%

def string2datetime(x):
    return dt.datetime.strptime(x,"%d/%m/%Y")

# %%
reference_index = temp['timestamp'].apply(lambda x: string2datetime(x)>dt.datetime.strptime("31/12/2013","%d/%m/%Y")).values
train = temp[reference_index == False]
test = temp[reference_index]
# %% Exportar a csv

train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)

# %%

def getReturns(x):
    return np.log(x[1:]/x[:-1])

returns = getReturns(temp["values"].apply(lambda x: np.float(x)).values)
df_returns = pd.DataFrame({"date":temp.timestamp.apply(string2datetime).values[1:],"returns":returns})
# %%



def estimateVolatility(reference_date,n_elem=21,alpha=0.5):

    def getVol(x):
        return np.std(x)*np.sqrt(252)

    def getPrediction(vector,alpha):
        mu = np.mean(vector)
        sigma = np.std(vector)
        return mu+alpha*sigma

    std_vector = []
    for i in range(20):
        upper_limit = (reference_date - dt.timedelta(days=1)) if (i==0) else (upper_limit - dt.timedelta(days=1))
        lower_limit = upper_limit - dt.timedelta(days=n_elem)
        _index = [i and j for i,j in zip((df_returns.date > lower_limit).values,(df_returns.date < upper_limit))]
        std_vector.append(getVol(df_returns[_index].returns.values))

    return getPrediction(std_vector,alpha)

def realVolatility(reference_date,n_elem=21):

    def getVol(x):
        return np.std(x)*np.sqrt(252)

    upper_limit = reference_date + dt.timedelta(days=n_elem)
    _index = [i and j for i,j in zip((df_returns.date > reference_date).values,(df_returns.date < upper_limit))]

    return getVol(df_returns[_index].returns.values)
# %% volatility for next day 1m
reference_date = dt.datetime.strptime("01/01/2014","%d/%m/%Y")
vol1m = estimateVolatility(reference_date,n_elem=21)
vol2m = estimateVolatility(reference_date,n_elem=42)
vol3m = estimateVolatility(reference_date,n_elem=63)
# %%
def getVolatilityEstimation(reference_date,n,vol_type=["1m","2m","3m"],alpha=0.5):
    n_elem_dict = {"1m":21,"2m":42,"3m":63,"6m":182,"1y":252}
    actual_date = reference_date - dt.timedelta(days=1)
    vol_est = {}
    dates = []
    for i in range(n):
        actual_date = actual_date + dt.timedelta(days=1)
        dates.append(actual_date)
        for j in vol_type:
            if j not in vol_est:
                vol_est[j] = []
            vol_est[j].append(estimateVolatility(actual_date,n_elem_dict[j],alpha))
    return pd.DataFrame(vol_est,index=dates)

def getRealVolatility(reference_date,n,vol_type=["1m","2m","3m"]):
    n_elem_dict = {"1m":21,"2m":42,"3m":63,"6m":182,"1y":252}
    actual_date = reference_date - dt.timedelta(days=1)
    vol_est = {}
    dates = []
    for i in range(n):
        actual_date = actual_date + dt.timedelta(days=1)
        dates.append(actual_date)
        for j in vol_type:
            if j not in vol_est:
                vol_est[j] = []
            vol_est[j].append(realVolatility(actual_date,n_elem_dict[j]))
    return pd.DataFrame(vol_est,index=dates)

# %%

# reference date
reference_date = dt.datetime.strptime("01/01/2014","%d/%m/%Y")

# number of dates
n = (dt.datetime.today() - reference_date).days

# desired volatilities
vol_type=["1m","2m","3m"]

# estimate vols
estimates = getVolatilityEstimation(reference_date,n-63)

# %%
real = getRealVolatility(reference_date,n-63)

# %%

error = (real - estimates).apply(lambda x: np.sqrt(x**2))
error.plot(kind="kde")
# %%

error.mean()
# %%
100*error.mean()/real.mean()

# %% 2017-03-29
new_ref_date = dt.datetime.strptime("29/03/2017","%d/%m/%Y")
new_df = getVolatilityEstimation(new_ref_date,63)

# %%
