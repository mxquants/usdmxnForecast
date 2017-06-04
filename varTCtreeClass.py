# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:50:47 2017

@author: danie
"""

#%%
import quanta as mx 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt 
from sklearn.ensemble import RandomForestClassifier
#%%

mx.data.getAvailableBanxicoSeries()

#%%

_fix = mx.data.getBanxicoSeries("usdmxn_fix")
_close = mx.data.getBanxicoSeries("usdmxn_close_long")
_open = mx.data.getBanxicoSeries("usdmxn_open_long")
_high = mx.data.getBanxicoSeries("usdmxn_max")
_low = mx.data.getBanxicoSeries("usdmxn_min")

#%%

df_dict = {"fix":_fix.copy(),"close":_close.copy(),
           "open":_open.copy(),"high":_high.copy(),
           "low":_low.copy()
           }

#%%

def string2datetime(x):
    """."""
    return dt.datetime.strptime(x, "%d/%m/%Y")


def numericDf(df):
    """."""
    df["timestamp"] = df["timestamp"].apply(string2datetime).values
    df["values"] = df["values"].apply(np.float).values
    return df
    

def variablesTC(df_dict):
    general_df = pd.DataFrame([])
               
    for k in df_dict:
        df = numericDf(df_dict[k])
        df.index =  df["timestamp"].values
        del  df["timestamp"]
        df.columns = [k]
        general_df = pd.concat([general_df,df], axis=1)
          
    return general_df
    
#%%
df = variablesTC(df_dict)
df = df.dropna()

# %% Add lags

close_values = df.close.values
high_values = df.high.values
low_values = df.low.values

df = df.iloc[1:]
df["close_1"] = close_values[:-1]
df["high_1"] = high_values[:-1]
df["low_1"] = low_values[:-1]
#%%
functions = {"close/high":lambda x: np.log(x["close"]/x["high"]),
             "close/low":lambda x: np.log(x["close"]/x["low"]),
             "close/open":lambda x: np.log(x["close"]/x["open"]),
             "high/low":lambda x: np.log(x["high"]/x["low"]),
             "high/high_1":lambda x: np.log(x["high"]/x["high_1"]),
             "low/low_1":lambda x: np.log(x["low"]/x["low_1"]),
             "close/close_1":lambda x: np.log(x["close"]/x["close_1"])}

relation_df = {}
for k in functions:
    relation_df[k] = df.apply(functions[k],1).values

relation_df = pd.DataFrame(relation_df)
#%%

RandomForestClassifier()

