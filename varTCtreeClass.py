# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:50:47 2017
import os
os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")
os.chdir("C://Users//danie//Documents//tcForecast")
@author: danie
"""
import quanta as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve
from metallic_blue_lizard.neural_net import simple_logistic


def string2datetime(x):
    """."""
    return dt.datetime.strptime(x, "%d/%m/%Y")


def numericDf(df):
    """."""
    df["timestamp"] = df["timestamp"].apply(string2datetime).values
    df["values"] = df["values"].apply(np.float).values
    return df


def variablesTC(df_dict):
    """."""
    general_df = pd.DataFrame([])

    for k in df_dict:
        df = df_dict[k]  # numericDf(df_dict[k])
        df.index = df["timestamp"].values
        del df["timestamp"]
        df.columns = [k]
        general_df = pd.concat([general_df, df], axis=1)

    return general_df


def cutDf(df, refence_date):
    """."""
    date = string2datetime(refence_date)
    reference_index = df["timestamp"] > date
    return df[reference_index]


_fix = mx.data.getBanxicoSeries("usdmxn_fix")
_close = mx.data.getBanxicoSeries("usdmxn_close_long")
_open = mx.data.getBanxicoSeries("usdmxn_open_long")
_high = mx.data.getBanxicoSeries("usdmxn_max")
_low = mx.data.getBanxicoSeries("usdmxn_min")

reference_date = "01/01/2008"
df_dict = {"fix": cutDf(numericDf(_fix.copy()), reference_date),
           "close": cutDf(numericDf(_close.copy()), reference_date),
           "open": cutDf(numericDf(_open.copy()), reference_date),
           "high": cutDf(numericDf(_high.copy()), reference_date),
           "low": cutDf(numericDf(_low.copy()), reference_date)
           }

df = variablesTC(df_dict).copy()
df = df.dropna()

close_values = df.close.values
high_values = df.high.values
low_values = df.low.values

df = df.iloc[1:]
df["close_1"] = close_values[:-1]
df["high_1"] = high_values[:-1]
df["low_1"] = low_values[:-1]

functions = {"close/high": lambda x: np.log(x["close"]/x["high"]),
             "close/low": lambda x: np.log(x["close"]/x["low"]),
             "close/open": lambda x: np.log(x["close"]/x["open"]),
             "high/low": lambda x: np.log(x["high"]/x["low"]),
             # "high/high_1": lambda x: np.log(x["high"]/x["high_1"]),
             "low/low_1": lambda x: np.log(x["low"]/x["low_1"]),
             # "close/close_1": lambda x: np.log(x["close"]/x["close_1"])
             }

relation_df = {}
for k in functions:
    relation_df[k] = df.apply(functions[k], 1).values

x_data_reg = pd.DataFrame(relation_df)[:-1]
x_data_class = x_data_reg.copy()
x_data_class.corr()

actual = df['fix'][1:]
anterior = df['fix'][:-1]
salida = np.array(actual) > np.array(anterior)

y_data_reg = df['fix'][1:]
y_data_class = pd.DataFrame(salida)


porcentajeAlza = sum(salida)/len(salida)
porcentajeBaja = 1 - porcentajeAlza

# Separate training and testing...
dataset_reg = mx.dataHandler.Dataset(input_data=x_data_reg,
                                     output_data=y_data_reg,
                                     normalize=None)
dataset_class = mx.dataHandler.Dataset(input_data=x_data_class,
                                       output_data=y_data_class,
                                       normalize=None)

# training model
RFR = RandomForestRegressor(n_estimators=50)
real_reg_train = dataset_reg.train[1].reshape((dataset_reg.train[1].shape[0],))
real_reg_test = dataset_reg.test[1].reshape((dataset_reg.test[1].shape[0],))
RFR.fit(dataset_reg.train[0],
        real_reg_train)
y_estimated_trainreg = RFR.predict(dataset_reg.train[0])
y_estimated_testreg = RFR.predict(dataset_reg.test[0])
y_estimated_reg = RFR.predict(dataset_reg.input_data)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
plt.subplot(121)
plt.plot(real_reg_train, real_reg_train, ".g", label="Real vals")
plt.plot(real_reg_train, y_estimated_trainreg, ".b", label="Estimates")
plt.title("Train")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.subplot(122)
plt.plot(real_reg_test, real_reg_test, ".g", label="Real vals")
plt.plot(real_reg_test, y_estimated_testreg, ".r", label="Estimates")
plt.title("Test")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.show()

RFC = RandomForestClassifier(n_estimators=10)
real_class_train = dataset_class.train[1].reshape(
                                    (dataset_class.train[1].shape[0],))
real_class_test = dataset_class.test[1].reshape(
                                    (dataset_class.test[1].shape[0],))
RFC.fit(dataset_class.train[0],
        real_class_train)
y_estimated_trainclass = RFC.predict(dataset_class.train[0])
y_estimated_testclass = RFC.predict(dataset_class.test[0])
y_estimated = RFC.predict(dataset_class.input_data)
sum(real_class_train == y_estimated_trainclass) / len(y_estimated_trainclass)
sum(real_class_test == y_estimated_testclass) / len(y_estimated_testclass)

y_score_trainclass = [i[-1] for i in RFC.predict_proba(dataset_class.train[0])]
y_score_testclass = [i[-1] for i in RFC.predict_proba(dataset_class.test[0])]

f,t,_ = roc_curve(real_class_test, y_score_testclass)

plt.plot(f,t)
plt.show()

SL = simple_logistic(x_data=dataset_class.train[0],
                     y_data=dataset_class.train[-1])
SL.train()
SL.evaluate()
y_train = SL.probability.values

SL.evaluate(x = dataset_class.test[0])
y_test = SL.probability.values



#roc_test = roc_curve(real_class_train, y_estimated_trainclass)
#plt.plot(roc_test)
