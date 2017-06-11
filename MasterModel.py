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


def generateKDE(datapoints, bandwidth=0.3, _plot=True):
    """Create KDE Estimation."""
    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
        datapoints.reshape(-1, 1))

    _min, _max = np.min(datapoints), np.max(datapoints)
    x_plot = np.arange(_min, _max+_max/2, (_max-_min)/10000)
    log_dens = kde.score_samples(x_plot.reshape(-1, 1))

    if _plot:
        plt.plot(x_plot, np.exp(log_dens))
        plt.title("Kernel Density Estimator: Errors")

    return pd.DataFrame({'x_data': x_plot, 'density': np.exp(log_dens)}), kde


def getReturns(vector):
    """Get returns."""
    return np.log(vector[1:]/vector[:-1])


# Variables
ndays = 1
nlags = 5

# Download prices
df = mx.data.getBanxicoSeries("usdmxn_fix")
df = numericDf(df)
df = cutDf(df, "01/01/2014")
df.shape
rend = np.log(df["values"].iloc[1:].values/df["values"].iloc[:-1].values)
rend_df = pd.DataFrame({"rends": rend})
rend_df.shape
prices = df[["values"]].iloc[1:]

prices.index = np.arange(1, len(prices)+1)
rend_df.index = prices.index
prices.iloc[-1]
# Create lags
price_lags = lagMatrix(prices.iloc[:-ndays], lag=nlags)
rend_lags = lagMatrix(rend_df.iloc[:-ndays], lag=nlags)
rend_lags.columns = [str(i)+"r" for i in rend_lags.columns]
price_lags.columns = [str(i)+"p" for i in price_lags.columns]

# Pattern matrix
patterns = pd.read_pickle("competitive_neurons.pkl")


def getDistance(series, patterns):
    """."""
    leading_neuron = None
    closest_distance = float("inf")
    for neuron in patterns:
        distance = np.linalg.norm(np.asarray(series) -
                                  patterns[neuron].values)
        if distance < closest_distance:
            closest_distance = distance
            leading_neuron = neuron
    return leading_neuron


detected_patterns = rend_lags.apply(lambda x: getDistance(x,
                                                          patterns=patterns),
                                    1)
np.unique(detected_patterns)
plt.hist(detected_patterns)
plt.show()
patterns_hist = {}
for i in detected_patterns:
    if i not in patterns_hist:
        patterns_hist[i] = 1
    else:
        patterns_hist[i] += 1
patterns_hist


def one_hot(attrs, attr):
    """."""
    def hot(x):
        if x == attr:
            return 1
        else:
            return 0
    return [hot(a) for a in attrs]


attrs = np.unique(detected_patterns)
pattern_df = detected_patterns.apply(lambda x: pd.Series(
                                                one_hot(attrs=attrs, attr=x)))


# input / output
output_data = prices.loc[nlags+ndays:]
input_data = pd.concat([price_lags, rend_lags], axis=1)


input_data.shape
output_data.shape

# Create model
dataset = mx.dataHandler.Dataset(input_data, output_data, normalize="minmax")
_epochs = 2000
_hdl = [10, 5, 5]

mlp = mx.neuralNets.mlpRegressor(hidden_layers=_hdl)
mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
train = mlp.train_results
train = pd.DataFrame(train)
test = pd.DataFrame(mlp.test(dataset=dataset))

# estimate
y_estimate = mlp.freeEval(dataset.norm_input_data.values)
plt.plot(dataset.norm_input_data[0].values,
         [np.asscalar(i) for i in dataset.norm_output_data.values], "b.")
plt.plot(dataset.norm_input_data[0].values, y_estimate, "r.")
plt.show()

# visualize the training performance
plt.plot(np.arange(_epochs), mlp.epoch_error)
plt.title("MSE per epoch.")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()


train.errors.plot(kind="kde")
test.errors.plot(kind="kde")
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
plt.subplot(121)
plt.plot(train.y, train.y, ".g", label="Real vals")
plt.plot(train.y, train.estimates, ".b", label="Estimates")
plt.title("Train")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.subplot(122)
plt.plot(test.y, test.y, ".g", label="Real vals")
plt.plot(test.y, test.estimates, ".r", label="Estimates")
plt.title("Test")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.show()

# Errors
error_data = train.errors.values
np.mean(error_data)
desnormalize(np.mean(error_data),dataset)

distribution, kde = generateKDE(error_data, bandwidth=0.005)
distribution.plot(x="x_data", y="density")
plt.show()


# Extract normalizer
def normalizeExternalData(vector, dataset):
    """."""
    _min, _max = dataset.min_x, dataset.max_x
    vector = np.asarray(vector)
    return (vector-_min)/(_max-_min)


def desnormalize(value, dataset):
    """."""
    _min, _max = dataset.min_y, dataset.max_y
    value = (_max-_min)*value + _min
    return np.asscalar(value)


dataset


vector = list(prices.values[-nlags:]) + \
                    list(getReturns(prices.values[-(nlags+1):]))
vector = np.asarray(vector)
vector = vector.reshape((vector.shape[0],))


vector = normalizeExternalData(vector, dataset)
raw_base_estimation = mlp.freeEval(pd.DataFrame({"vect": vector}).T.values)
base_estimation = desnormalize(raw_base_estimation, dataset)


def getForecastVector(base_estimation, kde, n):
    """."""
    return np.array([base_estimation + np.asscalar(kde.sample())
                    for i in range(n)])


forecast = getForecastVector(base_estimation, kde, 1000)
mean_estimation = np.mean(forecast)
pd.DataFrame({"forecast": forecast}).plot(kind="kde")
plt.show()
np.max(forecast)
np.min(forecast)

desnorm_train_y = np.array([desnormalize(i, dataset) for i in train.y])
desnorm_trainestim_y = np.array([desnormalize(i, dataset) for i in train.estimates])
desnorm_test_y = np.array([desnormalize(i, dataset) for i in test.y])
desnorm_testestim_y = np.array([desnormalize(i, dataset) for i in test.estimates])

np.sqrt(np.mean((desnorm_train_y - desnorm_trainestim_y)**2))
np.sqrt(np.mean((desnorm_test_y - desnorm_testestim_y)**2))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
plt.subplot(121)
plt.plot(desnorm_train_y, desnorm_train_y, ".g", label="Real vals")
plt.plot(desnorm_train_y, desnorm_trainestim_y, ".b", label="Estimates")
plt.title("Train")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.subplot(122)
plt.plot(desnorm_test_y, desnorm_test_y, ".g", label="Real vals")
plt.plot(desnorm_test_y, desnorm_testestim_y, ".r", label="Estimates")
plt.title("Test")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.show()
