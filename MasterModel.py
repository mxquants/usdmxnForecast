#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master Model.

import os
os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")
os.chdir("C://Users//danie//Documents//tcForecast")
"""

import numpy as np
import pandas as pd
import quanta as mx
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats.mstats import normaltest
from scipy import stats

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


def one_hot(attrs, attr):
    """."""
    def hot(x):
        if x == attr:
            return 1
        else:
            return 0
    return [hot(a) for a in attrs]


def numberOfWeights(dataset, hidden_layers, batch_ref=0.7):
    """Get the number of parameters to estimate."""
    n_input = np.shape(dataset.train[0])[-1]
    n_output = np.shape(dataset.train[-1])[-1]
    numbers = np.asarray([n_input]+list(hidden_layers)+[n_output])
    # params = np.prod(np.array([n_input]+list(hidden_layers)+[n_output]))
    params = np.sum([i * j for i, j in zip(numbers[:-1], numbers[1:])])
    n_elements = np.shape(dataset.train[0])[0]
    return params, n_elements, batch_ref*n_elements > params


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


def getForecastVector(base_estimation, kde, n):
    """."""
    return np.array([base_estimation + np.asscalar(kde.sample())
                    for i in range(n)])


def saveEstimate(estimate_log):
    """Save the forecast."""
    forecast_log = pd.read_pickle("forecast_log.pkl")
    _index = forecast_log.index.max()+1
    temp = pd.DataFrame(estimates_log, index=[_index])
    pd.concat([forecast_log, temp]).to_pickle("forecast_log.pkl")


def findLowerLimit(vector, alpha=0.05, step=0.0001):
    """."""
    _limit = np.min(vector)
    n = len(vector)
    while True:
        _alpha = sum(vector < _limit) / n
        if _alpha < alpha:
            _limit += step
            continue
        return _limit, _alpha


def findUpperLimit(vector, alpha=0.05, step=0.0001):
    """."""
    _limit = np.max(vector)
    n = len(vector)
    while True:
        _alpha = sum(vector > _limit) / n
        if _alpha < alpha:
            _limit -= step
            continue
        return _limit, _alpha


def getConfidenceInterval(vector, alpha, step=0.0001):
    """."""
    low, _ = findLowerLimit(vector, alpha, step)
    up, _ = findUpperLimit(vector, alpha, step)
    return low, up


def getVolatilityVect(vect, window):
    """Get the vol. vect."""
    vect = [i**2 for i in vect]
    n = len(vect)
    vol = []
    for i in range(n-window):
        vol.append(np.sqrt(252*np.mean(vect[i:(i+window)])))
    return vol


def findBestWindow(vect, reference_vect):
    """Find best volatility window."""
    _from = 5
    _to = 150
    correlation = []
    for i in np.arange(_from, _to+1):
        temp = getVolatilityVect(vect, i)
        n = len(temp)
        adequate_ref = reference_vect[i:(i+n)]
        correlation.append(np.corrcoef(temp, adequate_ref)[0, 1])
    return np.argmax(correlation)+_from


def movingAverage(vector, m):
    """Moving average."""
    n = len(vector)
    a = []
    for i in range(n-m):
        a.append(np.mean(vector[i:(i+m)]))
    return a


def FAC(vector, k):
    """Autocorrelation function."""
    vector_k = vector[:-k] if k != 0 else vector
    vector = vector[k:]
    covariance = np.cov(vector, vector_k)
    return covariance[0, 1] / covariance[0, 0]


def FACP(vector, k):
    """Partial autocorrelation function."""
    if k == 0:
        return 1
    matrix = np.matlib.ones((k, k))
    for i in range(k-1):
        rvector = np.asarray([FAC(vector, j) for j in range(1, k-i)])
        matrix[i, i+1:k] = rvector.reshape((1, k-i-1))
        matrix[i+1:k, i] = rvector.reshape((k-i-1, 1))
    return np.asscalar(np.linalg.inv(matrix).dot(np.array(
                [FAC(vector, i) for i in range(1, k+1)]).reshape((k, 1)))[-1])



a = np.matlib.ones((5,5))
a[:, 1] = np.array([0,0,0,0,0]).reshape((5,1))
a[1:, 1] = np.array([3,2,2,2]).reshape((4,1))


plt.plot([FACP(rend, i) for i in range(20)], ".")
plt.show()

plt.plot([FACP(df["values"].values - np.mean(df["values"].values), i)
          for i in range(20)], ".")
plt.show()



np.asscalar(a[-1])
np.linalg.inv(a)
a

FAC(rend, 4)
rend[:-0]
np.var(rend)

# Variables
ndays = 1
nlags = 5
nlow = 20
nhigh = 40
nvol = 86

# Download raw data
df_complete = mx.data.getBanxicoSeries("usdmxn_fix")
raw_interest_rate = mx.data.getBanxicoSeries("tiie28")
raw_high = mx.data.getBanxicoSeries("usdmxn_max")
raw_low = mx.data.getBanxicoSeries("usdmxn_min")

# str -> numeric
interest_rate = numericDf(raw_interest_rate.copy())
df = numericDf(df_complete.copy())
_high = numericDf(raw_high.copy())
_low = numericDf(raw_low.copy())

# query for date
reference_string_date = "01/01/2016"
df = cutDf(df, reference_string_date)
interest_rate = cutDf(interest_rate, reference_string_date)
_high = cutDf(_high, reference_string_date)
_low = cutDf(_low, reference_string_date)

# Plot prices
plt.plot(df["timestamp"], df["values"])
plt.show()

# Plot Interest Rate
plt.plot(interest_rate["timestamp"], interest_rate["values"])
plt.show()

# Timestamp as index
df.index = df.timestamp
_high.index = _high.timestamp
_low.index = _low.timestamp
interest_rate.index = interest_rate.timestamp

# Check if index between prices and interest rate are the same
ref_index = df.index
interest_rate = interest_rate.loc[ref_index]
_high = _high.loc[ref_index]
_low = _low.loc[ref_index]

interest_rate.shape
df.shape
_high.shape
_low.shape

# Calculate returns for interest rate
interest_returns = interest_rate["values"].values[1:] - \
                   interest_rate["values"].values[:-1]
interest_returns = pd.DataFrame({"interest": interest_returns})

stats.shapiro(interest_returns.interest.values) # not normal when 2nd < 0.05

# Calculate returns for prices
rend = np.log(df["values"].iloc[1:].values/df["values"].iloc[:-1].values)
rend_df = pd.DataFrame({"rends": rend})
rend_df.shape

stats.shapiro(rend) # not normal when 2nd < 0.05

# Prices as a dataframe (without first price)
prices = df[["values"]].iloc[1:]

# Volatility feature
nvol = findBestWindow(rend, prices["values"].values)
vols = getVolatilityVect(rend, nvol)
volatility_df = pd.DataFrame({"vols": vols})
len(vols)

# Moving averages
ma_low = movingAverage(_low["values"].values, nlow)
ma_high = movingAverage(_high["values"].values, nhigh)

trash0, trash1 = (0 if nlow > nhigh else nhigh-nlow), \
                 (0 if nlow < nhigh else nlow-nhigh)

moving_average = pd.DataFrame({"low": ma_low[trash0:],
                               "high": ma_high[trash1:]})

lowhigh = pd.DataFrame({"vals": moving_average.apply(
                                    lambda x: x["low"]-x["high"], 1).values})
lowhigh.plot()
plt.show()

# Adjust dataframes
max_cut = max([nvol, nlow, nhigh])
cut_lowhigh = max_cut - max([nlow, nhigh])
cut_vol = max_cut - nvol

prices = prices.iloc[max_cut:]
rend_df = rend_df.iloc[max_cut:]
interest_returns = interest_returns.iloc[max_cut:]
lowhigh = lowhigh.iloc[cut_lowhigh+1:]
volatility_df = volatility_df.iloc[cut_vol:]

prices.shape
rend_df.shape
interest_returns.shape
lowhigh.shape
volatility_df.shape

# Change index to numeric
numeric_index = np.arange(1, len(prices)+1)
prices.index = numeric_index
rend_df.index = numeric_index
interest_returns.index = numeric_index
lowhigh.index = numeric_index
volatility_df.index = numeric_index

# Last price
prices.iloc[-1]


# Create lags
price_lags = lagMatrix(prices.iloc[:-ndays], lag=nlags)
rend_lags = lagMatrix(rend_df.iloc[:-ndays], lag=nlags)
rend_lags.columns = [str(i)+"r" for i in rend_lags.columns]
price_lags.columns = [str(i)+"p" for i in price_lags.columns]

# Pattern matrix
patterns1 = pd.read_pickle("competitive_neurons.pkl")
patterns2 = pd.read_pickle("kmeans.pkl")

detected_patterns = rend_lags.apply(lambda x: getDistance(x,
                                                          patterns=patterns2),
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

attrs = np.unique(detected_patterns)
pattern_df = detected_patterns.apply(lambda x: pd.Series(
                                                one_hot(attrs=attrs, attr=x)))

prices.shape
price_lags.shape
rend_lags.shape
# pattern_df.shape
interest_returns.shape
lowhigh.shape
volatility_df.shape

# input / output
output_data = prices.loc[nlags+ndays+1:]
"""
input_data = pd.concat([price_lags, rend_lags, pattern_df,
                        interest_returns, lowhigh,
                        volatility_df], axis=1).dropna()
"""
input_data = pd.concat([price_lags, rend_lags,
                        interest_returns, lowhigh,
                        volatility_df], axis=1).dropna()

input_data.shape
output_data.shape

# Explore model configuration
dataset = mx.dataHandler.Dataset(input_data, output_data, normalize="minmax")
_epochs = 2000
max_hidden = 5
max_neurons = 20
mse = {"hidden": [], "neurons": [], "mse": []}
hidden_vector = [None]
temp_hidden_vector = [None]
best_per_layer = []
for i in range(1, max_hidden+1):
    print("\n")
    for j in range(1, max_neurons+1):
        temp_hidden_vector[i-1] = j
        nparams, nelements, not_warn = numberOfWeights(dataset,
                                                       temp_hidden_vector)
        if not not_warn:
            print("Not viable anymore.")
            break
        hidden_vector[i-1] = j
        mse["hidden"].append(i)
        mse["neurons"].append(j)
        mlp = mx.neuralNets.mlpRegressor(hidden_layers=hidden_vector)
        mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
        mse["mse"].append(np.mean(mlp.test(dataset=dataset)["square_error"]))
        print("Evaluating: ({i},{j}) => {mse:0.6f}".format(i=i, j=j,
                                                           mse=mse["mse"][-1]))
    if not not_warn:
        break
    temp = pd.DataFrame(mse)
    min_mse_arg = temp.query("hidden == {}".format(i)).mse.argmin()
    temp_hidden_vector[i-1] = temp["neurons"].iloc[min_mse_arg]
    hidden_vector[i-1] = temp["neurons"].iloc[min_mse_arg]
    best_per_layer.append(temp["mse"].iloc[min_mse_arg])
    hidden_vector.append(None)
    temp_hidden_vector.append(None)

hidden_vector
best_per_layer

plt.plot(np.arange(len(best_per_layer))+1, best_per_layer)
plt.title("MSE best per n-neurons per hidden layer index")
plt.show()

mse_df = pd.DataFrame(mse)
mse_df
x = mse["hidden"]
y = mse["neurons"]
z = mse["mse"]
min_z, max_z = min(z), max(z)
z = [(i-min_z)/(max_z-min_z) for i in z]
plt.scatter(x, y, c=z, s=100)
# plt.gray()
plt.xlabel("Number of hidden layers")
plt.ylabel("Number of neurons at last hl")
plt.grid()
plt.show()

plt.plot(x, mse["mse"])
# plt.gray()
plt.xlabel("Number of hidden layers")
plt.ylabel("mse")
plt.grid()
plt.show()


plt.plot(y, mse["mse"], '.b')
# plt.gray()
plt.xlabel("Number of neurons")
plt.ylabel("mse")
plt.grid()
plt.show()

# Grid eval
grid = [np.arange(1, max_neurons) for i in range(1, max_hidden)]
grid = pd.DataFrame(grid).T


def getMSE(architecture, dataset):
    """Return MSE or inf."""
    nparams, nelements, not_warn = numberOfWeights(dataset, architecture)
    if not not_warn:
        return np.float("inf")
    mlp = mx.neuralNets.mlpRegressor(hidden_layers=architecture)
    mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
    return np.mean(mlp.test(dataset=dataset)["square_error"])


mse_res = {}
save_min = float("inf")
save_architecture = None
for i in grid:
    hidden = i
    mse_res[hidden] = []
    for j in grid[hidden].values:

        architecture = [j]*hidden
        temp_mse = getMSE(architecture, dataset)
        if temp_mse < save_min:
            save_min = temp_mse
            save_architecture = architecture
        mse_res[hidden].append(temp_mse)
        print("({}, {}) => {}".format(hidden, j, mse))

mse_vals = pd.DataFrame(mse_res, index=grid[hidden].values)
mse_vals


min_arg, min_mse = None, float("inf")
for col in mse_vals:
    local_min = mse_vals[col].min()
    local_argmin = mse_vals[col].argmin()
    if local_min < min_mse:
        min_mse = local_min
        min_arg = local_argmin
        best_architecture = (col, min_arg)
    mse_vals[col].plot()
    plt.title("Fix number of hidden layers: {}".format(col))
    plt.xlabel("Number of neurons (for each layer)")
    plt.show()

best_architecture
min_mse

# Create model

_hdl = [5]
numberOfWeights(dataset, _hdl, batch_ref=0.7)

mlp = mx.neuralNets.mlpRegressor(hidden_layers=_hdl)
mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
train = mlp.train_results
train = pd.DataFrame(train)
test = pd.DataFrame(mlp.test(dataset=dataset))

# mse
mlp.test(dataset=dataset)["square_error"][-1]

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

# normaltest(np.random.normal(0,1,1000))
# stats.shapiro(np.random.normal(0,1,1000)) # is not normal if 2nd < 0.05

train.errors.plot()
plt.show()

normaltest(train.errors.values)
stats.shapiro(train.errors.values)


test.errors.plot()
plt.show()

normaltest(test.errors.values)
stats.shapiro(test.errors[test.errors.values < 0.2].values)

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
error_data = test.errors.values
np.mean(error_data)

n_distribution, n_kde = generateKDE(error_data, bandwidth=0.02, _plot=False)
n_distribution.plot(x="x_data", y="density")
plt.title("Error's kde distribution for test-data")
plt.xlabel("Error")
plt.show()


# Error in MXN

desnorm_train_y = np.array([desnormalize(i, dataset) for i in train.y])
desnorm_trainestim_y = np.array([desnormalize(i, dataset)
                                 for i in train.estimates])
desnorm_test_y = np.array([desnormalize(i, dataset) for i in test.y])
desnorm_testestim_y = np.array([desnormalize(i, dataset)
                                for i in test.estimates])

train_mxn_error = np.sqrt(np.mean((desnorm_train_y - desnorm_trainestim_y)**2))
test_mxn_error = np.sqrt(np.mean((desnorm_test_y - desnorm_testestim_y)**2))
train_mxn_error
test_mxn_error

error_data = (desnorm_test_y - desnorm_testestim_y)
distribution, kde = generateKDE(error_data, bandwidth=0.05, _plot=False)
distribution.plot(x="x_data", y="density")
plt.title("Error's kde distribution for test-data")
plt.xlabel("Error")
plt.show()


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

# Forecast
"""
vector = list(prices.values[-nlags:]) + \
                    list(getReturns(prices.values[-(nlags+1):])) + \
                    one_hot(attrs=attrs, attr=getDistance(
                        getReturns(prices.values[-(nlags+1):]), patterns2)) + \
                    [interest_returns.interest.values[-1]] + \
                    [lowhigh.vals.values[-1]] + \
                    [volatility_df.vols.values[-1]]


"""
vector = list(prices.values[-nlags:]) + \
                    list(getReturns(prices.values[-(nlags+1):])) + \
                    [interest_returns.interest.values[-1]] + \
                    [lowhigh.vals.values[-1]] + \
                    [volatility_df.vols.values[-1]]

vector = np.asarray(vector)
vector = vector.reshape((vector.shape[0],))


vector = normalizeExternalData(vector, dataset)
raw_base_estimation = mlp.freeEval(pd.DataFrame({"vect": vector}).T.values)
base_estimation = desnormalize(raw_base_estimation, dataset)
base_estimation

forecast = getForecastVector(base_estimation, kde, 10000)
mean_estimation = np.mean(forecast)
std_estimation = np.std(forecast)
mean_estimation
std_estimation

max_forecast = np.max(forecast)
min_forecast = np.min(forecast)
max_forecast
min_forecast

increase_prob = sum(forecast > np.asscalar(prices.values[-1])) / len(forecast)
confidence_level = 0.75
alpha = (1-confidence_level)/2
lower, upper = getConfidenceInterval(forecast, alpha)

text_current = """\
CURRENT INFO

Date: {date}
Price: $ {price:0.6f} MXN

""".format(
        date=str(df.timestamp.values[-1])[:10],
        price=np.asscalar(prices.values[-1])
)
text_forecast = """\
FORECAST

Base estimate: $ {base:0.6f} MXN
Test's sqrt(MSE): $ {mse:0.4f} MXN

Mean: $ {mean:0.6f} MXN
Std: $ {std:0.4f} MXN

Confidence level of: {cl:0.2f} %
{min:0.6f} < x < {max:0.6f}

prob(delta>0) = {increase:0.2f}
""".format(
        mean=mean_estimation,
        max=upper,
        min=lower,
        std=std_estimation,
        mse=test_mxn_error,
        base=base_estimation,
        cl=100*confidence_level,
        increase=100*increase_prob
)


# Plot forecast distribution

f_distr, f_kde = generateKDE(forecast, bandwidth=0.05, _plot=False)
df_forecast = f_distr.query("x_data < {} and x_data > {}".format(max_forecast,
                                                                 min_forecast))

_delta = mean_estimation-np.asscalar(prices.values[-1])

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
plt.subplot(211)
plt.plot(df_forecast.x_data, df_forecast.density)
plt.axvline(x=np.asscalar(prices.values[-1]), alpha=0.85, ls='--')
plt.xlabel("Forecast value (mxn)")
plt.title("Forecast distribution " +
          "(next day: change of $ {delta:0.4f} MXN)".format(
                delta=_delta))
plt.grid()
plt.subplot(223)
plt.text(0.5, 0.5,
         text_current,
         verticalalignment='center',
         horizontalalignment='center')
plt.axis('off')
plt.subplot(224)
plt.text(0.5, 0.5,
         text_forecast,
         verticalalignment='center',
         horizontalalignment='center')
plt.axis('off')
plt.show()


# Pack forecast info

estimate_generation_date = dt.datetime.now().strftime("%d/%m/%Y")
estimate_target_date = (dt.datetime.now() + dt.timedelta(days=1)).strftime(
                                                                "%d/%m/%Y")
estimate_val = mean_estimation
min_val = min_forecast
max_val = max_forecast
real_val = 0
delta = _delta

estimates_log = {
                 "base_estimation": base_estimation,
                 "estimate_generation_date": estimate_generation_date,
                 "estimate_target_date": estimate_target_date,
                 "estimate_val": estimate_val,
                 "real_val": real_val,
                 "min_val": min_val,
                 "max_val": max_val,
                 "delta": delta}
# saveEstimate(estimates_log)

same_vals_error = prices.values[1:] - prices.values[:-1]
np.sqrt(np.mean(list(map(lambda x: x**2, same_vals_error))))
np.mean(same_vals_error)
np.std(same_vals_error)

pd.DataFrame(same_vals_error).plot(kind="kde")
plt.show()

stats.shapiro(same_vals_error)
