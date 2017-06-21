#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delta-Hedging

Simulate some stuff...
"""

import numpy as np
import pandas as pd
import quanta as mx
import datetime as dt
from scipy.stats import norm
import matplotlib.pyplot as plt


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


def delta(st, k, r, sigma, T, t):
    """d1."""
    return norm.cdf((np.log(st/k)+(r+sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t)))


def trajectories(S0, kde, n, m):
    """Generate tarjectories."""
    def generateRandom(n, x):
        return list(kde.sample(n, random_state=x).reshape(n,))

    rnd = list(map(lambda x: generateRandom(n, x=x), range(m)))
    log_increment = [np.concatenate([np.array([np.log(S0)]), i]) for i in rnd]
    log_path = [np.cumsum(i) for i in log_increment]
    return pd.DataFrame(np.asmatrix([np.exp(i) for i in log_path]).T)


def estimateVolatility(df_returns, reference_date, n_elem=21):
    """."""
    def getVol(x):
        return np.std(x)*np.sqrt(252)

    std_vector = []
    for i in range(len(df_returns)-n_elem-1):
        upper_limit = (reference_date) if \
                            (i == 0) else (upper_limit - dt.timedelta(days=1))
        lower_limit = upper_limit - dt.timedelta(days=n_elem+1)
        _index = [i and j for i, j in zip((df_returns.index >
                                          lower_limit),
                                          (df_returns.index <= upper_limit))]
        vect_temp = df_returns[_index].rends.values
        if len(vect_temp) < 2:
            continue
        vol = getVol(vect_temp)
        std_vector.append(vol)

    temp_df, kde = generateKDE(np.asarray(std_vector),
                               bandwidth=0.001,
                               _plot=False)
    return kde


def analyticBlackScholes(_type, St, K, r, sigma, T, t):
    """Black-Schole valuation."""
    d1 = (np.log(St/K)+(r+sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1-sigma*np.sqrt(T-t)
    if _type == 'call':
        return St*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2)
    if _type == 'put':
        return -St*norm.cdf(-d1)+K*np.exp(-r*(T-t))*norm.cdf(-d2)
    print('Error: Type not found.')
    return None


# Download prices
df_complete = mx.data.getBanxicoSeries("usdmxn_fix")
df = numericDf(df_complete.copy())
df = cutDf(df, "01/06/2015")

# dataframes
rend = np.log(df["values"].iloc[1:].values/df["values"].iloc[:-1].values)
rend_df = pd.DataFrame({"rends": rend})
prices = df[["values"]].iloc[1:]
rend_df.index = df.timestamp.iloc[1:].values

# KDE Returns
n_kde = 10
_cut = 0.3
n = len(rend_df)
kdes, kdes_df = [], []
for i in range(n_kde):
    _index = rend_df.copy().index.values
    np.random.shuffle(_index)
    _split = np.int(np.asscalar(np.round(n*_cut)))
    vals = rend_df.copy().loc[_index[:_split]].values
    df_distr, kde_returns = generateKDE(rend_df.rends.values,
                                        bandwidth=0.005,
                                        _plot=False)
    kdes.append(kde_returns)
    kdes_df.append(df_distr.copy())

kdes_df[-1].plot(x="x_data", y="density")
plt.title("Return's kde distribution")
plt.xlabel("Returns")
plt.show()

# Trajectories with kde
last_price = np.asscalar(prices.iloc[-1].values)
n_elements = (dt.datetime.strptime("01/10/2017", "%d/%m/%Y") -
              dt.datetime.strptime("15/06/2017", "%d/%m/%Y")).days
n_trajectories = 100
df_trajectories = pd.DataFrame([])
for kde in kdes:
    df_trajectories = pd.concat([df_trajectories,
                                 trajectories(last_price, kde,
                                              n=n_elements,
                                              m=n_trajectories)], axis=1)

# Plot trajectories
for i in df_trajectories:
    plt.plot(np.arange(len(df_trajectories)),
             df_trajectories[i].values, '-b', alpha=0.3)
plt.show()

# KDE volatilities
reference_date = dt.datetime.strptime("15/06/2017", "%d/%m/%Y")

delta_list = {}
k = 18.00
T = n_elements
r = 0.0699

option_value = analyticBlackScholes("call", last_price, k, r,
                                    np.asscalar(kde_base_vols.sample(1)),
                                    n_elements/360, 0)

kde_base_vols = estimateVolatility(rend_df, reference_date, n_elem=5)
for i in np.arange(n_elements)[::-1]+1:
    print(i)
    kde_vol = estimateVolatility(rend_df, reference_date, n_elem=int(i))
    n_vol = [(np.asscalar(j) if j > 0 else 0.01)
             for j in kde_vol.sample(n_trajectories*n_kde)]
    price_vector = df_trajectories.iloc[n_elements-i].values
    t = n_elements-i
    delta_list[n_elements-i] = [delta(st, k, r, sigma, T/360, t/360)
                                for st, sigma in zip(price_vector, n_vol)]

df_delta = pd.DataFrame(delta_list).T
#df_delta.index = n_elements - df_delta.index.values
df_delta

PL = {}
for i in df_delta:
    vals = df_delta[i].values
    tc = df_trajectories.iloc[1:-1, i].values
    PL[i] = (vals[1:] - vals[:-1])*tc

PL = pd.DataFrame(PL, index=np.arange(len(PL[0]))+1)

new_line = {0:df_trajectories.iloc[0, :].values * df_delta.iloc[0, :].values}
new_line = pd.DataFrame(new_line).T

n_end = len(df_trajectories)-1
end_line = {n_end: -df_trajectories.iloc[-1, :].values *
            df_delta.iloc[-1, :].values}
end_line = pd.DataFrame(end_line).T

PL = pd.concat([new_line, PL, end_line], 0)

PL


profit = PL.sum().values

options_wins = [(i if i > 0 else 0)
                for i in df_trajectories.iloc[-1, :].values - k]
options_wins = np.array(options_wins)

np.mean(options_wins)
np.mean(profit)

profs = profit + options_wins - option_value
np.mean(profs)
np.min(profs)
np.max(profs)

# win probability
sum(profs > 0)/len(profs)

pd.DataFrame(profit+options_wins-option_value).plot(kind="kde")
plt.show()
