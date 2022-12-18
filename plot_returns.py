#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:35:58 2022

@author: Robbie von der Schmidt
@contributor: Daksh Sharma


"""
# TODO(robbiev): Move this entire thing to Colab so it can be iterated more easily.
# See https://colab.research.google.com/.

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import scipy.stats

# Start the timer.
t0 = time.perf_counter()

def SummaryStatistics(data):
    """
    Returns a NumPy darray of some summary statistics for the given data.
    Specifically, returns the length, minimum, maximum, mean, median, mode, stddev, skew, and kurtosis.
    NOTE for the reader: code is easy, basic statistics is hard.

    Args:
    - data: A Pandas dataframe.
    """

    sumstats = np.array([len(data), np.min(data), np.max(data), np.mean(data), np.median(data), float(scipy.stats.mode(data)[0]), np.std(data), float(scipy.stats.skew(data)[0]), float(scipy.stats.kurtosis(data)[0])])
    return sumstats

def TBW_KernelDensityEstimate(data, grid_search_range):
    """
    Returns a fitted SKL Estimator for the given training data
    using Kernel Density estimation with 20-fold cross-validation.

    Args:
    - data. A numpy ndarray of data.
    # TODO(robbiev): What should this data look like?

    """
     params = {'bandwidth': grid_search_range}
     # TODO(robbiev): is 20 excessive? Unclear. If not, remove this TODO.
     grid = GridSearchCV(KernelDensity(), params, cv = 20)
     return grid.fit(data.reshape(-1,1))

#return an array of densities for an array of values according to a KDE
def Score(KernelDensity, x_axis):
    """

    """
    log_densities = KernelDensity.score_samples(x_axis)
    densities = np.exp(log_densities)
    return densities 

#import data
df = pd.read_csv('MAAMG.csv')
GOOG = np.log((df['GOOG_Close'].iloc[1:].values/df['GOOG_Close'].iloc[:-1].values).reshape(-1,1))
META = np.log((df['META_Close'].iloc[1:].values/df['META_Close'].iloc[:-1].values).reshape(-1,1))

print(GOOG)

#split data into training and test sets 
GOOG_train, GOOG_test = train_test_split(GOOG, test_size = .1)
META_train, META_test = train_test_split(META, test_size = .1)

#decide on a range and increment to search for an optimal bandwidth 
bandwidth_search_range = np.linspace(0.001,.1,1000)

#find KDEs for GOOG and META 
GOOG_TBWKDE = TBW_KernelDensityEstimate(GOOG_train, bandwidth_search_range)
META_TBWKDE = TBW_KernelDensityEstimate(META_train, bandwidth_search_range)

#create x_axis for graphing 
x_axis = np.linspace(-.2, .2, 10000).reshape(-1,1)

#calculate densities for GOOG and META
GOOG_densities = Score(GOOG_TBWKDE, x_axis)
META_densities = Score(META_TBWKDE, x_axis)

#put summary statistics into a dataframe
SummaryStatisticsDF = pd.DataFrame(index = ["N", "MIN", "MAX", "MEAN", "MEDIAN", "MODE", "STD DEV", "SKEW", "KURT"], data = {"GOOG": SummaryStatistics(GOOG_train), "META": SummaryStatistics(META_train)})


#graph
plt.plot(x_axis, GOOG_densities, color = 'black', label = "GOOG_kde")
plt.plot(x_axis, scipy.stats.norm.pdf(x_axis, np.mean(GOOG_train),np.std(GOOG_train)), color = 'dimgrey', label = "GOOG_norm", linestyle = 'dashed')
plt.hist(GOOG_train, color = 'dimgrey', bins = 70, alpha = .25, label = "GOOG_hist")
plt.plot(x_axis, META_densities, color = 'blue', label = "META_kde")
plt.plot(x_axis, scipy.stats.norm.pdf(x_axis, np.mean(META_train), np.std(META_train)), color = 'cornflowerblue', label = 'META_norm', linestyle = 'dashed')
plt.hist(META_train, color = 'cornflowerblue', bins = 70, alpha = .25, label = "META_hist")
plt.title('Density of Log Daily Returns for GOOG and META (11/10/21 - 11/09/22)')
plt.xlabel("LN(Daily Return)")
plt.ylabel("Probability Density")
plt.legend()

#stop timer
t1 = time.perf_counter()

print("\nThis program took", round(t1-t0,2),"seconds to run :).")

print(SummaryStatisticsDF)

print("\nCross validation supports bandwidths of",GOOG_TBWKDE.best_estimator_,"for GOOG and",META_TBWKDE.best_estimator_,"for META")

=


"""
some poorly fleshed out scratch 
the most important function that i need to add is one that would allow me to get p values for log returns according ot the KDEs

"""
# a = [0]*100
# for i in range(100): 
#     a[i] = 1/10000*sum(np.exp(grid.score_samples(np.linspace(-i*.01,i*.01,10000).reshape(-1,1))))
# print(a[np.abs(np.asarray(a) - 1).argmin()])
# print(pd.DataFrame([list(range(100)),a]))
# z = pd.DataFrame([list(range(100)),a])
# print(np.array(grid.score([[-.0128]])).shape)
# print(sum(np.exp(grid.score_samples(x_axis[x_axis<=-.0128].reshape(-1,1))))/sum(np.exp(grid.score_samples(x_axis.reshape(-1,1)))))
# ####### grid.best_estimator_)
# # use these guys below for cross validation
# # scores = cross_validate(grid, X_train.reshape(-1,1))
# # print(sum(scores['test_score']))
# # print(np.var(X_train, axis = 0))
#from sklearn.model_selection import cross_validate


