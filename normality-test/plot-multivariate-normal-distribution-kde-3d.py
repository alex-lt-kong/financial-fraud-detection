#!/usr/bin/python3

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mu=np.array([1,10,20])
sigma=np.matrix([[4,10,0],[10,7,3],[0,2,100]])
data=np.random.multivariate_normal(mu,sigma,10000)
#print(data)

values = data.T

kde = stats.gaussian_kde(values)
density = kde(values)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
x, y, z = values
ax.scatter(x, y, z, c=density)
statistic, pvalue = stats.normaltest(data)
import pandas as pd

da = pd.DataFrame(data)
skewness = da.skew()
print('skewness: {}'.format(skewness))
print(stats.normaltest(da))
print(stats.mstats.normaltest(data, axis=0))
plt.show()

