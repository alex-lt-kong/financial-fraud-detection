#!/usr/bin/python3

import os
import pandas as pd
import pathlib
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Covariance
# https://datascienceplus.com/understanding-the-covariance-matrix/
def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
   # print('xbar: {}, ybar: {}'.format(xbar, ybar))
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

# Covariance matrix
def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1]), cov(X[0], X[2])],
                     [cov(X[1], X[0]), cov(X[1], X[1]), cov(X[1], X[2])],
                     [cov(X[2], X[0]), cov(X[2], X[1]), cov(X[2], X[2])]])

import statsmodels.api as sm
def plot_distribution_of_single_dimension(dataset, plot_title: str, i):
    plt.figure(i)
 #  print('values: {}'.format(values[0]))
    dataset_pd = pd.DataFrame(dataset)
   # statistic, pvalue = stats.normaltest(dataset)
   # skewness = float(dataset_pd.skew())
   # kurtosis = float(dataset_pd.kurtosis())
   # mu, std = stats.norm.fit(dataset_pd)
   # pvalue = float(pvalue)   
    plt.title('{}'.format(plot_title))
    #plt.plot(values[0])
    plt.hist(x=dataset, bins=200, histtype='bar', orientation='vertical', edgecolor='black', linewidth=0.2)  
    # hist stands for histogram here. However, the real origin of the name "histogram" is not clear.
    
    # https://docs.scipy.org/doc/scipy/reference/stats.html
   # sm.qqplot(data=dataset, dist=scipy.stats.distributions.logistic, line='45')
   # plt.show()
 
def main():
   
    parent_dir = pathlib.Path(__file__).parent.absolute()    
    df = pd.read_csv(os.path.join(parent_dir, 'hkex_standardized.csv'))
    #print(df[0])
    data = np.array(df)
    
    values = data.T
    #print(len(values))
    mu_vector = np.array(values[2: len(values) - 1].mean())
    #print(mu_vector)
    
    
    #kde = stats.gaussian_kde(values)
    #density = kde(values)
    
    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #x, y, z = values
    #ax.scatter(x, y, z, c=density)
    #statistic, pvalue = stats.normaltest(data)
    #import pandas as pd
    
    #da = pd.DataFrame(data)
    #skewness = da.skew()
    #plt.show()
    # Dr. Yu said that normality test should not work
    # but should I somehow test normality? Say for every one dimension?
    #print('skewness: {}'.format(skewness))
    #print(stats.normaltest(da))
    #print(stats.mstats.normaltest(data, axis=0))
    #plt.show()
    #plot_distribution_of_single_dimension(values[2])
    
    #cov = (cov_mat(values)) # (or with np.cov(X.T))
    #print(values[2:17])
    covariance_matrix = np.cov(values[2: len(values) - 1].astype(float))
    #print(covariance_matrix)
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)
    
    
    #plot_distribution_of_single_dimension(values[3], 'current ratio (annual)', 1)
    #plot_distribution_of_single_dimension(values[4], 'return on equity (ttm)', 2)
    #plot_distribution_of_single_dimension(values[5], 'asset turnover (annual)', 3)
    #plt.show()
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html
    mahalanobis_distance = []
    mahalanobis_distance_min, mahalanobis_distance_max = 12345, 0
    
    for i in range (0, len(data)):
     #   print('data: {}'.format(data[i][2:len(values) - 1]))
     #   print('mu_vector: {}'.format(mu_vector))
        mahalanobis_distance.append(scipy.spatial.distance.mahalanobis(data[i][2:len(values) - 1], mu_vector, covariance_matrix_inv))
        if mahalanobis_distance[len(mahalanobis_distance) - 1] < mahalanobis_distance_min:
            mahalanobis_distance_min = mahalanobis_distance[len(mahalanobis_distance) - 1]
        if mahalanobis_distance[len(mahalanobis_distance) - 1] > mahalanobis_distance_max:
            mahalanobis_distance_max = mahalanobis_distance[len(mahalanobis_distance) - 1]
    print(mahalanobis_distance_min, mahalanobis_distance_max)
    mahalanobis_distance = np.array(mahalanobis_distance)
    #plot_distribution_of_single_dimension(mahalanobis_distance, 'mahalanobis_distance')
    #data['mahalanobis_distance'] = mahalanobis_distance
    df_new = pd.DataFrame(data)
    #print(len(mahalanobis_distance))
    #print(len(np.insert(mahalanobis_distance, len(values) - 2, 26)))
    t = np.insert(mahalanobis_distance, 0, 26)
    #t = np.insert(t, 0, 26)
    print(len(t))
    print(len(df_new[0]))
    df_new[len(values)] = mahalanobis_distance
    #print(df.loc[df[0] == 699])
    #print(df.iloc[0])
    #df_new = pd.concat([pd.DataFrame(df.iloc[0]), df_new], ignore_index=True)
    df_new.to_csv(os.path.join(parent_dir, 'hkex_results.csv'), index = False, header=True)

if __name__ == '__main__':  
    main()
 