#!/usr/bin/python3
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x = np.linspace(start=-1, stop=3, num=100)
y = np.linspace(start=0, stop=4, num=100)
# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

X, Y = np.meshgrid(x, y)

pos = np.dstack((X, Y))

mu = np.array([1, 2])

cov = np.array([[.5, .25],[.25, .5]])
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
fig.show()
plt.show()
