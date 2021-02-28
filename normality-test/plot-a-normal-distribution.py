#!/usr/bin/python3

# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np

# Create the vectors X and Y
x = np.arange(-3,3,0.2)
print(x)
y = []
y_cdf = []
#y = x ** 2

mu, sigma = 0, 1
pi = 3.1415926
e = 2.71828
left = 1 / (sigma * ((2 * pi) ** (1/2)))

i = 0
while i < len(x):
    y.append(left * (e ** (- 0.5 * ((x[i] - mu)/sigma) ** 2.0)))
    if i != 0:
        y_cdf.append(y_cdf[i-1] + y[i])
    else:
        y_cdf.append(y[i])
    i += 1

plt.title('Plot PDF and CDF of a normal distribution')
plt.plot(x,y)
plt.plot(x,y_cdf)
plt.legend(['Probability Density Function', 'Cumulative Distribution Function'])
# Show the plot
plt.show()
