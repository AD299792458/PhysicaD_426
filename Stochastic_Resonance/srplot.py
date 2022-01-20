# Program to generate Figure 7.
import numpy
from matplotlib import pyplot as plt 

# For good plots
plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

# Import data
pre = numpy.load('check_srsingle.npy')
post = numpy.load('check_srpost.npy')

# Noise intensities used.
sigma = numpy.arange(0.001, 0.006, 0.0003)

# Scatter plot.
plt.scatter(sigma, pre, marker='o', edgecolor='green', facecolor='none', s=30, label='Presynaptic')
plt.scatter(sigma, post, marker='s', edgecolor='red', facecolor='none', s=30, label='Postsynaptic')

# Line plot.
plt.plot(sigma, pre, 'g-', linewidth=2.0)
plt.plot(sigma, post, 'r-', linewidth=2.0)

plt.xlabel('$\sigma$')
plt.xlim(0.0009, 0.0061)
plt.xticks(numpy.arange(0.001, 0.007, 0.001))
plt.ylabel('ISIH peak around $T_{0}$', labelpad=1.2)
plt.legend(loc='upper right', prop={'size': 16})
plt.show()