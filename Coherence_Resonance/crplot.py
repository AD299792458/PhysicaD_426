# Program to generate Figure 8.
import numpy
from matplotlib import pyplot as plt 

# For good plots.
plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

# Import data.
pre = numpy.load('check_crsingle.npy')
post = numpy.load('check_crpost.npy')

# Noise intensities used.
sigma = numpy.arange(0.0025, 0.023, 0.0005)

# Scatter plots.
plt.scatter(sigma, pre, marker='o', edgecolor='green', facecolor='none', s=25, label='Presynaptic neuron')
plt.scatter(sigma, post, marker='s', edgecolor='red', facecolor='none', s=25, label='Postsynaptic neuron')

# Line plots.
plt.plot(sigma, pre, 'g-', linewidth=1.5)
plt.plot(sigma, post, 'r-', linewidth=1.0)

plt.xlabel('$\sigma$')
plt.ylabel('CV')
plt.xlim(0.002, 0.016)
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.legend()
plt.show()