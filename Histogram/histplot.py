import numpy
from matplotlib import pyplot as plt 
import matplotlib.animation as animation

plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

single = numpy.load('check_singlespikes.npy')
single = numpy.ediff1d(single)
post = numpy.load('check_postspikes.npy')
post = numpy.ediff1d(post)

b = numpy.arange(0, 12, 0.05)
h1, b1 = numpy.histogram(single, b)
plt.subplot(1, 2, 1)
plt.plot(b1[:-1], h1, 'g-', label='Presynaptic neuron', linewidth=2.0)
plt.xlabel('ISI')
plt.ylabel('Number of events')
plt.ylim(0, 830)
plt.xlim(0, 10.5)

h2, b2 = numpy.histogram(post, b)
plt.subplot(1, 2, 2)
plt.plot(b2[:-1], h2, 'r-', label='Postsynaptic neuron', linewidth=2.0)
plt.xlabel('ISI')
plt.ylabel('Number of events')
plt.ylim(0, 830)
plt.xlim(0, 10.5)

plt.subplots_adjust(wspace=0.4)
plt.show()