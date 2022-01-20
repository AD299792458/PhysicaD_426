import numpy
from matplotlib import pyplot as plt 
plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

##################################################STOCHASTIC RESONANCE#############################################

# dt = 0.001
# A=0.03,f=0.5,sigma=0.001,T=10000,30 runs
 
sr1 = numpy.load('honeycutt_sr1.npy')
cr1 = numpy.load('honeycutt_cr1.npy')
print(sr1)
print(cr1)

f = 0.5
T_stimulus = 1/f
h1, b1 = numpy.histogram(sr1, numpy.arange(0, 5.5*T_stimulus, 0.05), density=True)
plt.figure()
plt.plot(b1[:-1], h1, 'g-', linewidth=2.2)
plt.xlabel('ISI (s)')
plt.ylabel('Probability density')

h2, b2 = numpy.histogram(cr1, numpy.arange(0, 7, 0.05), density=True)
plt.figure()
plt.plot(b2[:-1], h2, 'g-', linewidth=2.2)
plt.xlabel('ISI (s)')
plt.ylabel('Probability density')

plt.show()

