# Program to generate Figures 3 and 4: Threshold curves for excitatory and inhibitory input currents.

import numpy
from matplotlib import pyplot as plt 
import matplotlib.animation as animation

# For good plots.
plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

# Class for neuron.
class FHN():
	def __init__(self, params):
		self.eps = params[0]
		self.a = params[1]
		self.b = params[2]
		self.I = params[3]
		self.dt = params[4]
		self.T = params[5]
		self.v0 = params[6]
		self.w0 = params[7]
		self.v = numpy.zeros(int(self.T/self.dt))
		self.w = numpy.zeros(int(self.T/self.dt))
		self.synapse = numpy.zeros(len(self.v))		# Array for synaptic input current.		

	# Method for 4th order Runge-Kutta integration.
	def RK4(self):
		self.v[0] = self.v0
		self.w[0] = self.w0
		self.spikes = numpy.empty(0)
		flag=False
		for i in range(len(self.v)-1):
			k1 = (1/eps)*(self.v[i]*(self.v[i]-a)*(1-self.v[i]) - self.w[i] + self.I + self.synapse[i])
			l1 = self.v[i] - self.w[i] - b
			k2 = (1/eps)*((self.v[i]+dt*k1/2)*(self.v[i]+dt*k1/2-a)*(1-self.v[i]-dt*k1/2)-(self.w[i]+dt*l1/2)+ self.I + self.synapse[i])
			l2 = (self.v[i] + dt*k1/2) - (self.w[i] + dt*l1/2) - b
			k3 = (1/eps)*((self.v[i]+dt*k2/2)*(self.v[i]+dt*k2/2-a)*(1-self.v[i]-dt*k2/2)-(self.w[i]+dt*l2/2)+ self.I + self.synapse[i])
			l3 = (self.v[i] + dt*k2/2) - (self.w[i] + dt*l2/2) - b
			k4 = (1/eps)*((self.v[i]+dt*k3)*(self.v[i]+dt*k3-a)*(1-self.v[i]-dt*k3)-(self.w[i]+dt*l3)+ self.I + self.synapse[i])
			l4 = (self.v[i] + dt*k3) - (self.w[i] + dt*l3) - b

			self.v[i+1]=self.v[i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
			self.w[i+1]=self.w[i] + (dt/6)*(l1 + 2*l2 + 2*l3 + l4)
			if(self.v[i+1]>0.2 and self.v[i]<0.2):
				flag=True
			if(self.v[i+1]>0.9 and self.v[i]<0.9 and flag==True):
				self.spikes = numpy.append(self.spikes, (i+1)*dt)
				flag=False



eps = 0.005
a = 0.5
b = 0.15
I = 0
dt = 0.00005
T = 3.0
v0 = 0.11151
w0 = -0.03849 
sigma = 0

time = numpy.arange(0, T, dt)
# Initialize object for a neuron.
neuron = FHN([eps, a, b, I, dt, T, v0, w0])

A_T = 0.027		# This will be the maximum height of the staircase current. Set it to 0.027 for excitatory current, and -0.05 for inhibitory current (anodal break excitation).
tau = 0.3	# This will be the pulse width

# Array for time gaps between successive pulses comprising the staircase current.
mu = numpy.arange(0, 0.3, 0.001)
 
# For two pulses. 
N = 2
vmax = numpy.zeros(len(mu))
for j in range(len(mu)):
	print(str(j+1)+'/'+str(len(mu)))
	stimulus = numpy.zeros(len(time))	# Array for staircase current.
	for i in range(N):
		# Generate staircase current.
		stimulus = stimulus + (A_T/N)*(numpy.heaviside(time-0.01-i*mu[j], 1) - numpy.heaviside(time-0.01-i*mu[j]-tau, 1))

	neuron.synapse = stimulus	# Supply staircase current to neuron.
	neuron.RK4()	# Stimulate neuron.
	vmax[j] = numpy.amax(neuron.v)-v0	# Get maximum voltage attained from v0.
plt.plot(mu, vmax, 'g-', linewidth=1.2, label='$N_{min}=2$')	# Plot threshold curve.

# For three pulses.
N = 3
vmax = numpy.zeros(len(mu))
for j in range(len(mu)):
	print(str(j+1)+'/'+str(len(mu)))
	stimulus = numpy.zeros(len(time))	# Array for staircase current.
	for i in range(N):
		# Generate staircase current.
		stimulus = stimulus + (A_T/N)*(numpy.heaviside(time-0.01-i*mu[j], 1) - numpy.heaviside(time-0.01-i*mu[j]-tau, 1))

	neuron.synapse = stimulus	# Supply staircase current to neuron.
	neuron.RK4()	# Simulate neuron.
	vmax[j] = numpy.amax(neuron.v)-v0	# Get maximum voltage attained from v0.
plt.plot(mu, vmax, 'b-', linewidth=1.2, label='$N_{min}=3$')	# Plot threshold curve.

# For 4 pulses.
N = 4
vmax = numpy.zeros(len(mu))
for j in range(len(mu)):
	print(str(j+1)+'/'+str(len(mu)))
	stimulus = numpy.zeros(len(time))	# Array for staircase current.
	for i in range(N):
		# Generate staircase current.
		stimulus = stimulus + (A_T/N)*(numpy.heaviside(time-0.01-i*mu[j], 1) - numpy.heaviside(time-0.01-i*mu[j]-tau, 1))

	neuron.synapse = stimulus	# Supply staircase current to neuron.
	neuron.RK4()
	vmax[j] = numpy.amax(neuron.v)-v0	# Get maximum voltage attained from v0.
plt.plot(mu, vmax, 'r-', linewidth=1.2, label='$N_{min}=4$')	# Plot threshold curve.

# For 5 pulses.
N = 5
vmax = numpy.zeros(len(mu))
for j in range(len(mu)):
	print(str(j+1)+'/'+str(len(mu)))
	stimulus = numpy.zeros(len(time))	# Array for staircase current.
	for i in range(N):
		# Generate staircase current.
		stimulus = stimulus + (A_T/N)*(numpy.heaviside(time-0.01-i*mu[j], 1) - numpy.heaviside(time-0.01-i*mu[j]-tau, 1))

	neuron.synapse = stimulus	# Supply staircase current to neuron.
	neuron.RK4()
	vmax[j] = numpy.amax(neuron.v)-v0	# Get maximum voltage attained from v0.
plt.plot(mu, vmax, 'm-', linewidth=1.2, label='$N_{min}=5$')	# Plot threshold curve.
 

plt.xlabel('$\mu$')
plt.xlim(0,0.3)
plt.ylabel('$v_{max}-v_{0}$')
plt.legend(prop={'size': 14})
plt.show()

















