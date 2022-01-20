# Data for Figure 7: Program to simulate stochastic resonance in presynaptic neurons.

import numpy
from matplotlib import pyplot as plt 
import matplotlib.animation as animation

# For good plots
plt.rc('font', family='STIXGeneral', size=14)
plt.rc('xtick')
plt.rc('ytick')

class FHN():
	def __init__(self, params):
		self.eps = params[0]
		self.a = params[1]
		self.b = params[2]
		self.I = params[3]
		self.A = params[4]
		self.f = params[5]
		self.dt = params[6]
		self.T = params[7]
		self.v0 = params[8]
		self.w0 = params[9]
		self.sigma = params[10]
		self.v = numpy.zeros(int(self.T/self.dt))
		self.w = numpy.zeros(int(self.T/self.dt))

	def RK4(self):
		self.v[0] = self.v0
		self.w[0] = self.w0
		self.spikes = numpy.empty(0)
		flag = False
		for i in range(len(self.v)-1):
			k1 = (1/self.eps)*(self.v[i]*(self.v[i]-self.a)*(1-self.v[i])-self.w[i]+self.I+self.A*numpy.sin(2*numpy.pi*self.f*i*self.dt))
			l1 = self.v[i]-self.w[i]-self.b
			k2 = (1/self.eps)*((self.v[i]+self.dt*k1/2)*(self.v[i]+self.dt*k1/2-self.a)*(1-self.v[i]-self.dt*k1/2)-(self.w[i]+self.dt*l1/2)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt/2)))
			l2 = (self.v[i]+self.dt*k1/2)-(self.w[i]+self.dt*l1/2)-self.b
			k3 = (1/self.eps)*((self.v[i]+self.dt*k2/2)*(self.v[i]+self.dt*k2/2-self.a)*(1-self.v[i]-self.dt*k2/2)-(self.w[i]+self.dt*l2/2)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt/2)))
			l3 = (self.v[i]+self.dt*k2/2)-(self.w[i]+self.dt*l2/2)-self.b
			k4 = (1/self.eps)*((self.v[i]+self.dt*k3)*(self.v[i]+self.dt*k3-self.a)*(1-self.v[i]-self.dt*k3)-(self.w[i]+self.dt*l3)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt)))
			l4 = (self.v[i]+self.dt*k3)-(self.w[i]+self.dt*l3)-self.b

			self.v[i+1]=self.v[i]+(self.dt/6)*(k1+2*k2+2*k3+k4)
			self.w[i+1]=self.w[i]+(self.dt/6)*(l1+2*l2+2*l3+l4)
			if(self.v[i]<0.2 and self.v[i+1]>0.2):
				flag=True
			if(self.v[i]<0.9 and self.v[i+1]>0.9 and flag==True):
				self.spikes = numpy.append(self.spikes, (i+1)*self.dt)
				flag=False

	def euler_maruyama(self):
		self.v[0] = self.v0
		self.w[0] = self.w0
		self.spikes = numpy.empty(0)
		flag = False
		for i in range(len(self.v)-1):
			self.v[i+1] = self.v[i] + (1/self.eps)*self.dt*(self.v[i]*(self.v[i]-self.a)*(1-self.v[i])-self.w[i]+self.I+self.A*numpy.sin(2*numpy.pi*self.f*i*self.dt)) + self.sigma*numpy.sqrt(self.dt)*numpy.random.normal(0, 1)/self.eps
			self.w[i+1] = self.w[i] + self.dt*(self.v[i] - self.w[i] - self.b)
			if(self.v[i]<0.2 and self.v[i+1]>0.2):
				flag=True
			if(self.v[i]<0.9 and self.v[i+1]>0.9 and flag==True):
				self.spikes = numpy.append(self.spikes, (i+1)*self.dt)
				flag=False


eps = 0.005
a = 0.5
b = 0.15
I = 0
A = 0.03
f = 0.5
dt = 0.001
T = 2000.0
v0 = 0.11151
w0 = -0.03849 

sigma = numpy.arange(0.001, 0.006, 0.0003)		# Noise intensities used in this simulation.
T_stimulus = 1/f
ensemble_size = 40

time = numpy.arange(0, T, dt)
neuron = FHN([eps, a, b, I, A, f, dt, T, v0, w0, 0])

# Histogram bins
bins = numpy.arange(0, 5.5*T_stimulus, 0.05)
loc1 = numpy.where(bins==T_stimulus)[0][0]

# Use these to track histogram peaks at 2 and 3 times the stimulus period
#loc2 = numpy.where(bins==2*T_stimulus)[0][0]
#loc3 = numpy.where(bins==3*T_stimulus)[0][0]

# Array to store histogram peaks corresponding to each noise intensity.
peak1 = numpy.zeros(len(sigma))

# Use these to track histogram peaks at 2 and 3 times the stimulus period
#peak2 = numpy.zeros(len(sigma))
#peak3 = numpy.zeros(len(sigma))

for i in range(len(sigma)):
	print(i+1)
	neuron.sigma = sigma[i]		# Set noise intensity for presynaptic neuron.
	isi = numpy.empty(0)	# Array to store interspike intervals.
	for j in range(ensemble_size):
		neuron.euler_maruyama()		# Simulate presynaptic neuron.
		isi = numpy.append(isi, numpy.ediff1d(neuron.spikes))	# Dump interspike intervals from simulation into array.
	h, b = numpy.histogram(isi, bins)		# Get histogram for the whole ensemble.
	peak1[i] = numpy.amax(h[loc1-8:loc1+8])		# Get the peak height around the stimulus frequency.
	
	# Use these to track histogram peaks at 2 and 3 times the stimulus period
	#peak2[i] = numpy.amax(h[loc2-7:loc2+7])
	#peak3[i] = numpy.amax(h[loc3-7:loc3+7])

numpy.save('check_srsingle.npy', peak1)		# Save data.

# Use these to track histogram peaks at 2 and 3 times the stimulus period
#numpy.save('peak2_s.npy', peak2)
#numpy.save('peak3_s.npy', peak3)

