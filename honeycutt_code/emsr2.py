# Code to generate data for Figure 6: Presynaptic neuron.

import numpy
from matplotlib import pyplot as plt 
import matplotlib.animation as animation

# For good plots.
plt.rc('font', family='STIXGeneral', size=14)
plt.rc('xtick')
plt.rc('ytick')

# Define class for neuron
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

	# Class method for 4th order Runge-Kutta integration (for deterministic simulations).
	def RK4(self):
		self.v[0] = self.v0
		self.w[0] = self.w0
		self.spikes = numpy.empty(0)	# Array holds spike times
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
			# Voltages crossing 0.9 are considered as spikes only if they have crossed 0.2 previously
			if(self.v[i]<0.2 and self.v[i+1]>0.2):
				flag=True
			if(self.v[i]<0.9 and self.v[i+1]>0.9 and flag==True):
				self.spikes = numpy.append(self.spikes, (i+1)*self.dt)
				flag=False

	# Class method for Euler-Maruyama stochastic integration (for stochastic simulations).
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

# FitzHugh-Nagumo model parameters
eps = 0.005
a = 0.5
b = 0.15
I = 0
A = 0.03
f = 0.5
dt = 0.0001
T = 10000.0
# Initial condition: At the fixed point.
v0 = 0.11151
w0 = -0.03849 
sigma = 0.003
#T_stimulus = 1/f
time = numpy.arange(0, T, dt)

neuron = FHN([eps, a, b, I, A, f, dt, T, v0, w0, sigma])
ensemble_size = 30
isi = numpy.empty(0)

for i in range(ensemble_size):
	print(i+1)
	neuron.euler_maruyama()
	isi = numpy.append(isi, numpy.ediff1d(neuron.spikes))

numpy.save('emsr2.npy', isi)


