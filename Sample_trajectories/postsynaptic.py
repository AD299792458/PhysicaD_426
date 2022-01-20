# Program to generate Figure 5. Note that since this is a stochastic simulation, you will get a different plot. Figure 5 is only a representative illustration.

import numpy
from matplotlib import pyplot as plt 
import time

# For good plots.
plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

# Neuron class.
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
		self.N = int(params[10])
		self.height = params[11]
		self.width = params[12]
		self.sigma = params[13]
		self.p = params[14]		# This is a dummy parameter to be used in future work to simulate probabilistic firing.

	def RK4(self):
		#self.track = numpy.zeros((int(self.T/self.dt), self.N))
		#self.track[0] = self.v0		
		self.spktrain = numpy.zeros(int(self.T/self.dt))
		self.v = numpy.full(self.N, self.v0)
		self.w = numpy.full(self.N, self.w0)
		# Phase differences are introduced between the input current signals for presynaptic neurons.
		self.phase = numpy.random.normal(0, numpy.pi/4, self.N)
		vprevious = self.v
		# An array of flags is maintained to keep track of whether each presynaptic neuron crosses the lower threshold (v=0.2).
		flags = numpy.full(self.N, False)		
		for i in range(int(self.T/self.dt)-1):
			k1 = (1/self.eps)*(self.v*(self.v-self.a)*(1-self.v)-self.w+self.I+self.A*numpy.sin(2*numpy.pi*self.f*i*self.dt+self.phase))
			l1 = self.v-self.w-self.b
			k2 = (1/self.eps)*((self.v+self.dt*k1/2)*(self.v+self.dt*k1/2-self.a)*(1-self.v-self.dt*k1/2)-(self.w+self.dt*l1/2)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt/2)+self.phase))
			l2 = (self.v+self.dt*k1/2)-(self.w+self.dt*l1/2)-self.b
			k3 = (1/self.eps)*((self.v+self.dt*k2/2)*(self.v+self.dt*k2/2-self.a)*(1-self.v-self.dt*k2/2)-(self.w+self.dt*l2/2)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt/2)+self.phase))
			l3 = (self.v+self.dt*k2/2)-(self.w+self.dt*l2/2)-self.b
			k4 = (1/self.eps)*((self.v+self.dt*k3)*(self.v+self.dt*k3-self.a)*(1-self.v-self.dt*k3)-(self.w+self.dt*l3)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt)+self.phase))
			l4 = (self.v+self.dt*k3)-(self.w+self.dt*l3)-self.b

			self.v=self.v+(self.dt/6)*(k1+2*k2+2*k3+k4)
			self.w=self.w+(self.dt/6)*(l1+2*l2+2*l3+l4)
			#self.track[i+1] = self.v
			#Set corresponding flags to True whenever corresponding neuron crosses lower threshold
			flags = numpy.logical_or(flags, (numpy.heaviside(self.v-0.2, 1) - numpy.heaviside(vprevious-0.2, 1))==1)
			#Compute which neurons have crossed upper threshold
			activation = numpy.heaviside(self.v-0.9, 1) - numpy.heaviside(vprevious-0.9, 1)
			summ = numpy.sum(activation)
			#Locations of neurons crossing upper threshold 
			loc = numpy.where(activation>0)[0]
			#If some neuron fires, generate a pulse with height equal to number of neurons firing 
			if(summ>0):
				self.spktrain[i:i+self.width+1] = self.spktrain[i:i+self.width+1] + self.height*numpy.count_nonzero(flags[loc])
				flags[loc]=False
			vprevious = self.v

	def euler_maruyama(self):
		# TxN array to store trajectories of all presynaptic neurons.
		self.track = numpy.zeros((int(self.T/self.dt), self.N))
		self.track[0] = self.v0
		self.spktrain = numpy.zeros(int(self.T/self.dt))
		self.v = numpy.full(self.N, self.v0)
		self.w = numpy.full(self.N, self.w0)
		self.phase=0
		vprevious = self.v
		#Maintain an array of flags to track which presynaptic neuron has crossed lower threshold.
		flags = numpy.full(self.N, False)
		for i in range(int(self.T/self.dt)-1):	
			dW = numpy.random.normal(0, 1, self.N)
			self.v=self.v + (self.v*(self.v-self.a)*(1-self.v)-self.w+self.A*numpy.sin(2*numpy.pi*self.f*i*self.dt+self.phase))*self.dt/self.eps + (self.sigma/self.eps)*numpy.sqrt(self.dt)*dW
			self.w = self.w + (self.v - self.w - self.b)*self.dt
			self.track[i+1] = self.v
			#Set corresponding flags to True whenever corresponding neuron crosses lower threshold
			flags = numpy.logical_or(flags, (numpy.heaviside(self.v-0.2, 1) - numpy.heaviside(vprevious-0.2, 1))==1)
			#Compute which neurons have crossed upper threshold
			activation = numpy.heaviside(self.v-0.9, 1) - numpy.heaviside(vprevious-0.9, 1)
			#summ = numpy.sum(activation)
			#Locations of neurons crossing upper threshold 
			loc = numpy.where(activation>0)[0]
			#If some neuron fires, generate a pulse with height equal to number of neurons firing 
			if(len(loc)>0):
				self.spktrain[i:i+self.width+1] = self.spktrain[i:i+self.width+1] + self.height*numpy.count_nonzero(flags[loc])
				flags[loc]=False
			vprevious = self.v

			

class postFHN():
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
		self.synapse = numpy.zeros(len(self.v))		

	def RK4(self):
		self.v[0] = self.v0
		self.w[0] = self.w0
		self.spikes = numpy.empty(0)
		flag=False
		for i in range(len(self.v)-1):
			#if(i%1000==0):
			#	print(i*dt)		
			k1 = (1/eps)*(self.v[i]*(self.v[i]-a)*(1-self.v[i]) - self.w[i] + self.synapse[i])
			l1 = self.v[i] - self.w[i] - b
			k2 = (1/eps)*((self.v[i]+dt*k1/2)*(self.v[i]+dt*k1/2-a)*(1-self.v[i]-dt*k1/2)-(self.w[i]+dt*l1/2)+ self.synapse[i])
			l2 = (self.v[i] + dt*k1/2) - (self.w[i] + dt*l1/2) - b
			k3 = (1/eps)*((self.v[i]+dt*k2/2)*(self.v[i]+dt*k2/2-a)*(1-self.v[i]-dt*k2/2)-(self.w[i]+dt*l2/2)+ self.synapse[i])
			l3 = (self.v[i] + dt*k2/2) - (self.w[i] + dt*l2/2) - b
			k4 = (1/eps)*((self.v[i]+dt*k3)*(self.v[i]+dt*k3-a)*(1-self.v[i]-dt*k3)-(self.w[i]+dt*l3)+ self.synapse[i])
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
A = 0.03
f = 0.5
dt = 0.001
T = 10.0
t = numpy.arange(0, T, dt)
I = 0
# Initial conditions (Set to the stable fixed point).
v0 = 0.11151
w0 = -0.03849 
N = 5
A_thr = 0.027
N_min = 2
height = A_thr/N_min
width = int(0.3/dt)
sigma = 0.001	# Noise intensity.
p = 0
#T_stimulus = 1/f

neuron = FHN([eps, a, b, I, A, f, dt, T, v0, w0, N, height, width, sigma, p])	# Object for presynaptic neurons.
post = postFHN([eps, a, b, 0, dt, T, v0, w0])	# Object for postsynaptic neuron.
	
neuron.euler_maruyama()		# Simulate presynaptic neurons.
post.synapse = neuron.spktrain		# Supply dendritic current to postsynaptic neuron.
post.RK4()		# Simulate postsynaptic neuron.

print(len(post.spikes))		# Print the number of spikes of the postsynaptic neuron.

neuron.track = numpy.transpose(neuron.track)	# Transpose of the presynaptic trajectories matrix (for ease of plotting).
for i in range(N):
	plt.subplot((N+2), 1, i+1)
	plt.plot(t, neuron.track[i], 'b-', linewidth=1.0)	# Plot presynaptic voltage trajectory.
	plt.ylabel('v')
	plt.xlim(0,10)

plt.subplot((N+2), 1, N+1)
plt.plot(t, post.synapse, 'g-', linewidth=1.0)	# Plot synaptic current.
plt.ylabel('I', labelpad=-3)
plt.xlim(0,10)

plt.subplot((N+2), 1, N+2)
plt.plot(t, post.v, 'r-', linewidth=1.1)	# Plot postsynaptic voltage trajectory.
plt.ylim(-0.5, 1.5)
plt.xlim(0,10)
plt.subplots_adjust(hspace=0.9)
plt.xlabel('Time(s)', labelpad=-4)
plt.ylabel('v')
plt.show()
