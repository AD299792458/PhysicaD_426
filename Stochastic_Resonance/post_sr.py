# Data for Figure 7: Program to simulate stochastic resonance in postsynaptic neurons.
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
		self.N = int(params[10])
		self.height = params[11]
		self.width = params[12]
		self.sigma = params[13]
		self.p = params[14]	# Dummy parameter for future work with probabilistic firing or presynaptic neurons.

	def RK4(self):
		self.spktrain = numpy.zeros(int(self.T/self.dt))
		self.v = numpy.full(self.N, self.v0)
		self.w = numpy.full(self.N, self.w0)
		self.phase = numpy.random.normal(0, numpy.pi/4, self.N)
		vprevious = self.v
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
			#Set corresponding flags to True whenever corresponding neuron crosses lower threshold
			flags = numpy.logical_or(flags, (numpy.heaviside(self.v-0.2, 1) - numpy.heaviside(vprevious-0.2, 1))==1)
			#Compute which neurons have crossed upper threshold
			activation = numpy.heaviside(self.v-0.9, 1) - numpy.heaviside(vprevious-0.9, 1)
			summ = numpy.sum(activation)
			#Locations of neurons crossing upper threshold 
			loc = numpy.where(activation>0)[0]
			#If some neuron fires, generate a pulse with height equal to number of neurons firing 
			#if(len(loc)>0):
			if(summ>0):
				self.spktrain[i:i+self.width+1] = self.spktrain[i:i+self.width+1] + self.height*numpy.count_nonzero(flags[loc])
				flags[loc]=False
			vprevious = self.v

	def euler_maruyama(self):
		#self.track = numpy.zeros((int(self.T/self.dt), self.N))
		#self.track[0] = self.v0
		self.spktrain = numpy.zeros(int(self.T/self.dt))	# Array to store postsynaptic input current.
		self.v = numpy.full(self.N, self.v0)
		self.w = numpy.full(self.N, self.w0)
		self.phase = numpy.random.normal(0, numpy.pi/4, self.N)	# This is not used now. But can be used to include phase differences in the stimuli for presynaptic neurons.
		vprevious = self.v 		# Variable to store previous state.
		flags = numpy.full(self.N, False)	# Array of flags to keep track of whether presynaptic neurons have crossed lower threshold.

		for i in range(int(self.T/self.dt)-1):
			dW = numpy.random.normal(0, 1, self.N)
			self.v=self.v + (self.v*(self.v-self.a)*(1-self.v)-self.w+self.I+self.A*numpy.sin(2*numpy.pi*self.f*i*self.dt))*self.dt/self.eps + (self.sigma/self.eps)*numpy.sqrt(self.dt)*dW
			self.w = self.w + (self.v - self.w - self.b)*self.dt
			#self.track[i+1] = self.v
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
			k1 = (1/self.eps)*(self.v[i]*(self.v[i]-self.a)*(1-self.v[i]) - self.w[i] + self.I + self.synapse[i])
			l1 = self.v[i] - self.w[i] - self.b
			k2 = (1/self.eps)*((self.v[i]+self.dt*k1/2)*(self.v[i]+self.dt*k1/2-self.a)*(1-self.v[i]-self.dt*k1/2)-(self.w[i]+self.dt*l1/2)+ self.I + self.synapse[i])
			l2 = (self.v[i] + dt*k1/2) - (self.w[i] + dt*l1/2) - self.b
			k3 = (1/self.eps)*((self.v[i]+self.dt*k2/2)*(self.v[i]+self.dt*k2/2-self.a)*(1-self.v[i]-self.dt*k2/2)-(self.w[i]+self.dt*l2/2)+ self.I + self.synapse[i])
			l3 = (self.v[i] + self.dt*k2/2) - (self.w[i] + self.dt*l2/2) - self.b
			k4 = (1/self.eps)*((self.v[i]+self.dt*k3)*(self.v[i]+self.dt*k3-self.a)*(1-self.v[i]-self.dt*k3)-(self.w[i]+self.dt*l3)+ self.I + self.synapse[i])
			l4 = (self.v[i] + self.dt*k3) - (self.w[i] + self.dt*l3) - self.b

			self.v[i+1]=self.v[i] + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
			self.w[i+1]=self.w[i] + (self.dt/6)*(l1 + 2*l2 + 2*l3 + l4)
			if(self.v[i+1]>0.2 and self.v[i]<0.2):
				flag=True
			if(self.v[i+1]>0.9 and self.v[i]<0.9 and flag==True):
				self.spikes = numpy.append(self.spikes, (i+1)*self.dt)
				flag=False


# FitzHugh-Nagumo model parameters.
eps = 0.005
a = 0.5
b = 0.15
I = 0
A = 0.03
f =0.5		# Stimulus frequency.
dt = 0.001
T = 2000.0
v0 = 0.11151
w0 = -0.03849 
N = 5
A_thr = 0.035
N_min = 2
height = A_thr/N_min
width = int(0.3/dt)
sigma = numpy.arange(0.001, 0.006, 0.0003)	# Noise range used in this simulation
ensemble_size = 40
p = 0
T_stimulus = 1/f 	# Stimulus time period.
t = numpy.arange(0, T, dt)

neuron = FHN([eps, a, b, I, A, f, dt, T, v0, w0, N, height, width, 0, p])	# Initialize object for presynaptic neurons.
post = postFHN([eps, a, b, I, dt, T, v0, w0])		# Initialize object for postsynaptic neuron.

# Histogram bins
bins = numpy.arange(0, 5.5*T_stimulus, 0.05)
loc1 = numpy.where(bins==T_stimulus)[0][0]	# Location at stimulus time-period

# Use these to track histogram peaks at 2 and 3 times the stimulus period
#loc2 = numpy.where(bins==2*T_stimulus)[0][0]
#loc3 = numpy.where(bins==3*T_stimulus)[0][0]

# Array to store histogram peaks corresponding to each noise intensity.
peak1 = numpy.zeros(len(sigma))

# Use these to track histogram peaks at 2 and 3 times the stimulus period.
#peak2 = numpy.zeros(len(sigma))
#peak3 = numpy.zeros(len(sigma))

for i in range(len(sigma)):
	print(i+1)
	neuron.sigma = sigma[i]	# Set noise intensity for presynaptic neurons.
	isi = numpy.empty(0)	# This array will store the inter-spike intervals of the postsynaptic neuron.
	for j in range(ensemble_size):
		neuron.euler_maruyama()		# Simulate presynaptic neurons.
		post.synapse = neuron.spktrain		# Supply dendritic current to postsynaptic neuron.
		post.RK4()		# Simulate postsynaptic neuron.
		isi = numpy.append(isi, numpy.ediff1d(post.spikes))		# Dump interspike intervals from simulation into array.
	h, b = numpy.histogram(isi, bins)	# Get the histogram for the whole ensemble.
	peak1[i] = numpy.amax(h[loc1-8:loc1+8])		# Get the peak height around stimulus frequency.

	# Use these to track histogram peaks at 2 and 3 times the stimulus period.
	#peak2[i] = numpy.amax(h[loc2-7:loc2+7])
	#peak3[i] = numpy.amax(h[loc3-7:loc3+7])

numpy.save('check_srpost.npy', peak1)		# Save data into npy file.

# Use these to track histogram peaks at 2 and 3 times the stimulus period.
#numpy.save('peak2_l.npy', peak2)
#numpy.save('peak3_l.npy', peak3)
