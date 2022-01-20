
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
	def SRK4(self):
		self.v[0] = self.v0
		self.w[0] = self.w0
		self.spikes = numpy.empty(0)	# Array holds spike times
		flag = False
		for i in range(len(self.v)-1):
			psi = numpy.random.normal(0, 1)
			F1 = (1/self.eps)*(self.v[i]*(self.v[i]-self.a)*(1-self.v[i])-self.w[i]+self.I+self.A*numpy.sin(2*numpy.pi*self.f*i*self.dt))
			J1 = self.v[i]-self.w[i]-self.b
			F2 = (1/self.eps)*((self.v[i]+self.dt*F1+self.sigma*numpy.sqrt(self.dt)*psi/self.eps)*((self.v[i]+self.dt*F1+self.sigma*numpy.sqrt(self.dt)*psi/self.eps)-self.a)*(1-(self.v[i]+self.dt*F1+self.sigma*numpy.sqrt(self.dt)*psi/self.eps))-(self.w[i]+self.dt*J1)+self.I+self.A*numpy.sin(2*numpy.pi*self.f*(i*self.dt+self.dt)))
			J2 = (self.v[i]+self.dt*F1+self.sigma*numpy.sqrt(self.dt)*psi/self.eps) - (self.w[i] + self.dt*J1) - self.b
			self.v[i+1] = self.v[i] + (self.dt/2)*(F1 + F2) + self.sigma*numpy.sqrt(self.dt)*psi/self.eps
			self.w[i+1] = self.w[i] + (self.dt/2)*(J1 + J2)
			# Voltages crossing 0.9 are considered as spikes only if they have crossed 0.2 previously
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
dt = 0.001
T = 10000.0
# Initial condition: At the fixed point.
v0 = 0.11151
w0 = -0.03849 
sigma = 0.001
#T_stimulus = 1/f
time = numpy.arange(0, T, dt)

neuron = FHN([eps, a, b, I, A, f, dt, T, v0, w0, sigma])
ensemble_size = 30
isi = numpy.empty(0)

for i in range(ensemble_size):
	print(i+1)
	neuron.SRK4()
	isi = numpy.append(isi, numpy.ediff1d(neuron.spikes))

numpy.save('honeycutt_sr1.npy', isi)
