# Program to generate Figure 2.

import numpy
from matplotlib import pyplot as plt 

# For good plots
plt.rc('font', family='STIXGeneral', size=16)
plt.rc('xtick')
plt.rc('ytick')

# Neuron class
class FHN:
	def __init__(self, params):
		self.eps = params[0]
		self.a = params[1]
		self.b = params[2]
		self.dt = params[3]
		self.T = params[4]
		self.A = params[5]
		self.f = params[6]
		self.sigma = params[7]
		self.v0 = params[8]
		self.w0 = params[9]
		self.phase = 0.0
		self.V = numpy.zeros(int(self.T/self.dt))
		self.W = numpy.zeros(int(self.T/self.dt))
		self.I = 0.0207
		self.tau = 0.0
					
	# Method for 4th order Runge-Kutta integration.
	def rk4(self):
		time = numpy.arange(0, self.T, self.dt)
		self.feed = (self.I)*numpy.heaviside(time - 0.01, 1)
		self.V[0] = self.v0
		self.W[0] = self.w0
		N = len(self.V)

		for i in range(N-1):
			k1 = (1/self.eps)*(self.V[i]*(self.V[i]-self.a)*(1-self.V[i])-self.W[i]+self.feed[i])
			j1 = (self.V[i]-self.W[i]-self.b)

			k2 = (1/self.eps)*((self.V[i] + self.dt*k1/2)*((self.V[i] + self.dt*k1/2)-self.a)*(1-(self.V[i] + self.dt*k1/2))-(self.W[i] + self.dt*j1/2)+self.feed[i])
			j2 = ((self.V[i] + self.dt*k1/2)-(self.W[i] + self.dt*j1/2)-self.b)

			k3 = (1/self.eps)*((self.V[i] + self.dt*k2/2)*((self.V[i] + self.dt*k2/2)-self.a)*(1-(self.V[i] + self.dt*k2/2))-(self.W[i] + self.dt*j2/2)+self.feed[i])
			j3 = ((self.V[i] + self.dt*k2/2)-(self.W[i] + self.dt*j2/2)-self.b)

			k4 = (1/self.eps)*((self.V[i] + self.dt*k3)*((self.V[i] + self.dt*k3)-self.a)*(1-(self.V[i] + self.dt*k3))-(self.W[i] + self.dt*j3)+self.feed[i])
			j4 = ((self.V[i] + self.dt*k3)-(self.W[i] + self.dt*j3)-self.b)

			self.V[i+1] = self.V[i] + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
			self.W[i+1] = self.W[i] + (self.dt/6)*(j1 + 2*j2 + 2*j3 + j4)

	
eps = 0.005
a = 0.5
b = 0.15
dt = 0.0001
T = 3.0
A = 0.0
f = 0.2
sigma = 0.00	
v0 = 0.11151
w0 = -0.03849

run = FHN([eps, a, b, dt, T, A, f, sigma, v0, w0])

time = numpy.arange(0, T, dt)

I = numpy.arange(0.0, 0.05, 0.0001)		# Range of rectangular current heights used.
v_max = numpy.zeros(len(I))

for i in range(len(I)):
	print(str(i+1)+'/'+str(len(I)))
	run.I = I[i]
	run.rk4()
	v_max[i] = numpy.amax(run.V)-v0


plt.plot(I, v_max, 'k-', linewidth=1.3)
plt.xlabel('I')
plt.ylabel('$v_{max}-v_{0}$')
plt.xlim(0, 0.05)
plt.ylim(0, 1.0)
plt.show()



