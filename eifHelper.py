import numpy as np 
import matplotlib.pyplot as plt 
import subprocess as sb
import os
#
# Plot in a new figure the sample voltage trace of a EIF
# receiving a noisy current with sinusoidal modulation of
# the mean.
#
def plot_sampleV(T, I0, I1, F0, S, tau):
	cmdstr = "./eif " + str(T) + " " + str(I0) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 1"
	#!./eif 400 550 0 1 100 5 1
	return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program

	# Plot the result of the simulation
	i   = np.loadtxt('input.x', delimiter=' ')     # Load into memory the file output.x
	u   = np.loadtxt('output.x', delimiter=' ')     # Load into memory the file output.x
	tsp = np.loadtxt('spikes.x')                    # Load into memory the file spikes.x
	Nspikes = len(tsp)                              # Compute the number of spikes

	fig = plt.figure()
	ax1  = fig.add_subplot(211)	
	ax1.plot(u[:,0], u[:,1])                        # Make the actual plot versus time
	ax1.set_xlim( (0,400) )                             # Set the horizontal limits
	ax1.set_ylim( (-80,50) )                            # Set the vertical limits
	ax1.set_xlabel('time [ms]')                         # Label for the horizontal axis
	ax1.set_ylabel('u - membrane potential [mV]')       # Label for the vertical axis
	ax1.grid()                                        # "Grid" on
	ax1.set_aspect(aspect=1)

	ax2  = fig.add_subplot(212)	
	ax2.plot(i[:,0], i[:,1])                            # Make the actual plot versus time
	ax2.set_xlim( (0,400) )                             # Set the horizontal limits
	ax2.set_ylim( (-300,1000) )                         # Set the vertical limits
	ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
	ax2.set_ylabel('i - injected current [pA]')         # Label for the vertical axis
	ax2.grid()                                          # "Grid" on
	ax2.set_aspect(aspect=1)

	plt.show()
#---------------------------------------------------------------------------------------
def plot_FI(T, I0range, Srange):
	m  = np.size(I0range)
	n  = np.size(Srange)
	F  = np.zeros((m,n))

	I1 = 0.                  
	F0 = 0.                  
	tau= 5.

	myfile = 'spikes.x'

	for j in range(n):
		for i in range(m):
			I0= I0range[i]
			S = Srange[j]
			cmdstr = "./eif " + str(T) + " " + str(I0) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 0"
			return_code = sb.call(cmdstr, shell=True)
			if os.stat(myfile).st_size:
				tsp = np.loadtxt(myfile) 
				N   = np.size(tsp)
			else:
				N   = 0
			F[i,j] = 1000. * N / T;
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	ax.plot(I0range, F, 'o-', linewidth=3.0)		
	ax.set_xlim( (np.min(I0range),np.max(I0range)) )   # Set the horizontal limits
	#ax.set_ylim( (0,40) )                             # Set the vertical limits
	ax.set_xlabel('Mean input current [pA]')           # Label for the horizontal axis
	ax.set_ylabel('Mean Firing Rate [Hz]')             # Label for the vertical axis
	ax.grid()                                          # "Grid" on	
	plt.show()  
#---------------------------------------------------------------------------------------
	