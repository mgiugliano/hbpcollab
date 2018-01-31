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
	u   = np.loadtxt('output.x', delimiter=' ')     # Load into memory the file output.x
	tsp = np.loadtxt('spikes.x')                    # Load into memory the file spikes.x
	Nspikes = len(tsp)                              # Compute the number of spikes
	plt.plot(u[:,0], u[:,1])                        # Make the actual plot versus time
	plt.xlim( (0,400) )                             # Set the horizontal limits
	plt.ylim( (-80,50) )                            # Set the vertical limits
	plt.xlabel('time [ms]')                         # Label for the horizontal axis
	plt.ylabel('u - membrane potential [mV]')       # Label for the vertical axis
	plt.title(str(1000*Nspikes/T) + ' Hz')          # Figure title
	plt.grid()                                      # "Grid" on
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
	plt.plot(I0range, F, 'o-', linewidth=3.0)		
	plt.xlim( (np.min(I0range),np.max(I0range)) )   # Set the horizontal limits
	#plt.ylim( (0,40) )                             # Set the vertical limits
	plt.xlabel('Mean input current [pA]')           # Label for the horizontal axis
	plt.ylabel('Mean Firing Rate [Hz]')             # Label for the vertical axis
	plt.grid()                                      # "Grid" on	
	plt.show()  
#---------------------------------------------------------------------------------------
	