import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
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
	#ax1.set_aspect(aspect=1)

	ax2  = fig.add_subplot(212)	
	ax2.plot(i[:,0], i[:,1])                            # Make the actual plot versus time
	ax2.set_xlim( (0,400) )                             # Set the horizontal limits
	ax2.set_ylim( (-300,1000) )                         # Set the vertical limits
	ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
	ax2.set_ylabel('i - injected current [pA]')         # Label for the vertical axis
	ax2.grid()                                          # "Grid" on
	#ax2.set_aspect(aspect=1)

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
def plot_FI_and_sample(T, I0range, S, M):
	m  = np.size(I0range)
	F  = np.zeros((m,1))

	I1 = 0.                  
	F0 = 0.                  
	tau= 5.

	myfile = 'spikes.x'

	for i in range(m):
		I0= I0range[i]
		cmdstr = "./eif " + str(T) + " " + str(I0) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 0"
		return_code = sb.call(cmdstr, shell=True)
		if os.stat(myfile).st_size:
			tsp = np.loadtxt(myfile) 
			N   = np.size(tsp)
		else:
			N   = 0
		F[i,0] = 1000. * N / T;

	cmdstr = "./eif " + str(400.) + " " + str(M) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 1"
	return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program
	u   = np.loadtxt('output.x', delimiter=' ')     # Load into memory the file output.x

	cmdstr = "./eif " + str(T) + " " + str(M) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 1"
	return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program
	if os.stat(myfile).st_size:
		tsp = np.loadtxt(myfile) 
		N   = np.size(tsp)
	else:
		N   = 0

	fig = plt.figure(figsize=(14,4))
	ax1 = fig.add_subplot(121)	
	ax1.plot(I0range, F, 'o-', linewidth=3.0)		
	ax1.set_xlim( (np.min(I0range),np.max(I0range)) )   # Set the horizontal limits
	#ax1.set_ylim( (0,40) )                             # Set the vertical limits
	ax1.set_xlabel('Mean input current [pA]')           # Label for the horizontal axis
	ax1.set_ylabel('Mean Firing Rate [Hz]')             # Label for the vertical axis
	ax1.grid()                                          # "Grid" on	

	p = [M,1000. * N / T]
	xmin, xmax = ax1.get_xbound()
	ymin, ymax = ax1.get_ybound()
	l1 = mlines.Line2D([p[0],p[0]], [ymin,p[1]])
	l2 = mlines.Line2D([xmin,p[0]], [p[1],p[1]])
	ax1.add_line(l1)
	ax1.add_line(l2)

	ax2 = fig.add_subplot(122)	
	ax2.plot(u[:,0], u[:,1])                        # Make the actual plot versus time
	ax2.set_xlim( (0,400) )                             # Set the horizontal limits
	ax2.set_ylim( (-80,50) )                            # Set the vertical limits
	ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
	ax2.set_ylabel('u - membrane potential [mV]')       # Label for the vertical axis
	ax2.grid()                                        # "Grid" on

	plt.show()  
#---------------------------------------------------------------------------------------
def plot_FI_and_modulation(T, I0range, S, M, M1, FF):

	m  = np.size(I0range)
	F  = np.zeros((m,1))
	I1 = 0.                  
	F0 = 0.                  
	tau= 5.

	myfile = 'spikes.x'

	for i in range(m):
		I0= I0range[i]
		cmdstr = "./eif " + str(T) + " " + str(I0) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + 	" 0"
		return_code = sb.call(cmdstr, shell=True)
		if os.stat(myfile).st_size:
			tsp = np.loadtxt(myfile) 
			N   = np.size(tsp)
		else:
			N   = 0
		F[i,0] = 1000. * N / T;
	
	I1 = 0.                  
	F0 = 0.                  

	cmdstr = "./eif " + str(T) + " " + str(M) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 0"
	return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program
	if os.stat(myfile).st_size:
		tsp = np.loadtxt(myfile) 
		N   = np.size(tsp)
	else:
		N   = 0
	
	cmdstr = "./eif " + str(T) + " " + str(M*1.1) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + 	" 0"
	return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program
	if os.stat(myfile).st_size:
		tsp = np.loadtxt(myfile) 
		N1   = np.size(tsp)
	else:
		N1   = 0

	I1 = M1                  
	F0 = FF                  
	cmdstr = "./eif " + str(400.) + " " + str(M) + " " + str(I1) + " " + str(F0) + " " + str(S) + " " + str(tau) + " 	1"
	return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program
	u   = np.loadtxt('output.x', delimiter=' ')     # Load into memory the file output.x
	
	        
	fig = plt.figure(figsize=(14,4))
	ax1 = fig.add_subplot(121)	
	ax1.plot(I0range, F, 'o-', linewidth=3.0)		
	ax1.set_xlim( (np.min(I0range),np.max(I0range)) )   # Set the horizontal limits
	#ax1.set_ylim( (0,40) )                             # Set the vertical limits
	ax1.set_xlabel('Mean input current [pA]')           # Label for the horizontal axis
	ax1.set_ylabel('Mean Firing Rate [Hz]')             # Label for the vertical axis
	ax1.grid()                                          # "Grid" on	
	
	p = [M,1000. * N / T]
	xmin, xmax = ax1.get_xbound()
	ymin, ymax = ax1.get_ybound()
	l1 = mlines.Line2D([p[0],p[0]], [ymin,p[1]])
	l2 = mlines.Line2D([xmin,p[0]], [p[1],p[1]])
	ax1.add_line(l1)
	ax1.add_line(l2)
	
	alpha = 1000. * (N1-N) / (T * 0.1*M);
	t = np.arange(ymin, p[1], (p[1]-ymin)*0.005)
	i = M + M1 * np.cos(6.28*FF*t*0.02)
	ax1.plot(i,t)
	t = np.arange(xmin, p[0], (p[0]-xmin)*0.005)
	f = p[1] + alpha * M1 * np.cos(6.28*FF*t*0.002)
	ax1.plot(t,f)
	
	ax2 = fig.add_subplot(122)	
	ax2.plot(u[:,0], u[:,1])                        # Make the actual plot versus time
	tmp1 = -60 + 10 * np.cos(6.28*FF*u[:,0])
	ax2.plot(u[:,0], tmp1)
	ax2.set_xlim( (0,400) )                             # Set the horizontal limits
	ax2.set_ylim( (-80,50) )                            # Set the vertical limits
	ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
	ax2.set_ylabel('u - membrane potential [mV]')       # Label for the vertical axis
	ax2.grid()                                        # "Grid" on
	
	plt.show() 
#---------------------------------------------------------------------------------------
def plot_histogram(T, I0, S, I1, F):
    #F = 60          # Hz
    #T  = 200000.   # ms

    #I0 = 500.      # pA
    #S  = 300.      # pA
    #I1 = 80.       # pA
    tau= 5.        # ms
    
    myfile = 'spikes.x'
    cmdstr = "./eif " + str(T) + " " + str(I0) + " " + str(I1) + " " + str(F) + " " + str(S) + " " + str(tau) + " 	0"
    return_code = sb.call(cmdstr, shell=True)       # Launch the call to the external program
    if os.stat(myfile).st_size:
        tsp = np.loadtxt(myfile) 
        N   = np.size(tsp)
    else:
        N   = 0

    x   = np.cos(6.28318530718 * tsp * F / 1000.)/N
    y   = np.sin(6.28318530718 * tsp * F / 1000.)/N

    R0  = 1000. * N / T   # spikes/s
    R1  = 2*R0*np.absolute(np.complex(np.sum(x), np.sum(y)))
    PHI = np.angle(np.complex(np.sum(x), np.sum(y)), deg=False)
    tsp = np.remainder(tsp,  2 * 1000./F)
    C   = np.floor_divide(T, 2 * 1000./F)

    fig = plt.figure(figsize=(14,4))

    hist, bins = np.histogram(tsp, bins=40)
    hist = 1000. * hist / ((bins[3] - bins[2]) * C)   # Hz
    # R1 = (np.max(hist) - np.min(hist))/2

    W = (2. * 1000./F)/40.
    plt.bar(bins[:-1], hist, width=W, color='black')

    y = R0 + R1 * np.cos(6.28318530718 * bins * F / 1000. - PHI)
    z = R0 + R1 * np.cos(6.28318530718 * bins * F / 1000.)
    plt.plot(bins, y, 'r', linewidth=1)
    plt.plot(bins, z, 'g--', linewidth=1)

    ax = fig.add_subplot(111)
    ax.set_xlim( (0, 2 * 1000./F - (bins[3] - bins[2])) )                         # Set the horizontal limits
    # ax.set_ylim( (0, 2*R0) )                         # Set the horizontal limits
    ax.set_xlabel('time within two periods [ms]')           # Label for the horizontal axis
    ax.set_ylabel('Instantaneous firing rate [spikes/s]')   # Label for the vertical axis
    #ax.grid()                                               # "Grid" on

    plt.show()
#---------------------------------------------------------------------------------------
def plot_transferfunction(T, I0, S, I1, FRange):
    tau = 5
    myfile = "spikes.x"
    transfervec = np.zeros((len(FRange), 1))  # Create vector to store all transfer factor
    phasevec = np.zeros((len(FRange), 1))  # Create vector to store all the phase delay values
    counter = 0
    for freq in FRange:
        cmdstr = "./eif " + str(T) + " " + str(I0) + " " + str(I1) + " " + str(freq) + " " + str(S) + " " + str(tau) + " 	0"
        return_code = sb.call(cmdstr, shell=True)  # Launch the call to the external program
        if os.stat(myfile).st_size:
            tsp = np.loadtxt(myfile)
            N = np.size(tsp)
        else:
            N = 0

        x = np.cos(6.28318530718 * tsp * freq / 1000.) / N
        y = np.sin(6.28318530718 * tsp * freq / 1000.) / N
        R0 = 1000. * N / T  # spikes/s
        transfervec[counter] = np.absolute(np.complex(np.sum(x), np.sum(y)))
        phasevec[counter] = -np.angle(np.complex(np.sum(x), np.sum(y)), deg=False)
        counter += 1
    transfervec /= transfervec[0]  # Normalizing the transfer curve to the first value
    fig = plt.figure(figsize=(14,4))
    plt.subplot(211)
    plt.loglog(FRange, transfervec)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transfer factor')
    plt.subplot(212)
    plt.semilogx(FRange, phasevec)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase delay [Rad]')
    plt.show()
#---------------------------------------------------------------------------------------