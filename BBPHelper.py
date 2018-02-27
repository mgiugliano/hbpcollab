import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import subprocess as sb
import os
from neuron import h
import run as rn
import random
#
# Plot in a new figure the sample voltage trace of a EIF
# receiving a noisy current with sinusoidal modulation of
# the mean.
#
def load_model():
    #h.cvode.cache_efficient(0)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("constants.hoc")
    h.load_file("morphology.hoc")
    cell = rn.create_cell(add_synapses=0)
    return cell

def attach_noise_sin_clamp(cell, delay, dur, offset, amp, freq, dt, tau, sigma, mu, loc, seed):
    """Attach a sinusoidal current Clamp to a cell.

    :param cell: Cell object to attach the current clamp.
    :param delay: Onset of the injected current.
    :param dur: Duration of the stimulus.
    :param offset: Offset of the sine.
    :param amp: The amplitude of the sine.
    :param freq: The frequency of the sine.
    :param sigma: The standard deviation of the normrand.
    :param loc: Location on the dendrite where the stimulus is placed.
    """
    stim = h.IClampNoiseSin(cell.soma[0](loc))
    stim.delay = delay
    stim.dur = dur
    stim.std = sigma
    stim.offset = offset
    stim.amp = amp
    stim.freq = freq
    stim.dt = dt
    stim.tau = tau
    stim.mu = mu
    stim.new_seed = seed
    return stim

def run_model(dt, T):
    h.dt = dt
    h.tstop = T
    h.v_init = -70
    h.celsius = 37
    h.run()

def plot_sampleV(cell, T, I0, I1, F0, S, tau):
    delay = 0.
    dt = 0.05
    mu = 0
    loc = 0.5
    seed = random.randint(0, 1000)
    stim = attach_noise_sin_clamp(cell, delay, T, I0, I1, F0, dt, tau, S, mu, loc, seed)

    somav = h.Vector()
    somav.record(cell.soma[0](0.5)._ref_v)
    timevec = h.Vector()
    timevec.record(h._ref_t)
    iinj = h.Vector()
    iinj.record(stim._ref_i)

    run_model(dt, T)

    fig = plt.figure()
    ax1  = fig.add_subplot(211)
    ax1.plot(timevec, somav)                        # Make the actual plot versus time
    ax1.set_xlim( (0,400) )                             # Set the horizontal limits
    ax1.set_ylim( (-80,50) )                            # Set the vertical limits
    ax1.set_xlabel('time [ms]')                         # Label for the horizontal axis
    ax1.set_ylabel('u - membrane potential [mV]')       # Label for the vertical axis
    ax1.grid()                                        # "Grid" on
    #ax1.set_aspect(aspect=1)

    ax2  = fig.add_subplot(212)
    ax2.plot(timevec, iinj)                            # Make the actual plot versus time
    ax2.set_xlim( (0,400) )                             # Set the horizontal limits
    ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
    ax2.set_ylabel('i - injected current [pA]')         # Label for the vertical axis
    ax2.grid()                                          # "Grid" on
    #ax2.set_aspect(aspect=1)

    plt.show()
#---------------------------------------------------------------------------------------
def plot_FI(cell, T, I0range, Srange):
    delay = 0
    dt = 0.05
    mu = 0
    loc = 0.5

    m  = np.size(I0range)
    n  = np.size(Srange)
    F  = np.zeros((m,n))

    I1 = 0.
    F0 = 0.
    tau= 5.

    for j in range(n):
        for i in range(m):
            seed = random.randint(0, 1000)
            I0= I0range[i]
            S = Srange[j]
            stim = attach_noise_sin_clamp(cell, delay, T, I0, I1, F0, dt, tau, S, mu, loc, seed)

            counts = h.Vector()
            apc = h.APCount(cell.soma[0](0.5))
            apc.thresh = 0
            apc.record(counts)
            run_model(dt, T)

            N = len(counts)

            F[i,j] = 1000. * N / T

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
def plot_FI_and_sample(cell, T, I0range, S, M):
    m  = np.size(I0range)
    F  = np.zeros((m,1))

    delay = 0
    dt = 0.05
    mu = 0
    loc = 0.5
    I1 = 0.
    F0 = 0.
    tau= 5.

    for i in range(m):
        I0= I0range[i]
        seed = random.randint(0, 1000)
        I0 = I0range[i]
        stim = attach_noise_sin_clamp(cell, delay, T, I0, I1, F0, dt, tau, S, mu, loc, seed)

        counts = h.Vector()
        apc = h.APCount(cell.soma[0](0.5))
        apc.thresh = 0
        apc.record(counts)
        run_model(dt, T)
        N = len(counts)

        F[i, 0] = 1000. * N / T

    seed = random.randint(0, 1000)
    stim = attach_noise_sin_clamp(cell, delay, T, M, I1, F0, dt, tau, S, mu, loc, seed)

    somav = h.Vector()
    somav.record(cell.soma[0](0.5)._ref_v)
    timevec = h.Vector()
    timevec.record(h._ref_t)
    iinj = h.Vector()
    iinj.record(stim._ref_i)
    counts = h.Vector()
    apc = h.APCount(cell.soma[0](0.5))
    apc.thresh = 0
    apc.record(counts)

    run_model(dt, 400)

    N = len(counts)
    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(I0range, F, 'o-', linewidth=3.0)
    ax1.set_xlim( (np.min(I0range),np.max(I0range)) )   # Set the horizontal limits
    #ax1.set_ylim( (0,40) )                             # Set the vertical limits
    ax1.set_xlabel('Mean input current [pA]')           # Label for the horizontal axis
    ax1.set_ylabel('Mean Firing Rate [Hz]')             # Label for the vertical axis
    ax1.grid()                                          # "Grid" on

    p = [M,1000. * N / T]
    indthres = np.nonzero(I0range > M)
    crossind = indthres[0]
    rico = (F[crossind[0]]-F[crossind[0]-1])/(I0range[crossind[0]]-I0range[crossind[0]-1])
    p[1] = rico*(M-I0range[crossind[0]-1]) + F[crossind[0]-1]
    xmin, xmax = ax1.get_xbound()
    ymin, ymax = ax1.get_ybound()
    l1 = mlines.Line2D([p[0],p[0]], [ymin,p[1]])
    l2 = mlines.Line2D([xmin,p[0]], [p[1],p[1]])
    ax1.add_line(l1)
    ax1.add_line(l2)

    ax2 = fig.add_subplot(122)
    ax2.plot(timevec, somav)                        # Make the actual plot versus time
    ax2.set_xlim( (0,400) )                             # Set the horizontal limits
    ax2.set_ylim( (-80,50) )                            # Set the vertical limits
    ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
    ax2.set_ylabel('u - membrane potential [mV]')       # Label for the vertical axis
    ax2.grid()                                        # "Grid" on

    plt.show()
#---------------------------------------------------------------------------------------
def plot_FI_and_modulation(cell, T, I0range, S, M, M1, FF):

    m  = np.size(I0range)
    F  = np.zeros((m,1))
    I1 = 0.
    F0 = 0.
    tau= 5.
    FF = FF/1000
    delay = 0
    dt = 0.05
    mu = 0
    loc = 0.5

    for i in range(m):
        I0= I0range[i]
        seed = random.randint(0, 1000)
        I0 = I0range[i]
        stim = attach_noise_sin_clamp(cell, delay, T, I0, I1, F0, dt, tau, S, mu, loc, seed)

        counts = h.Vector()
        apc = h.APCount(cell.soma[0](0.5))
        apc.thresh = 0
        apc.record(counts)
        run_model(dt, T)

        N = len(counts)

        F[i,0] = 1000. * N / T

    I1 = 0.
    F0 = 0.

    seed = random.randint(0, 1000)
    stim = attach_noise_sin_clamp(cell, delay, T, M, I1, F0, dt, tau, S, mu, loc, seed)

    counts = h.Vector()
    apc = h.APCount(cell.soma[0](0.5))
    apc.thresh = 0
    apc.record(counts)
    run_model(dt, T)

    N = len(counts)

    stim = attach_noise_sin_clamp(cell, delay, T, M*1.1, I1, F0, dt, tau, S, mu, loc, seed)

    counts = h.Vector()
    apc = h.APCount(cell.soma[0](0.5))
    apc.thresh = 0
    apc.record(counts)
    run_model(dt, T)

    N1 = len(counts)

    I1 = M1
    F0 = FF

    seed = random.randint(0, 1000)
    stim = attach_noise_sin_clamp(cell, delay, T, M, I1, F0, dt, tau, S, mu, loc, seed)

    counts = h.Vector()
    apc = h.APCount(cell.soma[0](0.5))
    apc.thresh = 0
    apc.record(counts)
    somav = h.Vector()
    somav.record(cell.soma[0](0.5)._ref_v)
    timevec = h.Vector()
    timevec.record(h._ref_t)
    iinj = h.Vector()
    iinj.record(stim._ref_i)
    run_model(dt, T)


    fig = plt.figure(figsize=(14,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(I0range, F, 'o-', linewidth=3.0)
    ax1.set_xlim( (np.min(I0range),np.max(I0range)) )   # Set the horizontal limits
    #ax1.set_ylim( (0,40) )                             # Set the vertical limits
    ax1.set_xlabel('Mean input current [pA]')           # Label for the horizontal axis
    ax1.set_ylabel('Mean Firing Rate [Hz]')             # Label for the vertical axis
    ax1.grid()                                          # "Grid" on

    p = [M,1000. * N / T]

    indthres = np.nonzero(I0range > M)
    crossind = indthres[0]
    rico = (F[crossind[0]] - F[crossind[0] - 1]) / (I0range[crossind[0]] - I0range[crossind[0] - 1])
    p[1] = rico * (M - I0range[crossind[0] - 1]) + F[crossind[0] - 1]

    xmin, xmax = ax1.get_xbound()
    ymin, ymax = ax1.get_ybound()
    l1 = mlines.Line2D([p[0],p[0]], [ymin,p[1]])
    l2 = mlines.Line2D([xmin,p[0]], [p[1],p[1]])
    ax1.add_line(l1)
    ax1.add_line(l2)

    alpha = 1000. * (N1-N) / (T * 0.1*M)
    t = np.arange(ymin, p[1], (p[1]-ymin)*0.005)
    i = M + M1 * np.cos(6.28*FF*t*0.02*10000.)
    ax1.plot(i,t)
    f = p[1] + 20 * M1 * np.cos(6.28 * FF * t * 0.002 * 100000.)
    t = np.arange(xmin, p[0], (p[0]-xmin)*0.005)
    ax1.plot(t,f)

    ax2 = fig.add_subplot(122)
    ax2.plot(timevec, somav)                        # Make the actual plot versus time
    # timevecpyt = timevec.to_python()
    timevecpyt = np.arange(0., T, dt)
    tmp1 = -60 + 10 * np.cos(6.28*FF*timevecpyt)
    ax2.plot(timevecpyt, tmp1)
    ax2.set_xlim( (0,400) )                             # Set the horizontal limits
    ax2.set_ylim( (-80,50) )                            # Set the vertical limits
    ax2.set_xlabel('time [ms]')                         # Label for the horizontal axis
    ax2.set_ylabel('u - membrane potential [mV]')       # Label for the vertical axis
    ax2.grid()                                        # "Grid" on

    plt.show()
#---------------------------------------------------------------------------------------
def plot_histogram(cell, T, I0, S, I1, F):
    #F = 60          # Hz
    #T  = 200000.   # ms

    #I0 = 500.      # pA
    #S  = 300.      # pA
    #I1 = 80.       # pA
    tau = 5.        # ms
    delay = 0
    dt = 0.05
    mu = 0
    loc = 0.5
    seed = random.randint(0, 1000)
    F = F / 1000.
    stim = attach_noise_sin_clamp(cell, delay, T, I0, I1, F, dt, tau, S, mu, loc, seed)

    counts = h.Vector()
    apc = h.APCount(cell.soma[0](0.5))
    apc.thresh = 0
    apc.record(counts)
    run_model(dt, T)

    N = len(counts)
    tsp = np.array(counts.to_python())
    x   = np.cos(6.28318530718 * tsp * F)/N
    y   = np.sin(6.28318530718 * tsp * F)/N

    R0  = 1000. * N / T   # spikes/s
    R1  = 2*R0*np.absolute(np.complex(np.sum(x), np.sum(y)))
    PHI = np.angle(np.complex(np.sum(x), np.sum(y)), deg=False)
    tsp = np.remainder(tsp,  2 / F)
    C   = np.floor_divide(T, 2 / F)
    print(C)
    fig = plt.figure(figsize=(14, 4))

    hist, bins = np.histogram(tsp, bins=40)
    print(hist)
    hist = 1000. * hist / ((bins[3] - bins[2]) * C)   # Hz
    # R1 = (np.max(hist) - np.min(hist))/2

    W = (2. / F)/40.
    plt.bar(bins[:-1], hist, width=W, color='black')

    y = R0 + R1 * np.cos(6.28318530718 * bins * F - PHI)
    z = R0 + R1 * np.cos(6.28318530718 * bins * F )
    plt.plot(bins, y, 'r', linewidth=1)
    plt.plot(bins, z, 'g--', linewidth=1)

    ax = fig.add_subplot(111)
    ax.set_xlim( (0, 2./F - (bins[3] - bins[2])) )                         # Set the horizontal limits
    # ax.set_ylim( (0, 2*R0) )                         # Set the horizontal limits
    ax.set_xlabel('time within two periods [ms]')           # Label for the horizontal axis
    ax.set_ylabel('Instantaneous firing rate [spikes/s]')   # Label for the vertical axis
    #ax.grid()                                               # "Grid" on

    plt.show()
#---------------------------------------------------------------------------------------
def plot_transferfunction(cell, T, I0, S, I1, FRange):
    tau = 5
    delay = 0
    dt = 0.05
    mu = 0
    loc = 0.5

    myfile = "spikes.x"
    transfervec = np.zeros((len(FRange), 1))  # Create vector to store all transfer factor
    phasevec = np.zeros((len(FRange), 1))  # Create vector to store all the phase delay values
    counter = 0
    for freq in FRange:
        seed = random.randint(0, 1000)
        stim = attach_noise_sin_clamp(cell, delay, T, I0, I1, freq/1000., dt, tau, S, mu, loc, seed)

        counts = h.Vector()
        apc = h.APCount(cell.soma[0](0.5))
        apc.thresh = 0
        apc.record(counts)
        run_model(dt, T)

        N = len(counts)
        tsp = np.array(counts.to_python())

        x = np.cos(6.28318530718 * tsp * freq / 1000.) / N
        y = np.sin(6.28318530718 * tsp * freq / 1000.) / N
        R0 = 1000. * N / T  # spikes/s
        transfervec[counter] = np.absolute(np.complex(np.sum(x), np.sum(y)))
        phasevec[counter] = -np.angle(np.complex(np.sum(x), np.sum(y)), deg=False)
        counter += 1

    phasevec = phase_unwrap(phasevec)
    try:
        calc_cof(FRange, transfervec)
    except:
        print("Couldn't find a crossing between transfer curve and 70% line.")
    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(2, 1, 1)
    plt.grid('on')
    plt.loglog(FRange, transfervec, 'o-', linewidth=3)
    plt.loglog(FRange, transfervec[0]*0.707*np.ones((FRange.size)), linewidth=2, color='black')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transfer factor')
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.semilogx(FRange, phasevec, 'o-', linewidth=3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase delay [Rad]')
    plt.grid('on')
    plt.subplots_adjust(hspace=0)
    plt.show()
#---------------------------------------------------------------------------------------
def phase_unwrap(phasevec):
    for i in np.arange(1, len(phasevec)):
        currphaseval = phasevec[i]
        if currphaseval - phasevec[i-1] > 2:
            phasevec[i:] -= 2*np.pi
    return phasevec
#---------------------------------------------------------------------------------------
def calc_cof(FRange, transfervec):
    indthres = np.nonzero(transfervec < 0.707*transfervec[0])
    crossind = indthres[0]
    cof = FRange[crossind[0]-1] - ((transfervec[crossind[0]-1] - transfervec[0]*0.707) / (transfervec[crossind[0]-1] - transfervec[crossind[0]])) * (FRange[crossind[0]-1] - FRange[crossind[0]])
    strtoprint = "The cut-off frequency is at " + str(cof[0]) + " Hz."
    print(strtoprint)
#---------------------------------------------------------------------------------------
