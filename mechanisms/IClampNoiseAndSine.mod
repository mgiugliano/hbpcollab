COMMENT
  This .mod file introduces a pointprocess for current clamp made with sinusoidal noisy signal.
  To use, on linux, you have to locate it in the same folder as the .py file you are running, and run nrnivmodl before
  running the actual script. Doing this will create a x86_64 folder with inside all the compiled (in c) files needed
  by Neuron to use this point process.
  To use it in the .py fill, you call it as any other point process:
    stim = h.IClampNoise(soma(.5))
  and later you can edit all the PARAMETER fields

normrand(0,1(/nA))

ENDCOMMENT

NEURON {
  POINT_PROCESS IClampNoiseSin
  RANGE i,delay,dur,std,offset,amp,freq,dt,tau,mu,new_seed
  ELECTRODE_CURRENT i
}

UNITS {
  (nA) = (nanoamp)
}

PARAMETER {
  delay=50    (ms)
  dur=200   (ms)
  std=0.001   (nA)
  offset=0.05	(nA)
  amp=0.005	(nA)
  freq=1	(/ms)
  dt=0.0001	(ms)
  tau=10	(ms)
  mu=0		(nA)
  new_seed=0
}

ASSIGNED {
  ival (nA)
  i (nA)
  noise (nA)
  sinuspart (nA)
  ou (nA)
  lastou (nA)
  on (1)
}

PROCEDURE seed(x) {
  set_seed(x)
}

INITIAL {
  i = 0
  on = 0
  lastou = 0
  net_send(delay, 1)
  seed(new_seed)
}



BEFORE BREAKPOINT {
  if  (on) {
    noise = normrand(0,1(/nA))*1(nA)
    sinuspart = amp*sin(freq*2*3.1415927*t)
    ou = lastou*exp(-dt/tau)+mu*(1-exp(-dt/tau))+std*sqrt(1-exp(-2*dt/tau))*noise
    ival = offset+ou+sinuspart
    lastou = ou
  } else {
    ival = 0
  }
}

BREAKPOINT {
  i = ival
}

NET_RECEIVE (w) {
  if (flag == 1) {
    if (on == 0) {
      : turn it on
      on = 1
      : prepare to turn it off
      net_send(dur, 1)
    } else {
      : turn it off
      on = 0
    }
  }
}
