/***********************************************************************************************
  EIF.c (Exponential Integrate and Fire - for the HBP Collab)
  Antwerpen, 30/1/2018 
    
  Compile with:    gcc -o eif eif.c -lm -O
  Output files:    output.x, spikes.x
************************************************************************************************/

#include <stdio.h>              // Included (needed by fprintf, fopen, etc. ).
#include <stdlib.h>             // Standard library (needed by atof(), atoi(), etc.).
#include <math.h>               // Mathematical library (needed by exp(), etc.).
#include <time.h>               // Time library for initializing the random numbers 

typedef unsigned long int INT;  // Definition of a new LONG type, just for convenience.
#define TWOPI 6.283185307179586 // Definition of a useful constant

FILE *fopen();                  // File pointers to be used for logging results on disk.
FILE *output;		            // Output file, contains sample membrane voltage..
double gauss();					// Random number generator

// Numerical parameters of the model
double C     = 281.;  // Membrane capacitance [pF]
double gl    = 30.;   // Leak membrane conductance [nS]
double El    = -70.6; // Leak (or resting) membrane potential [mV]
double DT    = 2.;    // Steepness of the exponential IeF [mV]
double VT    = -50.4; // Excitability threshold parameter for the EIF [mV]
double theta = 40.;   // Conventional threshold for AP emission [mV]
double uH    = -70.6; // Reset membrane voltage after a spike
double Tarp  = 2.;    // Absolute refractory period [ms]
// Initialization of the state variables of the model
double u     = -70.6; // The membrane potential [mV]
double t0    = -9999; // The occurrence time of the last spike [ms]
// Initialization of simulation control variables
double t     = 0.;    // actual time [ms]
double dt    = 0.1;   // integration time step [ms]
double T     = 1000.; // Simulation lifetime [ms]
INT    N;             // Corresponding number of steps
// Initialization of the stimulation (external) current experienced by the model
double I0    = 550.;  // Value of an external DC offset current stimulus
double I1    = 0.;    // Value of an external sinusoidal amplitude current 
double F0    = 10./1000.;  // Value of an external sinusoidal frequency [kHz] current
double S     = 100.;  // Value of an external stddev of the noisy component
double tau   = 5.;    // Value of an external autocorr length of the noisy component


/********************************  MAIN ***********************************/ 
int main (int argc,char *argv[]) { 
  INT spikes;		// Counter over the number of spikes fired
  INT index   = 0;  // Generic index for the loops
  int printVt = 0;	// Flag: whether or not dump u(t) on file (output.x)
  double tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, udot, i;
  double *out, *tspikes;

  if (argc < 7)  {
      printf("USAGE: T I0 I1 F0 S tau printVt\n"); 
      exit(0);
  }

  T        = atof(argv[1]);  
  I0       = atof(argv[2]);   
  I1       = atof(argv[3]);                  
  F0       = atof(argv[4]);
  S        = atof(argv[5]);
  tau      = atof(argv[6]);
  printVt  = (argc == 8) ? atoi(argv[6]) : 0;	// Default is "no dump of u(t)"

  N        = (INT)(T/dt);
  out      = calloc(N, sizeof(double)); // Reserving memory for the N x 1 double array
  tspikes  = calloc(N, sizeof(double)); // Reserving memory for the N x 1 double array

  //printf("\nEIF January 2018 - M. Giugliano, HBP-SGA1,SP4\n\n");

  srand(time(NULL));   // Initialization of the random number generator

  spikes = 0;
  // Four constants defined and initialised here once for all, to save CPU time (EIF)   
  tmp1 = dt/C * gl * El;
  tmp2 = - dt/C * gl;
  tmp3 = dt/C * gl * DT;
  tmp4 = dt/C; 
  // Other three constants are defined and initialised here once for all, to save CPU time (O.U.)   
  tmp5 = exp(-dt/tau);
  tmp6 = S * sqrt(1. - exp(-2.*dt/tau));  
  tmp7 = I0 * (1. - exp(-dt/tau));
    
  i    = I0;   // Initial condition of the eq. used to simulate a O.U. process
  while (t<=T) { 
        if (u == theta) {          // A spike has just been detected!
            u    = uH;             // The membrane potential is reset to uH
            t0   = t;              // The current spike time is stored into t0
            tspikes[spikes++] = t; // Log each spike and increase the number of spikes  
        }      
        else if ((t-t0) >= Tarp) { // Are we out of absolute refractoriness ?
            udot  = tmp1 + tmp2 * u + tmp3 * exp((u-VT)/DT) + tmp4 * (i + I1 * cos(TWOPI * F0 * t));
            u     = ((u + udot) > theta) ? theta : (u + udot);
        }
        else {                     // Still in refractoriness...
            u = uH;                // I keep u clamped at uH
        }                
        t += dt;                   // Advance time (i.e. Euler's forward method)
        i = i * tmp5 + tmp6 * gauss() + tmp7; // Update the O.U. noisy current realisation 
        out[index++] = u;		   // Log the membrane potential u(t)
} // end while()

 output = fopen("spikes.x", "w");
 for (index=0;index<spikes;index++) fprintf(output, "%f\n", tspikes[index]);
 fclose(output);

if (printVt) {
 output = fopen("output.x", "w");
 for (index=0;index<N;index++) fprintf(output, "%f %f\n", index*dt, out[index]);
 fclose(output);
}

free(out);			// Free allocated memory
free(tspikes);	    // Free allocated memory
return 0;
} // end main
/********************************  MAIN ***********************************/ 


/********************************  GAUSS ***********************************/ 
double gauss() {	// Generate random deviates with normal distribution
	static int iset=0;
	static double gset;
	double fac,r,v1,v2;

	if  (iset == 0) {
		do {
			v1=2.0*((double)rand()/RAND_MAX)-1.0;
			v2=2.0*((double)rand()/RAND_MAX)-1.0;
			r=v1*v1+v2*v2;
		} while (r >= 1.0);
		fac=sqrt(-2.0*log(r)/r);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
} // end gauss()
/********************************  GAUSS ***********************************/ 

