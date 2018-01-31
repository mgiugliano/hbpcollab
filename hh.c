/***********************************************************************************************
  HH.c (Hodgkin Huxley - for the HBP Collab)
  Antwerpen, 31/1/2018 
    
  Compile with:    gcc -o hh h.c -lm -O
  Output files:    output.x, spikes.x
************************************************************************************************/

#include <stdio.h>              // Included (needed by fprintf, fopen, etc. ).
#include <stdlib.h>             // Standard library (needed by atof(), atoi(), etc.).
#include <math.h>               // Mathematical library (needed by exp(), etc.).
#include <time.h>               // Time library for initializing the random numbers 

typedef unsigned long int INT;  // Definition of a new LONG type, just for convenience.
#define TWOPI 6.283185307179586 // Definition of a useful constant
#define SIGMA(x,theta, sigma)   1./(1.+exp(-((x)-(theta))/(sigma)))

FILE *fopen();                  // File pointers to be used for logging results on disk.
FILE *output;		            // Output file, contains sample membrane voltage..
double gauss();					// Random number generator

// Numerical parameters of the model
double C   = 1.;         // [uF/cm^2] Membrane specific capacitance.
double V   = -69.965202; // [mV]      Membrane voltage.

double m = 0.;           // State variable for sodium current activation.
double h = 0.;           // State variable for sodium current inactivation.
double n = 0.;           // State variable for potassium current activation.

double Ina, Ik, Ileak, Iext;   // Membrane current densities for Na, K, Leak and Ext.

double hinf, tauh, thetah, sigmah, thetaht, sigmaht;  // Kinetics of inactivation (Na).
double minf, taum, thetam, sigmam;                    // Kinetics of activation (Na).
double ninf, taun, thetan, sigman, thetant, sigmant;  // Kinetics of activation (K).

double gna = 24.;      // [mS/cm^2] Specific conductance for sodium current.
double gk  = 3.;       // [mS/cm^2] Specific conductance for potassium current.
double gleak=0.25;     // [mS/cm^2] Specific conductance for leak current.

double Ena   = 55.;    // [mV] Sodium current reversal potential.
double Ek    = -90.;   // [mV] Potassium current reversal potential. 
double Eleak = -70.;   // [mV] Leakage current reversal potential.

double thetah = -53.;  // [mV] Kinetic parameter.
double sigmah = -7.;   // [mV] Kinetic parameter.
double thetaht= -40.5; // [mV] Kinetic parameter.
double sigmaht= -6.;   // [mV] Kinetic parameter.
 
double thetam=  -30.;   // [mV] Kinetic parameter.
double sigmam=  9.5;    // [mV] Kinetic parameter.
 
double thetan=  -30.;   // [mV] Kinetic parameter.
double sigman=  10.;    // [mV] Kinetic parameter.
double thetant= -27.;   // [mV] Kinetic parameter.
double sigmant= -15.;   // [mV] Kinetic parameter.

// Initialization of simulation control variables
double t     = 0.;    // actual time [ms]
double dt    = 0.001; // [ms] Integration time step (forward Euler method).
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
  int bool    = 0;	// Boolean variable for spike detection (by threshold crossing)
  double tmp1, tmp2, tmp3, tmp5, tmp6, tmp7, Vth, i;
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
  printVt  = (argc == 8) ? atoi(argv[7]) : 0;	// Default is "no dump of u(t)"

  N        = (INT)(T/dt);
  out      = calloc(N, sizeof(double)); // Reserving memory for the N x 1 double array
  tspikes  = calloc(N, sizeof(double)); // Reserving memory for the N x 1 double array
  Vth      = 0.;						// Peak detection by threshold crossing
  bool     = 0;							// 
  //printf("\HH January 2018 - M. Giugliano, HBP-SGA1,SP4\n\n");

  srand(time(NULL));   // Initialization of the random number generator

  spikes = 0;
  // Four constants defined and initialised here once for all, to save CPU time (EIF)   
  tmp1 = dt/C;
  tmp2 = dt/tauh;
  tmp3 = dt/taun;
  // Other three constants are defined and initialised here once for all, to save CPU time (O.U.)   
  tmp5 = exp(-dt/tau);
  tmp6 = S * sqrt(1. - exp(-2.*dt/tau));  
  tmp7 = I0 * (1. - exp(-dt/tau));
    
  i    = I0;   // Initial condition of the eq. used to simulate a O.U. process
  while (t<=T) { 
	minf = SIGMA(V,thetam,sigmam);
	hinf = SIGMA(V,thetah,sigmah);
	ninf = SIGMA(V,thetan,sigman);
	tauh = 0.37 + 2.78 * SIGMA(V,thetaht,sigmaht);
	taun = 0.37 + 1.85 * SIGMA(V,thetant,sigmant);
	
	Ina   = gna   * (minf*minf*minf)*h * (V - Ena);// The sodium current is updated.
	Ik    = gk    * (n*n*n*n)          * (V - Ek); // The potassium current is updated.
	Ileak = gleak * (V - Eleak);                   // The leak current is updated.
	
	//V   += tmp1 * (- Ina - Ik - Ileak + i);
	V   += tmp1 * (- Ileak - Ina);
	h   += tmp2 * (hinf - h);  // Please note: 'm' has been set to its equilibrium value.
	n   += tmp3 * (ninf - n);  
    i    = i * tmp5 + tmp6 * gauss() + tmp7; // Update the O.U. noisy current realisation 

    t += dt;                   // Advance time (i.e. Euler's forward method)
    out[index++] = V;		   // Log the membrane potential u(t)
    if (V>Vth && !bool) {
    	bool = 1;
    	tspikes[spikes++] = t;
    } else if (V<Vth && bool) bool = 0;
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

