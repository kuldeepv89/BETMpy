import os
import sys
import time
import emcee
import numpy as np
from multiprocessing import Pool
import _pickle as cPickle
import utils_general as ug
import ppd
import plots


# Path and starid
folder, starid = ("example", "Kepler-5")


# Whether to perform fit or analyse previously fitted data
fit = True


# Define the dimension and the MCMC chain related parameters
# ndim     : number of fitting parameters
# nwalkers : number of walkers
# nsteps   : number of steps
# nburn    : number of burn-in steps
ndim, nwalkers, nsteps, nburn = 10, 100, 5000, 4000


# Initial guess for the fitting parameters
# Per [d], T0 [d], Rp_div_Rs, Rs_div_a, b, a1, a2, a3, a4, logf
theta0 = (3.5485, 2455786.2342, 0.08, 0.15, 0.20, 0., 0.5, 0., 0., 0.)


# Regularisation parameter
lamda = 0.2


# Mission exposure time (same unit as the period)
# For Kepler, exp_time = 0
# For TESS, exp_time = 120 s = 120 / (3600 * 24) h
exp_time = 0.


#################### Automated from here onwards ######################
# Generate the same sequence of random numbers
np.random.seed(1)

# Initialize log file 
stdout = sys.stdout
path = os.path.join(folder, starid)
sys.stdout = ug.Logger(os.path.join(path, "log.txt"))

# Print header
print (88 * "=")
ug.prt_center("BAYESIAN EXOPLANET TRANSIT MODELLING", 88)
print ()
ug.prt_center("The BETMpy code", 88)
ug.prt_center("https://github.com/kuldeepv89/BETMpy", 88)
print (88 * "=")

# Print start time
t0 = time.localtime()
print (
    "\nRun started on {0}.\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S", t0))
)

# Analysing star
print ("Analysing star: %s\n" %(starid))

# Print input parameters
print ("Print input parameters...")
print (
    "ndim, nwalkers, nsteps, nburn, lamda, exp_time = %d, %d, %d, %d, %.2e, %.2e" 
    %(ndim, nwalkers, nsteps, nburn, lamda, exp_time)
)
theta0 = ug.initial_guess(theta0)
print (
    "theta0 = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n"
    %(
        theta0[0], theta0[1], theta0[2], theta0[3], theta0[4], theta0[5], 
        theta0[6], theta0[7], theta0[8], theta0[9]
    )
)

# Load the transit data
data = ug.load_data(path)
tim, flux, sig = (data[:, 0], data[:, 1], data[:, 2])


# Perform fit
if fit:

    # Parallelize emcee run
    with Pool() as pool:
    
        # Initialize walker positions
        pos = [theta0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    
        # Run MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, ppd.lnprob, args=[tim, flux, sig, lamda, exp_time], 
            pool=pool
        )
        sampler.run_mcmc(pos, nsteps, progress=True)
    
        # Print the diagnostics of the MCMC run
        # 0.2 < Mean acceptance fraction < 0.5
        print (
            "Mean acceptance fraction = %.2f\n" 
            %(np.mean(sampler.acceptance_fraction))
        )
        # Mean autocorrelation time < nsteps
        #print (
        #    "Mean autocorrelation time = %.2f\n" 
        #    %(np.mean(sampler.get_autocorr_time(tol=0)))
        #)
    
        # Write chain data
        with open(os.path.join(path, 'chain.pkl'), 'wb') as fp:
            cPickle.dump(sampler.chain, fp)


# Load stored MCMC chain
with open(os.path.join(path, 'chain.pkl'), 'rb') as fp:
    sampler_chain = cPickle.load(fp)

# Plot fitting parameters as a function of step number
_ = plots.plot_orbital_params(
    sampler_chain, fname=os.path.join(path, 'orbital_params.png')
)
_ = plots.plot_ld_params(
    sampler_chain, fname=os.path.join(path, 'ld_params.png')
)

# Discard "BAD" samples based on error scale factor value
samples = sampler_chain[:, nburn:nsteps, :].reshape((-1, ndim))
pars = np.zeros((ndim, 3))
pars[-1, 0] = np.percentile(samples[np.abs(samples[:, -1]) < 0.3, -1], 50)
pars[-1, 1] = pars[-1, 0] - np.percentile(samples[np.abs(samples[:, -1]) < 0.3, -1], 16)
pars[-1, 2] = np.percentile(samples[np.abs(samples[:, -1]) < 0.3, -1], 84) - pars[-1, 0]
sigma = 5 * min(pars[-1, 1], pars[-1, 2])
mask = np.logical_and(
    samples[:, -1] > (pars[-1, 0] - sigma),  
    samples[:, -1] < (pars[-1, 0] + sigma)
)
samples = samples[mask, :]
nsamples = samples.shape[0]
total_nsamples = nwalkers * (nsteps - nburn)
print (
    "Discarded sample percentage = %d\n" 
    %((total_nsamples - nsamples) * 100 // total_nsamples)
)

# Calculate fitted parameters and their uncertainties
print ("Fitted parameters...")
for i in range(ndim):
    pars[i, 0] = np.percentile(samples[:, i], 50)
    pars[i, 1] = pars[i, 0] - np.percentile(samples[:, i], 16)
    pars[i, 2] = np.percentile(samples[:, i], 84) - pars[i, 0]
    print ("par, ner, per = %.8f, %0.8f, %0.8f" %(pars[i, 0], pars[i, 1], pars[i, 2]))

# Make corner, light curve and phase diagrams
_ = plots.distribution(samples, fname=os.path.join(path, 'corner.png'))
_ = plots.light_curve_fit(
    tim, flux, sig, pars, exp_time=exp_time, 
    fname=os.path.join(path, 'light_curve.png')
)
_ = plots.phase_diagram(
    tim, flux, sig, pars, exp_time=exp_time/pars[0, 0], 
    fname=os.path.join(path, 'phase_diagram.png')
)

# Calculate h1', h2' and cov(h1', h2') 
mu1 = 2. / 3.
mu2 = 1. / 3.
h1 = np.zeros(nsamples)
h2 = np.zeros(nsamples)
for i in range(nsamples):
    h1[i] = (
        1. -
        samples[i, 5] * (1. - np.sqrt(mu1)) -
        samples[i, 6] * (1. - mu1) -
        samples[i, 7] * (1. - np.sqrt(mu1**3)) -
        samples[i, 8] * (1. - mu1**2)
    )
    h2[i] = (
        h1[i] -
        1. +
        samples[i, 5] * (1. - np.sqrt(mu2)) +
        samples[i, 6] * (1. - mu2) +
        samples[i, 7] * (1. - np.sqrt(mu2**3)) +
        samples[i, 8] * (1. - mu2**2)
    )
h1p = np.percentile(h1, 50)
h1p_nerr = h1p - np.percentile(h1, 16)
h1p_perr = np.percentile(h1, 84) - h1p
print (
    "\nh1_prime, ner, per = %.8f, %0.8f, %0.8f" %(h1p, h1p_nerr, h1p_perr)
)
h2p = np.percentile(h2, 50)
h2p_nerr = h2p - np.percentile(h2, 16)
h2p_perr = np.percentile(h2, 84) - h2p
print (
    "h2_prime, ner, per = %.8f, %0.8f, %0.8f" %(h2p, h2p_nerr, h2p_perr)
)
h12 = np.vstack((h1.reshape(1, nsamples), h2.reshape(1, nsamples)))
print ("Correlation coefficient = %0.8f\n" %(np.corrcoef(h12)[0, 1]))

# Print completion time 
t1 = time.localtime()
print(
    "\nFinished on {0}".format(time.strftime("%Y-%m-%d %H:%M:%S", t1)),
    "(runtime {0} s).".format(time.mktime(t1) - time.mktime(t0)),
)

# Save log file
sys.stdout = stdout
