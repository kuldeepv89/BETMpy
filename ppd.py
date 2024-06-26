import numpy as np
import batman


# Calculate logarithm of the prior
def lnprior(theta):
    Per, T0, Rp_div_Rs, Rs_div_a, b, a1, a2, a3, a4, logf = theta

    if (
        0. < Per < 500. and 
        0. < T0 < 2460300. and 
        0. < Rp_div_Rs < 0.5 and 
        0. < Rs_div_a < 0.5 and 
        -1. < b < 1. and 
        -50. < a1 < 50. and
        -50. < a2 < 50. and 
        -50. < a3 < 50. and 
        -50. < a4 < 50. and
        -0.5 < logf < 0.5
    ):
        return 0.0

    return -np.inf


# Calculate logarithm of the likelihood
def lnlikelihood(theta, tim, flux, sig, lamda, exp_time):
    Per, T0, Rp_div_Rs, Rs_div_a, b, a1, a2, a3, a4, logf = theta
    f = 10.**(logf)

    # Create an object containing BATMAN input parameters
    params = batman.TransitParams()
    params.t0 = T0
    params.per = Per
    params.rp = Rp_div_Rs
    params.a = 1. / Rs_div_a
    params.inc = np.arccos(b * Rs_div_a) * 180. / np.pi
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"
    params.u = [a1, a2, a3, a4]
    
    # Calculate model light curve
    m = batman.TransitModel(params, tim, fac=5e-3, exp_time=exp_time)
    mflux = m.light_curve(params)

    # Calculate log likelihood
    lnlike = - 0.5 * np.sum(
        np.log(2. * np.pi * f**2 * sig**2) + ((flux - mflux) / (f * sig))**2
    )

    # Add regularization term
    mu = np.linspace(0.01, 1., 100)
    d2I_div_dmu2 = (
        - (0.25 * a1 / np.sqrt(mu**3)) + (0.75 * a3 / np.sqrt(mu)) + 2. * a4
    )
    lnlike -= 0.5 * lamda**2 * np.sum(d2I_div_dmu2**2)

    return lnlike 


# Calculate logarithm of the posterior probability
def lnprob(theta, tim, flux, sig, lamda, exp_time):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlikelihood(theta, tim, flux, sig, lamda, exp_time)
