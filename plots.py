import numpy as np
import batman
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter
import seaborn as sns
from scipy.interpolate import CubicSpline


# Step sizes for major and minor ticks
def majMinTick(xmin, xmax, nxmajor=7, nxminor=5):
    '''
    Calculate step sizes for major and minor tick levels

    Parameters
    ----------
    xmin : float
        Minimum value of x
    xmax : float
        Maximum value of x
    nxmajor : int 
        Typical number of required major ticks on the x-axis
    nxminor : int 
        Number of required minor ticks between two consecutive major ticks on the x-axis

    Return
    ------
    xmajor : float
        Step size for major ticks on the x-axis
    xminor : float
        Step size for minor ticks on the x-axis
    '''

    xmajor = float("{:.0e}".format((xmax - xmin) / nxmajor))
    xminor = xmajor / nxminor

    return xmajor, xminor


# Plot the orbital parameters as a function of the step number
def plot_orbital_params(sampler_chain, fname='./orbital_params.png'):
    
    nwalkers, nsteps, ndim = sampler_chain.shape

    mpl.rc('font', family='serif')
    mpl.rc('font', serif='Times New Roman')
    mpl.rc('text', usetex='false')
    mpl.rcParams.update({'font.size': 12})

    x = np.arange(nsteps)

    plt.figure(1)
    plt.subplot(5, 1, 1)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 0], '-', color='darkred')
    plt.ylabel(r'$P \ [{\rm d}]$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))


    plt.subplot(5, 1, 2)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 1], '-', color='darkgreen')
    plt.ylabel(r'$T_0 \ [{\rm d}]$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))
    #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.7e'))

    plt.subplot(5, 1, 3)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 2], '-', color='darkblue')
    plt.ylabel(r'$R_p/R_*$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.subplot(5, 1, 4)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 3], '-', color='lime')
    plt.ylabel(r'$R_*/a$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.subplot(5, 1, 5)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 4], '-', color='purple')
    plt.xlabel('Step Number')
    plt.ylabel(r'$b$')
    plt.xlim(0, nsteps)
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))


    plt.gcf().subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.10, hspace=0.05)
    plt.gcf().set_size_inches(6, 8)
    
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close(1)

    return


# Plot the limb darkening parameters as a function of the step number
def plot_ld_params(sampler_chain, fname='./ld_params.png'):
    
    nwalkers, nsteps, ndim = sampler_chain.shape

    mpl.rc('font', family='serif')
    mpl.rc('font', serif='Times New Roman')
    mpl.rc('text', usetex='false')
    mpl.rcParams.update({'font.size': 12})

    x = np.arange(nsteps)

    plt.figure(2)
    plt.subplot(5, 1, 1)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 5], '-', color='darkred')
    plt.ylabel(r'$a_1$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))


    plt.subplot(5, 1, 2)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 6], '-', color='darkgreen')
    plt.ylabel(r'$a_2$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.subplot(5, 1, 3)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 7], '-', color='darkblue')
    plt.ylabel(r'$a_3$')
    plt.xlim(0, nsteps)
    plt.xticks([], [])
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.subplot(5, 1, 4)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 8], '-', color='lime')
    plt.xlabel('Step Number')
    plt.ylabel(r'$a_4$')
    plt.xlim(0, nsteps)
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.subplot(5, 1, 5)
    for i in range(nwalkers):
        plt.plot(x, sampler_chain[i, :, 9], '-', color='lime')
    plt.xlabel('Step Number')
    plt.ylabel(r'$\log f$')
    plt.xlim(0, nsteps)
    plt.gca().xaxis.set_major_locator(MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))


    plt.gcf().subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.10, hspace=0.05)
    plt.gcf().set_size_inches(6, 8)
    
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close(2)

    return


# Make corner plot for fitting parameters 
def distribution(samples, fname='./corner.png'):

    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.2, right=0.9)
    
    ax1 = fig.add_subplot(111)
    ax1.set_rasterization_zorder(-1)

    fig = corner.corner(
        samples, color='darkblue', labels=[r'$P$', r'$T_0$', r'$R_p/R_*$', r'$R_*/a$', 
        r'$b$', r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$', r'$\log f$'], 
        label_kwargs={"fontsize": 20}, quantiles=[0.16, 0.50, 0.84], show_titles=False, 
        title_fmt=None, title_kwargs=None, max_n_ticks=3, labelpad=0.2
    )

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=15)

    fig.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return


# Plot light curve and the fit
def light_curve_fit(tim, flux, sig, pars, exp_time=0., fname='./light_curve.png'):

    # Create BATMAN object
    params = batman.TransitParams()
    params.t0 = pars[1, 0]
    params.per = pars[0, 0]
    params.rp = pars[2, 0]
    params.a = 1. / pars[3, 0]
    params.inc = np.arccos(pars[3, 0] * pars[4, 0]) * 180. / np.pi
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"
    params.u = [pars[5, 0], pars[6, 0], pars[7, 0], pars[8, 0]]
    
    # Calculate the model light curve
    m = batman.TransitModel(params, tim, fac=5e-3, exp_time=exp_time)
    mflux = m.light_curve(params)

    # Phase plot
    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(
        bottom=0.12, right=0.95, top=0.98, left=0.15, wspace=0.25, hspace=0.15
    )
    
    ax1 = fig.add_subplot(3, 1, (1, 2))
    ax1.set_rasterization_zorder(-1)
    ax1.plot(tim, flux, ".", color="r")
    ax1.plot(tim, mflux, "b-", lw=2)
    plt.ylabel(r"Flux [arbitrary units]", fontsize=20)
    ymin, ymax = ax1.get_ylim()
    dy = (ymax - ymin)/ 10.
    plt.ylim((ymin - dy, ymax + dy))
    xmin, xmax = (tim[0], tim[-1])
    plt.xlim((xmin, xmax))
    plt.xticks(color="w")
    
    ax2 = fig.add_subplot(3, 1, (3, 3))
    ax2.set_rasterization_zorder(-1)
    f = 10.**(pars[-1, 0])
    ax2.plot(tim, (flux - mflux) / (f * sig), ".", color="r")
    ax2.axhline(y=0., ls='--', color='k', lw=2)
    dof = pars.shape[0]
    chi2r = np.sum(((flux - mflux) / sig)**2) / (len(flux) - dof)
    chi2rf = np.sum(((flux - mflux) / (f * sig))**2) / (len(flux) - dof)
    print ("\nReduced chi-squares = %.8f (sig), %.8f (f * sig)" %(chi2r, chi2rf))
    ymin, ymax = ax2.get_ylim()
    ymax = max(abs(ymin), abs(ymax))
    plt.xlabel(r"Barycentric Julian Date [d]", fontsize=20)
    plt.ylabel(r"S. R.", fontsize=20)
    plt.ylim((-1.5 * ymax, 1.5 * ymax))
    plt.xlim((xmin, xmax))
    
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return


# Make phase diagram
def phase_diagram(tim, flux, sig, pars, exp_time=0., fname='./phase_diagram.png'):

    # Calculate phase
    tmp = (tim - pars[1, 0]) / pars[0, 0]
    phase = np.zeros(len(tim))
    for i in range(len(tim)):
        if tmp[i] == 0.:
            phase[i] = 0. 
        elif tmp[i] > 0.:
            phase[i] = (tmp[i] % 1)
            if phase[i] > 0.5:
                phase[i] -= 1.
        else:
            phase[i] = np.abs(np.floor(tmp[i])) + tmp[i]
            if phase[i] > 0.5:
                phase[i] -= 1.

    # Create BATMAN object
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = pars[2, 0]
    params.a = 1. / pars[3, 0]
    params.inc = np.arccos(pars[3, 0] * pars[4, 0]) * 180. / np.pi
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"
    params.u = [pars[5, 0], pars[6, 0], pars[7, 0], pars[8, 0]]
    
    # Calculate the model light curve
    t = np.linspace(np.amin(phase), np.amax(phase), 1000)
    m = batman.TransitModel(params, t, fac=5e-3, exp_time=exp_time)
    mflux = m.light_curve(params)

    # Phase plot
    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(bottom=0.15, top=0.98, left=0.15, right=0.8, hspace=0.05)
    
    ax1 = fig.add_subplot(3, 1, (1, 2))
    ax1.set_rasterization_zorder(-1)

    ax1.plot(phase, flux, ".", ms=1, rasterized=True, color='#D55E00', zorder=1)
    ax1.plot(t, mflux, "-", lw=2, rasterized=True, color='#56B4E9', zorder=2)

    ax1.set_ylabel(r"Flux [arbitrary units]", fontsize=14, labelpad=2)

    ax1.tick_params(axis='y', labelsize=11, which='both', direction='inout', pad=2)
    ax1.tick_params(
        axis='x', labelsize=11, which='both', labelbottom=False, direction='inout',
        pad=2
    )

    xmin, xmax = (t[0], t[-1])
    ax1.set_xlim(left=xmin, right=xmax)
    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=7, nxminor=5)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    
    ymin, ymax = ax1.get_ylim()
    dy = (ymax - ymin)/ 10.
    ymin = ymin - dy
    ymax = ymax + dy
    ax1.set_ylim(bottom=ymin, top=ymax)
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=5, nxminor=5)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)


    ax2 = fig.add_subplot(3, 1, (3, 3))
    ax2.set_rasterization_zorder(-1)

    cs = CubicSpline(t, mflux)
    f = 10.**(pars[-1, 0])
    ax2.plot(
        phase, (flux - cs(phase)) / (f * sig), ".", ms=1, rasterized=True, color='#D55E00', 
        zorder=1
    )
    ax2.axhline(y=0., ls='--', lw=2, rasterized=True, color='#56B4E9', zorder=2)

    ax2.set_xlabel(r'Phase', fontsize=14, labelpad=2)
    ax2.set_ylabel(r"S.R.", fontsize=14, labelpad=2)

    ax2.tick_params(axis='y', labelsize=11, which='both', direction='inout', pad=2)
    ax2.tick_params(axis='x', labelsize=11, which='both', direction='inout', pad=2)

    ax2.set_xlim(left=xmin, right=xmax)
    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=7, nxminor=5)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)

    ymin, ymax = ax2.get_ylim()
    ymax = max(abs(ymin), abs(ymax))
    ymin = -1.5 * ymax
    ymax = 1.5 * ymax
    ax2.set_ylim(bottom=ymin, top=ymax)
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=3, nxminor=5)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return
