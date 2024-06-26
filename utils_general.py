import os
import sys
import glob
import numpy as np


# Load observed data
def load_data(path):

    # Check whether folder exists
    if not os.path.isdir(path):
        raise ValueError("ERROR: Data folder does not exist!")

    # Find list of data files
    wd = os.getcwd()
    os.chdir(path)
    dataList = glob.glob("*.csv")
    os.chdir(wd)

    # Check whether data files exist
    if len(dataList) == 0:
        raise FileNotFoundError("ERROR: No csv file found!")

    # Read data
    dataList = sorted(dataList)
    for i, filename in enumerate(dataList):
        if i == 0:
            data = np.genfromtxt(os.path.join(path, filename), delimiter=",")
        else:
            tmp = np.genfromtxt(os.path.join(path, filename), delimiter=",")
            data = np.vstack((data, tmp))

    return data 


# Initial guess for fitting parameters
def initial_guess(theta0):

    # Estimate period 
    if theta0[0] is None:
        raise NotImplementedError("ERROR: Guess for period cannot be None!")
    else:
        Per_0 = theta0[0]
    
    # Estimate time of mid transit
    if theta0[1] is None:
        T_0 = tim[np.argmin(flux)]
    else:
        T_0 = theta0[1]

    return (
        Per_0, T_0, theta0[2], theta0[3], theta0[4], theta0[5], theta0[6], 
        theta0[7], theta0[8], theta0[9]
    )


# Redefine stdout to terminal and an output file
class Logger(object):
    """
    Class used to redefine stdout to terminal and an output file.

    Parameters
    ----------
    outfilename : str
        Absolute path to an output file
    """

    # Credit: http://stackoverflow.com/a/14906787
    def __init__(self, outfilename):
        self.terminal = sys.stdout
        self.log = open(outfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# Centered printing
def prt_center(text, llen):
    """
    Prints a centered line

    Parameters
    ----------
    text : str
        The text string to print

    llen : int
        Length of the line

    Returns
    -------
    None
    """

    print("{0}{1}{0}".format(int((llen - len(text)) / 2) * " ", text))
