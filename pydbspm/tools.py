import os
from tempfile import TemporaryDirectory
import numpy as np

def get_fs(F, dz=0.075, k0=1800, f0=30300, n=10, conv=16.0217656):
    '''
    PPM function
    conversion of vertical force Fz to frequency shift 
    according to:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
       
    INPUTS:
            F: vertical force array in N.
            dz: step size in vertical direction in Armstrongs.
            k0: spring constant of the tip in eV/Armstrong².
            f0: natural frequency of the cantilever in Hz.
            n: along with dz, determines the oscillation amplitude of the 
               cantilever A = n * dz.
            conv: conversion factor. Default is 16.0217656 to convert the units of
                  k0 from eV/Armstrong² to N/m².
             
    '''
    # Set the integration intervals to calculate the corresponding weight
    x  = np.linspace(-1,1,int(n+1) )
    y  = np.sqrt(1-x*x)
    # Take derivate with middle finite differences
    dy =  ( y[1:] - y[:-1] )/(dz*n)
    fpi    = (n-2)**2
    prefactor = ( 1 + fpi*(2/np.pi) ) / (fpi+1) # correction for small n
    # Multiply the weights matrix by convolving the the weigths with the force along z.
    dFconv = -prefactor * np.apply_along_axis( lambda m: np.convolve(m, dy, mode='valid'), axis=2, arr=F )
    return dFconv*conv*f0/k0

class usetmpdir(object):
    '''Context manager to change python cwd to a temporary directory.
    It can take a suffix (suffix=str()) for the dir.'''
    def __init__(self):
        self.tdir = TemporaryDirectory()
        self.oldpwd = os.getcwd()
    def __enter__(self):
        return os.chdir(self.tdir.name)
    def __exit__(self, type, value, traceback):
        os.chdir(self.oldpwd)
        self.tdir.cleanup()