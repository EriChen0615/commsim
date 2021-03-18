"""
This is a library built for simulating digital communication
Author: Jinghong Chen
Email: jc2124@cam.ac.uk

This file contains classes and functions which implement equalisers below:
- Zero Forcing Equaliser (ZF)
- Minimum Mean Square Error Equaliser (MMSE)

An equaliser can be designed from the effective filter which defines
the inter-symbol interference (ISI). g(t) = p(t) * h(t) * q(t), and
g[k] = g[mT], where p(t) and q(t) are the modulating pulse and h(t) is
the channel's impulse response. (* represents convolution)
"""
import numpy as np

def design_zf_eq(g, K=200):
    """
    @parameters:
    - g : the impulse response of the effective filter
    - K : taps of the FIR filter (the number of coefficients)
    """
    L = len(g) # length of impulse response

    # Construct the linear equations
    M = np.zeros((K, K+L-1))
    for i in range(K):
        M[i, i:i+L] = np.flip(g)
    M = M[:, L-1:]

    b = np.zeros(K)
    b[0] = 1 # Enforce zero ISI
    
    h = np.linalg.inv(M).dot(b) # solve linear equation by linear inversion (wasteful! But we are working with small matrix so Okay)
    return h

def design_mmse_eq(g, E, N0, K=200):
    """
    g : effective filter
    E : average symbol energy
    N0 : N0 is the variance of the White Gaussian Noise
    K : taps of the FIR filter
    """
    L = len(g) # length of impulse response
    
    U = np.zeros((K, K+L-1))
    for i in range(K):
        U[i,i:i+L] = np.flip(g)
    
    # compute R and p
    R = E*U.dot(U.T) + N0 * np.identity(K)
    p = E*U[:, L-1]
    
    # compute coefficients
    c = np.linalg.inv(R).dot(p)
    return np.flip(c)

def compute_effective_filter(pf, h, L=100):
    """
    This function computes the effectißve filter, assuming all the filters are obtained
    using the same sampling rate
    @parameters:
    - pf: object of a Pulse Filter, assuming the same filter at tx and rx
    - h : impulse resposne of the channel
    - L : maximum length of the effective filter
    """
    g = np.convolve(pf.filter, h, 'same')
    g = np.convolve(g, pf.filter, 'same')
    gk = []
    k = pf.L
    while k < len(g) and len(gk) < L:
        gk.append(g[k]) # the effective ISI
        k += pf.symbol_period
    return gk

class MMSE_Equaliser:
    """
    This class implements a MMSE equaliser
    """
    def __init__(self, snr, h=None, gL=50, K=20):
        """
        @parameters:
        - pf      : pulse modulation object
        - channel : channel object
        - h       : additional filters which has `filter(x)` method
        - gL      : maximum length of the effective filter
        - K       : number of coefficients of equaliser (including h0)
        """
        self.K = K
        self.gL = gL
        self.g = None
        self.snr = snr
    
    def equalise(self, x):
        """
        equalise the signal, assuming the effective filter
        """
        y = np.convolve(x, self.eq_filter)[self.K-1:]
        return y

    def get_g(self):
        """
        return the computed effective filter
        """
        return self.g

    def design(self, gk):
        """
        return the computed effective filter
        Note that the effective filter is obtained from a simulation
        @parameters:
        - gk: the effective filter, will be stored in self.gß
        """
        self.g = gk # effective filter, obtained by simulate transmiting a one followed by a series of zeros
        self.eq_filter = design_mmse_eq(self.g, 10**(self.snr/10), 1, K=self.K)

class ZF_Equaliser:
    """
    This class implements a MMSE equaliser
    """
    def __init__(self, gL=50, K=20):
        """
        @parameters:
        - pf      : pulse modulation object
        - channel : channel object
        - h       : additional filters which has `filter(x)` method
        - gL      : maximum length of the effective filter
        - K       : number of coefficients of equaliser (including h0)
        """
        self.K = K
        self.gL = gL
        self.g = None
        self.eq_filter = None
       
    def equalise(self, x):
        """
        equalise the signal, assuming the effective filter
        """
        y = np.convolve(x, self.eq_filter)
        return y[:-self.K+1]

    def design(self, gk):
        """
        return the computed effective filter
        Note that the effective filter is obtained from a simulation
        @parameters:
        - gk: the effective filter, will be stored in self.gß
        """
        self.g = gk # effective filter, obtained by simulate transmiting a one followed by a series of zeros
        self.eq_filter = design_zf_eq(gk, K=self.K)

    def get_g(self):
        """
        return the computed effective filter
        """
        return self.g


