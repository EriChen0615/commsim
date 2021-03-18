"""
This is a library built for simulating digital communication
Author: Jinghong Chen
Email: jc2124@cam.ac.uk

This file contains the helper classes to simulate a channel in digital data communication

Available Channels include:
-   Additive White Gaussian Channel
-   Wireless Channel (impulse response [1/2, 1/4, 0, 1/4/])
-   Custom Channel (specify channel impulse response)

"""

import numpy as np

class AWGN_Channel:
    def __init__(self, snr):
        """
        @parameters:
        - snr: signal-to-noise ratio in dB
        """
        self.name = "AWGN"
        self.snr = snr
        self.h  = np.array([1])

    def pass_through(self, x):
        """
        add AWGN to the signal with snr specified
        @parameters:
        - x: passband data (optionally after convolving with channel impulse)
        """
        sig_power = np.mean(x**2)
        c = sig_power / 10**(self.snr/10)
        noise = c**0.5 * np.random.randn(len(x))
        return x + noise

class TimeDelay_Channel:
    def __init__(self, delay):
        """
        @parameters:
        - delay: float, a time delay of the channel in number of samples
        """
        self.name = "TimeDelay"
        self.delay = delay
        N = 21 # number of taps of the delay filter
        n = np.arange(N) # 0,1,2,3...
        h = np.sinc(n - self.delay) # calc filter taps
        h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
        self.h = h / np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power

    def pass_through(self, x):
        return np.convolve(x, self.h)
     

class FrequencyOffset_Channel:
    def __init__(self, f0, fs):
        """
        @parameters:
        - f0: offset frequency
        - fs: sampling frequency
        """
        self.name = "FrequencyOffset"
        self.f0 = f0
        self.fs = fs

    def pass_through(self, x):
        """
        add frequency offset to the signal, assuming the samples are samples at self.fs
        """
        N = np.arange(0, len(x))
        return np.real(x * np.exp(1j*2*np.pi*N*self.f0/self.fs))

class Custom_Channel:
    def __init__(self, name, channel_h):
        """
        @parameters:
        - channel_h: the impulse response of the channel
        """
        self.name = name
        self.h = channel_h

    def pass_through(self, x, noiseless=False):
        """
        pass through the channel, convolve with channel filter
        @parameters:
        - x: passband data
        """
        return np.convolve(x, self.h)