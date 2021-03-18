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
    def __init__(self, snr, delay=0):
        """
        @parameters:
        - snr: signal-to-noise ratio in dB
        - delay: float, a time delay of the channel in number of samples
        """
        self.snr = snr
        self.h  = np.array([1])
        self.delay = delay

        N = 21 # number of taps of the delay filter
        n = np.arange(N) # 0,1,2,3...
        h = np.sinc(n - self.delay) # calc filter taps
        h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
        self.delay_filter = h / np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
    
    def pass_through(self, x, noiseless=False):
        """
        add AWGN to the signal with snr specified
        @parameters:
        - x: passband data (optionally after convolving with channel impulse)
        """
        if noiseless:
            return x
        sig_power = np.mean(x**2)
        c = sig_power / 10**(self.snr/10)
        y = np.convolve(x, self.delay_filter)
        noise = c**0.5 * np.random.randn(len(y))
        y += noise
        return y

class FrequencyOffset_Channel:
    def __init__(self, f0, fs):
        """
        @parameters:
        - f0: offset frequency
        - fs: sampling frequency
        """
        self.f0 = f0
        self.fs = fs

    def pass_through(self, x):
        """
        add frequency offset to the signal, assuming the samples are samples at self.fs
        """
        N = np.arange(0, len(x))
        return x * np.exp(1j*2*np.pi*N*self.f0/self.fs)

class Custom_Channel:
    def __init__(self, channel_h, snr):
        """
        @parameters:
        - channel_h: the impulse response of the channel
        - snr: the signal to noise ratio for 
        """
        self.h = channel_h
        self.snr = snr
        self.AWGN_ch = AWGN_Channel(snr)

    def pass_through(self, x, noiseless=False):
        """
        pass through the channel, convolve with channel filter and add AWGN
        @parameters:
        - x: passband data
        """
        y = np.convolve(x, self.h)
        if noiseless:
            return y
        else:
            return self.AWGN_ch.pass_through(y)