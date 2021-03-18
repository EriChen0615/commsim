"""
This is a library built for simulating digital communication
Author: Jinghong Chen
Email: jc2124@cam.ac.uk

This file contains classes that encapsulate a simulation for digital data transmission

A simulation consists of three parts:
- A transmitter, defined by [encoder], symbol modulator, pulse filter and carrier frequency
- A channel, defined by its impulse response and signal-to-noise ratio (assuming AWGN)
- A receiver, defined by [decoder], symbol demodulator, pulser filter, equaliser and carrier frequency

The rate of transmission 
"""
import numpy as np
import matplotlib.pyplot as plt

import commsim.tx as tx
import commsim.rx as rx
from commsim.channel import AWGN_Channel, Custom_Channel
import commsim.equaliser as eq

class CommSim:
    """
    Simulate communication with carrier synchronization
    """
    def __init__(self, mod, demod, pf, channel, lp, cs, eq=None, fc=10000, fs=44100):
        """
        @parameters:
        - mod:      an object of Modulator
        - demod:    an object of Demodulator
        - pf:       an object of PulseFilter
        - channel:  an object of Channel
        - lp:       a lowpass filter object
        - cs:       a carrier synchronizer
        - eq:       equaliser object
        - fc:       the carrier frequency in Hz
        - fs:       the sampling frequency in Hz
        """
        self.mod = mod
        self.demod = demod
        self.pf = pf
        self.channel = channel
        self.Fc = fc
        self.Fs = fs
        self.d_true = []
        self.d_pred = []
        self.N = 0
        self.error_rate = 0
        self.eq = eq
        self.lp = lp
        self.cs = cs

    def transmit(self, data, cs_enable=True, eq_enable=True, verbose=True):
        """
        simulate data transmission through the system.
        @parameters:
        - data: data array to be transmitted (must be compatible with modulation scheme)
        - cs_enable: bool, enable the use of carrierSynchronisation
        - eq_enable: bool, enable the use of equaliser (it will be designed anyways)
        - verbose: display results and graphs
        """

        # preamble begin
        if cs_enable:
            d_pre = np.random.randint(self.mod.M, size=20)
            self.transmit(d_pre, cs_enable=False, eq_enable=False, verbose=False) # shouldn't use eq: it's not designed!
            self.cs.syncPhase(self.demod_symbol, self.true_symbol)
            
            if eq: # using an equaliser, the step is only performed after cs is done (otherwise pointless)
                x_true = np.zeros(self.eq.gL)
                x_true[0] = 1
                x_base = self.pf.modFilter(x_true)
                x_pass = tx.mult_cos_carrier(x_base.real, self.Fc, self.Fs)
                
                y_pass = self.channel.pass_through(x_pass, noiseless=True) # we don't want the noise to affect our 'theoretical effective filter'
                
                yr_base = tx.mult_cos_carrier(y_pass, self.Fc, self.Fs)
                yi_base = tx.mult_sin_carrier(y_pass, self.Fc, self.Fs) 
                yr_lp = self.lp.filter(yr_base)
                yi_lp = self.lp.filter(yi_base) # here we compensate for the phase lag manually
                yr_demo= self.pf.demodFilter(yr_lp) # 20 is order of low pass filter, we do need gain to counter the attenuation
                yi_demo = self.pf.demodFilter(yi_lp) # also the filter introduces a phase delay
                demod_symbol = np.array([complex(r, i) for r, i in zip(yr_demo, yi_demo)])
                demod_symbol = self.cs.correctPhase(demod_symbol) # by now we should have carrier in sync
                gk = demod_symbol.real # we transimit impulse for the real part, hence look at the real part
                self.eq.design(gk) # this is magical... But it only works this way

        # preamble end

        # transmitter begin
        d_true = data
        x_true = self.mod.modulate(d_true) # modulatation symbols (verified)
        x_base = self.pf.modFilter(x_true)
        x_pass = tx.mult_sin_carrier(x_base.imag, self.Fc, self.Fs) + tx.mult_cos_carrier(x_base.real, self.Fc, self.Fs)
        # transmitter end

        # channel begin
        y_pass = self.channel.pass_through(x_pass)
        # channel end

        # receiver begin
        yr_base = tx.mult_cos_carrier(y_pass, self.Fc, self.Fs)
        yi_base = tx.mult_sin_carrier(y_pass, self.Fc, self.Fs)

        yr_lp = self.lp.filter(yr_base)
        yi_lp = self.lp.filter(yi_base) # we compensate for the delay in the filter

        yr_demo = self.pf.demodFilter(yr_lp) # 20 is order of low pass filter, we do need gain to counter the attenuation
        yi_demo = self.pf.demodFilter(yi_lp) # also the filter introduces a phase delay

        yr_rec = self.eq.equalise(yr_demo) if self.eq and eq_enable else yr_demo
        yi_rec = self.eq.equalise(yi_demo) if self.eq and eq_enable else yi_demo
        
        self.demod_symbol = np.array([complex(r, i) for r, i in zip(yr_rec, yi_rec)]) # we "export" this to be used in carrier synchronisation
        if cs_enable:
            self.demod_symbol = self.cs.correctPhase(self.demod_symbol)

        d_pred = self.demod.demodulate(self.demod_symbol)

        # receiver end

        # used by the CarrierSynchronizer to lock in the phase
        self.true_symbol = x_true

        # Analytics begin
        if verbose:
            self.d_true = d_true
            self.d_pred = d_pred # export the results
            self.x_pass = x_pass # export passband signal

            yr = [y.real for y in self.demod_symbol]
            yi = [y.imag for y in self.demod_symbol]
            print("Carrier Phase Compensation(radian): ", self.cs.phi)
            print(d_true[:10])
            print(d_pred[:10])
            print("Error Rate: ",get_error_rate(d_true, d_pred))
        
            fig, ax = plt.subplots(3,2, figsize=(20, 16))
            fig.tight_layout()

            ax[0][0].plot(x_true.real[:10], 'o', label="True (real)")
            ax[0][0].plot(yr[:10], 'o', label="Received (real)")
            ax[0][0].plot(x_true.imag[:10], 'o', label="True (imag)")
            ax[0][0].plot(yi[:10], 'o', label="Received (imag)")

            ax[0][0].set_title('Symbols Sequence')
            ax[0][0].grid(True)
            ax[0][0].legend()

            x_coord = [ s.real for s in self.mod.symbols ]
            y_coord = [ s.imag for s in self.mod.symbols ]
            ax[0][1].scatter(yr, yi, label="Received")
            ax[0][1].scatter(x_coord, y_coord, label="True")
            ax[0][1].grid(True)
            ax[0][1].axhline(y=0, color='k')
            ax[0][1].axvline(x=0, color='k')
            ax[0][1].set_title('Symbols Distribution')
            ax[0][1].legend()

            ax[1][0].plot(x_base.real[:10*self.pf.symbol_period], label='True (real)')
            ax[1][0].plot(yr_lp[:10*self.pf.symbol_period], label='Received (real)')
            ax[1][0].plot(x_base.imag[:10*self.pf.symbol_period], label='True (imag)')
            ax[1][0].plot(yi_lp[:10*self.pf.symbol_period], label='Received (imag)')
            ax[1][0].set_title('Baseband Signal (real part)')
            ax[1][0].legend()

            ax[1][1].plot(yr_base[:10*self.pf.symbol_period], label='before')
            ax[1][1].plot(yr_lp[:10*self.pf.symbol_period], label='after')
            ax[1][1].set_title('Before and after lowpass')
            ax[1][1].legend()

            ax[2][0].plot(x_pass[:10*self.pf.symbol_period], label='before')
            ax[2][0].plot(y_pass[:10*self.pf.symbol_period], label='after')
            ax[2][0].set_title('Before and after channel')
            ax[2][0].legend()
            
            if self.eq:
                ax[2][1].plot(self.eq.get_g(), 'o')
            else:
                ax[2][1].plot(eq.compute_effective_filter(self.pf, self.channel.h))
            ax[2][1].grid(True)
            ax[2][1].set_title('Effective Filter')

            # Analytics end


def get_error_rate(d, dk):
    return (len(d) - np.count_nonzero(d==dk[:len(d)])) / len(d)
