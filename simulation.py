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

from tx import Transmitter
from rx import Receiver
from channel import AWGN_Channel, Custom_Channel, TimeDelay_Channel
#
# class CommSim:
#     """
#     Simulate communication with carrier synchronization
#     """
#     def __init__(self, mod, demod, pf, channels, lp, cfs, sbs, cs, eq=None, fc=10000, fs=44100):
#         """
#         @parameters:
#         - mod:      an object of Modulator
#         - demod:    an object of Demodulator
#         - pf:       an object of PulseFilter
#         - channels: a list of Channel objects
#         - lp:       a lowpass filter object
#         - cfs:      an object of Coarse Frequency Synchronizer
#         - sbs:      an object of Symbol Synchronizer
#         - cs:       a carrier synchronizer
#         - eq:       equaliser object
#         - fc:       the carrier frequency in Hz
#         - fs:       the sampling frequency in Hz
#         """
#         self.mod = mod
#         self.demod = demod
#         self.pf = pf
#         self.channels = channels
#         self.Fc = fc
#         self.Fs = fs
#         self.d_true = []
#         self.d_pred = []
#         self.N = 0
#         self.error_rate = 0
#         self.eq = eq
#         self.lp = lp
#         self.cs = cs
#         self.cfs = cfs
#         self.sbs = sbs
#
#     def pass_through_channel(self, x, noiseless=False, freqOff=True, timeDelay=True):
#         y = x
#         for ch in self.channels:
#             if noiseless and ch.name == "AWGN":
#                 continue
#             elif not freqOff and ch.name == "FrequencyOffset":
#                 continue
#             elif not timeDelay and ch.name == "TimeDelay":
#                 continue
#             else:
#                 y = ch.pass_through(y)
#         return y
#
#
#     def transmit(self, data, f0=0, cfs_enable=True, sbs_enable=True, cs_enable=True, eq_enable=True, verbose=True, ideal=False, preamble=False):
#         """
#         simulate data transmission through the system.
#         @parameters:
#         - data: data array to be transmitted (must be compatible with modulation scheme)
#         - f0: the frequency offset in Hz, model impecfection in the transmitter
#         - cfs_enable: bool, enable CoarseFrequencySynchronizer
#         - sbs_enable: bool, enable symbolSynchrnoizer
#         - cs_enable: bool, enable the use of carrierSynchronisation
#         - eq_enable: bool, enable the use of equaliser (it will be designed anyways)
#         - verbose: display results and graphs
#         - ideal:  bool, if true disable delay, frequency offset in the channel
#         - preamble: bool, send preamble (training) before the data
#         """
#
#         # preamble begin
#             # self.cs.syncPhase(self.demod_symbol, true_symbol=self.true_symbol)
#
#             # if eq: # using an equaliser, the step is only performed after cs is done (otherwise pointless)
#             #     x_true = np.zeros(self.eq.gL)
#             #     x_true[0] = 1
#             #     x_base = self.pf.modFilter(x_true)
#             #     x_pass = tx.mult_cos_carrier(x_base.real, self.Fc, self.Fs)
#
#             #     y_pass = self.pass_through_channel(x_pass, noiseless=True, freqOff=False) # we don't want the noise to affect our 'theoretical effective filter'
#
#             #     yr_base = tx.mult_cos_carrier(y_pass, self.Fc, self.Fs)
#             #     yi_base = tx.mult_sin_carrier(y_pass, self.Fc, self.Fs)
#             #     yr_lp = self.lp.filter(yr_base)
#             #     yi_lp = self.lp.filter(yi_base) # here we compensate for the phase lag manually
#             #     yr_demo= self.pf.demodFilter(yr_lp) # 20 is order of low pass filter, we do need gain to counter the attenuation
#             #     yi_demo = self.pf.demodFilter(yi_lp) # also the filter introduces a phase delay
#             #     demod_symbol = np.array([complex(r, i) for r, i in zip(yr_demo, yi_demo)])
#             #     demod_symbol = self.cs.correctPhase(demod_symbol) # by now we should have carrier in sync
#             #     gk = demod_symbol.real # we transimit impulse for the real part, hence look at the real part
#             #     self.eq.design(gk) # this is magical... But it only works this way
#
#         # preamble end
#
#         # transmitter begin
#         d_true = data
#         x_true = self.mod.modulate(d_true) # modulatation symbols (verified)
#         x_base = self.pf.modFilter(x_true)
#         x_pass = tx.mult_sin_carrier(x_base.imag, self.Fc, self.Fs) + tx.mult_cos_carrier(x_base.real, self.Fc, self.Fs)
#         # transmitter end
#
#         # channel begin
#         y_pass = self.pass_through_channel(x_pass) if not ideal else self.pass_through_channel(x_pass, freqOff=False, timeDelay=True)
#         # channel end
#
#         # receiver begin
#         yr_base = tx.mult_cos_carrier(y_pass, self.Fc, self.Fs, f0=f0)
#         yi_base = tx.mult_sin_carrier(y_pass, self.Fc, self.Fs, f0=f0) # notice we may have frequency offset here
#
#         yr_lp = self.lp.filter(yr_base)
#         yi_lp = self.lp.filter(yi_base) # we compensate for the delay in the filter
#
#
#         yr_q = self.pf.demodFilter(yr_base) # 20 is order of low pass filter, we do need gain to counter the attenuation
#         yi_q = self.pf.demodFilter(yi_base) # also the filter introduces a phase delay
#
#         (yr_fsync, yi_fsync) = self.cfs.syncFreq(yr_q , yi_q) if cfs_enable else (yr_q, yi_q)
#
#
#         yr_demo = []
#         yi_demo = []
#         if sbs_enable: # the symbol synchronizer acts as a matched filter as well
#             (yr_demo, yi_demo) = self.sbs.syncSymb(yr_fsync + 1j*yi_fsync, len(data))
#         else:
#             yr_matched = rx.match_filter(yr_fsync, len(data), self.pf.symbol_period)
#             yi_matched = rx.match_filter(yi_fsync, len(data), self.pf.symbol_period)
#             yr_demo = self.eq.equalise(yr_matched) if self.eq and eq_enable else yr_matched
#             yi_demo= self.eq.equalise(yi_matched) if self.eq and eq_enable else yi_matched
#
#         self.demod_symbol = np.array([complex(r, i) for r, i in zip(yr_demo, yi_demo)]) # we "export" this to be used in carrier synchronisation
#         # if cs_enable:
#         #     self.demod_symbol = self.cs.correctPhase(self.demod_symbol)
#
#         if not preamble:
#             self.y_cs = self.cs.fineSyncFreq(self.demod_symbol) if cs_enable else self.demod_symbol
#         else: # calibrate with true data to simulate a preabmle
#             self.y_cs = self.cs.fineSyncFreq(self.demod_symbol, x_true) if cs_enable else self.demod_symbol
#         d_pred = self.demod.demodulate(self.y_cs)
#
#         # receiver end
#
#         # used by the CarrierSynchronizer to lock in the phase
#         self.true_symbol = x_true
#
#         # Analytics begin
#         if verbose:
#             self.d_true = d_true
#             self.d_pred = d_pred # export the results
#             self.x_pass = x_pass # export passband signal
#
#             yr = np.array([y.real for y in self.y_cs])
#             yi = np.array([y.imag for y in self.y_cs])
#             print("Coarse Frequency Shift(Hz): ", self.cfs.f0)
#             print("Carrier Phase Compensation(radian): ", self.cs.phi)
#             print(d_true[:10])
#             print(d_pred[:10])
#             print("Error Rate: ",get_error_rate(d_true, d_pred))
#
#             fig, ax = plt.subplots(5,2, figsize=(16, 30))
#             # fig.tight_layout()
#
#             ax[0][0].plot(x_true.real[:10], 'o', label="True (real)")
#             ax[0][0].plot(yr[:10], 'o', label="Received (real)")
#             ax[0][0].plot(x_true.imag[:10], 'o', label="True (imag)")
#             ax[0][0].plot(yi[:10], 'o', label="Received (imag)")
#
#             ax[0][0].set_title('Symbols Sequence')
#             ax[0][0].grid(True)
#             ax[0][0].legend()
#
#             x_coord = [ s.real for s in self.mod.symbols ]
#             y_coord = [ s.imag for s in self.mod.symbols ]
#
#             colors = ['b','g','r','c']
#
#             for i in range(self.mod.M):
#                 ax[0][1].scatter(yr[d_true==i], yi[d_true==i], color=colors[i])
#                 ax[0][1].scatter(self.mod.symbols[i].real, self.mod.symbols[i].imag, marker='x', color=colors[i])
#                 pred_center = np.mean(yr[d_true==i]) + 1j*np.mean(yi[d_true==i])
#                 ax[0][1].plot([0, pred_center.real], [0, pred_center.imag], '-r')
#                 ax[0][1].set_xlim(-1.2, 1.2)
#                 ax[0][1].set_ylim(-1.2, 1.2)
#             # ax[0][1].scatter(yr, yi, label="Received")
#             # ax[0][1].scatter(x_coord, y_coord, label="True")
#             ax[0][1].grid(True)
#             ax[0][1].axhline(y=0, color='k')
#             ax[0][1].axvline(x=0, color='k')
#             ax[0][1].set_title('Signal Space Projection')
#             # ax[0][1].legend()
#
#             sp = tx.SquarePulseFilter(self.pf.symbol_period)
#             x_sq = sp.modFilter(x_true)
#             opt_x = np.arange(0, 10*self.pf.symbol_period, self.pf.symbol_period)
#             for i in opt_x:
#                 ax[1][0].axvline(i, c='r', ls='--')
#             ax[1][0].plot(x_sq.real[:10*self.pf.symbol_period], label='True (real)')
#             ax[1][0].plot(yr_q[:10*self.pf.symbol_period], label='Received (real)')
#             ax[1][0].plot(x_sq.imag[:10*self.pf.symbol_period], label='True (imag)')
#             ax[1][0].plot(yi_q[:10*self.pf.symbol_period], label='Received (imag)')
#             ax[1][0].set_title('Baseband Signal (real part)')
#             ax[1][0].legend()
#
#             ax[1][1].plot(yr_base[:10*self.pf.symbol_period], label='before')
#             ax[1][1].plot(yr_lp[:10*self.pf.symbol_period], label='after')
#             ax[1][1].set_title('Before and after lowpass')
#             ax[1][1].legend()
#
#             ax[2][0].plot(x_pass[:10*self.pf.symbol_period], label='before')
#             ax[2][0].plot(y_pass[:10*self.pf.symbol_period], label='after')
#             ax[2][0].set_title('Before and after channel')
#             ax[2][0].legend()
#
#
#             N = self.mod.M
#             y = yr_q + 1j*yi_q
#             y_sq = y ** N
#             psd_sq = np.fft.fftshift(np.abs(np.fft.fft(y_sq)))
#             f = np.linspace(-self.Fs/2.0, self.Fs/2.0, len(psd_sq))
#             print("Peak at ", f[np.argmax(psd_sq)], " Hz")
#             ax[3][0].plot(f, psd_sq)
#             ax[3][0].set_title(f"Spectrum After Raised to ${N}$ th Power")
#             ax[3][0].axvline(x=0, ls='-', color='k')
#             ax[3][0].axvline(x=f[np.argmax(psd_sq)], ls='--', color='r', label = f"Peak at {f[np.argmax(psd_sq)]:.2f} Hz")
#             ax[3][0].legend()
#
#             y_fsync = yr_fsync + 1j*yi_fsync
#             psd = np.fft.fftshift(np.abs(np.fft.fft(y_fsync**N)))
#             f = np.linspace(-self.Fs/2.0, self.Fs/2.0, len(psd))
#             ax[3][1].plot(f, psd)
#             ax[3][1].set_title(f"Spectrum After Coarse Frequency Sync (Raised {N})")
#             ax[3][1].axvline(x=0,ls='-', color='k')
#             ax[3][1].axvline(x=f[np.argmax(psd)], ls='--', color='r', label = f"Peak at {f[np.argmax(psd)]:.2f} Hz")
#             ax[3][1].legend()
#
#             if self.eq and not self.eq.get_g() is None:
#                 ax[2][1].plot(self.eq.get_g(), 'o')
#             ax[2][1].grid(True)
#             ax[2][1].set_title('Effective Filter')
#
#             # ax[4][0].plot(self.cs.freq_log)
#             # ax[4][0].set_title(f"Costa Loop Drift Frequency Estimate $beta={self.cs.beta}$ ")
#
#             # ax[4][0].scatter(self.demod_symbol.real, self.demod_symbol.imag, label="Recevied")
#             # ax[4][0].scatter(x_coord, y_coord, label="True")
#             for i in range(self.mod.M):
#                 ax[4][0].scatter(self.demod_symbol[d_true==i].real, self.demod_symbol[d_true==i].imag, color=colors[i])
#                 ax[4][0].scatter(self.mod.symbols[i].real, self.mod.symbols[i].imag, marker='x', color=colors[i])
#             ax[4][0].set_title("Before Fine Frequency Sync")
#             ax[4][0].grid(True)
#             ax[4][0].axhline(y=0, color='k')
#             ax[4][0].axvline(x=0, color='k')
#             # ax[4][0].legend()
#
#             ax[4][1].plot(self.cs.phi_log)
#             ax[4][1].set_title(f"Costa Loop Phase Error Estimate $alpha={self.cs.alpha}$")
#
#             # Analytics end
#
#         return get_error_rate(d_true, d_pred)

class Sim:
    def __init__(self, Fs=44100, Fc=3000, Ts=0.001, Channel=AWGN_Channel(20), ModScheme="QAM", ModPara={'M':4}, CodeScheme='', CodePara={}):
        """
        | Hyper Parameter       | Symbol       | Value                       | Note                          |
| :-------------------- | ------------ | --------------------------- | ----------------------------- |
| Sampling Frequency    | `Fs`         | float (Hz)                  |                               |
| Carrier Frequency     | `Fc`         | float (Hz)                  |                               |
| Symbol Rate           | `Ts`         | float (ms)                  | Symbol rate = `1/Ts`          |
| Modulation Scheme     | `ModScheme`  | string {"QAM","PSK","OFDM"} |                               |
| Modulation Parameters | `ModPara`    | dict {key: value}           | Value depends on `ModScheme`  |
| Coding Scheme         | `CodeScheme` | string {"LDPC","CONV"}      |                               |
| Coding Parameters     | `CodePara`   | ditct {key: value}          | Value depends on `CodeScheme` |
| Channel               | `Channel`    | Channel Object              |                               |
|                       |              |                             |                               |
        """
        self.tx = Transmitter(Fs=Fs, Fc=Fc, Ts=Ts, ModScheme=ModScheme, ModPara=ModPara, CodeScheme=CodeScheme, CodePara=CodePara)
        self.channel = Channel
        self.rx = Receiver(Fs=Fs, Fc=Fc, Ts=Ts, ModScheme=ModScheme, ModPara=ModPara, CodeScheme=CodeScheme, CodePara=CodePara)

    def transmit(self, N):
        """
        Transmit N data points that are randomly generated
        """
        data = np.random.randint(low=0,high=self.tx.mod.M,size=N)
        x_pass = self.tx.send(data)
        y_pass = self.channel.pass_through(x_pass)
        data_rec = self.rx.receive(y_pass, len(data))

        return (data, data_rec)


def get_error_rate(d, dk):
    return (len(d) - np.count_nonzero(d==dk[:len(d)])) / len(d)
