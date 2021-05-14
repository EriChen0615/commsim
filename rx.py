"""
This is a library built for simulating digital communication
Author: Jinghong Chen
Email: jc2124@cam.ac.uk

This file contains the helper classes to simulate a transmitter in digital data communication

A transmitter follows the following structure:

- Signal Flow
Received Signal   ->  Baseband Signal ->  Received Symbol    ->  [Equalised Symbol] -> Received Databits  -> [Decoded Data]

- System Diagram
CarrierMultiplier + low-pass filter ->  PulseFilter     ->  MatchedFilter      ->  Demodulator        ->  [Decoder]

"""
import numpy as np
import itertools
import scipy.signal as signal
from modulation import QAM_SymbolDemod, PSK_SymbolDemod
from sync import CoarseFreqSync, CarrierSync
from util import RRCPulseFilter, LP_Filter, mult_cos_carrier, mult_sin_carrier, merge2complex

class Receiver:
    """
    This is the high-level class for receiver. A receiver consists of
    - `pulseFilter`
    - `synchronizer`
    -  `demodulator`
    -  `decoder`
    """
    def __init__(self, Fs=44100, Fc=3000, Ts=0.01, ModScheme="QAM", ModPara={'M':4}, CodeScheme='', CodePara={}):
        """
        | Parameter             | Symbol         | Value                       | Note                          |
| --------------------- | -------------- | --------------------------- | ----------------------------- |
| Sampling Frequency    | `Fs`           | float (Hz)                  |                               |
| Carrier Frequency     | `Fc`           | float (Hz)                  |                               |
| Symbol Rate           | `Ts`           | float (s)                   | Symbol rate = `1/Ts`          |
| Modulation Scheme     | `ModScheme`    | string {"QAM","PSK","OFDM"} |                               |
| Modulation Parameters | `ModPara`      | dict {key: value}           | Value depends on `ModScheme`  |
| Coding Scheme         | `CodeScheme`   | string {"LDPC","CONV"}      |                               |
| Coding Parameters     | `CodePara`     | ditct {key: value}          | Value depends on `CodeScheme` |
| Pulse Filter          | `PulseFilter`  | PulseFilter Object          |                               |
| Synchronizer          | `Synchronizer` | Synchronizer Object         |                               |
        """
        self.Fs = Fs
        self.Fc = Fc
        self.Ts = Ts
        self.sps = int(self.Fs * self.Ts) # span for the pulseFilter

        # Demodulator
        if ModScheme == "QAM":
            self.demod = QAM_SymbolDemod(ModPara['M'])
        elif ModScheme == "PSK":
            self.demod = QAM_SymbolDemod(ModPara['M'])

        self.lowpassFilter = LP_Filter(np.arange(0,0.5/Ts,100), fcut=1.2/Ts, order=5, Fs=self.Fs)
        self.pulseFilter = RRCPulseFilter(0.35, self.sps, 5*self.sps) # default beta = 0.35, make variable later
       
        # synchronizers
        self.coarseSync = CoarseFreqSync(self.demod, self.Fs)
        self.fineSync = CarrierSync()

        self.log = {}
        


    def receive(self, y_raw, n, logging=True):
        '''
        @parameters:
        - y_rx: signal passed through the channel
        - n: number of symbols to receive
        '''
        # receiver begin

        # 1. Down-convert
        yr_base = mult_cos_carrier(y_raw, self.Fc, self.Fs)
        yi_base = mult_sin_carrier(y_raw, self.Fc, self.Fs) # notice we may have frequency offset here
        
        # 2. Pass through low pass filter
        yr_lp = self.lowpassFilter.filter(yr_base)
        yi_lp = self.lowpassFilter.filter(yi_base) # we compensate for the delay in the filter

        # 3. Pass through pulse filter
        yr_q = self.pulseFilter.demodFilter(yr_base) # 20 is order of low pass filter, we do need gain to counter the attenuation
        yi_q = self.pulseFilter.demodFilter(yi_base) # also the filter introduces a phase delay

        # 4. Coarse Frequency Correction
        # (yr_fsync, yi_fsync) = self.coarseSync.syncFreq(yr_q , yi_q)
        (yr_fsync, yi_fsync) = (yr_q , yi_q)

        # 5. Recover Symbols with matched filter
        yr_matched = match_filter(yr_fsync, n, self.sps) 
        yi_matched = match_filter(yi_fsync, n, self.sps)
        self.demod_symbol = np.array([complex(r, i) for r, i in zip(yr_matched, yi_matched)]) # we "export" this to be used in carrier synchronisation

        # 6. Fine Frequency Synchronization
        y_cs = self.fineSync.syncPhase(self.demod_symbol)
        # y_cs = self.demod_symbol

        # 7. Demodulation
        d_rec = self.demod.demodulate(y_cs)

        if logging:
            self.log['y_pass'] = y_raw
            self.log['y_base'] = merge2complex(yr_base, yi_base)
            self.log['y_q'] = merge2complex(yr_q, yi_q) # after applying pulse filter
            self.log['y_cfs'] = merge2complex(yr_fsync, yi_fsync) # after coarse frequency sync
            self.log['y_match'] = merge2complex(yr_matched, yi_matched)
            self.log['y_cs'] = y_cs # after carrier synchronization
            self.log['d_rec'] = d_rec
            self.log['coarseFreq'] = self.coarseSync.f0

        return d_rec
        # if cs_enable:
        #     self.demod_symbol = self.cs.correctPhase(self.demod_symbol)

        # if not preamble:
        #     self.y_cs = self.cs.fineSyncFreq(self.demod_symbol) if cs_enable else self.demod_symbol
        # else: # calibrate with true data to simulate a preabmle
        #     self.y_cs = self.cs.fineSyncFreq(self.demod_symbol, x_true) if cs_enable else self.demod_symbol
        # d_pred = self.demod.demodulate(self.y_cs)

        # receiver end
        

    
def match_filter(x, n, sps):
    """
    A matched filter
    @parameters:
    - x: incoming signal
    - n: number of symbols expected
    - sps: symbol period in number of samples
    """
    xm = []
    for i in range(n):
        xm.append(x[i*sps])
    return np.array(xm)
