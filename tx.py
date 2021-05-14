"""
This is a library built for simulating digital communication
Author: Jinghong Chen
Email: jc2124@cam.ac.uk

This file contains the helper classes to simulate a transmitter in digital data communication

A transmitter follows the following structure:

- Signal Flow
Data Bits         ->  [Coded Bits] ->  Symbols          ->  Baseband Signal  -> Passband Signal

- System Diagram
Data Generator    ->  [Encoder]    ->  SymbolModulator  ->  PulseFilter      -> CarrierMultiplier
"""


import numpy as np
import itertools
from util import mult_sin_carrier, mult_cos_carrier, RRCPulseFilter
from modulation import QAM_SymbolMod, PSK_SymbolMod

def generate_data(N, p1=0.5):
    """
    Generate binary data to transmit.
    @parameters:
    - N:  size of data
    - p1: probability of generating '1'
    """

    d = np.random.rand(N)
    return np.where(d<=p1, 1, 0)

class Transmitter:
    def __init__(self, Fs=44100, Fc=3000, Ts=0.01, ModScheme="QAM", ModPara={'M':4}, CodeScheme='', CodePara={}):
        """
        A transmitter consists of a `encoder`, a `modulator` and a `pulseFilter` 

The default constructor of a simulation system is

| Parameter             | Symbol        | Value                       | Note                          |
| --------------------- | ------------- | --------------------------- | ----------------------------- |
| Sampling Frequency    | `Fs`          | float (Hz)                  |                               |
| Carrier Frequency     | `Fc`          | float (Hz)                  |                               |
| Symbol Rate           | `Ts`          | float (ms)                  | Symbol rate = `1/Ts`          |
| Modulation Scheme     | `ModScheme`   | string {"QAM","PSK","OFDM"} |                               |
| Modulation Parameters | `ModPara`     | dict {key: value}           | Value depends on `ModScheme`  |
| Coding Scheme         | `CodeScheme`  | string {"LDPC","CONV"}      |                               |
| Coding Parameters     | `CodePara`    | ditct {key: value}          | Value depends on `CodeScheme` |
| Pulse Filter          | `PulseFilter` | PulseFilter Object          |                               |
        """
        self.Fs = Fs
        self.Fc = Fc
        self.Ts = Ts
        self.sps = int(self.Fs * self.Ts) # span for the pulseFilter

        self.encoder = None # encoder TODO

        # Modulator
        if ModScheme == "QAM":
            self.mod = QAM_SymbolMod(ModPara['M'])
        elif ModScheme == "PSK":
            self.mod = QAM_SymbolMod(ModPara['M'])
        
        # Pulse Filter
        self.pulseFilter = RRCPulseFilter(0.35, self.sps, 5*self.sps) # default beta = 0.35, make variable later
    
    def send(self, d_raw):
        """
        turn raw data into modulated passband signal
        - d_raw: data to transmit, take value from 0 to M-1
        """
        x_true = self.mod.modulate(d_raw)
        x_base = self.pulseFilter.modFilter(x_true)
        x_pass = mult_sin_carrier(x_base.imag, self.Fc, self.Fs) + mult_cos_carrier(x_base.real, self.Fc, self.Fs)
        return x_pass
        


        


        