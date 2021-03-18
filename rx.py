"""
This is a library built for simulating digital communication
Author: Jinghong Chen
Email: jc2124@cam.ac.uk

This file contains the helper classes to simulate a transmitter in digital data communication

A transmitter follows the following structure:

- Signal Flow
Received Singal   ->  Baseband Signal ->  Received Symbol    ->  [Equalised Symbol] -> Received Databits  -> [Decoded Data]

- System Diagram
CarrierMultiplier + low-pass filter ->  PulseFilter     ->  MatchedFilter      ->  Demodulator        ->  [Decoder]

"""
import numpy as np
import itertools
import scipy.signal as signal

class QAM_SymbolDemod:
    """
    class representing Quadrature Amplitude Demodulation
    """
    def __init__(self, M, d=1):
        """
        @parameters:
        - M: integer, number of constellations. Should be a power of 2
        - d: coordinate distance between adjacent constellations
        """
        assert (M != 0) and (M & (M-1) == 0) # M must be a power of 2
        self.M = M
        n = int(np.sqrt(M))
        self.n = n
        self.constellations = np.zeros((n, n), dtype=np.cdouble)
        self.symbols = []
        for p, q in itertools.product(np.arange(n), np.arange(n)):
            self.constellations[p][q] = np.cdouble(complex((-n/2+0.5) + p, (-n/2+0.5) + q))
            self.constellations *= d
        self.symbols = np.reshape(self.constellations, -1)

    def demodulate(self, y):
        """
        @parameters:
        - y: complex symbols received
        """
        d = []
        for cy in y:
            dist = self.symbols-cy
            d.append(np.argmin(np.linalg.norm([dist], axis=0)))
        return np.array(d)

def lowpass_filter(x, Wn, fs=44100):
    sos = signal.butter(20, Wn , 'low', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, x)
    return filtered

class LP_Filter:
    """
    This class implements a lowpass filter
    """
    def __init__(self, fsig, fcut, order=20, Fs=44100):
        """
        @parameters:
        - fsig  : frequencies of the signal used for compensation
        - fcut  : cutoff frequency for the lowpass filter
        - order : order of the lowpass filter
        - fs    : sampling frequency of the system       
        """
        self.fcut = fcut
        self.order = order
        self.Fs = Fs
        self.sos = signal.butter(order, self.fcut, 'low', fs=self.Fs, output='sos')
        b, a = signal.butter(order, self.fcut, 'low', fs=self.Fs)
        if fsig[0] == 0:
            fsig = fsig[1:]
        w, mag, phase = signal.dbode((b, a, 1/Fs), w=fsig/Fs) # compute mag
        self.K = np.mean(1/(10**(mag/20)))
        w, gd = signal.group_delay((b,a), w=fsig/Fs, fs=Fs) # compute delay
        self.lag = int(np.mean(gd))


    def filter(self, x):
        """
        @parameters:
        - x : data to be filtered
        """
        y = signal.sosfilt(self.sos, x)
        return self.K*y[self.lag:]

class CarrierSync:
    """
    This clss implements a carrier synchronizer based on a digital version of PLL
    """
    def __init__(self, Kp=1, L=20):
        """
        We assume the preamble be transmitted using 4-QAM
        @parameters:
        - commsim : a simulation object which will be used during preamble transmission
        - Kp      : feedback gain for PLL
        - L       : length of the preamble
        """
        self.Kp = Kp
        self.phi = 0 # initial phase guess

    def syncPhase(self, demod_symbol, true_symbol, mod='QAM'):
        """
        Workout the phae compensation given true symbol and demodulated symbol based on modulation scheme
        Note: this scheme assumes we know the true symbol
        @parameters:
        - demod_symbol: demodulated symbol from the preamble transmission
        - true_symobl:  true symbol in the preamble transmission
        - mod:          modulation used {'QAM', 'BPSK', 'QPSK',...} [only QAM implemented]
        """
        if mod == 'QAM':
            for x_pred, x_true in zip(demod_symbol, true_symbol):
                x_pred *= np.exp(1j*self.phi)
                e = x_true.real*x_pred.imag - x_true.imag*x_pred.real
                self.phi -= self.Kp * e # negative feedback
                if self.phi > 2*np.pi:
                    self.phi -= 2*np.pi

    def correctPhase(self, demo_x):
        return demo_x * np.exp(1j*self.phi)