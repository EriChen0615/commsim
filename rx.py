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

class _Demodulator:
    """
    Base Class for demodulator
    """
    def __init__(self):
        pass
    
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
    
    def mindist_symbol(self, s):
        dist = self.symbols - s
        return self.symbols[np.argmin(np.linalg.norm([dist],axis=0))]

class PSK_SymbolDemod(_Demodulator):
    """
    class representing Phase Shift Keying Demodulation
    """
    def __init__(self, M):
        """
        @parameters:
        - M: interger, number of PSK constellations
        """
        assert (M != 0) and (M & (M-1) == 0) # M must be a power of 2
        self.name = "PSK"
        self.M = M
        phi = np.arange(0, 2*np.pi, 2*np.pi/M)
        self.symbols = np.exp(1j*phi)

class QAM_SymbolDemod(_Demodulator):
    """
    class representing Quadrature Amplitude Demodulation
    """
    def __init__(self, M, d=np.sqrt(2)):
        """
        @parameters:
        - M: integer, number of constellations. Should be a power of 2
        - d: coordinate distance between adjacent constellations
        """
        assert (M != 0) and (M & (M-1) == 0) # M must be a power of 2
        self.M = M
        self.name = 'QAM'
        n = int(np.sqrt(M))
        self.n = n
        self.constellations = np.zeros((n, n), dtype=np.cdouble)
        self.symbols = []
        for p, q in itertools.product(np.arange(n), np.arange(n)):
            self.constellations[p][q] = np.cdouble(complex((-n/2+0.5) + p, (-n/2+0.5) + q))
            self.constellations[p][q] *= d
        self.symbols = np.reshape(self.constellations, -1)
    
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

    def syncPhase(self, demod_symbol, true_symbol=[], mod='QAM'):
        """
        Workout the phae compensation given true symbol and demodulated symbol based on modulation scheme
        Note: this scheme assumes we know the true symbol
        @parameters:
        - demod_symbol: demodulated symbol from the preamble transmission
        - true_symobl:  true symbol in the preamble transmission, if provided, will be used as training signals
        - mod:          modulation used {'QAM', 'BPSK', 'QPSK',...} [only QAM implemented]
        """
        corrected = []
        if mod == 'QAM':
            if len(true_symbol): # if true symbols are provided, i.e, in training
                for x_pred, x_true in zip(demod_symbol, true_symbol):
                    x_pred *= np.exp(1j*self.phi)
                    e = x_true.real*x_pred.imag - x_true.imag*x_pred.real
                    self.phi -= self.Kp * e # negative feedback
                    if self.phi > 2*np.pi:
                        self.phi -= 2*np.pi
                corrected.append(x_pred * np.exp(1j*self.phi))
            else:
                for x_pred in demod_symbol: # decision-directed
                    x_pred *= np.exp(1j*self.phi)
                    e = x_pred.real*x_pred.imag - x_pred.imag*x_pred.real
                    self.phi -= self.Kp * e # negative feedback
                    if self.phi > 2*np.pi:
                        self.phi -= 2*np.pi
                    corrected.append(x_pred * np.exp(1j*self.phi)) # correction
        return corrected

    def correctPhase(self, demo_x):
        return demo_x * np.exp(1j*self.phi)

class FineFreqSync:
    """
    This is a class for fine frequency sync using a costa loop
    """
    def __init__(self, demod, fs, alpha=0.132, beta=0.00932):
        """
        @parameters:
        - demod : symbol demodulator
        - fs  : sampling frequency
        - alpha: costa loop gain
        - beta:  costa loop gain
        """
        self.demod = demod
        self.fs = fs
        self.phi = 0
        self.alpha = alpha
        self.beta = beta
        self.freq_log = []
        self.phi_log = []
    
    def fineSyncFreq(self, samples, x_true=[]):
        """
        @parameters:
        - samples: complex samples to be sync
        - x_true: the true symbols. Used in preamble
        """
        N = len(samples)
        # self.phi = 0 # for preamble
        freq = 0
        # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
        out = np.zeros(N, dtype=np.complex)
        error = None
        for i in range(N):
            out[i] = samples[i] * np.exp(-1j*self.phi) # adjust the input sample by the inverse of the estimated phase offset
            if len(x_true): # in training, true symbol provided
               if self.demod.name == 'PSK' and self.demod.M == 2:
                    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
               elif self.demod.name == 'QAM' and self.demod.M == 4: # data aided
                    error =  x_true[i].real*out[i].imag - x_true[i].imag*out[i].real 
            else:
                if self.demod.name == 'PSK' and self.demod.M == 2:
                    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
                elif self.demod.name == 'QAM' and self.demod.M == 4:
                    x_pred = self.demod.mindist_symbol(out[i])
                    error =  x_pred.real*out[i].imag - x_pred.imag*out[i].real

            # Advance the loop (recalc phase and freq offset)
            freq += (self.beta * error)
            self.freq_log.append(freq / 50.0 * self.fs)

            self.phi += freq + (self.alpha * error)
            self.phi_log.append(self.phi)

            # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
            # while self.phi >= 2*np.pi:
            #     self.phi -= 2*np.pi
            # while self.phi < 0:
            #     self.phi += 2*np.pi
        return out

class CoarseFreqSync:
    """
    This class defines a coarse frequency synchrnoizer, assuming we know the modulation scheme
    """
    def __init__(self, mod, Fs):
        self.mod = mod
        self.fs = Fs
        self.f0 = 0
    
    def syncFreq(self, yr, yi):
        # work out the frequency shift
        N = self.mod.M
        y = yr + 1j*yi
        y_sq = y ** N # raised to the power of N
        psd_sq = np.fft.fftshift(np.abs(np.fft.fft(y_sq)))
        f = np.linspace(-self.fs/2.0, self.fs/2.0, len(psd_sq))
        self.f0 = f[np.argmax(psd_sq)] / N # this is the frequency shift
        t = np.arange(0, len(y))/self.fs
        y_sync = y * np.exp(-1j*2*np.pi*self.f0*t) # compensate for the frequency shift
        return y_sync.real, y_sync.imag

class SymbolSync:
    """
    This class defines a symbol synchronizer
    """
    def __init__(self, demo, pf, Kp=0.3):
        """
        @parameters:
        - pf : a pulse shaping object
        """
        self.demo = demo
        self.pf = pf
        self.Kp = Kp
        self.mu_list = []

    def syncSymb(self, samples, n):
        """
        This function implements a Muller-Meuller Algorithm for symbol synchronization
        @parameters:
        - samples: collected samples
        - n: number of data to retrieve
        """
        # mu = 0 # initial estimate of phase of sample
        # out = np.zeros(len(samples) + 10, dtype=np.complex)
        # out_rail = np.zeros(len(samples) + 10, dtype=np.complex) # stores values, each iteration we need the previous 2 values plus current value
        # i_in = 0 # input samples index
        # i_out = 2 # output index (let first two outputs be 0)
        # samples_interpolated = signal.resample_poly(samples, 16, 1) # interpolate the samples
        # self.mu_list = []
        # while i_out < len(samples) and i_in < len(samples):
        #     # out[i_out] = samples[i_in + int(mu)] # grab what we think is the "best" sample
        #     out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
        #     out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        #     x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        #     y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        #     mm_val = np.real(y - x)
        #     # mu += self.pf.symbol_period + self.Kp*mm_val # manual gain of 0.3
        #     mu += self.Kp*mm_val
        #     mu %= self.pf.symbol_period//2
        #     # i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        #     i_in += self.pf.symbol_period
        #     # mu = mu - np.floor(mu) # remove the integer part of mu
        #     i_out += 1 # increment output index
        #     self.mu_list.append(mu)
        # out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
        # return out.real, out.imag

        mu = 0.0
        x_pred = np.zeros(n, dtype=np.cdouble)
        e = np.zeros(n)
        out = np.zeros(n, dtype=np.cdouble)
        sps = self.pf.symbol_period
        x_pred[0] = self.demo.mindist_symbol(samples[0])
        out[0] = samples[0]
        samples_interpolated = signal.resample_poly(samples, 16, 1) # linear interpolation

        for i in range(1, n):
            out[i] = samples_interpolated[i*sps*16 + int(mu*16)]
            x_pred[i] = self.demo.mindist_symbol(out[i])
            e[i] = x_pred[i-1].real * samples[i*sps + int(mu)].real \
                    - x_pred[i].real * samples[(i-1)*sps + int(mu)].real \
                    + x_pred[i-1].imag * samples[i*sps + int(mu)].imag \
                    - x_pred[i].imag * samples[(i-1)*sps + int(mu)].imag
            mu += self.Kp * e[i]
            if mu > self.pf.symbol_period:
                mu -= self.pf.symbol_period
            if mu < -self.pf.symbol_period:
                mu += self.pf.symbol_period
            self.mu_list.append(mu)
        self.e = e
        return out.real, out.imag



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