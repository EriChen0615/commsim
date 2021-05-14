class RRCPulseFilter:
    """
    Class for Root Raised Consine Pulse Filter in digital domain
    """
    def __init__(self, beta, symbol_period, L):
        """
        @parameters:
        - beta: roll-off ratio [0,1]
        - symbol_period: The transmission period of each symbol, in array length
            - To obtain the true symbol period, we need to know the system's fundamental frequency. In the case of
            - audio speaker with 44.1kHz, a symbol_period of 10 corresponds to a true symbol frequency of 4.41kHz
        - L: the length of the filter is (2L+1)

        @example:
        x = [x1, x2, ..., xN] # symbol to transmit, audio frequency 44.1kHz
        symbol_period = 4
        xs = [x1, 0, 0, 0, x2, 0, 0, 0, x3, ... ] # padded
        rrcFilter = [h[-3], h[-2], h[-1], h[0], h[1], h[2], h[3]] # L = 3
        xb = np.convolve(xs, rrcFilter) # baseband signal, tranmission rate = 44.1/4 = 11 kHz
        """
        self.symbol_period = symbol_period
        self.Ts = self.symbol_period
        self.beta = beta
        self.L = L

        self.ts_ind = np.array([i for i in range(-L, L + 1)]).astype(np.double) / self.Ts
        self.filter = np.zeros(2 * L + 1)
        for i, t in enumerate(self.ts_ind):
            if t == 0.0:
                self.filter[i] = 1.0 - self.beta + (4 * self.beta / np.pi)
            elif beta != 0 and t == 1 / (4 * self.beta):
                self.filter[i] = (self.beta / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                              (np.sin(np.pi / (4 * self.beta)))) + ((1 - 2 / np.pi) * (
                    np.cos(np.pi / (4 * self.beta)))))
            else:
                self.filter[i] = (np.sin(np.pi * t * (1 - self.beta)) + \
                                  4 * self.beta * (t) * np.cos(np.pi * t * (1 + self.beta))) / \
                                 (np.pi * t * (1 - (4 * self.beta * t) * (4 * self.beta * t)))
        self.filter /= np.sqrt(self.Ts)

        # self.filter = np.sinc(t) * np.cos(np.pi*beta*t) / (1 - (2*beta*t)**2) # this formula is not from the notes

        # self.filter = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
        #            4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
        #            (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)
        # self.filter =  c * (np.cos((1+self.beta)*np.pi*t) + a*np.sinc((1-self.beta)*np.pi*t) ) / (1 - (4*self.beta*t)**2 )

    def modFilter(self, x):
        """
        @parameters:
        - x: array of symbol to apply the filter in Modulation (i.e., with oversampling)

        @return:
        - xb: digital baseband signal
        """
        xp = np.zeros(len(x) * self.symbol_period).astype(np.cdouble)
        for i, v in enumerate(x):
            xp[i * self.symbol_period] = v  # expanding the signal before convolution
        return np.convolve(xp, self.filter)

    def demodFilter(self, x):
        """
        @parameters:
        - x: array of symbol to apply the demodulation filter (with matched filter)

        @return:
        - xb: digital baseband signal
        """
        xp = np.convolve(x, self.filter)[2 * self.L:]  # note that we account for the delay of demodulation here.
        return xp

def mult_sin_carrier(xb, fc, fs, f0=0, sqrt2=True):
    """
    multiply the signal by a sine carrier at to up-convert to fc
    @parameters:
    - xb: baseband signal
    - fc: carrier frequency in Hz
    - f0: frequency offset
    - fs: sampling frequency of the system (e.g., for audio could be at 44.1kHz)
    """
    k = np.arange(len(xb))
    carrier = -np.sin(2*np.pi*(fc+f0) * (k / fs))
    if sqrt2:
        carrier *= np.sqrt(2)
    return xb * carrier

def mult_cos_carrier(xb, fc, fs, f0=0, sqrt2=True):
    """
    multiply the signal by a sine carrier at to up-convert to fc
    @parameters:
    - xb: baseband signal
    - fc: carrier frequency in Hz
    - f0: frequency offset
    - fs: sampling frequency of the system (e.g., for audio could be at 44.1kHz)
    """
    k = np.arange(len(xb))
    carrier = np.cos(2*np.pi*(fc+f0) * (k / fs))
    if sqrt2:
        carrier *= np.sqrt(2)
    return xb * carrier

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
