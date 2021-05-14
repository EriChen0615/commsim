class _Modulator:
    """
    Base Class for modulator
    """
    def __init__(self):
        self.symbols = []

    def modulate(self, data):
        """
        @parameters:
        - data: list of integer from [0, log2(M)), data to be converted to QAM constellation
        """
        return np.array([self.symbols[d] for d in data])

class PSK_SymbolMod(_Modulator):
    """
    Class reprensting the Phase Shift Keying Modulation
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


class QAM_SymbolMod(_Modulator):
    """
    Class representing Quadrature Amplitude Modulation
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
            dist = self.symbols - cy
            d.append(np.argmin(np.linalg.norm([dist], axis=0)))
        return np.array(d)

    def mindist_symbol(self, s):
        dist = self.symbols - s
        return self.symbols[np.argmin(np.linalg.norm([dist], axis=0))]


class PSK_SymbolDemod(_Demodulator):
    """
    class representing Phase Shift Keying Demodulation
    """

    def __init__(self, M):
        """
        @parameters:
        - M: interger, number of PSK constellations
        """
        assert (M != 0) and (M & (M - 1) == 0)  # M must be a power of 2
        self.name = "PSK"
        self.M = M
        phi = np.arange(0, 2 * np.pi, 2 * np.pi / M)
        self.symbols = np.exp(1j * phi)


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
        assert (M != 0) and (M & (M - 1) == 0)  # M must be a power of 2
        self.M = M
        self.name = 'QAM'
        n = int(np.sqrt(M))
        self.n = n
        self.constellations = np.zeros((n, n), dtype=np.cdouble)
        self.symbols = []
        for p, q in itertools.product(np.arange(n), np.arange(n)):
            self.constellations[p][q] = np.cdouble(complex((-n / 2 + 0.5) + p, (-n / 2 + 0.5) + q))
            self.constellations[p][q] *= d
        self.symbols = np.reshape(self.constellations, -1)
