import numpy as np
import itertools
import matplotlib.pyplot as plt

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
        return np.array([self.symbols[d] for d in data], dtype=np.complex)

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

class OFDM_SymbolMod(_Modulator):
    """This is the class for OFDM Modulator"""
    def __init__(self, Fs=44100, Ts=0.001, N=1024, L=50, ModScheme='QAM', ModPara={'M':4}, passband=True):
        """
        | Parameters            | Symbol      | Value                     | Note                                                |
| --------------------- | ----------- | ------------------------- | --------------------------------------------------- |
| Sampling Frequency    | `Fs`        | float (Hz)                |                                                     |
| Symbol Period         | `Ts`        | float (second)            | Bandwidth = `1/Ts`                                  |
| Block length          | `N`         | int                       | Subcarrier spacing = $1/(NT_s)$                     |
| Cyclic Prefix Length  | `L`         | int                       | Channel depend, if `0` set to `N/8`                 |
| Modulation Scheme     | `ModScheme` | String {'QAM', 'PSK',...} | The modulation scheme for each OFDM symbol          |
| Modulation Parameters | `ModPara`   | dict {}                   |                                                     |
| passband              | `passband`  | bool                      | If `True`, send passband signal, otherwise baseband |
        """
        assert 1/Ts < Fs/2

        self.N = N
        assert self.N % 2 == 0 # Only support even block length (for now)
        self.L = L
        self.sps = np.ceil(Ts * Fs)
        self.subcarriers = np.arange(N) / (N * Ts)  # sub-carriers frequencies
        self.passband = passband # if Ture, passband signal, if False, baseband signal

        # Modulation
        if ModScheme == 'QAM':
            self.symb_mod = QAM_SymbolMod(ModPara['M'])
        elif ModScheme == 'PSK':
            self.symb_mod = PSK_SymbolMod(ModPara['M'])
        else:
            raise RuntimeError('Unsupported Modulation Scheme! (QAM and PSK are available for OFDM)')

    def modulate(self, data):
        """Modulate an array of data (numpy array), with prefix appended"""
        symbols = self.symb_mod.modulate(data)

        if self.passband: # passband operation
            batch_num = len(symbols) // self.N + 1  # number of batches
            mod_data = np.zeros(batch_num * self.L + len(symbols))  # container for modulated data, faster than concatenate
            for i in range(batch_num): # split symbols into batches
                dft_batch = symbols[i*self.N: min((i+1)*self.N, len(symbols))]
                # ifft, convert from dft domain to time domain
                mod_data[i*(self.N+self.L) : min(( (i+1)*self.N+ i*self.L), len(mod_data)) ] = np.fft.ifft(dft_batch)
                # note that the prefix of 0s is already inplace
            return mod_data
        else: # baseband operation mod, [1:N//2] are conjugate of [N//2+1,:] (reduced efficiency)
            if self.N % 2 == 0: # even number
                data_per_batch = self.N//2 - 1 # baseband signal. Doesn't encode on the index 0 and N//2
                # padding in the end, so that symbols come in complete batches
                padded_data = None
                if len(data) % data_per_batch != 0:
                    padded_data = np.pad(data, (0, data_per_batch - len(symbols)%data_per_batch))
                else:
                    padded_data = data

                batch_num = len(padded_data) // data_per_batch
                time_symbols = np.zeros(batch_num * (self.N+self.L), dtype=np.complex)
                for i in range(batch_num):
                    dft_batch_symbols = np.zeros(self.N, dtype=np.complex)
                    # Useful signal in [1:self.N//2], 0 and N//2 not encoded
                    dft_batch_symbols[1:self.N//2] = self.symb_mod.modulate(padded_data[i*data_per_batch : (i+1)*data_per_batch])
                    # Negative frequencies are flip conjugate of positve ones
                    dft_batch_symbols[self.N//2+1:] = np.flip(np.conjugate(dft_batch_symbols[1:self.N//2]))
                    # note we have already included the cyclic prefix here
                    time_symbols[i*self.N+(i+1)*self.L : (i+1)*(self.N+self.L)] = np.fft.ifft(dft_batch_symbols)

                assert( np.isreal(time_symbols.all()) ) # Time symbols must be real for baseband transmission
                return time_symbols
            else: # odd number of block length
                return [] # TODO
            #
            # # each batch contain self.N // 2 + 1 data bits
            # data_per_batch = self.N // 2 + 1
            #
            # batch_num = len(symbols) // data_per_batch # number of batches
            # if len(symbols) % data_per_batch != 0:
            #     batch_num += 1 # One more batch for residual data
            #     symbols = np.pad(symbols, (0, data_per_batch - len(symbols) % data_per_batch))  # pad our symbols so that we don't need to worry about residual batch
            #     print(len(symbols))
            # # container for modulated data
            # mod_data = []
            # for i in range(batch_num):
            #     symbols_start_ind = i*data_per_batch
            #     symbols_end_ind = (i+1)*data_per_batch
            #     dft_batch = np.zeros(self.N, dtype=np.complex)
            #     # assign symbols to sub-carriers
            #     if self.N % 2 == 0: # even block length (carrying odd number of symbols)
            #         dft_batch[:self.N//2+1] += symbols[symbols_start_ind : symbols_end_ind]
            #         dft_batch[self.N//2:] += np.flip(np.conjugate( dft_batch[1:self.N//2+1] ))
            #         mod_data.extend(np.fft.ifft(dft_batch))
            #     else: # even block length (carrying even number of symbols)
            #         dft_batch[:self.N//2+1] += symbols[symbols_start_ind : symbols_end_ind]
            #         dft_batch[self.N//2+1:] += np.flip(np.conjugate( dft_batch[1:self.N//2+1] ))
            #         assert dft_batch[1:self.N//2+1] == np.flip(dft_batch[:-self.N//2:-1])
            #         dft_batch[0] = 0
            #         dft_batch[self.N//2] = 0
            #         mod_data.extend(np.fft.ifft(dft_batch))
            #
            # mod_data = np.array(mod_data)
            # print(mod_data[:100])
            # assert np.isreal(mod_data).all()  # make sure the we are transmitting real data
            # return mod_data



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

class OFDM_SymbolDemod(_Demodulator):
    """
    class for OFDM demodulation
    """
    def __init__(self, Fs=44100, Ts=0.001, N=1024, L=50, ModScheme='QAM', ModPara={'M':4}, passband=True):
        """
                | Parameters            | Symbol      | Value                     | Note                                                |
        | --------------------- | ----------- | ------------------------- | --------------------------------------------------- |
        | Sampling Frequency    | `Fs`        | float (Hz)                |                                                     |
        | Symbol Period         | `Ts`        | float (second)            | Bandwidth = `1/Ts`                                  |
        | Block length          | `N`         | int                       | Subcarrier spacing = $1/(NT_s)$                     |
        | Cyclic Prefix Length  | `L`         | int                       | Channel depend, if `0` set to `N/8`                 |
        | Modulation Scheme     | `ModScheme` | String {'QAM', 'PSK',...} | The modulation scheme for each OFDM symbol          |
        | Modulation Parameters | `ModPara`   | dict {}                   |                                                     |
        | passband              | `passband`  | bool                      | If `True`, send passband signal, otherwise baseband |
        """
        assert 1/Ts < Fs/2 # otherwise maximum data rate exceeded
        self.N = N
        assert self.N % 2 == 0 # Only support even block length now
        self.L = L
        self.sps = np.ceil(Ts * Fs)
        self.subcarriers = np.arange(N) / (N * Ts)
        self.passband = passband

        # Modulation
        if ModScheme == 'QAM':
            self.symb_demod = QAM_SymbolDemod(ModPara['M'])
        elif ModScheme == 'PSK':
            self.symb_demod = PSK_SymbolDemod(ModPara['M'])
        else:
            raise RuntimeError('Unsupported Modulation Scheme! (QAM and PSK are available for OFDM)')

    def demodulate(self, y):
        """Demodulate an array of incoming symbols (timedomain, numpy array)"""
        if self.passband: # passband operation
            pass # TODO
        else: # baseband operation
            if self.N % 2 == 0: # even block length
                data_per_batch = self.N // 2 - 1
                batch_num = len(y) // (self.N+self.L) # we assume data is padded
            dft_symbols = np.zeros(batch_num * data_per_batch, dtype=np.complex)
            for i in range(batch_num):
                time_batch_symbols = y[ i*(self.N+self.L) : (i+1)*self.N + i *self.L]
                # in baseband operation, only the positive frequency bins carry information.
                # The negative frequency bins (indexed > N//2) are conjugate to the positive ones
                dft_symbols[i*data_per_batch : (i+1)*data_per_batch] = np.fft.fft(time_batch_symbols)[1:self.N//2]

            return self.symb_demod.demodulate(dft_symbols)
