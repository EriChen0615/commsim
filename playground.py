from simulation import Sim
from channel import TimeDelay_Channel, FrequencyOffset_Channel
import matplotlib.pyplot as plt
import numpy as np

from modulation import OFDM_SymbolMod, QAM_SymbolMod, PSK_SymbolMod, OFDM_SymbolDemod

if __name__ == '__main__':
    sim = Sim(ModScheme='QAM', ModPara={'M':4})
    data, data_rec = sim.transmit(10000, analytics=False)

    block_length = 256
    ofdm_mod = OFDM_SymbolMod(N=256, L=0, passband=False) # no prefix situation
    ofdm_demod = OFDM_SymbolDemod(N=256, L=0, passband=False)

    time_symbols = ofdm_mod.modulate(data)
    pred_data = ofdm_demod.demodulate(time_symbols)

    print(data[:100])
    print(pred_data[:100])
    assert ( (data[:10000] == pred_data[:10000]).all() ) # verify modulation and demodulation works






