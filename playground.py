from simulation import Sim
from channel import TimeDelay_Channel, FrequencyOffset_Channel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim = Sim()
    data, data_rec = sim.transmit(100)
    print(data)
    print(data_rec)
