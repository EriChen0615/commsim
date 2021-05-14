from simulation import Sim
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim = Sim()
    data, data_rec = sim.transmit(10)
    print(data)
    print(data_rec)