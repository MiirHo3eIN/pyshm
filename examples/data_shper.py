import numpy as np 
import matplotlib.pyplot as plt 
from pyshm.data_shaper import shaper

def tester(): 
    N = 10_000
    fs = 100
    # Create a random signal 
    x = np.arange(N)
    x = np.sin(2*np.pi*fs*x/N)
    t = np.arange(len(x)) / fs
    x = x.reshape(-1, 1)
    shaper = shaper(window_size = 100, stride = 50)
    x_shaped = shaper(x)    

    plt.plot(x[:200], color = 'green', label = 'Original Signal')
    plt.plot(x_shaped[0, :], color = 'black', label = 'Shaped Signal', linestyle = '--')

    plt.legend()
    plt.show()
if __name__ == '__main__':  

    tester()