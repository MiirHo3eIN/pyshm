import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from pyshm.dataShaper import shaper

def tester(): 
    N = 10_000
    fs = 100
    # Create a random signal 
    x = np.arange(N)
    x = np.sin(2*np.pi*fs*x/N)
    t = np.arange(len(x)) / fs
    x = x.reshape(-1, 1)
    data_shaper = shaper(window_size = 100, stride = 50)
    x_shaped = data_shaper(x)    
    colors = ['red', 'blue', 'yellow', 'orange', 'purple']
    
    
    t = t.reshape(-1, 1)
    t_shaped = data_shaper(t)
    
    with sns.plotting_context("poster"):
        plt.figure(figsize= (16, 4))
        plt.plot(t[:500], x[:500], color = 'green', label = 'Original Signal')
        for i, clr in zip(range(5), colors): 
            
            plt.title("Overlapp with half a second stride")
            plt.plot(t_shaped[i, :] ,x_shaped[i, :], color = clr, label = f'Shaped Signal {i}', linestyle = '--')
            plt.legend()

    data_shaper = shaper(window_size = 100, stride = 100)
    x_shaped = data_shaper(x)
    t_shaped = data_shaper(t)

    with sns.plotting_context("poster"):
        plt.figure(figsize= (16, 4))
        plt.plot(t[:500], x[:500], color = 'green', label = 'Original Signal')
        for i, clr in zip(range(5), colors): 
            plt.title("No Overlapp for a sequence of 1 second stride")
            plt.plot(t_shaped[i, :] ,x_shaped[i, :], color = clr, label = f'Shaped Signal {i}', linestyle = '--')
            plt.legend()
    
    
    plt.show()
if __name__ == '__main__':  

    tester()