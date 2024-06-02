import numpy as np 
from pyshm.filters import LowPassFilter
# Plot the results 

import matplotlib.pyplot as plt 


def low_pass_filter_test_time_series(): 

    N = 500
    fs = 100
    # Create a random signal 
    x = np.arange(N)
    x = np.sin(2*np.pi*fs*x/N)
    t = np.arange(len(x)) / fs
    # Add some noise to the signal 
    x += 0.1*np.random.randn(500) 

    # Create a low pass filter 
    lpf = LowPassFilter(cutoff = 25, fs = 100, order = 4)

    # Apply the filter to the signal 
    y = lpf(x, axis = 0)

    plt.plot(t , x , label = 'Original Signal', color = 'green')
    plt.plot(t , y , label = 'Filtered Signal', color = 'black', linestyle = '--')
    plt.legend()
    plt.show()


def low_pass_filter_test_window(): 

    N = 500
    fs = 100
    # Create a random signal 
    x = np.arange(N)
    x = np.sin(2*np.pi*fs*x/N)
    t = np.arange(len(x)) / fs
    # Add some noise to the signal 
    x += 0.1*np.random.randn(500) 

    x = x.reshape(5, 100)
    # Create a low pass filter 
    lpf = LowPassFilter(cutoff = 25, fs = 100, order = 4)

    # Apply the filter to the signal 
    y = lpf(x, axis = 1)

    plt.plot(t[:100], x[0, :], label = 'Original Signal', color = 'green')
    plt.plot(t[:100], y[0, :], label = 'Filtered Signal', color = 'black', linestyle = '--')
    plt.legend()
    plt.show()


if __name__ == '__main__': 
    low_pass_filter_test_window()