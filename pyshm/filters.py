import numpy as np 
import torch 
from torch import nn

import scipy.signal as signal



class LowPassFilter: 
    __doc__ = r""" 

    @brief: This class is a simple wrapper around the scipy signal class 
            without any gradient capabilities for butterworth filters. 
            
            TODO: Implement a gradient version of this class deploying CNNs.
    @param: 
        - cutoff: The cutoff frequency of the filter 
        - fs: The sampling frequency of the signal 
        - order: The order of the filter
    """


    def __init__(self, cutoff: int, fs:int, order:int = 5): 
        self.cutoff = cutoff 
        self.fs = fs 
        self.order = order 
        self.sos = signal.butter(N = self.order, 
                                       Wn = self.cutoff, 
                                       btype='lowpass', 
                                       analog=False, 
                                       output='sos', 
                                       fs=self.fs)
    
    def forward(self, x:np.array, axis:int) -> np.array: 
        return signal.sosfiltfilt(self.sos, x, axis = axis)
    
    def __call__(self, x:np.array, axis:int) -> np.array:
        return self.forward(x, axis)