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
    

class EnergyFilter: 
    __doc__ = r""" 

    @brief: This class filters out the windows of input that has 
            energy below a certain threshold. This is to remove noisy 
            signals from the input for approaches that are effected 
            by white noise. 
    @param:
        - threshold: The threshold energy level
        - axis: The axis to compute the energy of the signal
                Input is a 2D signal with either I)[n_samples, n_features]
                or II) [n_features, n_samples]. In the first case, the axis = 1, 
                and in the second case, the axis = 0 must be set.  
            """
    
    def __init__(self, threshold: float = 0.1, axis:int = 0):
        
        
        self.threshold = threshold  
        self.axis = axis
    
    def _fit(self, x: np.array) -> np.array:
        __doc__ = r"""

        @breif: The method computes the energy of the input signal.
        @param:
            - x: The input signal
        @return:
            - The energy of the signal
        """  
        return np.sum(x**2, axis = self.axis)

    def _filter(self, x: np.array) -> np.array:
        __doc__ = r"""  
        
        @breif: The method filters out the windows of input that has 
                energy below a certain threshold. This is to remove noisy 
                signals from the input for approaches that are effected 
                by white noise.
        @param:
            - x: The input signal

        @return:
            - The filtered windows of the input signal

        """
        valid_energies = self._fit(x) > self.threshold 
        return x[valid_energies]
    

    def forward(self, x: np.array) -> np.array:
        return self._filter(x)
    
    def __call__(self, x) -> np.array:
        return self.forward(x)