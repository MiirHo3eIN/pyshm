import numpy as np 
import matplotlib.pyplot as plt


class shaper:
    __doc__ = r"""
    
    @breif: A class that shapes a 1D signal into overlapping sequences of a fixed window size.
            The class accepts a 1d numpy array and returns a 2d numpy array of overlapping sequences.
            Input shape: (n_samples, 1), 
            Output shape: (n_samples, sequence_lenght) with stride = stride, 
            Note: Seq_len is considered as the features of the NN model.
    @param:
        - sequence_len: The size of each small window.
        - stride: The stride of the window between each sequence

    """


    def __init__(self, sequence_len: int, stride: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 

        self.stride = stride
        self.bias = True
        self.dim = sequence_len 

    def shape_extractor(self, x: np.array): 
        nrows, _ = x.shape
        
        # Calculate the number of overlapping sequences that can be extracted
        N0 = (nrows - self.dim) // self.stride + 1
        return N0
    
    def forward(self, x: np.array) -> np.array:

        N0 = self.shape_extractor(x)
        overlapping_sequences = [np.expand_dims(x[i*self.stride:i*self.stride+self.dim], axis=0) for i in range(N0)]
        overlapping_sequences = np.concatenate(overlapping_sequences, axis=0)
        overlapping_sequences = np.squeeze(overlapping_sequences, axis = -1)
        print(overlapping_sequences.shape)
        return overlapping_sequences
    
    def __call__(self, x: np.array) -> np.array:
        return self.forward(x)




class MeanCentering:
    __doc__ = r"""
    
    @breif: A class that center each window at 0. 
    This class is critical for MEMS sensor data 
    where the sensor is subject to drift over time.
    
    @param: axis : The axis to center the signal. Default is 1.

    """

    def __init__(self, axis) -> None:
        self.axis = axis

    def forward(self, x): 
        __doc__ = r"""
        @breif: The forward method that centers the signal at 0.
        @param: x: The input signal must be a 2D numpy array 
        with shape (n_samples, n_features) for axis = 1. 
        For axis = 0, the shape must be (n_features, n_samples).

        @return: The centered signal at 0.
        """
        return x - np.mean(x, axis = self.axis)

    def __call__(self, x): 
        
        return self.forward(x)