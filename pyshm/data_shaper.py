import numpy as np 
import matplotlib.pyplot as plt


class shaper():
    def __init__(self, window_size: int, stride: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 

        self.stride = stride
        self.bias = True
        self.dim = window_size 

    def shape_extractor(self, x: np.array): 
        nrows, _ = x.shape
        
        # Calculate the number of overlapping sequences that can be extracted
        N0 = (nrows - self.dim) // self.stride + 1
        return N0
    
    def forward(self, x: np.array) -> np.array:

        N0 = self.shape_extractor(x)
        overlapping_sequences = [np.expand_dims(x[i*self.stride:i*self.stride+self.dim], axis=0) for i in range(N0)]
        overlapping_sequences = np.concatenate(overlapping_sequences, axis=0)
        print(overlapping_sequences.shape)
        return overlapping_sequences
    
    def __call__(self, x: np.array) -> np.array:
        return self.forward(x)




