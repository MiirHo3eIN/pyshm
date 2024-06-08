import numpy as np 
from sklearn.preprocessing import StandardScaler


class Ztranform:
    __doc__ = r"""
    @breif: A class that scales a 2D signal using the Z-transform.
    @param:
        - mean: The mean of the signal
        - std: The standard deviation of the signal
    """
    def __init__(self) -> None: 
        self.scalar = StandardScaler(with_mean=True, with_std=True)


    def _train(self, x: np.array) -> None:
        return self.scalar.fit(x)  

    def transform(self, x: np.array) -> np.array:        
        return self.scalar.transform(x)

    def forward(self, x: np.array, input_type: str) -> np.array:
        __doc__ = r"""

        @breif: The forward method that scales the input signal using the Z-transform.

        @param:
            - x: The input signal
            - input_type: The type of input signal. If 'train', the method will train the scalar and return the scaled signal.
                          If 'test', the method will scale the signal using the trained scalar.
        @return:    
            - The scaled signal
        """
        if input_type == 'train': 
            self._train(x)
            return self.transform(x)
        return self.transform(x)
    
    def __call__(self, x: np.array, input_type: str) -> np.array:
        return self.forward(x, input_type)
    





