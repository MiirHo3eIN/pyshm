import numpy as np 
from sklearn.preprocessing import StandardScaler
from dataShaper import shaper
import matplotlib.pyplot as plt
import seaborn as sns


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
    


def from_raw_to_scaled(train, valid):

    scaler = StandardScaler()
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    valid_scaled = scaler.transform(valid)
    with sns.plotting_context("poster"):
        plt.figure(figsize= (16, 4)) 
        plt.plot(train[0, :], color = 'green', label = 'Original Signal')
        plt.plot(train_scaled[0, :], color = 'red', label = 'Scaled Signal')
        plt.legend()

        plt.figure(figsize= (16, 4))
        plt.plot(valid[0, :], color = 'green', label = 'Original Signal')
        plt.plot(valid_scaled[0, :], color = 'red', label = 'Scaled Signal')
        plt.legend()
        plt.show()
    
    return train_scaled, valid_scaled



if __name__ == '__main__':
    N1 = 10_000
    N2 = 20_000
    fs = 100

    DataShaper = shaper(sequence_len = 100, stride = 100)
    # Create a random signal 

    

    x = np.arange(start= 0, stop = N1, step = 1)
    
    x_train , x_valid = np.split(x, 2)
    print(x_train.shape, x_valid.shape)
    
    


    x_train_org =  np.sin(2*np.pi*fs*x_train/(2*N1))
    x_valid_org =  np.sin(2*np.pi*fs*x_valid/(2*N1))

    plt.figure()
    plt.plot(x_valid_org[:100])
    plt.show()
    _ , _ = from_raw_to_scaled((x_train_org).reshape(-1, x_train_org.shape[-1]), x_valid_org.reshape(-1, x_valid_org.shape[-1]))
    exit()
    x_train = x_train_org.reshape(-1, 1)
    x_train_shaped = DataShaper(x_train)
    x_valid = x_valid_org.reshape(-1, 1)
    x_valid_shaped = DataShaper(x_valid)
    
    print(f"Train shape: {x_train_shaped.shape}, Valid shape: {x_valid_shaped.shape}")

    

    ztransform = Ztranform()
    x_tr_scaled = ztransform(x_train_shaped, input_type = 'train')
    print(x_tr_scaled.shape)
    x_val_scaled = ztransform(x_valid_shaped, input_type = 'test')
    print(x_val_scaled.shape)
    

    with sns.plotting_context("poster"):
        plt.figure(figsize= (16, 4))
        plt.plot(x_train_org[:100], color = 'green', label = 'Original Signal')
        plt.figure(figsize= (16, 4))

        plt.plot(x_train[0, :], color = 'green', label = 'Trained Scaled Signal')
        plt.plot(x_tr_scaled[0, :], color = 'red', label = 'Valid Scaled Signal')
        plt.legend()
        plt.show()
    