import numpy as np 
from sklearn.preprocessing import StandardScaler
from dataShaper import shaper, MeanCentering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

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

    print("Statistical properties of the signal before scaling:" ) 
    print(np.mean(train, axis = 1) )
    print(np.std(train, axis=1) )

    scaler = StandardScaler()
    scaler.fit(train)

    train_scaled = scaler.fit_transform(train)

    print("Statistical properties of the signal before scaling:" ) 
    print(np.mean(train_scaled, axis = 1) )
    print(np.std(train_scaled, axis=1) )
    
    valid_scaled = scaler.transform(valid)

    # print(f"Mean of the train signal: {scaler.mean_}, Standard deviation of the train signal: {np.sqrt(scaler.var_)}")
    with sns.plotting_context("poster"):
        plt.figure(figsize= (16, 4)) 
        plt.plot(train[0, :], color = 'green', label = 'Original Signal')
        plt.plot(train_scaled[0, :], color = 'red', label = 'Scaled Signal')
        plt.legend()

        # plt.figure(figsize= (16, 4))
        # plt.plot(valid[0, :], color = 'green', label = 'Original Signal')
        # plt.plot(valid_scaled[0, :], color = 'red', label = 'Scaled Signal')
        # plt.legend()
        flatten_data = train_scaled.flatten('C').reshape(-1, 1)
        print(flatten_data.shape)
        plt.figure()
        plt.plot(flatten_data[:, 0])

        plt.show()
    
    return train_scaled, valid_scaled



if __name__ == '__main__':
    N1 = 10_000
    N2 = 20_000
    fs = 100

    DataShaper = shaper(sequence_len = 100, stride = 100)
    MeanCenter = MeanCentering(axis = 0)
    df_test = pd.read_feather('../data/exp_2.feather')

    x = df_test['x'].values[4_000:]
    x_center = MeanCenter(x)

    plt.figure()
    plt.plot(x_center.flatten('C').reshape(-1, 1))
    plt.show()

    exit()
    print(x.shape)
    x_shaped = DataShaper(x.reshape( -1, 1))
    print(x_shaped.shape)
    x_centered = MeanCenter(x_shaped)
    # _ , _ = from_raw_to_scaled(x_shaped, x_shaped)
    # # exit()
    # x_train = x_train_org.reshape(-1, 1)
    # x_train_shaped = DataShaper(x_train)
    # x_valid = x_valid_org.reshape(-1, 1)
    # x_valid_shaped = DataShaper(x_valid)
    
    # print(f"Train shape: {x_train_shaped.shape}, Valid shape: {x_valid_shaped.shape}")

    

    ztransform = Ztranform()
    x_tr_scaled = ztransform(x_centered, input_type = 'train')
    print(x_tr_scaled.shape)


    exit()
    # x_val_scaled = ztransform(x_valid_shaped, input_type = 'test')
    # print(x_val_scaled.shape)
    

    with sns.plotting_context("poster"):
        # plt.figure(figsize= (16, 4))
        # plt.plot(x_centered[0, :], color = 'green', label = 'Original Signal')
        plt.figure(figsize= (16, 4))

        plt.plot(x_centered[0, :], color = 'green', label = 'Original Signal')
        plt.plot(x_tr_scaled[0, :], color = 'red', label = 'Trained Scaled Signal')
        plt.legend()
        plt.show()
    