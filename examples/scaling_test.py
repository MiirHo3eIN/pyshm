__doc__ = """   

    @breif: A script that demonstrates the use of the scaling classes. 
    Here we show Z-transform scaling.
    To run this script, we used the MEMS sensor data from the data folder to validate 
    the scaling classes. However, you can use any data of your choice.

    Consider that before scaling the data, we must first mean center and shape the data.
    
    """



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization


VERBOSE = True
sns.set_style(style="whitegrid")


def plot(x, x_scaled):
        plt.figure(figsize= (16, 4))
        plt.title(f"Original Signal")
        plt.plot(x[0, :], color = 'green', label = 'Original Signal')
        plt.legend()
        plt.figure(figsize= (16, 4))
        plt.title(f"Scaled Signal")
        plt.plot(x_scaled[0, :], color = 'red', label = 'Trained Scaled Signal')
        plt.legend()


if __name__ == '__main__':

    sequence_length = 100 
    windpw_stride = 100 
    data_range = (-1, 1)


    data_shaper = shaper(sequence_len = sequence_length, stride = windpw_stride)
    MeanCenter = MeanCentering(axis = 0)
    ztransform = Ztranform()
    minmax_norm = Normalization(feature_range= data_range, clip = True)
    # Load the data
    df_train = pd.read_feather('../data/exp_2.feather')
    df_test  = pd.read_feather('../data/exp_3.feather')
    # Extract and Mean center the data, then plot it
    x_train = df_train['x'].values[4_000:]
    x_test = df_test['x'].values[4_000:]
    
    x_tr_center = MeanCenter(x_train)
    x_ts_center = MeanCenter(x_test)

    sns.set_style(style="whitegrid")

    plt.figure()
    plt.title("Mean Centered Train Data")
    plt.plot(x_tr_center.flatten('C').reshape(-1, 1))
    plt.show()

    
    # Shape the data 
    x_tr_shaped = data_shaper(x_train.reshape( -1, 1))
    x_ts_shaped = data_shaper(x_test.reshape( -1, 1))

    if VERBOSE == True:  
        print(f"Train Shape = {x_tr_shaped.shape}")
        print(f"Test Shape = {x_ts_shaped.shape}")

    x_tr_scaled = ztransform(x_tr_shaped, input_type = 'train')
    x_ts_scaled = ztransform(x_ts_shaped, input_type = 'test')

    if VERBOSE == True:  
        print(f"Scaled Train Shape = {x_tr_scaled.shape}")
        print(f"Test Shape = {x_ts_scaled.shape}") 

    x_tr_norm = minmax_norm(x_tr_scaled, input_type = 'train')
    x_ts_norm = minmax_norm(x_ts_scaled, input_type = 'test') 

    
    # plot(x_tr_shaped, x_tr_scaled)
    # plot(x_ts_shaped, x_ts_scaled)

    plot(x_tr_scaled, x_tr_norm)
    plot(x_ts_scaled, x_ts_norm)
    
    
    plt.show()
    