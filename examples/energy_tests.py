import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import EnergyFilter

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
    minmax_norm = Normalization(feature_range= data_range, clip = False)
    energy_filtering = EnergyFilter(threshold = 5, axis = 1) 
    # Load the data
    df_train = pd.read_feather('../data/exp_2.feather')
    t  = pd.read_feather('../data/exp_3.feather')
    # Extract and Mean center the data, then plot it
    x_train = df_train['x'].values[4_000:]

    
    x_tr_center = MeanCenter(x_train)

    sns.set_style(style="whitegrid")


    
    # Shape the data 
    x_tr_shaped = data_shaper(x_train.reshape( -1, 1))
    

    if VERBOSE == True:  
        print(f"Train Shape = {x_tr_shaped.shape}")
        

    x_tr_scaled = ztransform(x_tr_shaped, input_type = 'train')
    

    if VERBOSE == True:  
        print(f"Scaled Train Shape = {x_tr_scaled.shape}")
         

    x_tr_norm = minmax_norm(x_tr_scaled, input_type = 'train')
    
    x_energy = energy_filtering(x_tr_norm)

    if VERBOSE == True:
        print(f"Energy Shape = {x_energy.shape}")

    plt.figure(figsize= (16, 4))
    plt.title("Normalized Data") 
    plt.plot(x_tr_norm.flatten('C').reshape(-1, 1 ), color = 'green', label = 'Normalized Signal')  
    plt.legend()

    plt.figure(figsize= (16, 4))
    plt.title("Energy Signals") 
    plt.plot(x_energy.flatten('C').reshape(-1, 1 ), color = 'green', label = 'Energy Signal')
    plt.legend()
    plt.show()


    
    