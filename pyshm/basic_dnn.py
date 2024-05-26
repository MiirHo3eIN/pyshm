__doc__ = r""" 

    This script contains the basic tools to work with 
    Deep Neural Network models. 
    Basic models like a single layer of CNN - LSTM joined 
    by their activation and batch normalization are 
    defined as different methods. 
"""
import torch 
from torch import nn


def get_activation_layer(activation:str) -> None: 
    __doc__ = r""" 
    Return the activation function that is required by the user.

    Parameters
    ----------
    activation : str
        Name of the activation layer.
    """


    match activation: 
        case"Relu" : 
            activation =  nn.ReLU()
        case "ELU":
            activation = nn.ELU()
        case "tanh":
            activatio = nn.Tanh()
        case "sigmoid": 
            activator = nn.Sigmoid()
        case _:
            raise ValueError("The given activation layer is not supported")

def conv_1d(in_channels: int, 
            out_channel: int, 
            kernel_size: int, 
            stride: int, 
            padding: str, 
            activation:str, 
            *args, **kwargs) -> nn.Sequential:
    __doc__ = r""" 
    Implementing a single convolutional layer with batch normalization and ELU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels. In the input layer this is equal to the number of the sensors used. 
    out_channel : int   
        Number of output channels. Modyfing this parameter effect the spatial relation between the sensors.   
    kernel_size : int
        Kernel size of the convolutional layer. 
    stride : int
        Stride of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    padding : str
        Padding of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    *args :
        Variable length argument list.
    **kwargs :
        Arbitrary keyword arguments.
    """

    

    return nn.Sequential( 
        nn.Conv1d(in_channels, out_channel, kernel_size, stride, padding, *args, **kwargs), 
        nn.BatchNorm1d(out_channel), 
        get_activation_layer(activation=activation)
        )

