from pyshm.autoencoders import FcAutoEncoder
import torch 
import torch.nn as nn
from torchinfo import summary





def main(): 
    # define a small batch of data 

    batch_size = 10
    input_dim = 10
    latent_dim = 3
    data = torch.randn(batch_size, input_dim)

    print(data.size())

    # define the autoencoder: 
    autoencoder = FcAutoEncoder(input_dim= input_dim, 
                                latent_dim=latent_dim, 
                                encoder_activation= "linear",
                                decoder_activation= "linear")
    # print the model 
    summary(autoencoder, input_size = data.size(), verbose = 1, depth = 5)


if __name__ == "__main__": 
    main()

