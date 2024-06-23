import torch 
from torch import nn
from torch.nn import functional as F


class FcEncoder(nn.Module): 

    def __init__(self,
        input_dim : int, 
        latent_dim: int,  
        activation: object = None):
        
        super().__init__()
        self.activation = activation
        match self.activation:
            case "linear":
                self.encoder_net = nn.Linear(in_features = input_dim, out_features = latent_dim)
            case "sigmoid":
                self.encoder_net = nn.Sequential(
                    nn.Linear(in_features = input_dim, out_features = latent_dim),
                    nn.Sigmoid()
                )
            case "tanh":
                self.encoder_net = nn.Sequential(
                    nn.Linear(in_features = input_dim, out_features = latent_dim),
                    nn.Tanh()
                )
            case "relu":
                self.encoder_net = nn.Sequential(
                    nn.Linear(in_features = input_dim, out_features = latent_dim),
                    nn.ReLU()
                )
            case "leaky_relu":
                raise ValueError("Leaky ReLU is not supported by GAP9") 
            case "elu":
                raise ValueError("ELU is not supported by GAP9")
        
        

    def forward(self, x): 
        x = self.encoder_net(x)
        return x



class FcDecoder(nn.Module): 

    def __init__(self,
        output_dim: int, 
        latent_dim: int,  
        activation: object = None): 

        super().__init__()
        self.activation = activation
        match self.activation:
            case "linear":
                self.decoder_net = nn.Linear(in_features = latent_dim, out_features = output_dim)
            case "sigmoid":
                self.decoder_net = nn.Sequential(
                    nn.Linear(in_features = latent_dim, out_features = output_dim),
                    nn.Sigmoid()
                )
            case "tanh":
                self.decoder_net = nn.Sequential(
                    nn.Linear(in_features = latent_dim, out_features = output_dim),
                    nn.Tanh()
                )
            case "relu":
                self.decoder_net = nn.Sequential(
                    nn.Linear(in_features = latent_dim, out_features = output_dim),
                    nn.ReLU()
                )
            case "leaky_relu":
                raise ValueError("Leaky ReLU is not supported by GAP9") 
            case "elu":
                raise ValueError("ELU is not supported by GAP9")
            
                                       

    def forward(self, x): 
        x = self.decoder_net(x)
        return x




class FcAutoEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        latent_dim: int,
        
        encoder_class: object = FcEncoder,
        encodeer_activation: str = "linear",
        decoder_class: object = FcDecoder,
        decoder_activation: str = "linear",

        device = 'cpu'
    ): 
        super().__init__()
        
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.device     = device
        self.encoder_activation = encodeer_activation
        self.decoder_activation = decoder_activation
        
        # Define the encoder and decoder
        self.encoder = encoder_class(input_dim = self.input_dim, 
                                     output_dim = self.latent_dim, 
                                     activation = self.encoder_activation)
        
        self.decoder = decoder_class(latent_dim = self.latent_dim, 
                                     output_dim = self.input_dim, 
                                     activation = self.decoder_activation)


    def forward(self, input_tensor):
        
        latent_space = self.encoder(input_tensor) # should be #of samples x #of features
        #latent_space = F.sigma(latent_space) 
        reconstructed_space = self.decoder(latent_space) # return the original space 
        
        return reconstructed_space
