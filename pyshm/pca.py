import torch 
from torch import svd

from .utils import MSE
class PCA:
    def __init__(self, preserved_energy):
        self.energy = preserved_energy


    

    def fit(self, x): 
        __doc__ = r"""
            This method computes the number of components needed to reach 
            the entered preserved energy. 
            It returns eigen values and eigen vectors using svd method. 
            Other methods are also suggested in the literature; However, 
            deploying SVD is the most optimum way of computing variance 
            matrix for eigen equation. 
        """
        # Accept the dataset Matrix and compute SVD
        U, S, V = svd(x)
        # Extract the number of components 
        # to have the minimum threshold entered by the user.
        lcumsum = torch.cumsum(S/torch.sum(S), dim=0)
        k_components = len(lcumsum[lcumsum < self.energy])
        
        # yield eigenVectors while keeping them as the attributes of the class. 
        self.n_components = k_components + 1
        # Extract eigenValue and eigenVectors
        self.eigen_vectors = V[:, :self.n_components]
        
        return self.n_components, self.eigen_vectors 
    

    def project_(self, x):
        __doc__ = r"""
            Projecting an input data to latent space. 
        """
        return torch.mm(x, self.eigen_vectors)

    def reconstruct_(self, x):
        __doc__ = r"""
            This method reconstruct the  
        """
        return torch.mm(self.eigen_vectors.T, x)

    def eval(self, x):
        __doc__ = r"""
            This method is to reconstruct the input file and compute the 
            error between original and the reconstructed signal. 
            It shall also return the reconstructed signals as well for 
            visualization purposes.

        """
        
        x_latent = self.project_(x)
        x_reconstructed = self.reconstruct_(x_latent)

        # Compute the difference
        mse_error = MSE(x, x_reconstructed)

        return x_reconstructed, mse_error

