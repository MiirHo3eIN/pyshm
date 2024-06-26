import torch 
import numpy as np 


from pyshm.pca import PCA


def main(): 
    
    batch_size = 50
    input_dim = 25
    pca_ = PCA(0.50)
    x_data = torch.randn(batch_size, input_dim)
    print(x_data.size())

    pca_n_comp, pca_eigen_vector = pca_.fit(x_data)
    
    print(pca_eigen_vector.T)
    print(pca_eigen_vector.T.shape)


if __name__ == "__main__":

    main()