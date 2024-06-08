import numpy as np
import tsaug
import torch



def data_augmentation(x: np.array) -> torch.Tensor: 


    X_aug  = torch.tensor(tsaug.AddNoise(scale=0.001).augment(x), dtype = torch.float32, device = "cpu")
    X_aug2 = torch.tensor(tsaug.AddNoise(scale=0.002).augment(x), dtype = torch.float32, device = "cpu")
    X_aug3 = torch.tensor(tsaug.AddNoise(scale=0.003).augment(x), dtype = torch.float32, device = "cpu")
    X_aug4 = torch.tensor(tsaug.Convolve(window="flattop", size=100).augment(x), dtype = torch.float32, device = "cpu")
    X_aug5 = torch.tensor(tsaug.Reverse().augment(x), dtype = torch.float32, device = "cpu")
    X_aug6 = torch.tensor(tsaug.Convolve(window="hann", size=100).augment(x), dtype = torch.float32, device = "cpu")
    X_aug7 = torch.tensor(tsaug.AddNoise(scale=0.001).augment(X_aug5.numpy()), dtype = torch.float32, device = "cpu")

    train_data_augmented = torch.cat([x, X_aug, X_aug2, X_aug3, X_aug4, X_aug5, X_aug6, X_aug7], dim=0)

    return train_data_augmented