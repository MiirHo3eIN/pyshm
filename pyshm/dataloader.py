import glob 
import pandas as pd
import torch 
from torch.utils.data import Dataset
# Profiling packages
import tracemalloc
import time

# TODO: Document the script 



class dataInitFeather: 
    def __init__(self,data_path:str,  dataset_type:str):
        self.dataset_type = dataset_type 
        self.data_path = data_path 

    def data_init(self) -> list: 
        match self.dataset_type:
            case "train":
                return [2,3,4]
            case "test":
                return [1] 
            case "validation":
                return [5]
            case "anomaly":
                # We add 5 to the anomaly level since the first 5 experiments are normal cases.
                return list(range(6, 15))
            case _:
                raise ValueError("Invalid type of data")
            
    def forward(self):
        experimenets = self.data_init()
        return experimenets
    def __repr__(self):
        return f"Data Initilization for {self.dataset_type} data"
    def __str__(self):
        return f"Data Initilization for {self.dataset_type} data"
    def __call__(self):
        return self.forward()


class dataInitHealthy(dataInitFeather):
    def __init__(self, data_path, dataset_type:str):
        super().__init__(data_path, dataset_type)

    def df_init(self) -> pd.DataFrame: 
        df_ = []
        for experiment in self.data_init():
            data_path = f"{self.data_path}/exp_{experiment}.feather"
            df_.append(pd.read_feather(f"{data_path}"))
        return pd.concat(df_)
    def forward(self):
        print(self.__str__())
        return self.df_init()
    def __call__(self):
        return self.forward()   

class dataInitAnomaly(dataInitFeather):
    def __init__(self, data_path:str):
        super().__init__(data_path, "anomaly")
    
    def df_init(self) -> dict: 
        df_ = {}
        for anomaly in self.data_init():
            data_path = f"{self.data_path}/exp_{anomaly}.feather"
            df_temp = (pd.read_feather(f"{data_path}"))
            df_[f"anomaly_{int(anomaly-5)}"] = df_temp
        return df_
    def forward(self):
        print(self.__str__())
        return self.df_init()
    def __call__(self):
        return self.forward()
    

class UnsandDataset(Dataset):
    __doc__ = r"""
        This is an adaptation of the Dataset Class to only load batches in the 
        CPU/GPU RAM.  

        Given the method explained in <https://www.analyticsvidhya.com/blog/2021/09/torch-dataset-and-dataloader-early-loading-of-data/>
        for images, in this class, we implement a version for the time series data. 
        Notice that this is a naive adaptaion for time-series data types where 
        each sample should be stored in memory with a fixed sequence length.   
        In this case each input is [1xSequence_length]. 
        Since we are training an unsupervised Multi-variant AutoEncoder, the labels are the input data. 
        """ 
    def __init__(self, path: str, device: str, gpu_number:int = 0): 
        super().__init__()
        __doc__ = r"""
            Args: 
                path (str): Directory of the files
                device (str): Device that you want to load your data into. 
                gpu_number (int): if device is cuda, you should insert the number of GPU. 
                                    Notice that it is not active for the CPU case.
        """
        self.path = path 
        self.file_list = glob.glob(self.path)
        self.device = device 
        self.gpu_number = gpu_number
    
    def __getitem__(self, item):
        file_idx = self.file_list[item] 
        sample = torch.load(file_idx, map_location = lambda storage, loc:storage.cuda(self.gpu_number)) if self.device == 'cuda' else torch.load(file_idx, map_location = torch.device('cpu'))
        return sample , sample
    def __len__(self): 
        return len(self.file_list)
    

def memory_allocation(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Allocated Memory used for dataloader = {current/1e6} MB")
        print(f"Max allocated Memory used by the python process = {peak/1e6} MB")
        tracemalloc.stop()
        return result
    return wrapper

def timeit(func): 
    def wrapper(*args, **kwargs): 
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Function {func.__name__} took {time.time() - start}sec' )
        return result 
    return wrapper