import pandas as pd 

# TODO: Document the script 



class dataInitFeather: 
    def __init__(self,data_path:str,  dataset_type:str):
        self.dataset_type = dataset_type
        # We add 5 to the anomaly level since the first 5 experiments are normal cases. 
        
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
        self.__str__()
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
        self.__str__()
        return self.df_init()
    def __call__(self):
        return self.forward()