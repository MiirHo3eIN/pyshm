import torch 




class TorchSampleLogger: 
    def __init__(self, save_path: str, device: str= 'cpu'):
        self._global_counter = 0
        self._save_path = save_path
        self._device = device

    @staticmethod
    def _shuffler(self, x:torch.tensor) -> torch.tensor:
        shuffled_indx = torch.randperm(len(x), dtype = torch.int64)
    
        return x[shuffled_indx]

    def _tensor_logger(self, x: torch.tensor) -> None: 
        torch.save(x, f'{self._save_path}sample_{self._global_counter}.pt')

    def _log_sample(self, x: torch.tensor) -> None:
        
        shuffled_data = self._shuffler(x)
        
        for idx in range(0, shuffled_data.shape[0]): 
            self._tensor_logger(shuffled_data[idx, :].to_device(self._device))
            self._global_counter += 1
    
    
    def forward(self, x: torch.tensor):
        x = self._log_sample(x)


    def __call__(self, x: torch.tensor):
        self.forward(x)   
