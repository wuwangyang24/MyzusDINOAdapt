# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
from torch import nn
from abc import abstractmethod
import torch

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        

    def encode(self, input: torch.Tensor):
        raise NotImplementedError

    def decode(self, input: torch.Tensor):
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs) -> torch.Tensor:
        pass