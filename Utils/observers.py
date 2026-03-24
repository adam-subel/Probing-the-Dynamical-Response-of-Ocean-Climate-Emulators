import torch
import functools
import abc
import sys
sys.path.append("../")

import Utils.transformations as transformations


class Observer(abc.ABC):
    def __init__(self,
                 network: torch.nn.Module,
                 input_transform: callable = transformations.NullTransform(),
                 output_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        self.network = network
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.device_name = device_name

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.input_transform.update_device(device_name)
        self.output_transform.update_device(device_name)
        self.device_name = device_name
        
    @abc.abstractmethod
    def observe(self, input_state):
        return input_state, output_state

class ObserverUpdate(Observer):
    def __init(self):
        super.__init__(self)
    
    def observe(self, input_state): 
        if len(input_state.shape) == 3:
            input_state = input_state.unsqueeze(dim = 0)
        input_state = input_state.to(device = self.device_name) 
        
        output_state = self.network(input_state)
        output_state = self.output_transform(output_state)
        
        return output_state, output_state
    
class ObserverPassive(Observer):
    def __init(self):
        super.__init__(self)
    
    def observe(self, input_state): 
        if len(input_state.shape) == 3:
            input_state = input_state.unsqueeze(dim = 0)
        input_state = input_state.to(device = self.device_name) 
        
        output_state = self.network(self.input_transform(input_state))
        output_state = self.output_transform(output_state)
        
        return input_state, output_state

class AutoEncodeObserver(Observer):
    def __init(self):
        super.__init__(self)
    
    def observe(self, input_state): 
        if len(input_state.shape) == 3:
            input_state = input_state.unsqueeze(dim = 0)
        input_state = input_state.to(device = self.device_name) 
        
        output_state, mu, logvar = self.network(self.input_transform(input_state))
        output_state = self.output_transform(output_state)
        
        return input_state, output_state, mu, logvar