from typing import Union, Tuple
from abc import ABC, abstractmethod
import math
import torch
from torchvision.transforms import Resize
from itertools import pairwise, chain
import functools



class LearnedFeatures(torch.nn.Module):
    def __init__(self,
                 shape: tuple[int,int],
                 num_features: int,
                 *,
                 intializer: callable = functools.partial(torch.nn.init.xavier_normal_),
                ):
        super().__init__()
        self.shape = (num_features, ) + shape
        self.features =  intializer(torch.zeros(self.shape))
        self.std = torch.std(self.features)
        self.features = torch.nn.Parameter(self.features)
        self.num_features = num_features
    def forward(self,fts):
        return self.features.repeat((fts.shape[0],) + (1,)*len(self.features.shape))/self.std

class FeatureWrapper(torch.nn.Module):
    def __init__(self,
                 network,
                 features,
                ):
        super().__init__()
        self.network = network
        self.features = features
    
    def forward(self,fts):
        features = self.features(fts)
        fts = torch.cat((fts,features), dim = 1)
        return self.network(fts)  

class DynamicTanh(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 *,
                 alpha_init: float = 0.5,                 
                ):
        super().__init__()
        self.input_size = input_size
        self.alpha = torch.nn.Parameter(torch.ones(1)*alpha_init)
        self.gamma = torch.nn.Parameter(torch.ones((self.input_size, 1, 1)))
        self.beta = torch.nn.Parameter(torch.zeros((self.input_size, 1, 1)))
        self.tanh = torch.nn.Tanh()
    def forward(self,fts):
        return self.gamma * self.tanh(self.alpha*fts) + self.beta

class DynamicTanhScaled(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 *,
                 alpha_init: float = 0.5, 
                 rescaling: float = 0.05,
                ):
        super().__init__()
        self.input_size = input_size
        self.rescaling = rescaling
        self.alpha = torch.nn.Parameter(torch.ones(1)*alpha_init*self.rescaling)
        self.gamma = torch.nn.Parameter(torch.ones((self.input_size, 1, 1))*self.rescaling)
        self.beta = torch.nn.Parameter(torch.zeros((self.input_size, 1, 1)))
        self.tanh = torch.nn.Tanh()
    def forward(self,fts):
        return (self.gamma/self.rescaling) * self.tanh((self.alpha/self.rescaling)*fts) + self.beta


class AreaWeightedMean(torch.nn.Module):
    """
    A helper module to compute the area-weighted mean of a tensor,
    accounting for channel-specific (e.g., depth-varying) wet masks.
    
    Args:
        area_weights (torch.Tensor): 
            A 2D tensor (H, W) of grid cell areas (e.g., areacello).
        wet_mask (torch.Tensor): 
            A 3D tensor (C, H, W) where ocean points are 1.0 
            and land points are 0.0. C must match the channel
            dimension of the network's output.
    """
    def __init__(self,
                 area_weights: torch.Tensor,
                 wet_mask: torch.Tensor,
                ):
        super().__init__()
        if area_weights.dim() != 2:
            raise ValueError(f"area_weights must be a 2D tensor (H, W), but got shape {area_weights.shape}")
        if wet_mask.dim() != 3:
            raise ValueError(f"wet_mask must be a 3D tensor (C, H, W), but got shape {wet_mask.shape}")
        

        area_weights_bc = area_weights.unsqueeze(0)
        masked_area_weights = wet_mask * area_weights_bc
        total_area_per_channel = masked_area_weights.sum(dim=(-2, -1))
        

        self.register_buffer('masked_area_weights_bc', masked_area_weights.unsqueeze(0))
        self.register_buffer('total_area_per_channel_bc', total_area_per_channel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    def forward(self, fts: torch.Tensor) -> torch.Tensor:

        if fts.shape[1] != self.masked_area_weights_bc.shape[1]:
            raise ValueError(f"Input tensor has {fts.shape[1]} channels, but AreaWeightedMean was initialized with a mask for {self.masked_area_weights_bc.shape[1]} channels.")
            
        weighted_sum = (fts * self.masked_area_weights_bc).sum(dim=(-2, -1), keepdim=True)
        
        mean = weighted_sum / (self.total_area_per_channel_bc + 1e-9)
        
        return mean

        

class ConvLayer(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 *,
                 dilation: int = 1,                 
                 kernel_size: tuple[int,int] = (3,3),
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pady = int((kernel_size[0]-1)/2*dilation)
        self.padx = int((kernel_size[1]-1)/2*dilation)
        
        self.layer = torch.nn.Conv2d(input_size,output_size,kernel_size,dilation=dilation,padding = 'valid')
    
    def forward(self,fts):
        fts = torch.nn.functional.pad(fts,(self.padx,self.padx,0,0),mode='circular')
        fts = torch.nn.functional.pad(fts,(0,0,self.pady,self.pady),mode='constant')
        return self.layer(fts)

class ConvNextBlock(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_node_ratio: int = 4,
                 *,
                 residual: bool = True,
                 batch_norm: bool = False,
                 batch_norm_function = torch.nn.InstanceNorm2d, 
                 dilation: int = 1,                                  
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False,
                 surya_block: bool = False,                 
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size  
        self.hidden_size = input_size*hidden_node_ratio
        self.activation = activation
        self.batch_norm_function = batch_norm_function
        if batch_norm:
            self.batch_norm_2D = self.batch_norm_function(self.hidden_size)
            self.batch_norm_1D = self.batch_norm_function(self.hidden_size)            
        else:
            self.batch_norm_2D = torch.nn.Identity()
            self.batch_norm_1D = torch.nn.Identity()
            
        self.activate_final = activate_final
        self.residual = residual
        if self.input_size != self.output_size:
            self.projection = torch.nn.Conv2d(self.input_size,
                                              self.output_size,
                                              kernel_size = (1,1),
                                              padding = 'valid',
                                              bias = False)
        else:
            self.projection = torch.nn.Identity()
        self.Conv2D = ConvLayer(self.input_size,self.hidden_size,kernel_size=kernel_size,dilation = dilation)
        if surya_block:
            self.Conv1D_Up = ConvLayer(self.hidden_size,
                                       self.hidden_size,
                                       kernel_size=kernel_size,
                                       dilation = dilation)
        else:
            self.Conv1D_Up = ConvLayer(self.hidden_size,self.hidden_size,kernel_size=(1,1))
            
        self.Conv1D_Down = ConvLayer(self.hidden_size,self.output_size,kernel_size=(1,1))

        
    def forward(self, fts):
        projected_fts = self.projection(fts)
        fts = self.Conv2D(fts)
        fts = self.batch_norm_2D(fts)
        fts = self.activation(fts)        
        fts = self.Conv1D_Up(fts)
        fts = self.batch_norm_1D(fts)        
        fts = self.activation(fts)
        fts = self.Conv1D_Down(fts)
        if self.residual:
            fts = fts + projected_fts
        if self.activate_final:
            fts = self.activation(fts)
        return fts

class ConvNet(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_hidden_nodes: int,
                 *,
                 dilation: int = 1,                                  
                 kernel_size: tuple[int,int] = (3,3),
                 num_hidden_layers: int = 0,
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False,
                 batch_norm: bool = False,
                 batch_norm_final: bool = False,
                 batch_norm_function = torch.nn.InstanceNorm2d,                  
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size        
        self.activation = activation
        self.activate_final = activate_final
        self.batch_norm_final = batch_norm_final
        self.batch_norm_function = batch_norm_function
        if batch_norm:
            self.batch_norm = self.batch_norm_function(num_hidden_nodes)
        else:
            self.batch_norm = torch.nn.Identity()        
        self.layers = []
        self.layers.append(ConvLayer(input_size,num_hidden_nodes,kernel_size=kernel_size,dilation = dilation))
        for i in range(num_hidden_layers):
            self.layers.append(ConvLayer(num_hidden_nodes,num_hidden_nodes,kernel_size=kernel_size,dilation = dilation))
        self.layers.append(ConvLayer(num_hidden_nodes,output_size,kernel_size=kernel_size,dilation = dilation))
        self.layers = torch.nn.ModuleList(self.layers)
        self.num_layers = len(self.layers)

        
    def forward(self, fts):
        for i, layer in enumerate(self.layers):
            fts = layer(fts)
            if i != (self.num_layers - 1) or self.batch_norm_final:
                fts = self.batch_norm(fts)
            if i != (self.num_layers - 1) or self.activate_final:
                fts = self.activation(fts)
        return fts


class ResBlock(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 *,
                 dilation: int = 1,                                  
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False,
                 batch_norm: bool = False,
                 batch_norm_function: callable = torch.nn.Identity, 
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size        
        self.activation = activation
        self.activate_final = activate_final     
        if self.input_size != self.output_size:
            self.projection = torch.nn.Conv2d(self.input_size,
                                              self.output_size,
                                              kernel_size = (1,1),
                                              padding = 'valid',
                                              bias = False)
        else:
            self.projection = torch.nn.Identity()      
        if batch_norm:
            self.norm1 = batch_norm_function(self.output_size)
            self.norm2 = batch_norm_function(self.output_size)
        else:
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()            
        self.first_conv = ConvLayer(self.input_size,
                                self.output_size,
                                kernel_size=kernel_size,
                                dilation = dilation)
        self.second_conv = ConvLayer(self.output_size,
                                self.output_size,
                                kernel_size=kernel_size,
                                dilation = dilation)
        
    def forward(self, fts):
        projected_fts = self.projection(fts)
        fts = self.first_conv(fts)
        fts = self.norm1(fts)
        fts = self.activation(fts)        
        fts = self.second_conv(fts)
        fts = self.norm2(fts)
        fts = fts + projected_fts
        if self.activate_final:
            fts = self.activation(fts)
        return fts

class PoolLayer():
    def __init__(self,
                 size,
                 layer,
                 resize_ratio,
                 ):
        del size
        self.layer = layer(resize_ratio)
        
    def __call__(self,fts):
        return self.layer(fts)    
    
class ResNet(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False ,
                 checkpoints: bool = False,      
                 batch_norm: bool = False,
                 batch_norm_function: callable = torch.nn.Identity,                 
                ):
        super().__init__()

        self.checkpoints = checkpoints        
        
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
            
        node_sizes = (input_shape[0],) + hidden_node_sizes 
        
        self.layers = []
        
        for (in_size, out_size), dilation in zip(pairwise(node_sizes),dilations[:-1]):
            self.layers.append(ResBlock(in_size,
                                   out_size,
                                   dilation = dilation,
                                   kernel_size = kernel_size,
                                   activation = activation,
                                   activate_final = True,
                                   batch_norm = batch_norm,
                                   batch_norm_function = batch_norm_function,
                                  ))
        self.layers.append(ResBlock(hidden_node_sizes[-1],
                               output_size,
                               dilation = dilation,
                               kernel_size = kernel_size,
                               activation = activation,
                               activate_final = activate_final,
                               batch_norm = False
                              ))
        self.layers = torch.nn.ModuleList(self.layers)       
        
    def forward_step (self, fts):
        for layer in self.layers:
            fts = layer(fts)
        return fts        
        
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)    
        
    
    
class UNet(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 kernel_size: tuple[int,int] = (3,3),
                 num_hidden_layers: int = 0,
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False ,
                 batch_norm: bool = False,
                 batch_norm_function = torch.nn.InstanceNorm2d,                                   
                 checkpoints: bool = False,                 
                ):
        super().__init__()

        self.checkpoints = checkpoints   
        self.batch_norm_function = batch_norm_function
        
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNet(input_size=in_size,
                                            output_size=out_size,
                                            num_hidden_nodes=out_size,
                                            dilation = dilation,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final=activate_final)
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)

    def forward_step (self, fts):
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
        return fts        
        
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)
    
class UNext(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True, 
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False, 
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()
        
        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNextBlock(input_size=in_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,
                                          
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)
    def forward_step(self,fts):
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
        if self.final_conv_layer:
            fts = self.final_layer(fts)
        return fts
    
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)



class PhysicalNetwork(torch.nn.Module):
    def __init__(self,
                 vars_input_shape: tuple[int,int,int],
                 num_vars: int,
                 num_boundary_vars: int,
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 vertical_kernel_size: int = 3,
                 dilations: int|tuple[int,...] = 1,
                 *,
                 boundary_embedding: bool = True,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True, 
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False, 
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()


        self.num_vars = num_vars
        self.num_boundary_vars = num_boundary_vars
        self.num_levels = vars_input_shape[0]
        self.vertical_kernel_size = vertical_kernel_size
        self.boundary_embedding = boundary_embedding
    
        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [vars_input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (num_vars,) + hidden_node_sizes
        
        for i, ((in_size, out_size), dilation, size) in enumerate(zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True)):            
            
            self.down_layers.append(ConvNextBlock(input_size=in_size*self.vertical_kernel_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1]*self.vertical_kernel_size,
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,
                                          
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size*self.vertical_kernel_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNextBlock(input_size=in_size*self.vertical_kernel_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )     
                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)

        if self.boundary_embedding:
            embedded_size = int((self.vertical_kernel_size-1)/2)*self.num_vars
            self.embedding_network = ConvNet(self.num_boundary_vars,
                                             self.num_vars,
                                             num_hidden_nodes = 30,
                                             dilation = 1,                                  
                                             num_hidden_layers = 2,
                                             activation = activation, 
                                             activate_final = False,
                                             batch_norm = batch_norm,
                                             batch_norm_final = False,
                                             batch_norm_function = self.batch_norm_function,                  
                                            )
        

    def _process_input(self,
                       tensor,
                       boundary_tensor = None):
        pad_size = int((self.vertical_kernel_size-1)/2)
        input_shape = tensor.shape
        batch_size = input_shape[0]
        tensor = torch.stack(torch.split(tensor,split_size_or_sections = self.num_levels, dim = -3),dim = -3)
        tensor = torch.nn.functional.pad(tensor,(0, 0, 0, 0, 0, 0, pad_size, pad_size))
        tensor = tensor.unfold(dimension=-4, size=self.vertical_kernel_size, step=1)  
        if boundary_tensor is not None:
            if pad_size == 1:
                boundary_tensor = boundary_tensor.unsqueeze(dim = -1)
            tensor[:,0,:,:,:,0:pad_size] = boundary_tensor           
        tensor = tensor.permute([0,1,2,-1,-3,-2])      
        tensor = torch.cat(torch.split(tensor,split_size_or_sections = 1, dim = -4),dim = -3).squeeze()
        tensor = torch.cat(torch.split(tensor,split_size_or_sections = 1, dim = 0),dim = -4).squeeze() 
        return tensor, batch_size

    def _process_output(self,tensor,batch_size):
        input_shape = tensor.shape
        pad_size = int((3-1)/2)        
        tensor = torch.stack(torch.split(tensor,split_size_or_sections = self.num_levels, dim = 0),dim = 0)
        tensor = torch.nn.functional.pad(tensor,(0, 0, 0, 0, 0, 0, pad_size, pad_size))
        tensor = tensor.unfold(dimension=-4, size=self.vertical_kernel_size, step=1)  
        tensor = tensor.permute([0,1,2,-1,-3,-2])      
        tensor = torch.cat(torch.split(tensor,split_size_or_sections = 1, dim = -4),dim = -3).squeeze()
        tensor = torch.cat(torch.split(tensor,split_size_or_sections = 1, dim = 0),dim = -4).squeeze() 
        return tensor

    def _process_final_output(self,tensor,batch_size):
        input_shape = tensor.shape  
        tensor = torch.stack(torch.split(tensor,split_size_or_sections = self.num_levels, dim = 0),dim = 0)
        tensor = torch.cat(torch.split(tensor,split_size_or_sections = 1, dim = -3),dim = -4).squeeze()
        return tensor
    
    def forward_step(self,fts,boundary_fts):
        embedded_boundary = self.embedding_network(boundary_fts)
        fts, batch_size = self._process_input(fts,embedded_boundary)
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            fts = self._process_output(fts, batch_size)
            carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        fts = self._process_output(fts, batch_size)
        
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
            if i != (len(self.up_layers) - 1):
                fts = self._process_output(fts, batch_size)
            
        if self.final_conv_layer:
            fts = self.final_layer(fts)
            
        return self._process_final_output(fts, batch_size)
    
    def forward(self,fts,boundary_fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,boundary_fts,use_reentrant=False)
        else:
            return self.forward_step(fts,boundary_fts)


class AutoEncoder(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 output_embedding: bool = True,
                 stochastic: bool = True,
                 var_scale: float = 1.0,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(PoolLayer,layer = torch.nn.AvgPool2d, resize_ratio = (2,2)),
                 kernel_size: tuple[int,int] = (3,3),
                 num_hidden_layers: int = 0,
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False ,
                 batch_norm: bool = False,
                 batch_norm_function = torch.nn.InstanceNorm2d,                                   
                 checkpoints: bool = False,                 
                ):
        super().__init__()

        self.checkpoints = checkpoints   
        self.batch_norm_function = batch_norm_function
        self.output_embedding = output_embedding
        self.stochastic = stochastic
        self.var_scale = var_scale
        
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNet(input_size=in_size,
                                            output_size=out_size,
                                            num_hidden_nodes=out_size,
                                            dilation = dilation,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.mu_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
        
        self.logvar_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )        
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final=activate_final)
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)

    def forward_step (self, fts):
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            fts = downsampler(fts)
            
        mu = self.mu_block(fts)
        logvar  = self.logvar_block(fts)
        if self.stochastic:
            fts =  mu + torch.exp(logvar/2)*self.var_scale * torch.randn_like(logvar) 
        else:
            fts =  mu

        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = layer(fts)
        if self.output_embedding:
            return fts, mu, logvar     
        else:
            return fts
        
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)


class UNetLatent(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 num_skipped_residuals: int = 0,
                 output_embedding: bool = True,
                 stochastic: bool = True,
                 var_scale: float = 1.0,
                 rho: float|None = None,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 kernel_size: tuple[int,int] = (3,3),
                 num_hidden_layers: int = 0,
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False,
                 batch_norm: bool = False,
                 batch_norm_function = torch.nn.InstanceNorm2d,                                   
                 checkpoints: bool = False,                 
                ):
        super().__init__()

        self.checkpoints = checkpoints   
        self.batch_norm_function = batch_norm_function
        self.output_embedding = output_embedding
        self.stochastic = stochastic
        self.var_scale = var_scale
        self.rho = rho
                    
        self.state = None
        
        self.num_skipped_residuals = num_skipped_residuals
        self.num_residual_layers = len(hidden_node_sizes) - num_skipped_residuals 
        
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNet(input_size=in_size,
                                            output_size=out_size,
                                            num_hidden_nodes=out_size,
                                            dilation = dilation,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final=activate_final)
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)

        self.mu_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
        
        self.logvar_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )        
    
    def forward_step (self, fts):
        carry = []
        for i, (layer, downsampler) in enumerate(zip(self.down_layers, self.downsample_layers)):
            fts = layer(fts)
            if i >= self.num_skipped_residuals:
                carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        
        mu = self.mu_block(fts)
        logvar  = self.logvar_block(fts)
        

        if self.rho is not None and self.state is not None:
            self.state = self.state*self.rho + torch.randn_like(self.state)
        else:
            self.state = torch.randn_like(logvar)
            
        if self.stochastic:
            fts =  mu + torch.exp(logvar/2)*self.var_scale * self.state 
        else:
            fts =  mu        
            
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):
            fts = upsampler(fts) 
            if i < self.num_residual_layers:
                fts = fts + carry[-(i+1)]
            fts = layer(fts)
            
        if self.output_embedding:
            return fts, mu, logvar     
        else:
            return fts
            
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)     

class UNetLatentLearnedRho(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 num_skipped_residuals: int = 0,
                 output_embedding: bool = True,
                 stochastic: bool = True,
                 var_scale: float = 1.0,
                 rho: float|None = None,
                 learned_rho: bool = False,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 kernel_size: tuple[int,int] = (3,3),
                 num_hidden_layers: int = 0,
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False,
                 batch_norm: bool = False,
                 batch_norm_function = torch.nn.InstanceNorm2d,                                   
                 checkpoints: bool = False,                 
                ):
        super().__init__()

        self.checkpoints = checkpoints   
        self.batch_norm_function = batch_norm_function
        self.output_embedding = output_embedding
        self.stochastic = stochastic
        self.var_scale = var_scale
        if learned_rho:
            self.rho_parameter = torch.nn.Parameter(torch.zeros(1))
            self.sigmoid = torch.nn.Sigmoid()
            self.rho = self.sigmoid(10*self.rho_parameter)
        else:
            self.rho = rho
            
        self.learned_rho = learned_rho
        
        self.state = None
        
        self.num_skipped_residuals = num_skipped_residuals
        self.num_residual_layers = len(hidden_node_sizes) - num_skipped_residuals 
        
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNet(input_size=in_size,
                                            output_size=out_size,
                                            num_hidden_nodes=out_size,
                                            dilation = dilation,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNet(input_size=in_size,
                                                output_size=out_size,
                                                num_hidden_nodes=out_size,
                                                dilation = dilation,
                                                activation = activation,
                                                activate_final=activate_final)
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)

        self.mu_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
        
        self.logvar_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )        

    def make_rho_learnable(self):
        if self.rho:
            self.rho_parameter = torch.nn.Parameter(torch.ones(1)*self.rho/10)
        else:
            self.rho_parameter = torch.nn.Parameter(torch.zeros(1))
        self.sigmoid = torch.nn.Sigmoid()
        self.rho = self.sigmoid(10*self.rho_parameter).detach()
        self.learned_rho = True
        
    def make_rho_fixed(self):
        self.rho = self.sigmoid(10*self.rho_parameter).detach()
        self.learned_rho = False
        self.rho_parameter = None
    
    def forward_step (self, fts):
        carry = []
        for i, (layer, downsampler) in enumerate(zip(self.down_layers, self.downsample_layers)):
            fts = layer(fts)
            if i >= self.num_skipped_residuals:
                carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        
        mu = self.mu_block(fts)
        logvar  = self.logvar_block(fts)
        
        if self.learned_rho:
            if self.state is not None:
                self.state = self.state*self.sigmoid(10*self.rho_parameter) + torch.randn_like(self.state)
            else:
                self.state = torch.randn_like(logvar)
                self.state = self.state*self.sigmoid(10*self.rho_parameter) + torch.randn_like(self.state)                
        else:
            if self.rho is not None and self.state is not None:
                self.state = self.state*self.rho + torch.randn_like(self.state)
            elif self.rho is not None:
                self.state = torch.randn_like(logvar)
                self.state = self.state*self.rho + torch.randn_like(self.state)
            else:
                self.state = torch.randn_like(logvar)
            
        if self.stochastic:
            fts =  mu + torch.exp(logvar/2)*self.var_scale * self.state 
        else:
            fts =  mu        
            
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):
            fts = upsampler(fts) 
            if i < self.num_residual_layers:
                fts = fts + carry[-(i+1)]
            fts = layer(fts)
            
        if self.output_embedding:
            return fts, mu, logvar     
        else:
            return fts
            
    def forward(self,fts):
        if self.learned_rho:
            with torch.no_grad():
                self.rho = self.sigmoid(10*self.rho_parameter)
        if self.state is not None and self.state.shape[0] !=  fts.shape[0]:
            self.state = None
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)   

class EncodeBoundaryUNext(torch.nn.Module):
    def __init__(self,
                 state_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 encode_network: callable,
                 state_indices: list[int,...],
                 boundary_indices: list[int,...],
                 dilations: int|tuple[int,...] = 1,
                 include_boundary: bool = True,
                 *,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True, 
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False, 
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()
        
        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [state_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (len(state_indices),) + hidden_node_sizes

        self.state_indices = state_indices
        self.boundary_indices = boundary_indices
        
        self.include_boundary = include_boundary
        self.encode_block = encode_network(output_size = hidden_node_sizes[-1])
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNextBlock(input_size=in_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,
                                          
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)


    def set_encoder_trainable_state(self,trainable: bool):
        for param in self.encode_block.parameters():
            param.requires_grad = trainable  
    
    def set_propagator_trainable_state(self,trainable: bool):
        for name, param in self.named_parameters():
            if 'encode_block' not in name:
                param.requires_grad = trainable  
                
    
    def forward_step(self,fts):
        boundary_fts = fts[...,self.boundary_indices,:,:]
        fts = fts[...,self.state_indices,:,:]
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)

        if self.include_boundary:
            fts += self.encode_block(boundary_fts)
        
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
        if self.final_conv_layer:
            fts = self.final_layer(fts)
        return fts
    
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)

class Encode(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 dilations: int|tuple[int,...] = 1,
                 *,
                 resize_ratio: tuple[int,int] = (2,2),
                 downsampler = functools.partial(Resize,antialias = True),
                 kernel_size: tuple[int,int] = (3,3),
                 num_hidden_layers: int = 0,
                 activation = torch.nn.functional.gelu, 
                 batch_norm: bool = False,
                 batch_norm_function = DynamicTanhScaled,                                   
                 checkpoints: bool = False,                 
                ):
        super().__init__()

        self.checkpoints = checkpoints   
        self.batch_norm_function = batch_norm_function
        
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNet(input_size=in_size,
                                            output_size=out_size,
                                            num_hidden_nodes=out_size,
                                            dilation = dilation,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNet(input_size=hidden_node_sizes[-1],
                                    output_size=output_size,
                                    num_hidden_nodes=hidden_node_sizes[-1],
                                    dilation = dilations[-1],
                                    activation = activation,
                                    activate_final = False,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                   )
                                   
        

    def forward_step (self, fts):
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        return fts        
        
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)

class EncodeBoundaryUNextRecursive(torch.nn.Module):
    def __init__(self,
                 state_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 encode_network: callable,
                 state_indices: list[int,...],
                 boundary_indices: list[int,...],
                 dilations: int|tuple[int,...] = 1,
                 include_boundary: bool = True,
                 *,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True, 
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False, 
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()
        
        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [state_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (len(state_indices),) + hidden_node_sizes

        self.state_indices = state_indices
        self.boundary_indices = boundary_indices
        
        self.include_boundary = include_boundary
        self.encode_block = encode_network(output_size = hidden_node_sizes[-1])
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNextBlock(input_size=in_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,
                                          
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)


    def set_encoder_trainable_state(self,trainable: bool):
        for param in self.encode_block.parameters():
            param.requires_grad = trainable  
    
    def set_propagator_trainable_state(self,trainable: bool):
        for name, param in self.named_parameters():
            if 'encode_block' not in name:
                param.requires_grad = trainable  
                
    
    def forward_step(self,fts):
        boundary_fts = fts[...,self.state_indices[:-1]+self.boundary_indices,:,:].clone()
        fts = fts[...,self.state_indices,:,:]
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        bottom_fts = self.bottom_block(fts)
        fts = bottom_fts.clone()
        
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
        if self.final_conv_layer:
            fts = self.final_layer(fts)

        if self.include_boundary:
            boundary_fts[...,self.state_indices[:-1],:,:] = fts.clone()
            fts = bottom_fts + self.encode_block(boundary_fts)
            for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):
    
                fts = upsampler(fts) 
                fts = fts + carry[-(i+1)]
                fts = layer(fts)
            if self.final_conv_layer:
                fts = self.final_layer(fts)            
            
        return fts
    
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)


class BoundaryUNextSequential(torch.nn.Module):
    def __init__(self,
                 state_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 state_indices: list[int,...],
                 boundary_indices: dict[str,list[int,...]],                 
                 encode_networks: dict[str,callable],
                 include_boundary: list[str]|None = None,
                 *,
                 dilations: int|tuple[int,...] = 1,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True, 
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False, 
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()
        
        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [state_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (len(state_indices),) + hidden_node_sizes

        self.state_indices = state_indices
        self.boundary_indices_dict = boundary_indices
        self.boundary_indices = list(chain.from_iterable(boundary_indices.values()))
        
        self.boundary_network_names = list(boundary_indices.keys())
        
        self.include_boundary = include_boundary
        
        self.encode_networks = {k:v(output_size = output_size) for k,v in encode_networks.items()}
        self.encode_networks = torch.nn.ModuleDict(self.encode_networks)
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNextBlock(input_size=in_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,
                                          
                                   )
                                   
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)


    def set_encoder_network_trainable_state(self,
                                            boundary_names:list,
                                            trainable: bool):
        for boundary_name in boundary_names:
            for param in self.encode_networks[boundary_name].parameters():
                param.requires_grad = trainable  
    
    def set_propagator_trainable_state(self,trainable: bool):
        for name, param in self.named_parameters():
            if 'encode_networks' not in name:
                param.requires_grad = trainable  
                
    
    def forward_step(self,fts):
        boundary_fts = fts[...,self.boundary_indices,:,:].clone()
        fts = fts[...,self.state_indices,:,:]
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        bottom_fts = self.bottom_block(fts)
        fts = bottom_fts.clone()
        
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
        if self.final_conv_layer:
            fts = self.final_layer(fts)

        if self.include_boundary is not None:
            for boundary in self.include_boundary:
                input_fts = torch.cat((boundary_fts[...,self.boundary_indices_dict[boundary],:,:],
                                       fts.clone()),dim = -3)
                fts += self.encode_networks[boundary](input_fts)            
            
        return fts
    
    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)

class UNextSequential(torch.nn.Module):
    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 state_indices: list[int,...],
                 boundary_indices: list[int,...],
                 encode_network: callable,
                 *,
                 dilations: int|tuple[int,...] = 1,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True,
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu,
                 activate_final: bool = False,
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()

        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)

        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))

        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes

        self.state_indices = state_indices
        self.boundary_indices = boundary_indices

        self.encode_network = encode_network(output_size = output_size)

        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):
            self.down_layers.append(ConvNextBlock(input_size=in_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))

        self.down_layers = torch.nn.ModuleList(self.down_layers)

        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,

                                   )

        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))

        up_node_list = reversed_hidden_nodes + (output_size,)


        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )
            self.upsample_layers.append(upsampler(size))
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)


    def forward_step(self,fts):
        boundary_fts = fts[...,self.boundary_indices,:,:].clone()
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)

        bottom_fts = self.bottom_block(fts)
        fts = bottom_fts.clone()

        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):

            fts = upsampler(fts)
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
        if self.final_conv_layer:
            fts = self.final_layer(fts)


        input_fts = torch.cat((fts.clone(),boundary_fts),dim = -3)
        fts = fts + self.encode_network(input_fts)

        return fts

    def forward(self,fts):
        if self.checkpoints:
            return torch.utils.checkpoint.checkpoint(self.forward_step,fts,use_reentrant=False)
        else:
            return self.forward_step(fts)


class UNextGlobalMean(torch.nn.Module):

    def __init__(self,
                 input_shape: tuple[int,int,int],
                 output_size: int,
                 hidden_node_sizes: tuple[int,...],
                 area_weights: torch.Tensor,
                 wet_mask: torch.Tensor,
                 dilations: int|tuple[int,...] = 1,
                 *,
                 mean_head_pool_size: tuple[int, int] = (2, 2),
                 mean_head_layers: list[int] = [256],
                 mean_head_activation = torch.nn.functional.gelu,
                 hidden_node_ratio: int = 4,
                 resize_ratio: tuple[int,int] = (2,2),
                 upsampler = functools.partial(Resize,antialias = True),
                 downsampler = functools.partial(Resize,antialias = True),
                 residual: bool = True, 
                 residual_final: bool = False,
                 kernel_size: tuple[int,int] = (3,3),
                 activation = torch.nn.functional.gelu, 
                 activate_final: bool = False, 
                 final_conv_layer: bool = False,
                 batch_norm: bool = True,
                 batch_norm_function = torch.nn.InstanceNorm2d,
                 checkpoints: bool = False,
                 surya_blocks: bool = False,
                ):
        super().__init__()
        

        self.checkpoints = checkpoints
        self.batch_norm_function = batch_norm_function
        if isinstance(dilations,int):
            dilations = (dilations,)*(len(hidden_node_sizes)+1)  
        
        sizes = [input_shape[-2:],]
        for i in range(len(hidden_node_sizes)):
            sizes.append(tuple([dim//2 for dim in sizes[i]]))
                    
        self.down_layers = []
        self.downsample_layers = []
        down_node_list = (input_shape[0],) + hidden_node_sizes
        
        for (in_size, out_size), dilation, size in zip(pairwise(down_node_list),dilations[:-1],sizes[1:],strict = True):            
            self.down_layers.append(ConvNextBlock(input_size=in_size,
                                            output_size=out_size,
                                            hidden_node_ratio=hidden_node_ratio,
                                            dilation = dilation,
                                            residual = residual,
                                            kernel_size = kernel_size,
                                            activation = activation,
                                            activate_final = True,
                                            batch_norm = batch_norm,
                                            batch_norm_function = self.batch_norm_function,
                                            surya_block = surya_blocks,
                                           )
                                   )
            self.downsample_layers.append(downsampler(size))
            
        self.down_layers = torch.nn.ModuleList(self.down_layers)
        
        self.bottom_block = ConvNextBlock(input_size=hidden_node_sizes[-1],
                                    output_size=hidden_node_sizes[-1],
                                    hidden_node_ratio=hidden_node_ratio,
                                    residual = residual,
                                    dilation = dilations[-1],
                                    kernel_size = kernel_size,
                                    activation = activation,
                                    activate_final = True,
                                    batch_norm = batch_norm,
                                    batch_norm_function = self.batch_norm_function,
                                    surya_block = surya_blocks,
                                   )
                                   
        self.mean_pool = torch.nn.AdaptiveAvgPool2d(mean_head_pool_size)
        bottleneck_channels = hidden_node_sizes[-1]
        
        in_features = bottleneck_channels * mean_head_pool_size[0] * mean_head_pool_size[1]
        
        mean_modules = []
        for h_dim in mean_head_layers:
            mean_modules.append(torch.nn.Linear(in_features, h_dim))
            in_features = h_dim 
            
        self.mean_final = torch.nn.Linear(in_features, output_size)
        self.mean_head_modules = torch.nn.ModuleList(mean_modules)
        self.mean_activations = mean_head_activation
        # --- 4. Standard UNext Up-sampling Path (Unchanged) ---
        self.up_layers = []
        self.upsample_layers = []
        reversed_hidden_nodes = tuple(reversed(hidden_node_sizes))
        reversed_sizes = tuple(reversed(sizes))
        reversed_dilations = tuple(reversed(dilations[:-1]))
        
        up_node_list = reversed_hidden_nodes + (output_size,)
        
        for i,((in_size, out_size), dilation, size) in enumerate(zip(pairwise(up_node_list),reversed_dilations,reversed_sizes[1:],strict = True)):
            if i < len(dilations)-2:
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                residual = residual,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final = True,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                surya_block = surya_blocks,
                                             )
                                       )
            else: # Final layer
                self.up_layers.append(ConvNextBlock(input_size=in_size,
                                                output_size=out_size,
                                                hidden_node_ratio=hidden_node_ratio,
                                                dilation = dilation,
                                                kernel_size = kernel_size,
                                                activation = activation,
                                                activate_final=activate_final,
                                                batch_norm = batch_norm,
                                                batch_norm_function = self.batch_norm_function,
                                                residual = residual_final,
                                                surya_block = surya_blocks,
                                                   )
                                       )                
            self.upsample_layers.append(upsampler(size))        
        self.up_layers = torch.nn.ModuleList(self.up_layers)
        
        self.final_conv_layer = final_conv_layer
        if self.final_conv_layer:
            self.final_layer = ConvLayer(out_size,
                                         out_size,
                                         kernel_size=kernel_size)
                                         
        # --- 5. MODIFIED: Area-weighted mean calculator ---
        if wet_mask.shape[0] != output_size:
            raise ValueError(f"wet_mask channel dimension ({wet_mask.shape[0]}) does not match network output_size ({output_size}).")
        
        self.area_mean_computer = AreaWeightedMean(
            area_weights=area_weights, 
            wet_mask=wet_mask
        )

    # --- forward_step and forward methods are UNCHANGED ---
    # (They correctly call self.area_mean_computer. The internal
    # logic of self.area_mean_computer is what we changed.)

    def forward_step(self, fts):
        """
        Runs the forward pass and returns the two components.
        """
        carry = []
        for layer, downsampler in zip(self.down_layers, self.downsample_layers):
            fts = layer(fts)
            carry.append(fts)
            fts = downsampler(fts)
            
        fts = self.bottom_block(fts)
        # --- Mean Head Path ---
        mean_fts = torch.flatten(self.mean_pool(fts), 1)
        for layer in self.mean_head_modules:
            mean_fts = layer(mean_fts)
            mean_fts = self.mean_activations(mean_fts)
        mean_fts = self.mean_final(mean_fts)
        global_mean_pred = mean_fts.unsqueeze(-1).unsqueeze(-1)

        # --- Anomaly Head Path (Standard U-Net Decoder) ---
        for i, (layer, upsampler) in enumerate(zip(self.up_layers, self.upsample_layers)):
            fts = upsampler(fts) 
            fts = fts + carry[-(i+1)]
            fts = layer(fts)
            
        if self.final_conv_layer:
            fts = self.final_layer(fts)
        

        mean_of_anomaly = self.area_mean_computer(fts)
        anomaly_pred = fts - mean_of_anomaly
        
        return global_mean_pred, anomaly_pred
    
    def forward(self, fts):
        """
        Main forward pass.
        """
        if self.checkpoints:
            (global_mean_pred, anomaly_pred) = torch.utils.checkpoint.checkpoint(
                self.forward_step, fts, use_reentrant=False
            )
        else:
            (global_mean_pred, anomaly_pred) = self.forward_step(fts)
        
        if self.training:
            return global_mean_pred, anomaly_pred
        else:
            return global_mean_pred + anomaly_pred        


def initialize_weights(model, init_func=None):
    """
    Globally initializes the weights of a given network model.
    Pass an init_func (e.g., functools.partial(torch.nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu'))
    to apply custom initialization to multidimensional weights (like convolutions and linear layers).
    """
    if init_func is None:
        return model

    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            continue

        if hasattr(m, 'weight') and m.weight is not None:
            # Multi-dimensional weights are typically Conv/Linear kernels
            if m.weight.dim() > 1:
                init_func(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    return model

def append_1d_coords(x):
    # x: (B, C, L)
    B, C, L = x.shape
    device = x.device
    coords = torch.linspace(-1, 1, L, device=device).reshape(1, 1, L).repeat(B, 1, 1)
    return torch.cat([x, coords], dim=1)

def append_2d_coords(x):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    device = x.device

    y = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).repeat(B, 1, 1, W)
    x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)

    return torch.cat([x, y, x_coords], dim=1)

class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 128, 128, 64)):
        super().__init__()
        layers = []
        in_dim = input_size

        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, h))
            # Use nn.GELU() class, not functional.gelu
            layers.append(torch.nn.GELU())
            in_dim = h

        layers.append(torch.nn.Linear(in_dim, output_size))
        self.net = torch.nn.Sequential(*layers)


class PixelShuffle1D(torch.nn.Module):
    """
    1D PixelShuffle equivalent.
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, length = x.size()
        new_channels = channels // self.upscale_factor
        new_length = length * self.upscale_factor
        x = x.view(batch_size, new_channels, self.upscale_factor, length)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, new_channels, new_length)
        return x

class ViT(torch.nn.Module):
    def __init__(self, input_size, output_size, patch_size=4, embed_dim=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1, window_size=None, positional_embedding=True, shifting_windows=False, use_relative_positional_embedding=False, spatial_dims=1, use_pixel_shuffle=False, padding_type='zeros', use_norm: bool = True, norm_layer = torch.nn.LayerNorm, checkpoints: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.output_size = output_size
        self.window_size = window_size
        self.positional_embedding = positional_embedding
        self.checkpoints = checkpoints
        self.shifting_windows = shifting_windows
        self.use_relative_positional_embedding = use_relative_positional_embedding
        self.num_heads = num_heads
        self.spatial_dims = spatial_dims
        self.use_pixel_shuffle = use_pixel_shuffle
        self.padding_type = padding_type
        self.use_norm = use_norm

        if self.spatial_dims not in [1, 2]:
            raise ValueError(f"spatial_dims must be 1 or 2, got {self.spatial_dims}")

        if self.shifting_windows and self.window_size is None:
             raise ValueError("window_size must be provided when shifting_windows is True.")

        if self.positional_embedding:
            if hasattr(self.positional_embedding, 'num_features'):
                input_size += self.positional_embedding.num_features

        # Patch Embedding
        if self.spatial_dims == 1:
            self.patch_embed = torch.nn.Conv1d(input_size, embed_dim, kernel_size=patch_size, stride=patch_size, padding_mode=self.padding_type)
        else:
            self.patch_embed = torch.nn.Conv2d(input_size, embed_dim, kernel_size=patch_size, stride=patch_size, padding_mode=self.padding_type)

        trans_in_dim = embed_dim

        # Transformer Layers
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])

        for layer in self.layers:
            if not self.use_norm:
                layer.norm1 = torch.nn.Identity()
                layer.norm2 = torch.nn.Identity()
            else:
                layer.norm1 = norm_layer(embed_dim)
                layer.norm2 = norm_layer(embed_dim)

        # Relative Positional Embedding
        if self.use_relative_positional_embedding:
            if self.window_size is not None:
                if self.shifting_windows:
                    self.window_len = self.window_size * 2 - 1
                else:
                    self.window_len = self.window_size * 2 + 1
            else:
                 self.window_len = 2 * 1024 - 1

            if self.spatial_dims == 1:
                self.relative_position_bias_table = torch.nn.Parameter(
                    torch.zeros(self.window_len, num_heads))
            else:
                self.relative_position_bias_table = torch.nn.Parameter(
                    torch.zeros(self.window_len * self.window_len, num_heads))
            torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # Output Head
        head_in_dim = embed_dim

        if not self.use_pixel_shuffle:
            if self.spatial_dims == 1:
                self.head = torch.nn.Linear(head_in_dim, output_size * patch_size)
            else:
                self.head = torch.nn.Linear(head_in_dim, output_size * patch_size * patch_size)
        else:
            scale = self.patch_size
            if self.spatial_dims == 1:
                self.head = torch.nn.Sequential(
                    torch.nn.Conv1d(head_in_dim, output_size * scale, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_type),
                    PixelShuffle1D(upscale_factor=scale),
                    torch.nn.Conv1d(output_size, output_size, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_type)
                )
            else:
                self.head = torch.nn.Sequential(
                    torch.nn.Conv2d(head_in_dim, output_size * (scale ** 2), kernel_size=3, stride=1, padding=1, padding_mode=self.padding_type),
                    torch.nn.PixelShuffle(upscale_factor=scale),
                    torch.nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_type)
                )

        self.embed_dim = embed_dim

    def generate_window_mask(self, size, window_size, device):
        # Sliding window mask
        if self.spatial_dims == 1:
            indices = torch.arange(size, device=device)
            diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
            mask = torch.where(diff <= window_size, torch.tensor(0.0, device=device), torch.tensor(float('-1e9'), device=device))
            return mask
        else:
            H, W = size
            # Create a 2D coordinate grid
            y = torch.arange(H, device=device)
            x = torch.arange(W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy, xx], dim=-1).view(H * W, 2)

            # Compute differences
            diff = coords.unsqueeze(0) - coords.unsqueeze(1) # (HW, HW, 2)
            dist = torch.max(torch.abs(diff), dim=-1)[0] # Chebyshev distance for sliding window (HW, HW)

            mask = torch.where(dist <= window_size, torch.tensor(0.0, device=device), torch.tensor(float('-1e9'), device=device))
            return mask

    def get_relative_position_bias(self, window_size, device):
         if self.shifting_windows:
             if self.spatial_dims == 1:
                 coords_h = torch.arange(window_size, device=device)
                 relative_coords = coords_h[:, None] - coords_h[None, :]
                 relative_coords += window_size - 1
                 return self.relative_position_bias_table[relative_coords.view(-1)].view(window_size, window_size, -1)
             else:
                 # 2D relative position bias
                 coords_h = torch.arange(window_size, device=device)
                 coords_w = torch.arange(window_size, device=device)
                 coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
                 coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                 # 2, Wh*Ww, Wh*Ww
                 relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
                 relative_coords[0, :, :] += window_size - 1  # shift to start from 0
                 relative_coords[1, :, :] += window_size - 1
                 relative_coords[0, :, :] *= 2 * window_size - 1
                 relative_position_index = relative_coords.sum(0)  # Wh*Ww, Wh*Ww

                 return self.relative_position_bias_table[relative_position_index.view(-1)].view(
                    window_size * window_size, window_size * window_size, -1)
         else:
              pass

    def _forward_layer(self, layer, i, x_curr, pad_mode, N, B, H_p, W_p):
        shift_size = 0
        if self.shifting_windows:
            if i % 2 == 1:
                shift_size = self.window_size // 2

            if self.spatial_dims == 1:
                pad_r = (self.window_size - N % self.window_size) % self.window_size
                if pad_r > 0:
                     x_curr = torch.nn.functional.pad(x_curr, (0, 0, 0, pad_r), mode=pad_mode)
                N_padded = x_curr.shape[1]

                if shift_size > 0:
                    shifted_x = torch.roll(x_curr, shifts=-shift_size, dims=1)
                else:
                    shifted_x = x_curr

                x_windows = shifted_x.reshape(B, N_padded // self.window_size, self.window_size, self.embed_dim)
                x_windows = x_windows.reshape(-1, self.window_size, self.embed_dim)

                attn_mask = None
                if shift_size > 0:
                     img_mask = torch.zeros(1, N_padded, 1, device=x_curr.device)
                     h_slices = (slice(0, -self.window_size),
                                 slice(-self.window_size, -shift_size),
                                 slice(-shift_size, None))
                     cnt = 0
                     for h in h_slices:
                         img_mask[:, h, :] = cnt
                         cnt += 1

                     mask_windows = torch.roll(img_mask, shifts=-shift_size, dims=1)
                     mask_windows = mask_windows.view(1, N_padded // self.window_size, self.window_size, 1)
                     mask_windows = mask_windows.view(-1, self.window_size, 1)

                     attn_mask = mask_windows - mask_windows.transpose(1, 2)
                     attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-1e9')).masked_fill(attn_mask == 0, float(0.0))
                     attn_mask = attn_mask.unsqueeze(0)
                     attn_mask = attn_mask.repeat(B, 1, 1, 1).view(-1, self.window_size, self.window_size)

                if self.use_relative_positional_embedding:
                     bias = self.get_relative_position_bias(self.window_size, x_curr.device)
                     bias = bias.permute(2, 0, 1).contiguous()
                     bias = 8.0 * torch.tanh(bias / 8.0)
                     bias = bias.unsqueeze(0)

                     if attn_mask is None:
                         batch_size_windows = x_windows.size(0)
                         attn_mask = bias.repeat(batch_size_windows, 1, 1, 1)
                         attn_mask = attn_mask.view(-1, self.window_size, self.window_size)
                     else:
                         attn_mask = attn_mask.unsqueeze(1)
                         attn_mask = attn_mask + bias
                         attn_mask = attn_mask.reshape(-1, self.window_size, self.window_size)
                elif attn_mask is not None:
                     attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                     attn_mask = attn_mask.view(-1, self.window_size, self.window_size)

                x_windows = layer(x_windows, src_mask=attn_mask)

                x_windows = x_windows.reshape(-1, self.window_size, self.embed_dim)
                shifted_x = x_windows.reshape(B, N_padded // self.window_size, self.window_size, self.embed_dim)
                shifted_x = shifted_x.reshape(B, N_padded, self.embed_dim)

                if shift_size > 0:
                    x_curr = torch.roll(shifted_x, shifts=shift_size, dims=1)
                else:
                    x_curr = shifted_x

                if pad_r > 0:
                     x_curr = x_curr[:, :N, :]

            else:
                # 2D Shifting Windows
                pad_h = (self.window_size - H_p % self.window_size) % self.window_size
                pad_w = (self.window_size - W_p % self.window_size) % self.window_size

                x_curr_2d = x_curr.view(B, H_p, W_p, self.embed_dim)

                if pad_h > 0 or pad_w > 0:
                    x_curr_2d = torch.nn.functional.pad(x_curr_2d, (0, 0, 0, pad_w, 0, pad_h), mode=pad_mode)

                Hp_padded = H_p + pad_h
                Wp_padded = W_p + pad_w

                if shift_size > 0:
                    shifted_x = torch.roll(x_curr_2d, shifts=(-shift_size, -shift_size), dims=(1, 2))
                else:
                    shifted_x = x_curr_2d

                x_windows = shifted_x.view(B, Hp_padded // self.window_size, self.window_size,
                                           Wp_padded // self.window_size, self.window_size, self.embed_dim)
                x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, self.embed_dim)

                attn_mask = None
                if shift_size > 0:
                    img_mask = torch.zeros((1, Hp_padded, Wp_padded, 1), device=x_curr.device)
                    h_slices = (slice(0, -self.window_size),
                                slice(-self.window_size, -shift_size),
                                slice(-shift_size, None))
                    w_slices = (slice(0, -self.window_size),
                                slice(-self.window_size, -shift_size),
                                slice(-shift_size, None))
                    cnt = 0
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1

                    mask_windows = torch.roll(img_mask, shifts=(-shift_size, -shift_size), dims=(1, 2))
                    mask_windows = mask_windows.view(1, Hp_padded // self.window_size, self.window_size,
                                                     Wp_padded // self.window_size, self.window_size, 1)
                    mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, 1)
                    attn_mask = mask_windows - mask_windows.transpose(1, 2)
                    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-1e9')).masked_fill(attn_mask == 0, float(0.0))
                    attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1).view(-1, self.window_size * self.window_size, self.window_size * self.window_size)

                if self.use_relative_positional_embedding:
                     bias = self.get_relative_position_bias(self.window_size, x_curr.device)
                     bias = bias.permute(2, 0, 1).contiguous()
                     bias = 8.0 * torch.tanh(bias / 8.0)
                     bias = bias.unsqueeze(0)

                     if attn_mask is None:
                         batch_size_windows = x_windows.size(0)
                         attn_mask = bias.repeat(batch_size_windows, 1, 1, 1)
                         attn_mask = attn_mask.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)
                     else:
                         attn_mask = attn_mask.unsqueeze(1)
                         attn_mask = attn_mask + bias
                         attn_mask = attn_mask.reshape(-1, self.window_size * self.window_size, self.window_size * self.window_size)
                elif attn_mask is not None:
                     attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                     attn_mask = attn_mask.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)

                x_windows = layer(x_windows, src_mask=attn_mask)

                shifted_x = x_windows.view(B, Hp_padded // self.window_size, Wp_padded // self.window_size,
                                           self.window_size, self.window_size, self.embed_dim)
                shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp_padded, Wp_padded, self.embed_dim)

                if shift_size > 0:
                    x_curr_2d = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
                else:
                    x_curr_2d = shifted_x

                if pad_h > 0 or pad_w > 0:
                    x_curr_2d = x_curr_2d[:, :H_p, :W_p, :].contiguous()

                x_curr = x_curr_2d.view(B, H_p * W_p, self.embed_dim)

        else:
            # Sliding / Global
            mask = None
            if self.window_size is not None:
                if self.spatial_dims == 1:
                    mask = self.generate_window_mask(N, self.window_size, x_curr.device)
                else:
                    mask = self.generate_window_mask((H_p, W_p), self.window_size, x_curr.device)

            if self.use_relative_positional_embedding:
                 if self.spatial_dims == 1:
                     coords = torch.arange(N, device=x_curr.device)
                     relative_coords = coords[:, None] - coords[None, :]
                     if self.window_size is not None:
                         table_size = 2 * self.window_size + 1
                         relative_coords += self.window_size
                         relative_coords = torch.clamp(relative_coords, 0, table_size - 1)
                     else:
                         offset = self.window_len // 2
                         relative_coords += offset
                         relative_coords = torch.clamp(relative_coords, 0, self.window_len - 1)

                     bias = self.relative_position_bias_table[relative_coords.view(-1)].view(N, N, -1)
                     bias = bias.permute(2, 0, 1).contiguous()
                     bias = 8.0 * torch.tanh(bias / 8.0)
                 else:
                     # 2D relative PE for global/sliding is complex.
                     # We'll approximate or use a simplified approach, similar to shifting windows
                     coords_h = torch.arange(H_p, device=x_curr.device)
                     coords_w = torch.arange(W_p, device=x_curr.device)
                     coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
                     coords_flatten = torch.flatten(coords, 1)
                     relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

                     if self.window_size is not None:
                         relative_coords[0, :, :] += self.window_size
                         relative_coords[1, :, :] += self.window_size
                         relative_coords = torch.clamp(relative_coords, 0, 2 * self.window_size)
                         relative_coords[0, :, :] *= 2 * self.window_size + 1
                     else:
                         offset = self.window_len // 2
                         relative_coords[0, :, :] += offset
                         relative_coords[1, :, :] += offset
                         relative_coords = torch.clamp(relative_coords, 0, self.window_len - 1)
                         relative_coords[0, :, :] *= self.window_len

                     relative_position_index = relative_coords.sum(0)
                     bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(N, N, -1)
                     bias = bias.permute(2, 0, 1).contiguous()
                     bias = 8.0 * torch.tanh(bias / 8.0)

                 if mask is not None:
                     mask_expanded = mask.unsqueeze(0)
                     combined = mask_expanded + bias
                     combined = combined.repeat(B, 1, 1)
                     mask = combined
                 else:
                     mask = bias.repeat(B, 1, 1)

            x_curr = layer(x_curr, src_mask=mask)
        return x_curr

    def forward(self, x):
        original_x = x
        pad_mode = 'constant' if self.padding_type == 'zeros' else self.padding_type

        if self.positional_embedding:
            if callable(self.positional_embedding):
                features = self.positional_embedding(x)
                x = torch.cat([x, features], dim=1)

        if self.spatial_dims == 1:
            B, C, L = x.shape

            pad_len = (self.patch_size - (L % self.patch_size)) % self.patch_size
            if pad_len > 0:
                x = torch.nn.functional.pad(x, (0, pad_len), mode=pad_mode)

            x_emb = self.patch_embed(x)
            x_emb = x_emb.transpose(1, 2)
            N = x_emb.shape[1]
            orig_size = L
            padded_size = L + pad_len

        else:
            B, C, H, W = x.shape

            pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
            pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size

            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

            x_emb = self.patch_embed(x) # (B, D, H_p, W_p)
            H_p, W_p = x_emb.shape[2], x_emb.shape[3]
            x_emb = x_emb.flatten(2).transpose(1, 2) # (B, H_p * W_p, D)
            N = x_emb.shape[1]
            orig_h, orig_w = H, W
            padded_h, padded_w = H + pad_h, W + pad_w

        if self.positional_embedding and not callable(self.positional_embedding) and not self.use_relative_positional_embedding:
            # Sinusoidal Absolute
            if self.spatial_dims == 1:
                pe = torch.zeros(N, self.embed_dim, device=x.device)
                position = torch.arange(0, N, dtype=torch.float, device=x.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / self.embed_dim))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                x_emb = x_emb + pe.unsqueeze(0)
            else:
                # 2D Sinusoidal PE
                pe = torch.zeros(N, self.embed_dim, device=x.device)
                y_pos = torch.arange(H_p, dtype=torch.float, device=x.device).unsqueeze(1).repeat(1, W_p).view(-1, 1)
                x_pos = torch.arange(W_p, dtype=torch.float, device=x.device).unsqueeze(0).repeat(H_p, 1).view(-1, 1)

                div_term = torch.exp(torch.arange(0, self.embed_dim // 2, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / (self.embed_dim // 2)))

                # y encodes first half, x encodes second half
                half_dim = self.embed_dim // 2
                pe[:, 0:half_dim:2] = torch.sin(y_pos * div_term)
                pe[:, 1:half_dim:2] = torch.cos(y_pos * div_term)
                pe[:, half_dim::2] = torch.sin(x_pos * div_term)
                pe[:, half_dim+1::2] = torch.cos(x_pos * div_term)

                x_emb = x_emb + pe.unsqueeze(0)

        x_curr = x_emb

        for i, layer in enumerate(self.layers):
            if self.checkpoints:
                x_curr = torch.utils.checkpoint.checkpoint(
                    self._forward_layer, layer, i, x_curr, pad_mode, N, B,
                    H_p if self.spatial_dims != 1 else None,
                    W_p if self.spatial_dims != 1 else None,
                    use_reentrant=False
                )
            else:
                x_curr = self._forward_layer(
                    layer, i, x_curr, pad_mode, N, B,
                    H_p if self.spatial_dims != 1 else None,
                    W_p if self.spatial_dims != 1 else None
                )

        if not self.use_pixel_shuffle:
            x_out = self.head(x_curr)

            if self.spatial_dims == 1:
                x_out = x_out.view(B, N, self.output_size, self.patch_size)
                x_out = x_out.permute(0, 2, 1, 3).reshape(B, self.output_size, -1)
                if pad_len > 0:
                    x_out = x_out[..., :orig_size]
            else:
                x_out = x_out.view(B, H_p, W_p, self.output_size, self.patch_size, self.patch_size)
                # B, H_p, W_p, C, P_h, P_w -> B, C, H_p, P_h, W_p, P_w
                x_out = x_out.permute(0, 3, 1, 4, 2, 5).contiguous()
                x_out = x_out.view(B, self.output_size, H_p * self.patch_size, W_p * self.patch_size)
                if pad_h > 0 or pad_w > 0:
                    x_out = x_out[:, :, :orig_h, :orig_w]
        else:
            if self.spatial_dims == 1:
                x_curr = x_curr.transpose(1, 2) # [B, head_in_dim, N]
                x_out = self.head(x_curr)
                if pad_len > 0:
                    x_out = x_out[..., :orig_size]
            else:
                x_curr = x_curr.transpose(1, 2) # [B, head_in_dim, N]
                x_curr = x_curr.view(B, -1, H_p, W_p).contiguous() # [B, head_in_dim, H_p, W_p]
                x_out = self.head(x_curr)
                if pad_h > 0 or pad_w > 0:
                    x_out = x_out[:, :, :orig_h, :orig_w]

        return x_out


class PatchTST(torch.nn.Module):
    def __init__(self, input_size, output_size, patch_size=4, embed_dim=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        if input_size != output_size:
            raise ValueError(f"PatchTST requires input_size ({input_size}) == output_size ({output_size})")

        self.input_size = input_size
        self.patch_size = patch_size

        # Shared backbone for 1 channel
        self.backbone = ViT(input_size=1, output_size=1, patch_size=patch_size,
                              embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                              dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)
        x = self.backbone(x)
        x = x.reshape(B, C, L)
        return x
