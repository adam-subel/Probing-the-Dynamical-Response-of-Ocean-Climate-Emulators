import torch

import numpy as np
import xarray as xr
import torch.nn.functional as F
import torch.nn as nn

class transformation:
    def __init__(self, transformations: tuple[str,...], device_name: str = 'cpu'):
        self.transformations = transformations
        for transform in self.transformations:
            transform.update_device(device_name)
    
    def update_device(self, device_name):
        for transform in self.transformations:
            transform.update_device(device_name)
    
    def __call__(self,x):
        for transform in self.transformations:
            x = transform(x)
        return x
            
class NullTransform:
    def __init__(self, device_name: str = 'cpu'):
        self.device_name = device_name
    def update_device(self, device_name):
        self.device_name = device_name    
    
    def __call__(self, x):
        return x
    

class normalize:
    def __init__(self, means, stds, device_name: str = 'cpu'):
        self.means = means.to(device = device_name)
        self.stds = stds.to(device = device_name)
        self.device_name = device_name
    
    def update_device(self, device_name):
        self.means = self.means.to(device = device_name)
        self.stds = self.stds.to(device = device_name)
        self.device_name = device_name    
    
    def __call__(self, x):
        return (x - self.means)/self.stds
        
        
class inv_normalize:
    def __init__(self, means, stds, device_name: str = 'cpu'):
        self.means = means.to(device = device_name)
        self.stds = stds.to(device = device_name)
        self.device_name = device_name
        
    def update_device(self, device_name):
        self.means = self.means.to(device = device_name)
        self.stds = self.stds.to(device = device_name)
        self.device_name = device_name    
    
    def __call__(self, x):
        return x*self.stds + self.means
    
class apply_wet_mask:
    def __init__(self, data, variables, device_name: str = 'cpu'):
        data = data[variables]
        if 'lev' not in list(data.coords):
            data = data.expand_dims('lev')
            
        mask = xr.where(np.isnan(data.isel(time = 0).to_array().stack(channel = ('variable','lev')).transpose('channel',...)),
                             0.0,1.0).to_numpy()
        self.mask = torch.asarray(mask).to(torch.float).to(device = device_name)
        self.device_name = device_name

    def update_device(self, device_name):
        self.mask = self.mask.to(device = device_name)
        self.device_name = device_name   
    
    def __call__(self, x):
        return x * self.mask



class FillLand:
    def __init__(self, 
                 data,
                 variables,
                 method: str = 'nearest',
                 *,
                 mean,
                 std,                 
                 device_name: str = 'cpu',
                 mask = None,
                ):
        if not mask:
            mask = xr.where(np.isnan(data[variables].isel(time = 0).to_array().stack(channel = ('variable','lev')).transpose('channel',...)),
                             1.0,0.0).to_numpy()
            
        self.mask = mask
        
        mean_values = data[variables].mean('time').compute()

        if method == 'nearest':
            interp_x = mean_values.interpolate_na(dim = 'x',method = 'nearest',fill_value="extrapolate")
            interp_y = mean_values.interpolate_na(dim = 'y',method = 'nearest',fill_value="extrapolate") 
            
            interp = ((interp_x.fillna(interp_y) + interp_y.fillna(interp_x))/2)
            interp = interp.fillna(interp.min(['x','y']))
        if method == 'linear':
            interp_x = mean_values.interpolate_na(dim = 'x',method = 'linear',period = 360)
            interp_y = mean_values.interpolate_na(dim = 'y',method = 'linear',period = 180) 
            
            interp = ((interp_x.fillna(interp_y) + interp_y.fillna(interp_x))/2)
            interp = interp.fillna(interp.min(['x','y']))
            
        fill_values = interp.to_array().stack(channel = ('variable','lev')).transpose('channel',...).to_numpy()
        fill_values = (fill_values - np.asarray(mean))/np.asarray(std)
        self.fill_values = torch.asarray(fill_values*self.mask).to(torch.float).to(device = device_name)
        
    def update_device(self, device_name):
        self.fill_values = self.fill_values.to(device = device_name)
        self.device_name = device_name   
    
    def __call__(self, x):
        return x + self.fill_values
    
class LevelWeight:
    def __init__(self, level_weights: torch.Tensor|np.ndarray|list,
                 label_values):
        level_weights = torch.asarray(level_weights)
        if len(level_weights) == label_values.input_size:
            self.level_weights = level_weights.reshape((-1,1,1))
        else:
            stacked_level_weights = torch.zeros(label_values.input_size)
            index_start = 0            
            for var in label_values.variables:
                var_size = label_values.variable_size[var]
                if var_size == 1:
                    variable_weights = 1.0
                else:
                    variable_weights = torch.asarray(level_weights)
                stacked_level_weights[index_start:index_start+var_size] = variable_weights
                index_start += var_size
            self.level_weights = stacked_level_weights.reshape((-1,1,1))

    def update_device(self, device_name):
        self.level_weights = self.level_weights.to(device = device_name)
        self.device_name = device_name          
            
    def __call__(self, output):
        return self.level_weights*output

class ExtractVariables:
    def __init__(self, variables: tuple[str,...], 
                 label_values,
                 ocean_surface: bool = False,
                 num_levels: int|None = None,
                ):

        indices = []
        index_start = 0            
        for var in label_values.variables:
            var_size = label_values.variable_size[var]
            if var in variables:
                if ocean_surface:
                    indices+=list(range(index_start,index_start+1))
                elif num_levels:
                    indices+=list(range(index_start,index_start+min(num_levels,var_size)))
                else:
                    indices+=list(range(index_start,index_start+var_size))
                    
            index_start += var_size
        self.indices = indices

    def update_device(self, device_name):
        self.device_name = device_name          
            
    def __call__(self, output):
        return output[...,self.indices,:,:]     

class add_noise:
    def __init__(self, 
                 noise_path: str = '/pscratch/sd/a/asubel/Data/Chapter_2/Noise_Map_so_thetao.nc',
                 variables: list = ['thetao','so'],
                 noise_level: float = .25,
                 device_name: str = 'cpu',
                 noise_frequency: float | None = None,
                 dt_scaling = None,
                 rand_function = torch.rand_like,
                ):
        noise_mask = xr.open_dataset(noise_path)
        noise_mask = noise_mask[variables]
        if 'lev' not in list(noise_mask.coords):
            noise_mask = noise_mask.expand_dims('lev')
        noise_mask = noise_mask.to_array().stack(channel = ('variable','lev')).transpose(...,'channel','y','x').to_numpy()
        if dt_scaling is not None:
            noise_mask = noise_mask*dt_scaling
        self.noise_mask = torch.asarray(noise_mask).to(torch.float).to(device = device_name)
        self.noise_level = noise_level
        self.device_name = device_name
        self.noise_frequency = noise_frequency
        self.spatial_dims = (1,)*len(self.noise_mask.shape)
        self.rand_function = rand_function
        
    def update_device(self, device_name):
        self.noise_mask = self.noise_mask.to(device = device_name)
        self.device_name = device_name   

    def __call__(self, x):
        if self.noise_frequency:
            indices = torch.rand(x.shape[0],*self.spatial_dims).to(device = self.device_name)<self.noise_frequency
            return x + self.noise_mask * self.rand_function(x)*indices*self.noise_level
        else:
            return x + self.noise_mask * self.rand_function(x)*self.noise_level


class add_noise_AR:
    def __init__(self, 
                 noise_mask,
                 rho = 0,
                 noise_level: float = .25,
                 device_name: str = 'cpu',
                 noise_frequency: float | None = None,
                 dt_scaling = None,
                 rand_function = torch.rand_like,
                ):
        if dt_scaling is not None:
            noise_mask = noise_mask*dt_scaling
        self.noise_mask = torch.asarray(noise_mask).to(torch.float).to(device = device_name)
        self.noise_level = noise_level
        self.device_name = device_name
        self.noise_frequency = noise_frequency
        self.spatial_dims = (1,)*len(self.noise_mask.shape)
        self.rand_function = rand_function
        self.rho = rho
        self.state = self.rand_function(self.noise_mask)
        
    def update_device(self, device_name):
        self.noise_mask = self.noise_mask.to(device = device_name)
        self.state = self.state.to(device = device_name)
        self.device_name = device_name   

    def __call__(self, x):
        self.state = self.state*self.rho + self.rand_function(self.state)
        if self.noise_frequency:
            indices = torch.rand(x.shape[0],*self.spatial_dims).to(device = self.device_name)<self.noise_frequency
            return x + self.noise_mask * self.state*indices*self.noise_level
        else:
            return x + self.noise_mask * self.state*self.noise_level


class mask_bcs:
    def __init__(self, 
                 boundary_variables: list = ['tauuo','tauvo','hfds'],
                 mask_frequency: float | dict = .3,
                 device_name: 'str' = 'cpu'
                ):
        
        self.device_name = device_name
        self.boundary_variables = boundary_variables
        self.n_variables = len(boundary_variables)
        if isinstance(mask_frequency,dict) :
            self.mask_frequency_tensor = torch.zeros(self.n_variables)
            for i,var in enumerate(boundary_variables):
                self.mask_frequency_tensor[i] = mask_frequency[var]
            self.mask_frequency_tensor = self.mask_frequency_tensor.reshape((-1,1,1))
        else:
            self.mask_frequency_tensor = torch.ones((self.n_variables,1,1))*mask_frequency
        self.mask_frequency_tensor = self.mask_frequency_tensor.to(device = self.device_name)
        
    def update_device(self, device_name):
        self.mask_frequency_tensor = self.mask_frequency_tensor.to(device = device_name)
        self.device_name = device_name   

    def __call__(self, x):
        indices = torch.rand((*x.shape[:-3],self.n_variables,1,1),device = self.device_name)>self.mask_frequency_tensor
        return x * indices


class LocalScaling():
    def __init__(self,
                 channels: int,
                 stencil_size: int,
                 wet_mask: torch.Tensor):
        """
        Initializes the LocalScalingLayer.

        Args:
            channels (int): The number of input channels (variables * levels).
            stencil_size (int): The size of the square stencil for local averaging (e.g., 5 for a 5x5 stencil).
            wet_mask (torch.Tensor): A tensor of shape (C, H, W) where ocean points are 1.0 and land points are 0.0.
        """
        if stencil_size % 2 == 0:
            raise ValueError("stencil_size must be an odd number.")

        self.channels = channels
        self.stencil_size = stencil_size

        self.pady = (stencil_size - 1) // 2
        self.padx = (stencil_size - 1) // 2


        self.avg_conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=stencil_size,
            groups=channels,
            bias=False,
            padding='valid' 
        )

        # Initialize the weights to 1, so it sums the values in the stencil.
        torch.nn.init.constant_(self.avg_conv.weight, 1)
        self.avg_conv.weight.requires_grad = False

        self.wet_mask = wet_mask.unsqueeze(0)
        with torch.no_grad(): 
            padded_mask = F.pad(self.wet_mask, (self.padx, self.padx, 0, 0), mode='circular')
            padded_mask = F.pad(padded_mask, (0, 0, self.pady, self.pady), mode='constant')
            
            sum_mask = self.avg_conv(padded_mask)
            sum_mask = sum_mask * wet_mask

        # Register the pre-computed sum_mask as a buffer.
        self.sum_mask = sum_mask

    def update_device(self, device_name):
        self.avg_conv = self.avg_conv.to(device = device_name)
        self.wet_mask = self.wet_mask.to(device = device_name)
        self.sum_mask = self.sum_mask.to(device = device_name)
        self.device_name = device_name  

    def __call__(self, fts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the local scaling transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            A tuple containing:
            - torch.Tensor: The locally centered data (x - local_mean).
            - torch.Tensor: The computed local mean.
        """
        masked_x = fts * self.wet_mask

        padded_x = F.pad(masked_x, (self.padx, self.padx, 0, 0), mode='circular')
        padded_x = F.pad(padded_x, (0, 0, self.pady, self.pady), mode='constant')

        sum_x = self.avg_conv(padded_x)

        local_mean = sum_x / (self.sum_mask + 1e-8)
        local_mean = torch.nan_to_num(local_mean, nan=0.0)
        local_mean = local_mean * self.wet_mask

        centered_x = fts - local_mean
        
        return centered_x, local_mean


class LocalSmoothing():
    def __init__(self,
                 channels: int,
                 stencil_size: int,
                 wet_mask: torch.Tensor,
                 device_name: str = 'cpu'):
        """
        Initializes the LocalScalingLayer.

        Args:
            channels (int): The number of input channels (variables * levels).
            stencil_size (int): The size of the square stencil for local averaging (e.g., 5 for a 5x5 stencil).
            wet_mask (torch.Tensor): A tensor of shape (C, H, W) where ocean points are 1.0 and land points are 0.0.
        """
        if stencil_size % 2 == 0:
            raise ValueError("stencil_size must be an odd number.")

        self.channels = channels
        self.stencil_size = stencil_size

        self.pady = (stencil_size - 1) // 2
        self.padx = (stencil_size - 1) // 2


        coords = torch.arange(-(self.stencil_size // 2), (self.stencil_size // 2) + 1)
        sigma = 0.3 * (stencil_size // 2 - 1) + 0.8
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_2d = torch.outer(g, g).reshape((1,1,self.stencil_size,self.stencil_size))

        
        self.avg_conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=stencil_size,
            groups=channels,
            bias=False,
            padding='valid' 
        )

        # Initialize the weights to 1, so it sums the values in the stencil.
        with torch.no_grad():
            self.avg_conv.weight.copy_(kernel_2d)
        self.avg_conv.weight.requires_grad = False

        self.wet_mask = wet_mask.unsqueeze(0)
        with torch.no_grad(): 
            padded_mask = F.pad(self.wet_mask, (self.padx, self.padx, 0, 0), mode='circular')
            padded_mask = F.pad(padded_mask, (0, 0, self.pady, self.pady), mode='constant')
            
            sum_mask = self.avg_conv(padded_mask)
            sum_mask = sum_mask * wet_mask
        self.sum_mask = sum_mask
            
        self.avg_conv = self.avg_conv.to(device = device_name)
        self.wet_mask = self.wet_mask.to(device = device_name)
        self.sum_mask = self.sum_mask.to(device = device_name)
        

    def update_device(self, device_name):
        self.avg_conv = self.avg_conv.to(device = device_name)
        self.wet_mask = self.wet_mask.to(device = device_name)
        self.sum_mask = self.sum_mask.to(device = device_name)
        self.device_name = device_name  

    def __call__(self, fts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the local scaling transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            A tuple containing:
            - torch.Tensor: The locally centered data (x - local_mean).
            - torch.Tensor: The computed local mean.
        """
        masked_x = fts * self.wet_mask

        padded_x = F.pad(masked_x, (self.padx, self.padx, 0, 0), mode='circular')
        padded_x = F.pad(padded_x, (0, 0, self.pady, self.pady), mode='constant')

        sum_x = self.avg_conv(padded_x)

        local_mean = sum_x / (self.sum_mask+1e-8)
        local_mean = torch.nan_to_num(local_mean, nan=0.0)
        local_mean = local_mean * self.wet_mask

        
        return local_mean   



class LaplacianDiffusion():
    def __init__(self, 
                 channels: int, 
                 diffusion_coeff: float | list[float], 
                 wet_mask: torch.Tensor,
                 device_name: str = 'cpu'):        

        if isinstance(diffusion_coeff, float):
            self.coeffs = torch.tensor([diffusion_coeff] * channels, device=device_name)
        else:
            self.coeffs = torch.tensor(diffusion_coeff, device=device_name)
        self.coeffs = self.coeffs.view(1, -1, 1, 1)

        self.wet_mask = wet_mask.to(device = device_name)
        
        kernel_base = torch.tensor([[0., 1., 0.],
                                    [1., 0., 1.],
                                    [0., 1., 0.]], device=device_name)
        
        self.kernel = kernel_base.expand(channels, 1, 3, 3)
        self.channels = channels
        
        # 4. Pre-compute the "Neighbor Count" for No-Flux condition
        # This tells us how many valid water neighbors each cell has (0 to 4)
        # We do this once during init to save time.
        with torch.no_grad():
            padded_mask = self._pad_periodic(self.wet_mask.float())
            self.valid_neighbor_count = F.conv2d(
                padded_mask, 
                self.kernel, 
                padding=0, 
                groups=self.channels
            )

    def update_device(self, device_name):

        self.wet_mask = self.wet_mask.to(device_name)
        self.coeffs = self.coeffs.to(device_name)
        self.kernel = self.kernel.to(device_name)
        self.valid_neighbor_count = self.valid_neighbor_count.to(device_name)

    def _pad_periodic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads:
        - Left/Right: Circular (Wraps around for Zonal Periodicity)
        - Top/Bottom: Constant 0 (No flux across poles/land)
        """
        # Pad width: (Left, Right, Top, Bottom)
        # 1. Circular pad West-East
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        # 2. Zero pad North-South (Land/Walls)
        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        return x
    
    def __call__(self, fts: torch.Tensor) -> torch.Tensor:
        """
        Computes nu * Laplacian(state) with no-flux boundaries.
        """
        # B. Sum of valid neighbors
        padded_fts = self._pad_periodic(fts)
        
        # 2. Sum of valid neighbors (Convolution)
        # Note: padding=0 because we manually padded
        sum_neighbors = F.conv2d(
            padded_fts, 
            self.kernel, 
            padding=0, 
            groups=self.channels
        )

        laplacian = sum_neighbors - (self.valid_neighbor_count * fts)
        
        # D. Apply diffusion coefficient
        tendency = self.coeffs * laplacian
        
        # E. Re-mask to ensure no diffusion generated on land (just in case)
        return tendency * self.wet_mask