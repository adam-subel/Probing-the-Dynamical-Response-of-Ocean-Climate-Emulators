import torch
import functools
import abc
import sys
import copy as copy
sys.path.append("../")

import xarray as xr
import numpy as np
import Utils.transformations as transformations
# import Utils.fft_tools as fft_tools
import torch.distributed as dist

class IdentityWeight():
    def __init__(self):
        self.function = torch.nn.Identity()
        
    def update_device(self, device_name):
        self.device_name = device_name          
    
    def __call__(self, output, step):
        del step
        return self.function(output)
    
class AreaWeight():
    def __init__(self, data: xr.DataArray,scaling_function: callable = torch.nn.Identity()):
        area = data.areacello/data.areacello.max()
        self.area_weight = torch.asarray(area.to_numpy()).to(torch.float).reshape((1,*area.shape))
        self.area_weight = scaling_function(self.area_weight)
        
    def update_device(self, device_name):
        self.area_weight = self.area_weight.to(device = device_name)
        self.device_name = device_name          
    
    def __call__(self, output, step):
        del step
        return self.area_weight*output        

class AreaWeightedMean():
    def __init__(self,
                 area_weights: torch.Tensor,
                 wet_mask: torch.Tensor,
                ):       
        area_weights = area_weights.squeeze() 
        wet_mask = wet_mask.squeeze()
        if area_weights.dim() != 2:
            raise ValueError(f"area_weights must be a 2D tensor (H, W), but got shape {area_weights.shape}")
        if wet_mask.dim() != 3:
            raise ValueError(f"wet_mask must be a 3D tensor (C, H, W), but got shape {wet_mask.shape}")
        

        area_weights_bc = area_weights.unsqueeze(0)
        masked_area_weights = wet_mask * area_weights_bc
        total_area_per_channel = masked_area_weights.sum(dim=(-2, -1))
        

        self.masked_area_weights_bc = masked_area_weights.unsqueeze(0)
        self.total_area_per_channel_bc = total_area_per_channel.reshape((1,-1,1,1))

    def update_device(self, device_name):
        self.masked_area_weights_bc = self.masked_area_weights_bc.to(device = device_name)
        self.total_area_per_channel_bc = self.total_area_per_channel_bc.to(device = device_name)
        self.device_name = device_name    

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        weighted_sum = (output * self.masked_area_weights_bc).sum(dim=(-2, -1), keepdim=True)
        
        mean = weighted_sum / (self.total_area_per_channel_bc + 1e-9)
        
        return mean


class SelectRegion():
    def __init__(self, data: xr.DataArray, lat_bounds, lon_bounds):
        lower_lat = np.argwhere(data.lat.values>lat_bounds[0]).min(axis=0)[0]
        upper_lat = np.argwhere(data.lat.values<lat_bounds[1]).max(axis=0)[0]
        lower_lon = np.argwhere(data.lon.values>lon_bounds[0]).min(axis=0)[1]
        upper_lon = np.argwhere(data.lon.values<lon_bounds[1]).max(axis=0)[1]        
        self.lat_slice = slice(lower_lat,upper_lat)
        self.lon_slice = slice(lower_lon,upper_lon)
        
        
    def update_device(self, device_name):
        self.device_name = device_name          
    
    def __call__(self, output, step):
        del step
        return output[...,self.lat_slice,self.lon_slice]


class LevelWeight():
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
            self.level_weights = stacked_level_weights.to(torch.float).reshape((-1,1,1))

    def update_device(self, device_name):
        self.level_weights = self.level_weights.to(device = device_name)
        self.device_name = device_name          
            
    def __call__(self, output, step):
        del step
        return self.level_weights*output

class StepWeightArray():
    def __init__(self, step_weights: torch.Tensor|np.ndarray|list):
        self.step_weights = step_weights
        
    def update_device(self, device_name):
        self.device_name = device_name      
        
    def __call__(self, output, step):
        return self.step_weights[step]*output

class StepWeightFunction():
    def __init__(self, step_weight_function: callable):
        self.step_weight_function = step_weight_function
        
    def update_device(self, device_name):
        self.device_name = device_name   
        
    def __call__(self, output, step):
        return self.step_weight_function(step).to*output 

class ComposedWeight():
    def __init__(self, weightings: tuple):
        self.weightings = weightings
    
    def update_device(self, device_name):
        for weighting in self.weightings:
            weighting.update_device(device_name)
        self.device_name = device_name
            
    def __call__(self, output, step):
        for weighting in self.weightings:
            output = weighting(output,step)        
        return output             
    
class SumWeights():
    def __init__(self, weightings: tuple):
        self.weightings = weightings
    
    def update_device(self, device_name):
        for weighting in self.weightings:
            weighting.update_device(device_name)
        self.device_name = device_name
            
    def __call__(self, output, step):
        final_output = 0
        for weighting in self.weightings:
            final_output += weighting(output,step)        
        return final_output      
    
class SelectVariables():
    def __init__(self, variables: tuple[str,...], label_values):

        stacked_level_weights = torch.zeros(label_values.input_size)
        index_start = 0            
        for var in label_values.variables:
            var_size = label_values.variable_size[var]
            if var_size == 1:
                variable_weights = 0.0
            else:
                variable_weights = torch.zeros(var_size)
            if var in variables:
                variable_weights += 1.0
            stacked_level_weights[index_start:index_start+var_size] = variable_weights
            index_start += var_size
        self.level_weights = stacked_level_weights.reshape((-1,1,1))

    def update_device(self, device_name):
        self.level_weights = self.level_weights.to(device = device_name)
        self.device_name = device_name          
            
    def __call__(self, output, step):
        del step
        return self.level_weights*output    
    
class SelectVariablesColumn():
    def __init__(self, variables: tuple[str,...], label_values):

        stacked_level_weights = torch.zeros(label_values.input_size)
        index_start = 0            
        for var in label_values.variables:
            var_size = label_values.variable_size[var]
            if var_size == 1:
                variable_weights = 0.0
            else:
                variable_weights = torch.zeros(var_size)
            if var in variables:
                variable_weights += 1.0
            stacked_level_weights[index_start:index_start+var_size] = variable_weights
            index_start += var_size
        self.level_weights = stacked_level_weights

    def update_device(self, device_name):
        self.level_weights = self.level_weights.to(device = device_name)
        self.device_name = device_name          
            
    def __call__(self, output, step):
        del step
        return self.level_weights*output    
        
class ExtractVariables:
    def __init__(self, variables: tuple[str,...], 
                 label_values,
                 ocean_surface: bool = False,
                 num_levels: int|None = None,
                 level_start: int|None = None
                ):

        indices = []
        index_start = 0            
        for var in label_values.variables:
            var_size = label_values.variable_size[var]
            if var in variables:
                if ocean_surface:
                    indices+=list(range(index_start,index_start+1))
                elif num_levels and level_start:
                    indices+=list(range(index_start+level_start,index_start+min(num_levels+level_start,var_size)))
                elif num_levels:
                    indices+=list(range(index_start,index_start+min(num_levels,var_size)))
                elif level_start:
                    indices+=list(range(index_start+level_start,index_start+min(num_levels+level_start,var_size)))                    
                else:
                    indices+=list(range(index_start,index_start+var_size))
                    
            index_start += var_size
        self.indices = indices

    def update_device(self, device_name):
        self.device_name = device_name          
            
    def __call__(self, output, step):
        del step
        return output[...,self.indices,:,:]    

class TendecyLoss():
    def __init__(self,
                 variables,
                 data,
                 label_values,
                 means,
                 stds,
                 wet_transform,
                ):
        self.variables = variables
        self.area_weight = AreaWeight(data)
        self.dz_weight = LevelWeight((data.dz.values/data.dz.values.max()),label_values) 
        self.inv_norms = transformations.inv_normalize(means.squeeze(), stds.squeeze())
        
        self.select_variables = []
        self.rescalings = []
        for var in variables:
            self.select_variables.append(SelectVariablesColumn([var,],label_values))
            self.rescalings.append(self.select_variables[-1](self.inv_norms(self.dz_weight(self.area_weight(wet_transform(torch.ones((1,) + label_values.input_shape)),0),0).sum([2,3])),0).sum(1))


        
    def update_device(self, device_name):
        self.area_weight.update_device(device_name)
        self.dz_weight.update_device(device_name)
        self.inv_norms.update_device(device_name)
        for select_variable in self.select_variables:
            select_variable.update_device(device_name)  
        for i, rescaling in enumerate(self.rescalings):
            self.rescalings[i] = rescaling.to(device = device_name)         
        self.device_name = device_name               
    
    def __call__(self,label,pred,step):
        del label
        del step 
        loss = 0
        for i, var in enumerate(self.variables):
            loss += (self.select_variables[i](self.inv_norms(self.dz_weight(self.area_weight(pred,0),0).sum([2,3])),0).sum(1))/self.rescalings[i] 
        return torch.abs(loss)
    
class ScalarWeight():
    def __init__(self, weight):
        self.weight = weight
        
    def update_device(self, device_name):
        self.device_name = device_name          
    
    def __call__(self, output, step):
        del step
        return output*self.weight

class DynamicLevelScaling():
    def __init__(self,
                 label_values,
                 post_weighting,
                 relaxation_period = 25,
                 std_vals = None,
                 world_size = None
                 ):
        self.level_weights = torch.ones(label_values.input_size).reshape((-1,1,1))
        self.post_weighting = post_weighting
        self.relaxation_period = relaxation_period
        self.world_size = world_size
        self.std_vals = (1/std_vals).reshape(-1,1,1)
            
        
    def update_device(self, device_name):
        self.level_weights = self.level_weights.to(device = device_name)
        if self.std_vals is not None:
            self.std_vals = self.std_vals.to(device = device_name)
        self.post_weighting.update_device(device_name)
        self.device_name = device_name          

    @torch.no_grad()
    def update_weights(self,label,pred,step):
        squared_difference = self.post_weighting((label-pred)**2,step)
        squared_difference = torch.where(squared_difference==0,1e-8,squared_difference)
        new_level_weights = 1/squared_difference.mean([-4,-2,-1]).reshape((-1,1,1))
        if self.std_vals is not None:
            new_level_weights = torch.minimum(new_level_weights,self.std_vals)
        if self.world_size:
            torch.distributed.all_reduce(new_level_weights,op= dist.ReduceOp.SUM)  
            new_level_weights = new_level_weights/self.world_size
        self.level_weights = (self.level_weights*(self.relaxation_period-1)+new_level_weights)/self.relaxation_period
        
    def __call__(self, output, step):
        del step
        return self.level_weights*output

class DynamicLevelScalingMultistep():
    def __init__(self,
                 label_values,
                 post_weighting,
                 max_steps,
                 relaxation_period = 100,
                 inv_max_vals = None,
                 max_ratio: float = 100.0,
                 world_size = None,
                 rand_threshold: int | None = None,
                 sqrt: bool = False,
                 ):
        self.level_weights = torch.ones((max_steps,label_values.input_size)).reshape((max_steps,-1,1,1))
        self.post_weighting = post_weighting
        self.relaxation_period = relaxation_period
        self.world_size = world_size
        self.max_vals = (1/inv_max_vals).reshape(-1,1,1) if inv_max_vals is not None else None
        self.new_level_weights = torch.ones_like(self.level_weights)
        self.rand_threshold = rand_threshold    
        self.max_ratio = max_ratio
        self.sqrt = sqrt
        
    def update_device(self, device_name) -> None:
        self.level_weights = self.level_weights.to(device = device_name)
        self.new_level_weights = self.new_level_weights.to(device = device_name)
        if self.max_vals is not None:
            self.max_vals = self.max_vals.to(device = device_name)
        self.post_weighting.update_device(device_name)
        self.device_name = device_name          
        
    @torch.no_grad()
    def update_new_weights(self,label,pred,step):
        squared_difference = self.post_weighting((label-pred)**2,step)
        squared_difference = torch.where(squared_difference==0,1e-8,squared_difference)
        if self.sqrt:
            new_level_weights = 1/torch.sqrt(squared_difference.mean([-4,-2,-1]).reshape((-1,1,1)))
        else:
            new_level_weights = 1/squared_difference.mean([-4,-2,-1]).reshape((-1,1,1))
            
        if self.max_vals is not None:
            new_level_weights = torch.minimum(new_level_weights,self.max_vals)
        new_min = new_level_weights.min()
        new_level_weights = new_level_weights.clamp(new_min,new_min*self.max_ratio)            
        self.new_level_weights[step] = new_level_weights

    def apply_new_weights(self):
        if self.world_size:
            torch.distributed.all_reduce(self.new_level_weights,op= dist.ReduceOp.SUM)  
            self.new_level_weights = self.new_level_weights/self.world_size        
        self.level_weights = (self.level_weights*(self.relaxation_period-1)+self.new_level_weights)/self.relaxation_period        
        self.new_level_weights = torch.ones_like(self.new_level_weights)

        
    def __call__(self, output, step):
        if self.rand_threshold:
            if torch.rand(1)[0] < self.rand_threshold:
                return self.level_weights[step]*output
            else:
                return output
        else:
            return self.level_weights[step]*output



class MeanLoss():
    def __init__(self,
                 wet_transform,
                 area_weighting,
                 pre_weighting: callable = IdentityWeight(),
                 post_weighting: callable = IdentityWeight()
                ):
        self.masked_area = wet_transform(area_weighting.area_weight.squeeze())
        self.level_rescaling = self.masked_area.sum(dim=(-1,-2))
        self.pre_weighting = pre_weighting
        self.post_weighting = post_weighting        
        
    def update_device(self, device_name):
        self.pre_weighting.update_device(device_name)
        self.post_weighting.update_device(device_name)
        self.masked_area = self.masked_area.to(device = device_name)
        self.level_rescaling = self.level_rescaling.to(device = device_name)
        self.device_name = device_name               
    
    def __call__(self,label,pred,step):
        label_mean = (self.pre_weighting(label,step)*self.masked_area).sum(dim=(-1,-2))
        pred_mean = (self.pre_weighting(pred,step)*self.masked_area).sum(dim=(-1,-2))
        error = torch.abs(label_mean-pred_mean)/self.level_rescaling
        return self.post_weighting(error,step).mean()
    

class GlobalMeanLoss():
    def __init__(self,
                 area_weights: torch.Tensor,
                 wet_mask: torch.Tensor,
                 mean_loss_weight: float = 1.0,
                 anomaly_loss_weight: float = 1.0,
                 total_loss_weight: float = 1.0,
                 anomaly_pre_weighting: callable = IdentityWeight(),
                 dynamic_weight_scaling: DynamicLevelScalingMultistep| None = None,
                 ):
        
        self.compute_area_mean = AreaWeightedMean(
            area_weights=area_weights,
            wet_mask=wet_mask
        )
        
        self.mean_loss_weight = mean_loss_weight
        self.anomaly_loss_weight = anomaly_loss_weight
        self.total_loss_weight = total_loss_weight
        
        self.anomaly_pre_weighting = anomaly_pre_weighting
        self.dynamic_weight_scaling = dynamic_weight_scaling
        
    def update_device(self, device_name):
        self.compute_area_mean.update_device(device_name)
        self.anomaly_pre_weighting.update_device(device_name)
        self.dynamic_weight_scaling.update_device(device_name)
        self.device_name = device_name               
    
    def __call__(self, label, pred_tuple, step):
        """
        Args:
            label (torch.Tensor): The ground truth tensor (B, C, H, W)
            pred_tuple (tuple): A tuple containing (global_mean_pred, anomaly_pred)
            step (int): The current time step index for weighting.
        """
        global_mean_pred, anomaly_pred = pred_tuple
        total_pred = global_mean_pred + anomaly_pred

        with torch.no_grad():
            true_global_mean = self.compute_area_mean(label)
            true_anomaly = label - true_global_mean
            

        loss_mean = (global_mean_pred.reshape(*global_mean_pred.shape,1,1)-true_global_mean.reshape(*true_global_mean.shape,1,1))**2
        loss_mean = self.dynamic_weight_scaling(loss_mean,step).mean()

        true_anomaly_pre = self.anomaly_pre_weighting(true_anomaly, step)
        anomaly_pred_pre = self.anomaly_pre_weighting(anomaly_pred, step)
        
        squared_diff_anomaly = (true_anomaly_pre - anomaly_pred_pre)**2
        weighted_squared_diff_anomaly = self.dynamic_weight_scaling(squared_diff_anomaly, step)
        
        loss_anomaly = weighted_squared_diff_anomaly.mean()
        
        true_pre = self.anomaly_pre_weighting(label, step)
        pred_pre = self.anomaly_pre_weighting(total_pred, step)

        squared_diff = (true_pre - pred_pre)**2
        weighted_squared_diff = self.dynamic_weight_scaling(squared_diff, step)
        
        loss = weighted_squared_diff.mean()

        total_loss = (self.mean_loss_weight * loss_mean) + (self.anomaly_loss_weight * loss_anomaly) + (self.total_loss_weight * loss) 
        return total_loss


class MSELoss():
    def __init__(self,
                 pre_weighting: callable = IdentityWeight(),
                 post_weighting: callable = IdentityWeight()):
        self.pre_weighting = pre_weighting
        self.post_weighting = post_weighting        
        
    def update_device(self, device_name):
        self.pre_weighting.update_device(device_name)
        self.post_weighting.update_device(device_name)
        self.device_name = device_name               
    
    def __call__(self,label,pred,step):
        label = self.pre_weighting(label,step)
        pred = self.pre_weighting(pred,step)
        squared_difference = self.post_weighting((label-pred)**2,step)
        return squared_difference.mean()

class STDLoss():
    def __init__(self,
                 pre_weighting: callable = IdentityWeight(),
                 post_weighting: callable = IdentityWeight()):
        self.pre_weighting = pre_weighting
        self.post_weighting = post_weighting        
        
    def update_device(self, device_name):
        self.pre_weighting.update_device(device_name)
        self.post_weighting.update_device(device_name)
        self.device_name = device_name               
    
    def __call__(self,label,pred,step):
        label = self.pre_weighting(label,step)
        pred = self.pre_weighting(pred,step)
        squared_difference = self.post_weighting((label.std(dim=-4)-pred.std(dim=-4))**2,step)
        return squared_difference.mean()
        
class KLLoss():
    def __init__(self,
                 lam = .0005):
        self.lam = lam        
        
    def update_device(self, device_name):
        self.device_name = device_name               
    
    def __call__(self,mu,logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss*self.lam

class AnomalyLoss():
    def __init__(self,
                 wet_transform,
                 area_weighting,
                 region_transform: callable = IdentityWeight(),
                 pre_weighting: callable = IdentityWeight(),
                 post_weighting: callable = IdentityWeight(),
                 return_sign = True,
                ):
        self.masked_area = region_transform(wet_transform(area_weighting.area_weight.squeeze()),0)
        self.level_rescaling = self.masked_area.sum(dim=(-1,-2))
        self.region_transform = region_transform
        self.pre_weighting = pre_weighting
        self.post_weighting = post_weighting      
        self.return_sign = return_sign
        
    def update_device(self, device_name):
        self.pre_weighting.update_device(device_name)
        self.region_transform.update_device(device_name)        
        self.post_weighting.update_device(device_name)
        self.masked_area = self.masked_area.to(device = device_name)
        self.level_rescaling = self.level_rescaling.to(device = device_name)
        self.device_name = device_name               
    
    def __call__(self,pred,clim,step,sign=None):
        anoms = self.region_transform(pred - clim,step)
        pred_mean = (self.pre_weighting(anoms,step)*self.masked_area).sum(dim=(-1,-2))
        anom_mean = (pred_mean/self.level_rescaling).sum(dim=(-1))
        new_sign = torch.sign(anom_mean)
        if sign is not None:
            anom_mean *= -sign
        else:
            anom_mean *= -new_sign 
        
        if self.return_sign:
            return self.post_weighting(anom_mean.mean(),step), new_sign
        else:
            return self.post_weighting(anom_mean.mean(),step)
        
def get_domain_fft(wet,steps = 500):
    region_wet = [0,0,0,0]
        
    for i in range(steps):
        Nx = wet.shape[1]
        Ny = wet.shape[0]

        i_init = np.random.randint(Nx)
        j_init = np.random.randint(Ny)

        i_min = i_init
        i_max = copy.copy(i_min)
        j_min = j_init
        j_max = copy.copy(j_min)

        again = True
        counter = 0
        while again and (counter<np.max([Nx,Ny])):
            again = False
            counter += 1
            i_min_temp = np.max([0,i_min-1])
            j_min_temp = np.max([0,j_min-1])
            i_max_temp = np.min([Nx-1,i_max+1])
            j_max_temp = np.min([Ny-1,j_max+1]) 

            if 0 not in wet[j_min:j_max,i_min_temp]:
                i_min = i_min_temp
                again = True
            if 0 not in wet[j_min:j_max,i_max_temp]:
                i_max = i_max_temp
                again = True        
            if 0 not in wet[j_min_temp,i_min:i_max]:
                j_min = j_min_temp    
                again = True        
            if 0 not in wet[j_max_temp,i_min:i_max]:
                j_max = j_max_temp     
                again = True
        if (i_max-i_min)*(j_max-j_min) > (region_wet[1]-region_wet[0])*(region_wet[3]-region_wet[2]):
            region_wet[0] = i_min
            region_wet[1] = i_max
            region_wet[2] = j_min
            region_wet[3] = j_max
    return region_wet


class SpectrumLoss():
    def __init__(self,
                 region_mask_path: str = '/pscratch/sd/a/asubel/Data/Chapter_2/Basin_Mask_Regions.zarr',
                 region: str = 'Pacific',
                 num_x_wavenumbers = 15,
                 num_y_wavenumbers = 10,
                 lam = .005,
                 post_weighting: callable = IdentityWeight(),                 
                ):
        masks = xr.open_zarr('/pscratch/sd/a/asubel/Data/Chapter_2/Basin_Mask_Regions.zarr').compute()
        region_fft = get_domain_fft(masks[region].fillna(0),200)
        self.lam = lam
        self.fft_shape = [int(np.floor((region_fft[3]-region_fft[2])/2))+1,
                          int(np.floor((region_fft[1]-region_fft[0])/2))+1]
        self.y_slice = slice(region_fft[2],region_fft[3])
        self.x_slice = slice(region_fft[0],region_fft[1])        
        self.num_x_wavenumbers = num_x_wavenumbers        
        self.num_y_wavenumbers = num_y_wavenumbers     
        self.fft_mask = torch.zeros(self.fft_shape)
        self.fft_mask[-self.num_y_wavenumbers] = 1.0
        self.fft_mask[:,-self.num_x_wavenumbers] = 1.0
        self.post_weighting = post_weighting
        
        
    def update_device(self, device_name):
        self.fft_mask = self.fft_mask.to(device = device_name)
        self.post_weighting.update_device(device_name)
        self.device_name = device_name               
    
    def __call__(self,label,pred,step):
        true_fft = torch.abs((torch.fft.rfft2(label[...,self.y_slice,self.x_slice],norm='forward'))**2)[...,:self.fft_shape[0],:]   
        pred_fft = torch.abs((torch.fft.rfft2(pred[...,self.y_slice,self.x_slice],norm='forward'))**2)[...,:self.fft_shape[0],:]  
        spectral_error = self.post_weighting(((torch.log(true_fft).mean(0)-torch.log(pred_fft).mean(0))**2)*self.fft_mask,step)
        return spectral_error.mean()

class ComposedLoss():
    def __init__(self,
                 losses: tuple):
        self.losses = losses
    
        
    def update_device(self, device_name):
        for loss in self.losses:
            loss.update_device(device_name)
        self.device_name = device_name               
    
    def __call__(self,label,pred,step):
        loss_value = 0
        for loss in self.losses:
            loss_value += loss(label,pred,step)
        return loss_value


class KernelPenalty():
    def __init__(self,
                 network,
                 integrated_variables_data,
                 boundary_variables_data,
                 label_variables_data,
                 weight: int = .1,
                ):
        
        self.network = network
        self.weight = weight
        self.conv_weights = []
        
        variable_sizes = integrated_variables_data.variable_size|boundary_variables_data.variable_size
        variable_sizes_output = label_variables_data.variable_size
        
        depth = np.array(list(variable_sizes.values())).max()
        input_size = integrated_variables_data.input_size + boundary_variables_data.input_size
        output_size = label_variables_data.input_size
        for param in self.network.parameters():
            param_shape = param.shape
            if len(param_shape) == 4:
                in_size = param_shape[-3]
                out_size = param_shape[-4]
                if in_size == input_size:
                    level_proxy_in = torch.zeros(input_size)
                    lev_start = 0
                    for levs in variable_sizes.values():
                        level_proxy_in[lev_start:lev_start+levs] = torch.arange(1,levs+1)
                        lev_start +=levs
                else:
                    level_proxy_in = torch.repeat_interleave(torch.arange(1,depth+1,dtype = torch.float),int(np.ceil(in_size/depth)))[:in_size]
                    
                if out_size == output_size:
                    level_proxy_out = torch.zeros(output_size)
                    lev_start = 0
                    for levs in variable_sizes_output.values():
                        level_proxy_out[lev_start:lev_start+levs] = torch.arange(1,levs+1)
                        lev_start +=levs
                else:
                    level_proxy_out = torch.repeat_interleave(torch.arange(1,depth+1,dtype = torch.float),int(np.ceil(out_size/depth)))[:out_size]
        
                level_proxy_out, level_proxy_in = torch.meshgrid(level_proxy_out,level_proxy_in)
                weights = torch.exp((torch.abs((level_proxy_in - level_proxy_out))))**0.5-1
                self.conv_weights.append(weights)
        
    def update_device(self, device_name):
        for i,weight in enumerate(self.conv_weights):
            self.conv_weights[i] = weight.to(device = device_name)
        self.device_name = device_name               
    
    def __call__(self,network):
        counter = 0
        loss = 0
        for param in network.parameters():
            if len(param.shape)==4:
                loss+= torch.abs(torch.abs(param).mean(dim = (-2,-1))*self.conv_weights[counter]).mean()
                counter+=1
        return loss*self.weight        