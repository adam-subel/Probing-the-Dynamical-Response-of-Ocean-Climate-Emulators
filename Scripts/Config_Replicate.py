import xarray as xr


import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import sys
sys.path.append("/pscratch/sd/a/asubel/Chapter_2/")

import Utils.networks as networks
import Utils.data_functions as data_functions
import torch.distributed as dist
import Utils.time_steppers as time_steppers
import Utils.transformations as transformations
import Utils.normalization_values as normalization_values
import Utils.parallel_multistep_dynamic as parallel
import Utils.losses as losses
import Utils.torch_functions as torch_functions

import os 
import random
import functools
import copy as copy


args = {}

DATA_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/CESM2_piControl_Detrended.zarr'
# DATA_PATH = '/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Processed_Detrended.zarr'


numerical_model = 'CM4_New'

n_train = 3500
n_val = 200

subsample = None
rand_seed = 16

torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)


  

# radiation is rsdt
integrated_variables = ['thetao','so']
label_variables = ['thetao','so']
# boundary_variables = ['tauuo','tauvo','hfds']
boundary_variables = ['hfds',]

include_radiation = True

data = xr.open_zarr(DATA_PATH,chunks = {})

if subsample != None:
    data = data.isel(time = slice(None,None,subsample))

data = data.sel(y = slice(-90,64))

data_train = data.isel(time = slice(None, n_train))
data_val = data.isel(time = slice(n_train, n_train + n_val))


if not(boundary_variables) and include_radiation:
    mean_radiation, std_radiation = normalization_values.get_normalization_values(model = numerical_model, variables = ['rsdt',])
    normalization_radiation = transformations.normalize(mean_radiation,std_radiation) 
                                                                                         
    boundary_values = data_functions.data_radiation(data_train,
                                                    transform = normalization_radiation)
    boundary_values_val = data_functions.data_radiation(data_val,
                                                    transform = normalization_radiation)    
elif boundary_variables:
    mean_boundary_vars, std_boundary_vars = normalization_values.get_normalization_values(model = numerical_model,
                                                                                          variables = boundary_variables)
    normalization_boundary_variables = transformations.normalize(mean_boundary_vars,std_boundary_vars) 
    wet_transformation_boundary = transformations.apply_wet_mask(data,boundary_variables)
    
    boundary_transformation = transformations.transformation([normalization_boundary_variables,wet_transformation_boundary])
    
    boundary_values = data_functions.data_from_single_source(data_train,
                                                             boundary_variables,
                                                             transform = boundary_transformation,
                                                             )
    boundary_values_val = data_functions.data_from_single_source(data_val,
                                                                 boundary_variables,
                                                                 transform = boundary_transformation,
                                                                 )
    if include_radiation:
        orbital_parameters_6ka = {
            'ecc': 0.018682,
            'obliquity': 24.105,
            'long_peri': 180.87,
            }                
        orbital_parameters_1850 = {
            'ecc': 0.016764,
            'obliquity': 23.459,
            'long_peri': 280.33,
            }              

        if DATA_PATH == '/pscratch/sd/a/asubel/Data/Chapter_2/CESM2_piControl_Detrended.zarr':
            orbital_parameters = orbital_parameters_1850
        elif DATA_PATH == '/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Processed_Detrended.zarr':
            orbital_parameters = orbital_parameters_6ka
            
        mean_radiation, std_radiation = normalization_values.get_normalization_values(model = numerical_model, variables = ['rsdt',])
        normalization_radiation = transformations.normalize(mean_radiation,std_radiation) 
        
        radiation_values = data_functions.data_radiation(data_train,
                                                        transform = normalization_radiation,
                                                        orb = orbital_parameters
                                                        )        
        radiation_values_val = data_functions.data_radiation(data_val,
                                                        transform = normalization_radiation,
                                                        orb = orbital_parameters)            
        boundary_values = data_functions.MergedDatasets((boundary_values,radiation_values))
        boundary_values_val = data_functions.MergedDatasets((boundary_values_val,radiation_values_val))
        
        
else:                                                             
    boundary_values = data_functions.data_from_single_source(data_train,
                                                               boundary_variables,
                                                              )                                                    
    boundary_values_val = data_functions.data_from_single_source(data_val,
                                                               boundary_variables,
                                                              )          



if integrated_variables:
    mean_integrated_vars, std_integrated_vars = normalization_values.get_normalization_values(model = numerical_model,
                                                                                              variables = integrated_variables)
    normalization_integrated_variables = transformations.normalize(mean_integrated_vars,std_integrated_vars)  
    wet_transformation_integrate = transformations.apply_wet_mask(data,integrated_variables)
    

    
    integrate_transformation = transformations.transformation([normalization_integrated_variables,wet_transformation_integrate])

    integrated_values_train = data_functions.data_from_single_source(data_train,
                                                               integrated_variables,
                                                               transform = integrate_transformation,
                                                              )
    integrated_values_val = data_functions.data_from_single_source(data_val,
                                                               integrated_variables,
                                                               transform = integrate_transformation,
                                                              )
else:
    integrated_values_train =  data_functions.data_from_single_source(data_train,
                                                               integrated_variables,
                                                              )    
    integrated_values_val =  data_functions.data_from_single_source(data_val,
                                                               integrated_variables,
                                                              )    
    
mean_label_vars, std_label_vars = normalization_values.get_normalization_values(model = numerical_model,
                                                                                variables = label_variables)
normalization_label_variables = transformations.normalize(mean_label_vars,std_label_vars)  
wet_transformation_label = transformations.apply_wet_mask(data,label_variables)
label_transformation = transformations.transformation([normalization_label_variables,wet_transformation_label])    
 
    
label_values_train = data_functions.data_from_single_source(data_train,
                                                            label_variables,
                                                            transform = label_transformation,
                                                            label = True,
                                                                 )
label_values_val = data_functions.data_from_single_source(data_val,
                                                          label_variables,
                                                          transform = label_transformation,
                                                          label = True,
                                                                 )

# of form [integrated, boundary, label]

args['train_datasets'] = data_functions.CombinedDataset(integrated_values_train,boundary_values,label_values_train)
args['val_datasets'] = data_functions.CombinedDataset(integrated_values_val,boundary_values_val,label_values_val)


os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = str(np.random.randint(1600,1800)) 

if boundary_values and integrated_values_train:
    input_shape = (integrated_values_train.input_size + boundary_values.input_size,) + integrated_values_train.input_shape[1:]
elif boundary_values:
    input_shape = boundary_values.input_shape
elif integrated_values_train:
    input_shape = integrated_values_train.input_shape
else:
    raise ValueError('No inputs selected for emulator') 

    
# learned_features = networks.LearnedFeatures(input_shape[1:],4)
learned_features = None

if learned_features:
    input_shape = (input_shape[0] + learned_features.num_features, ) + input_shape[1:]
    
    
downsampler = functools.partial(networks.PoolLayer,layer = nn.AvgPool2d, resize_ratio = (2,2))




propogater = networks.UNext(input_shape = input_shape,
                output_size = label_values_train.input_size,
                hidden_node_sizes = (200, 250, 300, 400),
                hidden_node_ratio = 4,
                dilations = (1, 2, 4, 8, 8),
                activate_final = True,
                residual_final = True,
                final_conv_layer = True,
                batch_norm = True,
                batch_norm_function = networks.DynamicTanhScaled,
                downsampler = downsampler,
                checkpoints = True,
                surya_blocks = True,
                       )



if learned_features:
    network = networks.FeatureWrapper(network=propogater,features=learned_features)
else:
    network = propogater
    
network =  nn.SyncBatchNorm.convert_sync_batchnorm(network)





args['network'] = network

mean_dt, std_dt = normalization_values.get_dt_normalization_values(model = numerical_model,
                                                                    variables = label_variables)

wet_transformation = transformations.apply_wet_mask(data,label_variables)
normalization_dt = transformations.inv_normalize(mean_dt,std_dt)  
dt_transformation = transformations.transformation([normalization_dt, wet_transformation])


forward_model = time_steppers.DirectStep(network = network, dt = 1, transform = wet_transformation)


args['forward_model'] = forward_model


args["save_model"] = True
args["World_Size"] = int(torch.cuda.device_count())


args["epochs"] = 75
args["batch_size"] = 6
args['lr'] = 2e-4
args['lr_decay_rate'] = .96

args['steps'] = [4,8]
args['step_transitions'] = [15,None,]
args['train_datasets'].set_max_steps(args['steps'][-1])


args['num_workers'] = 2

area_weighting = losses.AreaWeight(data,np.sqrt)

mean_weighting = losses.StepWeightArray([.025,.025,.025,.025,.025,.025,.025,.025,.025,.05,.05,.05,.05,.05,.05,.05,.05])

# std_dt[38:] = std_dt[38:]/100


dynamic_level_scaling = losses.DynamicLevelScalingMultistep(label_values=label_values_train,
                                        post_weighting=area_weighting,
                                        inv_max_vals=std_dt,
                                        max_steps = args['steps'][-1], 
                                        relaxation_period = 250,
                                        world_size = args["World_Size"],
                                        max_ratio = 10000,                   
                                        )




mse_loss = losses.MSELoss(post_weighting=dynamic_level_scaling)

mean_loss = losses.MeanLoss(wet_transform = wet_transformation,
                            area_weighting=area_weighting_true,
                            post_weighting=mean_weighting)

args['loss'] = mse_loss
args['network_loss'] = None

args['dynamic_weighting'] = dynamic_level_scaling


class OneCycleLRIgnoreEpoch(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        # The script passes a float (continuous epoch), but OneCycleLR expects
        # integer steps or None (to auto-increment). We ignore the argument 
        # so it behaves like step() and increments its own counter.
        super().step(epoch=None)

# 2. Calculate steps_per_epoch explicitly
# DistributedSampler pads data to be divisible by World_Size, then batched.
samples_per_gpu = np.ceil(n_train / args["World_Size"])
steps_per_epoch = int(np.ceil(samples_per_gpu / args["batch_size"]))

args['scheduler'] = functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                                      T_max=args['epochs'])

# args['scheduler'] = functools.partial(
#     OneCycleLRIgnoreEpoch,
#     max_lr=args['lr'],
#     epochs=args['epochs'],
#     steps_per_epoch=steps_per_epoch,
#     pct_start=0.2,     # (Optional) Fraction of cycle spent warming up
#     div_factor=25.0,   # (Optional) Initial LR = max_lr / div_factor
#     final_div_factor=1e4 # (Optional) Final LR = max_lr / (div_factor * final_div_factor)
# )

args['optimizer'] = functools.partial(torch.optim.Adam, lr=args['lr'], amsgrad=True)

    


area_weighting = losses.AreaWeight(data,np.sqrt)
level_weighting_upper = losses.LevelWeight([1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 0.,
                                            0., 0., 0., 0., 0.,
                                            0., 0., 0.,0.,],
                                           label_values_train)
level_weighting_mid = losses.LevelWeight([0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 1., 1., 1.,
                                          1., 1., 0., 0., 0., 0.,0.,],
                                         label_values_train)
level_weighting_lower = losses.LevelWeight([0., 0., 0., 0., 0., 0.,
                                            0., 0., 0., 0., 0., 0.,
                                            0., 0., 1., 1., 1.,1.,1.,],
                                           label_values_train)



mse_upper = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,level_weighting_upper)))
mse_mid = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,level_weighting_mid)))
mse_lower = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,level_weighting_lower)))
mse_total = losses.MSELoss(post_weighting=area_weighting)


select_thetao = losses.SelectVariables(('thetao',),label_values_train)
select_so = losses.SelectVariables(('so',),label_values_train)
select_uo = losses.SelectVariables(('uo',),label_values_train)
select_vo = losses.SelectVariables(('vo',),label_values_train)


mse_thetao = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,select_thetao)))
mse_so = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,select_so)))
mse_uo = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,select_uo)))
mse_vo = losses.MSELoss(post_weighting=losses.ComposedWeight((area_weighting,select_vo)))

args['evaluations'] = {'mse_upper':mse_upper,
                       'mse_mid':mse_mid,
                       'mse_lower':mse_lower,
                       'mse_total':mse_total,
                       'mse_thetao': mse_thetao,          
                       'mse_so': mse_so,               
                       # 'mse_uo': mse_uo,                                      
                       # 'mse_vo': mse_vo,                                      
                      }

args['times_to_evaluate'] = (1,8,20,40)

args['val_datasets'].set_max_steps(args['times_to_evaluate'][-1])
args['val_datasets'].set_step(args['times_to_evaluate'][-1])


args['experiment_name'] = 'Truly_Replicate_CESM2_Baseline_HFDS_Exact_Correct'


args['config_path'] = str(os.path.abspath(__file__))

args['restart_path'] = None

if __name__ == '__main__':
    torch.multiprocessing.spawn(parallel.train_network, nprocs=args["World_Size"], args=(args,))   
