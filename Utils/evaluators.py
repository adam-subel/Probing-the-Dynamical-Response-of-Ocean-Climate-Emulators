import wandb
import torch
import sys
sys.path.append("../")

import torch.distributed as dist
import Utils.data_functions as data_functions
import Utils.time_steppers as time_steppers


class ModelEval():
    def __init__(self,
                 evaluations: dict[str, callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self,
                 forward_model,
                 step,
                 global_rank,
                 additional_metrics = None):
        evaluation_metrics = {}
        
        device = forward_model.device_name
        forward_model.network.eval() 
        for k in self.evaluations.keys():
            for eval_step in self.times_to_evaluate:
                evaluation_metrics[k + '_step_'+ str(eval_step)] = torch.tensor(0.0,device = device)
        
        if self.training_step%self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        if i == 0:
                            out = forward_model.step_forward(integrated[:,0],boundary[:,0])
                        else:
                            out = forward_model.step_forward(out,boundary[:,i])       

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                metric = self.evaluations[k](out,labels[:,i].to(device=device),step)/self.num_batches
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric
            for k,v in evaluation_metrics.items():
                torch.distributed.all_reduce(v,op= dist.ReduceOp.SUM)
                evaluation_metrics[k] = v/self.world_size
                
            evaluation_metrics['recurrent_steps'] = step
            
            if global_rank == 0:
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics|additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 

        self.training_step += 1

class ModelEvalAutoEncoder():
    def __init__(self,
                 evaluations: dict[str, callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self,
                 forward_model,
                 step,
                 global_rank,
                 additional_metrics = None):
        
        evaluation_metrics = {}
        device = forward_model.device_name
        forward_model.network.eval() 

        # Fix 1: Initialize as Tensors on device, not floats
        for k in self.evaluations.keys():
            for eval_step in self.times_to_evaluate:
                evaluation_metrics[k + '_step_'+ str(eval_step)] = torch.tensor(0.0, device=device)
        
        if self.training_step % self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        # Fix 2: Unpack tuple (out, mu, logvar)
                        # Indexing Check: integrated[:,0] is correct for [batch, step, ...]
                        if i == 0:
                            out, _, _ = forward_model.step_forward(integrated[:,0], boundary[:,0])
                        else:
                            out, _, _ = forward_model.step_forward(out, boundary[:,i])       

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                # Accumulate locally first
                                metric = self.evaluations[k](out, labels[:,i].to(device=device), step) / self.num_batches
                                evaluation_metrics[k+ '_step_'+ str(i+1)] += metric

            # Fix 3: Global reduction happens ONCE after all batches are processed
            for k, v in evaluation_metrics.items():
                torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
                # Normalize by world_size to get the mean across GPUs
                evaluation_metrics[k] = v / self.world_size

            # Move to CPU for logging
            if global_rank == 0:
                evaluation_metrics['recurrent_steps'] = step
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics | additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 

        self.training_step += 1

class ModelEvalObserved():
    def __init__(self,
                 evaluations: dict[str, callable],
                 evaluations_obs: dict[str: callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.evaluations_obs = evaluations_obs        
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self, forward_model,
                 observer,
                 step,
                 global_rank,
                 additional_metrics = None):
        evaluation_metrics = {}
        evaluation_metrics['recurrent_steps'] = step

        device = forward_model.device_name
        forward_model.network.eval() 
        observer.network.eval()
        for step in self.times_to_evaluate:
            for k in self.evaluations.keys():
                evaluation_metrics[k + '_step_'+ str(step)] = 0.0
            for k in self.evaluations_obs.keys():
                evaluation_metrics[k + '_step_'+ str(step)] = 0.0

        if self.training_step%self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels, observations in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        if i == 0:
                            out = forward_model.step_forward(integrated[:,0],boundary[:,0])
                            out, observed_out =  observer.observe(out)
                        else:
                            out = forward_model.step_forward(out,boundary[:,i])  
                            out, observed_out =  observer.observe(out)                            

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                metric = self.evaluations[k](out,labels[i].to(device=device),step)/self.num_batches
                                torch.distributed.all_reduce(metric/self.world_size,op= dist.ReduceOp.SUM)
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric
                            for k in self.evaluations_obs.keys():
                                metric = self.evaluations_obs[k](observed_out,observations[i].to(device=device),step)/self.num_batches
                                torch.distributed.all_reduce(metric/self.world_size,op= dist.ReduceOp.SUM)
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric                                    
            if global_rank == 0:
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics|additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 
        observer.network.train()

        self.training_step += 1

class ModelEvalObservedPassive():
    def __init__(self,
                 evaluations: dict[str, callable],
                 evaluations_obs: dict[str: callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.evaluations_obs = evaluations_obs        
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self, forward_model,
                 observer,
                 step,
                 global_rank,
                 additional_metrics = None):
        evaluation_metrics = {}
        evaluation_metrics['recurrent_steps'] = step

        device = forward_model.device_name
        forward_model.network.eval() 
        observer.network.eval()
        for step in self.times_to_evaluate:
            for k in self.evaluations.keys():
                evaluation_metrics[k + '_step_'+ str(step)] = 0.0
            for k in self.evaluations_obs.keys():
                evaluation_metrics[k + '_step_'+ str(step)] = 0.0

        if self.training_step%self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels, observations in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        if i == 0:
                            out = forward_model.step_forward(integrated[:,0],boundary[:,0])
                            _, observed_out,_,_ =  observer.observe(torch.cat((out,boundary[:,i,-1:].to(device = device)),dim=-3))
                        else:
                            out = forward_model.step_forward(out,boundary[:,i])  
                            _, observed_out,_,_ =  observer.observe(torch.cat((out,boundary[:,i,-1:].to(device = device)),dim=-3))                            

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                metric = self.evaluations[k](out,labels[:,i].to(device=device),step)/self.num_batches
                                torch.distributed.all_reduce(metric/self.world_size,op= dist.ReduceOp.SUM)
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric
                            for k in self.evaluations_obs.keys():
                                metric = self.evaluations_obs[k](observed_out,observations[:,i].to(device=device),step)/self.num_batches
                                torch.distributed.all_reduce(metric/self.world_size,op= dist.ReduceOp.SUM)
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric                                    
            if global_rank == 0:
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics|additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 
        observer.network.train()

        self.training_step += 1
        
        
class ModelEvalForced():
    def __init__(self,
                 evaluations: dict[str, callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self,
                 forward_model,
                 step,
                 global_rank,
                 additional_metrics = None):
        evaluation_metrics = {}
        evaluation_metrics['recurrent_steps'] = step
        
        device = forward_model.device_name
        forward_model.network.eval() 
        for k in self.evaluations.keys():
            for step in self.times_to_evaluate:
                evaluation_metrics[k + '_step_'+ str(step)] = 0.0
        
        if self.training_step%self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        if i == 0:
                            out, dynamic_out = forward_model.step_forward(integrated[0],boundary[0])
                        else:
                            out, dynamic_out = forward_model.step_forward(out,boundary[i])       

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                metric = self.evaluations[k](out,labels[i].to(device=device),step)/self.num_batches
                                torch.distributed.all_reduce(metric/self.world_size,op= dist.ReduceOp.SUM)
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric
            if global_rank == 0:
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics|additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 

        self.training_step += 1
        
class ModelEvalForcedTendencies():
    def __init__(self,
                 evaluations: dict[str, callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self,
                 forward_model,
                 step,
                 global_rank,
                 additional_metrics = None):
        evaluation_metrics = {}
        evaluation_metrics['recurrent_steps'] = step
        
        device = forward_model.device_name
        forward_model.network.eval() 
        for k in self.evaluations.keys():
            for step in self.times_to_evaluate:
                evaluation_metrics[k + '_step_'+ str(step)] = 0.0
        
        if self.training_step%self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        if i == 0:
                            out, _, _ = forward_model.step_forward(integrated[0],boundary[0])
                        else:
                            out, _, _ = forward_model.step_forward(out,boundary[i])       

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                metric = self.evaluations[k](out,labels[i].to(device=device),step)/self.num_batches
                                torch.distributed.all_reduce(metric/self.world_size,op= dist.ReduceOp.SUM)
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric
            if global_rank == 0:
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics|additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 

        self.training_step += 1        


class ModelEvalNone():
    def __init__(self,
                 evaluations: dict[str, callable],
                 times_to_evaluate: tuple[int,...],
                 val_loader: data_functions.CombinedDataset,
                 world_size: int,
                 *,
                 num_batches: int = 2,
                 steps_between_evaluation: int = 250, 
                 ):
        self.evaluations = evaluations
        self.times_to_evaluate = times_to_evaluate
        self.num_batches = num_batches
        self.steps_between_evaluations = steps_between_evaluation
        self.world_size = world_size
        self.training_step = 0
        self.val_loader = val_loader

    def __call__(self,
                 forward_model,
                 step,
                 global_rank,
                 additional_metrics = None):
        evaluation_metrics = {}
        
        device = forward_model.device_name
        forward_model.network.eval() 
        for k in self.evaluations.keys():
            for eval_step in self.times_to_evaluate:
                evaluation_metrics[k + '_step_'+ str(eval_step)] = torch.tensor(0.0,device = device)
        
        if self.training_step%self.steps_between_evaluations == 0:
            batch = 0
            for integrated, boundary, labels in self.val_loader:
                if batch >= self.num_batches:
                    break
                else:
                    batch += 1
                with torch.no_grad():
                    for i in range(self.times_to_evaluate[-1]):
                        if i == 0:
                            out = forward_model.step_forward(integrated[0],boundary[:,0])
                        else:
                            out = forward_model.step_forward(out,boundary[:,i])       

                        if i+1 in self.times_to_evaluate:
                            for k in self.evaluations.keys():
                                metric = self.evaluations[k](out,labels[:,i].to(device=device),step)/self.num_batches
                                if global_rank == 0:
                                    evaluation_metrics[k+ '_step_'+ str(i+1)] += metric
            for k,v in evaluation_metrics.items():
                torch.distributed.all_reduce(v,op= dist.ReduceOp.SUM)
                evaluation_metrics[k] = v/self.world_size
                
            evaluation_metrics['recurrent_steps'] = step
            
            if global_rank == 0:
                evaluation_metrics['training_step'] = self.training_step 
                if additional_metrics:
                    wandb.log(evaluation_metrics|additional_metrics)   
                else:
                    wandb.log(evaluation_metrics)   
                    
        forward_model.network.train() 

        self.training_step += 1