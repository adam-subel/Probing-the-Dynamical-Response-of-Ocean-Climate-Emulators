import torch
import torch.nn as nn
import numpy as np
import os
import wandb

import Utils.time_steppers as time_steppers
import Utils.evaluators as evaluators

from Utils.collate import default_collate

import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from datetime import date
import csv


def train_step(forward_model,
               train_loader,
               loss_fn,
               network_loss,
               optimizer,
               scheduler,
               step,
               device,
               epoch,
               evaluator,
               global_rank,
               dynamic_weighting
               ):
    
    forward_model.network.train()
    iters_in_epoch = len(train_loader)
    for j, (integrated, boundary, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outs = forward_model.step_forward(integrated[:,0],boundary[:,0])        
        loss = loss_fn(labels[:,0].to(device), outs, 0)
        if dynamic_weighting:
            dynamic_weighting.update_new_weights(labels[:,0].to(device=device).detach(),outs.detach(),0)
        for i in range(1,step):
            outs = forward_model.step_forward(outs,boundary[:,i])
            loss += loss_fn(labels[:,i].to(device), outs, i)
            if dynamic_weighting:
                dynamic_weighting.update_new_weights(labels[:,i].to(device=device).detach(),
                                                 outs.detach(),i)            

        if network_loss is not None:
            loss_network = network_loss(forward_model.network)
            loss += loss_network
        loss.backward()

            
        torch.nn.utils.clip_grad_norm_(forward_model.network.parameters(), 1.0)
            

        optimizer.step()
        if dynamic_weighting:
                dynamic_weighting.apply_new_weights()

        loss_info = {'lr':scheduler.get_last_lr()[-1],
                  'loss':loss.detach().cpu(),
                  'dynamic_weights_0':dynamic_weighting.level_weights[0].cpu(),

                  }
        if network_loss is not None:
            loss_info = loss_info | {'loss_network':loss_network.detach().cpu()}
            
        evaluator(forward_model,
                  step,
                  global_rank,
                  loss_info
                  )
        
        scheduler.step(epoch + j/ iters_in_epoch)
    
    return loss           


def train_network(local_rank,args):

    global_rank = local_rank*1
    if args["World_Size"] > 2:
        dist.init_process_group(backend='nccl',world_size=args["World_Size"], rank=global_rank)
    else:
        dist.init_process_group(backend='gloo',world_size=args["World_Size"], rank=global_rank)
    
    device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
    device_name = "cuda:" + str(local_rank)
    
    dynamic_weighting = args['dynamic_weighting']
    if dynamic_weighting is not None:
        dynamic_weighting.update_device(device_name)
    
    if args['restart_path']:
        restart_config = torch.load(args['restart_path'],
                                    map_location=device_name,
                                    weights_only=True)
        forward_model = args['forward_model']
        forward_model.update_device(device_name)
        forward_model.network.load_state_dict(restart_config['network_state'])
        
        optimizer = args['optimizer'](forward_model.network.parameters()) 
        scheduler = args['scheduler'](optimizer)
        
        optimizer.load_state_dict(restart_config['optimizer_state'])
        scheduler.load_state_dict(restart_config['scheduler_state'])
        
        epoch = restart_config['epoch']
        step_number = restart_config['step_number']
        if args.get('override_experiment_folder',False):
            experiment_folder = args['experiment_name'] + '_' + str(date.today())
        else:
            experiment_folder = restart_config['experiment_folder']
        training_step = restart_config['training_step']
        if dynamic_weighting:
            dynamic_weighting.level_weights = restart_config['dynamic_weights']
   
    else:
        forward_model = args['forward_model']
        forward_model.update_device(device_name)

        experiment_folder = args['experiment_name'] + '_' + str(date.today())

        epoch = 0 
        step_number = 0
        
        optimizer = args['optimizer'](forward_model.network.parameters()) 
        scheduler = args['scheduler'](optimizer)
        training_step = None

    train_data = args['train_datasets']
    val_data = args['val_datasets']    
    
    step = args['steps'][step_number]
    
    train_data.set_step(step)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    args['train_datasets'],
    num_replicas=args["World_Size"],
    rank=global_rank
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
    args['val_datasets'],
    num_replicas=args["World_Size"],
    rank=global_rank
    )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args["batch_size"],
                                               num_workers = args['num_workers'],
                                               sampler=train_sampler,
                                              pin_memory = True, collate_fn=default_collate)
    
    
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=min(args["batch_size"],1),
                                              num_workers = args['num_workers'],
                                             sampler=test_sampler,
                                              pin_memory = True, collate_fn=default_collate)
    
    evaluations = args['evaluations']
    for v in evaluations.values():
        v.update_device(device_name)    
    evaluator = evaluators.ModelEval(evaluations,args['times_to_evaluate'],val_loader,args["World_Size"],                 steps_between_evaluation = 150, 
)
    if training_step:
        evaluator.training_step = training_step
        
    loss = args['loss']
    loss.update_device(device_name)
    network_loss = args['network_loss']
    if args['network_loss'] is not None:
        network_loss.update_device(device_name)
    
    base_folder_networks = '/pscratch/sd/a/asubel/Chapter_2/Networks/'
    experiment_folder_path = base_folder_networks + experiment_folder
    if not os.path.exists(experiment_folder_path):
        print('folders_created at path:' + experiment_folder_path )
        os.makedirs(experiment_folder_path)
        os.makedirs(experiment_folder_path+ '/Checkpoints')
        os.system('cp ' + args['config_path'] + ' ' + experiment_folder_path + '/Config.py')
        header = [['epoch', 'train_loss', 'val_loss']]
        with open(experiment_folder_path+'/Losses.csv',mode = 'w',newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(header)    
    if global_rank == 0:
        wandb.init(project = 'Recreate_Chapter2',
                   config={'experiment_name': args['experiment_name']},
                   id = experiment_folder,
                   resume = "allow",
                  )
        wandb.save(args['config_path'])
        
    
    for epoch in range(epoch, args["epochs"]):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        if epoch == args['step_transitions'][step_number]:
            step_number += 1
            step = args['steps'][step_number]
            train_data.set_step(step)
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args["batch_size"],
                                                       num_workers = args['num_workers'],
                                                       sampler=train_sampler,
                                                       pin_memory = True,
                                                       collate_fn=default_collate)
            
        loss_value = train_step(forward_model,
                                train_loader,
                                loss,
                                network_loss,
                                optimizer,
                                scheduler,
                                step,
                                device, 
                                epoch + 1, 
                                evaluator,
                                global_rank,
                                dynamic_weighting,
                               )                




        if global_rank == 0 and epoch%1 == 0:
            # torch.save(forward_model.network.state_dict(),experiment_folder_path+ '/Checkpoints/Model_Weights_epoch_' +str(epoch+1) + '.pt')        
            checkpoint_dict = {'network_state': forward_model.network.state_dict(),
                        'optimizer_state':optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch': epoch+1, 
                        'step_number': step_number,
                        'experiment_folder': experiment_folder,
                        'training_step': evaluator.training_step,
                       }
            if dynamic_weighting:
                checkpoint_dict = checkpoint_dict | {'dynamic_weights':dynamic_weighting.level_weights}
            torch.save(checkpoint_dict,
                        experiment_folder_path+ '/Checkpoints/Checkpoint_epoch_' +str(epoch+1) + '.tar')   
            del checkpoint_dict
        torch.distributed.barrier() 
            
    if global_rank ==0 and args["save_model"]:
        torch.save(forward_model.network.state_dict(),experiment_folder_path+ '/Checkpoints/Model_Weights_final.pt') 
        torch.save({'network_state': forward_model.network.state_dict(),
                    'optimizer_state':optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': epoch+1, 
                    'step_number': step_number,
                    'experiment_folder': experiment_folder,
                    'training_step': evaluator.training_step,
                   },
                    experiment_folder_path+ '/Checkpoints/Checkpoint_final.tar') 