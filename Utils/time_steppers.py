import torch
import functools
import abc
import sys
sys.path.append("../")

import Utils.transformations as transformations


class TimeStepper(abc.ABC):
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        self.network = network
        self.dt = dt
        self.transform = transform
        self.state_transform = state_transform
        self.boundary_transform = boundary_transform
        self.device_name = device_name

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.device_name = device_name
        
    @abc.abstractmethod
    def step_forward(self, state, boundary):
        return 
    
    
class DirectStep(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(DirectStep,self).__init__(network,
                                             dt,
                                             transform,
                                             state_transform,
                                             boundary_transform)
        
    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
        input_state.to(device=self.device_name)
        return self.transform(self.network(input_state))

class DirectStepLocalScaling(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 local_scaling_transform: callable,
                 local_scaling_stds: torch.tensor,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(DirectStepLocalScaling,self).__init__(network,
                                             dt,
                                             transform,
                                             state_transform,
                                             boundary_transform)
        self.local_scaling_transform = local_scaling_transform
        self.local_scaling_stds = local_scaling_stds

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.local_scaling_transform.update_device(device_name)
        self.local_scaling_stds = self.local_scaling_stds.to(device = device_name)        
        self.device_name = device_name
        
    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        state, local_mean = self.local_scaling_transform(state)
        state = state/self.local_scaling_stds
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
        input_state.to(device=self.device_name)
        return self.transform(self.network(input_state)*self.local_scaling_stds+local_mean) 


class DirectStepPhysical(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(DirectStepPhysical,self).__init__(network,
                                             dt,
                                             transform,
                                             state_transform,
                                             boundary_transform)
        
    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = self.state_transform(state.to(device = self.device_name))   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
        else:
            input_state = state
        return self.transform(self.network(state, boundary))

# class EulerStep(TimeStepper):
    
#     def __init__(self,
#                  network: torch.nn.Module,
#                  dt: int,
#                  transform: callable = transformations.NullTransform(),
#                  state_transform: callable = transformations.NullTransform(),
#                  boundary_transform: callable = transformations.NullTransform(),
#                  device_name: str = 'cpu',
#                 ):
#         super(EulerStep,self).__init__(network,
#                                              dt,
#                                              transform,
#                                              state_transform,
#                                              boundary_transform,
#                                              device_name)
        
#     def step_forward(self, state, boundary = None):
#         if len(state.shape) == 3:
#             state = state.unsqueeze(dim = 0)
#         state = state.to(device = self.device_name)   
#         if boundary is not None:
#             if len(boundary.shape) == 3:
#                 boundary = boundary.unsqueeze(dim = 0)
#             boundary = boundary.to(device=self.device_name)            
#             boundary = self.boundary_transform(boundary)
#             input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
#         else:
#             input_state = self.state_transform(state)
            
#         input_state.to(device=self.device_name)
#         return state.squeeze(dim = 0) + self.dt * self.transform(self.network(input_state))

class EulerStep(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 prev_state_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(EulerStep,self).__init__(network,
                                             dt,
                                             transform,
                                             state_transform,
                                             boundary_transform,
                                             device_name)
        self.prev_state_transform = prev_state_transform

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.prev_state_transform.update_device(device_name)
        self.device_name = device_name
        
    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
            
        input_state.to(device=self.device_name)
        return self.prev_state_transform(state).squeeze(dim = 0) + self.dt * self.transform(self.network(input_state))

class EulerStepNoise(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(EulerStepNoise,self).__init__(network,
                                             dt,
                                             transform,
                                             state_transform,
                                             boundary_transform,
                                             device_name)
        
    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = self.state_transform(state.to(device = self.device_name))   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((state,boundary), dim = 1)
        else:
            input_state = state
            
        input_state.to(device=self.device_name)
        return state.squeeze(dim = 0) + self.dt * self.transform(self.network(input_state))

class InvertStep(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(InvertStep,self).__init__(network,
                                        dt,
                                        transform,
                                        state_transform,
                                        boundary_transform,
                                        device_name
                                       )
        
    def step_forward(self, state, boundary):
        del state
        if len(boundary.shape) == 3:
            boundary = boundary.unsqueeze(dim = 0)
        boundary = boundary.to(device=self.device_name)            
        boundary = self.boundary_transform(boundary)
        return self.transform(self.network(boundary))

class AutoEncodeStep(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(AutoEncodeStep,self).__init__(network,
                                        dt,
                                        transform,
                                        state_transform,
                                        boundary_transform,
                                        device_name
                                       )
        
    def step_forward(self, state, boundary):
        del state
        if len(boundary.shape) == 3:
            boundary = boundary.unsqueeze(dim = 0)
        boundary = boundary.to(device=self.device_name)            
        boundary = self.boundary_transform(boundary)
        output, mu, logvar = self.network(boundary)
        return self.transform(output), mu, logvar

class AutoEncodeStepBoundary(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(AutoEncodeStepBoundary,self).__init__(network,
                                        dt,
                                        transform,
                                        state_transform,
                                        boundary_transform,
                                        device_name
                                       )
        
    def step_forward(self, state, boundary):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
            
        output, mu, logvar = self.network(input_state)
        return self.transform(output), mu, logvar

class AutoEncodeInvertStep(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(AutoEncodeInvertStep,self).__init__(network,
                                        dt,
                                        transform,
                                        state_transform,
                                        boundary_transform,
                                        device_name
                                       )
        
    def step_forward(self, state, boundary):
        del state
        if len(boundary.shape) == 3:
            boundary = boundary.unsqueeze(dim = 0)
        boundary = boundary.to(device=self.device_name)            
        boundary = self.boundary_transform(boundary)
        output, mu, logvar = self.network(boundary)
        return self.transform(output), mu, logvar

class EulerStepTendency(TimeStepper):
    
    def __init__(self,                 
                 network: torch.nn.Module,
                 tendency_network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 forcing_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
               ):
        super(EulerStepTendency,self).__init__(network,
                                               dt,
                                               transform,
                                               state_transform,
                                               boundary_transform,
                                               device_name,
                                              )
        self.forcing_transform = forcing_transform    
        self.tendency_network = tendency_network
        
    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.tendency_network = self.tendency_network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.forcing_transform.update_device(device_name)        
        self.device_name = device_name
        
    def step_forward(self, state, boundary = None, forcing = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        
        if len(boundary.shape) == 3:
            boundary = boundary.unsqueeze(dim = 0)
        boundary = boundary.to(device=self.device_name)            

        state_contribution = self.transform(self.network(self.state_transform(state)))
        boundary_contribution = self.forcing_transform(self.tendency_network(self.boundary_transform(boundary)))
        output_dynamics = state.squeeze(dim = 0) + self.dt * state_contribution
        output_forced = output_dynamics + self.dt * boundary_contribution
        return  output_forced, output_dynamics
    
class EulerStepOutputTendency(TimeStepper):
    
    def __init__(self,                 
                 network: torch.nn.Module,
                 tendency_network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 forcing_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
               ):
        super(EulerStepOutputTendency,self).__init__(network,
                                                     dt,
                                                     transform,
                                                     state_transform,
                                                     boundary_transform,
                                                     device_name,
                                                    )
        self.forcing_transform = forcing_transform    
        self.tendency_network = tendency_network
        
    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.tendency_network = self.tendency_network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.forcing_transform.update_device(device_name)        
        self.device_name = device_name
        
    def step_forward(self, state, boundary = None, forcing = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        
        if len(boundary.shape) == 3:
            boundary = boundary.unsqueeze(dim = 0)
        boundary = boundary.to(device=self.device_name)            

        state_contribution = self.transform(self.network(self.state_transform(state)))
        boundary_contribution = self.forcing_transform(self.tendency_network(self.boundary_transform(boundary)))
        next_state = state.squeeze(dim = 0) + self.dt * state_contribution + self.dt * boundary_contribution
        return next_state, state_contribution, boundary_contribution


class EulerStepCorrected(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 ae_network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 ae_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(EulerStepCorrected,self).__init__(network,
                                                dt,
                                                transform,
                                                state_transform,
                                                boundary_transform,
                                                device_name,
                                               )
        self.ae_network = ae_network
        self.ae_transform = ae_transform    

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.ae_network = self.ae_network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.ae_transform.update_device(device_name)  
        self.device_name = device_name
    
    def step_forward(self, state, boundary = None):

        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
            
        
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
            
        ae_state = self.ae_transform(self.ae_network(state))        
            
        input_state.to(device=self.device_name)
        tendency = self.dt * self.transform(self.network(input_state))
        next_state = ae_state.squeeze(dim = 0) + tendency
        return next_state     

        
class EulerStepCorrectedOutputTendency(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 corrector_network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 corrector_transform: callable = transformations.NullTransform(),
                 corrector_boundary_transform: callable = transformations.NullTransform(),
                 corrector_include_boundary: bool = False,
                 device_name: str = 'cpu',
                ):
        super(EulerStepCorrectedOutputTendency,self).__init__(network,
                                                dt,
                                                transform,
                                                state_transform,
                                                boundary_transform,
                                                device_name,
                                               )
        self.corrector_network = corrector_network
        self.corrector_transform = corrector_transform
        self.corrector_boundary_transform = corrector_boundary_transform
        self.corrector_include_boundary = corrector_include_boundary

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.corrector_network = self.corrector_network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.corrector_transform.update_device(device_name)   
        self.corrector_boundary_transform.update_device(device_name)                
        self.device_name = device_name
    
    def step_forward(self, state, boundary = None):
         
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
            
            
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)

        if self.corrector_include_boundary:
            corrector_input = torch.cat((state,self.corrector_boundary_transform(boundary)), dim = 1)
            state_corrected = self.corrector_transform(self.corrector_network(corrector_input))                
        else:
            state_corrected = self.corrector_transform(self.corrector_network(state))        
            
        input_state.to(device=self.device_name)
        tendency = self.dt * self.transform(self.network(input_state))
        next_state = state_corrected.squeeze(dim = 0) + tendency
        return next_state, state_corrected, tendency 
    

class DirectStepGlobalMean(TimeStepper):
    """
    DirectStep variant for networks that return a (mean, anomaly) tuple.
    It applies the transform to *both* components.
    
    In train mode, it returns:
        (composed_state, (transformed_mean, transformed_anomaly))
    In eval mode, it returns:
        composed_state
    """
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(DirectStepGlobalMean,self).__init__(network,
                                             dt,
                                             transform,
                                             state_transform,
                                             boundary_transform,
                                             device_name)
        

    def update_device(self, device_name):
        self.network = self.network.to(device = device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)             
        self.device_name = device_name

    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
        input_state.to(device=self.device_name)
        
        output = self.network(input_state)
        
        if self.network.training:
            mean_state, anom_state = output
            tf_mean_state = mean_state
            tf_anom_state = self.transform(anom_state)
            
            composed_state = tf_mean_state + tf_anom_state
            
            return composed_state, (tf_mean_state, tf_anom_state)
        else:
            tf_state = self.transform(output)
            return tf_state


class RK3Step(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(RK3Step, self).__init__(network,
                                      dt,
                                      transform,
                                      state_transform,
                                      boundary_transform,
                                      device_name)

    def update_device(self, device_name):
        self.network = self.network.to(device=device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.device_name = device_name
        
    def _get_tendency(self, state, transformed_boundary):
        """
        Helper function to calculate F(u) given state u and pre-transformed boundary.
        """
        input_state = self.state_transform(state)
        
        if transformed_boundary is not None:
            input_state = torch.cat((input_state, transformed_boundary), dim=1)
            
        return self.transform(self.network(input_state))

    def step_forward(self, state, boundary=None):
        # 1. Handle Dimensions and Device
        squeeze_output = False
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
            squeeze_output = True
            
        state = state.to(device=self.device_name)
        
        # 2. Prepare Boundary (Done once, reused for all RK stages)
        tf_boundary = None
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim=0)
            boundary = boundary.to(device=self.device_name)            
            tf_boundary = self.boundary_transform(boundary)

        # 3. Apply prev_state_transform to u^n (initial state)
        # We use this as the base for the additions to ensure masking consistency
        u_n = self.state_transform(state)
        
        # Stage 1
        # k1 = F(u^n)
        # u^1 = u^n + dt * k1
        k1 = self._get_tendency(u_n, tf_boundary)
        u1 = u_n + self.dt * k1
        
        # Stage 2
        # k2 = F(u^1)
        # u^2 = 3/4 * u^n + 1/4 * u^1 + 1/4 * dt * k2
        k2 = self._get_tendency(u1, tf_boundary)
        u2 = (3/4) * u_n + (1/4) * u1 + (1/4) * self.dt * k2
        
        # Stage 3
        # k3 = F(u^2)
        # u^{n+1} = 1/3 * u^n + 2/3 * u^2 + 2/3 * dt * k3
        k3 = self._get_tendency(u2, tf_boundary)
        u_next = (1/3) * u_n + (2/3) * u2 + (2/3) * self.dt * k3

        return u_next


class RK3StepDiffusion(TimeStepper):
    
    def __init__(self,
                 network: torch.nn.Module,
                 diffusion_module: transformations.LaplacianDiffusion,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super(RK3StepDiffusion, self).__init__(network,
                                      dt,
                                      transform,
                                      state_transform,
                                      boundary_transform,
                                      device_name)
        self.diffusion = diffusion_module


    def update_device(self, device_name):
        self.network = self.network.to(device=device_name)
        self.transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)
        self.device_name = device_name
        self.diffusion.update_device(device_name)


    def _get_tendency(self, state, transformed_boundary):
        """
        Helper function to calculate F(u) given state u and pre-transformed boundary.
        """
        input_state = self.state_transform(state)

        diffusion = self.diffusion(input_state)
        if transformed_boundary is not None:
            input_state = torch.cat((input_state, transformed_boundary), dim=1)
            
        return self.transform(self.network(input_state)) + diffusion

    def step_forward(self, state, boundary=None):
        # 1. Handle Dimensions and Device
        squeeze_output = False
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
            squeeze_output = True
            
        state = state.to(device=self.device_name)
        
        # 2. Prepare Boundary (Done once, reused for all RK stages)
        tf_boundary = None
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim=0)
            boundary = boundary.to(device=self.device_name)            
            tf_boundary = self.boundary_transform(boundary)

        # 3. Apply prev_state_transform to u^n (initial state)
        # We use this as the base for the additions to ensure masking consistency
        u_n = self.state_transform(state)
        
        # Stage 1
        # k1 = F(u^n)
        # u^1 = u^n + dt * k1
        k1 = self._get_tendency(u_n, tf_boundary)
        u1 = u_n + self.dt * k1
        
        # Stage 2
        # k2 = F(u^1)
        # u^2 = 3/4 * u^n + 1/4 * u^1 + 1/4 * dt * k2
        k2 = self._get_tendency(u1, tf_boundary)
        u2 = (3/4) * u_n + (1/4) * u1 + (1/4) * self.dt * k2
        
        # Stage 3
        # k3 = F(u^2)
        # u^{n+1} = 1/3 * u^n + 2/3 * u^2 + 2/3 * dt * k3
        k3 = self._get_tendency(u2, tf_boundary)
        u_next = (1/3) * u_n + (2/3) * u2 + (2/3) * self.dt * k3

        return u_next


class EulerStepDiffusion(TimeStepper):
    
    def __init__(self,
                 diffusion_module: torch.nn.Module,
                 network: torch.nn.Module,
                 dt: int,
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 prev_state_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        super().__init__(network,
                         dt,
                         transform,
                         state_transform,
                         boundary_transform,
                         device_name)
        self.diffusion = diffusion_module
        self.prev_state_transform = prev_state_transform

    def update_device(self, device_name):
        super().update_device(device_name)
        self.prev_state_transform.update_device(device_name)
        self.diffusion.update_device(device_name)
        
    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
            
        input_state.to(device=self.device_name)
        
        # Calculate tendencies
        network_tendency = self.transform(self.network(input_state))
        diffusion_tendency = self.diffusion(state)
        
        return self.prev_state_transform(state).squeeze(dim = 0) + self.dt * (network_tendency + diffusion_tendency).squeeze(dim = 0)


class CoupledDirectStep(TimeStepper):
    def __init__(self,
                 slow_network: torch.nn.Module,
                 fast_network: torch.nn.Module,
                 slow_indices: list[int],
                 fast_indices: list[int],
                 dt: int, # Not used for scaling in DirectStep, but kept for interface consistency
                 # Output Transforms (e.g. Wet Masks for T and U separately)
                 slow_transform: callable = transformations.NullTransform(),
                 fast_transform: callable = transformations.NullTransform(),
                 # Input State Transform (Normalization of the combined state)
                 state_transform: callable = transformations.NullTransform(),
                 # Boundary Transform
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        """
        Args:
            slow_network: Predicts T_{t+1}
            fast_network: Predicts U_{t+1}
            slow_indices: List of ALL channel indices belonging to slow vars (e.g., 0-37)
            fast_indices: List of ALL channel indices belonging to fast vars (e.g., 38-75)
        """
        # Initialize parent
        super().__init__(slow_network, dt, device_name=device_name)
        
        self.slow_network = slow_network
        self.fast_network = fast_network
        
        # Convert indices to tensors for efficient gathering
        self.slow_indices = torch.tensor(slow_indices, dtype=torch.long)
        self.fast_indices = torch.tensor(fast_indices, dtype=torch.long)
        
        self.slow_transform = slow_transform
        self.fast_transform = fast_transform
        self.state_transform = state_transform
        self.boundary_transform = boundary_transform
        
    def update_device(self, device_name):
        self.device_name = device_name
        self.slow_network = self.slow_network.to(device=device_name)
        self.fast_network = self.fast_network.to(device=device_name)
        
        self.slow_indices = self.slow_indices.to(device=device_name)
        self.fast_indices = self.fast_indices.to(device=device_name)
        
        self.slow_transform.update_device(device_name)
        self.fast_transform.update_device(device_name)
        self.state_transform.update_device(device_name)
        self.boundary_transform.update_device(device_name)

    def step_forward(self, state, boundary=None):
        # 1. Handle Input Dimensions
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
        state = state.to(device=self.device_name)
        
        # 2. Normalize and Split State
        # We normalize the full state first (if state_transform is provided)
        # Then we gather the specific channels for Slow vs Fast
        norm_state = self.state_transform(state)
        
        # Gather T_t and U_t
        # This works regardless of how many levels each variable has
        state_slow_t = torch.index_select(norm_state, 1, self.slow_indices)
        state_fast_t = torch.index_select(norm_state, 1, self.fast_indices)
        
        # 3. Prepare Boundary
        tf_boundary = None
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim=0)
            boundary = boundary.to(device=self.device_name)
            tf_boundary = self.boundary_transform(boundary)
            
        # --- PHASE 1: SLOW UPDATE (T_t -> T_t+1) ---
        # Input: T_t, U_t, Boundary
        if tf_boundary is not None:
            input_slow = torch.cat((state_slow_t, state_fast_t, tf_boundary), dim=1)
        else:
            input_slow = torch.cat((state_slow_t, state_fast_t), dim=1)
            
        # Predict T_{t+1} (Direct Step: Output is the state, not tendency)
        # Apply slow_transform (Wet Mask) immediately
        state_slow_next = self.slow_transform(self.slow_network(input_slow))
        
        # --- PHASE 2: FAST UPDATE (U_t -> U_t+1) ---
        # Input: U_t, T_t+1, Boundary
        # Note: We use the NEW slow state here
        if tf_boundary is not None:
            input_fast = torch.cat((state_fast_t, state_slow_next, tf_boundary), dim=1)
        else:
            input_fast = torch.cat((state_fast_t, state_slow_next), dim=1)
            
        # Predict U_{t+1}
        # Apply fast_transform (Wet Mask) immediately
        state_fast_next = self.fast_transform(self.fast_network(input_fast))
        
        # --- RECOMBINE ---
        # Scatter results back into the full state tensor in the correct order.
        # We create a container like norm_state and fill it.
        next_state = torch.zeros_like(norm_state)
        
        # We use simple slice assignment or index_copy_
        # index_copy_ is safer if indices are not contiguous, but standard assignment works 
        # if the tensors are on the same device.
        next_state[:, self.slow_indices] = state_slow_next
        next_state[:, self.fast_indices] = state_fast_next
        
        return next_state.squeeze(dim=0)


class DirectStepScaled(TimeStepper):
    def __init__(self,
                 network: torch.nn.Module,
                 dt: int,
                 residual_scales: torch.Tensor, # This is 'S'
                 transform: callable = transformations.NullTransform(),
                 state_transform: callable = transformations.NullTransform(),
                 boundary_transform: callable = transformations.NullTransform(),
                 device_name: str = 'cpu',
                ):
        """
        Args:
            residual_scales: Tensor (C,) containing the expected spatial std of the tendency.
                          Usually set to 1.5x or 2.0x the observed training std.
        """
        super().__init__(network, dt, transform, state_transform, boundary_transform, device_name)
        
        # Reshape for broadcasting: (1, Channels, 1, 1)
        if residual_scales.ndim == 1:
            self.residual_scales = residual_scales.view(1, -1, 1, 1)
        else:
            self.residual_scales = residual_scales
            
        self.residual_scales = self.residual_scales.to(device_name)

    def update_device(self, device_name):
        super().update_device(device_name)
        self.residual_scales = self.residual_scales.to(device_name)

    def step_forward(self, state, boundary = None):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim = 0)
        state = state.to(device = self.device_name)   
        if boundary is not None:
            if len(boundary.shape) == 3:
                boundary = boundary.unsqueeze(dim = 0)
            boundary = boundary.to(device=self.device_name)            
            boundary = self.boundary_transform(boundary)
            input_state = torch.cat((self.state_transform(state),boundary), dim = 1)
        else:
            input_state = self.state_transform(state)
        input_state.to(device=self.device_name)
        
        candidate_next_state = self.transform(self.network(input_state))
        raw_residual = candidate_next_state - state
        
        current_std = torch.std(raw_residual, dim=(-2, -1), keepdim=True)     

        eps = 1e-8
        scaling_factor = (self.residual_scales * torch.tanh(current_std / self.residual_scales)) / (current_std + eps)

        final_update = raw_residual * scaling_factor

        return state + final_update        