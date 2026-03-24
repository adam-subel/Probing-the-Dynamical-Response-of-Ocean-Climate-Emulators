import numpy as np
import torch
import xarray as xr

from climlab import constants as const
from climlab.solar.insolation import daily_insolation
from typing import Sequence
import pandas as pd

import Utils.transformations as transformations



class data_from_single_source(torch.utils.data.Dataset):
    def __init__(self,
                 data: xr.Dataset,
                 variables: Sequence[str],
                 *,
                 transform: callable = transformations.NullTransform(),
                 steps = 1,
                 max_steps = 4,
                 label = False,
                ):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if max_steps < steps:
             # Ensure max_steps accommodates the requested number of steps
             print(f"Warning: max_steps ({max_steps}) is less than steps ({steps}). Adjusting max_steps to {steps}.")
             max_steps = steps

        self.size = data.time.size - max_steps # Adjusted size calculation
        if self.size <= 0:
             raise ValueError(f"Dataset time dimension ({data.time.size}) is too small for max_steps ({max_steps}). Need at least {max_steps+1} time points.")

        self.max_steps = max_steps
        self.variables = variables
        self.steps = steps
        self.label = label
        self.transform = transform
        if self.label:
            # Offset ensures labels (time t+1) are not included in the input sequence (up to time t)
            self.offset = 1
        else:
            self.offset = 0


        if variables:
            # Select only specified variables
            data = data[variables]
            # Ensure 'lev' dimension exists, even if size 1 (for variables without vertical levels)
            if 'lev' not in list(data.coords):
                data = data.expand_dims('lev', axis=1) # Specify axis for clarity if needed

            # Pre-process data for efficiency:
            # Instead of keeping it as a Dataset and processing every __getitem__,
            # we convert to a DataArray, stack, and transpose here.
            self.data = (
                data
                .fillna(0)
                .to_array(name="ocean state variables")
                .stack(channel=('variable', 'lev'))
                .transpose('time', 'channel', 'y', 'x')
            )

            # Determine input size and shape based on one time step
            self.input_size = self.data.channel.size
            self.input_shape = (self.input_size, self.data.y.size, self.data.x.size)

            # Store size of 'lev' dimension for each variable
            self.variable_size = {}
            for var in variables:
                if 'lev' in data[var].dims:
                    self.variable_size[var] = int(data[var].lev.size)
                else:
                     self.variable_size[var] = 1 # Assuming it was expanded to size 1

        else:
            # Handle case with no variables selected
            self.input_size = 0
            # Define a plausible shape even with 0 channels
            self.input_shape = (0, data.y.size, data.x.size)
            self.variable_size = {}
            self.data = data['time'] # No data to store

        # --- End of __init__ modifications (mainly clarifications and robustness) ---

    def sampling_rule(self):
        #TODO allow for configurable sampling rules
        return None

    def set_step(self, step):
        if step < 1:
            raise ValueError("steps must be at least 1")
        # Adjust size calculation if steps change and impact max_steps implicitly
        if step > self.max_steps:
             print(f"Warning: new step ({step}) > current max_steps ({self.max_steps}). Adjusting max_steps.")
             self.set_max_steps(step) # Adjust max_steps which also adjusts size
        self.steps = step

    def set_max_steps(self, max_steps):
        if max_steps < self.steps:
             raise ValueError(f"max_steps ({max_steps}) cannot be less than steps ({self.steps}).")
        # Calculate the change in max_steps and adjust the dataset size accordingly
        
        self.max_steps = max_steps
        # The size depends on the original data time size minus the maximum look-ahead needed
        new_size = self.data.time.size - self.max_steps if self.data is not None else 0 # Recalculate based on original data
        if new_size <= 0 and self.data is not None:
             raise ValueError(f"Dataset time dimension ({self.data.time.size}) is too small for new max_steps ({self.max_steps}).")
        self.size = new_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # --- Start of __getitem__ modifications ---

        if not self.variables or self.data is None:
            # If no variables are selected, return None consistently.
            # Previously returned a tuple of Nones based on self.steps.
            # Returning a single None seems more appropriate now as we expect a single tensor output.
            return (None,)*self.steps

        # Determine the time indices needed based on the input index/slice `idx`
        # We need `self.steps` time points for each sample requested by `idx`.
        # The starting time index for a sample `i` in the original dataset is `i + self.offset`.
        # The subsequent steps are `i + self.offset + 1`, ..., `i + self.offset + self.steps - 1`.

        # We can fetch all required time steps more efficiently in one go using slicing.
        # Convert the input `idx` (potentially a slice) into a sequence of base indices.
        if isinstance(idx, slice):
            # Handle slice indexing: convert slice to range of indices
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.size
            step = idx.step if idx.step is not None else 1
            base_indices = range(start, stop, step)
            if not base_indices: # Handle empty slice
                 return torch.empty((self.steps, 0, *self.input_shape), dtype=torch.float) # Match dims but 0 batch
            is_slice = True

        elif isinstance(idx, int):
            # Handle integer indexing: wrap it in a list
            if idx < 0: # Handle negative index
                idx += self.size
            base_indices = [idx]
            is_slice = False
            
        elif isinstance(idx, list):
            base_indices = idx
            is_slice = False
            
        else:
             raise TypeError(f"Index must be int or slice, not {type(idx)}")        


        batch_tensors = []
        for i in base_indices:
            # Define the slice in the original time dimension to get `self.steps` points
            time_slice = slice(i + self.offset, i + self.offset + self.steps)

            # Select the sequence for this sample
            sequence_xr = self.data.isel(time=time_slice)

            # --- Verification (Optional but Recommended) ---
            # Ensure we actually got the expected number of steps
            if sequence_xr.time.size != self.steps:
                 raise ValueError(
                     f"Retrieved {sequence_xr.time.size} time points for "
                     f"base index {i} (offset={self.offset}, steps={self.steps}), "
                     f"expected {self.steps}. Time slice was {time_slice} on data "
                     f"with time size {self.data.time.size}."
                 )
            # --- End Verification ---

            # Convert to numpy and then to torch tensor directly
            numpy_array = sequence_xr.to_numpy()
            torch_tensor = torch.from_numpy(numpy_array).to(torch.float)
            batch_tensors.append(torch_tensor)

        if not batch_tensors:
             return torch.empty((self.steps, 0, *self.input_shape), dtype=torch.float)

        # batch_tensors has shape [step, channel, y, x] for each element
        # We want to stack along the batch dimension (dim=1)
        # to get [step, batch, channel, y, x]
        final_tensor = torch.stack(batch_tensors, dim=1)

        # 7. Apply transformation
        final_tensor = self.transform(final_tensor)

        # 8. Squeeze the batch dimension if the original index was an integer
        if not is_slice:
            # Batch dimension is dim=1
            final_tensor = final_tensor.squeeze(1) # Output: [step, channel, y, x]

        return final_tensor

class data_climatology(torch.utils.data.Dataset):
    def __init__(self,
                 data: xr.Dataset,         # Used for time context (dayofyear sequence)
                 climatology: xr.Dataset,  # Used for data values, indexed by dayofyear
                 variables: Sequence[str],
                 *,
                 time_unit: str = 'Month',
                 transform: callable = transformations.NullTransform(),
                 steps = 1,
                 max_steps = 4,
                 label = False,
                ):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if max_steps < steps:
             print(f"Warning: max_steps ({max_steps}) is less than steps ({steps}). Adjusting max_steps to {steps}.")
             max_steps = steps


        self.size = data.time.size - max_steps
        if self.size <= 0: # Should be caught above, but safety check
             raise ValueError("Calculated dataset size is negative.")
        # -------------------------------------------

        self.max_steps = max_steps
        self.variables = variables
        self.steps = steps
        self.label = label
        self.transform = transform
        self.offset = 1 if self.label else 0

        # --- Pre-process Day of Year from 'data' ---
        try:
            if time_unit == 'Day':
                if isinstance(data.time.values[0], np.datetime64):
                    dayofyear_vals = [pd.to_datetime(i).dayofyear for i in data.time.values]
                elif hasattr(data.time.values[0], 'dayofyr'): # Check for cftime objects
                    dayofyear_vals = [i.dayofyr for i in data.time.values]
                else:
                    raise TypeError(f"Unsupported time coordinate type in 'data': {type(data.time.values[0])}")
                # Store as a simple numpy array or list for easy slicing
                self.time_of_year = torch.asarray(dayofyear_vals) # Keep as numpy array potentially
            elif time_unit == 'Month':
                if isinstance(data.time.values[0], np.datetime64):
                    dayofyear_vals = [pd.to_datetime(i).month for i in data.time.values]
                elif hasattr(data.time.values[0], 'dayofyr'): # Check for cftime objects
                    dayofyear_vals = [i.month for i in data.time.values]
                else:
                    raise TypeError(f"Unsupported time coordinate type in 'data': {type(data.time.values[0])}")
                # Store as a simple numpy array or list for easy slicing
                self.time_of_year = torch.asarray(dayofyear_vals) # Keep as numpy array potentially
            else:
                raise TypeError(f"Unsupported time unit: {time_unit}")
                
        except Exception as e:
            raise ValueError(f"Failed to extract day of year from 'data' time coordinates: {e}")
        # -----------------------------------------

        # --- Process 'climatology' Data ---
        if variables:
            climatology = climatology[variables]
            if time_unit == 'Day':
                climatology = climatology.rename({'dayofyear':'timeofyear'})
            elif time_unit == 'Month':
                climatology = climatology.rename({'month':'timeofyear'})                
            # Ensure 'lev' dimension exists for consistency
            if 'lev' not in list(climatology.coords): # Check coords of the dataset
                climatology = climatology.expand_dims('lev')

            # Load climatology data if it's chunked and accessed frequently? Optional.
            # self.climatology = climatology.load()
            self.climatology = climatology

            # Use dayofyear=1 (or first available) for shape calculation, assuming structure is consistent.
            # Need to handle potential missing dayofyear=1 if climatology starts later.
            first_day = self.climatology.timeofyear.min().item()
            single_step_clim = self.climatology.sel(timeofyear=first_day).to_array().stack(channel=('variable', 'lev'))
                


            # --- Calculate Input Shape/Size ---
            self.input_size = single_step_clim.channel.size
            self.input_shape = single_step_clim.transpose("channel", 'y', 'x').shape

            self.variable_size = {}
            for var in variables:
                # Check individual variable for 'lev' dimension
                if 'lev' in climatology[var].dims:
                    self.variable_size[var] = int(climatology[var].lev.size)
                else:
                    # If a variable has no level, assign size 1, assuming it's handled in stacking
                    self.variable_size[var] = 1
                    print(f"Warning: Variable '{var}' in climatology lacks 'lev' dimension. Assigning size 1.")
            # ----------------------------------
        else:
            self.input_size = 0
            # Use coordinate sizes from climatology for placeholder shape
            self.input_shape = (0, climatology.sizes.get('y', 0), climatology.sizes.get('x', 0))
            self.variable_size = {}
            self.climatology = None # No data to store

    def sampling_rule(self):
        #TODO allow for configurable sampling rules
        return None

    def set_step(self, step):
        if step < 1:
            raise ValueError("steps must be at least 1")
        if step > self.max_steps:
             print(f"Warning: new step ({step}) > current max_steps ({self.max_steps}). Adjusting max_steps.")
             self.set_max_steps(step) # This will adjust self.size based on 'data' size
        self.steps = step

    def set_max_steps(self, max_steps):
        if max_steps < self.steps:
             raise ValueError(f"max_steps ({max_steps}) cannot be less than steps ({self.steps}).")
        # Calculate the change in max_steps and adjust the dataset size accordingly
        self.max_steps = max_steps
        # The size depends on the original data time size minus the maximum look-ahead needed
        new_size = len(self.time_of_year) - self.max_steps if self.time_of_year is not None else 0 # Recalculate based on original data
        if new_size <= 0 and self.data is not None:
             raise ValueError(f"Dataset time dimension ({len(self.time_of_year)}) is too small for new max_steps ({self.max_steps}).")
        self.size = new_size

    def __len__(self):
        return max(0, self.size) # Ensure non-negative

    def __getitem__(self, idx):
        if not self.variables or self.climatology is None:
            return None # Return single None if no data/variables

        # 1. Determine the base indices from idx
        if isinstance(idx, slice):
            # Handle slice indexing: convert slice to range of indices
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.size
            step = idx.step if idx.step is not None else 1
            base_indices = range(start, stop, step)
            if not base_indices: # Handle empty slice
                 return torch.empty((self.steps, 0, *self.input_shape), dtype=torch.float) # Match dims but 0 batch
            is_slice = True

        elif isinstance(idx, int):
            # Handle integer indexing: wrap it in a list
            if idx < 0: # Handle negative index
                idx += self.size
            base_indices = [idx]
            is_slice = False
            
        elif isinstance(idx, list):
            base_indices = idx
            is_slice = False
            
        else:
             raise TypeError(f"Index must be int or slice, not {type(idx)}")        

        # --- Start of revised __getitem__ for climatology ---

        # 2. Select the climatology sequence for each base index and prepare for concat
        batch_xr_list = []
        for i in base_indices:
            # Define the slice in the original time dimension of 'data'
            # to get the indices for the day-of-year sequence
            time_index_slice = slice(i + self.offset, i + self.offset + self.steps)

            # Get the sequence of day-of-year values for this sample
            # Ensure indices are within bounds of self.time_of_year
           
            day_of_year_sequence_values = self.time_of_year[time_index_slice]

            # Select the sequence from climatology using the day-of-year values.
            # .sel should handle selecting multiple points along the 'dayofyear' dim.
            # Convert list/array to xr.DataArray for sel if needed, though list often works.
            climatology_sequence_xr = self.climatology.sel(timeofyear=day_of_year_sequence_values)

            # Rename the 'dayofyear' dimension to 'step'.
            # Assign integer coordinates to this new 'step' dimension.
            processed_sequence_xr = climatology_sequence_xr.rename({'timeofyear': 'step'}).assign_coords(
                {'step': range(self.steps)}
            )

            batch_xr_list.append(processed_sequence_xr)

        # 3. Concatenate along a new 'batch' dimension
        if not batch_xr_list: # Should be caught earlier by empty slice/list checks
             return torch.empty((self.steps, 0, *self.input_shape), dtype=torch.float)

        batch_coord = pd.Index(range(len(base_indices)), name='batch')
        # Concatenate the list of sequence datasets along the new 'batch' dimension
        # The 'step' dimension and its coordinates are now consistent across items
        concatenated_data = xr.concat(batch_xr_list, dim=batch_coord)
        # Resulting dims expected: [batch, step, (variables), lev, y, x]

        # 4. Process the concatenated xarray object
        # Fill NaNs, convert to array, stack variables/levels, transpose
        processed_data = (
            concatenated_data
            .fillna(0)
            .to_array(name="climatology variables") # Combine variables -> dim 'variable'
            .stack(channel=('variable', 'lev')) # Stack into 'channel'
            # Transpose to desired final order: [step, batch, channel, y, x]
            .transpose('step', 'batch', 'channel', 'y', 'x')
        )

        # 5. Convert to NumPy array
        numpy_array = processed_data.to_numpy()

        # 6. Convert to PyTorch tensor
        torch_tensor = torch.from_numpy(numpy_array).to(torch.float)

        # 7. Apply transformation
        final_tensor = self.transform(torch_tensor)

        # 8. Squeeze the batch dimension if the original index was an integer
        if not is_slice: # Only squeeze if original idx was int
            # Batch dimension is dim=1
            final_tensor = final_tensor.squeeze(1) # Output: [step, channel, y, x]

        # Expected final shape: [step, batch, channel, y, x] or [step, channel, y, x]
        return final_tensor

class data_radiation(torch.utils.data.Dataset):
    def __init__(self,
                 data: xr.Dataset,
                 *,
                 transform: callable = transformations.NullTransform(),
                 steps = 1,
                 max_steps = 4,
                 orb: dict[str,float]={'ecc': 0.017236, 'long_peri': 281.37, 'obliquity': 23.446},
                ):
        super().__init__()
        self.size = data.time.size - max_steps
        self.max_steps = max_steps
        self.variables = ['rsdt',]
        self.steps = steps
        self.transform = transform
        self.orb = orb
        self.input_size = 1
        self.data = data.copy(deep=True)
        if isinstance(data.time.values[0],np.datetime64):
            self.data['time'] = np.array([pd.Timestamp(i).day_of_year for i in data.time.values])
        else:
            self.data['time'] = np.array([i.dayofyr for i in data.time.values])
        self.input_shape = (1,) + self.data.lat.shape
        

        self.area = (self.data.areacello/self.data.areacello.max()).compute()
        self.lat = self.data.lat.load().expand_dims('lev',0)
        
        self.data = self.data['time']
        self.variable_size = {'rsdt': 1}

    def sampling_rule(self):
        #TODO allow for configurable sampling rules 
        return None
    
    def set_step(self, step):
        if step < 1:
            raise ValueError("steps must be at least 1")
        # Adjust size calculation if steps change and impact max_steps implicitly
        if step > self.max_steps:
             print(f"Warning: new step ({step}) > current max_steps ({self.max_steps}). Adjusting max_steps.")
             self.set_max_steps(step) # Adjust max_steps which also adjusts size
        self.steps = step

    def set_max_steps(self, max_steps):
        if max_steps < self.steps:
             raise ValueError(f"max_steps ({max_steps}) cannot be less than steps ({self.steps}).")
        # Calculate the change in max_steps and adjust the dataset size accordingly
        self.max_steps = max_steps
        # The size depends on the original data time size minus the maximum look-ahead needed
        new_size = self.data.time.size - self.max_steps if self.data is not None else 0 # Recalculate based on original data
        if new_size <= 0 and self.data is not None:
             raise ValueError(f"Dataset time dimension ({self.data.time.size}) is too small for new max_steps ({self.max_steps}).")
        self.size = new_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.variables:
            return (None,)*self.steps
        
        if isinstance(idx, slice):
            # Handle slice indexing: convert slice to range of indices
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.size
            step = idx.step if idx.step is not None else 1
            base_indices = range(start, stop, step)
            if not base_indices: # Handle empty slice
                 return torch.empty((self.steps, 0, *self.input_shape), dtype=torch.float) # Match dims but 0 batch
            is_slice = True

        elif isinstance(idx, int):
            # Handle integer indexing: wrap it in a list
            if idx < 0: # Handle negative index
                idx += self.size
            base_indices = [idx]
            is_slice = False
            
        elif isinstance(idx, list):
            base_indices = idx
            is_slice = False
            
        else:
             raise TypeError(f"Index must be int or slice, not {type(idx)}")    

        
        batch_xr_list = []
        for i in base_indices:
            # Define the slice in the original time dimension to get `self.steps` day-of-year values
            time_slice = slice(i, i + self.steps) # self.offset is 0

            # Select the sequence of day-of-year values for this sample
            day_of_year_sequence = self.data.isel(time=time_slice)

            # --- Verification (Optional) ---
            if day_of_year_sequence.time.size != self.steps:
                 raise ValueError(
                     f"Retrieved {day_of_year_sequence.time.size} day-of-year points for "
                     f"base index {i} (steps={self.steps}), "
                     f"expected {self.steps}. Time slice was {time_slice} on data "
                     f"with time size {self.data.time.size}."
                 )
            # --- End Verification ---

            # Calculate insolation sequence. Assume daily_insolation accepts a 1D DataArray
            # of day-of-year values and broadcasts latitude correctly, returning an
            # xarray object with dimensions [time, lev, y, x] where 'time' corresponds
            # to the sequence steps.
            # Ensure self.lat has coords that don't conflict with day_of_year_sequence if daily_insolation uses them.
            insolation_sequence_xr = daily_insolation(self.lat,
                                                      day_of_year_sequence,
                                                     orb = self.orb)

            # Apply area weighting (assuming self.area has dims [y, x] or broadcastable)
            weighted_insolation_xr = insolation_sequence_xr * self.area

            # Rename the 'time' dimension (inherited from day_of_year_sequence) to 'step'.
            # Assign integer coordinates to this new 'step' dimension.
            processed_sequence_xr = weighted_insolation_xr.rename({'time': 'step'}).assign_coords(
                {'step': range(self.steps)}
            )

            batch_xr_list.append(processed_sequence_xr)

        # 3. Concatenate along a new 'batch' dimension
        if not batch_xr_list: # Should be caught earlier by empty slice/list checks
             return torch.empty((self.steps, 0, *self.input_shape), dtype=torch.float)

        batch_coord = pd.Index(range(len(base_indices)), name='batch')
        concatenated_data = xr.concat(batch_xr_list, dim=batch_coord)
        # Resulting dims expected: [batch, step, lev, y, x]

        # 4. Transpose to desired final order
        # No need for .to_array() or .stack() here, as we have effectively one variable ('rsdt')
        # and the 'lev' dimension acts as the channel.
        transposed_data = concatenated_data.transpose('step', 'batch', 'lev', 'y', 'x')

        # 5. Convert to NumPy array (consider .compute() if data is large and dask-backed)
        # numpy_array = transposed_data.compute().to_numpy() # Use compute() if needed
        numpy_array = transposed_data.to_numpy() # Assume data fits in memory


        # 6. Convert to PyTorch tensor
        torch_tensor = torch.from_numpy(numpy_array).to(torch.float)

        # 7. Apply transformation
        final_tensor = self.transform(torch_tensor)

        # 8. Squeeze the batch dimension if the original index was an integer
        if not is_slice: # Only squeeze if original idx was int
            # Batch dimension is dim=1
            final_tensor = final_tensor.squeeze(1) # Output: [step, lev=1, y, x]

        # Expected final shape: [step, batch, lev=1, y, x] or [step, lev=1, y, x]
        return final_tensor
    
class MergedDatasets(torch.utils.data.Dataset):
    def __init__(self,
                 datasets: tuple,
                ):
        super().__init__()
        
        self.datasets = datasets        
        self.size = min(list([dataset.size for dataset in datasets]))     
        self.steps = min(list([dataset.steps for dataset in datasets]))       
        for dataset in self.datasets:
            dataset.set_step(self.steps) 
        self.max_steps = min(list([dataset.max_steps for dataset in datasets]))       
        for dataset in self.datasets:
            dataset.set_max_steps(self.max_steps)   
        self.variable_size = {}
                
        self.input_size = 0
        self.variables = []
        for dataset in self.datasets:
            self.input_size += dataset.input_size
            for var in dataset.variables:
                self.variables.append(var)
                self.variable_size[var] = dataset.variable_size[var]
                
            if dataset.input_shape[1:] != self.datasets[0].input_shape[1:]:
                raise ValueError('Trailing Spatial Dimensions do not Agree.')
        self.input_shape = (self.input_size,) + self.datasets[0].input_shape[1:]
        
    def set_step(self, step):
        for dataset in self.datasets:
            dataset.set_step(step)
        self.steps = step
        
    def set_max_steps(self, max_steps):
        for dataset in self.datasets:
            dataset.set_max_steps(max_steps)
        self.size = min(list([dataset.size for dataset in self.datasets]))          
        self.max_steps = max_steps

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        stacked_data = [data[idx] for data in self.datasets]
        return torch.cat(stacked_data,dim=-3)
    
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 integrated,
                 boundary,
                 labels,
                 observations = None,
                 climatology = None,
                ):
        super().__init__()
        self.size = integrated.size        
        self.integrated = integrated
        self.boundary = boundary
        self.labels = labels
        if observations:
            self.observations = observations
            self.has_observations = True
        else:
            self.has_observations = False
        if climatology:
            self.climatology = climatology
            self.has_climatology = True
        else:
            self.has_climatology = False            

    def sampling_rule(self):
        #TODO allow for configurable sampling rules 
        return None
    
    def set_step(self, step):
        self.boundary.set_step(step)
        self.labels.set_step(step)
        if self.has_observations:
            self.observations.set_step(step)
        if self.has_climatology:
            self.climatology.set_step(step)            
        
    def set_max_steps(self, max_steps):
        self.boundary.set_max_steps(max_steps)
        self.labels.set_max_steps(max_steps)
        self.integrated.set_max_steps(max_steps)
        if self. has_observations:
            self.observations.set_max_steps(max_steps)
        if self.has_climatology:
            self.climatology.set_max_steps(max_steps)            
        self.size = self.integrated.size        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):        
        if self.has_observations and self.has_climatology:
            return (self.integrated[idx],
                    self.boundary[idx], 
                    self.labels[idx], 
                    self.observations[idx], 
                    self.climatology[idx])
        elif self.has_observations:
            return (self.integrated[idx],
                    self.boundary[idx], 
                    self.labels[idx], 
                    self.observations[idx]
                   )
        elif self.has_climatology:
            return (self.integrated[idx],
                    self.boundary[idx], 
                    self.labels[idx], 
                    self.climatology[idx]
                   )            
        else:
            return self.integrated[idx], self.boundary[idx], self.labels[idx]
            