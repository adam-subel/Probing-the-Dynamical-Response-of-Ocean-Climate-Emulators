import xarray as xr
import numpy as np
import torch
import sys
import cftime
import os
import dask
import shutil

# Add utility modules path
sys.path.append("../")
import Utils.networks as networks
import Utils.data_functions as data_functions
import Utils.time_steppers as time_steppers
import Utils.transformations as transformations

# ==============================================================================
# 1. MASTER CONFIGURATION
# ==============================================================================
# --- Select Emulator and Epochs to run ---
# EXPERIMENT_PATH = '/pscratch/sd/a/asubel/Chapter_2/Networks/Direct_Step_CESM2_Finetune_Holocene_New_2025-08-18/'
# EXPERIMENT_PATH = '/pscratch/sd/a/asubel/Chapter_2/Networks/Direct_Step_CESM2_Finetune_piControl_New_Rad_2025-08-21/'
# EXPERIMENT_PATH = '/pscratch/sd/a/asubel/Chapter_2/Networks/Direct_Step_CESM2_Finetune_piControl_More_Data_2025-08-25/'
EXPERIMENT_PATH = '/pscratch/sd/a/asubel/Chapter_2/Networks/Truly_Replicate_CESM2_Baseline_Exact_2026-03-12/'
# EXPERIMENT_PATH = '/pscratch/sd/a/asubel/Chapter_2/Networks/Replicate_CESM2_Holocene_Fixed_2026-03-10/'
# EPOCHS_TO_RUN = [71]  # List of epochs to process, e.g., [50, 100, 'final']
# EPOCHS_TO_RUN = [40,50,,60,65,70]
# EPOCHS_TO_RUN = [60,65,70,75]
EPOCHS_TO_RUN = [40,45,50]

NAME_MODIFIER = 'Reconstructed_Baseline_Fixed'

# --- Define Base Climate (must match the emulator) and Perturbation Target ---
BASE_CLIMATE = 'piControl'
TARGET_CLIMATE = 'Holocene'

# BASE_CLIMATE = 'Holocene'
# TARGET_CLIMATE = 'piControl'

# --- Define Perturbation Experiments for EACH epoch ---
# 'forcings_to_perturb': None runs the control with base climate forcing.
PERTURBATION_EXPERIMENTS = [
    {'forcings_to_perturb': None},
    {'forcings_to_perturb': ['hfds', 'tauuo', 'tauvo', 'orb']},
    {'forcings_to_perturb': ['hfds',]},
    {'forcings_to_perturb': ['tauuo',]},
    {'forcings_to_perturb': ['tauvo',]},
]

# --- Rollout and System Parameters ---
IC_ROLLOUT_YEARS = 100  # Length of the initial monthly rollout for ICs
RESPONSE_ROLLOUT_YEARS = 150  # Length of each response experiment
NUM_INIT_TIMES = 5  # Number of ensemble members for response runs
INITIAL_CONDITION_SPACING = 24  # Spacing in months between initial conditions
YEARS_PER_SAVE = 10 # For response runs, save data every N years

# --- Paths and Device ---
DEVICE_NAME = 'cuda'
BASE_SAVE_PATH = '/pscratch/sd/a/asubel/Chapter_2/Preds/'

# --- Ground Truth Data Configuration ---
CLIMATE_DATA_CONFIG = {
    'piControl': {
        'path': '/pscratch/sd/a/asubel/Data/Chapter_2/CESM2_piControl_Detrended.zarr/',
        'time_slice': slice('0900-01-01', '1100-12-31'),
        'ic_time_slice': slice('1100-01-01', f'{1100 + IC_ROLLOUT_YEARS}-02-01')

    },
    'Holocene': {
        'path': '/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Processed_Detrended.zarr/',
        'time_slice': slice('0500-01-01', '0700-12-31'),
        'ic_time_slice': slice('0595-01-01', f'0{595 + IC_ROLLOUT_YEARS}-02-01')
    }
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def extend_cftime_array(original_times, n_months_to_add):
    """Extends a cftime array by a given number of months."""
    if not len(original_times):
        raise ValueError("Original time array cannot be empty.")
    monthly_template = {
        date.month: {
            "day": date.day, "hour": date.hour, "minute": date.minute,
            "second": date.second, "microsecond": date.microsecond,
            "has_year_zero": getattr(date, 'has_year_zero', False)
        } for date in original_times[:12]
    }
    last_date = original_times[-1]
    current_year, current_month = last_date.year, last_date.month
    new_dates = []
    for _ in range(n_months_to_add):
        current_month += 1
        if current_month > 12:
            current_month, current_year = 1, current_year + 1
        template = monthly_template[current_month]
        new_dates.append(cftime.DatetimeNoLeap(current_year, current_month, **template))
    return np.concatenate((original_times, np.array(new_dates, dtype=object)))

# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================


def generate_initial_condition_rollout(forward_model, inv_norm_labels, device, save_path, *args):
    """Generates and saves the long monthly rollout to be used for ICs."""
    label_vars, bound_vars, int_vars, inc_rad, bound_trans, int_trans, norm_rad = args
    
    # Load ground truth data for forcing
    data = xr.open_zarr(CLIMATE_DATA_CONFIG[BASE_CLIMATE]['path']).sel(y=slice(-90, 64))
    data = data.sel(time=CLIMATE_DATA_CONFIG[BASE_CLIMATE]['ic_time_slice'])
    wet = (data.isel(time=0)[label_vars] * 0 + 1).drop_vars('time').compute()

    # Prepare boundary and initial state values
    boundary_values, integrated_values = prepare_data_tensors(
        data,
        bound_vars,
        int_vars,
        inc_rad,
        bound_trans,
        int_trans,
        norm_rad,
        rad_clim = BASE_CLIMATE,
        is_climatology=False
    )
    label_values = data_functions.data_from_single_source(data, label_vars, max_steps=1, steps=1)
    
    # Create Zarr store
    rollout_length = data.time.size - 1
    ds_temp = xr.Dataset(coords=data.isel(time=slice(1, rollout_length + 1)).coords)
    for var in label_vars:
        shape = (rollout_length,) + data[var].shape[1:]
        ds_temp[var] = xr.DataArray(dask.array.zeros(shape), dims=('time',) + data[var].dims[1:])
    ds_temp.to_zarr(save_path, compute=False, mode='w')
    
    # Run rollout
    steps_per_save_chunk = 50
    preds_buffer = torch.zeros((steps_per_save_chunk,) + label_values.input_shape)

    with torch.no_grad():
        out = None
        for i in range(rollout_length):
            if i % 120 == 0: print(f"    IC Gen: Year {i//12 + 1}/{IC_ROLLOUT_YEARS}")
            
            if i == 0:
                state = integrated_values[0][0]
                boundary = boundary_values[0][0]
            else:
                state = out
                boundary = boundary_values[i][0]

            out = forward_model.step_forward(state, boundary)
            preds_buffer[i % steps_per_save_chunk] = out

            # Save in chunks
            is_last_step = (i == rollout_length - 1)
            is_chunk_end = (i + 1) % steps_per_save_chunk == 0
            if is_chunk_end or is_last_step:
                num_steps_in_chunk = i % steps_per_save_chunk + 1
                preds_to_save = preds_buffer[:num_steps_in_chunk]
                preds_to_save = inv_norm_labels(preds_to_save).cpu()

                start_idx = i - (num_steps_in_chunk - 1)
                end_idx = i + 1
                chunk_coords = ds_temp.isel(time=slice(start_idx, end_idx)).coords
                
                pred_array = xr.Dataset(coords=chunk_coords)
                tensor_idx_start = 0
                for var in label_vars:
                    var_size = label_values.variable_size[var]
                    var_data = preds_to_save[:, tensor_idx_start : tensor_idx_start + var_size]
                    
                    if 'lev' in data[var].dims:
                        var_dims = ('time', 'lev', 'y', 'x')
                    else:
                        var_dims = ('time', 'y', 'x')
                    
                    pred_array[var] = (xr.DataArray(var_data.numpy(), coords=chunk_coords, dims=var_dims) * wet[var])
                    tensor_idx_start += var_size
                
                region_to_write = {'time': slice(start_idx, end_idx)}
                pred_array.drop_vars(list(pred_array.coords.keys())).to_zarr(save_path, region=region_to_write)
                
                if not is_last_step:
                     preds_buffer.zero_()

    print(f"IC rollout generation complete. Saved to: {save_path}")


def run_response_experiment(forward_model, inv_norm_labels, device, epoch, ic_path, exp_config, *args):
    """Runs a single multi-member response experiment for a given forcing setup."""
    label_vars, bound_vars, int_vars, inc_rad, bound_trans, int_trans, norm_rad = args
    forcings_to_perturb = exp_config['forcings_to_perturb']

    # --- Prepare data and climatological boundary conditions ---
    data_truth_base = xr.open_zarr(CLIMATE_DATA_CONFIG[BASE_CLIMATE]['path']).sel(y=slice(-90, 64))
    data_truth_target = xr.open_zarr(CLIMATE_DATA_CONFIG[TARGET_CLIMATE]['path']).sel(y=slice(-90, 64))
    
    wet = (data_truth_base[label_vars].isel(time=0) * 0 + 1).drop_vars('time').compute()

    boundary_climatology = data_truth_base[bound_vars].sel(
        time=CLIMATE_DATA_CONFIG[BASE_CLIMATE]['time_slice']
    ).groupby('time.month').mean('time').compute()

    if forcings_to_perturb and forcings_to_perturb != ['orb']:
        dyn_forcings = [f for f in forcings_to_perturb if f != 'orb']
        if dyn_forcings:
            boundary_climatology_modify = data_truth_target[dyn_forcings].sel(
                time=CLIMATE_DATA_CONFIG[TARGET_CLIMATE]['time_slice']
            ).groupby('time.month').mean('time').compute()
            # boundary_climatology[dyn_forcings] = boundary_climatology_modify[dyn_forcings]
            boundary_climatology.update(boundary_climatology_modify[dyn_forcings])
    data_ics = xr.open_zarr(ic_path)
    
    # --- Prepare data tensors ---
    rollout_length_months = RESPONSE_ROLLOUT_YEARS * 12
    extended_times = extend_cftime_array(data_ics.time.values, rollout_length_months)
    
    climatological_boundary_ds = xr.zeros_like(
        data_truth_base[bound_vars].reindex({"time": extended_times}, fill_value=0)
    ).groupby('time.month') + boundary_climatology

    if forcings_to_perturb is None:
        rad_clim = BASE_CLIMATE
    elif 'orb' in forcings_to_perturb:
        rad_clim = TARGET_CLIMATE
    else:
        rad_clim = BASE_CLIMATE
    
    boundary_values, integrated_values = prepare_data_tensors(
        climatological_boundary_ds, bound_vars, int_vars, inc_rad, bound_trans, int_trans, norm_rad, 
        is_climatology=True, rad_clim = rad_clim, ic_source=data_ics
    )
    label_values = data_functions.data_from_single_source(data_ics, label_vars, max_steps=1, steps=1)

    # --- Setup Zarr store for yearly means ---
    init_indices = [i * INITIAL_CONDITION_SPACING for i in range(NUM_INIT_TIMES)]
    init_times = data_ics.time.values[init_indices]
    
    folder_name = f"BVP_CESM2_{NAME_MODIFIER}_{BASE_CLIMATE}_Epoch_{epoch}_Response_{TARGET_CLIMATE}_BC"
    if forcings_to_perturb:
        folder_name += ''.join(['_' + f for f in forcings_to_perturb])
    
    save_path = os.path.join(BASE_SAVE_PATH, folder_name, 'yearly_means.zarr')
    
    if os.path.exists(os.path.dirname(save_path)): shutil.rmtree(os.path.dirname(save_path))
    os.makedirs(os.path.dirname(save_path))
    
    ds_yearly_template = create_yearly_zarr_template(
        data_truth_base, label_vars, NUM_INIT_TIMES, RESPONSE_ROLLOUT_YEARS, init_times
    )
    ds_yearly_template.to_zarr(save_path, compute=False, mode='w')

    # --- Main Rollout Loop ---
    for init_idx, model_step_idx in enumerate(init_indices):
        print(f"    Running Initial Condition {init_idx+1}/{NUM_INIT_TIMES}...")
        yearly_means_buffer = []
        yearly_accumulator = torch.zeros(label_values.input_shape)
        month_counter = 0

        with torch.no_grad():
            out = None
            for i in range(rollout_length_months):
                current_step_idx = model_step_idx + i
                if i == 0:
                    state = integrated_values[model_step_idx][0].to(device)
                else:
                    state = out
                
                boundary = boundary_values[current_step_idx][0]
                out = forward_model.step_forward(state, boundary)

                yearly_accumulator += out.squeeze().cpu()
                month_counter += 1

                if month_counter == 12 or i == rollout_length_months - 1:
                    yearly_means_buffer.append(yearly_accumulator / month_counter)
                    yearly_accumulator.zero_()
                    month_counter = 0
                    
                    current_year_index = i // 12
                    if (current_year_index + 1) % YEARS_PER_SAVE == 0 or i == rollout_length_months - 1:
                        save_yearly_chunk(
                            yearly_means_buffer, inv_norm_labels, label_values, label_vars, wet,
                            ds_yearly_template, save_path, init_idx, current_year_index
                        )
                        yearly_means_buffer = []

    print(f"    Response experiment complete. Saved to: {save_path}")

def prepare_data_tensors(data,
                         bound_vars,
                         int_vars,
                         inc_rad,
                         bound_trans,
                         int_trans,
                         norm_rad,
                         is_climatology,
                         rad_clim,
                         ic_source=None,
                         ):
    """Helper to prepare boundary and integrated value tensors."""
    orb_params = {
        'Holocene': {'ecc': 0.018682, 'obliquity': 24.105, 'long_peri': 180.87},
        'piControl': {'ecc': 0.016764, 'obliquity': 23.459, 'long_peri': 280.33}
    }
    # This logic is complex and derived from your scripts. It may need fine-tuning.
    if len(bound_trans.transformations) > 2:
        bound_trans.transformations = bound_trans.transformations[:2]

    source = ic_source if ic_source is not None else data
    
    if bound_vars:
        boundary_values = data_functions.data_from_single_source(
            data, bound_vars, transform=bound_trans, max_steps=1, steps=1
        )
        if inc_rad:
            # This part needs the logic from your script to determine which orb_values to use
            # based on experiment config. Assuming BASE_CLIMATE for simplicity here.
            radiation_values = data_functions.data_radiation(
                data, transform=norm_rad, max_steps=1, steps=1, orb=orb_params[rad_clim]
            )
            boundary_values = data_functions.MergedDatasets((boundary_values, radiation_values))
    else: # Fallback for no boundary vars
        boundary_values = data_functions.data_from_single_source(data, [], max_steps=1, steps=1)

    integrated_values = data_functions.data_from_single_source(
        source, int_vars, transform=int_trans, max_steps=1, steps=1
    )
    return boundary_values, integrated_values

def create_yearly_zarr_template(data_template, label_vars, n_inits, n_years, init_times):
    """Creates the empty Xarray dataset for the yearly-mean Zarr store."""
    ds_yearly = xr.Dataset()
    for var in label_vars:
        yearly_shape = (n_inits, n_years) + data_template[var].shape[1:]
        dims = ('init_time', 'year') + data_template[var].dims[1:]
        ds_yearly[var] = xr.DataArray(dask.array.zeros(yearly_shape), dims=dims)
    ds_yearly = ds_yearly.assign_coords({
        'init_time': ('init_time', init_times),
        'year': ('year', np.arange(n_years))
    })
    for coord in ['lev', 'y', 'x']:
        if coord in data_template.coords:
            ds_yearly = ds_yearly.assign_coords({coord: data_template[coord]})
    return ds_yearly

def save_yearly_chunk(buffer, inv_norm, label_vals, label_vars, wet, template_ds, path, init_idx, current_year):
    """Saves a chunk of yearly means to the Zarr store."""
    preds_tensor = torch.stack(buffer)
    preds_tensor = inv_norm(preds_tensor)

    start_year = current_year - len(buffer) + 1
    end_year = current_year + 1
    
    pred_chunk = xr.Dataset()
    idx_start = 0
    for var in label_vars:
        var_size = label_vals.variable_size[var]
        var_data = preds_tensor[:, idx_start:idx_start + var_size].unsqueeze(0).cpu().numpy()
        
        if 'lev' in template_ds[var].dims:
            var_data = var_data.reshape(
                1, len(buffer), template_ds.lev.size, template_ds.y.size, template_ds.x.size
            )
        pred_chunk[var] = xr.DataArray(var_data, dims=template_ds[var].dims)
        idx_start += var_size
    
    region = {'init_time': slice(init_idx, init_idx + 1), 'year': slice(start_year, end_year)}
    # FIX: Drop coordinate variables before writing to a region to avoid dimension mismatch.
    (pred_chunk * wet).drop_vars(list(set(pred_chunk.coords.keys())|set(wet.coords.keys()))).to_zarr(path, region=region)
    print(f"      ...saved chunk of {len(buffer)} years to Zarr.")


## Rolling out the code

device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")

for epoch in EPOCHS_TO_RUN:
    print(f"\n{'='*60}\nProcessing Epoch: {epoch}\n{'='*60}")

    # --- 3.1 Load Model and Configuration for the Current Epoch ---
    print(f"--> Loading model from epoch {epoch}...")
    sys.path.append(EXPERIMENT_PATH)
    from Config import forward_model, normalization_label_variables, label_variables, \
                         boundary_variables, integrated_variables, include_radiation, \
                         boundary_transformation, integrate_transformation, normalization_radiation

    model_path = os.path.join(EXPERIMENT_PATH, f'Checkpoints/Checkpoint_epoch_{epoch}.tar')
    
    forward_model.network.load_state_dict(torch.load(model_path, map_location=device)['network_state'])
    forward_model.state_transform = transformations.NullTransform()
    forward_model.update_device(DEVICE_NAME)
    forward_model.network = forward_model.network.eval()
    inv_norm_labels = transformations.inv_normalize(
        means=normalization_label_variables.means,
        stds=normalization_label_variables.stds
    )
    
    # --- 3.2 Generate 100-Year Initial Condition Rollout ---
    print(f"\n--> Stage 1: Generating {IC_ROLLOUT_YEARS}-year monthly rollout for initial conditions...")
    ic_rollout_path = os.path.join(BASE_SAVE_PATH, f"BVP_{NAME_MODIFIER}_{BASE_CLIMATE}_IC_Rollout_Epoch_{epoch}.zarr")

    if os.path.exists(ic_rollout_path):
        print(f"    Found existing IC rollout at: {ic_rollout_path}. Skipping generation.")
    else:
        generate_initial_condition_rollout(
            forward_model, inv_norm_labels, device, ic_rollout_path,
            label_variables, boundary_variables, integrated_variables,
            include_radiation, boundary_transformation, integrate_transformation,
            normalization_radiation
        )

    # --- 3.3 Run Perturbation Response Experiments ---
    print(f"\n--> Stage 2: Running perturbation experiments using generated ICs...")
    for i, exp_config in enumerate(PERTURBATION_EXPERIMENTS):
        print(f"\n--- Running Perturbation Experiment {i+1}/{len(PERTURBATION_EXPERIMENTS)} ---")
        print(f"    Forcings perturbed: {exp_config['forcings_to_perturb']}")
        run_response_experiment(
            forward_model, inv_norm_labels, device, epoch, ic_rollout_path, exp_config,
            label_variables, boundary_variables, integrated_variables,
            include_radiation, boundary_transformation, integrate_transformation,
            normalization_radiation
        )