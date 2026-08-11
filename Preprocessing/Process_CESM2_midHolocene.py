"""Put the CESM2 midHolocene ocean output on the emulator grid (Section 2.1).

Same two-step regridding as Process_CESM2_piControl.py -- conservative in the vertical
onto 19 levels with partial cells, then conservative-normed onto a 1 degree Gaussian
grid -- with one addition. The midHolocene archive does not publish the total surface
stress on the ocean, only the stress at the bottom of the atmosphere and at the bottom
of the sea ice, so tauuo and tauvo are combined here, weighted by the local sea ice
concentration.

The store this writes is the regridded output only. The figure notebooks read the
`*_Detrended` store, which additionally removes the control drift from thetao, so and
hfds; that step is not part of this script. Everything else -- uo, vo, tauuo, tauvo and
the land mask -- is bit-identical between the two.

The four input stores are the concatenated CMIP6 midHolocene output
(doi:10.22033/ESGF/CMIP6.7674) on its native grid:

    ocean     Omon   thetao, so, hfds, and the wetmask / dz / cell corners
    velocity  Omon   uo, vo
    atmos     Amon   tauu, tauv
    ice       SImon  siconc, sistrxubot, sistryubot

Usage:
    python Process_CESM2_midHolocene.py init      # allocate the output store
    python Process_CESM2_midHolocene.py 0 4800    # fill those time steps
"""

import sys
from itertools import pairwise

import dask.array
import numpy as np
import xarray as xr
import xesmf as xe

# ------------------------------------------------------------------ configuration ---
PATH_OCEAN = "/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Ocean_Data.zarr"
PATH_VELOCITY = "/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Velocity_Data.zarr"
PATH_ATMOS = "/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Atm_Data.zarr"
PATH_ICE = "/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Ice_Data.zarr"

PATH_TARGET_GRID = "/pscratch/sd/a/asubel/Data/CM2x_grids/gaussian_grid_180_by_360.nc"
PATH_OUT = "/pscratch/sd/a/asubel/Data/Chapter_2/Holocene_Processed.zarr"

SURFACE_VARIABLES = ["hfds"]     # tauuo and tauvo are built after the horizontal regrid
STRESS_CLIP = 0.4                # N/m2; drops a handful of unphysical coastal points
SAMPLES_PER_STEP = 5             # time steps regridded per pass

TARGET_LEVEL_BOUNDS = np.array([0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650,
                                900, 1200, 1600, 2100, 2700, 3500, 4500, 5500, 6750])
TARGET_LEVEL_CENTERS = np.array([2.5, 10, 22.5, 40, 65, 105, 165, 250, 375,
                                 550, 775, 1050, 1400, 1850, 2400, 3100, 4000, 5000, 6000])


# ------------------------------------------------------------------------ helpers ---
def horizontal_regrid(ds, ds_target, na_thres=0.5, coarse_wetmask=None):
    """Regrid `ds` horizontally, and conserve the integral in space"""
    regridder_kwargs = dict(ignore_degenerate=True, periodic=True, unmapped_to_nan=True)

    s = xr.Dataset(
        coords={co: ds[co].astype("float128") for co in ["lon", "lat", "lon_b", "lat_b"]}
    )
    t = xr.Dataset(
        coords={co: ds_target[co].astype("float128") for co in ["lon", "lat", "lon_b", "lat_b"]}
    )
    if coarse_wetmask is not None:
        t["mask"] = coarse_wetmask

    regridder = xe.Regridder(s, t, "conservative_normed", **regridder_kwargs)
    ds_regridded = regridder(ds, skipna=True, na_thres=na_thres)

    lon, lat = ds_target.lon, ds_target.lat
    lon_b, lat_b = ds_target.lon_b, ds_target.lat_b
    r_earth = 6356  # in km
    new_area = xe.util.cell_area(ds_target, r_earth) * 1e6

    ds_regridded = ds_regridded.drop_vars(["lon_b", "lat_b"])
    ds_regridded = ds_regridded.assign_coords(
        lon=lon, lat=lat, lon_b=lon_b, lat_b=lat_b, areacello=new_area,
        x=lon.isel(y=0), y=lat.isel(x=0),
    )
    ds_regridded.attrs = ds.attrs | ds_regridded.attrs
    return ds_regridded


def load_source():
    """The ocean, atmosphere and sea ice stores on their shared time window."""
    data = xr.open_zarr(PATH_OCEAN)
    data_vel = xr.open_zarr(PATH_VELOCITY)
    data["uo"] = data_vel["uo"]
    data["vo"] = data_vel["vo"]
    data_atm = xr.open_zarr(PATH_ATMOS)
    data_ice = xr.open_zarr(PATH_ICE)

    # The sea ice record starts later than the ocean record and ends earlier.
    time_slice = slice(data_ice.time.values[0], data.time.values[-1])
    data = data.sel(time=time_slice)
    data_atm = data_atm.sel(time=time_slice)
    data_ice = data_ice.sel(time=time_slice)

    for ds in (data, data_atm, data_ice):
        for variable in set(ds.variables) - set(ds.coords):
            ds[variable] = ds[variable].astype(np.float32)
    return data, data_atm, data_ice


def load_target_grid():
    ds_target_grid = xr.open_dataset(PATH_TARGET_GRID).load()
    return ds_target_grid.rename({
        "grid_x": "x_b", "grid_y": "y_b", "grid_xt": "x", "grid_yt": "y",
        "grid_lon": "lon_b", "grid_lat": "lat_b", "grid_lont": "lon", "grid_latt": "lat",
    })


def vertical_operator(data):
    """Overlap of every native cell with every target cell, in metres.

    `deltas[k, l]` is the thickness of native level `l` that falls inside target level
    `k`, so a native cell straddling a target boundary contributes to both. Summing it
    over the native levels gives the target cell thickness `new_dz`, which is what makes
    the partial cells of Section 2.1 rather than a fixed thickness per level.
    """
    thkcello = (data["wetmask"] * data["dz"]).data
    data = data.assign_coords(thkcello=(["lev", "y", "x"], thkcello))

    new_dz = np.zeros((TARGET_LEVEL_CENTERS.size, *data.lon.shape))
    new_levels = xr.Dataset(coords={
        "lev_new": ("lev_new", TARGET_LEVEL_CENTERS),
        "lev_bounds_new": ("lev_bounds_new", TARGET_LEVEL_BOUNDS),
        "dz": (["lev_new", "y", "x"], new_dz),
        "x": ("x", data.x.data),
        "y": ("y", data.y.data),
    })

    zero_boundary = xr.zeros_like(data["thkcello"][0])
    zero_boundary["lev"] = 0.0
    thk_bnds = xr.concat([zero_boundary, data["thkcello"].cumsum("lev")], "lev")
    thk_bnds = thk_bnds.transpose("lev", "y", "x").compute()
    thk_centers = (thk_bnds[1:] + thk_bnds[0:-1].data) / 2

    deltas = xr.zeros_like(data["thkcello"])
    deltas = deltas.expand_dims(dim={"lev_new": new_levels.lev_new}, axis=0).copy()
    for i, (upper_bnd, lower_bnd) in enumerate(pairwise(new_levels.lev_bounds_new.values)):
        upper_bound = xr.ufuncs.maximum(thk_bnds[:-1], upper_bnd).values
        lower_bound = xr.ufuncs.minimum(thk_bnds[1:], lower_bnd).values
        update = np.where(lower_bound - upper_bound > 0, lower_bound - upper_bound, 0)
        new_dz[i] = update.sum(axis=0)
        deltas[i] = update

    return data, deltas.compute(), new_dz, new_levels, thk_bnds, thk_centers


def process_block(data, data_atm, data_ice, start, end, grid,
                  deltas, new_dz, new_levels, thk_bnds, thk_centers):
    """Vertically then horizontally regrid one block of time steps."""
    ds = data.isel(time=slice(start, end))
    ds_atm = data_atm.isel(time=slice(start, end)).transpose("y", "x", ...)
    ds_ice = data_ice.isel(time=slice(start, end)).transpose("y", "x", ...)

    ds = ds.chunk({"time": 1, "y": ds.y.size, "x": ds.x.size})
    ds["lev_bounds"] = data["lev_bnds"]
    ds = ds.assign_coords(thk_bnds=thk_bnds.transpose("lev", "y", "x"))
    ds = ds.assign_coords(thk_centers=thk_centers.transpose("lev", "y", "x"))

    ds_surface = ds[SURFACE_VARIABLES]
    ds = ds.drop_vars(SURFACE_VARIABLES)

    ds_vert = ((ds * deltas).sum("lev") / new_levels.dz)
    ds_vert = ds_vert.drop_vars(["lev_bounds", "lev_bnds"]).transpose("time", "lev_new", ...)
    ds_vert = ds_vert.rename({"lev_new": "lev"})
    ds_vert = ds_vert.assign_coords(wetmask=xr.where(
        deltas.sum("lev").rename({"lev_new": "lev"}).transpose("lev", "y", "x"), 1, np.nan))
    ds_vert = ds_vert * ds_vert["wetmask"]
    ds_vert = ds_vert.assign_coords(dz=(["lev", "y", "x"], new_dz)).reset_coords("dz")
    for variable in SURFACE_VARIABLES:
        ds_vert[variable] = ds_surface[variable]

    ds_regridded = horizontal_regrid(ds_vert, grid, na_thres=1)
    ds_regridded = ds_regridded.chunk({"time": 1, "x": ds_regridded.x.size,
                                       "y": ds_regridded.y.size, "lev": ds_regridded.lev.size})
    ds_regridded = ds_regridded.assign_coords(
        {"dz": (["lev", "y", "x"], ds_regridded["dz"].data)})

    # Total stress on the ocean: the atmospheric stress over open water plus the stress
    # under the ice, each weighted by the local ice concentration.  The atmospheric
    # stress is masked to the ocean first so land values cannot leak across the coast.
    ds_atm = horizontal_regrid(ds_atm, grid, na_thres=1)
    ocean_mask = (ds_regridded["hfds"].isel(time=0) * 0 + 1).data
    ds_atm[["tauu", "tauv"]] = ds_atm[["tauu", "tauv"]] * ocean_mask
    ds_ice = horizontal_regrid(ds_ice, grid, na_thres=1)

    ds_atm["tauu"] = xr.where(np.abs(ds_atm["tauu"]) > STRESS_CLIP, np.nan, ds_atm["tauu"])
    ds_atm["tauv"] = xr.where(np.abs(ds_atm["tauv"]) > STRESS_CLIP, np.nan, ds_atm["tauv"])

    ice_fraction = ds_ice["siconc"] / 100
    tauuo = (ds_atm["tauu"] * (1 - ice_fraction)
             - ds_ice["sistrxubot"].fillna(0.0) * ice_fraction)
    tauvo = (ds_atm["tauv"] * (1 - ice_fraction)
             - ds_ice["sistryubot"].fillna(0.0) * ice_fraction)
    ds_regridded["tauuo"] = (["time", "y", "x"], tauuo.values)
    ds_regridded["tauvo"] = (["time", "y", "x"], tauvo.values)
    return ds_regridded


def init_store(data, example):
    """Allocate the output store so the time blocks can be written as regions."""
    template = xr.Dataset(coords=example.coords)
    template["time"] = data.time
    for variable in set(example.variables) - set(example.coords.variables):
        shape = example[variable].shape
        if "time" in example[variable].dims:
            shape = (data.time.size,) + shape[1:]
        # Chunking comes from the explicit `.chunk()` below, and `to_zarr(compute=False)`
        # writes metadata only, so these zeros are never materialised.
        template[variable] = xr.DataArray(
            dask.array.zeros(shape), dims=example[variable].dims)

    template = template.chunk({"time": 1, "x": example.x.size, "y": example.y.size,
                               "lev": example.lev.size})
    for variable in set(template.variables) - set(template.coords):
        template[variable] = template[variable].astype("float32")

    encoding = template.encoding
    for variable in template.variables:
        encoding.setdefault(variable, {})["compressor"] = None

    template.to_zarr(PATH_OUT, encoding=encoding, compute=False)
    template[list(template.coords)].to_zarr(PATH_OUT, mode="a")
    print(f"initialised {PATH_OUT} with {data.time.size} time steps")


def main():
    data, data_atm, data_ice = load_source()
    grid = load_target_grid()
    data, deltas, new_dz, new_levels, thk_bnds, thk_centers = vertical_operator(data)
    operator = (grid, deltas, new_dz, new_levels, thk_bnds, thk_centers)

    if sys.argv[1] == "init":
        init_store(data, process_block(data, data_atm, data_ice, 0, 1, *operator))
        return

    begin, finish = int(sys.argv[1]), int(sys.argv[2])
    for j in range(begin // SAMPLES_PER_STEP,
                   int(np.ceil(finish / SAMPLES_PER_STEP))):
        start, end = j * SAMPLES_PER_STEP, (j + 1) * SAMPLES_PER_STEP
        print(start, end, flush=True)
        ds_regridded = process_block(data, data_atm, data_ice, start, end, *operator)
        ds_regridded.drop_vars(list(ds_regridded.coords.variables)).to_zarr(
            PATH_OUT, region={"time": slice(start, end)})


if __name__ == "__main__":
    main()
