"""
Usage:
    python Process_CESM2_1pctCO2.py init      # allocate the output store
    python Process_CESM2_1pctCO2.py 0 1800    # fill those time steps
"""

import sys
import time
from itertools import pairwise

import dask.array
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

# ------------------------------------------------------------------ configuration ---
EXPERIMENT = "1pctCO2"

# thetao must stay first: it is the dataset the others are merged into, and it carries
# the `lev_bnds` the vertical operator needs.
REQUIRED_VARIABLES = ["thetao", "so", "hfds", "tauuo", "tauvo"]
OPTIONAL_VARIABLES = ["uo", "vo"]      # written when published, skipped when not
SURFACE_VARIABLES = ["hfds", "tauuo", "tauvo"]

CATALOG_URL = "https://cmip6.storage.googleapis.com/pangeo-cmip6.csv"
SOURCE_ID = "CESM2"
GRID_LABEL = "gn"
PREFERRED_MEMBER = "r1i1p1f1"

# `variable_id` alone is ambiguous -- tas is in Amon/day/3hr, thetao in Omon and Oyr.
TABLE_IDS = {
    "thetao": "Omon", "so": "Omon", "uo": "Omon", "vo": "Omon",
    "hfds": "Omon", "tauuo": "Omon", "tauvo": "Omon",
    "areacello": "Ofx", "deptho": "Ofx",
}
# The native grid is the same in every CESM2 experiment, so the time-invariant Ofx
# fields can be taken from another experiment when they are not published for this one.
GRID_EXPERIMENTS = [EXPERIMENT, "piControl", "historical"]

PATH_TARGET_GRID = "/pscratch/sd/a/asubel/Data/CM2x_grids/gaussian_grid_180_by_360.nc"
PATH_OUT = "/pscratch/sd/a/asubel/Data/Chapter_2/CESM2_1pctCO2_remade.zarr"

N_MONTHS = 1800            # the full 150-year ramp
SAMPLES_PER_STEP = 5       # time steps regridded per pass

TARGET_LEVEL_BOUNDS = np.array([0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650,
                                900, 1200, 1600, 2100, 2700, 3500, 4500, 5500, 6750])
TARGET_LEVEL_CENTERS = np.array([2.5, 10, 22.5, 40, 65, 105, 165, 250, 375,
                                 550, 775, 1050, 1400, 1850, 2400, 3100, 4000, 5000, 6000])


# --------------------------------------------------------------- catalogue access ---
def retry(action, description, attempts=4, delay=5):
    """Run `action`, retrying transient cloud failures with a backing-off delay."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:                                  # noqa: BLE001
            if attempt == attempts:
                raise RuntimeError(f"{description} failed after {attempts} attempts") from error
            print(f"  {description}: attempt {attempt} failed ({error!r}); "
                  f"retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay *= 2


def load_catalog():
    return retry(lambda: pd.read_csv(CATALOG_URL), "reading the Pangeo CMIP6 catalogue")


def catalog_rows(df, variable, experiment, member=None):
    """The catalogue rows for one variable, newest version first.

    Same filter as the original `df.query`, plus the `table_id` for this variable and,
    when asked, one ensemble member. Sorting by version makes `.values[0]` deterministic.
    """
    rows = df[(df.source_id == SOURCE_ID)
              & (df.experiment_id == experiment)
              & (df.variable_id == variable)
              & (df.grid_label == GRID_LABEL)]
    table_id = TABLE_IDS.get(variable)
    if table_id is not None:
        rows = rows[rows.table_id == table_id]
    if member is not None:
        rows = rows[rows.member_id == member]
    return rows.sort_values("version", key=lambda column: column.astype(str), ascending=False)


def pick_member(df):
    """One ensemble member that publishes every required variable."""
    available = {variable: set(catalog_rows(df, variable, EXPERIMENT).member_id)
                 for variable in REQUIRED_VARIABLES}

    missing = [variable for variable, members in available.items() if not members]
    if missing:
        raise RuntimeError(
            f"{SOURCE_ID} {EXPERIMENT}: no catalogue entry for {', '.join(missing)} "
            f"(grid_label {GRID_LABEL}, table "
            f"{', '.join(sorted({TABLE_IDS[v] for v in missing}))})")

    shared = set.intersection(*available.values())
    if not shared:
        summary = "; ".join(f"{v}: {sorted(m)}" for v, m in available.items())
        raise RuntimeError(f"{SOURCE_ID} {EXPERIMENT}: no single member publishes all of "
                           f"{REQUIRED_VARIABLES} -- {summary}")

    return PREFERRED_MEMBER if PREFERRED_MEMBER in shared else sorted(shared)[0]


def variable_store(df, variable, member):
    """The zarr store for one time-varying variable, for the chosen member."""
    rows = catalog_rows(df, variable, EXPERIMENT, member)
    if rows.empty:
        raise RuntimeError(f"{SOURCE_ID} {EXPERIMENT} {member}: no catalogue entry for "
                           f"{variable} ({TABLE_IDS.get(variable, 'any table')}, "
                           f"grid_label {GRID_LABEL})")
    if len(rows) > 1:
        print(f"  {variable}: {len(rows)} catalogue entries, taking version "
              f"{rows.version.values[0]}")
    return rows.zstore.values[0]


def grid_store(df, variable):
    """The zarr store for a time-invariant grid field, falling back across experiments."""
    for experiment in dict.fromkeys(GRID_EXPERIMENTS):
        rows = catalog_rows(df, variable, experiment)
        if not rows.empty:
            if experiment != EXPERIMENT:
                print(f"  {variable}: not published for {EXPERIMENT}, taking it from "
                      f"{experiment} (the native grid is identical across experiments)")
            return rows.zstore.values[0]
    raise RuntimeError(f"{SOURCE_ID}: no {variable} in the catalogue for any of "
                       f"{GRID_EXPERIMENTS}")


def open_store(zstore):
    return retry(lambda: xr.open_zarr(fsspec.get_mapper(zstore)), f"opening {zstore}")


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
    """The native-grid CMIP6 fields, with the cell corners the regridder needs."""
    df = load_catalog()
    member = pick_member(df)
    print(f"{SOURCE_ID} {EXPERIMENT}: member {member}", flush=True)

    variables = list(REQUIRED_VARIABLES)
    for variable in OPTIONAL_VARIABLES:
        if catalog_rows(df, variable, EXPERIMENT, member).empty:
            print(f"  {variable}: not published for {member}, leaving it out")
        else:
            variables.append(variable)
    print(f"  variables: {', '.join(variables)}", flush=True)

    # Cell corners: CMIP bounds are per-cell, so the shared vertices are assembled by
    # taking corner 0 of every cell plus the outer edges of the last row and column.
    area = open_store(grid_store(df, "areacello")).rename({"nlat": "y", "nlon": "x"})
    vertex_shape = tuple(i + 1 for i in area.lat.shape)
    lon_b, lat_b = np.zeros(vertex_shape), np.zeros(vertex_shape)
    for bounds, corners in [(lon_b, area["lon_bnds"]), (lat_b, area["lat_bnds"])]:
        bounds[:-1, :-1] = corners[:, :, 0]
        bounds[-1, :-1] = corners[-1, :, 3]
        bounds[:-1, -1] = corners[:, -1, 1]
        bounds[-1, -1] = corners[-1, -1, 2]

    # thetao is the base dataset; the rest are merged into it. The time axes are compared
    # first because a bare assignment would reindex a mismatched axis and fill with NaN.
    data = open_store(variable_store(df, variables[0], member))
    for variable in variables[1:]:
        ds = open_store(variable_store(df, variable, member))
        if not ds.time.equals(data.time):
            raise RuntimeError(
                f"{variable}: time axis does not match {variables[0]} "
                f"({ds.time.size} steps, {ds.time.values[0]} to {ds.time.values[-1]} "
                f"vs {data.time.size} steps, {data.time.values[0]} to "
                f"{data.time.values[-1]})")
        if ds[variable].sizes.get("nlat") != data.sizes["nlat"] \
                or ds[variable].sizes.get("nlon") != data.sizes["nlon"]:
            raise RuntimeError(f"{variable}: horizontal shape does not match {variables[0]}")
        data[variable] = ds[variable]

    if data.sizes["nlat"] != area.sizes["y"] or data.sizes["nlon"] != area.sizes["x"]:
        raise RuntimeError("the areacello grid does not match the data grid: "
                           f"{(area.sizes['y'], area.sizes['x'])} vs "
                           f"{(data.sizes['nlat'], data.sizes['nlon'])}")
    if data.time.size < N_MONTHS:
        raise RuntimeError(f"{EXPERIMENT} has {data.time.size} months, "
                           f"fewer than the {N_MONTHS} this script asks for")

    data = data.rename({"nlat": "y", "nlon": "x"})
    data = data.drop_vars(["time_bnds", "lat_bnds", "lon_bnds"], errors="ignore")

    data = data.assign_coords(
        wetmask=(["lev", "y", "x"], ((data["so"][0] * data["thetao"][0]) * 0 + 1).values))
    dz = np.abs(data.lev_bnds[:, 0].values - data.lev_bnds[:, 1].values)
    data = data.assign_coords(dz=(["lev"], dz))
    data = data.assign_coords(lon_b=(["y_b", "x_b"], lon_b))
    data = data.assign_coords(lat_b=(["y_b", "x_b"], lat_b))

    data = data.isel(time=slice(-N_MONTHS, None))
    return data.chunk({"time": 5, "x": 384, "y": 320, "lev": 10})


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


def process_block(data, start, end, grid, deltas, new_dz, new_levels, thk_bnds, thk_centers):
    """Vertically then horizontally regrid one block of time steps."""
    ds = data.isel(time=slice(start, end))
    ds = ds.chunk({"time": 1, "y": ds.y.size, "x": ds.x.size})
    ds = ds.assign_coords(thk_bnds=thk_bnds.transpose("lev", "y", "x"))
    ds = ds.assign_coords(thk_centers=thk_centers.transpose("lev", "y", "x"))

    ds_surface = ds[SURFACE_VARIABLES]
    ds = ds.drop_vars(SURFACE_VARIABLES)

    ds_vert = ((ds * deltas).sum("lev") / new_levels.dz).transpose("time", "lev_new", ...)
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

    fields = list(set(ds_regridded.variables) - set(ds_regridded.coords))
    ds_regridded[fields] = ds_regridded[fields].astype("float32")
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
    data = load_source()
    grid = load_target_grid()
    data, deltas, new_dz, new_levels, thk_bnds, thk_centers = vertical_operator(data)
    operator = (grid, deltas, new_dz, new_levels, thk_bnds, thk_centers)

    if sys.argv[1] == "init":
        init_store(data, process_block(data, 0, 1, *operator))
        return

    begin, finish = int(sys.argv[1]), int(sys.argv[2])
    for j in range(begin // SAMPLES_PER_STEP,
                   int(np.ceil(finish / SAMPLES_PER_STEP))):
        start, end = j * SAMPLES_PER_STEP, (j + 1) * SAMPLES_PER_STEP
        print(start, end, flush=True)
        ds_regridded = process_block(data, start, end, *operator)
        ds_regridded.drop_vars(list(ds_regridded.coords.variables)).to_zarr(
            PATH_OUT, region={"time": slice(start, end)})


if __name__ == "__main__":
    main()
