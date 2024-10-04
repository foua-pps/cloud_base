import numpy as np
import xarray as xr
from xarray import Dataset
from tqdm import tqdm
from typing import List, Tuple, Dict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


KEYS = [
    "time",
    "cloud_base",
    "cloud_top",
    "cloud_layers",
    "M05",
    "M07",
    "M12",
    "M15",
    "M16",
    "z_surface",
    "p_surface",
    "t250",
    "t500",
    "t700",
    "t850",
    "t900",
    "t950",
    "t1000",
    "rh250",
    "rh500",
    "rh700",
    "rh850",
    "rh900",
    "rh950",
    "rh1000",
    "h_2meter",
    "ciwv",
    "tclw",
    "cloud_fraction",
    "flag_base",
    "vis_optical_depth",
    "latitude",
    "longitude",
    "cth",  # PPS paramters
    "ctp",
    "ctt",
    "ctp_quality",
    "ct",
    "ct_quality",
    "cmic_phase",
    "cmic_quality",
    "cmic_lwp",
    "elevation",
    "land_use",
    "tb17",  # ATMS parameters
    "tb18",
    "tb19",
    "tb20",
    "tb21",
    "tb22",
    "view_ang",
]

KEYS_NEIGHBOURHOOD = [
    "tb11coldest",
    "tb12coldest",
    "tb11warmest",
    "tb12warmest",
    "var_M05",
    "var_M07",
    "var_M12",
    "var_M15",
    "var_M16",
    "var_tb17",
    "var_tb18",
    "var_tb19",
    "var_tb20",
    "var_tb21",
    "var_tb22",
]

ELIGIBLE_CHANNELS = [
    "M05",
    "M07",
    "M12",
    "M15",
    "M16",
    "tb17",
    "tb18",
    "tb19",
    "tb20",
    "tb21",
    "tb22",
]


def cth2asl(da: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    """convert pps cth to above sea level"""
    da["cth"]["data"] = da["cth"]["data"] - da["elevation"]["data"]
    return da


def find_coldest_warmest_temp_in_neigh(
    cb: np.ndarray, tb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """warmest and coldest pixels in neighborhood"""
    coldest = []
    warmest = []

    indices = np.argwhere(cb[:, :] > 0)
    for ix, iy in indices:
        ix1 = max([0, ix - 5])
        ix2 = min([100, ix + 5])
        iy1 = max([0, iy - 5])
        iy2 = min([20, iy + 5])
        coldest.append(tb[ix1:ix2, iy1:iy2].flatten().min())
        warmest.append(tb[ix1:ix2, iy1:iy2].flatten().max())
    return np.array(coldest), np.array(warmest)


def find_variance_in_neigh(ds: Dataset) -> Dict[str, float]:
    variance = {}
    indices = np.argwhere(ds.cloud_base.values[:, :] > 0)
    for ix, iy in indices:
        ix1 = max([0, ix - 5])
        ix2 = min([100, ix + 5])
        iy1 = max([0, iy - 5])
        iy2 = min([20, iy + 5])
        for ichannel in ELIGIBLE_CHANNELS:
            tbs = ds[ichannel][ix1:ix2, iy1:iy2].values
            tbs[tbs < 0] = np.nan
            variance[ichannel] = np.nanvar(tbs)
    return variance


def make_pixel_dataset(input_files: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """make xr dataset with all input parameters in keys"""
    training_data = {key: [] for key in KEYS}
    for key in KEYS_NEIGHBOURHOOD:
        training_data[key] = []

    for input_file in tqdm(input_files[:]):
        with xr.open_dataset(input_file) as ds:
            # ds = ds.sel(
            # {
            #     "npix": ds["npix"].values[8:-8],
            #     "nscan": ds["nscan"].values[8:-8],
            # }
            # ).load()
            mask = ds.cloud_base.values > 0
            for key in KEYS:
                try:
                    training_data[key].append(ds[key].values[mask])
                except KeyError:
                    logging.warning(f"Parameter '{key}' is missing in the dataset.")
                    continue
                except Exception as e:
                    logging.error(
                        f"An unexpected error occurred while processing '{key}': {e}"
                    )
                    continue
            cold_tb11, warm_tb11 = find_coldest_warmest_temp_in_neigh(
                ds.cloud_base.values, ds.M15.values
            )
            cold_tb12, warm_tb12 = find_coldest_warmest_temp_in_neigh(
                ds.cloud_base.values, ds.M16.values
            )

            training_data["tb11coldest"].append(cold_tb11)
            training_data["tb12coldest"].append(cold_tb12)
            training_data["tb11warmest"].append(warm_tb11)
            training_data["tb12warmest"].append(warm_tb12)

            variance = find_variance_in_neigh(ds)
            for key in variance:
                key_name = "var_" + key
                print(variance[key])
                training_data[key_name].append(variance[key])

    da = {}
    print(training_data)
    for key in training_data:
        print(key)
        training_data[key] = np.concatenate(training_data[key])
        da[key] = {"dims": ("n"), "data": training_data[key]}

    # convert cth above sea level
    da = cth2asl(da)
    return da
