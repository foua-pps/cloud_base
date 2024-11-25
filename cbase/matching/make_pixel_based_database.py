import numpy as np
import xarray as xr
from xarray import Dataset
from tqdm import tqdm
from typing import List, Tuple, Dict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

N = 5

VGAC_KEYS = [
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
    "satzenith",
    "satazimuth",
]

VGAC_PPS_KEYS = [
    "cth",  # PPS paramters
    "ctp",
    "ctt",
    "ctp_quality",
    "ct",
    "ctp16",
    "cth16",
    "ctt16",
    "ctp84",
    "cth84",
    "ctt84",
    "ct_quality",
    "cmic_phase",
    "cmic_quality",
    "cmic_lwp",
    "cmic_cot",
    "elevation",
    "land_use",
]

ATMS_KEYS = [
    "tb17",  # ATMS parameters
    "tb18",
    "tb19",
    "tb20",
    "tb21",
    "tb22",
    "view_ang",
]


VGAC_KEYS_NEIGHBOURHOOD = [
    "tb11coldest",
    "tb12coldest",
    "tb11warmest",
    "tb12warmest",
    "var_M05",
    "var_M07",
    "var_M12",
    "var_M15",
    "var_M16",
]

ATMS_KEYS_NEIGHBOURHOOD = [
    "var_tb17",
    "var_tb18",
    "var_tb19",
    "var_tb20",
    "var_tb21",
    "var_tb22",
]

ELIGIBLE_VGAC_CHANNELS = [
    "M05",
    "M07",
    "M12",
    "M15",
    "M16",
]

ELIGIBLE_ATMS_CHANNELS = [
    "tb17",
    "tb18",
    "tb19",
    "tb20",
    "tb21",
    "tb22",
]


def cth2asl(
    da: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """convert pps cth to above sea level"""
    da["cth"]["data"] = da["cth"]["data"] - da["elevation"]["data"]
    return da


def find_coldest_warmest_temp_in_neigh(
    cb: np.ndarray, tb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """warmest and coldest pixels in neighborhood"""
    coldest = []
    warmest = []
    imax = cb.shape[0]
    jmax = cb.shape[1]
    indices = np.argwhere(cb[:, :] > 0)
    for ix, iy in indices:
        ix1 = max([0, ix - N])
        ix2 = min([imax, ix + N])
        iy1 = max([0, iy - N])
        iy2 = min([jmax, iy + N])
        coldest.append(tb[ix1:ix2, iy1:iy2].flatten().min())
        warmest.append(tb[ix1:ix2, iy1:iy2].flatten().max())
    return np.array(coldest), np.array(warmest)


def find_variance_in_neigh(ds: Dataset, do_atms) -> Dict[str, float]:
    all_channels = ELIGIBLE_VGAC_CHANNELS
    if do_atms:
        all_channels += ELIGIBLE_ATMS_CHANNELS
    variance = {key: [] for key in all_channels}
    indices = np.argwhere(ds.cloud_base.values[:, :] > 0)
    imax = ds.cloud_base.shape[0]
    jmax = ds.cloud_base.shape[1]
    for ix, iy in indices:
        ix1 = max([0, ix - N])
        ix2 = min([imax, ix + N])
        iy1 = max([0, iy - N])
        iy2 = min([jmax, iy + N])
        for ichannel in all_channels:
            tbs = ds[ichannel][ix1:ix2, iy1:iy2].values
            tbs[tbs < 0] = np.nan
            variance[ichannel].append(np.nanvar(tbs))
    return variance


def make_pixel_dataset(
    input_files: List[str], do_atms: bool = False
) -> Dict[str, Dict[str, np.ndarray]]:
    """make xr dataset with all input parameters in keys"""
    parameter_names = VGAC_KEYS + VGAC_PPS_KEYS
    parameter_neighbourhood = VGAC_KEYS_NEIGHBOURHOOD
    if do_atms:
        parameter_names += ATMS_KEYS
        parameter_neighbourhood += ATMS_KEYS_NEIGHBOURHOOD

    training_data = {key: [] for key in parameter_names}
    for key in parameter_neighbourhood:
        training_data[key] = []

    for input_file in tqdm(input_files[:]):
        with xr.open_dataset(input_file) as ds:
            mask = ds.cloud_base.values > 0
            for key in parameter_names:
                try:
                    training_data[key].append(ds[key].values[mask])
                except KeyError:
                    logging.warning(
                        f"Parameter '{key}' is missing in the dataset."
                    )
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
                training_data[key_name].append(variance[key])

    da = {}
    for key in training_data:
        print(key)
        training_data[key] = np.concatenate(training_data[key])
        print(training_data[key].shape)
        da[key] = {"dims": ("n"), "data": training_data[key]}

    # convert cth above sea level
    da = cth2asl(da)
    return da
