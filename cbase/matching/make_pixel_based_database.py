import numpy as np
import xarray as xr
from xarray import Dataset
from tqdm import tqdm
from typing import List, Tuple, Dict
import logging
from scipy.ndimage import uniform_filter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

N = 3  # number of neighbours allowed to get the neighborhood info

VGAC_KEYS = [
    "time",
    "cloud_base",
    "cloud_top",
    "cloud_layers",
    "cloud_base_temp",
    "base_pressure",
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "M06",
    "M07",
    "M08",
    "M09",
    "M10",
    "M11",
    "M12",
    "M13",
    "M14",
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
    "q250",
    "q500",
    "q700",
    "q850",
    "q900",
    "q950",
    "q1000",
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
    "sunzenith",
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
    "tb01",
    "tb02",
    "tb03",
    "tb04",
    "tb05",
    "tb06",
    "tb07",
    "tb08",
    "tb09",
    "tb10",
    "tb11",
    "tb12",
    "tb13",
    "tb14",
    "tb15",
    "tb16",
    "tb17",
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
    "var_M09",
    "var_M11",
    "var_M12",
    "var_M14",
    "var_M15",
    "var_M16",
]

ATMS_KEYS_NEIGHBOURHOOD = [
    "var_tb16",
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
    "M09",
    "M11",
    "M12",
    "M14",
    "M15",
    "M16",
]

ELIGIBLE_ATMS_CHANNELS = [
    "tb16",
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


# def find_coldest_warmest_temp_in_neigh(
#     cb: np.ndarray, tb: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """warmest and coldest pixels in neighborhood"""
#     coldest = []
#     warmest = []
#     imax = cb.shape[0]
#     jmax = cb.shape[1]
#     indices = np.argwhere(cb[:, :] > 0)
#     for ix, iy in indices:
#         ix1 = max([0, ix - N])
#         ix2 = min([imax, ix + N])
#         iy1 = max([0, iy - N])
#         iy2 = min([jmax, iy + N])
#         coldest.append(tb[ix1:ix2, iy1:iy2].flatten().min())
#         warmest.append(tb[ix1:ix2, iy1:iy2].flatten().max())
#     return np.array(coldest), np.array(warmest)


def find_coldest_warmest_temp_in_neigh(
    cb: np.ndarray, tb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the warmest and coldest pixels in the neighborhood
    and their indices."""
    coldest = []
    warmest = []
    coldest_indices = []
    warmest_indices = []

    imax, jmax = cb.shape
    indices = np.argwhere(cb > 0)

    for ix, iy in indices:
        ix1 = max(0, ix - N)
        ix2 = min(imax, ix + N + 1)
        iy1 = max(0, iy - N)
        iy2 = min(jmax, iy + N + 1)

        neighborhood = tb[ix1:ix2, iy1:iy2]

        min_val = np.nanmin(neighborhood)
        max_val = np.nanmax(neighborhood)
        coldest.append(min_val)
        warmest.append(max_val)
        if np.isnan(min_val) or np.isnan(max_val):
            coldest_indices.append([None, None])
            warmest_indices.append([None, None])
        else:
            min_pos = np.argwhere(neighborhood == min_val)[0]
            max_pos = np.argwhere(neighborhood == max_val)[0]
            coldest_indices.append((ix1 + min_pos[0], iy1 + min_pos[1]))
            warmest_indices.append((ix1 + max_pos[0], iy1 + max_pos[1]))
    return (
        np.array(coldest),
        np.array(warmest),
        np.array(coldest_indices),
        np.array(warmest_indices),
    )


def get_values_from_indices(tb: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Values from tb using the given indices"""
    values = []
    for idx in indices:
        if idx[0] is None:
            values.append(np.nan)
        else:
            x, y = idx
            values.append(tb[x, y])
    return np.array(values)


def _textures(array_in: np.array, ks: int = N) -> Dict[str, np.ndarray]:
    """copied from CTTH algorithm
    computes variance and then texture
    """
    n = ks * ks * 1.0
    not_ok = array_in < 0 | np.isnan(array_in)
    true_N = n - n * uniform_filter(
        not_ok.astype(np.float64), size=(ks, ks), mode="reflect"
    )
    true_N[true_N < ks] = ks - 1.0
    array_in[not_ok] = 0.0
    K = np.median(array_in)
    array_in -= K
    array_in[not_ok] = 0.0

    mean = (n / true_N) * uniform_filter(array_in, size=(ks, ks), mode="reflect")
    mean_of_squared = (n / true_N) * uniform_filter(
        array_in**2, size=(ks, ks), mode="reflect"
    )
    squared = mean_of_squared - mean**2

    return np.sqrt(true_N / (true_N - 1) * np.abs(squared))


def find_variance_in_neigh(ds: Dataset, do_atms: bool) -> Dict[str, float]:
    all_channels = ELIGIBLE_VGAC_CHANNELS
    if do_atms:
        all_channels = all_channels + ELIGIBLE_ATMS_CHANNELS
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
            n = np.sum(np.isfinite(tbs))
            texture = np.sqrt(np.abs(np.nanvar(tbs) * (n / (n - 1))))
            variance[ichannel].append(texture)
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
                    logging.warning(f"Parameter '{key}' is missing in the dataset.")
                    continue
                except Exception as e:
                    logging.error(
                        f"An unexpected error occurred while processing '{key}'"
                        + f": {e}"
                    )
                    continue
            cold_tb11, warm_tb11, cold_indices, warm_indices = (
                find_coldest_warmest_temp_in_neigh(ds.cloud_base.values, ds.M15.values)
            )
            cold_tb12 = get_values_from_indices(ds.M16.values, cold_indices)
            warm_tb12 = get_values_from_indices(ds.M16.values, warm_indices)
            training_data["tb11coldest"].append(cold_tb11)
            training_data["tb12coldest"].append(cold_tb12)
            training_data["tb11warmest"].append(warm_tb11)
            training_data["tb12warmest"].append(warm_tb12)

            variance = find_variance_in_neigh(ds, do_atms)
            for key in variance:
                key_name = "var_" + key
                training_data[key_name].append(variance[key])

    da = {}
    for key in training_data:
        print(key, len(training_data[key]))
        training_data[key] = np.concatenate(training_data[key])
        da[key] = {"dims": ("n"), "data": training_data[key]}

    # convert cth above sea level
    da = cth2asl(da)
    return da
