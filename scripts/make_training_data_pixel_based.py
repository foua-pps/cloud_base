import os
from random import shuffle
import numpy as np
import xarray as xr
from glob import glob
from tqdm import tqdm

input_path = "/nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/test/split/"
input_path1 = "/nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/split/"

keys = [
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
    "q250",
    "q500",
    "q700",
    "q850",
    "q900",
    "q950",
    "q1000",
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
    #    "cth",
    #    "ctp",
    #    "ctt",
    #    "ctp_quality",
    #    "ct",
    #    "ct_quality",
    #    "cmic_phase",
    #    "cmic_quality",
    #    "cmic_lwp#",
    #    "cmic_cot",
    #    "elevation",
    #    "land_use",
    "tb11coldest",
    "tb12coldest",
    "tb11warmest",
    "tb12warmest",
]


def cth2asl(da):
    da["cth"]["data"] = da["cth"]["data"] - da["elevation"]["data"]
    return da


def find_coldest_warmest_temp_in_neigh(cb, tb):
    """..."""
    coldest = []
    warmest = []
    npix = tb.shape[0]
    nscan = tb.shape[1]

    indices = np.argwhere(cb[:, :] > 0)
    for ix, iy in indices:
        ix1 = max([0, ix - 5])
        ix2 = min([npix, ix + 5])
        iy1 = max([0, iy - 5])
        iy2 = min([nscan, iy + 5])
        # ix1 = max([0, ix - 5])
        # ix2 = min([100, ix + 5])
        # iy1 = max([0, iy - 5])
        # iy2 = min([20, iy + 5])
        coldest.append(tb[ix1:ix2, iy1:iy2].flatten().min())
        warmest.append(tb[ix1:ix2, iy1:iy2].flatten().max())
    return np.array(coldest), np.array(warmest)


def make_pixel_dataset(input_files):
    training_data = {key: [] for key in keys}
    print(training_data)
    print("length of input files", len(input_files))
    for input_file in tqdm(input_files[:]):
        with xr.open_dataset(input_file) as ds:
            # ds = ds.sel(
            # {
            #     "npix": ds["npix"].values[8:-8],
            #     "nscan": ds["nscan"].values[8:-8],
            # }
            # ).load()
            mask = ds.cloud_base.values > 0

            for key in keys[:-4]:
                training_data[key].append(ds[key].values[mask])

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

    da = {}
    for key in training_data:
        training_data[key] = np.concatenate(training_data[key])
        da[key] = {"dims": ("n"), "data": training_data[key]}

    # convert cth above sea level
    #    da = cth2asl(da)
    return da


input_files = glob(input_path + "/cnn_data*nc")
# input_files1 = glob(input_path1 + "/cnn_data*nc")
print(input_files)
# input_files = input_files[0:22000]
input_files = input_files
shuffle(input_files)
s = len(input_files)
s1 = int(s * 0.5)
s2 = int(s * 0.25)
s3 = int(s * 0.25)

# da = make_pixel_dataset(input_files[:s1])
# xr.Dataset.from_dict(da).to_netcdf(os.path.join(input_path, "training_data_vgac_cloudsat_caliop_pixel_pps_5min.nc"))

# da = make_pixel_dataset(input_files[s1:s1+s2])
# xr.Dataset.from_dict(da).to_netcdf(os.path.join(input_path, "validation_data_vgac_cloudsat_caliop_pixel_pps_5min.nc"))

da = make_pixel_dataset(input_files[s1 + s2 :])
xr.Dataset.from_dict(da).to_netcdf(
    os.path.join(input_path, "testing_data_vgac_cloudsat_caliop_pixel_pps_5min.nc")
)
