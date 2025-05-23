import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import pytz
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF  # HC
from pyhdf import VS
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

FILL_VALUE = -999.9


@dataclass
class BaseDate:
    """
    string format for base date
    "%Y%m%d%H%M"
    """

    base_date: str

    def __str__(self):
        return self.base_date


@dataclass
class CloudsatData:
    """
    Class to read and handle Cloudsat data
    """

    longitude: np.array
    latitude: np.array
    cloud_top: np.array
    cloud_base: np.array
    cloud_layers: np.array
    flag_base: np.array
    cloud_fraction: np.array
    vis_optical_depth: np.array
    cloud_base_temp: np.array
    time: np.array
    name: str

    @classmethod
    def from_files(cls, cldclass_lidar_file: Path, dardar_cloud_file: Path):
        """
        read cloudsat data and
        generate constructor from file
        """
        if cldclass_lidar_file and dardar_cloud_file:
            csat_dict = read_cloudsat_hdf4(cldclass_lidar_file.as_posix())
            vis_optical_depth, temp_profile, height = get_dardar_parameters(
                dardar_cloud_file.as_posix()
            )
            cloud_base_temp = get_base_temp(
                csat_dict["LayerBase"], temp_profile, height
            )
            return cls(
                csat_dict["Longitude"].ravel() % 360,
                csat_dict["Latitude"].ravel(),
                get_top_height(csat_dict["LayerTop"]),
                get_base_height(csat_dict["LayerBase"]),
                csat_dict["CloudLayers"].ravel(),
                csat_dict["FlagBase"][:, 0],  # which instrument gives base height
                get_cloud_fraction(csat_dict["CloudFraction"]),
                # csat_dict["CloudLayerType"][:, 0],
                vis_optical_depth,
                cloud_base_temp,
                get_time(csat_dict),
                os.path.basename(cldclass_lidar_file.as_posix()),
            )

        raise ValueError(
            "Both cldclass_lidar_file and dardar_cloudfile need to be provided"
        )


def get_top_height(cth: np.array) -> np.array:
    """get height of highest cloud out of n layers"""
    cth_copy = cth.copy()
    top_height = np.ones(len(cth)) * FILL_VALUE
    all_missing = np.all(cth < 0, axis=1)
    valid_indices = np.argwhere(~all_missing)[:, 0]
    top_height[valid_indices] = np.max(cth_copy[valid_indices, :], axis=1)

    return top_height


def get_base_height(cbh: np.array) -> np.array:
    """get height of lowest cloud out of n layers"""
    cbh_copy = np.float32(cbh.copy())
    cbh_copy[cbh_copy < 0] = np.nan
    base_height = np.ones(len(cbh)) * FILL_VALUE
    all_missing = np.all(cbh < 0, axis=1)
    valid_indices = np.argwhere(~all_missing)[:, 0]
    cbh_copy[cbh_copy < 0] = np.nan
    base_height[valid_indices] = np.nanmin(cbh_copy[valid_indices, :], axis=1)
    return base_height


def get_base_temp(cbh_profile, temp_profile, height):
    cbh = get_base_height(cbh_profile)
    return np.array(
        [
            interp1d(height, temp_profile[i, :])(cbh[i])
            for i in range(temp_profile.shape[0])
        ]
    )


def read_cloudsat_hdf4(filepath: str) -> dict:
    """access variables in a hdf4 file, and dump them all to a dict"""
    all_data = {}
    h4file = SD(filepath, SDC.READ)
    datasets = h4file.datasets()

    for _, sds in enumerate(datasets.keys()):
        all_data[sds] = np.array(h4file.select(sds).get())

    h4file = HDF(filepath, SDC.READ)
    vs = h4file.vstart()
    data_info_list = vs.vdatainfo()
    for item in data_info_list:
        # 1D data compound/Vdata
        name = item[0]
        all_data[name] = np.array(vs.attach(name)[:])

    return all_data


def get_dardar_parameters(dardarfile: str):
    with xr.open_dataset(dardarfile) as dardar:
        return (
            dardar.vis_optical_depth.values,
            dardar.temperature.values,
            dardar.height.values,
        )


def get_cloud_fraction(cf: np.array) -> np.array:
    """max cloud fraction in multiple layers"""
    cf_copy = cf.astype(float)
    cloud_fraction = np.ones(len(cf)) * FILL_VALUE
    all_missing = np.all(cf < 0, axis=1)
    valid_indices = np.argwhere(~all_missing)[:, 0]
    cf_copy[cf < 0] = np.nan
    cloud_fraction[valid_indices] = np.nanmax(cf_copy[valid_indices, :], axis=1)

    return cloud_fraction


def get_time(all_data):
    """convert time from TAI units to datetime"""
    dsec = time.mktime((1993, 1, 1, 0, 0, 0, 0, 0, 0)) - time.timezone
    sec_1970 = all_data["Profile_time"].ravel() + all_data["TAI_start"].ravel() + dsec

    times = convert2datetime(sec_1970, BaseDate("197001010000"))
    return times


def convert2datetime(times: np.array, base_date_string: BaseDate) -> np.array:
    """convert time from secs to datetime objects"""

    base_date = datetime.strptime(base_date_string.base_date, "%Y%m%d%H%M").replace(
        tzinfo=pytz.UTC
    )
    return np.array([base_date + timedelta(seconds=value) for value in times])
