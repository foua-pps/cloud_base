import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import pytz
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF  # HC
import xarray as xr
import pyhdf.VS
import numpy as np


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
            vis_optical_depth = get_vod_from_dardar(dardar_cloud_file.as_posix())

            return cls(
                csat_dict["Longitude"].ravel(),
                csat_dict["Latitude"].ravel(),
                get_cloud_top(csat_dict),
                csat_dict["LayerBase"][:, 0],  # base height of bottommost layer
                csat_dict["CloudLayers"].ravel(),
                csat_dict["FlagBase"][:, 0],  # which instrument gives base height
                get_cloud_fraction(csat_dict["CloudFraction"]),
                # csat_dict["CloudLayerType"][:, 0],
                vis_optical_depth,
                get_time(csat_dict),
                os.path.basename(cldclass_lidar_file.as_posix()),
            )
        else:
            raise ValueError(
                "Both cldclass_lidar_file and dardar_cloudfile need to be provided"
            )


def get_cloud_top(csat_dict):
    nlayers = csat_dict["CloudLayers"].ravel()
    cth = csat_dict["LayerTop"][:, nlayers]


# def get_cloud_variables(all_data: dict) -> CloudVariables:
#     """get variables from CLDCLASS-LIDAR dataset"""
#     return (
#         all_data["CloudLayerTop"][:, 0],
#         all_data["CloudLayerBase"][:, 0],
#         all_data["Cloudlayer"].ravel(),
#         all_data["CloudLayerType"][:, 0],
#     )


def get_vod_from_dardar(dardarfile: str):
    """get visible optical depth"""
    with xr.open_dataset(dardarfile) as dardar:
        return dardar.vis_optical_depth.values


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


def get_cloud_fraction(cf: np.array) -> np.array:
    """max cloud fraction in multiple layers"""
    cf_copy = cf.astype(float)
    cloud_fraction = np.ones(len(cf)) * -9
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
    """
    convert time from secs to datetime objects
    """

    base_date = datetime.strptime(base_date_string.base_date, "%Y%m%d%H%M").replace(
        tzinfo=pytz.UTC
    )
    return np.array([base_date + timedelta(seconds=value) for value in times])
