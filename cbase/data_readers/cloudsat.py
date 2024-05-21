import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import pytz
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
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
    validation_height_base: np.array
    validation_height: np.array
    cloud_fraction: np.array
    n_layers: float
    time: np.array
    name: str

    @classmethod
    def from_file(cls, filepath: Path):
        """
        read cloudsat data and
        generate constructor from file
        """
        csat_dict = read_cloudsat_hdf4(filepath.as_posix())

        return cls(
            csat_dict["Longitude"].ravel(),
            csat_dict["Latitude"].ravel(),
            get_base_height(csat_dict["CloudLayerBase"]),
            get_top_height(csat_dict["CloudLayerTop"]),
            get_cloud_fraction(csat_dict["Cloudlayer"]),
            csat_dict["Cloudlayer"].ravel(),
            get_time(csat_dict),
            os.path.basename(filepath),
        )


def read_cloudsat_hdf4(filepath: str) -> dict:
    """access variables in a hdf4 file, and dump them all to a dict"""
    all_data = {}
    h4file = SD(filepath.as_posix(), SDC.READ)
    datasets = h4file.datasets()

    for _, sds in enumerate(datasets.keys()):
        # 2d data
        all_data[sds] = h4file.select(sds).get()

    h4file = HDF(filepath, SDC.READ)
    vs = h4file.vstart()
    data_info_list = vs.vdatainfo()
    for item in data_info_list:
        # 1D data compound/Vdata
        all_data[item[0]] = vs.attach(item[0])[:]

    return all_data


def get_top_height(cth: np.array) -> np.array:
    """get height of highest cloud out of 10 layers"""
    cth_copy = cth.copy()
    top_height = np.ones(len(cth)) * -9
    all_missing = np.all(cth < 0, axis=1)
    valid_indices = np.argwhere(~all_missing)[:, 0]
    cth_copy[cth < 0] = np.nan
    top_height[valid_indices] = np.nanmax(cth_copy[valid_indices, :], axis=1)

    return top_height


def get_base_height(cbh: np.array) -> np.array:
    """get height of lowest cloud out of 10 layers"""
    cbh_copy = cbh.copy()
    base_height = np.ones(len(cbh)) * -9
    all_missing = np.all(cbh < 0, axis=1)
    valid_indices = np.argwhere(~all_missing)[:, 0]
    cbh_copy[cbh < 0] = np.nan
    base_height[valid_indices] = np.nanmin(cbh_copy[valid_indices, :], axis=1)

    return base_height


def get_cloud_fraction(cf: np.array) -> np.array:
    """max cloud fraction in 10 layers"""
    cf_copy = cf.copy()
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
