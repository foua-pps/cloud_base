from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import xarray as xr
from xarray.core.dataset import Dataset
import numpy as np
from cbase.utils.utils import BaseDate


@dataclass
class VGACData:
    """
    Class to read and handle VGAC data
    """

    longitude: np.ndarray
    latitude: np.ndarray
    validation_height_base: np.ndarray
    time: np.ndarray


def read_vgac(filepath: Path) -> VGACData:
    """read data from netCDF file"""
    with xr.open_dataset(filepath, decode_times=False) as da:
        time = convert_time2datetime(da)
        validation_height_base = -999.9 * np.ones_like(da.lat.values)
        vgac = VGACData(da.lon.values, da.lat.values, validation_height_base, time)
        return vgac


def convert_time2datetime(
    da: Dataset, base_date_string=BaseDate("201001010000")
) -> datetime:
    """
    required conversion of time,
    time in VGAC is supplied as days from 20100101
    """
    base_date = datetime.strptime(base_date_string.base_date, "%Y%m%d%H%M") + timedelta(
        days=da.proj_time0.values.item()
    )
    return np.array([base_date + timedelta(hours=value) for value in da.time.values])
