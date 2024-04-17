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

    longitude: float
    latitude: float
    time: float


def read_data(filepath: Path) -> VGACData:
    """read data from netCDF file"""
    with xr.open_dataset(filepath) as da:
        time = convert_time2datetime(da)

        vgac = VGACData(da.longitude.values, da.latitude.values, time)
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
