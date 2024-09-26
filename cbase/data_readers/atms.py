import os
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
import numpy as np
from satpy import Scene
import xarray as xr
import re
from datetime import datetime
from cbase.utils.utils import datetime64_to_datetime, microseconds_to_datetime

atms_data = {
    "latitude" : [],
    "longitude" : [],
    "time": [],
    "tb17" : [],
    "tb18": [],
    "tb19": [],
    "tb20": [],
    "tb21": [],
    "tb22": []
}

@dataclass
class ATMSData:
    """
    A container for ATMS data
    Can read in one file or multiple and concatenate the files together
    Concatenation option for example helps to read in data for one day together
    or multiple swaths from same hour together
    """

    latitude: np.ndarray
    longitude: np.ndarray
    time: np.ndarray
    tb17: np.ndarray
    tb18: np.ndarray
    tb19: np.ndarray
    tb20: np.ndarray
    tb21: np.ndarray
    tb22: np.ndarray
        
    @classmethod
    def from_file(cls, atmsfiles: list):
        """read data from netCDF file"""
        for atmsfile in sorted(atmsfiles):
            print(atmsfile)
            with xr.open_dataset(atmsfile) as da:
                atms_data["latitude"].append(da.lat.values)
                atms_data["longitude"].append(da.lon.values)
                atms_data["tb17"].append(da.antenna_temp.values[:, :, 16])
                atms_data["tb18"].append(da.antenna_temp.values[:, :, 17])
                atms_data["tb19"].append(da.antenna_temp.values[:, :, 18])
                atms_data["tb20"].append(da.antenna_temp.values[:, :, 19])
                atms_data["tb21"].append(da.antenna_temp.values[:, :, 20])
                atms_data["tb22"].append(da.antenna_temp.values[:, :, 21])
                atms_time = convert_to_datetime(da.obs_time_utc.values)
                atms_data["time"].append(atms_time)

        return ATMSData(
            np.concatenate(atms_data["latitude"]),
            np.concatenate(atms_data["longitude"]),
            np.concatenate(atms_data["tb17"]),
            np.concatenate(atms_data["tb18"]),
            np.concatenate(atms_data["tb19"]),
            np.concatenate(atms_data["tb20"]),
            np.concatenate(atms_data["tb21"]),
            np.concatenate(atms_data["tb22"]),
            np.concatenate(atms_data["time"])              
        )        
                

def convert_to_datetime(utc_array) -> np.ndarray:
    """convert ATMS timestamps to datetime
    ATMS time stamps come as tuples of 8 values
    pertaining to names of the elements of UTC when
    it is expressed as an array of 
    integers year,month,day,hour,minute,second,
    millisecond,microsecond
    """
    year = utc_array[:, :, 0].astype(int)
    month = utc_array[:, :, 1].astype(int)
    day = utc_array[:, :, 2].astype(int)
    hour = utc_array[:, :, 3].astype(int)
    minute = utc_array[:, :, 4].astype(int)
    second = utc_array[:, :, 5].astype(int)
    nrows = utc_array.shape[0]
    ncols = utc_array.shape[1]

    # Create datetime objects
    datetime_objects = np.array(
        [[datetime(year[i, j], month[i, j], day[i, j],
                    hour[i, j], minute[i, j], second[i, j])
          for j in range(ncols)] for i in range(nrows)]
    )
    
    return datetime_objects




