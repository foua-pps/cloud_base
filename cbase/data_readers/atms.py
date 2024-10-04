from dataclasses import dataclass
import numpy as np
import xarray as xr
from datetime import datetime


ATMS_KEYS = [
    "latitude",
    "longitude",
    "time",
    "tb17",
    "tb18",
    "tb19",
    "tb20",
    "tb21",
    "tb22",
    "view_ang",
]


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
    view_ang: np.ndarray

    @classmethod
    def from_file(cls, atmsfiles: list):
        """read data from netCDF file"""
        atms_data = {key: [] for key in ATMS_KEYS}
        for atmsfile in sorted(atmsfiles):
            with xr.open_dataset(atmsfile) as da:
                atms_data["latitude"].append(da.lat.values)
                atms_data["longitude"].append(da.lon.values % 360)
                atms_data["tb17"].append(da.antenna_temp.values[:, :, 16])
                atms_data["tb18"].append(da.antenna_temp.values[:, :, 17])
                atms_data["tb19"].append(da.antenna_temp.values[:, :, 18])
                atms_data["tb20"].append(da.antenna_temp.values[:, :, 19])
                atms_data["tb21"].append(da.antenna_temp.values[:, :, 20])
                atms_data["tb22"].append(da.antenna_temp.values[:, :, 21])
                atms_data["view_ang"].append(da.view_ang.values)
                atms_time = convert_to_datetime(da.obs_time_utc.values)
                atms_data["time"].append(atms_time)

        return ATMSData(
            np.concatenate(atms_data["latitude"]),
            np.concatenate(atms_data["longitude"]),
            np.concatenate(atms_data["time"]),
            np.concatenate(atms_data["tb17"]),
            np.concatenate(atms_data["tb18"]),
            np.concatenate(atms_data["tb19"]),
            np.concatenate(atms_data["tb20"]),
            np.concatenate(atms_data["tb21"]),
            np.concatenate(atms_data["tb22"]),
            np.concatenate(atms_data["view_ang"]),
        )


def convert_to_datetime(utc_array) -> np.ndarray[datetime]:
    """convert ATMS timestamps to datetime
    ATMS time stamps come as tuples of 8 values
    pertaining to names of the elements of UTC when
    it is expressed as an array of
    integers year,month,day,hour,minute,second,
    millisecond,microsecond
    """

    year = utc_array[:, :, 0].astype(float)
    month = utc_array[:, :, 1].astype(float)
    day = utc_array[:, :, 2].astype(float)
    hour = utc_array[:, :, 3].astype(float)
    minute = utc_array[:, :, 4].astype(float)
    second = utc_array[:, :, 5].astype(float)

    datetime_objects = np.full(year.shape, None, dtype=object)
    nan_mask = (
        np.isnan(year)
        | np.isnan(month)
        | np.isnan(day)
        | np.isnan(hour)
        | np.isnan(minute)
        | np.isnan(second)
    )
    datetime_objects[nan_mask] = np.nan
    valid_mask = ~nan_mask
    # Create datetime objects
    datetime_objects[valid_mask] = np.array(
        [
            datetime(int(y), int(m), int(d), int(h), int(mi), int(s))
            for y, m, d, h, mi, s in zip(
                year[valid_mask],
                month[valid_mask],
                day[valid_mask],
                hour[valid_mask],
                minute[valid_mask],
                second[valid_mask],
            )
        ]
    )
    return datetime_objects
