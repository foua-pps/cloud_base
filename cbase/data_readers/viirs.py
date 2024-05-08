import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from satpy import Scene
from cbase.utils.utils import datetime64_to_datetime


VGAC_PARAMETER_LIST = [
    "latitude",
    "longitude",
    "scanline_timestamps",
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
]


@dataclass
class VGACData:
    """
    A container for VGAC data read in using satpy
    """

    latitude: np.ndarray
    longitude: np.ndarray
    time: np.ndarray
    M01: np.ndarray
    M02: np.ndarray
    M03: np.ndarray
    M04: np.ndarray
    M05: np.ndarray
    M06: np.ndarray
    M07: np.ndarray
    M08: np.ndarray
    M09: np.ndarray
    M10: np.ndarray
    M11: np.ndarray
    M12: np.ndarray
    M13: np.ndarray
    M14: np.ndarray
    M15: np.ndarray
    M16: np.ndarray
    validation_height_base: np.ndarray
    name: str

    @classmethod
    def from_file(cls, filepath: Path):
        """alternative constructor from file"""
        scn = Scene(reader="viirs_vgac_l1c_nc", filenames=[filepath])
        scn.load(VGAC_PARAMETER_LIST)
        d = scn.to_xarray()
        return cls(
            d.latitude.values,
            d.longitude.values % 360,
            datetime64_to_datetime(d.scanline_timestamps.values),
            d.M01.values,
            d.M02.values,
            d.M03.values,
            d.M04.values,
            d.M05.values,
            d.M06.values,
            d.M07.values,
            d.M08.values,
            d.M09.values,
            d.M10.values,
            d.M11.values,
            d.M12.values,
            d.M13.values,
            d.M14.values,
            d.M15.values,
            d.M16.values,
            np.zeros_like(d.latitude.values),  # initialise the base height,
            os.path.basename(filepath),
        )


# @dataclass
# class VGACData:
#     """
#     Class to read and handle VGAC data
#     """

#     longitude: np.ndarray
#     latitude: np.ndarray
#     validation_height_base: np.ndarray
#     time: np.ndarray


# def read_vgac(filepath: Path) -> VGACData:
#     """read data from netCDF file"""
#     with xr.open_dataset(filepath, decode_times=False) as da:
#         time = convert_time2datetime(da)
#         validation_height_base = -999.9 * np.ones_like(da.lat.values)
#         vgac = VGACData(da.lon.values, da.lat.values, validation_height_base, time)
#         return vgac


# def convert_time2datetime(
#     da: Dataset, base_date_string=BaseDate("201001010000")
# ) -> datetime:
#     """
#     required conversion of time,
#     time in VGAC is supplied as days from 20100101
#     """
#     base_date = datetime.strptime(base_date_string.base_date, "%Y%m%d%H%M") + timedelta(
#         days=da.proj_time0.values.item()
#     )
#     return np.array([base_date + timedelta(hours=value) for value in da.time.values])
