import pygrib
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from cbase.utils.utils import create_dataset

era5file = "/home/a002602/data/cloud_base/NWP/GAC_ECMWF_ERA5_201801030300+000H00M"


@dataclass
class Era5:
    """data reader for ERA5 data in grib format"""

    tclw: np.ndarray
    t: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    @classmethod
    def from_file(cls, filepath: Path):
        grbs = pygrib.open(filepath)
        grb = grbs.select(name="Total column cloud liquid water")[0]
        lats, lons = grb.latlons()
        return cls(grb.values, grb.values * 100, lats, lons)
