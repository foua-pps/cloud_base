import pygrib
from pathlib import Path
import numpy as np
from cbase.utils.utils import create_dataset

era5file = "/home/a002602/data/cloud_base/NWP/GAC_ECMWF_ERA5_201801030300+000H00M"


class Era5:
    """data reader for ERA5 data in grib format"""

    def __init__(self, filename: Path):
        self.grbs = pygrib.open(filename)

        self.tclw = self._load_tclw()

    def _load_tclw(self):
        grb = self.grbs.select(name="Total column cloud liquid water")[0]
        lats, lons = grb.latlons()
        return create_dataset(lats, lons, grb.values, "tclw")
