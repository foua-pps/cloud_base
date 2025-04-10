import os
from typing import Dict
from dataclasses import dataclass
import xarray as xr
from xarray import Dataset, DataArray
import numpy as np
from pathlib import Path
import glob
import pytz
from datetime import datetime, timedelta
from atrain_match.utils.match import match_lonlat
from scipy.interpolate import LinearNDInterpolator
from cbase.data_readers.atms import ATMSData
from cbase.matching.config import ATMS_PARAMETERS
from cbase.utils.utils import (
    extract_timestamp_from_atms_filename,
    check_lon_range,
    adapt_lonrange,
)

VGAC_ATMS_TDIFF = 40  # minutes
LONRANGE = {"-180--180": 1, "0--360": 2}


@dataclass
class LatLonBox:
    """class to define the extent of a lat lon to crop
    images"""

    lower_lat: float
    upper_lat: float
    left_lon: float
    right_lon: float


class MatchATMSVGAC:
    """class to match ATMS data to VGAC scenes created for CNN"""

    def __init__(self, vgacfile: Path, atmspath: Path, outpath: Path):
        self.vgac_file = vgacfile
        self.outfile = os.path.join(outpath, os.path.basename(vgacfile))

        self.vgac = xr.open_dataset(vgacfile.as_posix())
        self.atms_files = self.find_matching_atms(atmspath)
        if len(self.atms_files) == 0:
            self.atms = None
            print(f"No match found for {vgacfile}")
        else:
            self.atms = ATMSData.from_file(self.atms_files)

    def find_matching_atms(self, atmspath: Path) -> list:
        """find matching ATMS files within +- TDIFF of VGAC file"""
        vgc_time = datetime.fromtimestamp(self.vgac.time.values[0, 0], pytz.utc)
        time_string = (
            f"{vgc_time.year}{vgc_time.strftime('%m')}{vgc_time.strftime('%d')}T"
        )

        atmsfiles = sorted(glob.glob(f"{atmspath}/*{time_string}*"))
        date_str = np.array(
            [extract_timestamp_from_atms_filename(atmsfile) for atmsfile in atmsfiles]
        )

        tdiff_mask = (date_str > vgc_time - timedelta(minutes=VGAC_ATMS_TDIFF)) & (
            date_str < vgc_time + timedelta(minutes=VGAC_ATMS_TDIFF)
        )

        if tdiff_mask is None:
            return np.array([])

        return np.array(atmsfiles)[tdiff_mask]

    def matching(self):
        """match the ATMS data to VGAC grid and write out a netcdf file"""
        if self.atms is None:
            interpolated_atms = self.add_fillvalue_atms_data()
        else:
            lon_string = check_lon_range(self.vgac.longitude.values)
            self.vgac["longitude"] = adapt_lonrange(self.vgac.longitude, lon_string)
            self.atms.longitude = adapt_lonrange(self.atms.longitude, lon_string)

            latlon_box = self.get_latlon_bounds()
            latlon_mask = self.get_latlon_mask(latlon_box)

            try:
                matcher_mask = self.get_closest_matches(latlon_mask)
            except Exception as e:
                print(f"Problem with finding matches {e}")
                interpolated_atms = self.add_fillvalue_atms_data()

            try:
                interpolated_atms = self.interpolate_atms2vgac(
                    latlon_mask, matcher_mask
                )
            except Exception as e:
                print(f"Problem with interpolation {e}")
                interpolated_atms = self.add_fillvalue_atms_data()

        self.vgac = add2vgac_dataset(self.vgac, interpolated_atms)
        self.vgac.to_netcdf(self.outfile)

    def get_latlon_bounds(self, buffer=0) -> LatLonBox:
        """gives lat lon bounds of where VGAC scene is located"""
        return LatLonBox(
            self.vgac.latitude.values.min() - buffer,
            self.vgac.latitude.values.max() + buffer,
            (self.vgac.longitude.values).min() - buffer,
            (self.vgac.longitude.values).max() + buffer,
        )

    def get_latlon_mask(self, latlon_box: LatLonBox) -> np.ndarray[np.bool_]:
        """gives a mask for ATMS data according to VGAC bounds"""
        return (
            (self.atms.latitude >= latlon_box.lower_lat)
            & (self.atms.latitude <= latlon_box.upper_lat)
            & (self.atms.longitude >= latlon_box.left_lon)
            & (self.atms.longitude <= latlon_box.right_lon)
        )

    def get_closest_matches(self, latlon_mask: np.ndarray) -> np.ndarray[np.bool_]:
        """find closest matches of ATMS with VGAc using atrain _match"""
        source = (self.vgac.longitude.values, self.vgac.latitude.values)
        target = (
            self.atms.longitude[latlon_mask],
            self.atms.latitude[latlon_mask],
        )

        _, distances = match_lonlat(
            source, target, n_neighbours=2, radius_of_influence=1000 * 8
        )
        return np.any(distances > 0, axis=1)

    def interpolate_atms2vgac(self, latlon_mask, matcher_mask) -> Dict[str, np.ndarray]:
        """interpolates the ATMS TBs to VGAC grid"""
        interpolated_atms = {}
        for parameter in ATMS_PARAMETERS:
            values = getattr(self.atms, parameter)[latlon_mask][matcher_mask]
            points = np.array(
                [
                    self.atms.longitude[latlon_mask][matcher_mask],
                    self.atms.latitude[latlon_mask][matcher_mask],
                ]
            ).T

            interpolated_atms[parameter] = LinearNDInterpolator(points, values)(
                np.stack((self.vgac.longitude, self.vgac.latitude), axis=-1)
            )
        return interpolated_atms

    def add_fillvalue_atms_data(self) -> Dict[str, np.ndarray]:
        """if no ATMS files are present, fillvalues are written to final file"""
        fill_value_atms = {}
        for parameter in ATMS_PARAMETERS:
            fill_value_atms[parameter] = -999.9 * np.ones_like(self.vgac.latitude)
        return fill_value_atms


def make_dataarray(data, dims, coords) -> DataArray:
    """make a xarray dataarray"""
    return xr.DataArray(data=data, dims=dims, coords=coords)


def add2vgac_dataset(vgac, interpolated_atms) -> Dataset:
    """add ATMS paramters to existing VGAC dataset"""
    dims = ["npix", "nscan"]
    coords = dict(
        npix=(["npix"], vgac.npix.values),
        nscan=(["nscan"], vgac.nscan.values),
    )
    for key in interpolated_atms.keys():
        vgac[key] = make_dataarray(interpolated_atms[key], dims, coords)
    return vgac
