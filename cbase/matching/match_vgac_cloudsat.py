from datetime import timedelta
from typing import Union
import numpy as np
import xarray as xr
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData
from cbase.utils.utils import haversine_distance
from .config import I1, I2, COLLOCATION_THRESHOLD, TIME_WINDOW


class DataMatcher:
    """Class to match VGAC and CLOUDSAT data"""

    def __init__(self, cloudsat: CloudsatData, vgac: VGACData, era5: xr.Dataset):
        self.cloudsat = cloudsat
        self.vgac = vgac
        self.era5 = era5

        self.count_collocations = np.zeros_like(self.vgac.latitude)

        if not self.check_overlapping_time():
            raise ValueError("The two passes are not at same time")

    def check_overlapping_time(self) -> tuple[list, list]:
        """
        check if the two satellites passes are overlapping
        """
        t1 = self.cloudsat.time[0]  # start time
        t2 = self.cloudsat.time[-1]  # end time
        if t1 > t2:
            raise ValueError("start time cannot be after end time")
        for dt in self.vgac.time:
            if t1 <= dt <= t2:
                return True
        return False

    def _get_ascending_descending_orbit(self):
        """divide the complete orbit to Ascending and Descending nodes"""

        cld_lat_diff = np.diff(
            self.cloudsat.latitude, prepend=self.cloudsat.latitude[0]
        )
        vgc_lat_diff = np.diff(
            self.vgac.latitude[:, 400], prepend=self.vgac.latitude[0, 400]
        )
        return (
            cld_lat_diff < 0,
            ~(cld_lat_diff < 0),
            vgc_lat_diff < 0,
            ~(vgc_lat_diff < 0),
        )

    def _broadcast_arrays(self, i: int, icld):
        """
        broadcast lat/lon arrays to 2d to enable vectorized calculation
        of distances
        """
        icld1, icld2 = icld[0], icld[1]
        lon_c = np.broadcast_to(
            self.cloudsat.longitude[icld1:icld2].reshape(-1, 1),
            (
                self.cloudsat.longitude[icld1:icld2].shape[0],
                self.vgac.longitude[i, I1:I2].shape[0],
            ),
        )
        lat_c = np.broadcast_to(
            self.cloudsat.latitude[icld1:icld2].reshape(-1, 1),
            (
                self.cloudsat.latitude[icld1:icld2].shape[0],
                self.vgac.latitude[i, I1:I2].shape[0],
            ),
        )
        lon_v = np.broadcast_to(
            self.vgac.longitude[i, I1:I2].reshape(1, -1),
            (
                self.cloudsat.longitude[icld1:icld2].shape[0],
                self.vgac.longitude[i, I1:I2].shape[0],
            ),
        )
        lat_v = np.broadcast_to(
            self.vgac.latitude[i, I1:I2].reshape(1, -1),
            (
                self.cloudsat.latitude[icld1:icld2].shape[0],
                self.vgac.latitude[i, I1:I2].shape[0],
            ),
        )
        return lon_c, lat_c, lon_v, lat_v

    def _process_matching_iteration(self, i: int, icld: tuple[int, int]):
        """the matching process is run for each VGAC scan,
        Cloudsat pixels within radius given by COLLOCATION_THRESHOLD
        are averaged to give the cloud base height at eligible pixels of
        VGAC
        """

        lon_c, lat_c, lon_v, lat_v = self._broadcast_arrays(i, icld)

        # calculate haversine distance
        distances = haversine_distance(lat_v, lon_v, lat_c, lon_c)

        # find args where distance is below threshold
        x_argmin, y_argmin = np.where(distances <= COLLOCATION_THRESHOLD)

        icld1, icld2 = icld[0], icld[1]
        # check tdiff between cloudsat and VGAC collocations
        tdiff = self.cloudsat.time[icld1:icld2][x_argmin] - self.vgac.time[i]
        tdiff_minutes = np.array([t.seconds / 60 for t in tdiff])

        # get x and y indices
        valid_indices = np.where(
            (self.cloudsat.validation_height_base[icld1:icld2][x_argmin] > 0)
            & (tdiff_minutes < 20.0)
        )[0]

        # update base height and count number of cloudsat obs used for each VGAC pixel
        for valid_index in valid_indices:
            self.vgac.validation_height_base[i, I1:I2][
                y_argmin[valid_index]
            ] += self.cloudsat.validation_height_base[icld1:icld2][
                x_argmin[valid_index]
            ]
            self.count_collocations[i, I1:I2][y_argmin[valid_index]] += 1

    def _get_closest_cloudsat_guess(self, itime: int) -> Union[tuple[int, int], None]:
        """
        select part of Cloudsat track crossing the selected VGAC pixel
        """
        tmask = (
            self.cloudsat.time
            >= self.vgac.time[itime] + timedelta(minutes=TIME_WINDOW[0])
        ) & (
            self.cloudsat.time
            <= self.vgac.time[itime] + timedelta(minutes=TIME_WINDOW[1])
        )
        if np.all(~tmask):
            return None
        else:
            distance = haversine_distance(
                self.vgac.latitude[itime, 400],
                self.vgac.longitude[itime, 400] % 360,
                self.cloudsat.latitude[tmask],
                self.cloudsat.longitude[tmask] % 360,
            )
            argmin = np.argmin(distance)
            index = np.arange(0, len(self.cloudsat.time), 1)[tmask][argmin]
            return (index - 25, index + 25)

    def match_vgac_cloudsat(self):
        """
        For each VGAC scan, matches from Cloudsat are found
        """

        for itime in range(len(self.vgac.time)):

            if self._get_closest_cloudsat_guess(itime) is None:
                continue
            else:
                icld1, icld2 = self._get_closest_cloudsat_guess(itime)
                if np.all(self.cloudsat.validation_height_base[icld1:icld2] < 0):
                    continue
                else:
                    self._process_matching_iteration(itime, [icld1, icld2])
