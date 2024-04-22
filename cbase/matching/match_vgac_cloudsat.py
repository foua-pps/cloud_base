from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData
from cbase.utils.utils import haversine_distance

I1, I2 = 300, 400


class DataMatcher:
    """ """

    def __init__(self, cloudsat: CloudsatData, vgac: VGACData):
        self.cloudsat = cloudsat
        self.vgac = vgac

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

    def match_vgac_cloudsat(self):
        """ """
        for itime in enumerate(self.vgac.time):
            tmask_c = (
                self.cloudsat.time >= self.vgac.time[itime] + timedelta(minutes=10)
            ) & (self.cloudsat.time <= self.vgac.time[itime] + timedelta(minutes=15))
            if np.sum(tmask_c) == 0:
                continue
            else:
                self._process_matching_iteration(itime, tmask_c)

    def _process_matching_iteration(self, i: int, tmask_c: bool):

        def _broadcast_arrays(self):
            lon_c = np.broadcast_to(
                self.cloudsat.longitude[tmask_c].reshape(-1, 1),
                (
                    self.cloudsat.longitude[tmask_c].shape[0],
                    self.vgac.longitude[i, i1:i2].shape[0],
                ),
            )
            lat_c = np.broadcast_to(
                self.cloudsat.latitude[tmask_c].reshape(-1, 1),
                (
                    self.cloudsat.latitude[tmask_c].shape[0],
                    self.vgac.latitude[i, i1:i2].shape[0],
                ),
            )
            lon_v = np.broadcast_to(
                self.vgac.longitude[i, i1:i2].reshape(1, -1),
                (
                    self.cloudsat.longitude[tmask_c].shape[0],
                    self.vgac.longitude[i, i1:i2].shape[0],
                ),
            )
            lat_v = np.broadcast_to(
                self.vgac.latitude[i, i1:i2].reshape(1, -1),
                (
                    self.cloudsat.latitude[tmask_c].shape[0],
                    self.vgac.latitude[i, i1:i2].shape[0],
                ),
            )
            return lon_c, lat_c, lon_v, lat_v

        lon_c, lat_c, lon_v, lat_v = self._broadcast_arrays()
        distances = haversine_distance(lat_v, lon_v, lat_c, lon_c)
        min_index = np.unravel_index(np.argmin(distances), distances.shape)

        x_arg_min, y_arg_min = min_index
