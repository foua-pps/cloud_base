import numpy as np
from datetime import timedelta
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData
from cbase.utils.utils import haversine_distance
from .config import I1, I2, COLLOCATION_THRESHOLD, TIME1, TIME2


class DataMatcher:
    """Class to match VGAC and CLOUDSAT data"""

    def __init__(self, cloudsat: CloudsatData, vgac: VGACData):
        self.cloudsat = cloudsat
        self.vgac = vgac

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

    def _broadcast_arrays(self, i: int, tmask_c: bool):
        """
        broadcast lat/lon arrays to 2d to enable vectorized calculation
        of distances
        """
        lon_c = np.broadcast_to(
            self.cloudsat.longitude[tmask_c].reshape(-1, 1),
            (
                self.cloudsat.longitude[tmask_c].shape[0],
                self.vgac.longitude[i, I1:I2].shape[0],
            ),
        )
        lat_c = np.broadcast_to(
            self.cloudsat.latitude[tmask_c].reshape(-1, 1),
            (
                self.cloudsat.latitude[tmask_c].shape[0],
                self.vgac.latitude[i, I1:I2].shape[0],
            ),
        )
        lon_v = np.broadcast_to(
            self.vgac.longitude[i, I1:I2].reshape(1, -1),
            (
                self.cloudsat.longitude[tmask_c].shape[0],
                self.vgac.longitude[i, I1:I2].shape[0],
            ),
        )
        lat_v = np.broadcast_to(
            self.vgac.latitude[i, I1:I2].reshape(1, -1),
            (
                self.cloudsat.latitude[tmask_c].shape[0],
                self.vgac.latitude[i, I1:I2].shape[0],
            ),
        )
        return lon_c, lat_c, lon_v, lat_v

    def _process_matching_iteration(self, i: int, tmask_c: bool):
        """the matching process is run for each VGAC scan,
        Cloudsat pixels within radius given by COLLOCATION_THRESHOLD
        are averaged to give the cloud base height at eligible pixels of
        VGAC
        """

        lon_c, lat_c, lon_v, lat_v = self._broadcast_arrays(i, tmask_c)

        # calculate haversine distance
        distances = haversine_distance(lat_v, lon_v, lat_c, lon_c)

        # find args where distance is below threshold
        x_argmin, y_argmin = np.where(distances <= COLLOCATION_THRESHOLD)

        # check tdiff between cloudsat and VGAC collocations
        tdiff = self.cloudsat.time[tmask_c][x_argmin] - self.vgac.time[i]
        tdiff_minutes = np.array([t.seconds / 60 for t in tdiff])

        # get x and y indices
        valid_indices = (
            self.cloudsat.validation_height_base[tmask_c][x_argmin] > 0
        ) & (tdiff_minutes < 20.0)

        # update base height and count number of cloudsat obs used for each VGAC pixel
        for valid_index in valid_indices:
            self.vgac.validation_height_base[i, I1:I2][
                y_argmin[valid_index]
            ] += self.cloudsat.validation_height_base[tmask_c][x_argmin[valid_index]]
            self.count_collocations[i, I1:I2][y_argmin[valid_index]] += 1

    def _get_time_mask(self, itime: int):
        """
        Mask to select part of Cloudsat track crossing the selected VGAC pixel
        """

        return (
            self.cloudsat.time >= self.vgac.time[itime] + timedelta(minutes=TIME1)
        ) & (self.cloudsat.time <= self.vgac.time[itime] + timedelta(minutes=TIME2))

    def match_vgac_cloudsat(self):
        """
        For each VGAC scan, matches from Cloudsat are found
        """

        for itime in range(len(self.vgac.time)):

            tmask_c = self._get_time_mask(itime)
            if np.sum(tmask_c) == 0:
                continue
            else:
                self._process_matching_iteration(itime, tmask_c)
