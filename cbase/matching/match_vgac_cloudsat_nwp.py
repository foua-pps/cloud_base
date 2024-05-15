import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Union
import numpy as np
import xarray as xr
from pps_nwp.gribfile import GRIBFile
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData
from cbase.utils.utils import haversine_distance
from .config import (
    I1,
    I2,
    COLLOCATION_THRESHOLD,
    TIME_WINDOW,
    IMAGE_SIZE,
    CNN_NWP_PARAMETERS,
    CNN_SAT_PARAMETERS,
    SWATH_CENTER,
    TIME_DIFF_ALLOWED,
    SECS_PER_MINUTE,
    OUTPUT_PATH,
)


@dataclass
class BoundingBox:
    """class to define the extent of a bounding box to crop images
    for CNN training input data"""

    i1: int
    i2: int
    j1: int
    j2: int


class DataMatcher:
    """Class to match VGAC and CLOUDSAT data, add NWP data to selected scenes"""

    def __init__(self, cloudsat: CloudsatData, vgac: VGACData, era5: GRIBFile):
        self.cloudsat = cloudsat
        self.vgac = vgac
        self.era5 = era5

        self.out_filename = os.path.join(
            OUTPUT_PATH, f"cnn_data_{self.cloudsat.name[:22]}_VGAC.nc"
        )
        self.count_collocations = np.zeros_like(self.vgac.latitude)

        if not self.check_overlapping_time():
            raise ValueError("The two passes are not at same time")

    def check_overlapping_time(self) -> bool:
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

    def _broadcast_arrays(
        self, i: int, icld: tuple[int, int]
    ) -> list[float, float, float, float]:
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

        # check tdiff between cloudsat and VGAC collocations
        tdiff = np.abs(
            self.cloudsat.time[icld[0] : icld[1]][x_argmin] - self.vgac.time[i]
        )

        tdiff_minutes = np.array([t.seconds / SECS_PER_MINUTE for t in tdiff])

        # get x and y indices
        valid_indices = np.where(
            self.cloudsat.validation_height_base[icld[0] : icld[1]][x_argmin]
            > 0 & (tdiff_minutes < TIME_DIFF_ALLOWED)
        )[0]
        #  print(f"indicies, {i}, {valid_indices}, {icld}")
        # update base height and count number of cloudsat obs used for each VGAC pixel
        for valid_index in valid_indices:
            self.vgac.validation_height_base[i, I1:I2][
                y_argmin[valid_index]
            ] += self.cloudsat.validation_height_base[icld[0] : icld[1]][
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

        # check if the guess can be made from previous iteration
        _, iscan = np.where(self.count_collocations[itime - 5 : itime, :] > 0)
        if len(iscan) == 0:
            index = SWATH_CENTER  # center of swath
            distance = haversine_distance(
                self.vgac.latitude[itime, index],
                self.vgac.longitude[itime, index],
                self.cloudsat.latitude[tmask],
                self.cloudsat.longitude[tmask],
            )
            argmin = np.argmin(distance)
            index = np.arange(0, len(self.cloudsat.time), 1)[tmask][argmin]

            cld_mask = np.argwhere(
                (self.cloudsat.latitude > self.cloudsat.latitude[index] - 1.5)
                & (self.cloudsat.latitude < self.cloudsat.latitude[index] + 1.5)
                & (self.cloudsat.longitude > self.cloudsat.longitude[index] - 1.5)
                & (self.cloudsat.longitude < self.cloudsat.longitude[index] + 1.5)
            )
            return (cld_mask[0][0], cld_mask[-1][0])

        index = iscan[-1]  # last scan position where cloudsat and VGAC intersected
        distance = haversine_distance(
            self.vgac.latitude[itime, index],
            self.vgac.longitude[itime, index],
            self.cloudsat.latitude[tmask],
            self.cloudsat.longitude[tmask],
        )
        argmin = np.argmin(distance)
        index = np.arange(0, len(self.cloudsat.time), 1)[tmask][argmin]
        return (index - 10, index + 10)

        # # buffer zone defines the extra cloudsat swath which needs to be
        # # considered to find the best collocation
        # # this is a crude way to subset the Cloudsat swath, in future
        # # a better way could be used
        # buffer_zone = int(np.min(distance) / 4)
        # return (index - buffer_zone, index + buffer_zone)

    def match_vgac_cloudsat(self):
        """
        For each VGAC scan, matches from Cloudsat are found
        """

        for itime in range(len(self.vgac.time)):

            if self._get_closest_cloudsat_guess(itime) is None:
                continue
            icld1, icld2 = self._get_closest_cloudsat_guess(itime)
            if np.all(self.cloudsat.validation_height_base[icld1:icld2] < 0):
                continue
            self._process_matching_iteration(itime, [icld1, icld2])

        valid_indices = self.count_collocations > 0

        self.vgac.validation_height_base[valid_indices] = (
            self.vgac.validation_height_base[valid_indices]
            / self.count_collocations[valid_indices]
        )
        self.vgac.validation_height_base[~valid_indices] = -999.9

    def _bounding_box(self, i: int, j: int):
        """bounding box for CNN input image"""
        N = len(self.vgac.time)
        return BoundingBox(
            max(0, i - int(IMAGE_SIZE / 2)),
            min(N, i + int(IMAGE_SIZE / 2)),
            max(0, j - int(IMAGE_SIZE / 2)),
            min(N, j + int(IMAGE_SIZE / 2)),
        )

    def _interpolate_nwp_data(
        self, parameter: str, projection: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:

        if parameter not in CNN_NWP_PARAMETERS:
            raise ValueError(
                f"the NWP paramter: {parameter} is not present in gribfile"
            )
        return self.era5.get_data(parameter, projection)

    def create_cnn_dataset_with_nwp(self, to_file=True) -> xr.Dataset:
        """
        crop VGAC images to required size for CNN, add required NWP data,
        and collect data into a xarray dataset
        """

        def _make_dataset(lists_sat_data: dict, lists_nwp_data: dict) -> xr.Dataset:
            ds = xr.Dataset()

            nscene = np.arange(len(lists_sat_data["latitude"]))
            npix = np.arange(IMAGE_SIZE)
            nscan = np.arange(IMAGE_SIZE)
            for parameter in CNN_SAT_PARAMETERS:
                print(parameter, len(lists_sat_data[parameter]))
                ds[parameter] = xr.DataArray(
                    np.stack(lists_sat_data[parameter]),
                    dims=("nscene", "npix", "nscan"),
                    coords={"npix": npix, "nscan": nscan, "nscene": nscene},
                )
            for parameter in CNN_NWP_PARAMETERS:
                ds[parameter] = xr.DataArray(
                    np.stack(lists_nwp_data[parameter]),
                    dims=("nscene", "npix", "nscan"),
                    coords={"npix": npix, "nscan": nscan, "nscene": nscene},
                )

            return ds

        lists_sat_data = {name: [] for name in CNN_SAT_PARAMETERS}
        lists_nwp_data = {name: [] for name in CNN_NWP_PARAMETERS}

        inum = 0
        for ipix in range(0, len(self.vgac.time), IMAGE_SIZE):
            iscan = np.where(self.vgac.validation_height_base[ipix, :] > 0)[0]
            if len(iscan) > 0:
                iscan = iscan[0]
                box = self._bounding_box(ipix, iscan)
                if (box.i2 - box.i1, box.j2 - box.j1) == (IMAGE_SIZE, IMAGE_SIZE):
                    for parameter, values in lists_sat_data.items():

                    remap_lats = lists_sat_data["latitude"][inum]
                    remap_lons = lists_sat_data["longitude"][inum]
                    projection = (
                        remap_lons,
                        remap_lats,
                    )  # projection to regrid ERA5 data
                    values.append(self._interpolate_nwp_data(parameter, projection))
                inum += 1
                        data = getattr(self.vgac, parameter)
                        values.append(data[box.i1 : box.i2, box.j1 : box.j2])

                    for parameter, values in lists_nwp_data.items():
                        remap_lats = lists_sat_data["latitude"][inum]
                        remap_lons = lists_sat_data["longitude"][inum]
                        projection = (
                            remap_lons,
                            remap_lats,
                        )  # projection to regrid ERA5 data
                        values.append(self._interpolate_nwp_data(parameter, projection))
                    inum += 1
        ds = _make_dataset(lists_sat_data, lists_nwp_data)

        if to_file is True:
            ds.to_netcdf(self.out_filename)
