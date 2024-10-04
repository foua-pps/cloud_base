import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Union
import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator
from pps_nwp.gribfile import GRIBFile
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData
from cbase.utils.utils import haversine_distance
from .config import (
    COLLOCATION_THRESHOLD,
    TIME_WINDOW,
    XIMAGE_SIZE,
    YIMAGE_SIZE,
    CNN_NWP_PARAMETERS,
    CNN_VGAC_PARAMETERS,
    CNN_MATCHED_PARAMETERS,
    TIME_DIFF_ALLOWED,
    SECS_PER_MINUTE,
    OUTPUT_PATH,
    PIXEL_DATA,
    CNN_DATA,
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

        if not self.check_overlapping_time():
            raise ValueError("The two passes are not at same time")

        self.out_filename = os.path.join(
            OUTPUT_PATH, f"cnn_data_{self.cloudsat.name[:22]}_VGAC.nc"
        )
        self.collocated_data = self.initialize_collocated_data()

    def initialize_collocated_data(self) -> dict:
        """Initialize the collocated data dictionary"""
        collocated_dict = {}
        for key in CNN_MATCHED_PARAMETERS:
            collocated_dict[key] = np.ones_like(self.vgac.latitude) * -999.9
        return collocated_dict

    def check_overlapping_time(self) -> bool:
        """
        check if the two satellites passes are overlapping
        """
        t1 = self.cloudsat.time[0]  # start time
        t2 = self.cloudsat.time[-1]  # end time
        if t1 > t2:
            raise ValueError("start time cannot be after end time")
        for dt in self.vgac.time[:, 0]:
            if t1 <= dt <= t2:
                return True
        return False

    def broadcast_arrays(
        self, i: int, icld: tuple[int, int]
    ) -> list[float, float, float, float]:
        """
        broadcast lat/lon arrays to 2d to enable vectorized calculation
        of distances
        """

        def _broadcast(arr, shape1, shape2):
            return np.broadcast_to(arr, (shape1, shape2))

        icld1, icld2 = icld[0], icld[1]
        shape1 = self.cloudsat.longitude[icld1:icld2].shape[0]
        shape2 = self.vgac.longitude[i, :].shape[0]

        lon_c = _broadcast(
            self.cloudsat.longitude[icld1:icld2].reshape(-1, 1), shape1, shape2
        )
        lat_c = _broadcast(
            self.cloudsat.latitude[icld1:icld2].reshape(-1, 1), shape1, shape2
        )
        lon_v = _broadcast(self.vgac.longitude[i, :].reshape(1, -1), shape1, shape2)
        lat_v = _broadcast(self.vgac.latitude[i, :].reshape(1, -1), shape1, shape2)

        return lon_c, lat_c, lon_v, lat_v

    def process_matching_iteration_nearest(self, i: int, icld: tuple[int, int]):
        """the matching process is run for each VGAC scan,
        Cloudsat pixels within radius given by COLLOCATION_THRESHOLD
        are averaged to give the cloud base height at eligible pixels of
        VGAC
        """

        lon_c, lat_c, lon_v, lat_v = self.broadcast_arrays(i, icld)

        # calculate haversine distance & find args where distance is below
        #  threshold
        distances = haversine_distance(lat_v, lon_v, lat_c, lon_c)
        args = np.argwhere(distances <= COLLOCATION_THRESHOLD)
        x_argmin, y_argmin = args[:, 0], args[:, 1]
        # print(x_argmin, y_argmin)
        # check tdiff between cloudsat and VGAC collocations
        tdiff = np.abs(
            self.cloudsat.time[icld[0] : icld[1]][x_argmin] - self.vgac.time[i, 0]
        )
        tdiff_minutes = np.array([t.seconds / SECS_PER_MINUTE for t in tdiff])
        # find indices where values are greater than zero
        valid_indices = np.where(
            self.cloudsat.cloud_base[icld[0] : icld[1]][x_argmin]
            > 0 & (tdiff_minutes < TIME_DIFF_ALLOWED)
        )[0]
        # print(x_argmin[valid_indices])
        if len(valid_indices) > 0:
            self._collocate_data(
                i, icld, x_argmin[valid_indices], y_argmin[valid_indices]
            )

    def _collocate_data(self, i, icld, ix, iy):
        """update height/cf/layers and count number of cloudsat obs used for each VGAC pixel"""

        def _interpolate_nearest(x, y, z, x_new, y_new):
            return NearestNDInterpolator(np.vstack([x, y]).T, z)(
                np.vstack((x_new, y_new)).T
            )

        for key in CNN_MATCHED_PARAMETERS:
            c_data = getattr(self.cloudsat, key)
            v_data = self.collocated_data[key]
            v_data[i, :][iy] = _interpolate_nearest(
                self.cloudsat.latitude[icld[0] : icld[1]][ix],
                self.cloudsat.longitude[icld[0] : icld[1]][ix],
                c_data[icld[0] : icld[1]][ix],
                self.vgac.latitude[i, :][iy],
                self.vgac.longitude[i, :][iy],
            )
        self.collocated_data[key] = v_data

    def get_tmask(self, itime: int):
        """get part of cloudsat track within the time window"""
        return (
            self.cloudsat.time
            >= self.vgac.time[itime, 0] + timedelta(minutes=TIME_WINDOW[0])
        ) & (
            self.cloudsat.time
            <= self.vgac.time[itime, 0] + timedelta(minutes=TIME_WINDOW[1])
        )

    def get_index_closest_cloudsat_track(
        self, itime: int
    ) -> Union[tuple[int, int], None]:
        """
        select part of Cloudsat track crossing the selected VGAC pixel
        """
        # select part of cloudsat swath within TIME WINDOW
        tmask = self.get_tmask(itime)
        if len(tmask) > 0:
            index = int(self.vgac.latitude.shape[1] / 2)  # center of swath
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
        return None

    def match_vgac_cloudsat(self):
        """
        For each VGAC scan, matches from Cloudsat are found
        """

        for itime in range(len(self.vgac.time)):
            if self.get_index_closest_cloudsat_track(itime) is None:
                continue
            icld1, icld2 = self.get_index_closest_cloudsat_track(itime)

            if np.all(self.cloudsat.cloud_base[icld1:icld2] < 0):
                continue
            # get the matching data for the selected part of swath
            self.process_matching_iteration_nearest(itime, [icld1, icld2])

    def _bounding_box(self, i: int, j: int):
        """bounding box for CNN input image"""
        n = len(self.vgac.time)
        return BoundingBox(
            max(0, i - int(YIMAGE_SIZE / 2)),
            min(n, i + int(YIMAGE_SIZE / 2)),
            max(0, j - int(XIMAGE_SIZE / 2)),
            min(n, j + int(XIMAGE_SIZE / 2)),
        )

    def _interpolate_nwp_data(
        self, parameter: str, projection: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        try:
            return self.era5.get_data(parameter, projection)
        except Exception as e:
            return np.ones_like([projection[0].size, projection[1].size]) * -999.9

    def _make_cnn_data_matched_parameters(
        self, lists_collocated_data: dict, box: BoundingBox
    ):
        for parameter, values in lists_collocated_data.items():
            data = self.collocated_data[parameter]
            values.append(data[box.i1 : box.i2, box.j1 : box.j2])

    def _make_cnn_data_vgac_parameters(self, lists_vgac_data: dict, box: BoundingBox):
        for parameter, values in lists_vgac_data.items():
            data = getattr(self.vgac, parameter)
            values.append(data[box.i1 : box.i2, box.j1 : box.j2])

    def _make_cnn_data_vgac_parameters_pixel(
        self, lists_vgac_data: dict, box: BoundingBox
    ):
        for parameter, values in lists_vgac_data.items():
            data = getattr(self.vgac, parameter)
            values.append(data[box.i1 : box.i2, box.j1 : box.j2])

    def _make_cnn_data_nwp_parameters(
        self, lists_vgac_data: dict, lists_nwp_data: dict, inum: int
    ):
        for parameter, values in lists_nwp_data.items():
            print(parameter)
            remap_lats = lists_vgac_data["latitude"][inum]
            remap_lons = lists_vgac_data["longitude"][inum]
            projection = (
                remap_lons,
                remap_lats,
            )  # projection to regrid ERA5 data
            values.append(self._interpolate_nwp_data(parameter, projection))

    def create_cnn_dataset_with_nwp(self, to_file=True) -> xr.Dataset:
        """
        crop VGAC images to required size for CNN, add required NWP data,
        and collect data into a xarray dataset
        """

        lists_vgac_data = {name: [] for name in CNN_VGAC_PARAMETERS}
        lists_collocated_data = {name: [] for name in CNN_MATCHED_PARAMETERS}
        lists_nwp_data = {name: [] for name in CNN_NWP_PARAMETERS}

        inum = 0
        print("collocated_data", len(self.collocated_data["cloud_base"]))
        for ipix in range(0, len(self.vgac.time), YIMAGE_SIZE):
            iscan = np.where(self.collocated_data["cloud_base"][ipix, :] > 0)[0]
            if len(iscan) > 0:
                print(iscan)
                iscan = iscan[0]
                if PIXEL_DATA == True:
                    self._make_cnn_data_vgac_parameters(lists_vgac_data, ipix, iscan)
                    self._make_cnn_data_matched_parameters(
                        lists_collocated_data, ipix, iscan
                    )
                    self._make_cnn_data_nwp_parameters(
                        lists_vgac_data, lists_nwp_data, inum
                    )

                if CNN_DATA == True:
                    box = self._bounding_box(ipix, iscan)
                    if (box.i2 - box.i1, box.j2 - box.j1) == (YIMAGE_SIZE, XIMAGE_SIZE):
                        self._make_cnn_data_vgac_parameters(lists_vgac_data, box)
                        self._make_cnn_data_matched_parameters(
                            lists_collocated_data, box
                        )
                        self._make_cnn_data_nwp_parameters(
                            lists_vgac_data, lists_nwp_data, inum
                        )
                        inum += 1
        if len(lists_vgac_data) > 0:
            ds = self._make_dataset(
                lists_vgac_data, lists_collocated_data, lists_nwp_data
            )
            if to_file is True:
                ds.to_netcdf(self.out_filename)
        else:
            raise ValueError("No matched found within 30 minutes")

    def _make_dataset(
        self, lists_vgac_data: dict, lists_collocated_data: dict, lists_nwp_data: dict
    ) -> xr.Dataset:
        ds = xr.Dataset()

        nscene = np.arange(len(lists_vgac_data["latitude"]))
        npix = np.arange(YIMAGE_SIZE)
        nscan = np.arange(XIMAGE_SIZE)
        print(nscene, npix, nscan)
        parameter_types = {
            "VGAC": (CNN_VGAC_PARAMETERS, lists_vgac_data),
            "MATCHED": (CNN_MATCHED_PARAMETERS, lists_collocated_data),
            "NWP": (CNN_NWP_PARAMETERS, lists_nwp_data),
        }

        for _, (parameters, data_list) in parameter_types.items():
            for parameter in parameters:
                if (
                    parameter == "time"
                ):  # time cannot be stred as datetime in netcdf file
                    values = np.vectorize(lambda dt: dt.timestamp())(
                        np.stack(data_list[parameter])
                    )
                else:
                    values = np.stack(data_list[parameter])
                ds[parameter] = xr.DataArray(
                    values,
                    dims=("nscene", "npix", "nscan"),
                    coords={"npix": npix, "nscan": nscan, "nscene": nscene},
                )

        return ds
