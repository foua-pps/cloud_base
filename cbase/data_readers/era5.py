from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.interpolate import interp1d
from pps_nwp.gribfile import GRIBFile


class PRESSURE_LEVELS(Enum):
    T250 = 250  # hPa
    T500 = 500
    T700 = 700
    T850 = 850
    T900 = 900
    Q250 = 250  # hPa
    Q500 = 500
    Q700 = 700
    Q850 = 850
    Q900 = 900


@dataclass
class Era5:
    """data reader for ERA5 data in grib format
    Uses PPS_NWP to read in grib data
    """

    grb: GRIBFile

    @classmethod
    def from_file(cls, filepath: Path):
        """a new wrapper class for GRIB data, uses PPS_NWP,"""
        return Era5(GRIBFile(filepath))

        # grb = grbs.select(name="Total column cloud liquid water")[0]
        # lats, lons = grb.latlons()
        # return cls(grb.values, grb.values * 100, lats, lons)

    def get_data(
        self, parameter: str, projection=[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """read in the required parameter and also allows to set the projection"""

        self.grb.set_projection(projection)

        values = None

        if parameter == "ciwv":
            values = self.grb.get_ciwv()[:]
        elif parameter == "tclw":
            values = self.grb.get_tclw()[:]
        elif parameter == "p_surface":
            values = self.grb.get_p_surface()[:]
        elif parameter == "z_surface":
            values = self.grb.get_z_surface()[:]
        elif parameter == "t_2meter":
            values = self.grb.get_t_2meter()[:]
        elif parameter == "t_vertical":
            values = self.grb.get_t_vertical()[:]
        elif parameter == "q_vertical":
            values = self.grb.get_q_vertical()[:]
        elif parameter == "h_2meter":
            values = self.grb.get_h_2meter()[:]
        elif parameter == "pressure_levels":
            values = self.grb.get_p_vertical()[:]
        elif parameter in ["t250", "t500", "t700", "t850", "t900"]:
            values = self.grb.get_t_pressure(PRESSURE_LEVELS[parameter.upper()].value)[
                :
            ]
        elif parameter in ["q250", "q500", "q700", "q850", "q900"]:
            values = self.grb.get_q_pressure(PRESSURE_LEVELS[parameter.upper()].value)[
                :
            ]
        if values is not None:
            return values
        else:
            raise ValueError("Invalid parameter name")

    def _interpolate_to_pressure_level(self, field: np.ndarray, new_level: float):
        p = self.grb.get_p_vertical()[:]
        p_flat = p.reshape(len(p), -1)
        field_interp = np.zeros_like(p[0, :, :]).flatten()
        field_flat = field.reshape(len(p), -1)
        npoints = field_flat.shape[1]

        for i in range(npoints):
            field_interp[i] = interp1d(p_flat[:, i], field_flat[:, i])(new_level)
        return field_interp.reshape(field.shape[1], field.shape[2])


if __name__ == "__main__":
    era5file = "/home/a002602/data/cloud_base/NWP/GAC_ECMWF_ERA5_201801010100+000H00M"

    era_obj = Era5.from_file(era5file)
    print(era_obj.grb.lonlat(), era_obj.grb.get_ciwv()[:])
