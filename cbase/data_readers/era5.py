from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.interpolate import interp1d
from pps_nwp.gribfile import GRIBFile
from pps_nwp.water.humidity import sph2rh


class PressureLevels(Enum):
    """Enum representing pressure levels"""

    # T100 = 100
    T250 = 250  # hPa
    T400 = 400
    T500 = 500
    T700 = 700
    T850 = 850
    T900 = 900
    T950 = 950
    T1000 = 1000
    # RH100 = 100
    RH250 = 250  # hPa
    RH400 = 400
    RH500 = 500
    RH700 = 700
    RH850 = 850
    RH900 = 900
    RH950 = 950
    RH1000 = 1000
    Q250 = 250  # hPa
    Q400 = 400
    Q500 = 500
    Q700 = 700
    Q850 = 850
    Q900 = 900
    Q950 = 950
    Q1000 = 1000



@dataclass
class Era5:
    """data reader for ERA5 data in grib format
    Uses PPS_NWP to read in grib data
    """

    grb: GRIBFile

    @classmethod
    def from_file(cls, filepath: Path):
        """a new wrapper class for GRIB data, uses PPS_NWP,"""
        return Era5(GRIBFile(filepath.as_posix()))

    def get_data(
        self, parameter: str, projection=tuple[np.ndarray, np.ndarray]
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
        elif parameter == "PressureLevels":
            values = self.grb.get_p_vertical()[:]
        elif parameter in [
            "t100",
            "t250",
            "t400",
            "t500",
            "t700",
            "t850",
            "t900",
            "t950",
            "t1000",
        ]:
            values = self.grb.get_t_pressure(PressureLevels[parameter.upper()].value)[:]
        elif parameter in [
            "rh100",
            "rh250",
            "rh400",
            "rh500",
            "rh700",
            "rh850",
            "rh900",
            "rh950",
            "rh1000",
        ]:
            q = self.grb.get_q_pressure(PressureLevels[parameter.upper()].value)[:]
            t = self.grb.get_t_pressure(PressureLevels[parameter.upper()].value)[:]
            values = sph2rh(q, t, PressureLevels[parameter.upper()].value)
        elif parameter in [
            "q100",
            "q250",
            "q400",
            "q500",
            "q700",
            "q850",
            "q900",
            "q950",
            "q1000",
        ]:
            values = self.grb.get_q_pressure(PressureLevels[parameter.upper()].value)[:]
        elif parameter == "snow_mask":
            values = self.grb.get_snow_depth()[:]
        elif parameter == "t_land":
            values = self.grb.get_t_land()[:]
        elif parameter == "t_sea":
            values = self.grb.get_t_sea()[:]
        if values is not None:
            return values
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
