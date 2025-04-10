import os
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
import numpy as np
from satpy import Scene
import xarray as xr
import re
from cbase.utils.utils import datetime64_to_datetime

VGAC_PPS_PATH = "/nobackup/smhid20/proj/safcm/work/PPS/PPS2021_3_CLARA_VGAC/CALIPSO_matchups/SNPP/VIIRS/export/"


VGAC_PARAMETER_LIST = [
    "latitude",
    "longitude",
    "scanline_timestamps",
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "M06",
    "M07",
    "M08",
    "M09",
    "M10",
    "M11",
    "M12",
    "M13",
    "M14",
    "M15",
    "M16",
]


@dataclass
class VGACData:
    """
    A container for VGAC data read in using satpy
    """

    latitude: np.ndarray
    longitude: np.ndarray
    time: np.ndarray
    M01: np.ndarray
    M02: np.ndarray
    M03: np.ndarray
    M04: np.ndarray
    M05: np.ndarray
    M06: np.ndarray
    M07: np.ndarray
    M08: np.ndarray
    M09: np.ndarray
    M10: np.ndarray
    M11: np.ndarray
    M12: np.ndarray
    M13: np.ndarray
    M14: np.ndarray
    M15: np.ndarray
    M16: np.ndarray
    name: str

    @classmethod
    def from_file(cls, filepath: Path):
        """alternative constructor from file"""
        scn = Scene(reader="viirs_vgac_l1c_nc", filenames=[filepath])
        scn.load(VGAC_PARAMETER_LIST)
        d = scn.to_xarray()
        time_scanline = datetime64_to_datetime(d.scanline_timestamps.values)
        time = np.tile(time_scanline, (d.latitude.values.shape[1], 1)).T
        return cls(
            d.latitude.values,
            d.longitude.values % 360,
            time,
            d.M01.values,
            d.M02.values,
            d.M03.values,
            d.M04.values,
            d.M05.values,
            d.M06.values,
            d.M07.values,
            d.M08.values,
            d.M09.values,
            d.M10.values,
            d.M11.values,
            d.M12.values,
            d.M13.values,
            d.M14.values,
            d.M15.values,
            d.M16.values,
            os.path.basename(filepath),
        )


@dataclass
class VGACPPSData:
    """
    Class to read and handle VGAC data
    """

    longitude: np.ndarray
    latitude: np.ndarray
    time: np.ndarray
    validation_height_base: np.ndarray
    M01: np.ndarray
    M02: np.ndarray
    M03: np.ndarray
    M04: np.ndarray
    M05: np.ndarray
    M06: np.ndarray
    M07: np.ndarray
    M08: np.ndarray
    M09: np.ndarray
    M10: np.ndarray
    M11: np.ndarray
    M12: np.ndarray
    M13: np.ndarray
    M14: np.ndarray
    M15: np.ndarray
    M16: np.ndarray
    satzenith: np.ndarray
    satazimuth: np.ndarray
    sunzenith: np.array
    ctp: np.ndarray
    cth: np.ndarray
    ctt: np.ndarray
    ctp_quality: np.ndarray
    ctp16: np.ndarray
    cth16: np.ndarray
    ctt16: np.ndarray
    ctp84: np.ndarray
    cth84: np.ndarray
    ctt84: np.ndarray
    ct: np.ndarray
    ct_quality: np.ndarray
    cmic_phase: np.ndarray
    cmic_lwp: np.ndarray
    cmic_cot: np.ndarray
    cmic_quality: np.ndarray
    elevation: np.ndarray
    land_use: np.ndarray

    @classmethod
    def from_file(cls, filepath: Path):
        """read data from netCDF file"""
        with xr.open_dataset(filepath) as da:
            validation_height_base = -999.9 * np.ones_like(da.lat.values)
            time_scanline = datetime64_to_datetime(da.scanline_timestamps.values)
            time = np.tile(time_scanline, (da.lat.shape[1], 1)).T
        pps_data = get_pps_data(filepath)
        vgac = VGACPPSData(
            da.lon.values % 360,
            da.lat.values,
            time,
            validation_height_base,
            da.image20.values[0, :, :],  # M01
            da.image21.values[0, :, :],  # M02
            da.image22.values[0, :, :],  # M03
            da.image23.values[0, :, :],  # M04
            da.image1.values[0, :, :],  # M05
            da.image24.values[0, :, :],  # M06
            da.image2.values[0, :, :],  # M07
            da.image25.values[0, :, :],  # M08
            da.image8.values[0, :, :],  # M09
            da.image6.values[0, :, :],  # M10
            da.image9.values[0, :, :],  # M11
            da.image5.values[0, :, :],  # M12
            da.image26.values[0, :, :],  # M13
            da.image7.values[0, :, :],  # M14
            da.image3.values[0, :, :],  # M15
            da.image4.values[0, :, :],  # M16
            da.satzenith.values[0, :, :],
            da.satazimuth.values[0, :, :],
            da.sunzenith.values[0, :, :],
            extract_pps_parameter(pps_data, "ctth_pres"),
            extract_pps_parameter(pps_data, "ctth_alti"),
            extract_pps_parameter(pps_data, "ctth_tempe"),
            extract_pps_parameter(pps_data, "ctth_quality"),
            extract_pps_parameter(pps_data, "ctth16_pres"),
            extract_pps_parameter(pps_data, "ctth16_alti"),
            extract_pps_parameter(pps_data, "ctth16_tempe"),
            extract_pps_parameter(pps_data, "ctth84_pres"),
            extract_pps_parameter(pps_data, "ctth84_alti"),
            extract_pps_parameter(pps_data, "ctth84_tempe"),
            extract_pps_parameter(pps_data, "ct"),
            extract_pps_parameter(pps_data, "ct_quality"),
            extract_pps_parameter(pps_data, "cmic_phase"),
            extract_pps_parameter(pps_data, "cmic_lwp"),
            extract_pps_parameter(pps_data, "cmic_cot"),
            extract_pps_parameter(pps_data, "cmic_quality"),
            extract_pps_parameter(pps_data, "elevation"),
            extract_pps_parameter(pps_data, "land_use"),
        )
        return vgac


def get_pps_data(input_path: Path) -> dict:
    (
        output_path,
        aux_path,
        ctth_filename,
        ctth16_filename,
        ctth84_filename,
        ct_filename,
        cmic_filename,
        aux_filename,
    ) = get_pps_data_files(input_path)
    pps_data = {}
    print(
        output_path,
        aux_path,
        ctth_filename,
        ctth16_filename,
        ctth84_filename,
        ct_filename,
        cmic_filename,
        aux_filename,
    )
    with xr.open_dataset(os.path.join(output_path, ctth_filename)) as da:
        pps_data["ctth_pres"] = da.ctth_pres.values[0, :, :]
        pps_data["ctth_tempe"] = da.ctth_tempe.values[0, :, :]
        pps_data["ctth_quality"] = da.ctth_quality.values[0, :, :]
        pps_data["ctth_alti"] = da.ctth_alti.values[0, :, :]
    with xr.open_dataset(os.path.join(output_path, ctth16_filename)) as da:
        pps_data["ctth16_pres"] = da.ctth_pres.values[0, :, :]
        pps_data["ctth16_tempe"] = da.ctth_tempe.values[0, :, :]
        pps_data["ctth16_alti"] = da.ctth_alti.values[0, :, :]
    with xr.open_dataset(os.path.join(output_path, ctth84_filename)) as da:
        pps_data["ctth84_pres"] = da.ctth_pres.values[0, :, :]
        pps_data["ctth84_tempe"] = da.ctth_tempe.values[0, :, :]
        pps_data["ctth84_alti"] = da.ctth_alti.values[0, :, :]
    with xr.open_dataset(os.path.join(output_path, ct_filename)) as da:
        pps_data["ct"] = da.ct.values[0, :, :]
        pps_data["ct_quality"] = da.ct_quality.values[0, :, :]
    with xr.open_dataset(os.path.join(output_path, cmic_filename)) as da:
        pps_data["cmic_phase"] = da.cmic_phase.values[0, :, :]
        pps_data["cmic_cot"] = da.cmic_cot.values[0, :, :]
        pps_data["cmic_lwp"] = da.cmic_lwp.values[0, :, :]
        pps_data["cmic_quality"] = da.cmic_quality.values[0, :, :]
    with xr.open_dataset(os.path.join(aux_path, aux_filename)) as da:
        pps_data["elevation"] = da.elevation.values[0, :, :]
        pps_data["land_use"] = da.landuse[0, :, :]
    return pps_data


def extract_pps_parameter(pps_data, parameter) -> np.ndarray:
    try:
        return pps_data[parameter]
    except:
        raise Exception(f"{parameter} not present")


def get_year_month_day(file_path: Path) -> tuple[str, str, str]:
    year = file_path.parts[-4]
    month = file_path.parts[-3]
    day = file_path.parts[-2]
    return year, month, day


def get_pps_data_files(input_path: Path) -> Tuple[str, str, str, str]:
    """Extract PPS file names from the input path of L1C files"""
    input_filename = input_path.as_posix().split("/")[-1]
    year, month, day = get_year_month_day(input_path)
    output_path = os.path.join(VGAC_PPS_PATH, year, month, day)
    # output_path = output_path.replace("AVHRR_HERITAGE/", "")
    # output_path = output_path.replace("NO_SBAF/", "NO_SBAF/export/")
    ctth_filename = re.sub(r"S_NWC_viirs_npp", "S_NWC_CTTH_npp", input_filename)
    ctth16_filename = re.sub(r"S_NWC_viirs_npp", "S_NWC_CTTHUpP16_npp", input_filename)
    ctth84_filename = re.sub(r"S_NWC_viirs_npp", "S_NWC_CTTHLoP84_npp", input_filename)
    ct_filename = re.sub(r"S_NWC_viirs_npp", "S_NWC_CT_npp", input_filename)
    cmic_filename = re.sub(r"S_NWC_viirs_npp", "S_NWC_CMIC_npp", input_filename)
    aux_path = output_path.replace("export", "intermediate/AUX_remapped/")
    aux_filename = re.sub(r"S_NWC_viirs_npp", "S_NWC_physiography_npp", input_filename)
    return (
        output_path,
        aux_path,
        ctth_filename,
        ctth16_filename,
        ctth84_filename,
        ct_filename,
        cmic_filename,
        aux_filename,
    )
