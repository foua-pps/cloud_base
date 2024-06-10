import re
from sys import argv
import argparse
from datetime import datetime, timedelta
import numpy as np
from cbase.matching.config import (
    SECS_PER_MINUTE,
    MINUTES_PER_HOUR,
)


def create_datetime_from_year_doy_hour_minute(
    year: int, doy: int, hour: int, minute: int
) -> datetime:
    """make a datetime obj"""

    base_date = datetime(year, 1, 1)
    target_date = base_date + timedelta(days=doy - 1)
    target_date = target_date.replace(hour=hour, minute=minute)
    return target_date


def get_cloudsat_time(
    cloudsat_file: str,
) -> datetime:
    """get CSAT orbit time"""
    try:
        pattern = r"(\d{13})_(\d{5})"
        match = re.search(pattern, cloudsat_file)

        year = match.group(1)[:4]
        doy = match.group(1)[4:7]
        hour = match.group(1)[7:9]
        minutes = match.group(1)[9:11]
        ctime = create_datetime_from_year_doy_hour_minute(
            int(year), int(doy), int(hour), int(minutes)
        )
        return ctime
    except Exception as exc:
        raise ValueError(
            f"the pattern is of type *2018150001814_64370*, check file name: {cloudsat_file}"
        ) from exc


def get_vgac_time(vgac_file: str) -> datetime:
    """get VGAC orbit time"""
    try:
        pattern = r"(\d{7})_(\d{4})"
        match = re.search(pattern, vgac_file)
        year = match.group(1)[:4]
        doy = match.group(1)[4:7]
        hour = match.group(2)[0:2]
        minutes = match.group(2)[2:4]
        vtime = create_datetime_from_year_doy_hour_minute(
            int(year), int(doy), int(hour), int(minutes)
        )
        return vtime
    except Exception as exc:
        raise ValueError(
            f"the pattern is of type *2018150_0136*, check file name: {vgac_file}"
        ) from exc


def is_valid_match(ctime: datetime, vtime: datetime) -> bool:
    """check if time diff between two sat passes is optimal"""

    tdiff = (ctime - vtime).total_seconds() / SECS_PER_MINUTE
    return np.abs(tdiff) < 0.5 * MINUTES_PER_HOUR  # 30 minutes tolerance


def _find_matching_files(ctime, files: list, key: str) -> list:
    """Find matching DARDAR/VGAC/NWP files"""

    def _matching_string(time: datetime, key: str):
        if key in ["vgac", "nwp", "dardar"]:
            if key == "vgac":
                return time.strftime("%Y%j_%H")
            if key == "nwp":
                return time.strftime("%Y%m%d%H")
            if key == "dardar":
                return time.strftime("%Y%j%H%M")
        raise ValueError(
            "please check key, only one of ['vgac', 'nwp', 'dardar'] are allowed"
        )

    matched_files = []
    matched_files += [file for file in files if _matching_string(ctime, key) in file]

    if key in ["vgac", "nwp"]:
        # also add in files from prev hour
        prev_hour = ctime - timedelta(hours=1)
        matched_files += [
            file for file in files if _matching_string(prev_hour, key) in file
        ]
    return matched_files


def get_matching_cloudsat_vgac_nwp_files(
    cfiles: list, dfiles: list, vfiles: list, nfiles: list
) -> tuple[list, list, list]:
    """get matching VGAC/NWP/DARDAR filename for CSAT orbit file"""
    matched_vfiles = []
    matched_cfiles = []
    matched_nfiles = []
    matched_dfiles = []
    for cfile in cfiles:
        print(cfile)
        ctime = get_cloudsat_time(cfile)
        d_matches = _find_matching_files(ctime, dfiles, "dardar")
        v_matches = _find_matching_files(ctime, vfiles, "vgac")
        n_matches = _find_matching_files(ctime, nfiles, "nwp")

        if v_matches and n_matches and d_matches:
            for vfile, nfile in zip(v_matches, n_matches):
                # check for closest VGAC file
                vtime = get_vgac_time(vfile)
                if is_valid_match(ctime, vtime):
                    matched_vfiles.append(vfile)
                    matched_cfiles.append(cfile)
                    matched_dfiles.append(d_matches[0])
                    matched_nfiles.append(nfile)
                break
    return matched_cfiles, matched_dfiles, matched_vfiles, matched_nfiles
