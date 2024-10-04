import re
from datetime import datetime, timedelta, timezone
import pytz
import numpy as np
import xarray as xr


R = 6371.0  # Earth's radius in kilometers


def check_lon_range(lons):
    within_range = (-180 <= lons) & (lons < 0)
    if np.any(within_range):
        return "-180--180"
    within_range = (0 <= lons) & (lons < 360)
    if np.any(within_range):
        return "0--360"


def adapt_lonrange(lons, flag="0--360"):
    if flag == "0--360":
        return lons % 360
    elif flag == "-180--180":
        return (lons + 180) % 360 - 180
    else:
        raise ValueError("only flgas allowed are : '0--360' and '-180--180'")


def extract_timestamp_from_atms_filename(filename):
    """Regular expression to capture the YYYYMMDDTHHMM pattern
    from ATMS files"""
    pattern = r"(\d{8}T\d{4})"
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        timestamp = datetime.strptime(date_str, "%Y%m%dT%H%M")
        timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp
    else:
        # If the pattern is not found, return None
        return None


def datetime64_to_datetime(times: np.datetime64) -> datetime:
    """convert np.datetime64 to datetime"""
    return np.array(
        [
            datetime.fromtimestamp(np.datetime64(time).astype("uint64") / 1e9, pytz.UTC)
            for time in times
        ]
    )


def microseconds_to_datetime(microseconds) -> datetime:
    """
    Convert microseconds since 1970-01-01 (Unix epoch) to a UTC datetime object.
    """
    seconds = microseconds / 10e6
    return np.array(
        [
            datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sec)
            for sec in seconds
        ]
    )


def haversine_distance(
    lat1: float, lon1: float, lat2: np.array, lon2: np.array
) -> np.array:
    """
    find distance between two points on earth
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Calculate the Haversine distance
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance


def create_dataset(
    lats: np.ndarray, lons: np.ndarray, values: np.ndarray, parameter_name: str
) -> xr.DataArray:
    """create xarray dataset"""
    return xr.DataArray(
        data=values,
        dims=["x", "y"],
        coords=dict(
            lon=(["x", "y"], lons),
            lat=(["x", "y"], lats),
        ),
        name=parameter_name,
    )
