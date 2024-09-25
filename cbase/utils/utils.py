from datetime import datetime, timedelta, timezone
import pytz
import numpy as np
import xarray as xr


R = 6371.0  # Earth's radius in kilometers


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
        [datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sec) for sec in seconds]
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
