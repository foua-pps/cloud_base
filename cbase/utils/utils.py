from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta


R = 6371.0  # Earth's radius in kilometers


@dataclass
class BaseDate:
    """
    string format for base date
    "%Y%m%d%H%M"
    """

    base_date: str

    def __str__(self):
        return self.base_date


def convert2datetime(times: np.array, base_date_string: BaseDate) -> np.array:
    """
    convert time from secs to datetime objects
    """

    base_date = datetime.strptime(base_date_string.base_date, "%Y%m%d%H%M")
    return np.array([base_date + timedelta(seconds=value) for value in times])


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
