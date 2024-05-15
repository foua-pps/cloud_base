import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import pytz
import numpy as np
import matplotlib.pyplot as plt
from atrain_match.truths.cloudsat import (
    read_cloudsat_hdf4,
    add_validation_ctth_cloudsat,
)

# from cbase.utils.utils import convert2datetime, BaseDate


@dataclass
class BaseDate:
    """
    string format for base date
    "%Y%m%d%H%M"
    """

    base_date: str

    def __str__(self):
        return self.base_date


@dataclass
class CloudsatData:
    """
    Class to read and handle Cloudsat data
    """

    longitude: float
    latitude: float
    validation_height_base: float
    validation_height: float
    time: float
    name: str

    @classmethod
    def from_file(cls, filepath: Path):
        """
        read cloudsat data using atrain_match and
        generate constructor from file
        """
        cloudsat = read_cloudsat_hdf4(filepath.as_posix())
        cloudsat = add_validation_ctth_cloudsat(cloudsat)
        times = convert2datetime(cloudsat.sec_1970, BaseDate("197001010000"))
        return cls(
            cloudsat.longitude % 360,
            cloudsat.latitude,
            cloudsat.validation_height_base,
            cloudsat.validation_height,
            times,
            os.path.basename(filepath),
        )


def convert2datetime(times: np.array, base_date_string: BaseDate) -> np.array:
    """
    convert time from secs to datetime objects
    """

    base_date = datetime.strptime(base_date_string.base_date, "%Y%m%d%H%M").replace(
        tzinfo=pytz.UTC
    )
    return np.array([base_date + timedelta(seconds=value) for value in times])


if __name__ == "__main__":

    filename = "/home/a002602/data/cloud_base/cloudsat/2018150150536_64379_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"
    retv = CloudsatData.from_file(filename)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.scatter(
        retv.longitude,
        retv.latitude,
        c=retv.validation_height - retv.validation_height_base,
    )
    fig.savefig("cloudsat.png")
