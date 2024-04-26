import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from atrain_match.truths.cloudsat import (
    read_cloudsat_hdf4,
    add_cloudsat_cloud_fraction,
    add_validation_ctth_cloudsat,
)
from cbase.utils.utils import convert2datetime, BaseDate


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


def read_cloudsat(filename: Path):
    """
    read cloudsat data using atrain_match and
    generate a CloudsatData object
    """

    cloudsat = read_cloudsat_hdf4(filename)
    cloudsat = add_validation_ctth_cloudsat(cloudsat)
    times = convert2datetime(cloudsat.sec_1970, BaseDate("197001010000"))
    return CloudsatData(
        cloudsat.longitude,
        cloudsat.latitude,
        cloudsat.validation_height_base,
        cloudsat.validation_height,
        times,
    )


if __name__ == "__main__":
    filename = "/home/a002602/data/cloud_base/cloudsat/2018150150536_64379_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"
    retv = read_cloudsat(filename)
    print(retv.all_arrays.keys())
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.scatter(
        retv.longitude,
        retv.latitude,
        c=retv.validation_height - retv.validation_height_base,
    )
    fig.savefig("cloudsat.png")
