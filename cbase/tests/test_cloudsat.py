import os
from pathlib import Path
from unittest.mock import patch
import pytz
import numpy as np
import pytest
from datetime import datetime
from cbase.data_readers.cloudsat_products import (
    CloudsatData,
    BaseDate,
    read_cloudsat_hdf4,
    get_top_height,
    get_base_height,
    get_cloud_fraction,
    get_time,
)

# sample data for testing
sample_filepath = Path("sample_file.hdf")
sample_data = {
    "Longitude": np.array([17.2, 17.25]),
    "Latitude": np.array([45.6, 45.8]),
    "CloudLayerBase": np.array(
        [[0.8, 1.2, 6.1, -99.0, -99.0, -99.0], [0.9, 1.3, 3.4, -99.0, -99.0, -99.0]]
    ),
    "CloudLayerTop": np.array(
        [[1.2, 3.2, 8.1, -99, -99, -99], [1.5, 2.1, 5.5, -99.0, -99.0, -99.0]]
    ),
    "CloudFraction": np.array(
        [[0.9, 0.3, 0.3, -99.0, -99.0, -99.0], [0.1, 0.3, 0.0, -99.0, -99.0, -99.0]]
    ),
    "Cloudlayer": np.array([3, 3]),
    "Profile_time": np.array([3600, 7200]),
    "TAI_start": np.array([3600]),
}


@pytest.fixture
def mock_read_cloudsat_hdf4():
    def mock_function(sample_filepath):
        return sample_data

    with patch(
        "cbase.data_readers.cloudsat_products.read_cloudsat_hdf4", mock_function
    ):
        yield


def test_get_top_height():
    cth = sample_data["CloudLayerTop"]
    expected_result = np.array([8.1, 5.5])
    assert np.array_equal(get_top_height(cth), expected_result)


def test_get_base_height():
    cbh = sample_data["CloudLayerBase"]
    expected_result = np.array([0.8, 0.9])
    assert np.array_equal(get_base_height(cbh), expected_result)


def test_get_cloud_fraction():
    cf = sample_data["CloudFraction"]
    expected_result = np.array([0.9, 0.3])
    assert np.array_equal(get_cloud_fraction(cf), expected_result)


def test_get_time():
    expected_result = np.array(
        [
            datetime(1993, 1, 1, 2, 0, 0, tzinfo=pytz.UTC),
            datetime(1993, 1, 1, 3, 0, 0, tzinfo=pytz.UTC),
        ]
    )
    print(get_time(sample_data))
    assert np.array_equal(get_time(sample_data), expected_result)
