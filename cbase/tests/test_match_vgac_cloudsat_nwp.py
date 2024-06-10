from unittest.mock import MagicMock
from datetime import datetime, timedelta
import numpy as np
import pytest
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData
from cbase.matching.match_vgac_cloudsat_nwp import DataMatcher
from cbase.tests.mock_data import (
    mock_lon,
    mock_lat,
    mock_interp_cloud_base,
    mock_interp_cloud_top,
    mock_interp_flag_base,
    mock_interp_cloud_fraction,
    mock_interp_vis_optical_depth,
)


@pytest.fixture
def cloudsat_data():
    """Mock CloudsatData object"""
    cloudsat = MagicMock(spec=CloudsatData)
    cloudsat.time = np.array(
        [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i) for i in range(10)]
    )
    cloudsat.latitude = np.linspace(-37, -36.85, 10)
    cloudsat.longitude = np.linspace(168.6, 169.0, 10)
    cloudsat.cloud_base = np.arange(200, 1200, 100)
    cloudsat.cloud_top = np.arange(1000, 2000, 100)
    cloudsat.cloud_layers = np.repeat(1, 10)
    cloudsat.flag_base = np.array([1, 2, 3, 1, 2, 2, 1, 1, 2, 2])
    cloudsat.cloud_fraction = np.arange(0, 1, 0.1)
    cloudsat.vis_optical_depth = np.arange(10, 20, 1)
    cloudsat.name = "cloudsat_file.hdf"
    return cloudsat


@pytest.fixture
def vgac_data():
    """mock VGACData object"""
    vgac = MagicMock(spec=VGACData)
    vgac.time = np.array(
        [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i) for i in range(5)]
    )
    vgac.latitude = mock_lat
    vgac.longitude = mock_lon
    return vgac


@pytest.fixture
def setup_data_matcher(cloudsat_data, vgac_data):
    """mock DataMatcher object"""
    return DataMatcher(cloudsat_data, vgac_data, None)


def test_check_overlapping_time(setup_data_matcher):
    """test overlapping orbits"""
    assert setup_data_matcher.check_overlapping_time() is True
    setup_data_matcher.cloudsat.time = np.array(
        [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 1)]
    )
    setup_data_matcher.vgac.time = np.array([datetime(2024, 1, 1, 0, 2)])
    assert setup_data_matcher.check_overlapping_time() is False


def test_match_vgac_cloudsat(setup_data_matcher):
    """test interpolation of cloudsat to vgac grid"""
    dm = setup_data_matcher  # class obj data matcher

    dm.match_vgac_cloudsat()
    print(dm.collocated_data["vis_optical_depth"])
    print(dm.collocated_data["cloud_fraction"])
    assert np.array_equal(dm.collocated_data["cloud_base"], mock_interp_cloud_base)
    assert np.array_equal(dm.collocated_data["cloud_top"], mock_interp_cloud_top)
    assert np.array_equal(dm.collocated_data["flag_base"], mock_interp_flag_base)
    assert np.array_equal(
        dm.collocated_data["vis_optical_depth"], mock_interp_vis_optical_depth
    )
