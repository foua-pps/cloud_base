import os
import numpy as np
import xarray as xr
from unittest.mock import MagicMock
import pytest
from datetime import datetime
from cbase.data_readers.viirs import VGACData
from cbase.data_readers.cloudsat import CloudsatData


@pytest.fixture
def cloudsat_data():
    # Mocking CloudsatData object for testing
    cloudsat = MagicMock(spec=CloudsatData)
    cloudsat.time = np.array([datetime(2024, 5, 1, 12, 0, 0)])
    cloudsat.cloud_base = np.array([[1000.0]])
    cloudsat.cloud_top = np.array([[2000.0]])
    cloudsat.latitude = np.array([[0.0]])
    cloudsat.longitude = np.array([[0.0]])
    return cloudsat


@pytest.fixture
def vgac_data():
    # Mocking VGACData object for testing
    vgac = MagicMock(spec=VGACData)
    vgac.time = np.array([datetime(2024, 5, 1, 12, 0, 0)])
    vgac.latitude = np.array([[0.0]])
    vgac.longitude = np.array([[0.0]])
    return vgac
