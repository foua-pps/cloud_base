from datetime import datetime, timedelta
from cbase.matching.get_matching_csat_vgac_filenames import (
    create_datetime_from_year_doy_hour_minute,
    get_cloudsat_time,
    get_vgac_time,
    is_valid_match,
    get_matching_cloudsat_vgac_nwp_files,
)

# Mocked lists for testing
CLOUDSAT_FILES = ["2018150015649_64371_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"]
VGAC_FILES = [
    "VGAC_VJ102MOD_A2018150_0130_n002738_K005.nc",
    "VGAC_VJ102MOD_A2018145_0318_n002668_K005.nc",
]
NWP_FILES = [
    "GAC_ECMWF_ERA5_201805300100+000H00M",
    "GAC_ECMWF_ERA5_201805310100+000H00M",
]


def test_create_datetime_from_year_doy_hour_minute():
    # Test create_datetime_from_year_doy_hour_minute function
    dt = create_datetime_from_year_doy_hour_minute(2022, 150, 12, 34)
    assert isinstance(dt, datetime)
    assert dt.year == 2022
    assert dt.day == 30
    assert dt.month == 5
    assert dt.hour == 12
    assert dt.minute == 34


def test_get_cloudsat_time():
    # Test get_cloudsat_time function
    ctime = get_cloudsat_time(CLOUDSAT_FILES[0])
    truth = datetime(2018, 5, 30, 1, 56)
    assert ctime == truth


def test_get_vgac_time():
    # Test get_vgac_time function
    vtime = get_vgac_time(VGAC_FILES[0])
    truth = datetime(2018, 5, 30, 1, 30)
    assert vtime == truth


def test_is_valid_match():
    # Test is_valid_match function
    ctime = get_vgac_time(VGAC_FILES[0])
    vtime = get_cloudsat_time(CLOUDSAT_FILES[0])
    print(ctime, vtime)
    assert is_valid_match(ctime, vtime)


def test_get_matching_cloudsat_vgac_nwp_files():
    # Test get_matching_cloudsat_vgac_files function
    csat_files, vgac_files, nwp_files = get_matching_cloudsat_vgac_nwp_files(
        CLOUDSAT_FILES, VGAC_FILES, NWP_FILES
    )
    assert csat_files == CLOUDSAT_FILES
    assert nwp_files == [NWP_FILES[0]]
    assert vgac_files == [VGAC_FILES[0]]
