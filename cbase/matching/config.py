I1, I2 = 0, 800  # index to select the part of scan where collocations can lie
COLLOCATION_THRESHOLD = 4  # km
TIME_WINDOW = [-80, 80]
IMAGE_SIZE = 32
PADDING = 1.0  # some extra margin when interpolating ERA5 (in degrees)
TIME_DIFF_ALLOWED = 60  # minutes
SECS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
CLOUDSAT_PATH = "/home/a002602/data/cloud_base/cloudsat/"
VGAC_PATH = "/home/a002602/data/cloud_base/vgac/"
NWP_PATH = "/home/a002602/data/cloud_base/NWP/"
OUTPUT_PATH = "/home/a002602/data/cloud_base/collocated_data/"

CNN_NWP_PARAMETERS = [
    "h_2meter",
    "t_2meter",
    "p_surface",
    "z_surface",
    "ciwv",
    "tclw",
    # "pressure_levels",
    "t100",
    "t250",
    "t400",
    "t500",
    "t700",
    "t850",
    "t900",
    "t950",
    "t1000",
    "q100",
    "q250",
    "q400",
    "q500",
    "q700",
    "q850",
    "q900",
    "q950",
    "q1000",
    "snow_mask",
    "t_sea",
    "t_land",
    "ice_mask",
]
CNN_VGAC_PARAMETERS = [
    "latitude",
    "longitude",
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "M06",
    "M07",
    "M08",
    "M09",
    "M10",
    "M11",
    "M12",
    "M13",
    "M14",
    "M15",
    "M16",
]
CNN_MATCHED_PARAMETERS = [
    "cloud_top",
    "cloud_base",
    "cloud_layers",
    "vis_optical_depth",
    "flag_base",
    "cloud_fraction",
]
