I1, I2 = 0, 800  # index to select the part of scan where collocations can lie
COLLOCATION_THRESHOLD = 4  # km
SWATH_CENTER = 400  # VGAC SWATH CENTER
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
    "t250",
    "t500",
    "t700",
    "t850",
    "t900",
    "q250",
    "q500",
    "q700",
    "q850",
    "q900",
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
    "validation_height_base",
    "validation_height",
    "cloud_fraction",
    "cloud_layers",
    "vis_optical_depth",
]
