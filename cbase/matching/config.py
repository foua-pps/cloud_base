I1, I2 = 0, 800  # index to select the part of scan where collocations can lie
COLLOCATION_THRESHOLD = 4  # km
SWATH_CENTER = 400  # VGAC SWATH CENTER
TIME_WINDOW = [-5, 20]
IMAGE_SIZE = 32
PADDING = 1.0  # some extra margin when interpolating ERA5 (in degrees)
CNN_NWP_PARAMETERS = [
    # "h_2meter",
    # "t_2meter",
    # "p_surface",
    # "z_surface",
    # "ciwv",
    # "tclw",
    # "pressure_levels",
    "t250",
    # "t500",
    # "t700",
    # "t850",
    # "t900",
    # "q250",
    # "q500",
    # "q700",
    # "q850",
    # "q900",
]
CNN_SAT_PARAMETERS = [
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
    "validation_height_base",
]
