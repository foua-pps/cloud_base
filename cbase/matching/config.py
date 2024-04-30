I1, I2 = 300, 500  # index to select the part of scan where collocations can lie
COLLOCATION_THRESHOLD = 4  # km
TIME_WINDOW = [-5, 20]
IMAGE_SIZE = 32
PADDING = 1.0  # some extra margin when interpolating ERA5 (in degrees)
CNN_NWP_PARAMETERS = ["tclw"]
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
