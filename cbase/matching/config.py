I1, I2 = 0, 800  # index to select the part of scan where collocations can lie
COLLOCATION_THRESHOLD = 4  # km
TIME_WINDOW = [-200, 200]  # just to find cloudsat swath closest to VGAC
XIMAGE_SIZE = 20
YIMAGE_SIZE = 100
PADDING = 1.0  # some extra margin when interpolating ERA5 (in degrees)
TIME_DIFF_ALLOWED = 30  # minutes
SECS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
PIXEL_DATA = False
CNN_DATA = True
CLOUDSAT_PATH = "/home/a002602/data/cloud_base/cloudsat/"
VGAC_PATH = "/home/a002602/data/cloud_base/vgac/"
NWP_PATH = "/home/a002602/data/cloud_base/NWP/"
OUTPUT_PATH = "/nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/"

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
    "rh100",
    "rh250",
    "rh400",
    "rh500",
    "rh700",
    "rh850",
    "rh900",
    "rh950",
    "rh1000",
    "snow_mask",
    "t_sea",
    "t_land",
    # "ice_mask",
]
CNN_VGAC_PARAMETERS = [
    "time",
    "latitude",
    "longitude",
    "M05",
    "M07",
    "M12",
    "M15",
    "M16",
    "ctt",
    "cth",
    "ctp",
    "ctp_quality",
    "ct",
    "ct_quality",
    "cmic_phase",
    "cmic_quality",
    "cmic_lwp",
    "elevation",
    "land_use",
]

CNN_MATCHED_PARAMETERS = [
    "cloud_top",
    "cloud_base",
    "cloud_layers",
    "vis_optical_depth",
    "flag_base",
    "cloud_fraction",
]

ATMS_PARAMETERS = ["tb17", "tb18", "tb19", "tb20", "tb21", "tb22", "view_ang"]
