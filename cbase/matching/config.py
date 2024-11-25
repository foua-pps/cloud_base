I1, I2 = 0, 800  # index to select VGAC scan where collocations can lie
COLLOCATION_THRESHOLD = 4  # km
TIME_WINDOW = [-200, 200]  # just to find cloudsat swath closest to VGAC
XIMAGE_SIZE = 32
YIMAGE_SIZE = 32
PADDING = 1.0  # some extra margin when interpolating ERA5 (in degrees)
TIME_DIFF_ALLOWED = 5  # minutes
SECS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
CLOUDSAT_PATH = "/home/a002602/data/cloud_base/cloudsat/"
VGAC_PATH = "/home/a002602/data/cloud_base/vgac/"
NWP_PATH = "/home/a002602/data/cloud_base/NWP/"
ATMS_PATH = "/nobackup/smhid17/proj/foua/data/satellit/ATMS/"
OUTPUT_PATH = (
    "/nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/cth_with16p_84p"
)

CNN_NWP_PARAMETERS = [
    "h_2meter",
    "t_2meter",
    "p_surface",
    "z_surface",
    "ciwv",
    "tclw",
    "t250",
    "t400",
    "t500",
    "t700",
    "t850",
    "t900",
    "t950",
    "t1000",
    "rh250",
    "rh400",
    "rh500",
    "rh700",
    "rh850",
    "rh900",
    "rh950",
    "rh1000",
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
]
CNN_VGAC_PARAMETERS = [
    "time",
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
CNN_VGAC_PPS_PARAMETERS = [
    "time",
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
    "ctt",
    "cth",
    "ctp",
    "ctp_quality",
    "ct",
    "ct_quality",
    "ctp16",
    "cth16",
    "ctt16",
    "ctp84",
    "cth84",
    "ctt84",
    "cmic_phase",
    "cmic_quality",
    "cmic_lwp",
    "cmic_cot",
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
