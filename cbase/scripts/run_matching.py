from pathlib import Path
import multiprocessing
from sys import argv
import argparse
from cbase.matching.match_csat_vgac_nwp_filenames import (
    get_matching_cloudsat_vgac_nwp_files,
)
from cbase.data_readers import cloudsat, viirs, era5
from cbase.matching.match_vgac_cloudsat_nwp import DataMatcher

# some demo filenames to check the processing
INPUT_FILENAMES = {
    "CLOUDSAT_FILE": Path(
        "/home/a002602/data/cloud_base/cloudsat/2018150015649_64371_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"
    ),
    "VGAC_FILE": Path(
        "/home/a002602/data/cloud_base/vgac/VGAC_VJ102MOD_A2018150_0130_n002738_K005.nc"
    ),
    "NWP_FILE": Path(
        "/home/a002602/data/cloud_base/NWP/GAC_ECMWF_ERA5_201801010100+000H00M"
    ),
}


def process(cloudsat_file: Path, vgac_file: Path, nwp_file: Path):
    """main process"""
    vgc = viirs.VGACData.from_file(vgac_file)
    nwp = era5.Era5.from_file(nwp_file.absolute().as_posix())
    cld = cloudsat.CloudsatData.from_file(cloudsat_file.absolute().as_posix())

    # create matching object
    dm = DataMatcher(cld, vgc, nwp)
    dm.match_vgac_cloudsat()
    dm.create_cnn_dataset_with_nwp()


def cli(args_list: list[str]) -> None:
    """command line args"""
    parser = argparse.ArgumentParser(
        description="Find matching Cloudsat, VGAC and NWP (ERA5) files"
    )
    parser.add_argument(
        "-C",
        "--cloudsat_files",
        type=str,
        nargs="+",
        help="Full path to Cloudsat level1b file(s)",
        metavar="CLOUDSAT_FILE",
        required=True,
    )
    parser.add_argument(
        "-V",
        "--vgac_files",
        type=str,
        nargs="+",
        help="Full path to available VGAC file(s).",
        metavar="VGAC_FILE",
        required=True,
    )
    parser.add_argument(
        "-N",
        "--nwp_files",
        type=str,
        nargs="+",
        help="Full path to available NWP (ERA5) file(s).",
        metavar="NWP_FILE",
        required=True,
    )
    args = parser.parse_args(args_list)
    return get_matching_cloudsat_vgac_nwp_files(
        cfiles=[Path(f) for f in args.cloudsat_files],
        vfiles=[Path(f) for f in args.vgac_files],
        nfiles=[Path(f) for f in args.nwp_files],
    )


# read in command line args and get list of files to process
cloudsat_files, vgac_files, nwp_files = cli(argv[1:])

# Create a multiprocessing Pool
with multiprocessing.Pool() as pool:
    # Map the function to the list of files
    pool.starmap(
        process,
        zip(cloudsat_files, vgac_files, nwp_files),
    )

# process(
#     INPUT_FILENAMES["CLOUDSAT_FILE"],
#     INPUT_FILENAMES["VGAC_FILE"],
#     INPUT_FILENAMES["NWP_FILE"],
# )
