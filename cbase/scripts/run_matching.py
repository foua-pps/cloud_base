from pathlib import Path
from sys import argv
import argparse
from cbase.matching.match_csat_vgac_nwp_filenames import (
    get_matching_cloudsat_vgac_nwp_files,
)
from cbase.data_readers import cloudsat, viirs, era5
from cbase.matching.match_vgac_cloudsat_nwp import DataMatcher


def process(cloudsat_file: Path, vgac_file: Path, nwp_file: Path):
    """main process"""
    # read in data
    vgc = viirs.VGACData.from_file(vgac_file)
    nwp = era5.Era5.from_file(nwp_file)
    cld = cloudsat.CloudsatData.from_file(cloudsat_file)

    # create matching object
    dm = DataMatcher(cld, vgc, nwp)
    dm.match_vgac_cloudsat()
    dm.create_cnn_dataset_with_nwp()


def cli(args_list: list[str]) -> None:
    """command line args"""
    parser = argparse.ArgumentParser(
        description="Find matching Cloudsat, VGAC and NWP (ERA5) files"
        "or provide individual matched filenames"
    )

    parser.add_argument(
        "-CPATH",
        "--available_cloudsat_files",
        type=str,
        nargs="+",
        help="Full path to Cloudsat level1b file(s) which you want to process",
        metavar="CLOUDSAT_FILES_PATH",
    )
    parser.add_argument(
        "-VPATH",
        "--available_vgac_files",
        type=str,
        nargs="+",
        help="Full path to available VGAC file(s) which you want to process",
        metavar="VGAC_FILES_PATH",
    )
    parser.add_argument(
        "-NPATH",
        "--available_nwp_files",
        type=str,
        nargs="+",
        help="Full path to available NWP (ERA5) file(s) which you want to",
        metavar="NWP_FILES_PATH",
    )
    parser.add_argument(
        "-CFILE",
        "--matched_cloudsat_file",
        type=str,
        nargs="+",
        help="Matched Cloudsat level1b file(s) which you want to process",
        metavar="CLOUDSAT_FILE",
    )
    parser.add_argument(
        "-VFILE",
        "--matched_vgac_file",
        type=str,
        nargs="+",
        help="Matched VGAC file(s) which you want to process",
        metavar="VGAC_FILE",
    )
    parser.add_argument(
        "-NFILE",
        "--matched_nwp_file",
        type=str,
        nargs="+",
        help="Matched NWP (ERA5) file(s) which you want to",
        metavar="NWP_FILE",
    )
    args = parser.parse_args(args_list)

    # Check mutual exclusivity
    options = [
        args.available_cloudsat_files,
        args.available_vgac_files,
        args.available_nwp_files,
    ]
    alternatives = [
        args.matched_cloudsat_file,
        args.matched_vgac_file,
        args.matched_nwp_file,
    ]

    opt_flag = sum(bool(opt) for opt in options) == 3
    alt_flag = sum(bool(alt) for alt in alternatives) == 3
    print(opt_flag, alt_flag)
    if opt_flag and alt_flag:
        parser.error(
            "OBS! Please provide only one set of options"
            "either [-CPATH, -VPATH, -NPATH] or [-CFILE, -VFILE, -NFILE]"
        )
    elif opt_flag:
        cloudsat_files, vgac_files, nwp_files = get_matching_cloudsat_vgac_nwp_files(
            cfiles=[Path(f) for f in args.available_cloudsat_files],
            vfiles=[Path(f) for f in args.available_vgac_files],
            nfiles=[Path(f) for f in args.available_nwp_files],
        )
        if cloudsat_files:
            for cloudsat_file, vgac_file, nwp_file in zip(
                cloudsat_files, vgac_files, nwp_files
            ):
                print(f"Processing files: {cloudsat_file}, {vgac_file}, {nwp_file}")
                process(cloudsat_file, vgac_file, nwp_file)
        else:
            raise ValueError("Tyvärr! No Matching files found!")

    elif alt_flag:
        print(alternatives)
        process(
            Path(args.matched_cloudsat_file[0]),
            Path(args.matched_vgac_file[0]),
            Path(args.matched_nwp_file[0]),
        )
    else:
        parser.error(
            "OBS! Options [-CPATH, -VPATH, and -NPATH] must occur together"
            "or Options [-CFILE, -VFILE, -NFILE] must occur together"
        )


# read in command line args and get process
cli(argv[1:])
