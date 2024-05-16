from pathlib import Path
from sys import argv
import argparse
from cbase.matching.match_csat_vgac_nwp_filenames import (
    get_matching_cloudsat_vgac_nwp_files,
)


def get_matches(cloudsat_files: list, vgac_files: list, nwp_files: list):

    def _write_to_file(filename: str, items: list):
        with open(filename, "w") as file:
            file.write("\n".join(items))

    matched_csat, matched_vgac, matched_nwp = get_matching_cloudsat_vgac_nwp_files(
        cloudsat_files, vgac_files, nwp_files
    )
    if matched_csat:
        _write_to_file("cloudsat_matches.txt", matched_csat)
        _write_to_file("vgac_matches.txt", matched_vgac)
        _write_to_file("nwp_matches.txt", matched_nwp)
    else:
        raise ValueError("No Matches found")


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
        required=True,
    )
    parser.add_argument(
        "-VPATH",
        "--available_vgac_files",
        type=str,
        nargs="+",
        help="Full path to available VGAC file(s) which you want to process",
        metavar="VGAC_FILES_PATH",
        required=True,
    )
    parser.add_argument(
        "-NPATH",
        "--available_nwp_files",
        type=str,
        nargs="+",
        help="Full path to available NWP (ERA5) file(s) which you want to",
        metavar="NWP_FILES_PATH",
        required=True,
    )

    args = parser.parse_args(args_list)
    return (
        args.available_cloudsat_files,
        args.available_vgac_files,
        args.available_nwp_files,
    )


# read in command line args and get process
cfiles, vfiles, nfiles = cli(argv[1:])
get_matches(cfiles, vfiles, nfiles)
