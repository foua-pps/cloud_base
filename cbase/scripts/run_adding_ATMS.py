import argparse
import glob
import os
from pathlib import Path
from cbase.matching.add_ATMS import MatchATMSVGAC


def main():
    parser = argparse.ArgumentParser(
        description="Match ATMS data to VGAC scenes for CNN processing."
    )
    parser.add_argument(
        "--vgacpath",
        type=str,
        required=True,
        help="Path to the VGAC files to be populated with ATMS data",
    )
    parser.add_argument(
        "--atmspath",
        type=str,
        required=True,
        help="Path to the data directory containing ATMS input files",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        help="Path to the output data directory containing ATMS populated files",
    )
    args = parser.parse_args()
    vgacfiles = glob.glob(os.path.join(args.vgacpath, "cnn*2012*"))
    for vgacfile in vgacfiles:
        print(vgacfile)
        matcher = MatchATMSVGAC(Path(vgacfile), Path(args.atmspath), Path(args.outpath))
        matcher.matching()


if __name__ == "__main__":
    main()
