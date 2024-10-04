import argparse
import glob
import os
from pathlib import Path
from cbase.matching.add_ATMS import MatchATMSVGAC


def main():
    parser = argparse.ArgumentParser(
        description="Match ATMS data to VGAC scenes for CNN processing."
    )
    # argument for vgacfile
    parser.add_argument(
        "vgacpath", type=str, help="Path to the VGAC files used for matching"
    )
    args = parser.parse_args()
    vgacfiles = glob.glob(os.path.join(args.vgacpath, "cnn*2012*"))

    for vgacfile in vgacfiles:
        print(vgacfile)
        matcher = MatchATMSVGAC(Path(vgacfile))
        matcher.matching()


if __name__ == "__main__":
    main()
