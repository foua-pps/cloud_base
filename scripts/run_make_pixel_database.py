import os
from random import shuffle
import xarray as xr
from glob import glob
import argparse
from cbase.matching.make_pixel_based_database import make_pixel_dataset


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Process CNN based scenes to training, validation, and testing pixel datasets."
    )

    parser.add_argument(
        "--inpath",
        type=str,
        required=True,
        help="Path to the input data directory containing NetCDF files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Ratio of files used for training (default: 0.6).",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of files used for validation (default: 0.2).",
    )

    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        help="Output path for the generated datasets.",
    )

    return parser.parse_args()


def main():
    # Parse the command-line arguments
    args = parse_args()

    input_files = glob(os.path.join(args.inpath, "cnn*nc"))

    if len(input_files) == 0:
        print("No input files found with the given pattern.")
        return

    shuffle(input_files)

    total_files = len(input_files)
    s1 = int(total_files * args.train_ratio)
    s2 = int(total_files * args.val_ratio)

    # File splits for train, validation, and test
    train_files = input_files[:s1]
    val_files = input_files[s1 : s1 + s2]
    test_files = input_files[s1 + s2 :]

    # Generate and save the training dataset
    print("Processing training dataset")
    da_train = make_pixel_dataset(train_files)
    xr.Dataset.from_dict(da_train).to_netcdf(
        os.path.join(
            args.outpath, "training_data_vgac_cloudsat_caliop_atms_pps_pixel.nc"
        )
    )

    # Generate and save the validation dataset
    print("Processing validation dataset")
    da_val = make_pixel_dataset(val_files)
    xr.Dataset.from_dict(da_val).to_netcdf(
        os.path.join(
            args.outpath, "validation_data_vgac_cloudsat_caliop_atms_pps_pixel.nc"
        )
    )

    # Generate and save the testing dataset
    print("Processing testing dataset")
    da_test = make_pixel_dataset(test_files)
    xr.Dataset.from_dict(da_test).to_netcdf(
        os.path.join(
            args.outpath, "testing_data_vgac_cloudsat_caliop_atms_pps_pixel.nc"
        )
    )


if __name__ == "__main__":
    main()

# USAGE python run_make_pixel_databse.py --inpath /nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/split_data/ --outpath /nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/split_data/
# can provide train validation ratio, otherwise default are used
