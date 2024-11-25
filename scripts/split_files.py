import xarray as xr
import glob
import os
from tqdm import tqdm


def split_files(files):
    for ix, file in tqdm(enumerate(files)):
        with xr.open_dataset(file) as ds:
            for i in range(len(ds.nscene)):
                outfile = os.path.basename(file)[:-3] + f"_{i}.nc"
                # if np.sum(ds.sel(nscene=i).cloud_base.values < 60):
                ds.sel(nscene=i).to_netcdf(os.path.join(outpath, outfile))


files = glob.glob(
    "/nobackup/smhid20/users/sm_indka/collocated_data/including_clearsky/*cnn*.nc"
)
outpath = "/nobackup/smhid20/users/sm_indka/collocated_data/including_clearsky/split_data/"
split_files(files)
