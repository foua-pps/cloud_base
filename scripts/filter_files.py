import xarray as xr
from glob import glob
import numpy as np
from tqdm import tqdm
import os

training_data_path = (
    "/nobackup/smhid20/users/sm_indka/collocated_data/split_data/"
)
output_path = (
    "/nobackup/smhid20/users/sm_indka/collocated_data/split_data/filtered_data/"
)
input_files = glob(training_data_path + "/*.nc*")[:]
good_files = []
for i, file in tqdm(enumerate(input_files[:])):
    with xr.open_dataset(file) as ds:
        mask = (ds.cloud_base.values > 0) & (ds.cloud_base.values < 5000)
        if np.sum(mask) >= 50:
            print(os.path.join(output_path, os.path.basename(file)))
            os.rename(file, os.path.join(output_path, os.path.basename(file)))
