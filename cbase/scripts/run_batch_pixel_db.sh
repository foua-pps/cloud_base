#!/bin/bash -l                                                                                                    
#SBATCH -A safnwc
#SBATCH -n 3 # lower case n!
#SBATCH -t 1:00:0

module load Anaconda/2023.09-0-hpc1
conda activate cbase
source /home/sm_indka/pps_nwp/source_me.bash
python run_adding_ATMS.py --vgacpath /nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/split_data/ --atmspath /nobackup/smhid17/proj/foua/data/satellit/ATMS/ --outpath /nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/split_data/ATMS/
python run_make_pixel_database.py --inpath /nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/split_data/ATMS/ --outpath /nobackup/smhid20/users/sm_indka/collocated_data/VGAC_PPS/pixel_database --train-ratio 0.01 
