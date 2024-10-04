#!/bin/bash -l                                                                                                    
#SBATCH -A safnwc
#SBATCH -n 3 # lower case n!
#SBATCH -t 0:65:00 #Maximum time of 65 minutes
#SBATCH --array=0-200	 #Run as array job 0 to 10 but at most 4 at one time

module load Anaconda/2023.09-0-hpc1
conda activate cbase
source /home/sm_indka/pps_nwp/source_me.bash

# Get the Nth line from my_files.txt
csatfile_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" < cloudsat_matches.txt)
dardarfile_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" < dardar_matches.txt)
vgacfile_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" < vgac_matches.txt)
nwpfile_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" < nwp_matches.txt)

python run_process.py -CFILE ${csatfile_name} -DFILE ${dardarfile_name} -VFILE ${vgacfile_name} -NFILE ${nwpfile_name}
