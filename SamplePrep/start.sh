#! /bin/bash

# Script to run everything
# Source environment/bashrc
source ~/.bashrc

# Replace the following with your own environment location
source activate /home/sinhai/anaconda3/envs/tf

source /usr/local/gromacs/bin/GMXRC

# 1 Clean Files, calculate SASA, calculate SAP:
#sh run_sasa.sh
#cd ep-sap
#sh sap_calc.sh
#cd ..

# 2 Create Occupancy Maps:
#python VoxelParse_SAP.py

# 3 Divide Into Patches:
python PartitionPatches_SAP.py
