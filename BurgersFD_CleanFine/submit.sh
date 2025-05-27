#!/bin/bash

# Set the Python script name without the extension
PYTHON_SCRIPT="run_HPROM"
SCRIPT_NAME=$(basename $PYTHON_SCRIPT .py)

#SBATCH --job-name=${SCRIPT_NAME}           # Job name based on Python script name
#SBATCH --output=${SCRIPT_NAME}_output_%j.log  # Standard output log (%j expands to jobID)
#SBATCH --error=${SCRIPT_NAME}_error_%j.log    # Error log file
#SBATCH --time=48:00:00                        # Time limit (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=cfarhat                    # Use the owners partition
#SBATCH --mail-type=END,FAIL                   # Send email on job completion or failure
#SBATCH --mail-user=sadpr@stanford.edu         # Email for notifications

# Set environment variables for multi-threading
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export BLIS_NUM_THREADS=24
export GOTO_NUM_THREADS=24

# Load the Python module
module load python/3.9.0

# Activate the virtual environment (with all dependencies already installed)
source ~/myenv/bin/activate

# Run the Python script
python3 /scratch/users/sadpr/BurgersFD_CleanFine/${PYTHON_SCRIPT}.py

