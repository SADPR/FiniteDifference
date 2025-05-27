#!/bin/bash
#SBATCH --job-name=pod_rbf_hrom_job           # Job name
#SBATCH --output=output_%j.log           # Standard output and error log (%j expands to jobID)
#SBATCH --error=error_%j.log             # Error log file
#SBATCH --time=04:00:00                  # Time limit (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=cfarhat               # Use the owners partition
#SBATCH --mail-type=END,FAIL             # Send email on job completion or failure
#SBATCH --mail-user=sadpr@stanford.edu   # Email for notifications

# Load the Python module
module load python/3.9.0

# Activate the virtual environment (with all dependencies already installed)
source ~/myenv/bin/activate

# Run your Python script
bash
python3 /scratch/users/sadpr/BurgersFD_CleanCoarse/run_POD_RBF_HPROM.py
