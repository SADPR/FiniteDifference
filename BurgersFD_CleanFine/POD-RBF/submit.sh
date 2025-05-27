#!/bin/bash
#SBATCH --job-name=pod_rbf_job           # Job name
#SBATCH --output=output_%j.log           # Standard output and error log (%j expands to jobID)
#SBATCH --error=error_%j.log             # Error log file
#SBATCH --time=48:00:00                  # Time limit (adjust as needed)
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
python3 /scratch/users/sadpr/BurgersFD_CleanFine/POD-RBF/pod_rbf_reconstruction_nearest_neighbours_dynamic_hyperparameter_exploration.py
