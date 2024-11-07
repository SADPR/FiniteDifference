import os
import numpy as np
import run_fom
import run_HPROM_ecm
import run_HRNM_ecm
import run_POD_RBF_HPROM_ecm

# Define the mu1 and mu2 parameters
mu1_values = [5.19, 4.56, 4.75]
mu2_values = [0.026, 0.019, 0.02]

# Check if FOM times/errors exist in rom_results.npz
if os.path.exists('rom_results.npz'):
    fom_data = np.load('rom_results.npz', allow_pickle=True)
    fom_times = list(fom_data.get('fom_times', [None] * len(mu1_values)))
    fom_errors = list(fom_data.get('fom_errors', [None] * len(mu1_values)))
else:
    fom_times = [None] * len(mu1_values)
    fom_errors = [None] * len(mu1_values)

# If FOM times/errors were not found, check rom_results_hprom.npz
if all(x is None for x in fom_times) and os.path.exists('rom_results_hprom.npz'):
    fom_hprom_data = np.load('rom_results_hprom.npz', allow_pickle=True)
    fom_times = list(fom_hprom_data.get('fom_times', fom_times))
    fom_errors = list(fom_hprom_data.get('fom_errors', fom_errors))

# Load HPROM results from rom_results_hprom.npz if available
if os.path.exists('rom_results_hprom.npz'):
    hprom_data = np.load('rom_results_hprom.npz', allow_pickle=True)
    hprom_times = list(hprom_data.get('hprom_times', [None] * len(mu1_values)))
    hprom_errors = list(hprom_data.get('hprom_errors', [None] * len(mu1_values)))
    hrnm_times = list(hprom_data.get('hrnm_times', [None] * len(mu1_values)))
    hrnm_errors = list(hprom_data.get('hrnm_errors', [None] * len(mu1_values)))
    pod_rbf_hprom_times = list(hprom_data.get('pod_rbf_hprom_times', [None] * len(mu1_values)))
    pod_rbf_hprom_errors = list(hprom_data.get('pod_rbf_hprom_errors', [None] * len(mu1_values)))
else:
    hprom_times = [None] * len(mu1_values)
    hprom_errors = [None] * len(mu1_values)
    hrnm_times = [None] * len(mu1_values)
    hrnm_errors = [None] * len(mu1_values)
    pod_rbf_hprom_times = [None] * len(mu1_values)
    pod_rbf_hprom_errors = [None] * len(mu1_values)

# Helper function to check if the .npy file exists
def check_if_exists(file_name):
    return os.path.exists(file_name)

# Loop through each combination of mu1 and mu2
for i, (mu1, mu2) in enumerate(zip(mu1_values, mu2_values)):
    print(f"Running for mu1 = {mu1}, mu2 = {mu2}")

    # Run FOM if results are missing or need to be updated
    fom_file = f"hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(fom_file) or fom_times[i] is None:
        print(f"FOM results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running FOM...")
        fom_time, fom_error = run_fom.main(mu1, mu2)
        fom_times[i] = fom_time
        fom_errors[i] = fom_error
    else:
        print(f"FOM results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping FOM run.")

    # Run HPROM if results are missing or need to be updated
    hprom_file = f"hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(hprom_file) or hprom_times[i] is None:
        print(f"HPROM results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running HPROM...")
        hprom_time, hprom_error = run_HPROM_ecm.main(mu1, mu2)
        hprom_times[i] = hprom_time
        hprom_errors[i] = hprom_error
    else:
        print(f"HPROM results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping HPROM run.")

    # Run HRNM if results are missing or need to be updated
    hrnm_file = f"hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(hrnm_file) or hrnm_times[i] is None:
        print(f"HRNM results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running HRNM...")
        hrnm_time, hrnm_error = run_HRNM_ecm.main(mu1, mu2)
        hrnm_times[i] = hrnm_time
        hrnm_errors[i] = hrnm_error
    else:
        print(f"HRNM results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping HRNM run.")

    # Run POD-RBF HPROM if results are missing or need to be updated
    pod_rbf_hprom_file = f"pod_rbf_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(pod_rbf_hprom_file) or pod_rbf_hprom_times[i] is None:
        print(f"POD-RBF HPROM results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running POD-RBF HPROM...")
        pod_rbf_hprom_time, pod_rbf_hprom_error = run_POD_RBF_HPROM_ecm.main(mu1, mu2)
        pod_rbf_hprom_times[i] = pod_rbf_hprom_time
        pod_rbf_hprom_errors[i] = pod_rbf_hprom_error
    else:
        print(f"POD-RBF HPROM results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping POD-RBF HPROM run.")

# Save the updated results in the .npz file, overwriting old results
np.savez('rom_results_hprom.npz',
         fom_times=fom_times, fom_errors=fom_errors,
         hprom_times=hprom_times, hprom_errors=hprom_errors,
         hrnm_times=hrnm_times, hrnm_errors=hrnm_errors,
         pod_rbf_hprom_times=pod_rbf_hprom_times, pod_rbf_hprom_errors=pod_rbf_hprom_errors)

print("Results updated and saved to 'rom_results_hprom.npz'")





