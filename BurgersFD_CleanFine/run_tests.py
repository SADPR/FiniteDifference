import os
import numpy as np
import run_fom
import run_prom
#import run_RNM
#import run_POD_RBF

# Define the mu1 and mu2 parameters
mu1_values = [5.19, 4.56, 4.75]
mu2_values = [0.026, 0.019, 0.02]

# Load existing results if they exist
if os.path.exists('rom_results.npz'):
    existing_data = np.load('rom_results.npz', allow_pickle=True)
    fom_times = list(existing_data['fom_times'])
    fom_errors = list(existing_data['fom_errors'])
    prom_times = list(existing_data['prom_times'])
    prom_errors = list(existing_data['prom_errors'])
    rnm_times = list(existing_data['rnm_times'])
    rnm_errors = list(existing_data['rnm_errors'])
    pod_rbf_times = list(existing_data['pod_rbf_times'])
    pod_rbf_errors = list(existing_data['pod_rbf_errors'])
else:
    fom_times = [None] * len(mu1_values)
    fom_errors = [None] * len(mu1_values)
    prom_times = [None] * len(mu1_values)
    prom_errors = [None] * len(mu1_values)
    rnm_times = [None] * len(mu1_values)
    rnm_errors = [None] * len(mu1_values)
    pod_rbf_times = [None] * len(mu1_values)
    pod_rbf_errors = [None] * len(mu1_values)

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

    # Run PROM if results are missing or need to be updated
    prom_file = f"rom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(prom_file) or prom_times[i] is None:
        print(f"PROM results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running PROM...")
        prom_time, prom_error = run_prom.main(mu1, mu2)
        prom_times[i] = prom_time
        prom_errors[i] = prom_error
    else:
        print(f"PROM results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping PROM run.")

    '''
    # Run RNM if results are missing or need to be updated
    rnm_file = f"rnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(rnm_file) or rnm_times[i] is None:
        print(f"RNM results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running RNM...")
        rnm_time, rnm_error = run_RNM.main(mu1, mu2)
        rnm_times[i] = rnm_time
        rnm_errors[i] = rnm_error
    else:
        print(f"RNM results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping RNM run.")

    # Run POD-RBF if results are missing or need to be updated
    pod_rbf_file = f"pod_rbf_prom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    if not check_if_exists(pod_rbf_file) or pod_rbf_times[i] is None:
        print(f"POD-RBF results not found or missing for mu1 = {mu1}, mu2 = {mu2}. Running POD-RBF...")
        pod_rbf_time, pod_rbf_error = run_POD_RBF.main(mu1, mu2)
        pod_rbf_times[i] = pod_rbf_time
        pod_rbf_errors[i] = pod_rbf_error
    else:
        print(f"POD-RBF results already exist for mu1 = {mu1}, mu2 = {mu2}. Skipping POD-RBF run.")
    '''

# Save the updated results in the .npz file, overwriting old results
np.savez('rom_results.npz',
         fom_times=fom_times, fom_errors=fom_errors,
         prom_times=prom_times, prom_errors=prom_errors,
         rnm_times=rnm_times, rnm_errors=rnm_errors,
         pod_rbf_times=pod_rbf_times, pod_rbf_errors=pod_rbf_errors)

print("Results updated and saved to 'rom_results.npz'")




