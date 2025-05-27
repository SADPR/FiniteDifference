#!/bin/bash

# Remove the existing pod_rbf_global_model directory
rm -r pod_rbf_global_model

# Copy the pod_rbf_global_model directory from POD-RBF_global
cp -r POD-RBF_global/pod_rbf_global_model .

# Copy the ECSW weights to the new pod_rbf_global_model directory
cp pod_rbf_global_model_imq/ecsw_weights_rbf_global.npy pod_rbf_global_model/

# Print a success message
echo "Model updated successfully!"

