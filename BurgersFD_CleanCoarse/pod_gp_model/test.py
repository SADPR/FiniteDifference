
import pickle

# Load your trained GP model
with open('gp_model.pkl', 'rb') as file:
    gp_model = pickle.load(file)

# Print optimized hyperparameters
print(gp_model.kernel_)

