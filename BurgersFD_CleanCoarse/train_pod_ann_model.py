import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
from hypernet2D import load_or_compute_snaps, POD, make_2D_grid

EPOCHS = 50000
ROM_SIZE = 40
LR_INIT = 1e-3
LR_PATIENCE = 100
COMPLETION_PATIENCE = 500
MODEL_PATH = 'autoenc.pt'
CARLBERG = False
BATCH_SIZE = 16
TRAIN_FRAC = 0.9
SNAP_FOLDER = "param_snaps"

SEED = 1234557

## PROBLEM-WIDE CONSTANTS
#   These define the underlying HDM, so set them once and use the same values for all ROM
#   and neural network runs
DT = 0.07
NUM_STEPS = 500
NUM_CELLS = 250
XL, XU = 0, 100
U0 = np.ones((NUM_CELLS, NUM_CELLS))
V0 = U0.copy()
W0 = np.concatenate((U0.ravel(), V0.ravel()))
GRID_X, GRID_Y = make_2D_grid(XL, XU, XL, XU, NUM_CELLS, NUM_CELLS)
MU1_RANGE = 4.25, 5.5
MU2_RANGE = 0.015, 0.03
SAMPLES_PER_MU = 3

def get_snapshot_params():
    MU1_LOW, MU1_HIGH = MU1_RANGE
    MU2_LOW, MU2_HIGH = MU2_RANGE
    mu1_samples = np.linspace(MU1_LOW, MU1_HIGH, SAMPLES_PER_MU)
    mu2_samples = np.linspace(MU2_LOW, MU2_HIGH, SAMPLES_PER_MU)
    mu_samples = []
    for mu1 in mu1_samples:
        for mu2 in mu2_samples:
            mu_samples += [[mu1, mu2]]
    return mu_samples

# Define the ANN model
class POD_ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(POD_ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        x = self.fc6(x)
        return x

def main(primary_num_modes=10, secondary_num_modes=140, compute_basis=False):
    # Define file paths and parameters
    snap_folder = 'param_snaps'
    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    w0 = np.ones((num_cells_y * num_cells_x, 2)).flatten()

    mu_samples = get_snapshot_params()

    # Collect snapshots over a range of parameters
    snap_count = len(mu_samples)
    snapshot_shape = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder).shape
    total_snaps = snapshot_shape[1] * snap_count  # Total number of snapshots

    # Pre-allocate memory for all snapshots
    snaps = np.zeros((snapshot_shape[0], total_snaps))

    # Collect snapshots
    col_offset = 0
    for mu in mu_samples:
        snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu
        col_offset += snap_mu.shape[1]

    # Load or compute the HDM snapshots and build the Reduced Order Basis (ROB)
    if not compute_basis:
        # Load a precomputed basis
        basis = np.load('basis.npy', allow_pickle=True)
    else:
        # Perform POD to compute the ROB
        pod_method = 'rsvd'
        if pod_method == 'rsvd':
            basis, sigma = POD(snaps, num_modes=primary_num_modes + secondary_num_modes, method='rsvd', random_state=42)
        else:
            basis, sigma = POD(snaps, num_modes=primary_num_modes + secondary_num_modes, method='svd')

        np.save('basis', basis)
        np.save('sigma', sigma)

    # Prepare training data
    qs = basis.T @ snaps  # Shape: (primary_num_modes + secondary_num_modes, total_snaps)
    q_p = qs[:primary_num_modes, :]  # Principal modes
    q_s = qs[primary_num_modes:primary_num_modes + secondary_num_modes, :]  # Secondary modes

    # Prepare parameter labels for input to the neural network
    mu_labels0, mu_labels1 = [], []
    for mu in mu_samples:
        mu_labels0.extend([mu[0]] * (num_steps + 1))
        mu_labels1.extend([mu[1]] * (num_steps + 1))

    mu_labels0 = np.array(mu_labels0)
    mu_labels1 = np.array(mu_labels1)

    # Concatenate primary modes and parameter labels
    q_p_with_mu = np.concatenate((q_p.T, mu_labels0[:, None], mu_labels1[:, None]), axis=1)
    # Now, q_p_with_mu has shape: (total_snaps, primary_num_modes + 2)

    # Convert to PyTorch tensors
    q_p_tensor = torch.tensor(q_p_with_mu, dtype=torch.float32)
    q_s_tensor = torch.tensor(q_s.T, dtype=torch.float32)

    # Adjust input dimension
    input_dim = primary_num_modes + 2  # Adding two mu labels
    output_dim = secondary_num_modes

    # Split data into training and testing sets
    q_p_train, q_p_test, q_s_train, q_s_test = train_test_split(q_p_tensor, q_s_tensor, test_size=0.2, random_state=42)

    # Create data loaders
    train_dataset = TensorDataset(q_p_train, q_s_train)
    test_dataset = TensorDataset(q_p_test, q_s_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = POD_ANN(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    # Train the ANN model
    train_ANN(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'pod_ann_model.pth')

# Train the ANN model
def train_ANN(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100, clip_value=1.0):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Clip gradients
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)

        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

if __name__ == "__main__":
    main(compute_basis=False)

