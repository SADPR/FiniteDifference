import os
import numpy as np
import torch
import time
from hypernet2D import make_2D_grid, load_or_compute_snaps, inviscid_burgers_implicit2D_ae_LSPG
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

# === Autoencoder definition (must match training) ===
class ScaledAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, mu_in, sig_in):
        super().__init__()
        self.mu_in = torch.tensor(mu_in).float().view(1, -1)
        self.sig_in = torch.tensor(sig_in).float().view(1, -1)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim)
        )

    def scale_in(self, x):
        return (x - self.mu_in.to(x.device)) / self.sig_in.to(x.device)

    def scale_out(self, y):
        return y * self.sig_in.to(y.device) + self.mu_in.to(y.device)

    def forward(self, x):
        z = self.encoder(self.scale_in(x))
        return self.scale_out(self.decoder(z))

# === Run AE-ROM ===
def run_rom_autoencoder(mu1=4.75, mu2=0.02):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR = "models"
    LATENT_DIM = 10

    # Grid and initial condition
    snap_folder = 'param_snaps'  # Folder where snapshots are stored
    num_modes = 10  # Number of modes to keep after truncating the basis

    # Time-stepping and grid setup for the 2D problem
    dt = 0.05  # Time step size
    num_steps = 500  # Number of time steps
    num_cells_x, num_cells_y = 50, 50  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)  # Create the 2D grid
    
    # Initial conditions for u and v (2D velocity components or state variables)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))  # Concatenate initial conditions

    # Parameter values for ROM evaluation
    mu_rom = [mu1, mu2]

    # Load model + scaler
    scaler = np.load(os.path.join(MODEL_DIR, "scaler.npz"))
    model = ScaledAutoencoder(w0.size, LATENT_DIM, scaler["mu"], scaler["sigma"]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder_best.pth"), map_location=DEVICE))
    model.eval()

    # Run ROM
    t0 = time.time()
    rom_snaps, _ = inviscid_burgers_implicit2D_ae_LSPG(grid_x, grid_y, w0, DT, NUM_STEPS, mu_rom, model)
    elapsed = time.time() - t0

    # Load HDM
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder="param_snaps")

    # Save
    np.save(f"rom_snaps_AE_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy", rom_snaps)

    # Relative error
    error = 100 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)
    print(f"[Autoencoder-ROM] Relative error: {error:.2f}%, Time: {elapsed:.2f}s")

if __name__ == "__main__":
    run_rom_autoencoder()
