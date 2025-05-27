import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def make_2D_grid(x_low, x_up, y_low, y_up, num_cells_x, num_cells_y):
    """Creates a 2D grid with specified boundaries and number of cells."""
    grid_x_edges = np.linspace(x_low, x_up, num_cells_x + 1)
    grid_y_edges = np.linspace(y_low, y_up, num_cells_y + 1)
    cell_centers_x = (grid_x_edges[:-1] + grid_x_edges[1:]) / 2
    cell_centers_y = (grid_y_edges[:-1] + grid_y_edges[1:]) / 2
    return cell_centers_x, cell_centers_y

def create_animation(u_x_snaps, inds_to_plot, label, grid_x, grid_y, output_file, dt, fps=30):
    """Create and save a 2D animation of the u_x field over time."""
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    vmin, vmax = u_x_snaps.min(), u_x_snaps.max()

    im = ax.imshow(u_x_snaps[:, inds_to_plot[0]].reshape((len(grid_y), len(grid_x)), order='C'), 
                   extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    time_text = ax.set_title(f'{label}, Time: {inds_to_plot[0] * dt:.2f}s')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('u_x')

    def update(frame):
        data = u_x_snaps[:, frame].reshape((len(grid_y), len(grid_x)), order='C')
        im.set_data(data)
        time_text.set_text(f'{label}, Time: {frame * dt:.2f}s')
        return [im]

    ani = FuncAnimation(fig, update, frames=inds_to_plot, interval=1000 / fps, blit=False)
    ani.save(output_file, writer="pillow", dpi=150, fps=fps)
    plt.close(fig)
    print(f"2D animation saved as '{output_file}'")

def plot_characteristic_snapshot(u_x_snaps, grid_x, grid_y, dt, output_file):
    """Plot 4 snapshots of the 2D solution at different times."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    times = [0, len(u_x_snaps[0])//3, 2*len(u_x_snaps[0])//3, len(u_x_snaps[0])-1]
    
    for i, ax in enumerate(axs.flat):
        data = u_x_snaps[:, times[i]].reshape((len(grid_y), len(grid_x)), order='C')
        im = ax.imshow(data, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], 
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title(f'Time: {times[i] * dt:.2f}s')
        fig.colorbar(im, ax=ax)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"2D characteristic snapshot plot saved as '{output_file}'")

if __name__ == "__main__":
    output_folder = 'hrom_animations'
    os.makedirs(output_folder, exist_ok=True)

    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    dt = 0.1
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)

    mu1, mu2 = 5.19, 0.026
    fom_snap_file = f'hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy'
    fom_snaps = np.load(fom_snap_file)

    u_x_snaps = fom_snaps[:num_cells_x * num_cells_y, :]
    inds_to_plot = range(0, 500)

    create_animation(u_x_snaps, inds_to_plot, f'HDM u_x field at mu1={mu1}, mu2={mu2}',
                     grid_x, grid_y, os.path.join(output_folder, f"animation_2D_mu1_{mu1}_mu2_{mu2}.gif"), dt)

    plot_characteristic_snapshot(u_x_snaps, grid_x, grid_y, dt, 
                                 os.path.join(output_folder, f"characteristic_2D_mu1_{mu1}_mu2_{mu2}.png"))
