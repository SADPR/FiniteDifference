import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def make_2D_grid(x_low, x_up, y_low, y_up, num_cells_x, num_cells_y):
    """
    Creates a 2D grid with specified boundaries and number of cells.
    Returns grid points for cell centers.
    """
    # Grid points (edges)
    grid_x_edges = np.linspace(x_low, x_up, num_cells_x + 1)  # x-direction grid points
    grid_y_edges = np.linspace(y_low, y_up, num_cells_y + 1)  # y-direction grid points
    # Cell centers
    cell_centers_x = (grid_x_edges[:-1] + grid_x_edges[1:]) / 2
    cell_centers_y = (grid_y_edges[:-1] + grid_y_edges[1:]) / 2
    return cell_centers_x, cell_centers_y

def create_animation(u_x_snaps, inds_to_plot, label, grid_x, grid_y, output_file, dt, fps=30):
    """
    Create and save an animation of the 2D u_x field over time.
    """
    interval = 1000 / fps  # Milliseconds per frame
    fig, ax = plt.subplots(figsize=(8, 6))
    num_points_x = len(grid_x)
    num_points_y = len(grid_y)

    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]

    # Compute global vmin and vmax for consistent color scaling
    vmin = u_x_snaps.min()
    vmax = u_x_snaps.max()
    print('Data min:', vmin, 'Data max:', vmax)

    # Initial data
    data_flat = u_x_snaps[:, inds_to_plot[0]]
    data = data_flat.reshape((num_points_y, num_points_x), order='C')
    im = ax.imshow(data, extent=extent,
                   origin='lower', cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    time = inds_to_plot[0] * dt
    ax.set_title(f'{label}, Time: {time:.2f}s')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('u_x')

    def init():
        """Initialize the plot."""
        im.set_data(np.zeros((num_points_y, num_points_x)))
        return [im]

    def update(frame):
        """Update the plot for each frame."""
        data_flat = u_x_snaps[:, frame]
        data = data_flat.reshape((num_points_y, num_points_x), order='C')
        im.set_data(data)
        time = frame * dt
        ax.set_title(f'{label}, Time: {time:.2f}s')
        return [im]

    ani = FuncAnimation(fig, update, frames=inds_to_plot, init_func=init, interval=interval, blit=False)

    # Save the animation as GIF
    ani.save(output_file, writer="pillow", dpi=150, fps=fps)
    plt.close(fig)
    print(f"2D animation saved as '{output_file}'")

def create_3d_animation(u_x_snaps, inds_to_plot, label, grid_x, grid_y, output_file, dt, fps=10):
    """
    Create and save a 3D animation of the u_x field over time.
    """
    interval = 1000 / fps  # Milliseconds per frame
    num_points_x = len(grid_x)
    num_points_y = len(grid_y)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(grid_x, grid_y)

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute global z limits for consistent scaling
    zmin = u_x_snaps.min()
    zmax = u_x_snaps.max()
    print('Data min:', zmin, 'Data max:', zmax)

    # Initial data
    data_flat = u_x_snaps[:, inds_to_plot[0]]
    Z = data_flat.reshape((num_points_y, num_points_x), order='C')
    surf = [ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')]

    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u_x')
    time = inds_to_plot[0] * dt
    ax.set_title(f'{label}, Time: {time:.2f}s')

    def init():
        """Initialize the plot."""
        ax.clear()
        surf[0] = ax.plot_surface(X, Y, np.zeros((num_points_y, num_points_x)), cmap='viridis', edgecolor='none')
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u_x')
        return surf

    def update(frame):
        """Update the plot for each frame."""
        ax.clear()
        data_flat = u_x_snaps[:, frame]
        Z = data_flat.reshape((num_points_y, num_points_x), order='C')
        surf[0] = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_zlim(zmin, zmax)
        time = frame * dt
        ax.set_title(f'{label}, Time: {time:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u_x')
        return surf

    ani = FuncAnimation(fig, update, frames=inds_to_plot, init_func=init, interval=interval, blit=False)

    # Save the animation as GIF
    ani.save(output_file, writer="pillow", dpi=80, fps=fps)
    plt.close(fig)
    print(f"3D animation saved as '{output_file}'")

if __name__ == "__main__":
    # Define file paths and grid data
    output_folder = 'hrom_animations'

    num_cells_x, num_cells_y = 250, 250  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    dt = 0.1  # Time step size, adjust if different
    # Create the 2D grid at cell centers
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Specify the parameters
    mu1 = 5.19
    mu2 = 0.026

    # Snapshot file path
    fom_snap_file = f'hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy'

    # Load the snapshots
    fom_snaps = np.load(fom_snap_file)  # Expected shape: (125000, num_snapshots)
    print('fom_snaps.shape:', fom_snaps.shape)

    # Extract u_x component
    num_cells = num_cells_x * num_cells_y  # 250 * 250 = 62,500
    u_x_snaps = fom_snaps[:num_cells, :]  # Shape: (62,500, num_snapshots)
    print('u_x_snaps.shape:', u_x_snaps.shape)
    print('u_x_snaps min:', u_x_snaps.min(), 'max:', u_x_snaps.max())

    # Define indices for animation (snapshots from 0 to 500)
    inds_to_plot = range(0, 500)  # Include all 500 time steps

    # Optional: Verify data ordering
    data_flat = u_x_snaps[:, 0]  # First snapshot
    data_C = data_flat.reshape((num_cells_y, num_cells_x), order='C')

    # Plot using 'C' order
    plt.figure()
    plt.imshow(data_C, extent=[xl, xu, yl, yu], origin='lower', cmap='viridis', aspect='auto')
    plt.title('u_x field (order=C)')
    plt.colorbar(label='u_x')
    plt.show()

    # Create animation for HDM (FOM) data only
    create_animation(
        u_x_snaps=u_x_snaps,
        inds_to_plot=inds_to_plot,
        label=f'HDM u_x field at mu1={mu1}, mu2={mu2}',
        grid_x=grid_x,
        grid_y=grid_y,
        output_file=os.path.join(output_folder, f"animation_hdm_mu1_{mu1}_mu2_{mu2}.gif"),
        dt=dt,  # Pass the time step size
        fps=30  # Smooth playback with 30 frames per second
    )

    # Create 3D animation for HDM (FOM) data only
    create_3d_animation(
        u_x_snaps=u_x_snaps,
        inds_to_plot=inds_to_plot,
        label=f'HDM u_x field at mu1={mu1}, mu2={mu2}',
        grid_x=grid_x,
        grid_y=grid_y,
        output_file=os.path.join(output_folder, f"3d_animation_hdm_mu1_{mu1}_mu2_{mu2}.gif"),
        dt=dt,  # Pass the time step size
        fps=30  # Lower FPS due to rendering complexity
    )
