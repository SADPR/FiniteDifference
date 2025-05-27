#!/usr/bin/env python3
import os
import numpy as np
import pyvista as pv

def create_3d_animation_pyvista(u_x_snaps, inds_to_plot, label,
                                grid_x, grid_y, output_file,
                                dt=0.1, fps=60):
    """
    Create and save a 3D animation of the u_x field over time using PyVista.
    Draws a bounding box behind the mesh but has no axis titles or tick labels.
    Z-axis range is locked [0, zmax].
    """
    nx = len(grid_x)
    ny = len(grid_y)
    X, Y = np.meshgrid(grid_x, grid_y)

    zmin = 0.0
    zmax = float(u_x_snaps.max())

    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(output_file)

    for frame_idx in inds_to_plot:
        Z = u_x_snaps[:, frame_idx].reshape((ny, nx), order='C')

        points = np.column_stack((X.ravel(order='C'),
                                  Y.ravel(order='C'),
                                  Z.ravel(order='C')))
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = [nx, ny, 1]

        plotter.clear()

        # Add the surface
        plotter.add_mesh(
            grid,
            scalars=Z.ravel(order='C'),
            cmap='viridis',
            show_scalar_bar=True,
            scalar_bar_args={'title': 'u_x', 'font_family': 'arial'},
            clim=[zmin, zmax]
        )

        # Time label (text in corner)
        plotter.add_text(f"{label}, Time: {frame_idx*dt:.2f}s", font_size=14)

        # Write out the current frame
        plotter.write_frame()

    plotter.close()
    print(f"PyVista 3D animation saved as '{output_file}'")


def plot_characteristic_snapshot_3d_pyvista(u_x_snaps, grid_x, grid_y,
                                            dt, output_file):
    """
    Plot 4 snapshots of the 3D solution at different times using PyVista,
    2x2 layout, bounding box behind the mesh, no axis/tick labels.
    """
    nx = len(grid_x)
    ny = len(grid_y)
    X, Y = np.meshgrid(grid_x, grid_y)

    # Pick 4 times: first, 1/3, 2/3, last
    times = [
        0,
        u_x_snaps.shape[1] // 3,
        2 * u_x_snaps.shape[1] // 3,
        u_x_snaps.shape[1] - 1
    ]

    zmin = 0.0
    zmax = float(u_x_snaps.max())

    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1200))

    for i, t_idx in enumerate(times):
        row, col = divmod(i, 2)
        plotter.subplot(row, col)

        Z = u_x_snaps[:, t_idx].reshape((ny, nx), order='C')
        points = np.column_stack((X.ravel(order='C'),
                                  Y.ravel(order='C'),
                                  Z.ravel(order='C')))
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = [nx, ny, 1]

        plotter.add_mesh(
            grid,
            scalars=Z.ravel(order='C'),
            cmap='viridis',
            show_scalar_bar=True,
            scalar_bar_args={'title': 'u_x', 'font_family': 'arial'},
            clim=[zmin, zmax]
        )

        # A small text label for the time in each subplot
        plotter.add_text(f"Time: {t_idx*dt:.2f}s", font_size=14)

    # Save the final 2x2 figure
    plotter.screenshot(output_file)
    plotter.close()
    print(f"PyVista 3D characteristic snapshot plot saved as '{output_file}'")


if __name__ == "__main__":
    # Example usage
    output_folder = 'hrom_animations'
    os.makedirs(output_folder, exist_ok=True)

    mu1, mu2 = 5.19, 0.026
    fom_snap_file = f'hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy'
    dt = 0.1

    # Load your snapshot data
    fom_snaps = np.load(fom_snap_file)

    # Grid
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x = np.linspace(xl, xu, num_cells_x)
    grid_y = np.linspace(yl, yu, num_cells_y)

    # Extract u_x
    u_x_snaps = fom_snaps[: num_cells_x * num_cells_y, :]

    # Animate frames [0..499]. If you want to skip frames to shorten
    # the overall length, e.g. range(0, 500, 2).
    inds_to_plot = range(0, 500, 10)

    # 1) Create a faster GIF (60 fps, though high fps may not be strictly honored by all GIF viewers)
    create_3d_animation_pyvista(
        u_x_snaps,
        inds_to_plot,
        f'HDM u_x field at mu1={mu1}, mu2={mu2}',
        grid_x, grid_y,
        os.path.join(output_folder, f"animation_3D_mu1_{mu1}_mu2_{mu2}.gif"),
        dt=dt,
        fps=60
    )

    # 2) Create the 2x2 snapshots PNG
    plot_characteristic_snapshot_3d_pyvista(
        u_x_snaps,
        grid_x,
        grid_y,
        dt,
        os.path.join(output_folder, f"characteristic_3D_mu1_{mu1}_mu2_{mu2}.png")
    )


