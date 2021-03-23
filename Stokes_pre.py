import numpy as np
import matplotlib.pyplot as plt

def main(input, xlen, ylen, nx, ny ,nx_m, ny_m):
    dx = xlen / (nx - 1)
    dy = ylen / (ny - 1)
    dx_m = dx / nx_m
    dy_m = dy / ny_m

    # define the mesh, mesh unit, m
    # x=[0 dx 2dx ... (nx-1)dx], y=[...], nodes
    x1 = np.arange(0, xlen + dx, dx)
    y1 = np.arange(0, ylen + dy, dy)
    x, y = np.meshgrid(x1, y1)
    del x1, y1

    # define the markers
    # xm=[0.5dx_m 1.5dx_m ... (all-0.5)dx_m], y=[...], markers
    x1 = np.arange(0.5 * dx_m, xlen, dx_m)
    y1 = np.arange(0.5 * dy_m, ylen, dy_m)
    xm, ym = np.meshgrid(x1, y1)
    del x1, y1

    # input parameters for makers
    cp = 1.0

    if input == 1:
        (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_box(nx, nx_m, ny, ny_m, xm, ym)
    # if input == 2:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_Fun(nx, nx_m, ny, ny_m, xm, ym, dx_m, dy_m)
    # if input == 3:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_LitMod(nx, nx_m, ny, ny_m, xm, ym)
    # if input == 4:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_circle(nx, nx_m, ny, ny_m, xm, ym)

    fig_out = plt.figure(figsize=(16, 12))
    # meshes
    ax = fig_out.add_subplot(231); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(x, y, 'No', ax, 'Grid: ' + str(nx-1) + ' × ' + str(ny-1), None, 1)

    ax = fig_out.add_subplot(232); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, density_m, ax, 'Density', 'ρ (kg/$\mathregular{m^3}$)', 2)

    ax = fig_out.add_subplot(233); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, np.log10(viscosity_m), ax, 'Viscosity', 'log$_{10}$η ($Pa·s$)', 2)

    ax = fig_out.add_subplot(234); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, rock_m, ax, 'Type of rock', None, 2)

    ax = fig_out.add_subplot(235); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, mTT, ax, 'Temperture', 'Temperture ($deg$)', 2)

    ax = fig_out.add_subplot(236); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, mkk, ax, 'Thermal conductivity', 'k ($W/(m·K)$)', 2)

    return rock_m, density_m, viscosity_m, mkk, mTT, nx_m, ny_m, xm, ym, nx, ny


def input_parameters_box(nx, nx_m, ny, ny_m, xm, ym):  # input=1
    print('  The shape of model: Box' )
    rock_m      = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    density_m   = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    viscosity_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    mkk         = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    mTT         = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))

    for im in range(nx_m * (nx - 1)):
        for jm in range(ny_m * (ny - 1)):
            if 200000 < xm[jm, im] < 300000 and 100000 < ym[jm, im] < 200000:
                rock_m[jm, im] = 1
                density_m[jm, im] = 3000
                viscosity_m[jm, im] = 1e22
                mkk[jm, im] = 1e3
                mTT[jm, im] = (xm[jm, im] + ym[jm, im])/1000
            else:
                rock_m[jm, im] = 2
                density_m[jm, im] = 2800
                viscosity_m[jm, im] = 1e20
                mkk[jm, im] = 1e4
                mTT[jm, im] = (xm[jm, im] + ym[jm, im])/1000

    return rock_m, density_m, viscosity_m, mkk, mTT


def Plot_fig(x, y, z, ax, title, label, type):
    if type == 1:
        ax.plot(x / 1000, y / 1000, 'g+', label="mesh")
    if type == 2:
        plt.pcolor(x / 1000, y / 1000, z[:-1, :-1], cmap='Spectral_r')  # Spectral_r
    if type == 3:
        plt.pcolor(x / 1000, y / 1000, z, cmap='Spectral_r')  # Spectral_r
    ax.set_title(title)
    ax.set_xlabel('Distance ($km$)')
    ax.set_ylabel('Depth ($km$)')
    ax.grid(True, linestyle='dotted', linewidth=0.5)
    ax.invert_yaxis()
    if label: # if label=None; skip
        Cbar = plt.colorbar()
        Cbar.set_label(label, fontsize=10)  # ,fontweight='bold'