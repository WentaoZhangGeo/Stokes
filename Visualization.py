import numpy as np
import matplotlib.pyplot as plt

def main(nx, ny, vx1, vy1, viscosity, dx ,dy):
    VunitType = 1
    if VunitType == 0:
        scale = 1
        Vlable = ' ${m}/{s}$'
    if VunitType == 1:
        scale = 1 * 365.25 * 24 * 3600 * 100
        Vlable = ' ${cm}/{y}$'
    Vx = vx1 * scale
    Vy = vy1 * scale
    V = (Vx ** 2 + Vy ** 2) ** 0.5

    # stress
    Stress_xx_d = np.zeros((ny-1, nx-1))
    Stress_yy_d = np.zeros((ny-1, nx-1))
    Stress_xy = np.zeros((ny-1, nx-1))
    Stress_yx = np.zeros((ny-1, nx-1))
    # strain rate
    Strain_xx_d = np.zeros((ny-1, nx-1))
    Strain_yy_d = np.zeros((ny-1, nx-1))
    Strain_xy = np.zeros((ny-1, nx-1))
    Strain_yx = np.zeros((ny-1, nx-1))
    Strain_II = np.zeros((ny-1, nx-1))
    for i in range(nx-2):
        for j in range(ny-2):
            # stress / 6 = deviatoric stress - P * dij ; Thus, stress_xx_d[j, i] = stress_xx[j, i] + Pressure[j, i]
            # strain / E = deviatoric strain ; Thus, stress_xy[j, i] = stress_xy_d[j, i]
            Strain_xx_d[j, i] = (vx1[j, i + 1] - vx1[j, i]) / dx
            Stress_xx_d[j, i] = 2 * 0.25 * (
                        viscosity[j, i] + viscosity[j, i + 1] + viscosity[j + 1, i] + viscosity[j + 1, i + 1]) * Strain_xx_d[
                                    j, i]

            Strain_yy_d[j, i] = (vy1[j + 1, i] - vy1[j, i]) / dy
            Stress_yy_d[j, i] = 2 * 0.25 * (
                        viscosity[j, i] + viscosity[j, i + 1] + viscosity[j + 1, i] + viscosity[j + 1, i + 1]) * Strain_yy_d[
                                    j, i]

            Strain_xy[j, i] = ((vy1[j, i] - vy1[j, i - 1]) / dx + (vx1[j, i] - vx1[j - 1, i]) / dy) / 2
            Stress_xy[j, i] = 2 * viscosity[j, i] * Strain_xy[j, i]

            # 6 xy = 6 yx;    E xy = E yx
            Strain_yx[j, i] = Strain_xy[j, i]
            Stress_yx[j, i] = Stress_xy[j, i]

            Strain_II[j, i] = (0.5 * (Strain_xx_d[j, i] **2 + Strain_yy_d[j, i] **2 + Strain_xy[j, i] **2 + Strain_yx[j, i] **2)) ** 0.5


def plot_fig(x, y, z, ax, title, label, type):
    if type == 1:
        ax.plot(x / 1000, y / 1000, 'g+', label="mesh")
    if type == 2:
        plt.pcolor(x / 1000, y / 1000, z[:-1, :-1], cmap='Spectral_r')  # Spectral_r
    # if type == 3:
    #     ax.plot(x / 1000, y / 1000, 'g+', label="mesh")
    ax.set_title(title)
    ax.set_xlabel('Distance ($km$)')
    ax.set_ylabel('Depth ($km$)')
    ax.grid(True, linestyle='dotted', linewidth=0.5)
    ax.invert_yaxis()
    if label:   # if label=None; skip
        cbar = plt.colorbar()
        cbar.set_label(label, fontsize=10)  # ,fontweight='bold'


def plot_slip(y, f, ax, title, xlabel, legend):
    ax.plot(f, y / 1000, '.', label=legend)  # Spectral_r
    ax.legend(ncol=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth ($km$)')
    ax.grid(True, linestyle='dotted', linewidth=0.5)
    ax.invert_yaxis()
