import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def main():
    OutFile = 'Model/Vis_'
    Model_inf = 'Model/Model_inf.npz'
    Model_result = 'Model/Model_result_TimeStep=0_IterationNumber=1.npz'
    Plot_All(Model_inf, Model_result, OutFile)
    plt.show()


def Plot_All(inf, result, OutFile):
    Strain_II_m_pre = np.load("Model/Strain_II_m.npy")
    Model_inf = np.load(inf)
    input = Model_inf['input']
    xlen, ylen, nx, ny, nx_m, ny_m = \
        Model_inf['xlen'], Model_inf['ylen'], Model_inf['nx'], Model_inf['ny'], Model_inf['nx_m'], Model_inf['ny_m']
    dx, dy, dx_m, dy_m = \
        Model_inf['dx'], Model_inf['dy'], Model_inf['dx_m'], Model_inf['dy_m']
    xx, yy, xm, ym, x_n, y_n = \
        Model_inf['xx'], Model_inf['yy'], Model_inf['xm'], Model_inf['ym'], Model_inf['x_n'], Model_inf['y_n']

    Model_result = np.load(result)
    vx1, vy1, Deviation, IterationNumber = \
        Model_result['vx1'], Model_result['vy1'], Model_result['Deviation'], Model_result['IterationNumber']
    rock_m, density_m, viscosity_m, mkk, mTT = \
        Model_result['rock_m'], Model_result['density_m'], Model_result['viscosity_m'], Model_result['mkk'], Model_result['mTT']
    rock, density, viscosity, kk, TT = \
        Model_result['rock'], Model_result['density'], Model_result['viscosity'], Model_result['kk'], Model_result['TT']

    VunitType = 1
    if VunitType == 0:
        Scale = 1
        Vlable = ' ${m}/{s}$'
    if VunitType == 1:
        Scale = 1 * 365.25 * 24 * 3600 * 100
        Vlable = ' ${cm}/{y}$'

    Vx = vx1 * Scale
    Vy = vy1 * Scale
    V = (Vx ** 2 + Vy ** 2) ** 0.5

    x1 = np.arange(0.5 * dx, xlen, dx)
    y1 = np.arange(0.5 * dy, ylen, dy)
    xx, yy = np.meshgrid(x1, y1)
    del x1, y1

    m = 1
    vx10, vy10 = vx1, vy1
    del vx1, vy1
    vx1 = vx10[0:ny:m, 0:nx:m]
    vy1 = vy10[0:ny:m, 0:nx:m]
    del vx10, vy10

    Stress_xx_d = np.zeros((ny - 1, nx - 1))
    Stress_yy_d = np.zeros((ny - 1, nx - 1))
    Stress_xy = np.zeros((ny - 1, nx - 1))
    Stress_yx = np.zeros((ny - 1, nx - 1))
    Strain_xx_d = np.zeros((ny - 1, nx - 1))
    Strain_yy_d = np.zeros((ny - 1, nx - 1))
    Strain_xy = np.zeros((ny - 1, nx - 1))
    Strain_yx = np.zeros((ny - 1, nx - 1))
    Strain_II = np.zeros((ny - 1, nx - 1))
    for i in range(nx - 2):
        for j in range(ny - 2):
            # stress / 6 = deviatoric stress - P * dij ; Thus, stress_xx_d[j, i] = stress_xx[j, i] + Pressure[j, i]
            # strain / E = deviatoric strain ; Thus, stress_xy[j, i] = stress_xy_d[j, i]
            Strain_xx_d[j, i] = (vx1[j, i + 1] - vx1[j, i]) / dx
            Stress_xx_d[j, i] = 2 * 0.25 * (
                    viscosity[j, i] + viscosity[j, i + 1] + viscosity[j + 1, i] + viscosity[j + 1, i + 1]) * \
                                Strain_xx_d[
                                    j, i]

            Strain_yy_d[j, i] = (vy1[j + 1, i] - vy1[j, i]) / dy
            Stress_yy_d[j, i] = 2 * 0.25 * (
                    viscosity[j, i] + viscosity[j, i + 1] + viscosity[j + 1, i] + viscosity[j + 1, i + 1]) * \
                                Strain_yy_d[
                                    j, i]

            Strain_xy[j, i] = ((vy1[j, i] - vy1[j, i - 1]) / dx + (vx1[j, i] - vx1[j - 1, i]) / dy) / 2
            Stress_xy[j, i] = 2 * viscosity[j, i] * Strain_xy[j, i]

            # 6 xy = 6 yx;    E xy = E yx
            Strain_yx[j, i] = Strain_xy[j, i]
            Stress_yx[j, i] = Stress_xy[j, i]

            Strain_II[j, i] = (0.5 * (
                    Strain_xx_d[j, i] ** 2 + Strain_yy_d[j, i] ** 2 + Strain_xy[j, i] ** 2 + Strain_yx[
                j, i] ** 2)) ** 0.5

    fig_out = plt.figure(figsize=(16, 12))
    # meshes
    ax = fig_out.add_subplot(231); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, 'No', ax, 'Grid: ' + str(nx-1) + ' × ' + str(ny-1), None, 1)

    ax = fig_out.add_subplot(232); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, density_m, ax, 'Density', 'ρ (kg/$\mathregular{m^3}$)', 2)

    ax = fig_out.add_subplot(233); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, np.log10(viscosity_m), ax, 'Viscosity', 'log$_{10}$η ($Pa·s$)', 2)

    ax = fig_out.add_subplot(235); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, rock_m, ax, 'Type of rock', 'Number', 2)

    ax = fig_out.add_subplot(236); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, mTT, ax, 'Temperture', 'Temperture ($deg$)', 2)

    # ax = fig_out.add_subplot(236); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    # Plot_fig(xm, ym, mkk, ax, 'Thermal conductivity', 'k ($W/(m·K)$)', 2)
    # plt.savefig(OutFile + 'marker_IterationNumber=' + str(IterationNumber))


    ################ vx1,vy1,viscosity,Density / nodes of mesh

    fig_out = plt.figure(figsize=(12, 12))
    plt.suptitle('Stress', fontsize=16, fontweight='bold')
    ax = fig_out.add_subplot(221);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Stress_xx_d, ax, 'Stress_xx_d', 'Stress_xx_d / Pa', 2)
    ax = fig_out.add_subplot(222);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Stress_yy_d, ax, 'Stress_yy_d', 'Stress_yy_d / Pa', 2)
    ax = fig_out.add_subplot(223);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Stress_xy, ax, 'Stress_xy', 'Stress_xy / Pa', 2)
    ax = fig_out.add_subplot(224);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Stress_yx, ax, 'Stress_yx', 'Stress_yx / Pa', 2)
    plt.savefig(OutFile + 'Stress_IterationNumber=' + str(IterationNumber))

    ################ Strain_II
    fig_out = plt.figure(figsize=(12, 12))
    plt.suptitle('Strain_II', fontsize=16, fontweight='bold')
    ax = fig_out.add_subplot(221);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, Strain_II_m_pre, ax, 'Strain_II before', 'Strain / $\mathregular{s^-1}$', 2)

    points = np.column_stack((xx.flatten(), yy.flatten()))
    Strain_II_m_new = griddata(points, Strain_II.flatten(), (xm, ym), method='nearest')
    for im in range(nx_m * (nx - 1)):
        for jm in range(ny_m * (ny - 1)):
            if Strain_II_m_new[jm, im] == 0:  # sticky air
                Strain_II_m_new[jm, im] = 1e-18
    # Strain_II_m_new[np.isnan(Strain_II_m_new)] = 0
    # Strain_II_m_new[np.where(Strain_II_m_new == 0)] = 1e-18
    ax = fig_out.add_subplot(222);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, Strain_II_m_new, ax, 'Strain_II after', 'Strain / $\mathregular{s^-1}$', 2)

    ax = fig_out.add_subplot(223);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    # plt.plot(Strain_II_m_new.flatten(), 'b.')
    # plt.plot(Strain_II_m_pre-Strain_II_m_new, 'y*')
    Plot_fig(xm, ym, Strain_II_m_new - Strain_II_m_pre, ax, 'Strain_II after',
             'Strain / $\mathregular{s^-1}$', 2)
    # np.save("Strain_II_m.npy", Strain_II_m_new)

    Strain_II_D = (Strain_II_m_new - Strain_II_m_pre) / Strain_II_m_pre
    Strain_II_D2 = (Strain_II_D ** 2) ** 0.5

    ax = fig_out.add_subplot(224)
    plt.plot(Strain_II_D.flatten(), 'r.')
    # plt.savefig('/home/ictja/Videos/' + str(IterationNumber))
    plt.savefig(OutFile + 'Strain_II_IterationNumber=' + str(IterationNumber))

    #  Strain_xx, xy, yx, yy
    fig_out = plt.figure(figsize=(12, 12))
    plt.suptitle('Strain', fontsize=16, fontweight='bold')
    ax = fig_out.add_subplot(221);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Strain_xx_d, ax, 'Strain_xx_d', 'Strain / $\mathregular{s^-1}$', 2)
    ax = fig_out.add_subplot(222);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Strain_yy_d, ax, 'Strain_yy_d', 'Strain / $\mathregular{s^-1}$', 2)
    ax = fig_out.add_subplot(223);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Strain_xy, ax, 'Strain_xy', 'Strain / $\mathregular{s^-1}$', 2)
    ax = fig_out.add_subplot(224);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Strain_yx, ax, 'Strain_yx', 'Strain / $\mathregular{s^-1}$', 2)

    plt.savefig(OutFile + 'Strain_IterationNumber=' + str(IterationNumber))

    # viscosity /Plotting viscosity: vx1, vy1
    Qkey = np.max(V)  # Qkey = (np.max(Vx) ** 2 + np.max(Vy) ** 2) ** 0.5
    fig_out = plt.figure(figsize=(12, 12))
    plt.suptitle('Viscosity', fontsize=16, fontweight='bold')
    # ax = fig_out.add_subplot(221); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    # Plot_fig(xx, yy, V, ax, 'V', None, 3)
    # Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
    # plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

    ax = fig_out.add_subplot(222);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, V, ax, 'V', 'V' + Vlable, 2)
    # Plot_fig(xm, ym, np.log10(viscosity_m), ax, 'Viscosity', 'log$_{10}$η ($Pa·s$)', 2)
    Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
    plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

    ax = fig_out.add_subplot(223);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Vx, ax, 'Vx', 'Vx' + Vlable, 2)
    Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
    plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

    ax = fig_out.add_subplot(224);
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, Vy, ax, 'Vy', 'Vy' + Vlable, 2)
    Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
    plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')
    plt.savefig(OutFile + 'velocity_IterationNumber=' + str(IterationNumber))


def Plot_fig(x, y, z, ax, title, label, type):
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
    if label:  # if label=None; skip
        Cbar = plt.colorbar()
        Cbar.set_label(label, fontsize=10)  # ,fontweight='bold'


def Plot_Slip(y, f, ax, title, xlabel, legend):
    ax.plot(f, y / 1000, '.', label=legend)  # Spectral_r
    ax.legend(ncol=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth ($km$)')
    ax.grid(True, linestyle='dotted', linewidth=0.5)
    ax.invert_yaxis()


if __name__ == '__main__':
    main()
