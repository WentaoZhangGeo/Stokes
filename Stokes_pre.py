import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def main(input, xlen, ylen, nx, ny ,nx_m, ny_m, LitMod_file):
    fig_switch = 0  # Check out the fig, 1 is off, other is on
    PA = 200000 # vertical_slice
    PB = 600000  # vertical_slice

    dx = xlen / (nx - 1)
    dy = ylen / (ny - 1)
    dx_m = dx / nx_m
    dy_m = dy / ny_m

    # define the centre of each mesh, mesh unit, m
    # x=[0.5dx 1.5dx 2.5dx ... (nx-0.5)dx], y=[...], nodes
    x1 = np.arange(0.5 * dx, xlen, dx)
    y1 = np.arange(0.5 * dy, ylen, dy)
    xx, yy = np.meshgrid(x1, y1)

    # define the markers
    # xm=[0.5dx_m 1.5dx_m ... (all-0.5)dx_m], y=[...], markers
    x2 = np.arange(0.5 * dx_m, xlen, dx_m)
    y2 = np.arange(0.5 * dy_m, ylen, dy_m)
    xm, ym = np.meshgrid(x2, y2)

    # define the nodes, mesh unit, m
    # x=[0 dx 2dx ... (nx-1)dx], y=[...], nodes
    x3 = np.arange(0, xlen + dx, dx)
    y3 = np.arange(0, ylen + dy, dy)
    x_n, y_n = np.meshgrid(x3, y3)
    del x1, y1, x2, y2, x3, y3

    # input parameters for makers
    cp = 1.0

    if input == 1:
        (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_box(nx, nx_m, ny, ny_m, xm, ym)
    # if input == 2:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_Fun(nx, nx_m, ny, ny_m, xm, ym, dx_m, dy_m)
    if input == 3:
        (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_LitMod(nx, nx_m, ny, ny_m, xm, ym, LitMod_file)
    # if input == 4:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_circle(nx, nx_m, ny, ny_m, xm, ym)

    fig_out = plt.figure(figsize=(16, 12))
    # meshes
    ax = fig_out.add_subplot(231); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xx, yy, 'No', ax, 'Grid: ' + str(nx-1) + ' × ' + str(ny-1), None, 1)

    ax = fig_out.add_subplot(232); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, density_m, ax, 'Density', 'ρ (kg/$\mathregular{m^3}$)', 2)

    ax = fig_out.add_subplot(233); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, np.log10(viscosity_m), ax, 'Viscosity', 'log$_{10}$η ($Pa·s$)', 2)

    ax = fig_out.add_subplot(234); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, rock_m, ax, 'Type of rock', 'Number', 2)

    ax = fig_out.add_subplot(235); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, mTT, ax, 'Temperture', 'Temperture ($deg$)', 2)

    ax = fig_out.add_subplot(236); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Plot_fig(xm, ym, mkk, ax, 'Thermal conductivity', 'k ($W/(m·K)$)', 2)

    if fig_switch == 0:
        plt.close()

    # vertical_slice
    im_find = abs(xm[1, :] - PA)
    im = np.argmin(im_find)
    legend = 'x=' + str(xm[1, im] / 1000) + ' km'

    fig_out = plt.figure(figsize=(8, 12))
    ax = fig_out.add_subplot(121)
    Plot_Slip(ym[:, im], np.log10(viscosity_m[:, im]), ax, 'viscosity profile', 'log$_{10}$Viscosity (Pa s)', legend)

    im_find = abs(xm[1, :] - PB)
    im = np.argmin(im_find)
    legend = 'x=' + str(xm[1, im] / 1000) + ' km'
    ax = fig_out.add_subplot(122)
    Plot_Slip(ym[:, im], np.log10(viscosity_m[:, im]), ax, 'viscosity profile', 'log$_{10}$Viscosity (Pa s)', legend)

    plt.close()


    return rock_m, density_m, viscosity_m, mkk, mTT, dx, dy, dx_m, dy_m, xx, yy, xm, ym, x_n, y_n


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

def input_parameters_LitMod(nx, nx_m, ny, ny_m, xm, ym, LitMod_file):
    print('  The shape of model: LitMod2D_2.0' )
    rock_m      = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    density_m   = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    viscosity_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    mkk         = np.ones((ny_m * (ny - 1), nx_m * (nx - 1)))
    mTT         = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))

    file = LitMod_file + '/dens_node2.dat'
    data = np.loadtxt(file)
    points = np.column_stack((data[:, 0], -1 * data[:, 1])) * 1000
    density_m = griddata(points, data[:, 2], (xm, ym), method='nearest')
    del data, points

    file = LitMod_file + '/tempout.dat'
    data = np.loadtxt(file)
    points = np.column_stack((data[:, 0], -1 * data[:, 1])) * 1000
    mTT = griddata(points, data[:, 2], (xm, ym), method='nearest')
    del data, points

    file = LitMod_file + '/post_processing_output.dat'
    data = np.loadtxt(file)
    points = np.column_stack((data[:, 0], -1 * data[:, 1])) * 1000
    rock_m = griddata(points, data[:, 7], (xm, ym), method='nearest')
    Pressure = griddata(points, data[:, 3], (xm, ym), method='nearest')

    Strain_II = np.load("Strain_II_m.npy")
    for im in range(nx_m * (nx - 1)):
        for jm in range(ny_m * (ny - 1)):
            if rock_m[jm, im] < 0:              # sticky air
                rock_m[jm, im] = 0
                viscosity_m[jm, im] = 1e18
            elif 0 <= rock_m[jm, im] <= 10:     # sediment & crust
                rock_m[jm, im] = 1
                viscosity_m[jm, im] = 1e20
            elif 10 < rock_m[jm, im] <= 90:     # Lit_mantle
                rock_m[jm, im] = 2
                viscosity_m[jm, im] = Rheology_HK03(Pressure[jm, im], mTT[jm, im], Strain_II[jm, im])
            elif 91 < rock_m[jm, im] <= 100:    # Sub_mantle
                rock_m[jm, im] = 3
                viscosity_m[jm, im] = Rheology_HK03(Pressure[jm, im], mTT[jm, im], Strain_II[jm, im])


    return rock_m, density_m, viscosity_m, mkk, mTT

def Rheology_HK03(P, T, Strain_II):
    sii_dis = 0.5 * Strain_II  # second invariant of the deviatoric strain rate
    sii_dif = 0.5 * Strain_II  # second invariant of the deviatoric strain rate
    d = 1e-2  # Grain size, m
    R = 8.314  # gas constant
    # # Dislocation creep / dry
    # A_dis = 3.5e22  # material constant, unit, 1/(Pa^n * s * m^m)
    # n_dis = 3.5  # stress exponent
    # m_dis = 0  # grain size exponent
    # Ea_dis = 540000  # activation energy, unit, J/mol
    # Va_dis = 20.0e-6  # activation volume, unit, m3/mol    15 - 25
    # # Difusion creep / dry
    # A_dif = 8.7e15
    # n_dif = 1.00
    # m_dif = -2.5
    # Ea_dif = 300000
    # Va_dif = 6.0

    # Dislocation creep / wet
    A_dis = 2.28e-18  # Pa m3 s-1
    n_dis = 3.5  # stress exponent
    m_dis = 0  # stress exponent
    Ea_dis = 480000  # J/mol
    Va_dis = 1.1e-5  # activation volume, unit, m3/mol    15 - 25
    # Difusion creep / wet
    A_dif = 4.7e-16
    n_dif = 1
    m_dif = 3
    Ea_dif = 335000
    Va_dif = 4.0e-6

    viscosity_dis = 0.5 * (A_dis ** (-1 / n_dis)) * (d ** (m_dis / n_dis)) * (sii_dis ** (1 / n_dis - 1)) * np.exp(
        (Ea_dis + Va_dis * P) / (n_dis * R * (T + 273.15)))
    viscosity_dif = 0.5 * (A_dif ** (-1 / n_dif)) * (d ** (m_dif / n_dif)) * (sii_dif ** (1 / n_dif - 1)) * np.exp(
        (Ea_dif + Va_dif * P) / (n_dif * R * (T + 273.15)))

    viscosity = 1 / (1 / viscosity_dis + 1 / viscosity_dif)

    if viscosity > 1e23:        # Max viscosity for upper mantle is 1e23 Pa s
        viscosity = 1e23

    return viscosity


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
    if label: # if label=None; skip
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