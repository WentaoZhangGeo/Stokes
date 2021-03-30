import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import Visualization as vis
# import main

def marker(Model_inf):
    fig_switch = 0  # Check out the fig, 1 is off, other is on
    PA = 200000  # vertical_slice
    PB = 600000  # vertical_slice
    outfile = 'Model_marker.npz'
    global xx, yy, xm, ym, x_n, y_n
    Model_inf = np.load(Model_inf)
    input = Model_inf['input']
    xlen, ylen, nx, ny, nx_m, ny_m = \
        Model_inf['xlen'], Model_inf['ylen'], Model_inf['nx'], Model_inf['ny'], Model_inf['nx_m'], Model_inf['ny_m']
    dx, dy, dx_m, dy_m = \
        Model_inf['dx'], Model_inf['dy'], Model_inf['dx_m'], Model_inf['dy_m']
    xx, yy, xm, ym, x_n, y_n = \
        Model_inf['xx'], Model_inf['yy'], Model_inf['xm'], Model_inf['ym'], Model_inf['x_n'], Model_inf['y_n']

    if input == 1:
        (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_box(nx, nx_m, ny, ny_m, xm, ym)
    # if input == 2:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_Fun(nx, nx_m, ny, ny_m, xm, ym, dx_m, dy_m)
    if input == 3:
        (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_LitMod(nx, nx_m, ny, ny_m, xm, ym)
    # if input == 4:
    #     (rock_m, density_m, viscosity_m, mkk, mTT) = input_parameters_circle(nx, nx_m, ny, ny_m, xm, ym)

    fig_out = plt.figure(figsize=(16, 12))
    # meshes
    ax = fig_out.add_subplot(231)
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    vis.plot_fig(x_n, y_n, 'No', ax, 'Grid: ' + str(nx - 1) + ' × ' + str(ny - 1), None, 1)

    ax = fig_out.add_subplot(232)
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    vis.plot_fig(xm, ym, density_m, ax, 'Density', 'ρ (kg/$\mathregular{m^3}$)', 2)

    ax = fig_out.add_subplot(233)
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    vis.plot_fig(xm, ym, np.log10(viscosity_m), ax, 'Viscosity', 'log$_{10}$η ($Pa·s$)', 2)

    ax = fig_out.add_subplot(234)
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    vis.plot_fig(xm, ym, rock_m, ax, 'Type of rock', 'Number', 2)

    ax = fig_out.add_subplot(235)
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    vis.plot_fig(xm, ym, mTT, ax, 'Temperture', 'Temperture ($deg$)', 2)

    ax = fig_out.add_subplot(236)
    ax.axis([0, xlen / 1000, 0, ylen / 1000])
    vis.plot_fig(xm, ym, mkk, ax, 'Thermal conductivity', 'k ($W/(m·K)$)', 2)

    if fig_switch == 0:
        plt.close()

    # vertical_slice
    im_find = abs(xm[1, :] - PA)
    im = np.argmin(im_find)
    legend = 'x=' + str(xm[1, im] / 1000) + ' km'

    fig_out = plt.figure(figsize=(8, 12))
    ax = fig_out.add_subplot(121)
    vis.plot_slip(ym[:, im], np.log10(viscosity_m[:, im]), ax, 'viscosity profile', 'log$_{10}$Viscosity (Pa s)',
                  legend)

    im_find = abs(xm[1, :] - PB)
    im = np.argmin(im_find)
    legend = 'x=' + str(xm[1, im] / 1000) + ' km'
    ax = fig_out.add_subplot(122)
    vis.plot_slip(ym[:, im], np.log10(viscosity_m[:, im]), ax, 'viscosity profile', 'log$_{10}$Viscosity (Pa s)',
                  legend)

    if fig_switch == 0:
        plt.close()

    np.savez(outfile, rock_m=rock_m,
             density_m=density_m, viscosity_m=viscosity_m,
             mkk=mkk, mTT=mTT)
    return outfile

def class2str(Model):
    input = Model.input
    xlen = Model.xlen
    ylen = Model.ylen
    nx = Model.nx
    ny = Model.ny
    nx_m = Model.nx_m
    ny_m = Model.ny_m
    dx = Model.dx
    dy = Model.dy
    dx_m = Model.dx_m
    dy_m = Model.dy_m
    return input, xlen, ylen, nx, ny, nx_m, ny_m, dx, dy, dx_m, dy_m


def input_parameters_box(nx, nx_m, ny, ny_m, xm, ym):  # input=1
    print('  The shape of model: Box')
    rock_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    density_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    viscosity_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    mkk = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    mTT = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))

    for im in range(nx_m * (nx - 1)):
        for jm in range(ny_m * (ny - 1)):
            if 200000 < xm[jm, im] < 300000 and 100000 < ym[jm, im] < 200000:
                rock_m[jm, im] = 1
                density_m[jm, im] = 3000
                viscosity_m[jm, im] = 1e22
                mkk[jm, im] = 1e3
                mTT[jm, im] = (xm[jm, im] + ym[jm, im]) / 1000
            else:
                rock_m[jm, im] = 2
                density_m[jm, im] = 2800
                viscosity_m[jm, im] = 1e20
                mkk[jm, im] = 1e4
                mTT[jm, im] = (xm[jm, im] + ym[jm, im]) / 1000

    return rock_m, density_m, viscosity_m, mkk, mTT


def input_parameters_LitMod(nx, nx_m, ny, ny_m, xm, ym):
    print('  The shape of model: LitMod2D_2.0')
    rock_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    density_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    viscosity_m = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))
    mkk = np.ones((ny_m * (ny - 1), nx_m * (nx - 1)))
    mTT = np.zeros((ny_m * (ny - 1), nx_m * (nx - 1)))

    file = '/home/ictja/PycharmProjects/dens_node2.dat'
    data = np.loadtxt(file)
    points = np.column_stack((data[:, 0], -1 * data[:, 1])) * 1000
    density_m = griddata(points, data[:, 2], (xm, ym), method='nearest')
    del data, points

    file = '/home/ictja/PycharmProjects/tempout.dat'
    data = np.loadtxt(file)
    points = np.column_stack((data[:, 0], -1 * data[:, 1])) * 1000
    mTT = griddata(points, data[:, 2], (xm, ym), method='nearest')
    del data, points

    file = '/home/ictja/PycharmProjects/post_processing_output.dat'
    data = np.loadtxt(file)
    points = np.column_stack((data[:, 0], -1 * data[:, 1])) * 1000
    rock_m = griddata(points, data[:, 7], (xm, ym), method='nearest')
    Pressure = griddata(points, data[:, 3], (xm, ym), method='nearest')

    Strain_II = np.load("Strain_II_m.npy")
    for im in range(nx_m * (nx - 1)):
        for jm in range(ny_m * (ny - 1)):
            if rock_m[jm, im] < 0:  # sticky air
                rock_m[jm, im] = 0
                viscosity_m[jm, im] = 1e18
            elif 0 <= rock_m[jm, im] <= 10:  # sediment & crust
                rock_m[jm, im] = 1
                viscosity_m[jm, im] = 1e20
            elif 10 < rock_m[jm, im] <= 90:  # Lit_mantle
                rock_m[jm, im] = 2
                viscosity_m[jm, im] = Rheology_HK03(Pressure[jm, im], mTT[jm, im], Strain_II[jm, im])
            elif 91 < rock_m[jm, im] <= 100:  # Sub_mantle
                rock_m[jm, im] = 3
                viscosity_m[jm, im] = Rheology_HK03(Pressure[jm, im], mTT[jm, im], Strain_II[jm, im])
    return rock_m, density_m, viscosity_m, mkk, mTT


def marker2node(Model_inf, outfile_m):
    outfile_n = 'Model_node.npz'

    Model_inf = np.load(Model_inf)
    input = Model_inf['input']
    xlen, ylen, nx, ny, nx_m, ny_m = \
        Model_inf['xlen'], Model_inf['ylen'], Model_inf['nx'], Model_inf['ny'], Model_inf['nx_m'], Model_inf['ny_m']
    dx, dy, dx_m, dy_m = \
        Model_inf['dx'], Model_inf['dy'], Model_inf['dx_m'], Model_inf['dy_m']
    xx, yy, xm, ym, x_n, y_n = \
        Model_inf['xx'], Model_inf['yy'], Model_inf['xm'], Model_inf['ym'], Model_inf['x_n'], Model_inf['y_n']

    npzf = np.load(outfile_m)
    rock_m, density_m, viscosity_m, mkk, mTT = \
        npzf['rock_m'], npzf['density_m'], \
        npzf['viscosity_m'], npzf['mkk'], npzf['mTT']

    rock = np.zeros((ny, nx))
    density = np.zeros((ny, nx))
    viscosity = np.zeros((ny, nx))
    kk = np.zeros((ny, nx))
    TT = np.zeros((ny, nx))
    weight = np.zeros((ny, nx))

    for im in range(nx_m * (nx - 1)):
        for jm in range(ny_m * (ny - 1)):
            i = int(xm[jm, im] / dx)
            j = int(ym[jm, im] / dy)
            if i < 0: i = 0
            if i > nx - 2: i = nx - 2
            xm_del = xm[jm, im] / dx - i

            if j < 0: j = 0
            if i > nx - 2: i = nx - 2
            ym_del = ym[jm, im] / dy - j

            #
            rock[j, i] = rock[j, i] + rock_m[jm, im] * (1 - xm_del) * (1 - ym_del)
            rock[j, i + 1] = rock[j, i + 1] + rock_m[jm, im] * xm_del * (1 - ym_del)
            rock[j + 1, i + 1] = rock[j + 1, i + 1] + rock_m[jm, im] * xm_del * ym_del
            rock[j + 1, i] = rock[j + 1, i] + rock_m[jm, im] * (1 - xm_del) * ym_del
            #
            density[j, i] = density[j, i] + density_m[jm, im] * (1 - xm_del) * (1 - ym_del)
            density[j, i + 1] = density[j, i + 1] + density_m[jm, im] * xm_del * (1 - ym_del)
            density[j + 1, i + 1] = density[j + 1, i + 1] + density_m[jm, im] * xm_del * ym_del
            density[j + 1, i] = density[j + 1, i] + density_m[jm, im] * (1 - xm_del) * ym_del
            #
            viscosity[j, i] = viscosity[j, i] + viscosity_m[jm, im] * (1 - xm_del) * (1 - ym_del)
            viscosity[j, i + 1] = viscosity[j, i + 1] + viscosity_m[jm, im] * xm_del * (1 - ym_del)
            viscosity[j + 1, i + 1] = viscosity[j + 1, i + 1] + viscosity_m[jm, im] * xm_del * ym_del
            viscosity[j + 1, i] = viscosity[j + 1, i] + viscosity_m[jm, im] * (1 - xm_del) * ym_del
            #
            weight[j, i] = weight[j, i] + (1 - xm_del) * (1 - ym_del)
            weight[j, i + 1] = weight[j, i + 1] + xm_del * (1 - ym_del)
            weight[j + 1, i + 1] = weight[j + 1, i + 1] + xm_del * ym_del
            weight[j + 1, i] = weight[j + 1, i] + (1 - xm_del) * ym_del

            kk[j, i] = kk[j, i] + mkk[jm, im] * (1 - xm_del) * (1 - ym_del)
            kk[j, i + 1] = kk[j, i + 1] + mkk[jm, im] * xm_del * (1 - ym_del)
            kk[j + 1, i + 1] = kk[j + 1, i + 1] + mkk[jm, im] * xm_del * ym_del
            kk[j + 1, i] = kk[j + 1, i] + mkk[jm, im] * (1 - xm_del) * ym_del

            TT[j, i] = TT[j, i] + mTT[jm, im] * (1 - xm_del) * (1 - ym_del)
            TT[j, i + 1] = TT[j, i + 1] + mTT[jm, im] * xm_del * (1 - ym_del)
            TT[j + 1, i + 1] = TT[j + 1, i + 1] + mTT[jm, im] * xm_del * ym_del
            TT[j + 1, i] = TT[j + 1, i] + mTT[jm, im] * (1 - xm_del) * ym_del
    for i in range(nx):
        for j in range(ny):
            if weight[j, i] != 0:
                rock[j, i] = rock[j, i] / weight[j, i]
                density[j, i] = density[j, i] / weight[j, i]
                viscosity[j, i] = viscosity[j, i] / weight[j, i]
                kk[j, i] = kk[j, i] / weight[j, i]
                TT[j, i] = TT[j, i] / weight[j, i]

    np.savez(outfile_n, rock=rock, density=density, viscosity=viscosity, kk=kk, TT=TT)

    return outfile_n


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

    if viscosity > 1e23:  # Max viscosity for upper mantle is 1e23 Pa s
        viscosity = 1e23

    return viscosity
