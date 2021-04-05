#!/usr/bin/python3.8
#########################################################################
# This code consists of subroutines to solve 2D Stokes equation.        #
#=======================================================================#
# Author: Wentao Zhang, wzhang@geo3bcn.csic.es                          #
# Created 09/03/2021                                                    #
#########################################################################

# Stokes equation
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import Stokes_pre

start = time.time()
print('Start time: %s ' % time.ctime(start))
# define the size of model, mesh unit, m
# Domain
# input = 1  # input:        the shape of model
# xlen, ylen = 1000000.0, 400000.0      # m

input = 3  # input:        the shape of model
xlen, ylen = 1070000.0, 400000.0      # m

# define the mesh, mesh unit, m
nx, ny = 101, 41

# define the markers
nx_m, ny_m = 4, 2

dx = xlen / (nx - 1)
dy = ylen / (ny - 1)
dx_m = dx / nx_m
dy_m = dy / ny_m

# Velocity Boundary condition specified by BC_left,BC_right,BC_top,bbot
# (1=free slip 0=no slip) are implemented from ghost nodes
# No slip: vx[i,j]=0 # Free slip dvx/dy=0: vx[i,j]-vx[i+1,j]=0
# directly into Stokes and continuity equations
BC_left = 1
BC_right = 1
BC_top = 0
BC_bottom = 0
# Pressure in the upermost, leftmost (first) cell
prfirst = 0.0
cp = 1.0

print('  The length of model: %0.2f km' % (xlen/1000))
print('  The depth of model: %0.2f km' % (ylen/1000))
print('  The number of nodes: %d × %d' % (nx, ny))
print('  The number of markers in each mesh: %d × %d' % (nx_m, ny_m))

# First Pre-Processing
print('Pre-Processing start' )

# time_eta_cal = 8
# if time_eta_cal == 1:
#     Strain_II_m_pre = np.ones((ny_m * (ny - 1), nx_m * (nx - 1)))
#     Strain_II_m_pre = Strain_II_m_pre * 1e-15
#     np.save("Strain_II_m.npy", Strain_II_m_pre)
# Strain_II_m_pre = np.load("Strain_II_m.npy", )
# fig_outsii = plt.figure(figsize=(18, 12))
# plt.suptitle('Time = ' + str(time_eta_cal), fontsize=16, fontweight='bold')
# ax = fig_outsii.add_subplot(231)
# plt.plot(Strain_II_m_pre, '.')
# ax = fig_outsii.add_subplot(234); ax.axis([0, xlen / 1000, 0, ylen / 1000])
# x1 = np.arange(0.5 * dx_m, xlen, dx_m)
# y1 = np.arange(0.5 * dy_m, ylen, dy_m)
# xm, ym = np.meshgrid(x1, y1)
# del x1, y1
# Stokes_pre.Plot_fig(xm, ym, Strain_II_m_pre, ax, 'Strain_II_m_pre', 'Strain / $\mathregular{s^-1}$', 2)

# Setting Initial Strain_II for calculating viscosity
IterationNumber = 0
MaxIterationNumber = 100
Deviation = []
Strain_II_m_pre = np.ones((ny_m * (ny - 1), nx_m * (nx - 1))) * 1e-15
np.save("Strain_II_m.npy", Strain_II_m_pre)

(rock_m, density_m, viscosity_m, mkk, mTT, nx_m, ny_m, xm, ym, nx, ny) = Stokes_pre.main(input, xlen, ylen, nx, ny ,nx_m, ny_m)
print('Pre-Processing end' )
print('Pre-Processing total time: %f s \n' % (time.time()-start))



##  (2) Main
print('Stokes start' )
print('  Computing: 0%' )
Stokesstart   = time.time()


dt      = 1
for time_step in range(1, 2, 1):  # time step
    while IterationNumber < MaxIterationNumber:
        Strain_II_m_pre = np.load("Strain_II_m.npy")
        t = dt * time_step
        IterationNumber += 1

        (rock_m, density_m, viscosity_m, mkk, mTT, nx_m, ny_m, xm, ym, nx, ny) = Stokes_pre.main(input, xlen, ylen, nx,
                                                                                                 ny, nx_m, ny_m)

        # parameters for mesh
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


        L = np.zeros((nx * ny * 3, nx * ny * 3))
        R = np.zeros((nx * ny * 3, 1))

        gx = 0
        gy = 9.8
        Pscale = viscosity[0, 0] / (dx / 2 + dy / 2)

        L = np.zeros((nx * ny * 3, nx * ny * 3))
        R = np.zeros((nx * ny * 3, 1))

        gx = 0
        gy = 9.8
        Pscale = viscosity[0, 0] / (dx / 2 + dy / 2)

        for i in range(nx):
            for j in range(ny):
                ip = (i * ny + j) * 3
                ivx = ip + 1
                ivy = ip + 2

                # B.C.
                # x direction
                if i == 0:  # left boundary,
                    L[ivx, ivx] = Pscale  # vx[i,j]
                    L[ivx, ivx + 3] = -Pscale * BC_left  # vx[i+1,j]
                    R[ivx, 0] = 0
                elif i == nx - 1:  # right boundary
                    L[ivx, ivx] = Pscale  # vx[i,j]
                    L[ivx, ivx - 3] = -Pscale * BC_right  # vx[i-1,j]
                    R[ivx, 0] = 0
                elif j == 0:  # upper boundary
                    L[ivx, ivx] = Pscale  # vx[i,j]
                    L[ivx, ivx + 3] = -Pscale * BC_top  # vx[i+1,j]
                    R[ivx, 0] = 0
                elif j == ny - 2:  # lower boundary
                    L[ivx, ivx] = Pscale
                    L[ivx, ivx - 3] = -Pscale * BC_bottom  # vx[i-1,j]
                    R[ivx, 0] = 0
                elif j == ny - 1:  # lower boundary?
                    L[ivx, ivx] = Pscale  # vx[i,j]
                    R[ivx, 0] = 0
                else:  # internal node

                    L[ivx, ivx] = -viscosity[j + 1, i] / (dy * dy) - viscosity[j, i] / (dy * dy) - (
                                viscosity[j, i] + viscosity[j + 1, i] + viscosity[j, i + 1] + viscosity[j + 1, i + 1]) / (
                                              2 * dx * dx) - (
                                              viscosity[j, i] + viscosity[j + 1, i] + viscosity[j, i - 1] + viscosity[
                                          j + 1, i - 1]) / (2 * dx * dx)
                    L[ivx, ivx + 3] = viscosity[j + 1, i] / (dy * dy)
                    L[ivx, ivx - 3] = viscosity[j, i] / (dy * dy)
                    L[ivx, ivx + 3 * ny] = (viscosity[j, i] + viscosity[j + 1, i] + viscosity[j, i + 1] + viscosity[
                        j + 1, i + 1]) / (2 * dx * dx)
                    L[ivx, ivx - 3 * ny] = (viscosity[j, i - 1] + viscosity[j + 1, i - 1] + viscosity[j, i] + viscosity[
                        j + 1, i]) / (2 * dx * dx)

                    L[ivx, ip + 3 + 3 * ny] = -Pscale / dx
                    L[ivx, ip + 3] = Pscale / dx

                    L[ivx, ivy] = -viscosity[j, i] / (dx * dy)
                    L[ivx, ivy - 3 * ny] = viscosity[j, i] / (dx * dy)
                    L[ivx, ivy + 3 - 3 * ny] = -viscosity[j + 1, i] / (dx * dy)
                    L[ivx, ivy + 3] = viscosity[j + 1, i] / (dx * dy)

                    R[ivx, 0] = -(density[j, i] + density[j + 1, i]) / 2 * gx

                    L[ivy, ivy] = (-viscosity[j, i + 1] - viscosity[j, i]) / (dx * dx)

                # y direction
                if j == 0:  # top boundary,
                    L[ivy, ivy] = 1 * Pscale
                    L[ivy, ivy + 3] = -Pscale * BC_top  # vy[i,j+1]
                    R[ivy, 0] = 0
                elif j == ny - 1:  # lower boundary
                    L[ivy, ivy] = 1 * Pscale
                    L[ivy, ivy - 3] = -Pscale * BC_bottom  # vy[i,j-1]
                    R[ivy, 0] = 0
                elif i == 0:  # left boundary
                    L[ivy, ivy] = 1 * Pscale
                    L[ivy, ivy + 3 * ny] = -1 * Pscale * BC_left
                    R[ivy, 0] = 0
                elif i == nx - 2:  # right boundary
                    L[ivy, ivy] = Pscale
                    L[ivy, ivy - 3 * ny] = -Pscale * BC_right
                    R[ivy, 0] = 0
                elif i == nx - 1:
                    L[ivy, ivy] = 1 * Pscale
                    R[ivy, 0] = 0
                else:

                    L[ivy, ivy] = -viscosity[j, i + 1] / (dx * dx) - viscosity[j, i] / (dx * dx) - (
                                viscosity[j, i] + viscosity[j + 1, i] + viscosity[j, i + 1] + viscosity[j + 1, i + 1]) / (
                                              2 * dy * dy) - (
                                              viscosity[j, i] + viscosity[j - 1, i] + viscosity[j, i + 1] + viscosity[
                                          j - 1, i + 1]) / (2 * dy * dy)
                    L[ivy, ivy + 3] = (viscosity[j, i] + viscosity[j, i + 1] + viscosity[j + 1, i] + viscosity[
                        j + 1, i + 1]) / (2 * dy * dy)
                    L[ivy, ivy - 3] = (viscosity[j, i] + viscosity[j, i + 1] + viscosity[j - 1, i] + viscosity[
                        j - 1, i + 1]) / (2 * dy * dy)
                    L[ivy, ivy + 3 * ny] = viscosity[j, i + 1] / (dx * dx)
                    L[ivy, ivy - 3 * ny] = viscosity[j, i] / (dx * dx)

                    L[ivy, ivx] = -viscosity[j, i] / (dx * dy)
                    L[ivy, ivx - 3] = viscosity[j, i] / (dx * dy)
                    L[ivy, ivx + 3 * ny] = viscosity[j, i + 1] / (dx * dy)
                    L[ivy, ivx - 3 + 3 * ny] = -viscosity[j, i + 1] / (dx * dy)

                    L[ivy, ip + 3 + 3 * ny] = -Pscale / dy
                    L[ivy, ip + 3 * ny] = Pscale / dy

                    R[ivy, 0] = -(density[j, i] + density[j, i + 1]) / 2 * gy

                # solution of the continuity equation
                if i == 0 or j == 0:
                    L[ip, ip] = 1 * Pscale
                    R[ip, 0] = 0
                elif (i == 1 and j == 1) or (i == 1 and j == ny - 1) or (i == nx - 1 and j == 1) or (
                        i == nx - 1 and j == ny - 1):
                    L[ip, ip] = 1 + Pscale
                    R[ip, 0] = 0
                elif i == 2 and j == 1:
                    L[ip, ip] = 1 * Pscale
                    R[ip, 0] = 10 ^ 5
                else:
                    L[ip, ivx - 3] = 1 * Pscale
                    L[ip, ivx - 3 - 3 * ny] = -1 * Pscale
                    L[ip, ivy - 3 * ny] = dx / dy * Pscale
                    L[ip, ivy - 3 - 3 * ny] = -dx / dy * Pscale
                    R[ip, 0] = 0

        XX = np.linalg.solve(L, R)
        print('  Computing: 50%')

        P0 = np.zeros((ny, nx))
        vx0 = np.zeros((ny, nx))
        vy0 = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                ip = (i * ny + j) * 3
                ivx = ip + 1
                ivy = ip + 2

                P0[j, i] = XX[ip, 0] * Pscale
                vx0[j, i] = XX[ivx, 0]
                vy0[j, i] = XX[ivy, 0]

        # vmax = 0
        # v0 = np.zeros((ny, nx))
        # for i in range(nx):
        #     for j in range(ny):
        #         v0[j,i] = (vx0[j,i]**2+vy0[j,i]**2)**0.5
        #         vmax = np.max(vmax,v0[j,i])

        P = np.zeros((ny - 1, nx - 1))
        for i in range(nx - 1):
            for j in range(ny - 1):
                P[j, i] = P0[j + 1, i + 1]

        vx = np.zeros((ny - 1, nx))
        for i in range(nx):
            for j in range(ny - 1):
                vx[j, i] = vx0[j, i]

        vy = np.zeros((ny, nx - 1))
        for i in range(nx - 1):
            for j in range(ny):
                vy[j, i] = vy0[j, i]

        p1 = np.zeros((ny - 1, nx - 1))
        vx1 = np.zeros((ny - 1, nx - 1))
        vy1 = np.zeros((ny - 1, nx - 1))
        for i in range(nx - 1):
            for j in range(ny - 1):
                p1[j, i] = P[j, i]
                vx1[j, i] = (vx[j, i] + vx[j, i + 1]) / 2
                vy1[j, i] = (vy[j, i] + vy[j + 1, i]) / 2

        LL = np.zeros((nx * ny, nx * ny))
        RR = np.zeros((nx * ny, 1))

        for i in range(nx):
            for j in range(ny):
                iTT = i * ny + j
                if i == 0:
                    LL[iTT, iTT] = 1
                    LL[iTT, iTT + ny] = -1
                    RR[iTT, 0] = 0
                elif i == nx - 1:
                    LL[iTT, iTT] = 1
                    LL[iTT, iTT - ny] = -1
                    RR[iTT, 0] = 0
                elif j == 0:
                    LL[iTT, iTT] = 1
                    LL[iTT, iTT + 1] = -1
                    RR[iTT, 0] = 0
                elif j == ny - 1:
                    LL[iTT, iTT] = 1
                    LL[iTT, iTT - 1] = -1
                    RR[iTT, 0] = 0
                else:
                    LL[iTT, iTT] = -density[j, i] * cp \
                                   - (kk[j, i] + kk[j, i + 1]) / (2 * dx * dx) * dt \
                                   - (kk[j, i - 1] + kk[j, i]) / (2 * dx * dx) * dt \
                                   - (kk[j + 1, i] + kk[j, i]) / (2 * dy * dy) * dt \
                                   - (kk[j, i] + kk[j - 1, i]) / (2 * dy * dy) * dt
                    LL[iTT, iTT + ny] = (kk[j, i] + kk[j, i + 1]) / (2 * dx * dx) * dt
                    LL[iTT, iTT - ny] = (kk[j, i] + kk[j, i - 1]) / (2 * dx * dx) * dt
                    LL[iTT, iTT + 1] = (kk[j + 1, i] + kk[j, i]) / (2 * dy * dy) * dt
                    LL[iTT, iTT - 1] = (kk[j - 1, i] + kk[j, i]) / (2 * dy * dy) * dt

                    RR[iTT, 0] = -density[j, i] * cp * TT[j, i]

        YY = np.linalg.solve(LL, RR)

        TTnew = np.zeros((ny, nx))
        delTT = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                iT = i * ny + j
                TTnew[j, i] = YY[iT, 0]
                delTT[j, i] = TTnew[j, i] - TT[j, i]

        for i in range(nx):
            for j in range(ny):
                # iT = (i - 1) * ny + j
                TT[j, i] = TTnew[j, i]

        # transfer the properties on the nodes to the markers
        vxm = np.zeros(((ny - 1) * ny_m, (nx - 1) * nx_m))
        vym = np.zeros(((ny - 1) * ny_m, (nx - 1) * nx_m))
        delTTm = np.zeros(((ny - 1) * ny_m, (nx - 1) * nx_m))
        TTm = np.zeros(((ny - 1) * ny_m, (nx - 1) * nx_m))

        # dt2 = 0.25 * min_density * min_cp * min_dxdy / min_kk
        for im in range((nx - 1) * nx_m):
            for jm in range((ny - 1) * ny_m):
                # vx
                i = int(round(xm[jm, im] / dx - 0.5))
                if i < 0:
                    i = 0
                else:
                    if i > nx - 2:
                        i = nx - 2

                delxm = xm[jm, im] / dx - i

                # j=floor(ym[jm,im]/dy)
                j = int(round(ym[jm, im] / dy - 0.5 - 0.5))
                if j < 0:
                    j = 0
                else:
                    if j > ny - 3:
                        j = ny - 3

                delym = ym[jm, im] / dy - 0.5 - j
                vxm[jm, im] = vx[j, i] * (1 - delxm) * (1 - delym) \
                              + vx[j, i + 1] * (delxm) * (1 - delym) \
                              + vx[j + 1, i] * (1 - delxm) * (delym) \
                              + vx[j + 1, i + 1] * (delxm) * (delym)
                #         weight     = (1-delym)*(1-delym)\
                #                     +(delym)*(1-delym)\
                #                     +(1-delym)*(delym)\
                #                     +(delym)*(delym)

                # vy
                i = int(round(xm[jm, im] / dx - 0.5))
                if i < 0:
                    i = 0
                else:
                    if i > nx - 3:
                        i = nx - 3
                delxm = xm[jm, im] / dx - i - 0.5

                # j=floor(ym[jm,im]/dy)
                j = int(round(ym[jm, im] / dy - 0.5 - 0.5))
                if j < 0:
                    j = 0
                else:
                    if j > ny - 2:
                        j = ny - 2

                delym = ym[jm, im] / dy - j
                vym[jm, im] = vy[j, i] * (1 - delxm) * (1 - delym) \
                              + vy[j, i + 1] * (delxm) * (1 - delym) \
                              + vy[j + 1, i] * (1 - delxm) * (delym) \
                              + vy[j + 1, i + 1] * (delxm) * (delym)

                # TT
                i = int(round(xm[jm, im] / dx - 0.5))
                if i < 0:
                    i = 0
                else:
                    if i > nx - 2:
                        i = nx - 2

                delxm = xm[jm, im] / dx - i
                # j=floor(ym[jm,im]/dy)
                j = int(round(ym[jm, im] / dy - 0.5))
                if j < 0:
                    j = 0
                else:
                    if j > ny - 2:
                        j = ny - 2

                delym = ym[jm, im] / dy - j

                delTTm[jm, im] = delTT[j, i] * (1 - delxm) * (1 - delym) \
                                 + delTT[j, i + 1] * (delxm) * (1 - delym) \
                                 + delTT[j + 1, i] * (1 - delxm) * (delym) \
                                 + delTT[j + 1, i + 1] * (delxm) * (delym)
                TTm[jm, im] = mTT[jm, im] + delTTm[jm, im]

        # if dt>min(dt1,dt2)
        # dt=min(dt1,dt2)
        #
        #
        for im in range((nx - 1) * nx_m):
            for jm in range((ny - 1) * ny_m):
                xm[jm, im] = xm[jm, im] + vxm[jm, im] * dt
                ym[jm, im] = ym[jm, im] + vym[jm, im] * dt

        print('Stokes end')
        print('Stokes-solver time: %f s \n' % (time.time() - Stokesstart))

        # (3) visualization, Velocity, Viscosity, Density, Stress, Strain, Pressure, -Thermal parameters(T,Q,HF,K,k,cp,A and so on)
        # Result: vx1,vy1,velocity / center of mesh
        #   Unit transformation
        # viscosity /Plotting viscosity: vx1, vy1, m/s

        VunitType = 1
        if VunitType == 0:
            Scale = 1
            Vlable = ' ${m}/{s}$'
        if VunitType == 1:
            Scale = 1 * 365.25 * 24 * 3600 * 100
            Vlable = ' ${cm}/{y}$'

        Vx = vx1 * Scale
        Vy = vy1 * Scale
        V = (Vx ** 2 +Vy ** 2) ** 0.5

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

        Stress_xx_d = np.zeros((ny-1, nx-1))
        Stress_yy_d = np.zeros((ny-1, nx-1))
        Stress_xy = np.zeros((ny-1, nx-1))
        Stress_yx = np.zeros((ny-1, nx-1))
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

        ## vx1,vy1,viscosity,Density / nodes of mesh

        fig_out = plt.figure(figsize=(12, 12))
        plt.suptitle('Stress', fontsize=16, fontweight='bold')
        ax = fig_out.add_subplot(221); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Stress_xx_d, ax, 'Stress_xx_d', 'Stress_xx_d / Pa', 2)
        ax = fig_out.add_subplot(222); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Stress_yy_d, ax, 'Stress_yy_d', 'Stress_yy_d / Pa', 2)
        ax = fig_out.add_subplot(223); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Stress_xy, ax, 'Stress_xy', 'Stress_xy / Pa', 2)
        ax = fig_out.add_subplot(224); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Stress_yx, ax, 'Stress_yx', 'Stress_yx / Pa', 2)
        plt.close()
        fig_out = plt.figure(figsize=(12, 12))

        plt.suptitle('Strain_II', fontsize=16, fontweight='bold')
        ax = fig_out.add_subplot(221); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xm, ym, Strain_II_m_pre, ax, 'Strain_II before', 'Strain / $\mathregular{s^-1}$', 2)

        points = np.column_stack((xx.flatten(), yy.flatten()))
        Strain_II_m_new = griddata(points, Strain_II.flatten(), (xm, ym), method='nearest')
        for im in range(nx_m * (nx - 1)):
            for jm in range(ny_m * (ny - 1)):
                if Strain_II_m_new[jm, im] == 0:              # sticky air
                    Strain_II_m_new[jm, im] = 1e-18
        # Strain_II_m_new[np.isnan(Strain_II_m_new)] = 0
        # Strain_II_m_new[np.where(Strain_II_m_new == 0)] = 1e-18
        ax = fig_out.add_subplot(222); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xm, ym, Strain_II_m_new, ax, 'Strain_II after', 'Strain / $\mathregular{s^-1}$', 2)

        ax = fig_out.add_subplot(223); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        # plt.plot(Strain_II_m_new.flatten(), 'b.')
        # plt.plot(Strain_II_m_pre-Strain_II_m_new, 'y*')
        Stokes_pre.Plot_fig(xm, ym, Strain_II_m_new - Strain_II_m_pre, ax, 'Strain_II after', 'Strain / $\mathregular{s^-1}$', 2)
        np.save("Strain_II_m.npy", Strain_II_m_new)

        Strain_II_D = (Strain_II_m_new - Strain_II_m_pre) / Strain_II_m_pre
        Strain_II_D2 = (Strain_II_D ** 2) ** 0.5
        Deviation.append(Strain_II_D2.mean())

        ax = fig_out.add_subplot(224)
        plt.plot(Strain_II_D.flatten(), 'r.')
        plt.savefig('/home/wentao/Videos/'+str(IterationNumber))
        # plt.show()

        fig_out = plt.figure(figsize=(12, 12))
        plt.suptitle('Strain', fontsize=16, fontweight='bold')
        ax = fig_out.add_subplot(221); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Strain_xx_d, ax, 'Strain_xx_d', 'Strain / $\mathregular{s^-1}$', 2)
        ax = fig_out.add_subplot(222); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Strain_yy_d, ax, 'Strain_yy_d', 'Strain / $\mathregular{s^-1}$', 2)
        ax = fig_out.add_subplot(223); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Strain_xy, ax, 'Strain_xy', 'Strain / $\mathregular{s^-1}$', 2)
        ax = fig_out.add_subplot(224); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Strain_yx, ax, 'Strain_yx', 'Strain / $\mathregular{s^-1}$', 2)

        #
        plt.close()
        plt.close()
        # viscosity /Plotting viscosity: vx1, vy1

        Qkey = np.max(V) # Qkey = (np.max(Vx) ** 2 + np.max(Vy) ** 2) ** 0.5
        fig_out = plt.figure(figsize=(12, 12))
        plt.suptitle('Viscosity', fontsize=16, fontweight='bold')
        # ax = fig_out.add_subplot(221); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        # Stokes_pre.Plot_fig(xx, yy, V, ax, 'V', None, 3)
        # Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
        # plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

        ax = fig_out.add_subplot(222); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, V, ax, 'V', 'V' + Vlable, 2)
        # Stokes_pre.Plot_fig(xm, ym, np.log10(viscosity_m), ax, 'Viscosity', 'log$_{10}$η ($Pa·s$)', 2)
        Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
        plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

        ax = fig_out.add_subplot(223); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Vx, ax, 'Vx', 'Vx' + Vlable, 2)
        Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
        plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

        ax = fig_out.add_subplot(224); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        Stokes_pre.Plot_fig(xx, yy, Vy, ax, 'Vy', 'Vy' + Vlable, 2)
        Q = plt.quiver(xx / 1000, yy / 1000, Vx, -1 * Vy, units='xy', color='red')
        plt.quiverkey(Q, 0.8, -0.1, Qkey, str(Qkey) + Vlable, labelpos='E', color='red', coordinates='axes')

        plt.close()

        # points = np.column_stack((xx.flatten(), yy.flatten()))
        # Strain_II_m = griddata(points, Strain_II.flatten(), (xm, ym), method='nearest')
        # for im in range(nx_m * (nx - 1)):
        #     for jm in range(ny_m * (ny - 1)):
        #         if Strain_II_m[jm, im] == 0:              # sticky air
        #             Strain_II_m[jm, im] = 1e-18

        # fig_out = plt.figure(figsize=(12, 12))
        # plt.suptitle('Time = ' + str(time_eta_cal), fontsize=16, fontweight='bold')
        # ax = fig_outsii.add_subplot(232)
        # plt.plot(Strain_II_m, '.')

        # ax = fig_outsii.add_subplot(235); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        # Stokes_pre.Plot_fig(xm, ym, Strain_II_m, ax, 'Strain_II_m', 'Strain / $\mathregular{s^-1}$', 2)
        # np.save("Strain_II_m.npy", Strain_II_m)

        # Strain_II_D = (Strain_II_m_pre-Strain_II_m)
        #
        # ax = fig_outsii.add_subplot(233)
        # plt.plot(Strain_II_D, '.')
        #
        # ax = fig_outsii.add_subplot(236); ax.axis([0, xlen / 1000, 0, ylen / 1000])
        # Stokes_pre.Plot_fig(xm, ym, Strain_II_D, ax, 'Strain_II_D', 'Strain / $\mathregular{s^-1}$', 2)
        # # plt.savefig('Time = ' + str(time_eta_cal) + '.png')


# plt.show()
plt.figure()
plt.plot(Deviation, '*')
plt.show()
plt.savefig('/home/wentao/Videos/Deviation.png')
