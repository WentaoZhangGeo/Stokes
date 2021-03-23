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
import Stokes_pre

start = time.time()
print('Start time: %s ' % time.ctime(start))
# define the size of model, mesh unit, m
# Domain
input = 1  # input:        the shape of model
xlen, ylen = 1000000.0, 400000.0      # m

# define the mesh, mesh unit, m
nx, ny = 31, 21

# define the markers
nx_m, ny_m = 4, 2

# Velocity Boundary condition specified by BC_left,BC_right,BC_top,bbot
# (1=free slip 0=no slip) are implemented from ghost nodes
# No slip: vx[i,j]=0# Free slip dvx/dy=0: vx[i,j]-vx[i+1,j]=0
# directly into Stokes and continuity equations
BC_left = 0
BC_right = 0
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
(rock_m, density_m, viscosity_m, mkk, mTT, nx_m, ny_m, xm, ym, nx, ny) = Stokes_pre.main(input, xlen, ylen, nx, ny ,nx_m, ny_m)
print('Pre-Processing end' )
print('Pre-Processing total time: %f s \n' % (time.time()-start))

##  (2) Main
print('Stokes start' )
print('  Computing: 0%' )
Stokesstart   = time.time()
dx = xlen / (nx - 1)
dy = ylen / (ny - 1)

dt      = 1
for time_step in range(1,2,1): # time step
    t = dt * time_step
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
    x1 = np.arange(dx/2, xlen, dx)
    y1 = np.arange(dy/2, ylen, dy)
    xx, yy = np.meshgrid(x1, y1)
    del x1, y1

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

    fig_out = plt.figure(figsize=(12, 12))
    plt.suptitle('Strain_II', fontsize=16, fontweight='bold')
    ax = fig_out.add_subplot(221); ax.axis([0, xlen / 1000, 0, ylen / 1000])
    Stokes_pre.Plot_fig(xx, yy, Strain_II, ax, 'Strain_II', 'Strain / $\mathregular{s^-1}$', 2)

    fig_out.add_subplot(222)
    plt.plot(Strain_II.flatten(), '.')
    avg = sum(Strain_II.flatten())/len(Strain_II.flatten())
    print(avg)

    sii0 = 1e-15
    # sii0 = 7.8e-16

    # plt.plot(Strain_II.flatten() - sii0, '.')
    line = np.zeros((2, 2))
    line[0, 0] = 0
    line[0, 1] = len(Strain_II.flatten())
    line[1, 0] = sii0
    line[1, 1] = sii0
    plt.plot(line[0, :], line[1, :], '-')
    ax = plt.gca()
    ax.set_title('Strain_II distribution map')
    ax.set_xlabel('Point')
    ax.set_ylabel('Strain / $\mathregular{s^-1}$')
    error = (sum((Strain_II.flatten() - sii0) ** 2)) ** 0.5 / len(Strain_II.flatten())
    print(error)
    # plt.plot(1,len(Strain_II.flatten()), '*')


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

    # viscosity /Plotting viscosity


plt.show()