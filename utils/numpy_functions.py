import numpy as np
from utils.parameters import nx, ny

def np_macroscopic(fin, v):
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u


def np_equilibrium(rho, u, v, t):
    usqr = 3 / 2 * (u[0] ** 2 + u[1] ** 2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        cu = 3 * (v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * t[i] * (1 + cu + 0.5 * cu ** 2 - usqr)
    return feq


def np_obstacle_fun(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2


def np_inivel(d, x, y, ly, uLB):
    return (1.0 - d) * uLB * (1.0 + 1e-4 * np.sin(y / ly * 2.0 * np.pi))


def np_outflow(fin, col3, nx):
    fin[col3, nx - 1, :] = fin[col3, nx - 2, :]


def np_inflow(u, vel, rho, fin, col2, col3):
    u[:, 0, :] = vel[:, 0, :]
    rho[0, :] = (
        1
        / (1 - u[0, 0, :])
        * (np.sum(fin[col2, 0, :], axis=0) + 2 * np.sum(fin[col3, 0, :], axis=0))
    )


def np_update_fin(fin, feq):
    fin[[0, 1, 2], 0, :] = feq[[0, 1, 2], 0, :] + fin[[8, 7, 6], 0, :] - feq[[8, 7, 6], 0, :]


def np_collision(fin, feq, omega):
    return fin - omega * (fin - feq)


def np_bounce_back(fout, fin, obstacle):
    for i in range(9):
        fout[i, obstacle] = fin[8 - i, obstacle]


def np_streaming_step(fin, fout, v):
    for i in range(9):
        fin[i, :, :] = np.roll(np.roll(fout[i, :, :], v[i, 0], axis=0), v[i, 1], axis=1)
