import numpy as np
from numba import int64

from utils.numba_kernels import *
from utils.parameters import *
from utils.numpy_functions import np_obstacle_fun, np_inivel

from functools import partial

INTNX = int64(nx)
INTNY = int64(ny)

TPB1D, BPG1D = dispatch1D(ny)
TPB2D, BPG2D = dispatch(nx, ny)


def main():
    # Numpy part
    obstacle = np.fromfunction(partial(np_obstacle_fun, cx=cx, cy=cy, r=r), (nx, ny))
    vel = np.fromfunction(partial(np_inivel, ly=ly, uLB=uLB), (2, nx, ny))
    rho = np.full((nx, ny), 1)
    omega_ = np.full((nx, ny), omega)

    # GPU part
    d_omega, d_obstacle = map(cuda.to_device, (omega_, obstacle))

    d_rho, d_vel, d_v, d_t = map(cuda.to_device, (rho, vel, v, t))
    d_fin = cuda.device_array((9, nx, ny))
    d_u = cuda.device_array((2, nx, ny))
    d_feq = cuda.device_array((9, nx, ny))
    d_fout = cuda.device_array((9, nx, ny))

    equilibrium[BPG2D, TPB2D](d_rho, d_vel, d_v, d_t, d_fin, INTNX, INTNY)

    for time in range(maxIter):
        outflow[BPG1D, TPB1D](d_fin, INTNX, INTNY)

        macroscopic[BPG2D, TPB2D](d_fin, d_v, d_rho, d_u, INTNX, INTNY)

        inflow[BPG1D, TPB1D](d_u, d_vel, d_rho, d_fin, INTNY)

        # if time == 1:
        #     start = cuda.event()
        #     start.record()
        #     equilibrium[BPG2D, TPB2D](d_rho, d_u, d_v, d_t, d_feq, INTNX, INTNY)
        #     cuda.synchronize()
        #     stop = cuda.event()
        #     stop.record()
        #     print(cuda.event_elapsed_time(start, stop))
        # else:
        equilibrium[BPG2D, TPB2D](d_rho, d_u, d_v, d_t, d_feq, INTNX, INTNY)

        update_fin[BPG1D, TPB1D](d_fin, d_feq, INTNY)

        collision[BPG2D, TPB2D](d_omega, d_fin, d_feq, d_fout, INTNX, INTNY)

        bounce_back[BPG2D, TPB2D](d_fout, d_feq, d_obstacle, INTNX, INTNY)

        streaming_step[BPG2D, TPB2D](d_fin, d_fout, d_v, INTNX, INTNY)


if __name__ == "__main__":
    main()
