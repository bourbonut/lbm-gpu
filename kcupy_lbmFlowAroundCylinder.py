import numpy
import cupy

import cv2
import cmapy

from utils.cupy_kernels import *
from utils.parameters import *
from utils.numpy_functions import np_obstacle_fun, np_inivel

from functools import partial

INTNX = nx
INTNY = ny

TPB1D, BPG1D = dispatch1D(ny)
TPB2D, BPG2D = dispatch(nx, ny)

frameSize = (INTNX, INTNY)
path_video = "output_video.avi"
bin_loader = cv2.VideoWriter_fourcc(*"DIVX")
out = cv2.VideoWriter(path_video, bin_loader, 120, frameSize)


def main():
    # Numpy part
    obstacle = numpy.fromfunction(partial(np_obstacle_fun, cx=cx, cy=cy, r=r), (nx, ny))
    vel = numpy.fromfunction(partial(np_inivel, ly=ly, uLB=uLB), (2, nx, ny))

    # GPU part
    d_rho = cupy.full((nx, ny), 1.0)
    d_omega = cupy.full((nx, ny), omega)

    d_obstacle, d_vel, d_v, d_t = map(cupy.array, (obstacle, vel, v, t))
    d_fin = cupy.zeros((9, nx, ny))
    d_u = cupy.zeros((2, nx, ny))
    d_feq = cupy.zeros((9, nx, ny))
    d_fout = cupy.zeros((9, nx, ny))

    equilibrium[BPG2D, TPB2D](d_rho, d_vel, d_v, d_t, d_fin, INTNX, INTNY)

    for time in range(maxIter + 1):
        outflow[BPG1D, TPB1D](d_fin, INTNX, INTNY)

        macroscopic[BPG2D, TPB2D](d_fin, d_v, d_rho, d_u, INTNX, INTNY)

        inflow[BPG1D, TPB1D](d_u, d_vel, d_rho, d_fin, INTNY)

        equilibrium[BPG2D, TPB2D](d_rho, d_u, d_v, d_t, d_feq, INTNX, INTNY)

        update_fin[BPG1D, TPB1D](d_fin, d_feq, INTNY)

        collision[BPG2D, TPB2D](d_omega, d_fin, d_feq, d_fout, INTNX, INTNY)

        bounce_back[BPG2D, TPB2D](d_fout, d_feq, d_obstacle, INTNX, INTNY)

        streaming_step[BPG2D, TPB2D](d_fin, d_fout, d_v, INTNX, INTNY)

        if time % 10 == 0 and time != 0:
            print(round(100 * time / maxIter, 3), "%")
            u = d_u.get()
            arr = np.sqrt(u[0] ** 2 + u[1] ** 2).transpose()
            new_arr = ((arr / arr.max()) * 255).astype("uint8")
            img_colorized = cv2.applyColorMap(new_arr, cmapy.cmap("plasma"))
            out.write(img_colorized)

    out.release()


if __name__ == "__main__":
    main()
