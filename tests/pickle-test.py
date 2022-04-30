import numpy as np

from utils.parameters import *
from utils.numpy_functions import *

import pickle
from functools import partial
from pathlib import Path

PATH = Path().absolute() / "tests" / "picklefiles"

if not (PATH.exists()):
    PATH.mkdir()

obstacle = np.fromfunction(partial(np_obstacle_fun, cx=cx, cy=cy, r=r), (nx, ny))
vel = np.fromfunction(partial(np_inivel, ly=ly, uLB=uLB), (2, nx, ny))
rho = np.full((nx, ny), 2)

maxIter = 10

for i in range(maxIter):
    with open(PATH / "equilibrium-test-{}.pkl".format(1 + i * 8), "wb") as file:
        pickle.dump([rho, vel, v, t], file)
    print("{}/{} files".format(1 + i * 8, maxIter * 8))

    fin = np_equilibrium(rho, vel, v, t)

    with open(PATH / "outflow-test-{}.pkl".format(2 + i * 8), "wb") as file:
        pickle.dump([fin, col3], file)
    print("{}/{} files".format(2 + i * 8, maxIter * 8))

    np_outflow(fin, col3, nx)

    with open(PATH / "macroscopic-test-{}.pkl".format(3 + i * 8), "wb") as file:
        pickle.dump([fin, v], file)
    print("{}/{} files".format(3 + i * 8, maxIter * 8))

    rho, u = np_macroscopic(fin, v)

    with open(PATH / "inflow-test-{}.pkl".format(4 + i * 8), "wb") as file:
        pickle.dump([u, vel, rho, fin, col2, col3], file)
    print("{}/{} files".format(4 + i * 8, maxIter * 8))

    np_inflow(u, vel, rho, fin, col2, col3)
    feq = np_equilibrium(rho, u, v, t)

    with open(PATH / "updatefin-test-{}.pkl".format(5 + i * 8), "wb") as file:
        pickle.dump([fin, feq], file)
    print("{}/{} files".format(5 + i * 8, maxIter * 8))

    np_update_fin(fin, feq)

    with open(PATH / "collision-test-{}.pkl".format(6 + i * 8), "wb") as file:
        pickle.dump([fin, feq, omega], file)
    print("{}/{} files".format(6 + i * 8, maxIter * 8))

    fout = np_collision(fin, feq, omega)

    with open(PATH / "bounceback-test-{}.pkl".format(7 + i * 8), "wb") as file:
        pickle.dump([fout, feq, obstacle], file)
    print("{}/{} files".format(7 + i * 8, maxIter * 8))

    np_bounce_back(fout, fin, obstacle)

    with open(PATH / "streaming-test-{}.pkl".format(8 + i * 8), "wb") as file:
        pickle.dump([fin, fout, v], file)
    print("{}/{} files".format(8 + i * 8, maxIter * 8))

    np_streaming_step(fin, fout, v)
