import numpy as np

from utils.parameters import *
from utils.numpy_functions import *

import pickle
from functools import partial
from pathlib import Path

PATH = Path().absolute() / "tests" / "picklefiles"

obstacle = np.fromfunction(partial(np_obstacle_fun, cx=cx, cy=cy, r=r), (nx, ny))
vel = np.fromfunction(partial(np_inivel, ly=ly, uLB=uLB), (2, nx, ny))
rho = np.full((nx, ny), 2)

with open(PATH / "equilibrium-test.pkl", "wb") as file:
    pickle.dump([rho, vel, v, t], file)
print("1/8 files")

fin = np_equilibrium(rho, vel, v, t)

with open(PATH / "outflow-test.pkl", "wb") as file:
    pickle.dump([fin, col3], file)
print("2/8 files")

np_outflow(fin, col3, nx)

with open(PATH / "macroscopic-test.pkl", "wb") as file:
    pickle.dump([fin, v], file)
print("3/8 files")

rho, u = np_macroscopic(fin, v)

with open(PATH / "inflow-test.pkl", "wb") as file:
    pickle.dump([u, vel, rho, fin, col2, col3], file)
print("4/8 files")

np_inflow(u, vel, rho, fin, col2, col3)
feq = np_equilibrium(rho, u, v, t)

with open(PATH / "updatefin-test.pkl", "wb") as file:
    pickle.dump([fin, feq], file)
print("5/8 files")

np_update_fin(fin, feq)

with open(PATH / "collision-test.pkl", "wb") as file:
    pickle.dump([fin, feq, omega], file)
print("6/8 files")

fout = np_collision(fin, feq, omega)

with open(PATH / "bounceback-test.pkl", "wb") as file:
    pickle.dump([fout, feq, obstacle], file)
print("7/8 files")

np_bounce_back(fout, fin, obstacle)

with open(PATH / "streaming-test.pkl", "wb") as file:
    pickle.dump([fin, fout, v], file)
print("8/8 files")

# np_streaming_step(fin, fout, v)
