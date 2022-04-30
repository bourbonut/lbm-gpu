import cupy
import numpy

from utils.cupy_kernels import *
from utils.parameters import *
from utils.numpy_functions import *

import pickle
from functools import partial
from pathlib import Path

PATH = Path().absolute() / "tests" / "picklefiles"

# Colors
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def testing(name=""):
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                norms = function(*args, **kwargs)
                if all((n < 1e-10 for n in norms)):
                    print(f"{bcolors.OKGREEN}{name}: Passed !\n{bcolors.ENDC}")
                else:
                    print(bcolors.WARNING + str(norms) + bcolors.ENDC)
                    print(f"{bcolors.FAIL}{name}: >>> Failed ! <<<\n{bcolors.ENDC}")
            except Exception as e:
                print(bcolors.WARNING + str(e) + bcolors.ENDC)
                print(f"{bcolors.FAIL}{name}: >>> Failed ! <<<\n{bcolors.ENDC}")
                raise

        return wrapper

    return decorator


@testing(name="Equilibrium")
def test_equilibrium(i):
    rho, u, v, t = pickle.load(open(PATH / "equilibrium-test-{}.pkl".format(i), "rb"))
    feq = cupy.zeros((9, nx, ny))

    d_rho, d_u, d_v, d_t = map(cupy.array, (rho, u, v, t))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    equilibrium[blockspergrid, threadsperblock](d_rho, d_u, d_v, d_t, feq, nx, ny)
    a = feq.get()
    b = np_equilibrium(rho, u, v, t)
    return [np.linalg.norm(a - b)]


@testing(name="Outflow")
def test_outflow(i):
    fin, col3 = pickle.load(open(PATH / "outflow-test-{}.pkl".format(i), "rb"))

    d_fin = cupy.array(fin)

    threadsperblock, blockspergrid = dispatch1D(ny)
    outflow[blockspergrid, threadsperblock](d_fin, nx, ny)
    a = d_fin.get()
    np_outflow(fin, col3, nx)
    b = fin
    return [np.linalg.norm(a - b)]


@testing(name="Macroscopic")
def test_macroscopic(i):
    fin, v = pickle.load(open(PATH / "macroscopic-test-{}.pkl".format(i), "rb"))
    u = cupy.zeros((2, nx, ny))
    rho = cupy.zeros((nx, ny))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    d_fin, d_v = map(cupy.array, (fin, v))
    macroscopic[blockspergrid, threadsperblock](d_fin, d_v, rho, u, nx, ny)
    a1 = rho.get()
    a2 = u.get()
    b1, b2 = np_macroscopic(fin, v)
    return [np.linalg.norm(a1 - b1), np.linalg.norm(a2 - b2)]


@testing(name="Inflow")
def test_inflow(i):
    u, vel, rho, fin, col2, col3 = pickle.load(
        open(PATH / "inflow-test-{}.pkl".format(i), "rb")
    )

    threadsperblock, blockspergrid = dispatch1D(ny)
    d_u, d_vel, d_rho, d_fin = map(cupy.array, (u, vel, rho, fin))
    inflow[blockspergrid, threadsperblock](d_u, d_vel, d_rho, d_fin, ny)
    a1 = d_rho.get()
    a2 = d_u.get()
    np_inflow(u, vel, rho, fin, col2, col3)
    b1, b2 = rho, u
    return [np.linalg.norm(a1 - b1), np.linalg.norm(a2 - b2)]


@testing(name="Update fin")
def test_updatefin(i):
    fin, feq = pickle.load(open(PATH / "updatefin-test-{}.pkl".format(i), "rb"))

    d_fin, d_feq = map(cupy.array, (fin, feq))

    threadsperblock, blockspergrid = dispatch1D(ny)
    update_fin[blockspergrid, threadsperblock](d_fin, d_feq, ny)
    a = d_fin.get()
    np_update_fin(fin, feq)
    b = fin
    return [np.linalg.norm(a - b)]


@testing(name="Collision")
def test_collision(i):
    fin, feq, omega = pickle.load(open(PATH / "collision-test-{}.pkl".format(i), "rb"))
    fout = cupy.zeros((9, nx, ny))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    omega_ = np.full((nx, ny), omega)
    d_fin, d_feq, d_omega = map(cupy.array, (fin, feq, omega_))
    collision[blockspergrid, threadsperblock](d_omega, d_fin, d_feq, fout, nx, ny)
    a = fout.get()
    b = np_collision(fin, feq, omega)
    return [np.linalg.norm(a - b)]


@testing(name="Bounce back")
def test_bounce_back(i):
    fout, feq, obstacle = pickle.load(open(PATH / "bounceback-test-{}.pkl".format(i), "rb"))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    d_fout, d_feq, d_obstacle = map(cupy.array, (fout, feq, obstacle))
    bounce_back[blockspergrid, threadsperblock](d_fout, d_feq, d_obstacle, nx, ny)
    a = d_fout.get()
    np_bounce_back(fout, feq, obstacle)
    b = fout
    return [np.linalg.norm(a - b)]


@testing(name="Streaming step")
def test_streaming_step(i):
    fin, fout, v = pickle.load(open(PATH / "streaming-test-{}.pkl".format(i), "rb"))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    d_fin, d_fout, d_v = map(cupy.array, (fin, fout, v))
    streaming_step[blockspergrid, threadsperblock](d_fin, d_fout, d_v, nx, ny)
    a = d_fin.get()
    np_streaming_step(fin, fout, v)
    b = fin
    return [np.linalg.norm(a - b)]


maxIter = 10
for i in range(maxIter):
    print("Step :", i)
    test_equilibrium(1 + i * 8)
    test_outflow(2 + i * 8)
    test_macroscopic(3 + i * 8)
    test_inflow(4 + i * 8)
    test_updatefin(5 + i * 8)
    test_collision(6 + i * 8)
    test_bounce_back(7 + i * 8)
    test_streaming_step(8 + i * 8)
