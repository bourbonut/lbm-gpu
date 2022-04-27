from numba import cuda, int64

from utils.numba_kernels import *
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
def test_equilibrium():
    rho, u, v, t = pickle.load(open(PATH / "equilibrium-test.pkl", "rb"))
    feq = cuda.device_array((9, nx, ny))

    d_rho, d_u, d_v, d_t = map(cuda.to_device, (rho, u, v, t))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    equilibrium[blockspergrid, threadsperblock](d_rho, d_u, d_v, d_t, feq, int64(nx), int64(ny))
    a = feq.copy_to_host()
    b = np_equilibrium(rho, u, v, t)
    return [np.linalg.norm(a - b)]


@testing(name="Outflow")
def test_outflow():
    fin, col3 = pickle.load(open(PATH / "outflow-test.pkl", "rb"))

    d_fin = cuda.to_device(fin)

    threadsperblock, blockspergrid = dispatch1D(ny)
    outflow[blockspergrid, threadsperblock](d_fin, int64(nx), int64(ny))
    a = d_fin.copy_to_host()
    np_outflow(fin, col3, nx)
    b = fin
    return [np.linalg.norm(a - b)]


@testing(name="Macroscopic")
def test_macroscopic():
    fin, v = pickle.load(open(PATH / "macroscopic-test.pkl", "rb"))
    u = cuda.device_array((2, nx, ny))
    rho = cuda.device_array((nx, ny))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    d_fin, d_v = map(cuda.to_device, (fin, v))
    macroscopic[blockspergrid, threadsperblock](d_fin, d_v, rho, u, int64(nx), int64(ny))
    a1 = rho.copy_to_host()
    a2 = u.copy_to_host()
    b1, b2 = np_macroscopic(fin, v)
    return [np.linalg.norm(a1 - b1), np.linalg.norm(a2 - b2)]


@testing(name="Inflow")
def test_inflow():
    u, vel, rho, fin, col2, col3 = pickle.load(open(PATH / "inflow-test.pkl", "rb"))

    threadsperblock, blockspergrid = dispatch1D(ny)
    d_u, d_vel, d_rho, d_fin = map(cuda.to_device, (u, vel, rho, fin))
    inflow[blockspergrid, threadsperblock](d_u, d_vel, d_rho, d_fin, int64(ny))
    a1 = d_rho.copy_to_host()
    a2 = d_u.copy_to_host()
    np_inflow(u, vel, rho, fin, col2, col3)
    b1, b2 = rho, u
    return [np.linalg.norm(a1 - b1), np.linalg.norm(a2 - b2)]


@testing(name="Update fin")
def test_updatefin():
    fin, feq = pickle.load(open(PATH / "updatefin-test.pkl", "rb"))

    d_fin, d_feq = map(cuda.to_device, (fin, feq))

    threadsperblock, blockspergrid = dispatch1D(ny)
    update_fin[blockspergrid, threadsperblock](d_fin, d_feq, int64(ny))
    a = d_fin.copy_to_host()
    np_update_fin(fin, feq)
    b = fin
    return [np.linalg.norm(a - b)]


@testing(name="Collision")
def test_collision():
    fin, feq, omega = pickle.load(open(PATH / "collision-test.pkl", "rb"))
    fout = cuda.device_array((9, nx, ny))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    omega_ = np.full((nx, ny), omega)
    d_fin, d_feq, d_omega = map(cuda.to_device, (fin, feq, omega_))
    collision[blockspergrid, threadsperblock](d_omega, d_fin, d_feq, fout, int64(nx), int64(ny))
    a = fout.copy_to_host()
    b = np_collision(fin, feq, omega)
    return [np.linalg.norm(a - b)]


@testing(name="Bounce back")
def test_bounce_back():
    fout, feq, obstacle = pickle.load(open(PATH / "bounceback-test.pkl", "rb"))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    d_fout, d_feq, d_obstacle = map(cuda.to_device, (fout, feq, obstacle))
    bounce_back[blockspergrid, threadsperblock](d_fout, d_feq, d_obstacle, int64(nx), int64(ny))
    a = d_fout.copy_to_host()
    np_bounce_back(fout, feq, obstacle)
    b = fout
    return [np.linalg.norm(a - b)]


@testing(name="Streaming step")
def test_streaming_step():
    fin, fout, v = pickle.load(open(PATH / "bounceback-test.pkl", "rb"))

    threadsperblock, blockspergrid = dispatch(nx, ny)
    d_fin, d_fout, d_v = map(cuda.to_device, (fin, fout, v))
    streaming_step[blockspergrid, threadsperblock](d_fin, d_fout, d_v, int64(nx), int64(ny))
    a = d_fin.copy_to_host()
    np_streaming_step(fin, fout, v)
    b = fin
    return [np.linalg.norm(a - b)]


test_equilibrium()
test_outflow()
test_macroscopic()
test_inflow()
test_updatefin()
test_collision()
test_bounce_back()
test_streaming_step()
