import cupy as np
import numpy

from utils.parameters import *

v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]])
numpy_v = numpy.array(
    [[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]]
)
t = np.array([1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

col1 = np.array([0, 1, 2])
col2 = np.array([3, 4, 5])
col3 = np.array([6, 7, 8])


def macroscopic(fin):
    """Compute macroscopic variables (density, velocity)
    fluid density is 0th moment of distribution functions
    fluid velocity components are 1st order moments of dist. functions
    """
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u


def equilibrium(rho, u):
    """Equilibrium distribution function."""
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        cu = 3 * (v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * t[i] * (1 + cu + 0.5 * cu ** 2 - usqr)
    return feq


def obstacle_fun(x, y):
    return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2


def inivel(d, x, y):
    return (1.0 - d) * uLB * (1.0 + 1e-4 * numpy.sin(y / ly * 2.0 * numpy.pi))


def main():

    # create obstacle mask array from element-wise function
    obstacle = numpy.fromfunction(obstacle_fun, (nx, ny))

    # initial velocity field vx,vy from element-wise function
    # vel is also used for inflow border condition
    vel = numpy.fromfunction(inivel, (2, nx, ny))

    obstacle, vel = map(np.array, (obstacle, vel))
    # Initialization of the populations at equilibrium
    # with the given velocity.
    fin = equilibrium(1, vel)

    ###### Main time loop ########
    for time in range(maxIter):

        # if time == 1:
        #     start = pf()

        # Right wall: outflow condition.
        # we only need here to specify distrib. function for velocities
        # that enter the domain (other that go out, are set by the streaming step)
        fin[col3, nx - 1, :] = fin[col3, nx - 2, :]

        # Compute macroscopic variables, density and velocity.
        rho, u = macroscopic(fin)

        # Left wall: inflow condition.
        u[:, 0, :] = vel[:, 0, :]
        rho[0, :] = (
            1
            / (1 - u[0, 0, :])
            * (np.sum(fin[col2, 0, :], axis=0) + 2 * np.sum(fin[col3, 0, :], axis=0))
        )

        # Compute equilibrium.
        feq = equilibrium(rho, u)
        fin[[0, 1, 2], 0, :] = (
            feq[[0, 1, 2], 0, :] + fin[[8, 7, 6], 0, :] - feq[[8, 7, 6], 0, :]
        )

        # Collision step.
        fout = fin - omega * (fin - feq)

        # Bounce-back condition for obstacle.
        # in python language, we "slice" fout by obstacle
        for i in range(9):
            fout[i, obstacle] = fin[8 - i, obstacle]

        # Streaming step.
        for i in range(9):
            fin[i, :, :] = np.roll(
                np.roll(fout[i, :, :], numpy_v[i, 0], axis=0), numpy_v[i, 1], axis=1
            )

        # Visualization of the velocity.
        # if time % 100 == 0:
        #     plt.clf()
        #     plt.imshow(np.sqrt(u[0] ** 2 + u[1] ** 2).transpose(), cmap=cm.Reds)
        #     plt.savefig("vel.{0:04d}.png".format(time // 100))
    # print(pf() - start)


if __name__ == "__main__":
    # execute only if run as a script
    main()
