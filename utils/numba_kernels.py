from numba import cuda, float64
from math import log, floor, ceil

SM = 22


def dispatch(m, n):
    r = min(floor(log((m * n) // SM, 2)), 10)
    inf, sup = (m, n) if m < n else (n, m)
    c = log(sup, 2) - log(inf, 2)
    a, b = min(10, int(floor(r / 2 - c))), min(10, int(ceil(r / 2 + c)))
    tx, ty = (2 ** a, 2 ** b) if m < n else (2 ** b, 2 ** a)
    blockspergrid_x = int(m // tx + bool(m % tx))
    blockspergrid_y = int(n // ty + bool(n % ty))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    threadsperblock = (int(max(1, tx)), int(max(1, ty)))
    return threadsperblock, blockspergrid


def dispatch1D(n):
    t = threadsperblock = int(2 ** min(10, floor(log(max(1, n // 216), 2))))
    blockspergrid = int(n // t + bool(n % t))
    return threadsperblock, blockspergrid


@cuda.jit
def equilibrium(rho, u, v, t, feq, nx, ny):
    cv = cuda.const.array_like(v)
    ct = cuda.const.array_like(t)
    row, col = cuda.grid(2)
    if row < nx and col < ny:
        vx = u[0, row, col]
        vy = u[1, row, col]
        usqr = 1.5 * (vx * vx + vy * vy)
        for i in range(9):
            cu = 3 * (cv[i, 0] * vx + cv[i, 1] * vy)
            feq[i, row, col] = rho[row, col] * ct[i] * (1 + cu + 0.5 * cu * cu - usqr)


@cuda.jit
def macroscopic(fin, v, rho, u, nx, ny):
    cv = cuda.const.array_like(v)
    row, col = cuda.grid(2)
    if row < nx and col < ny:
        trho = float64(0.0)
        tu0 = float64(0.0)
        tu1 = float64(0.0)
        for i in range(9):
            fvalue = fin[i, row, col]
            trho += fvalue
            tu0 += cv[i, 0] * fvalue
            tu1 += cv[i, 1] * fvalue

        rho[row, col] = trho
        u[0, row, col] = tu0 / trho
        u[1, row, col] = tu1 / trho


@cuda.jit
def outflow(fin, nx, ny):
    col = cuda.grid(1)
    if col < ny:
        for i in range(3):
            fin[6 + i, nx - 1, col] = fin[6 + i, nx - 2, col]


@cuda.jit
def inflow(u, vel, rho, fin, ny):
    col = cuda.grid(1)
    if col < ny:
        u[0, 0, col] = vel[0, 0, col]
        u[1, 0, col] = vel[1, 0, col]
        t2 = fin[3, 0, col] + fin[4, 0, col] + fin[5, 0, col]
        t3 = fin[6, 0, col] + fin[7, 0, col] + fin[8, 0, col]
        rho[0, col] = (t2 + 2 * t3) / (1 - u[0, 0, col])


@cuda.jit
def update_fin(fin, feq, ny):
    col = cuda.grid(1)
    if col < ny:
        for i in range(3):
            fin[i, 0, col] = feq[i, 0, col] + fin[8 - i, 0, col] - feq[8 - i, 0, col]


@cuda.jit
def collision(omega, fin, feq, fout, nx, ny):
    row, col = cuda.grid(2)
    comega = cuda.const.array_like(omega)
    if row < nx and col < ny:
        for i in range(9):
            vomega = comega[row, col]
            fout[i, row, col] = (1 - vomega) * fin[i, row, col] + vomega * feq[i, row, col]


@cuda.jit
def bounce_back(fout, fin, obstacle, nx, ny):
    row, col = cuda.grid(2)
    if row < nx and col < ny:
        if obstacle[row, col]:
            for i in range(9):
                fout[i, row, col] = fin[8 - i, row, col]


@cuda.jit
def streaming_step(fin, fout, v, nx, ny):
    row, col = cuda.grid(2)
    if row < nx and col < ny:
        for k in range(9):
            i = row + v[k, 0]
            j = col + v[k, 1]
            if i == nx:
                i = 0
            elif i == -1:
                i = nx - 1
            if j == ny:
                j = 0
            elif j == -1:
                j = ny - 1
            fin[k, i, j] = fout[k, row, col]
