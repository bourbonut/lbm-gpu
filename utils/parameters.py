import numpy as np

###### Flow definition #################################################
maxIter = 4  # 200000  # Total number of time iterations.
Re = 150.0  # Reynolds number.
nx, ny = 1048, 22 * 512  # Number of lattice nodes.
ly = ny - 1  # Height of the domain in lattice units.
cx, cy, r = nx // 4, ny // 2, ny // 9  # Coordinates of the cylinder.
uLB = 0.04  # Velocity in lattice units.
nulb = uLB * r / Re
# Viscoscity in lattice units.
omega = 1 / (3 * nulb + 0.5)
# Relaxation parameter.

###### Lattice Constants ###############################################
v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]])
t = np.array([1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

col1 = np.array([0, 1, 2])
col2 = np.array([3, 4, 5])
col3 = np.array([6, 7, 8])
