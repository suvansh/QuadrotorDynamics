import numpy as np
from utils import rk4


def step(quad, x, u, h, z=None):
    if z is None:
        xn = rk4(x, lambda _x: quad.dynamics(_x, u)[0], h)
        zn = None
    else:
        xzn = rk4(hstack((x, z)), lambda xz: hstack(quad.dynamics(xz[:quad.xdim], u, xz[quad.xdim:])), h)
        xn, zn = torch.split(xzn, [quad.xdim, quad.zdim])
    return xn, zn


def simulate(quad, x0, us, time, h, z0=None):
    t = i = 0
    xs = [x0]
    zs = [z0]
    for i, t in enumerate(np.arange(0, time + 1e-7, h)):
        xn, zn = step(quad, xs[-1], us[i], h, zs[-1])
        xs.append(xn)
        zs.append(zn)
    return xs, zs
