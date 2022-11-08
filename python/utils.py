import os
import pandas as pd
import numpy as np
from torch import tensor, vstack, hstack, stack, \
                    zeros, diag, eye, block_diag, \
                    sqrt, dot, median, diff


def hat(vec):
    return tensor([[0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])

""" Quaternion utils """
H = vstack((zeros(1, 3), eye(3)))
T = diag(tensor([1., -1, -1, -1]))

def L(q):
    return vstack((hstack((q[:1], -q[1:])),
            hstack((q[1:,None], q[0]*eye(3) + hat(q[1:])))))

def R(q):
    return vstack((hstack((q[:1], -q[1:])),
            hstack((q[1:,None], q[0]*eye(3) - hat(q[1:])))))

def A(q):
    return H.T @ L(q) @ R(q)' @ H.T

def qtoQ(q):
    return H.T @ T @ L(q) * T @ L(q) @ H

def G(q):
    return L(q) @ H

def rptoq(ϕ):
    return 1 / sqrt(1 + dot(ϕ, ϕ))

def Drptoq(ϕ):
    return 1 / sqrt(1 + dot(ϕ, ϕ)) * H - ((1 + dot(ϕ, ϕ))**-1.5) * vstack((tensor([1]), ϕ[:, None])) * ϕ[None]

def qtorp(q):
    return q[1:] / q[0]

def E(q):
    return block_diag(eye(3), G(q), eye(6))

def rk4(x, f, h):
    """ updates x using f = dx/dt for time h """
    t1 = f(x)
    t2 = f(x + 0.5 * h * t1)
    t3 = f(x + 0.5 * h * t2)
    t4 = f(x + h * t3)
    xnext = x + h/6 * (t1 + 2*t2 + 2*t3 + t4)
    if len(xnext) > 6:
        return torch.hstack((xnext[:3],
                            xnext[3:7] / torch.norm(xnext[3:7]),  # normalize quat
                            xnext[7:]))
    else:
        return xnext

def symmetrize(vec):
    return stack((
        vec[:3],
        tensor([vec[1], vec[3], vec[4]]),
        tensor([vec[2], vec[4], vec[5]])
    ))

def apply_along_axis(function, x, axis=0):
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)

def read_all_neuroBEM(config, save_path=None):
    if save_path is not None and os.path.isfile(save_path):
        # load from path instead of regenerating
        d = torch.load(save_path)
        return d["xs"], d["xdots"], d["us"], d["dt"]
    data_path = config["data_path"]
    xs, ẋs, us, dt = [], [], [], []
    for file in os.listdir(data_path):
        xsi, ẋsi, usi, dti = read_neuroBEM(file)
        xs.append(xsi)
        ẋs.append(ẋsi)
        us.append(usi)
        dt.append(dti)
    xs, ẋs, us, dt = vstack(xs), vstack(ẋs), vstack(us), np.median(dts)
    if save_path is not None:  # # save to path
        torch.save({"xs": xs, "xdots": ẋs, "us": us, "dt": dt}, save_path)
    return xs, ẋs, us, dt

def read_neuroBEM(csv_path):
    arr = tensor(np.genfromtxt(csv_path, delimiter=","))
    dt = median(diff(arr[:, 0]))
    q̇s = apply_along_axis(lambda row: 0.5*R(row[7:11]) * vstack((tensor([0]), row[4:7, None])),
                            axis=1,
                            arr=arr)
    vw = apply_along_axis(lambda row: A(row[7:11]) * row[14:17],
                            axis=1,
                            arr=arr)
    aw = apply_along_axis(lambda row: A(row[7:11]) * row[11:14],
                            axis=1,
                            arr=arr)
    xs = hstack((arr[:, 17:20], arr[:, 7:11], vw, arr[:, 4:7]))
    ẋs = hstack((vw, q̇s, aw, arr[:, 1:4]))
    us = arr[:, 20:24]
    return xs, ẋs, us, dt
