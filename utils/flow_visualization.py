#!/usr/env/python3

import numpy as np
import matplotlib.pyplot as plt
import flow_utils


def plot_optical_flow(image_file, flow_file):
    flow = flow_utils.read_kitti_file(flow_file)

    (h, w) = flow.shape[0:2]
    du = flow[:, :, 0]
    dv = flow[:, :, 1]
    valid = flow[:, :, 2]
    U = du * valid
    V = dv * valid
    M = np.hypot(U, V)

    X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    step = 10
    plt.figure()
    # plt.title("pivot='mid'; every third arrow; units='inches'")

    im = plt.imread(image_file)
    plt.imshow(im, cmap='gray')

    plt.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        M[::step, ::step],
        pivot='tail',
        units='xy',
        color='r',
        angles='xy',
        scale_units='xy',
        scale=.7
    )
    # plt.scatter(X[::step, ::step], Y[::step, ::step], color='r', s=2)
    plt.show()
