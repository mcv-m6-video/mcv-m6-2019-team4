import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.colors as cl
from PIL import Image


def plot_optical_flow(image_file, flow_file):
    flow = read_kitti_file(flow_file)

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


"""
Copyright (c) 2019 LI RUOTENG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Modified on March 2019 in the framework of a MsC project. We only take the parts of the original script 'flowlib.py'
 concerning optical flow: read/write + visualization in both, KITTI format ('.png') and Middlebury ('.flo'), but not in 
 '.pfm'. 
For the complete code, please refer to: https://github.com/liruoteng/OpticalFlowToolkit
All rights go to LI RUOTENG.

"""

# Define maximum/minimum flow values to avoid NaN/Inf
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
TAU_MOTION = 3  # error smaller or equal than 3 pixels are not taken into account
VERBOSE = 0  # Do not show informational print's

"""
    Visualization
"""


def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()


def show_flow_pair(filename, filename2):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file 1
    :param filename: optical flow file 2
    :return: None
    """

    flow = read_flow(filename)
    img = flow_to_image(flow)
    flow2 = read_flow(filename2)
    img2 = flow_to_image(flow2)

    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    ax[0, 0] = plt.imshow(img)
    ax[0, 1] = plt.imshow(img2)
    plt.show()


def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()

        return None


"""
    Read/write interfaces (see 'auxiliar_functions' for details)
"""


# Overwrite 'read_png_file', 'save_'


def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    elif filename.endswith('.png'):
        # flow = read_png_file(filename)
        flow = read_kitti_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def save_flow_image(flow, image_file):
    """
    save flow visualization into image file
    :param flow: optical flow data
    :param image_file: image destination
    :return: None
    """
    flow_img = flow_to_image(flow)
    img_out = Image.fromarray(flow_img)
    img_out.save(image_file)


def flowfile_to_imagefile(flow_file, image_file):
    """
    convert flowfile into image file
    :param flow_file: optical flow file path
    :param image_file: image destination
    :return: None
    """
    flow = read_flow(flow_file)
    save_flow_image(flow, image_file)


"""
    (brief explanation added on March '19): returns a image with labels from 0 to 8 detailing to
    which class (depending on the flow orientation) a pixel belongs
"""


def segment_flow(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    u[idx2] = 0.00001
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg


# Useful for a more 'canonical' flow representation (and easier to interpret)
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


# Auxiliar functions to read/write flow files and visualizations in PNG format
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(
        np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(
        np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(
        np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(
        np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def read_flo_file(flow_file):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(flow_file, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if VERBOSE:
            print("Reading %d x %d flow file in .flo format" % (h, w))

        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


# def read_png_file(flow_file):
#     """
#     Read from KITTI .png file
#     :param flow_file: name of the flow file
#     :return: optical flow data in matrix
#     March 2019: we need the data as w x h x channels
#     So we transpose each element of flow at the end
#     """
#     flow_object = png.Reader(filename=flow_file)
#     flow_direct = flow_object.asDirect()
#     flow_data = list(flow_direct[2])
#     (w, h) = flow_direct[3]['size']
#     if VERBOSE:
#         print("Reading %d x %d flow file in .png format" % (h, w))
#
#     flow = np.zeros((h, w, 3), dtype=np.float64)
#     for i in range(len(flow_data)):
#         flow[i, :, 0] = flow_data[i][0::3]
#         flow[i, :, 1] = flow_data[i][1::3]
#         flow[i, :, 2] = flow_data[i][2::3]
#
#     invalid_idx = (flow[:, :, 2] == 0)
#     flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
#     flow[invalid_idx, 0] = 0
#     flow[invalid_idx, 1] = 0
#
#     # Transpose to have (width, height, channels)
#     flow[:, :, 0] = np.transpose(flow[:, :, 0])
#     flow[:, :, 1] = np.transpose(flow[:, :, 1])
#     flow[:, :, 2] = np.transpose(flow[:, :, 2])
#
#     return flow


def read_kitti_file(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix (hor_field, vert_field, val_mask)
    adapted from original devkit_kitti MATLAB code

    """
    flow_kitti = cv2.imread(flow_file, -1)  # Read as R, G, B (instead of BGR)

    # Convert to floating point
    u_gt = (flow_kitti[:, :, 2] - 2. ** 15) / 64
    v_gt = (flow_kitti[:, :, 1] - 2. ** 15) / 64
    # Read validity mask
    valid_gt = flow_kitti[:, :, 0]
    # Transpose array to match reference MATLAB code
    F_kitti = np.transpose(np.array([u_gt, v_gt, valid_gt]))

    return F_kitti


# From user soply on githubgist (https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1)
def show_images(images, cols=1, titles=None, cmap='gray', scaling=1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in
                                 range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2 and cmap == 'gray':
            plt.gray()

        plt.imshow(image, cmap=cmap)
        a.set_title(title)
    fig.set_size_inches(scaling * np.array(fig.get_size_inches()) * n_images)

    plt.show()

# TODO: add tests for visualization tools used
