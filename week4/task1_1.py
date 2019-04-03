from pathlib import Path

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import random

from block_matching import BlockedImage
from utils import optical_flow as utils_of
from evaluation import optical_flow as eval_of
from paths import DATA_DIR


def show_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()


def forward_compensated_image(past_image, curr_image, block_size,
                              search_area_radius, search_step,
                              dist_error_method, scan_method):
    past = BlockedImage(past_image, block_size, scan_method)
    curr = BlockedImage(curr_image, block_size, scan_method)

    for past_block in past.getBlocks():
        curr_block = curr.blockMatch(past_block, search_area_radius,
                                     search_step, dist_error_method)
        # print(f"Current block at ({curr_block.x},{curr_block.y}) "
        #     f"corresponds to block in past ({past_block.x},{past_block.y})")
        past_block.x = curr_block.x
        past_block.y = curr_block.y
        past.setBlock(past_block)

    return past.paintBlocks()


def backward_compensated_image(past_image, curr_image, block_size,
                               search_area_radius, search_step,
                               dist_error_method, scan_method):
    past = BlockedImage(past_image, block_size, scan_method)
    curr = BlockedImage(curr_image, block_size, scan_method)

    for curr_block in curr.getBlocks():
        past_block = past.blockMatch(curr_block, search_area_radius,
                                     search_step, dist_error_method)
        # print(f"Current block at ({curr_block.x},{curr_block.y}) "
        #     f"corresponds to block in past ({past_block.x},{past_block.y})")
        curr_block.image = past_block.image
        curr.setBlock(curr_block)

    return curr.paintBlocks()


def forward_compensated_optical_flow(past_image, curr_image, block_size,
                                     search_area_radius, search_step,
                                     dist_error_method, scan_method):
    # 3 channels: du, dv and valid
    opt_flow = np.zeros((past_image.shape[0], past_image.shape[1], 3))
    past = BlockedImage(past_image, block_size, scan_method)
    curr = BlockedImage(curr_image, block_size, scan_method)

    for past_block in past.getBlocks():
        curr_block = curr.blockMatch(past_block, search_area_radius,
                                     search_step, dist_error_method)
        dv = curr_block.x - past_block.x
        du = curr_block.y - past_block.y
        valid = 1

        if opt_flow[curr_block.y:curr_block.y + block_size,
           curr_block.x:curr_block.x + block_size].shape == (
            block_size, block_size, 3):
            opt_flow[curr_block.y:curr_block.y + block_size,
            curr_block.x:curr_block.x + block_size] = \
                np.full((block_size, block_size, 3), (du, dv, valid))

    return opt_flow


def backward_compensated_optical_flow(past_image, curr_image, block_size,
                                      search_area_radius, search_step,
                                      dist_error_method, scan_method):
    # 3 channels: du, dv and valid
    opt_flow = np.zeros((past_image.shape[0], past_image.shape[1], 3))
    past = BlockedImage(past_image, block_size, scan_method)
    curr = BlockedImage(curr_image, block_size, scan_method)
    for curr_block in curr.getBlocks():
        past_block = past.blockMatch(curr_block, search_area_radius,
                                     search_step, dist_error_method)

        dv = past_block.x - curr_block.x
        du = past_block.y - curr_block.y
        valid = 1

        if opt_flow[curr_block.y:curr_block.y + block_size,
           curr_block.x:curr_block.x + block_size].shape == (
            block_size, block_size, 3):
            opt_flow[curr_block.y:curr_block.y + block_size,
            curr_block.x:curr_block.x + block_size] = \
                np.full((block_size, block_size, 3), (du, dv, valid))

    return opt_flow


def format_filename(prefix: str, config: dict, extension: str) -> str:
    return (
        f"{prefix}_"
        f"{config['search_area_radius']}_"
        f"{config['block_size']}_"
        f"{config['search_step']}_"
        f"{config['dist_error_method']}_"
        f"{config['scan_method']}.{extension}"
    )


def compute_optical_flow(current, past, config):
    """ Computes Forward and Backward optical flow

    Args:
        current: current frame CV2.Image
        past: past frame CV2.Image
    """
    return (
        forward_compensated_optical_flow(past, current, **config),
        backward_compensated_optical_flow(past, current, **config),
    )


def evaluate_optical_flow(gt: np.array, pred: np.array):
    """ Compute metrics for non-occluded pixels.

    Returns:
        MSEN, PEPN
    """
    gt_value = 0  # reject 0's in validity masks

    tu: np.ndarray = gt[:, :, 0]
    tv = gt[:, :, 1]
    mask = gt[:, :, 2].astype(np.uint64)
    u = pred[:, :, 0]
    v = pred[:, :, 1]

    # print(f'maxmin: {max(tu.flatten()), min(tu.flatten())}')
    # print(f'maxmin: {max(u.flatten()), min(u.flatten())}')
    # print(f'maxmin: {max(tv.flatten()), min(tv.flatten())}')
    # print(f'maxmin: {max(v.flatten()), min(v.flatten())}')

    msen = eval_of.flow_error(tu, tv, u, v, mask, gt_value, method='MSEN')
    pepn = eval_of.flow_error(tu, tv, u, v, mask, gt_value, method='PEPN')

    return msen, pepn


if __name__ == "__main__":
    import time
    from utils import optical_flow

    # Convert GT to image for visualization(
    gt_path = "../data/seq45/gt/noc/000045_10.png"
    gt = optical_flow.read_flow(gt_path)
    gt = gt.transpose((1, 0, 2))
    optical_flow.save_flow_image(gt, 'gt.png')

    past_path = '../data/seq45/000045_10.png'
    curr_path = '../data/seq45/000045_11.png'
    past_image = cv2.imread(past_path)
    curr_image = cv2.imread(curr_path)

    config = dict(
        search_area_radius=30,
        block_size=20,
        search_step=5,
        dist_error_method='MSD',
        scan_method='centered',
    )

    config = dict(
        search_area_radius=30,
        block_size=20,
        search_step=None,
        dist_error_method=None,
        scan_method='ncc',
    )

    start = time.clock()
    of_fwd, of_bwd = compute_optical_flow(current=curr_image,
                                          past=past_image,
                                          config=config)
    runtime = time.clock() - start

    msen, pepn = evaluate_optical_flow(gt, of_fwd)

    print(
        f'Computing optical flow...\n'
        f'Config {config}\n'
        f'MSEN: {msen:.2f}\n'
        f'PEPN: {pepn:.2f}\n'
        f'Runtime computing FWD and BWD: {runtime:.3g} seconds'
    )

    msen, pepn = evaluate_optical_flow(gt, of_bwd)

    print(
        f'Computing optical flow...\n'
        f'Config {config}\n'
        f'MSEN: {msen:.2f}\n'
        f'PEPN: {pepn:.2f}\n'
        f'Runtime computing FWD and BWD: {runtime:.3g} seconds'
    )

    optical_flow.save_flow_image(of_fwd, format_filename('fwd', config, 'png'))
    optical_flow.save_flow_image(of_bwd, format_filename('bwd', config, 'png'))

    utils_of.plot_optical_flow_raw(
        curr_image,
        of_fwd,
        10,
        Path(format_filename('img_fwd', config, 'png'))
    )
    utils_of.plot_optical_flow_raw(
        curr_image,
        of_bwd,
        10,
        Path(format_filename('img_bwd', config, 'png'))
    )
    utils_of.plot_optical_flow_raw(curr_image, gt, 10)

    # lk = utils_of.read_flow('../data/seq45/LKflow_000045_10.png')
    # lk = lk.transpose((1, 0, 2))
    # utils_of.plot_optical_flow_raw(curr_image, lk, 1)
