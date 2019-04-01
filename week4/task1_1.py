import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import random

from block_matching import BlockedImage


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


if __name__ == "__main__":
    from utils.optical_flow import (
    plot_optical_flow_raw,
    plot_optical_flow_colours,
    save_flow_image,
    flow_to_image,
    write_flow)

    past_path = "../data/seq45/000045_10.png"
    curr_path = "../data/seq45/000045_11.png"
    past_image = cv2.imread(past_path)
    curr_image = cv2.imread(curr_path)

    config = dict(
        block_size=20,
        search_area_radius=30,
        search_step=5,
        dist_error_method='MSD',
        scan_method='centered',
    )

    past = BlockedImage(past_image, 5, config['scan_method'])
    curr = BlockedImage(curr_image, 5, config['scan_method'])

    # FORWARD COMPENSATION
    of = forward_compensated_optical_flow(
        past_image,
        curr_image,
        **config,
    )
    plot_optical_flow_raw(curr_image, of, 10)
    write_flow(of, 'fw.png')

    # BACKWARD COMPENSATION
    of = backward_compensated_optical_flow(
        past_image,
        curr_image,
        **config,
    )
    plot_optical_flow_raw(curr_image, of, 10)
    write_flow(of, 'bw.png')
