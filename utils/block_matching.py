import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import random


class Block:
    # Block of an image
    #   x and y are the coordinates of the top left of the block
    #       on the original image
    #   block_size is the side of the block in pixels

    def __init__(self, block_size, x, y, image):
        self.image = image
        self.block_size = block_size
        self.x = x
        self.y = y

    def error(self, other, method):
        # measures the error distance between two blocks (this and other)
        # by several methods

        if self.image.shape != other.image.shape:
            return np.Infinity

        if method == "MSD":
            dist = np.mean((self.image - other.image) ** 2)

        return dist


class BlockedImage:
    # Representation of an image divided into blocks of side block_size
    #
    # Internally it is composed of source and destination images
    # Source image is the original one provided at initialization.
    # src_blocks is a list of all the blocks of side block_size of the
    # source image. Blocks can be retrived by getBlocks()
    #
    # Destination image is initially black and the same size as source image.
    # dst_blocks is a list of blocks. setBlock(block) can be used to add
    # blocks to dst blocks. paintBlocks() gets each block in dst_blocks and
    # and paints it at its x and y position, returning the resulting image.
    #
    # blockMatch() is provided a block and window radius. A window is defined
    # around the center position of the block with given radius. Inside that
    # window a block is searched by sliding windows algorightm (given step)
    # that minimizes a given error distance function between sliding window and
    # given block.

    def __init__(self, image, block_size):
        self.src_image = image
        self.dst_image = np.zeros(self.src_image.shape, np.uint8)
        self.block_size = block_size
        self.block_rows = 0

        self.src_blocks = []
        for r in range(0, self.src_image.shape[0] - self.block_size,
                       self.block_size):
            self.block_rows += 1
            for c in range(0, self.src_image.shape[1] - self.block_size,
                           self.block_size):
                self.src_blocks.append(Block(
                    self.block_size,
                    c,
                    r,
                    self.src_image[r:r + self.block_size,
                    c:c + self.block_size],
                ))

        self.block_cols = math.ceil(len(self.src_blocks) / self.block_rows)

        self.dst_blocks = []

    def getBlocks(self):
        return self.src_blocks

    def getBlockRows(self):
        return self.block_rows

    def getBlockCols(self):
        return self.block_cols

    def getBlock(self, row, col):
        # print(len(self.blocks) , ((self.block_cols) * (row-1)) + (col-1) )
        return self.src_blocks[(self.block_cols * (row - 1)) + (col - 1)]

    def setBlock(self, block):
        self.dst_blocks.append(block)

    def clearDstImage(self):
        self.dst_image = np.zeros(self.src_image.shape, np.uint8)
        self.dst_blocks = []

    def paintBlocks(self):
        for b in self.dst_blocks:
            if b.image.shape == self.dst_image[b.y:b.y + b.block_size,
                                b.x:b.x + b.block_size].shape:
                self.dst_image[b.y:b.y + b.block_size,
                b.x:b.x + b.block_size] = b.image

        return self.dst_image

    def src2dst(self):
        self.dst_blocks = self.src_blocks

    def blockMatch(self, block, window_radius, step, dist_method):
        y_block_center = block.y + math.ceil(block.block_size / 2)
        x_block_center = block.x + math.ceil(block.block_size / 2)
        row_start = max(0, y_block_center - window_radius)
        row_end = y_block_center + window_radius
        col_start = max(0, x_block_center - window_radius)
        col_end = x_block_center + window_radius

        # print(y_block_center, x_block_center, block.y, block.x, row_start,
        #       row_end, col_start, col_end)
        src = self.src_image[row_start:row_end, col_start:col_end]

        cmp_blocks = []
        for r in range(0, src.shape[0], step):
            for c in range(0, src.shape[1], step):
                cmp_blocks.append(Block(block.block_size,
                                        col_start + c, row_start + r,
                                        src[r:r + block.block_size,
                                        c:c + block.block_size]))

        dists = [block.error(other, dist_method) for other in cmp_blocks]
        idx = np.argmin(dists)
        best = cmp_blocks[idx]
        # print(dists[idx])
        return best


def show_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()


def forward_compensated_image(past_image, curr_image, block_size,
                              search_area_radius, search_step,
                              dist_error_method):
    past = BlockedImage(past_image, block_size)
    curr = BlockedImage(curr_image, block_size)
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
                               dist_error_method):
    past = BlockedImage(past_image, block_size)
    curr = BlockedImage(curr_image, block_size)
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
                                     dist_error_method):
    # 3 channels: du, dv and valid
    opt_flow = np.zeros((past_image.shape[0], past_image.shape[1], 3))
    past = BlockedImage(past_image, block_size)
    curr = BlockedImage(curr_image, block_size)
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
                                      dist_error_method):
    # 3 channels: du, dv and valid
    opt_flow = np.zeros((past_image.shape[0], past_image.shape[1], 3))
    past = BlockedImage(past_image, block_size)
    curr = BlockedImage(curr_image, block_size)
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
