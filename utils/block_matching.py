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
            dist = np.Infinity

        elif method == "MSD":
            dist = np.mean((self.image - other.image) ** 2)

        elif method == "NCC":
            res = cv2.matchTemplate(image=self.image,
                                    templ=other.image,
                                    method=cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            dist = 1 - max_val
        else:
            raise NotImplementedError

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

    def __init__(self, image, block_size, method):
        """

        Args:
            method: block match scanning method: {'linear', 'centered'}
        """
        self.src_image = image
        self.dst_image = np.zeros(self.src_image.shape, np.uint8)
        self.block_size = block_size
        self.block_rows = 0

        block_match_options = {
            'linear': self.blockMatchLinear,
            'centered': self.blockMatchCentered,
            'ncc': self.blockMatchNCC,
        }
        self.block_match_method = block_match_options[method]

        self.src_blocks = []
        for r in range(0,
                       self.src_image.shape[0] - self.block_size,
                       self.block_size):
            self.block_rows += 1
            for c in range(0,
                           self.src_image.shape[1] - self.block_size,
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

    def blockMatchNCC(self, block, window_radius, *args):
        """ Block match using normed cross-correlation """
        y_block_center = block.y + math.ceil(block.block_size / 2)
        x_block_center = block.x + math.ceil(block.block_size / 2)

        col_start = max(0, x_block_center - window_radius)
        col_end = x_block_center + window_radius
        row_start = max(0, y_block_center - window_radius)
        row_end = y_block_center + window_radius

        origin = (col_start, row_start)
        search_window = self.src_image[row_start:row_end, col_start:col_end]
        res = cv2.matchTemplate(image=search_window,
                                templ=block.image,
                                method=cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        # dist = 1 - max_val

        top_left = max_loc
        bottom_right = (
            top_left[0] + block.block_size,
            top_left[1] + block.block_size,
        )

        top_left_abs = (
            origin[0] + top_left[0],
            origin[1] + top_left[1],
        )
        bottom_right_abs = (
            origin[0] + bottom_right[0],
            origin[1] + bottom_right[1],
        )

        best_image: np.array = search_window[
                               top_left[1]:bottom_right[1],
                               top_left[0]: bottom_right[0],
                               ]
        # best_imag: np.array = self.src_image[
        #                        top_left_abs[1]:bottom_right_abs[1],
        #                        top_left_abs[0]: bottom_right_abs[0],
        #                        ]
        # assert 0 == np.sum(best_imag.flatten() - best_image.flatten())

        best = Block(
            block_size=block.block_size,
            x=top_left_abs[0],
            y=top_left_abs[1],
            image=best_image,
        )
        return best

    def blockMatchCentered(self, block, window_radius, step, dist_method):
        """ Find best match scanning from center to outside

        Takes into consideration the L1 distance from the source block
        center to priorise matches.
        """
        y_block_center = block.y + math.ceil(block.block_size / 2)
        x_block_center = block.x + math.ceil(block.block_size / 2)
        row_start = max(0, y_block_center - window_radius)
        row_end = y_block_center + window_radius
        col_start = max(0, x_block_center - window_radius)
        col_end = x_block_center + window_radius

        src = self.src_image[row_start:row_end, col_start:col_end]

        cmp_blocks = []
        for r in range(0, src.shape[0], step):
            for c in range(0, src.shape[1], step):
                cmp_blocks.append(Block(
                    block.block_size,
                    col_start + c,
                    row_start + r,
                    src[r:r + block.block_size, c:c + block.block_size]
                ))

        def distance_from_center(block):
            """
            Returns the pixel distance from a block to the source block center
            """
            y_center = block.y + math.ceil(block.block_size / 2)
            x_center = block.x + math.ceil(block.block_size / 2)
            return abs(x_center - x_block_center) + \
                   abs(y_center - y_block_center)

        # Sort blocks by its distance from the original block (increasing)
        cmp_blocks.sort(key=distance_from_center)

        dists = [block.error(other, dist_method) for other in cmp_blocks]
        idx = np.argmin(dists)
        best = cmp_blocks[idx]
        return best

    def blockMatchLinear(self, block, window_radius, step, dist_method):
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
                cmp_blocks.append(Block(
                    block.block_size,
                    col_start + c,
                    row_start + r,
                    src[r:r + block.block_size, c:c + block.block_size]
                ))

        dists = [block.error(other, dist_method) for other in cmp_blocks]
        idx = np.argmin(dists)
        best = cmp_blocks[idx]
        # print(dists[idx])
        return best

    def blockMatch(self, block, window_radius, step, dist_method):
        return self.block_match_method(
            block,
            window_radius,
            step,
            dist_method
        )
