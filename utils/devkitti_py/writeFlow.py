# Based on the original Matlab code 'flow_write.m' included in the devkit for KITTI 2012/2015
# Source code can be downloaded from: http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow
# All rights belong to the original authors, we only provide a Python version for those that do not have/want Matlab
# Notes: 'shiftdim' in the original implementation is used to ensure that there are no singleton dimensions (i.e.: those with one element). Here we use np.squeeze() which effectively does the same.

import cv2
import numpy as np


def write_flow(F, filename):
	# Write flow field F to 'filename.flo' (for encoding details see original code/readme.txt)
    F = np.double(F)

	# Convert from floating precision back to uint64 and store in a PNG file w. 3 channels
	I[:,:,0] = np.max(np.min(np.squeeze(F[:,:,0]) * 64 + 2 ** 15, 2 ** 16 - 1), 0).astype(np.uint64)

