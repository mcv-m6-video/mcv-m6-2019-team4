# Based on the original Matlab code 'flow_read.m' included in the devkit for KITTI 2012/2015
# Source code can be downloaded from: http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow
# All rights belong to the original authors, we only provide a Python version for those that do not have/want Matlab
import cv2
import numpy as np

def read_flow(filename):
    # Loads flow field F from png file (for encoding details see original code/readme.txt)
    I = cv2.imread(filename)
	I_d = np.double(I)
	height, width, channels = I_d.shape
	# Read horizontal flow field (F_u) and convert it to floating point values
	F_u = (I_d[:,:,0] - 2 ** 15) / 64
	# Do the same for the vertical flow field (F_v)
	F_v = (I_d[:,:,1] - 2 ** 15) / 64
	# Read validity mask (third channel) and ensure we only have 0s and 1s
	F_valid = np.minimum(I_d[:,:,2], np.ones((height, width)))

    # Set invalid flow estimates (F_u, F_v) to 0
    F_u[F_valid == 0] = 0.0
	F_v[F_valid == 0] = 0.0

	# Compose 3-channel output containing the KITTI-formatted flow
	F = np.zeros(I_d.shape)
    F[:,:,0] = F_u
	F[:,:,1] = F_v
	F[:,:,2] = F_valid

    return F

