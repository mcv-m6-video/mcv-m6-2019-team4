import numpy as np
import utils.pyflow
import pyflow
import os
import time
import cv2
from paths import DATA_DIR

class OpticalFlowTracker(object):


    def __init__(self, image, roi):
        self.last_image = image
        self.last_roi = roi

        # Flow Options:
        self.alpha = 0.012
        self.ratio = 0.75
        self.minWidth = 20
        self.nOuterFPIterations = 7
        self.nInnerFPIterations = 1
        self.nSORIterations = 30
        self.colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    def predict(self, new_image, obj_id, frame_id, online):


        #im1 = self.last_image[ytl:ybr, xtl:xbr, :]
        #im2 = new_image[ytl:ybr, xtl:xbr, :]
        im1 = self.last_image
        im2 = new_image
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
        of_file = DATA_DIR.joinpath('results/outFlow_frame{0}.npy'.format(frame_id))
        if online:
            if not os.path.isfile(of_file):
                u, v, im2W = pyflow.coarse2fine_flow(
                    im1, im2, self.alpha, self.ratio, self.minWidth, self.nOuterFPIterations, self.nInnerFPIterations,
                    self.nSORIterations, self.colType)
                flow = np.concatenate((u[..., None], v[..., None]), axis=2)
                np.save(of_file, flow)
                plot_flow(flow)
            else:
                flow = np.load(of_file)
        else:
            if os.path.isfile(of_file):
                flow = np.load(of_file)
            else:
                flow = None

        new_center = self.estimate_center(flow)
        return new_center


    def correct(self, new_image, new_roi):
        self.last_image = new_image
        self.last_roi = new_roi


    def estimate_center(self, flow):

        #return [[0],[0]]
        #if flow == None:
        #    return [[0], [0]]
        #u = flow[:, :, 0]
        #v = flow[:, :, 1]
        # TODO: Implement center computation from optical flow

        u = flow[:, :, 0]
        v = flow[:, :, 1]
        alfa = 20
        xtl = self.last_roi.xTopLeft
        xbr = self.last_roi.xBottomRight
        ytl = self.last_roi.yTopLeft
        ybr = self.last_roi.yBottomRight
        xtl = max(int(xtl-alfa*(xbr-xtl)/100),0)
        xbr = min(int(xbr+alfa*(xbr-xtl)/100),u.shape[1]-1)
        ytl = max(int(ytl-alfa*(ybr-ytl)/100),0)
        ybr = min(int(ybr+alfa*(ybr-ytl)/100),u.shape[0]-1)
        u = u[ytl:ybr, xtl:xbr]
        v = v[ytl:ybr, xtl:xbr]
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        plot_flow(flow)
        mag, ang = cv2.cartToPolar(u, v)
        # Otsu's thresholding

        aux = np.array(mag * 255 / mag.max(), np.dtype(np.uint8))
        ret, th = cv2.threshold(aux, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mean_u = np.array(u[aux > ret]).mean()
        mean_v = np.array(v[aux > ret]).mean()

        center = self.last_roi.center()
        #print('old center: {}'.format(center))
        center[0] = center[0] + mean_u
        center[1] = center[1] + mean_v
        #print('new center: {}'.format(center))

        return center

def plot_flow(flow):
    shape = flow.shape
    shape = (shape[0],shape[1],3)
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Image", rgb)
    cv2.waitKey(1)

def run():
    frame_id = 1000;

    of_file = DATA_DIR.joinpath('results/outFlow_frame{0}.npy'.format(frame_id))
    flow = np.load(of_file)
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Otsu's thresholding
    ret, th = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mean_u = np.array(u[mag > th]).mean()
    mean_v = np.array(v[mag > th]).mean()

    return new_center


if __name__ == '__main__':
    run()