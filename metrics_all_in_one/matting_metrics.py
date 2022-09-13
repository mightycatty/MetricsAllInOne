# @Time    : 2022/7/18 7:23 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
"""metrics for matting
Modified from https://github.com/PeterL1n/RobustVideoMatting/blob/17d1774b032fd503bfe53c57d295db719f9e3da1/evaluation/evaluate_lr.py
"""
import numpy as np
import cv2
from metric_interface import MetricsInterface
import os


class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred - true).mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)

    def __call__(self, pred, true):
        pred_normed = np.zeros_like(pred)
        true_normed = np.zeros_like(true)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss / 1000

    def gauss_gradient(self, img):
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x ** 2 + img_filtered_y ** 2)

    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x ** 2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma ** 2


class MetricCONN:
    def __call__(self, pred, true):
        step = 0.1
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(true)
        for i in range(1, len(thresh_steps)):
            true_thresh = true >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(true)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        true_diff = true - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        true_phi = 1 - true_diff * (true_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        connectivity_error = np.sum(np.abs(true_phi - pred_phi))
        return connectivity_error / 1000


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = np.sum(dtSSD) / true_t.size
        dtSSD = np.sqrt(dtSSD)
        return dtSSD * 1e2


class MattingMetricsPy(MetricsInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.mad_list = []
        self.mse_list = []
        self.grad_list = []
        self.conn_list = []
        return

    def get_scores(self, verbose=False):
        scores = {
            'MAD': np.mean(self.mad_list),
            'MSE': np.mean(self.mse_list),
            'GRAD': np.mean(self.grad_list),
            'CONN': np.mean(self.conn_list)
        }
        if verbose: print(scores)
        return scores

    def update(self, pred, mask):
        self.mad_list.append(MetricMAD()(pred, mask))
        self.mse_list.append(MetricMSE()(pred, mask))
        self.grad_list.append(MetricGRAD()(pred, mask))
        self.conn_list.append(MetricCONN()(pred, mask))
        return

    def test_folder(self, pred_f: str, gt_f: str, verbose: bool = False, *args, **kwargs):
        pred_mask_list = os.listdir(pred_f)
        for item in pred_mask_list:
            if '.png' in item:
                mask_pred_src = os.path.join(pred_f, item)
                mask_gt_src = os.path.join(gt_f, item)
                pred = cv2.imread(mask_pred_src, 0)
                gt = cv2.imread(mask_gt_src, 0)
                self.update(pred, gt)
        scores = self.get_scores()
        if verbose: print(scores)
        return scores


if __name__ == '__main__':
    gt_f = r'/Users/shuai.he/Data/segmentation/object/val/pm/mask'
    pred_f = r'/Users/shuai.he/Data/segmentation/object/val/pm/mask_0902'
    print(MattingMetricsPy().test_folder(pred_f, gt_f))
