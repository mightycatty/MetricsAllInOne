# @Time    : 2022/7/18 4:23 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
'''
requirement:
    pip install pysodmetrics
'''
import os
import cv2
import torch
from torch.nn import functional as nnF
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
import torch.distributed as torch_dist
import numpy as np
from metric_interface import MetricsInterface
__all__ = ['SODMetricsTorch', 'SODMetricsPy']


def _get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()


def _reduce_tensor(inp, op='mean', in_place=False):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = _get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        if in_place:
            reduced_inp = inp
        else:
            reduced_inp = inp.clone()
        if op == 'max':
            torch.distributed.reduce(reduced_inp, dst=0, op=torch.distributed.ReduceOp.MAX)
        else:
            torch.distributed.reduce(reduced_inp, dst=0)
    if op == 'mean':
        return reduced_inp / world_size
    else:
        return reduced_inp


class SODMetricsTorch(MetricsInterface):
    def __init__(self, is_distributed=False, norm_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_distributed = is_distributed
        self._norm_size = norm_size
        self.reset()

    def reset(self):
        self.mae_list = []
        self.fm_list = []
        self.fa_list = []
        self.s_list = []

    def get_scores(self, verbose=False):
        scores = {
            'MAE': np.mean(self.mae_list),
            'MaxF': np.mean(self.fm_list),
            'AvgF': np.mean(self.fa_list),
            'Sm': np.mean(self.s_list),
        }
        if verbose: print(scores)
        return scores

    def update(self, pred, mask):
        pred = self.input_format(pred)
        mask = self.input_format(mask)
        mae, max_f, avg_f, s_score = self.cal_total_metrics(pred, mask)
        if self._is_distributed:
            mae = _reduce_tensor(mae).item()
            max_f = _reduce_tensor(max_f, 'max').item()
            avg_f = _reduce_tensor(avg_f).item()
            s_score = _reduce_tensor(s_score).item()
        self.mae_list.append(mae)
        self.fm_list.append(max_f)
        self.fa_list.append(avg_f)
        self.s_list.append(s_score)
        return mae, max_f, avg_f, s_score

    def input_format(self, mask):
        if isinstance(mask, np.ndarray):
            # format to (n, c, h, w)
            mask = np.squeeze(mask)
            mask = torch.from_numpy(mask)
            mask = torch.unsqueeze(mask, dim=0)
            mask = torch.unsqueeze(mask, dim=0)
            mask = mask.float() / 255.
        if self._norm_size:
            mask = nnF.interpolate(mask, self._norm_size, mode='bilinear')
        return mask

    def cal_total_metrics(self, pred, mask):
        # MAE
        mae = torch.mean(torch.abs(pred - mask))
        # MaxF measure
        beta2 = 0.3
        prec, recall = self._eval_pr(pred, mask, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        max_f = f_score.max()
        # AvgF measure
        avg_f = f_score.mean()
        # S measure
        alpha = 0.5
        y = mask.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            Q = alpha * self._S_object(pred, mask) + (1 - alpha) * self._S_region(pred, mask)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        s_score = Q
        return mae, max_f, avg_f, s_score

    def _eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, mask):
        fg = torch.where(mask == 0, torch.zeros_like(pred), pred)
        bg = torch.where(mask == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, mask)
        o_bg = self._object(bg, 1 - mask)
        u = mask.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, mask):
        temp = pred[mask == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, mask):
        X, Y = self._centroid(mask)
        mask1, mask2, mask3, mask4, w1, w2, w3, w4 = self._divideGT(mask, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, mask1)
        Q2 = self._ssim(p2, mask2)
        Q3 = self._ssim(p3, mask3)
        Q4 = self._ssim(p4, mask4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, mask):
        rows, cols = mask.size()[-2:]
        mask = mask.view(rows, cols)
        if mask.sum() == 0:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
        else:
            total = mask.sum()
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((mask.sum(dim=0) * i).sum() / total)
            Y = torch.round((mask.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, mask, X, Y):
        h, w = mask.size()[-2:]
        area = h * w
        mask = mask.view(h, w)
        LT = mask[:Y, :X]
        RT = mask[:Y, X:w]
        LB = mask[Y:h, :X]
        RB = mask[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, mask):
        mask = mask.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = mask.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((mask - y) * (mask - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (mask - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def test_folder(self, pred_f, gt_f, verbose=False, num_bits=None):
        scores = None
        return scores


class SODMetricsPy(MetricsInterface):
    def __init__(self, norm_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._norm_size = norm_size
        self.mae = MAE()
        self.fm = Fmeasure()
        self.sm = Smeasure()
        self.em = Emeasure()
        self.wfm = WeightedFmeasure()

    def get_scores(self, num_bits=3):
        fm_info = self.fm.get_results()
        fm = fm_info["fm"]
        pr = fm_info["pr"]
        wfm = self.wfm.get_results()["wfm"]
        sm = self.sm.get_results()["sm"]
        em = self.em.get_results()["em"]
        mae = self.mae.get_results()["mae"]

        numerical_results = {
            "SM": sm,
            "MAE": mae,
            "maxE": em["curve"].max(),
            "avgE": em["curve"].mean(),
            "adpE": em["adp"],
            "maxF": fm["curve"].max(),
            "avgF": fm["curve"].mean(),
            "adpF": fm["adp"],
            "wFm": wfm,
        }
        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        return numerical_results

    def update(self, pred, mask):
        assert pred.shape == mask.shape
        assert pred.dtype == np.uint8
        assert mask.dtype == np.uint8

        self.mae.step(pred, mask)
        self.sm.step(pred, mask)
        self.fm.step(pred, mask)
        self.em.step(pred, mask)
        self.wfm.step(pred, mask)
        return

    def test_folder(self, pred_f, gt_f, verbose=False, num_bits=None):
        pred_mask_list = os.listdir(pred_f)
        for item in pred_mask_list:
            if '.png' in item:
                mask_pred_src = os.path.join(pred_f, item)
                mask_gt_src = os.path.join(gt_f, item)
                pred = cv2.imread(mask_pred_src, 0)
                gt = cv2.imread(mask_gt_src, 0)
                self.update(pred, gt)
        scores = self.get_scores(num_bits=num_bits)
        if verbose: print(scores)
        return scores


def folder_test(pred_f, gt_f=None):
    from tqdm import tqdm
    sod_m = SODMetricsTorch()
    if gt_f is None:
        gt_f = r'/Users/shuai.he/Data/segmentation/object/val/pm/mask'
        # gt_f = r'/Users/shuai.he/Data/segmentation/object/val/DUTS-TE/mask'
    item_list = os.listdir(pred_f)
    bar = tqdm(total=len(item_list))
    for item in item_list:
        bar.update()
        if '.png' in item:
            try:
                gt_src = os.path.join(gt_f, item)
                pred_src = os.path.join(pred_f, item)
                gt = cv2.imread(gt_src, 0)
                pred = cv2.imread(pred_src, 0)
                sod_m.update(pred, gt)
            except Exception as e:
                print(e)
                print(item)
    print(sod_m.get_scores())


if __name__ == '__main__':
    gt_f = r'/Users/shuai.he/Data/segmentation/object/val/pm/mask'
    pred_f = r'/Users/shuai.he/Data/segmentation/object/val/pm/mask_0907'
    # print(SODMetricsPy().test_folder(pred_f, gt_f))
    folder_test(pred_f)
