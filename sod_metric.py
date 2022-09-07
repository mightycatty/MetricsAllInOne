# @Time    : 2022/7/18 4:23 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
'''
reference
[1] https://raw.githubusercontent.com/lartpang/PySODMetrics/main/py_sod_metrics/sod_metrics.py
'''
import os
import cv2
import torch
from torch.nn import functional as nnF

try:
    from ..engine.utils.distributed import reduce_tensor
except:
    from semseg.engine.utils.distributed import reduce_tensor

# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

_EPS = np.spacing(1)  # the different implementation of epsilon (extreme min value) between numpy and matlab
_TYPE = np.float64


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    A numpy-based function for preparing ``pred`` and ``gt``.

    - for ``pred``, it looks like ``mapminmax(im2double(...))`` of matlab;
    - ``gt`` will be binarized by 128.

    :param pred: prediction
    :param gt: mask
    :return: pred, gt
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.

    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)


class Fmeasure(object):
    def __init__(self, beta: float = 0.3):
        """
        F-measure for SOD.

        ::

            @inproceedings{Fmeasure,
                title={Frequency-tuned salient region detection},
                author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
                booktitle=CVPR,
                number={CONF},
                pages={1597--1604},
                year={2009}
            }

        :param beta: the weight of the precision
        """
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the adaptive F-measure.

        :return: adaptive_fm
        """
        # ``np.count_nonzero`` is faster and better
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        """
        Calculate the corresponding precision and recall when the threshold changes from 0 to 255.

        These precisions and recalls can be used to obtain the mean F-measure, maximum F-measure,
        precision-recall curve and F-measure-threshold curve.

        For convenience, ``changeable_fms`` is provided here, which can be used directly to obtain
        the mean F-measure, maximum F-measure and F-measure-threshold curve.

        :return: precisions, recalls, changeable_fms
        """
        # 1. 获取预测结果在真值前背景区域中的直方图
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)  # 最后一个bin为[255, 256]
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量, 这里仅计算了前景区域
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        # 3. 使用不同阈值的结果计算对应的precision和recall
        # p和r的计算的真值是pred==1&gt==1，二者仅有分母不同，分母前者是pred==1，后者是gt==1
        # 为了同时计算不同阈值的结果，这里使用hsitogram&flip&cumsum 获得了不同各自的前景像素数量
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        # 为防止除0，这里针对除0的情况分析后直接对于0分母设为1，因为此时分子必为0
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)
        # TODO: T=0 或者 特定阈值下fg_w_thrs=0或者bg_w_thrs=0，这些都会包含在TPs[i]=0的情况中，
        #  但是这里使用TPs不便于处理列表
        precisions = TPs / Ps
        recalls = TPs / T

        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        """
        Return the results about F-measure.

        :return: dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))
        """
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=_TYPE), axis=0)  # N, 256
        recall = np.mean(np.array(self.recalls, dtype=_TYPE), axis=0)  # N, 256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))


class MAE(object):
    def __init__(self):
        """
        MAE(mean absolute error) for SOD.

        ::

            @inproceedings{MAE,
                title={Saliency filters: Contrast based filtering for salient region detection},
                author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
                booktitle=CVPR,
                pages={733--740},
                year={2012}
            }
        """
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the mean absolute error.

        :return: mae
        """
        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        """
        Return the results about MAE.

        :return: dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=mae)


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        """
        S-measure(Structure-measure) of SOD.

        ::

            @inproceedings{Smeasure,
                title={Structure-measure: A new way to eval foreground maps},
                author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
                booktitle=ICCV,
                pages={4548--4557},
                year={2017}
            }

        :param alpha: the weight for balancing the object score and the region score
        """
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the S-measure.

        :return: s-measure
        """
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the region score.
        """
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.

        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the ssim score.
        """
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        """
        Return the results about S-measure.

        :return: dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)


class Emeasure(object):
    def __init__(self):
        """
        E-measure(Enhanced-alignment Measure) for SOD.

        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38

        ::

            @inproceedings{Emeasure,
                title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
                author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
                booktitle=IJCAI,
                pages="698--704",
                year={2018}
            }
        """
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the adaptive E-measure.

        :return: adaptive_em
        """
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the changeable E-measure, which can be used to obtain the mean E-measure,
        the maximum E-measure and the E-measure-threshold curve.

        :return: changeable_ems
        """
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        """
        Calculate the E-measure corresponding to the specific threshold.

        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``

        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = (
                        2
                        * (combination[0] * combination[1])
                        / (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the E-measure corresponding to the threshold that varies from 0 to 255..

        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``

        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = (
                        2
                        * (combination[0] * combination[1])
                        / (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(
            self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
    ):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        """
        Return the results about E-measure.

        :return: dict(em=dict(adp=adaptive_em, curve=changeable_em))
        """
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFmeasure(object):
    def __init__(self, beta: float = 1):
        """
        Weighted F-measure for SOD.

        ::

            @inproceedings{wFmeasure,
                title={How to eval foreground maps?},
                author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
                booktitle=CVPR,
                pages={248--255},
                year={2014}
            }

        :param beta: the weight of the precision
        """
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the weighted F-measure.
        """
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        # 注意这里使用mask索引矩阵的时候不可使用Ew[gt]，这实际上仅在索引Ew的0维度
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        """
        Return the results about weighted F-measure.

        :return: dict(wfm=weighted_fm)
        """
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=weighted_fm)


class SODMetrics:
    def __init__(self, is_distributed=False):
        self._is_distributed = is_distributed
        self._norm_size = (256, 256)
        self.reset()

    def reset(self):
        self.mae_list = []
        self.fm_list = []
        self.fa_list = []
        self.s_list = []

    def get_scores(self):
        return {
            'MAE': np.mean(self.mae_list),
            'MaxF': np.mean(self.fm_list),
            'AvgF': np.mean(self.fa_list),
            'Sm': np.mean(self.s_list),
        }

    def update(self, pred, mask):
        pred = self.input_format(pred)
        mask = self.input_format(mask)
        mae, max_f, avg_f, s_score = self.cal_total_metrics(pred, mask)
        if self._is_distributed:
            mae = reduce_tensor(mae).item()
            max_f = reduce_tensor(max_f, 'max').item()
            avg_f = reduce_tensor(avg_f).item()
            s_score = reduce_tensor(s_score).item()
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


class SODMetricsPy:
    def __init__(self, norm_size=None):
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

        sequential_results = {
            "fm": np.flip(fm["curve"]),
            "em": np.flip(em["curve"]),
            "p": np.flip(pr["p"]),
            "r": np.flip(pr["r"]),
        }
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


def folder_test(pred_f, gt_f=None):
    from tqdm import tqdm
    sod_m = SODMetrics()
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
    # print(sod_m.update(pred, gt))


if __name__ == '__main__':
    pred_f = r'/Users/shuai.he/Data/segmentation/object/val/pm/mask_object_resample_0907_True_Wed_Sep__7_10-15-19_2022'
    folder_test(pred_f)
"""
mask_0810
{'MAE': 0.07104788, 'MaxF': 0.86587447, 'AvgF': 0.86108404, 'Sm': 0.86495245}
0818_pixel
{'MAE': 0.06277633, 'MaxF': 0.8724629, 'AvgF': 0.86452025, 'Sm': 0.87220806}
0818_item
{'MAE': 0.06389283, 'MaxF': 0.88162255, 'AvgF': 0.8736997, 'Sm': 0.8783835}
0902
{'MAE': 0.049846523, 'MaxF': 0.9109147, 'AvgF': 0.9031521, 'Sm': 0.89538497}
"""
