# @Time    : 2021/11/21 6:37 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as nnF
from ai_toolkit.optical_flow.flowlib import FlowToolkit
import cv2
import numpy as np


# from semseg.metrics.cls_metrics import TwocatMetricPth

def dice_coef_np(mask_0, mask_1, smooth=1.):
    mask_0 = cv2.resize(mask_0, (256, 256))
    mask_1 = cv2.resize(mask_1, (256, 256))
    mask_0 = np.float32(mask_0) / (mask_0.max() + 1e-6)
    mask_1 = np.float32(mask_1) / (mask_1.max() + 1e-6)
    return (2. * np.sum(mask_0 * mask_1) + smooth) / \
           (np.sum(np.square(mask_0)) + np.sum(np.square(mask_1)) + smooth)


def l1(mask):
    mask = np.float32(mask) / (mask.max() + 1e-6)
    value = np.sum(mask)
    return value


def bce_tf(x, y, scale_norm=True):
    tf.enable_eager_execution()
    tf.executing_eagerly()
    x = cv2.resize(x, (256, 256))
    y = cv2.resize(y, (256, 256))
    x = np.expand_dims(x, axis=0).astype(np.float32) / 255.
    y = np.expand_dims(y, axis=0).astype(np.float32) / 255.
    union = np.sum(np.bitwise_or(x > 0.5 * x.max(), y > 0.5 * y.max())) + 10.
    x_tensor = tf.convert_to_tensor(x)
    y_tensor = tf.convert_to_tensor(y)
    loss = tf.keras.backend.binary_crossentropy(x_tensor, y_tensor).numpy()
    if scale_norm:
        loss = np.sum(loss) / (union + 1e-6)
    else:
        loss = np.mean(loss)
    return loss


def bce_np(x, y, scale_norm=True):
    x = cv2.resize(x, (256, 256))
    y = cv2.resize(y, (256, 256))
    x = x = np.float32(x) / ()
    y = np.expand_dims(y, axis=0).astype(np.float32) / 255.
    inter = np.sum(np.bitwise_or(x > 0.5 * x.max(), y > 0.5 * y.max())) + 10.
    x_tensor = tf.convert_to_tensor(x)
    y_tensor = tf.convert_to_tensor(y)
    loss = tf.keras.backend.binary_crossentropy(x_tensor, y_tensor).numpy()
    if scale_norm:
        loss = np.sum(loss) / (inter + 1e-6)
    else:
        loss = np.mean(loss)
    return loss


def mean_iou(y_true, y_pred, norm_size=(256, 256)):
    fg_iou = iou(y_true, y_pred, norm_size=norm_size)
    bg_iou = iou(255 - y_true, 255 - y_pred, norm_size=norm_size)
    miou = (fg_iou + bg_iou) * 0.5
    return miou


def _get_boundary(mask, bandwidth=5, mul=False):
    kernel = np.ones((bandwidth, bandwidth), np.uint8)
    boundary = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
    if mul:
        boundary[boundary > 0.] = 1
        vis = [boundary * 255]
        boundary = boundary * mask
        vis.append(boundary)
        vis = np.hstack(vis)
    # plt.imshow(boundary)
    # plt.show()
    return boundary


def iou(y_true, y_pred, norm_size=(256, 256), binary=False, boundary_only=False, bandwidth=7):
    """
    iou per image
    :param y_true:
    :param y_pred:
    :param norm_size:
    :param binary: cal iou over binary mask
    :return:
    """
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    y_true = cv2.resize(y_true, norm_size)
    y_pred = cv2.resize(y_pred, norm_size)
    if boundary_only:
        y_true = _get_boundary(y_true, bandwidth=bandwidth)
        y_pred = _get_boundary(y_pred, bandwidth=bandwidth)
    y_pred = y_pred / (y_pred.max() + 1e-6)
    y_true = y_true / (y_true.max() + 1e-6)
    if binary:
        y_pred = np.where(y_pred > 0.5, 1., 0)
        y_true = np.where(y_true > 0.5, 1., 0)
    intersection = np.min(np.stack([y_true, y_pred], axis=-1), axis=-1)
    intersection = np.sum(intersection)
    union = np.max(np.stack([y_true, y_pred], axis=-1), axis=-1)
    union = np.sum(union)
    if np.sum(y_true) < 10:
        esp = 400
    else:
        esp = 1e-6
    iou_value = (intersection + esp) / (union + esp)
    return iou_value


def boundary_l1(y_true, y_pred, bandwidth=7, norm_size=(256, 256)):
    y_true = cv2.resize(y_true, norm_size)
    y_pred = cv2.resize(y_pred, norm_size)
    y_true = _get_boundary(y_true, bandwidth=bandwidth)
    y_pred = _get_boundary(y_pred, bandwidth=bandwidth)
    l1 = np.abs(y_true - y_pred)
    l1 = np.mean(l1) / 255.
    return l1


def fpr_tpr(y_true, y_pred, norm_size=(256, 256), binary=True):
    y_true = cv2.resize(y_true, norm_size)
    y_pred = cv2.resize(y_pred, norm_size)
    y_true = np.float32(y_true) / (y_true.max() + 1e-6)
    y_pred = np.float32(y_pred) / (y_pred.max() + 1e-6)
    if binary:
        y_true = np.uint8(y_true > 0.5)
        y_pred = np.uint8(y_pred > 0.5)

    tpr = np.sum(y_true * y_pred) / float(np.sum(y_true) + 1e-6)
    fpr = np.sum((1. - y_true) * y_pred) / float(np.sum(1. - y_true) + 1e-6)
    return fpr, tpr


# 二类分割质量评估，很受二值化强度影响
class TwocatMetric:
    def __init__(self):
        self.iou_list = []
        self.binary_fg_iou_list = []
        self.miou_list = []
        self.biou_list = []
        self.b_l1_list = []
        self.fpr_list = []
        self.tpr_list = []

    def update(self, gt, pred):
        gt = np.squeeze(gt)
        pred = np.squeeze(pred)
        if gt.ndim == 2:
            gt = gt[np.newaxis]
            pred = pred[np.newaxis]
        for gt_item, pred_item in zip(gt, pred):
            if str(gt_item.dtype) != 'uint8':
                gt_item = np.uint8(gt_item * 255)
            if str(pred_item.dtype) != 'uint8':
                pred_item = np.uint8(pred_item * 255)
            gt_item = cv2.resize(gt_item, (256, 256))
            pred_item = cv2.resize(pred_item, (256, 256))
            self.iou_list.append(iou(gt_item, pred_item))
            self.binary_fg_iou_list.append(iou(gt_item, pred_item, binary=True))
            self.miou_list.append(mean_iou(gt_item, pred_item))
            self.biou_list.append(iou(gt_item, pred, boundary_only=True))
            self.b_l1_list.append(boundary_l1(gt_item, pred_item))
            fpr_item, tpr_item = fpr_tpr(gt_item, pred_item)
            self.fpr_list.append(fpr_item)
            self.tpr_list.append(tpr_item)

    def get_scores(self):
        return {
            'fg_iou': np.mean(self.iou_list),
            'binary_fg_iou': np.mean(self.binary_fg_iou_list),
            'boundary_iou': np.mean(self.biou_list),
            'boundary_l1': np.mean(self.b_l1_list),
            'miou': np.mean(self.miou_list),
            'fpr': np.mean(self.fpr_list),
            'tpr': np.mean(self.tpr_list),
        }


class MetricByConfuseMatrix(object):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        # mask = (label_true >= 0) & (label_true < n_class)
        # 头发分割计算mIoU的时候pred 也会计算ignore
        mask = (label_true >= 0) & (label_true < n_class) & (label_pred >= 0) & (label_pred < n_class)
        # use label only when label value is valid
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix.copy()
        acc = np.diag(hist).sum() / hist.sum()
        cls_acc = np.diag(hist) / hist.sum(axis=1)
        cls_rec = np.diag(hist) / hist.sum(axis=0)
        mean_acc = np.nanmean(cls_acc)  # 忽略nana，计算均值，很有意义，因为并不是每一个分割的类别都在图片里面存在
        f1 = np.diag(hist) * 2 / (hist.sum(axis=1) + hist.sum(axis=0))
        mean_f1 = np.nanmean(f1)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)  # 忽略nan，计算均值
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * cls_acc[freq > 0]).sum()
        fwaviou = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        return {
            # 'Overall Acc': acc,
            #     'Mean Acc': mean_acc,
            #     'Cls Acc': cls_acc,
            #     'FreqW Acc': fwavacc,
            'Mean IoU': mean_iu,
            # 'Mean F1': mean_f1,
            'Cls IoU': cls_iu,
            # 'FreaW IoU': fwaviou,
            # 'Cls Rec': cls_rec
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


# 广义连续二类分割metrics
class TwocatMetricPth:
    def __init__(self, n_classes=1, is_distributed=False):
        self.iou_list = []
        self.bg_iou_list = []
        self.miou_list = []
        self._is_distributed = is_distributed
        self._norm_size = 256  # metric eval on the same size for all models

    def _iou(self, gt, pred):
        gt_pred_stack = torch.stack([gt, pred], dim=-1)
        inter, _ = gt_pred_stack.min(dim=-1)
        inter = inter.sum()
        union, _ = gt_pred_stack.max(dim=-1)
        union = union.sum(dim=-1)
        iou_value = (inter + 1e-6) / (union + 1e-6)
        if self._is_distributed:
            iou_value = reduce_tensor(iou_value)  # avg over batch first
        iou_value = iou_value.mean(dim=0)
        return iou_value

    def update(self, gt, pred):  # nchw
        # gt = torch.squeeze(gt)
        # pred = torch.squeeze(pred)
        batch_size = gt.size(0)
        gt = nnF.interpolate(gt, size=(self._norm_size, self._norm_size), mode='bilinear', align_corners=False)  # n w h
        pred = nnF.interpolate(pred, size=(self._norm_size, self._norm_size), mode='bilinear', align_corners=False)
        gt = gt.view(batch_size, -1)
        pred = pred.view(batch_size, -1)
        iou_value = self._iou(gt, pred)
        bg_iou_value = self._iou(1. - gt, 1. - pred)
        iou_value = float(iou_value.cpu().numpy().copy())
        bg_iou_value = float(bg_iou_value.cpu().numpy().copy())
        miou = (iou_value + bg_iou_value) / 2
        self.iou_list.append(iou_value)
        self.bg_iou_list.append(bg_iou_value)
        self.miou_list.append(miou)

    def get_scores(self):
        return {
            'fg-iou': np.mean(self.iou_list),
            'bg-iou': np.mean(self.bg_iou_list),
            'miou': np.mean(self.miou_list)
        }


class MTC:
    def __init__(self):
        self.flow_instance = FlowToolkit()
        self.mTC = []

    def update(self, frames_seq, masks_seq, flows_seq=None, vis_flag=False):
        self.flow_instance = FlowToolkit()
        iou_list = []
        for idx in range(len(frames_seq) - 1):
            img_0 = frames_seq[idx]
            img_1 = frames_seq[idx + 1]
            mask_0 = masks_seq[idx]
            mask_1 = masks_seq[idx + 1]
            if flows_seq is not None:
                flow_src = flows_seq[idx]
                mask_1_warp = self.flow_instance.warp_mask_with_flow(mask_0, flow_src)
                img_1_warp = self.flow_instance.warp_mask_with_flow(img_0, flow_src)
            else:
                mask_1_warp = self.flow_instance.warp_mask(img_0, img_1, mask_0, flow_vis=vis_flag)
                img_1_warp = self.flow_instance.warp_mask(img_0, img_1, img_0, flow_vis=vis_flag)
            iou_value = iou(mask_1, mask_1_warp, binary=False)
            if vis_flag:
                print(iou_value)
                img_vis = np.hstack([img_0, img_1, img_1_warp])
                mask_vis = np.hstack([mask_1, mask_1_warp])
                mask_vis = np.stack([mask_vis] * 3, axis=-1)
                vis = np.hstack([img_vis, mask_vis])
                cv2.imshow('flow_vis', vis)
                cv2.waitKey(0)
            iou_list.append(iou_value)
        mtc_item = np.mean(iou_list)
        self.mTC.append(mtc_item)
        return mtc_item

    def update_with_gt(self, masks_seq, masks_seq_gt, vis_flag=False):
        iou_list = []
        for mask_pred, mask_gt in zip(masks_seq, masks_seq_gt):
            iou_value = iou(mask_pred, mask_gt, binary=False)
            if vis_flag:
                print(iou_value)
                vis = np.hstack([mask_pred, mask_gt])
                cv2.imshow('mask', vis)
                cv2.waitKey(0)
            iou_list.append(iou_value)
        mtc_item = np.mean(iou_list)
        self.mTC.append(mtc_item)
        return mtc_item

    def reset(self):
        self.mTC = []

    def get_scores(self):
        return {
            'mTC': np.mean(self.mTC),
            'allTC': self.mTC,
            'TC_std': np.std(self.mTC)
        }


if __name__ == '__main__':
    test_img = r'/Users/shuai.he/Downloads/1489523495612668991.png-1000000-404.png'
    img = cv2.imread(test_img, 0)
    mask_0 = np.hsplit(img, indices_or_sections=4)[1]
    mask_1 = np.hsplit(img, indices_or_sections=4)[2]
    quality_value = mask_quality_by_laplacian(mask_0)
    quality_value_1 = mask_quality_by_laplacian(mask_1)
    print('{}-{}'.format(quality_value, quality_value_1))
    plt.imshow(img)
    plt.show()
