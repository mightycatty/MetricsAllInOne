# -*- coding: utf-8 -*-

# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import sys

import numpy as np
import torch

from ..engine.utils.distributed import reduce_tensor
from ..transforms.img_process_torch import get_boundary_torch
import torch.nn.functional as nnF
from skimage.morphology import square, opening
import cv2

#
def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num


def iou(gt, pred):
    batch_size = pred.size(0)
    pred = pred.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    gt_pred_stack = torch.stack([gt, pred], dim=-1)
    inter, _ = gt_pred_stack.min(dim=-1)
    inter = inter.sum(dim=-1)
    union, _ = gt_pred_stack.max(dim=-1)
    union = union.sum(dim=-1)
    iou_value = (inter + 1e-6) / (union + 1e-6)
    return iou_value


def get_slim_structure(mask, k_size=9):
    mask_open = opening(mask, square(k_size))
    mask_slim_structure = cv2.absdiff(mask, mask_open)
    return mask_slim_structure


def slim_structure_iou(gt, pred):
    gt_slim = get_slim_structure(gt)
    pred_slim = get_slim_structure(pred)
    return iou(gt_slim, pred_slim)


# 广义非离散二类分割metrics
class TwocatMetricPth:
    def __init__(self, n_classes=1, is_distributed=False):
        self.reset()
        self._is_distributed = is_distributed
        self._norm_size = 256  # metric eval on the same size for all models

    def _iou(self, gt, pred):
        gt_pred_stack = torch.stack([gt, pred], dim=-1)
        inter, _ = gt_pred_stack.min(dim=-1)
        inter = inter.sum(dim=-1)
        union, _ = gt_pred_stack.max(dim=-1)
        union = union.sum(dim=-1)
        iou_value = (inter + 1e-6) / (union + 1e-6)
        if self._is_distributed:
            iou_value = reduce_tensor(iou_value)
        iou_value = iou_value.mean(dim=0)  # avg over batch first
        return iou_value

    def update(self, gt, pred):  # nchw
        # gt = torch.squeeze(gt)
        # pred = torch.squeeze(pred)
        batch_size = gt.size(0)
        gt = nnF.interpolate(gt, size=(self._norm_size, self._norm_size), mode='bilinear', align_corners=False)  # n w h
        pred = nnF.interpolate(pred, size=(self._norm_size, self._norm_size), mode='bilinear', align_corners=False)
        # boundary
        gt_boundary = get_boundary_torch(gt).view(batch_size, -1)
        pred_boundary = get_boundary_torch(pred).view(batch_size, -1)
        iou_boundary = self._iou(gt_boundary, pred_boundary)
        iou_boundary = float(iou_boundary.cpu().numpy().copy())
        self.boundary_iou_list.append(iou_boundary)
        # fg/bg iou
        gt = gt.view(batch_size, -1)
        pred = pred.view(batch_size, -1)
        iou_value = self._iou(gt, pred)
        bg_iou_value = self._iou(1. - gt, 1. - pred)
        iou_value = float(iou_value.cpu().numpy().copy())
        bg_iou_value = float(bg_iou_value.cpu().numpy().copy())
        miou = (iou_value + bg_iou_value) / 2
        # binary fg iou
        gt = (gt > 0.5).float()
        pred = (pred > 0.5).float()
        binary_iou_value = self._iou(gt, pred)
        binary_iou_value = float(binary_iou_value.cpu().numpy().copy())

        self.binary_fg_iou_list.append(binary_iou_value)
        self.iou_list.append(iou_value)
        self.bg_iou_list.append(bg_iou_value)
        self.miou_list.append(miou)

    def get_scores(self):
        return {
            'fg-iou': np.mean(self.iou_list),
            'binary-fg-iou': np.mean(self.binary_fg_iou_list),
            'bg-iou': np.mean(self.bg_iou_list),
            'miou': np.mean(self.miou_list),
            'b-iou': np.mean(self.boundary_iou_list)
        }

    def reset(self):
        self.iou_list = []
        self.binary_fg_iou_list = []
        self.bg_iou_list = []
        self.miou_list = []
        self.boundary_iou_list = []


class cls_metric_np(object):

    def __init__(self, n_classes):
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

    def get_scores(self, dataset=''):
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
        return {'Overall Acc': acc,
                'Mean Acc': mean_acc,
                'Cls Acc': cls_acc,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu,
                'Mean F1': mean_f1,
                'Cls IoU': cls_iu,
                'FreaW IoU': fwaviou,
                'Cls Rec': cls_rec}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class cls_metric_pth(object):

    def __init__(self, n_classes, is_distributed=False):
        self.n_classes = n_classes
        self.is_distributed = is_distributed
        self.confusion_matrix = torch.zeros((n_classes, n_classes)).long()
        if self.is_distributed:
            self.confusion_matrix = self.confusion_matrix.cuda()

    def _fast_hist(self, label_true, label_pred, n_class):
        # mask = (label_true >= 0) & (label_true < n_class)
        # 头发分割计算mIoU的时候pred 也会计算ignore
        # print(f"label_true.shape: {label_true.shape}, label_pred.shape: {label_pred.shape}, n_class: {n_class}")
        mask = (label_true >= 0) & (label_true < n_class) & (label_pred >= 0) & (label_pred < n_class)
        # use label only when label value is valid
        hist = torch.bincount(
            n_class * label_true[mask] +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            if lt.shape != lp.shape:
                print("Warning: preds have not been transformed into labels !!!")
                lp_argmax = torch.argmax(lp, dim=0)
                # print(f"lt.shape: {lt.shape}, lp.shape: {lp.shape}, lp_argmax.shape: {lp_argmax.shape}")
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp_argmax.flatten(), self.n_classes)
            else:
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, dataset=''):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix.float()
        if self.is_distributed:
            hist = reduce_tensor(hist)
        hist = hist.cpu()
        acc = torch.diag(hist).sum() / hist.sum()  # 跟python2一样，整数相除会出问题
        cls_acc = torch.diag(hist) / hist.sum(dim=1)
        cls_rec = torch.diag(hist) / hist.sum(axis=0)
        # mean_acc = torch.mean(cls_acc)
        mean_acc = torch_nanmean(cls_acc)
        tp = torch.diag(hist)
        f1 = tp * 2 / (tp * 2 + (hist.sum(dim=1) - tp) + (hist.sum(dim=0) - tp))
        mean_f1 = torch_nanmean(f1)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        # mean_iu = torch.mean(iu)
        mean_iu = torch_nanmean(iu)
        freq = hist.sum(dim=1) / hist.sum()
        fwavacc = (freq[freq > 0] * cls_acc[freq > 0]).sum()
        fwaviou = (freq[freq > 0] * iu[freq > 0]).sum()

        iu = iu.cpu().numpy()
        cls_iu = dict(zip(range(self.n_classes), iu))

        if self.is_distributed:
            return {'Overall Acc': acc.cpu().numpy(),
                    'Mean Acc': mean_acc.cpu().numpy(),
                    'Cls Acc': cls_acc.cpu().numpy(),
                    'FreqW Acc': fwavacc.cpu().numpy(),
                    'Mean F1': mean_f1.cpu().numpy(),
                    'Mean IoU': mean_iu.cpu().numpy(),
                    'Cls IoU': cls_iu,
                    'FreaW IoU': fwaviou.cpu().numpy(),
                    'Cls Rec': cls_rec.cpu().numpy()}
        else:
            return {'Overall Acc': acc,
                    'Mean Acc': mean_acc,
                    'Cls Acc': cls_acc,
                    'FreqW Acc': fwavacc,
                    'Mean IoU': mean_iu,
                    'Mean F1': mean_f1,
                    'Cls IoU': cls_iu,
                    'FreaW IoU': fwaviou,
                    'Cls Rec': cls_rec}

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes)).long()
        if self.is_distributed:
            self.confusion_matrix = self.confusion_matrix.cuda()
