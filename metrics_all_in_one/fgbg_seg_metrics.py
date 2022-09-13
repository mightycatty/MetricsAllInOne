"""metrics specifically for two category segmentation(or regression), for exmaple, portrait/human/face/skin segmentation that commonly used in production
"""

import cv2
import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn.functional as nnF
from skimage.morphology import square, opening
from metric_interface import MetricsInterface


def _get_boundary_torch(mask, ksize=7, multi_original=False):
    dilated = nnF.max_pool2d(mask, ksize, 1, ksize // 2)
    eroded = - nnF.max_pool2d(-mask, ksize, 1, ksize // 2)
    boundary = dilated - eroded
    if multi_original:
        boundary[boundary > 0.05] = 1.
        boundary = boundary * mask
    boundary = torch.clamp(boundary, 0., 1.)
    return boundary


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


def _torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num


def _iou_torch(gt, pred):
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
    return _iou_torch(gt_slim, pred_slim)


# 广义非离散二类(或reg)分割metrics
class FgBgSegMetricTorch(MetricsInterface):
    def __init__(self, n_classes=1, is_distributed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            iou_value = _reduce_tensor(iou_value)
        iou_value = iou_value.mean(dim=0)  # avg over batch first
        return iou_value

    def update(self, gt, pred):  # nchw
        # gt = torch.squeeze(gt)
        # pred = torch.squeeze(pred)
        batch_size = gt.size(0)
        gt = nnF.interpolate(gt, size=(self._norm_size, self._norm_size), mode='bilinear', align_corners=False)  # n w h
        pred = nnF.interpolate(pred, size=(self._norm_size, self._norm_size), mode='bilinear', align_corners=False)
        # boundary
        gt_boundary = _get_boundary_torch(gt).view(batch_size, -1)
        pred_boundary = _get_boundary_torch(pred).view(batch_size, -1)
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

    def get_scores(self, *args, **kwargs):
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
