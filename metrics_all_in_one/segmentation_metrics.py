"""metrics for segmentation
"""

import cv2
import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn.functional as nnF
from skimage.morphology import square, opening
from metric_interface import MetricsInterface


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


class SegMetricsPy(MetricsInterface):
    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

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

    def _fast_hist(self, label_true, label_pred, n_class):
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

    def get_scores(self, *args, **kwargs):
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


class SegMetricsTorch(MetricsInterface):
    def __init__(self, n_classes, is_distributed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.is_distributed = is_distributed
        self.confusion_matrix = torch.zeros((n_classes, n_classes)).long()
        if self.is_distributed:
            self.confusion_matrix = self.confusion_matrix.cuda()

    def _fast_hist(self, label_true, label_pred, n_class):
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

    def get_scores(self, *args, **kwargs):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix.float()
        if self.is_distributed:
            hist = _reduce_tensor(hist)
        hist = hist.cpu()
        acc = torch.diag(hist).sum() / hist.sum()  # 跟python2一样，整数相除会出问题
        cls_acc = torch.diag(hist) / hist.sum(dim=1)
        cls_rec = torch.diag(hist) / hist.sum(axis=0)
        # mean_acc = torch.mean(cls_acc)
        mean_acc = _torch_nanmean(cls_acc)
        tp = torch.diag(hist)
        f1 = tp * 2 / (tp * 2 + (hist.sum(dim=1) - tp) + (hist.sum(dim=0) - tp))
        mean_f1 = _torch_nanmean(f1)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        # mean_iu = torch.mean(iu)
        mean_iu = _torch_nanmean(iu)
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
