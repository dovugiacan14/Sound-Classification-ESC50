import os
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


def clip_bce(pred, target):
    """Binary crossentropy loss."""
    return F.binary_cross_entropy(pred, target)


def clip_ce(pred, target):
    return F.cross_entropy(pred, target)


def get_loss_func(loss_type):
    if loss_type == "clip_bce":
        return clip_bce
    if loss_type == "clip_ce":
        return clip_ce
    if loss_type == "asl_loss":
        loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
        return loss_func


def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [
        np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)
    ]
    return np.array(mixup_lambdas).astype(np.float32)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x  # without sigmoid since it has been computed
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()
