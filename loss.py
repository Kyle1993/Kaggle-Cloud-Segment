import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

nn.BCELoss()

def f_score(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = (2 * tp + eps) / (2* tp + fn + fp + eps)

    return score

class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)

class FocalLoss2d(nn.Module):
    def __init__(self,alpha=1.4, gamma=2, activation='sigmoid'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if activation is None or activation == "none":
            self.activation = lambda x: x
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d"
            )

    def forward(self, outputs, targets):
        outputs = self.activation(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        # non_ignored = targets.view(-1) != self.ignore_index
        # targets = targets.view(-1)[non_ignored].float()
        # outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)*(targets*self.alpha+(1-targets)*(2-self.alpha))).mean()

class FocalLoss2d2(nn.Module):
    def __init__(self, alpha=1.4, gamma=2, activation='sigmoid'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if activation is None or activation == "none":
            self.activation = lambda x: x
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d"
            )

    def forward(self, outputs, targets):
        outputs = self.activation(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        res = (-self.alpha*targets*torch.log(outputs)*((1-outputs)**self.gamma) - (2-self.alpha)*(1-targets)*torch.log(1-outputs)*(outputs**self.gamma))
        # res = (-self.alpha*targets*torch.log(outputs) - (2-self.alpha)*(1-targets)*torch.log(1-outputs))
        # print(res)
        return res.mean()

class FocalBCEDiceLoss(nn.Module):
    def __init__(self, eps=1e-7, gamma=2, activation='sigmoid'):
        super(FocalBCEDiceLoss,self).__init__()
        self.focal = FocalLoss2d(gamma=gamma,activation=activation)
        self.dice = DiceLoss(eps=eps,activation=activation)

    def forward(self, y_pr, y_gt):
        dice = self.dice(y_pr, y_gt)
        focal = self.focal(y_pr, y_gt)
        return dice + focal

class FocalLoss1d(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss1d, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.Tensor([alpha,1-alpha])
        else:
            self.alpha = None

        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

if __name__ == '__main__':
    f1 = FocalLoss2d()
    f2 = FocalLoss2d2(alpha=1)
    # x = torch.rand((2,4,480,640))
    # x[x>0.5] = 1
    # x[x<=0.5] = 0
    # y = torch.rand((2,4,480,640))

    x = np.load('masks.npy')
    y = np.load('outputs.npy')

    print(x.shape,x.min(),x.max())
    print(y.shape,y.min(),y.max())

    l1 = f1(y,x)
    l2 = f2(y,x)
    print(l1,l2)
