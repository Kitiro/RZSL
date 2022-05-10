#!/usr/bin/env python
# coding=utf-8
"""
Author: Kitiro
Date: 2020-11-03 20:59:36
LastEditTime: 2020-11-05 11:51:08
LastEditors: Kitiro
Description: 
FilePath: /zzc/exp/Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/utils.py
"""

import numpy as np
import os
import torch
from torch import nn
import random
import matplotlib.pyplot as plt


def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def cal_mean_feature(cluster, input_features):
    k = len(np.unique(cluster))
    features = []
    record_times = [0] * k
    for i in range(k):
        features.append([0] * len(input_features[0]))  # e.g. 3*2048 all zero
    for index, clus in enumerate(cluster):
        features[clus] += input_features[index]
        record_times[clus] += 1
    return np.array([f / record_times[i] for i, f in enumerate(features)])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


# classify using NN
def Classify(source, search_space, labels):

    sapce_dim = search_space.shape[
        0
    ]  # shape[0] stands for the num of classes to be decice

    diff = np.tile(source, (sapce_dim, 1)) - search_space  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    max_index = np.argsort(distance)[0]
    target_label = labels[max_index]

    return target_label


def compute_accuracy(model, test_att, test_visual, test_id, test_label):
    """[summary]

    Args:
        model ([type]): [description]
        test_att ([type]): semantic feature
        test_visual ([type]): visual feature
        test_id ([type]): att2label
        test_label ([type]): x2label
    """
    model.eval()
    with torch.no_grad():
        #test_att = torch.tensor(test_att).unsqueeze(1).unsqueeze(1).float().cuda()  # for conv. model
        test_att = torch.tensor(test_att).float().cuda()
        attr_feature = model(test_att)

        outpred = [0] * test_visual.shape[0]
        test_label = test_label.astype("float32")
        # 将类的属性映射到visual space. 得到att_pred。然后每一张图片去找距离最近的attr of unseen
        for i in range(test_visual.shape[0]):
            outputLabel = Classify(
                test_visual[i, :], attr_feature.cpu().detach().numpy(), test_id
            )
            outpred[i] = outputLabel

        outpred = np.array(outpred)
        acc = np.equal(outpred, test_label).mean()
    return acc


def plot_img(his_dict, path, posfix):
    key = "Acc_ZSL"
    data = his_dict[key]
    plt.clf()
    plt.plot(range(len(data)), data, color="#99CC33", label=key)
    plt.axhline(y=max(data), color="#FF0033", linestyle="-", label="Best_Acc")
    plt.title(key)
    plt.legend(loc="right")
    plt.text(0, max(data) + 0.001, str(round(max(data), 6)))
    plt.savefig(os.path.join(path, "ACC", f"{posfix}_{key}.png"))

    plt.clf()
    color = ["#FF9966", "#99CCFF", "#FF99CC", "#663366"]  # orange, blue, pink, purple

    for index, (key, data) in enumerate(his_dict.items()):
        if key != "Acc_ZSL":
            plt.plot(range(len(data)), data, color=color[index - 1], label=key)
    plt.legend(loc="upper right")
    plt.title("Total_Loss")
    plt.savefig(os.path.join(path, "LOSS", f"{posfix}_Loss.png"))


def export_log(log_text, path):
    with open(path, "a") as file:
        file.write(log_text)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim).cuda()
            )
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )

        classes = torch.arange(self.num_classes).long()

        if self.use_gpu:
            distmat = distmat.cuda()
            classes = classes.cuda()

        distmat.addmm_(
            1, -2, x, self.centers.t()
        )  # 1*distmat + (-2)*(x*self.centers'T)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'{score}/{self.best_score} EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        