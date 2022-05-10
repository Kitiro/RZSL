#!/usr/bin/env python3
# coding=utf-8
"""
Author: Kitiro
Date: 2021-10-31 20:55:26
LastEditTime: 2021-10-31 20:55:26
LastEditors: Kitiro
Description: 
FilePath: /web_zsl/train_zsl.py
"""

import numpy as np
import torch
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
from torchsummary import summary
import json
import random

# Author Defined
from model import MyModel, APYModel, LinearModel
from utils import (
    compute_accuracy,
    set_seed,
    weights_init,
    export_log,
    plot_img,
)
from dataset_unseen import DataSet


def export_best(args, best):
    with open(f"best_result_{args.dataset}.txt", "a") as file:
        file.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                args.dataset, args.seed, args.attr_num, args.c_w, best
            )
        )

def eval(model, loader):

    mse_loss = nn.MSELoss()
    with torch.set_grad_enabled(False):

        total_num = 0
        running_loss = 0.0

        # Iterate over data.
        for visual_batch, attr_batch, label_batch in loader:
            visual_batch = visual_batch.float().cuda()

            attr_batch = (
                attr_batch.float()
                .reshape(visual_batch.shape[0], 1, 1, attr_batch.shape[1])
                .cuda()
            )

            out_visual = model(attr_batch)  # semantic -> visual space

            loss = mse_loss(out_visual, visual_batch)

            # statistics loss and acc every epoch
            running_loss += loss.item() * visual_batch.shape[0]

            total_num += visual_batch.shape[0]

    return running_loss / total_num


def train_and_test(loader, dataset, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    visual_feature_dim = dataset.train_feature.shape[1]
    semantic_feature_dim = dataset.attr_dim[1]
    class_num = dataset.attr_dim[0]
    if args.dataset == "APY":
        model = APYModel(semantic_feature_dim, visual_feature_dim)
    else:
        model = LinearModel(semantic_feature_dim, visual_feature_dim)
    model.cuda()
    # model.apply(weights_init)

    summary(model, input_size=(1, 256, semantic_feature_dim))
    mse_loss = nn.MSELoss()

    best_zsl = 0.0
    best_h = 0.0
    best_h_line = ""
    h_line = ""

    postfix = "C_W-{}_GenAttr-{}".format(args.c_w, args.attr_num)
    if args.screen:
        postfix += "_screen"
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4, betas=[0.9, 0.999]
    )  # here lr is the overall learning rate
    # scheduler = lr_scheduler.MultiStepLR(
    #     opt, milestones=list(range(20, 300, 20)), gamma=0.5
    # )

    history = {
        "Acc_ZSL": [],
        "Loss_attr": [],
        "Loss_cluster": [],
        "Loss_center": [],
        "Loss_Total": [],
    }
    log_head = "Original_attr: {}\tGenerated_attr: {}\tIs_screen: {}\tScreen_ratio: {}\tTotal_attr: {}\n".format(
        dataset.attr1_dim[1],
        dataset.generative_attribute_num,
        dataset.screen,
        dataset.screen_ratio,
        dataset.attr_dim[1],
    )
    log_head += "Epoch\tAcc_ZSL\tBest_Acc\tLoss_attr\tLoss_center(c_w)\tTotal_Loss\n"
    log_path = os.path.join(
        args.output, args.dataset, "LOG", f"Log_{postfix}_Seed_{args.seed}.log"
    )
    if os.path.exists(log_path):
        os.remove(log_path)
    # export_log(log_head, log_path)

    with torch.set_grad_enabled(True):
        for epoch in range(args.epochs):
            model.train()
            for visual_batch, attr_batch, label_batch in loader:
                visual_batch = visual_batch.float().cuda()
                attr_batch = attr_batch.float().cuda()
                # attr_batch = (attr_batch.float().reshape(visual_batch.shape[0], 1, 1, semantic_feature_dim).cuda())
                opt.zero_grad()

                out_visual = model(attr_batch)  # semantic -> visual space

                loss = mse_loss(out_visual, visual_batch)
                opt.zero_grad()
                loss.backward()

                opt.step()

            # scheduler.step()
            model.eval()

            acc_zsl = compute_accuracy(
                model,
                dataset.test_att_map_unseen,
                dataset.test_feature_unseen,
                dataset.test_id_unseen,
                dataset.test_label_unseen,
            )
            acc_seen_gzsl = compute_accuracy(
                model,
                dataset.attribute,
                dataset.test_feature_seen,
                np.arange(class_num),
                dataset.test_label_seen,
            )
            acc_unseen_gzsl = compute_accuracy(
                model,
                dataset.attribute,
                dataset.test_feature_unseen,
                np.arange(class_num),
                dataset.test_label_unseen,
            )
            H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)

            if acc_zsl > best_zsl:
                best_zsl = acc_zsl
                if args.save_model:
                    save_path = os.path.join(
                        args.output, "{}/Model_{}.pth".format(args.dataset, postfix)
                    )
                    torch.save(model.state_dict(), save_path)
                    print("model has been stored in ", save_path)

            if H > best_h:
                best_h = H
                best_h_line = "gzsl: seen=%.4f, unseen=%.4f, h=%.4f" % (
                    acc_seen_gzsl,
                    acc_unseen_gzsl,
                    H,
                )

            print("Epoch:", epoch, "--------")
            print("zsl:", acc_zsl)
            print("best_zsl:", best_zsl)

            print("Total-loss:{}".format(loss.item()))

            print("lr:", opt.param_groups[0]["lr"])
            h_line = "gzsl: seen={:.4f}, unseen={:.4f}, h={:.4f}".format(
                acc_seen_gzsl, acc_unseen_gzsl, H
            )
            print(h_line)
            print(best_h_line)

            # log_info = f"{epoch}\t{acc_zsl}\t{best_zsl}\t{loss_attr.item()}\t{loss_center.item()}\t{loss.item()}\n{h_line}\t{best_h_line}\n"
            history["Acc_ZSL"].append(acc_zsl)
            history["Loss_Total"].append(loss.item())

            # export_log(log_info, log_path)

            if args.plot:
                plot_img(history, os.path.join(args.output, args.dataset), postfix)
    export_best(args, best_zsl)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="AWA2", help="dataset for experiments"
)
parser.add_argument(
    "--attr_num", type=int, default=0, help="num of generative attribute for training"
)
parser.add_argument("--output", type=str, default="./output/", help="Output directory")
parser.add_argument("--alpha", type=float, default="0.0")
parser.add_argument("--c_w", type=float, default="0", help="Center loss weight")
parser.add_argument("--seed", type=int, default=-1)  # 8729
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument(
    "--screen", action="store_true", default=False, help="Whether to screen attributes"
)
parser.add_argument(
    "--plot", action="store_true", default=False, help="Whether to plot curve"
)
parser.add_argument(
    "--aux", action="store_true", default=False, help="Whether to use aux feature"
)
parser.add_argument(
    "--save_model", action="store_true", default=False, help="Whether to plot curve"
)
parser.add_argument("--gpu", default="0", help="GPU")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")

args = parser.parse_args()

# 设置随机数种子
if args.seed == -1:
    args.seed = random.randint(1, 10000)
    set_seed(args.seed)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    print("Running parameters:")
    print(json.dumps(vars(args), indent=4, separators=(",", ":")))
    dataset = DataSet(
        name=args.dataset,
        generative_attribute_num=args.attr_num,
        screen=args.screen,
        screen_ratio=0.2,
        is_aux_feat=args.aux,
    )

    dataset_train = TensorDataset(
        torch.from_numpy(dataset.train_feature),
        torch.from_numpy(dataset.train_att),
        torch.from_numpy(dataset.train_label),
    )
    train_loader = DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    train_and_test(train_loader, dataset, args)


if __name__ == "__main__":
    main()

