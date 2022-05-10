#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-10-31 14:29:42
LastEditTime: 2021-12-18 10:24:42
LastEditors: Kitiro
Description: 
FilePath: /exp/web_zsl/extract_feature.py
'''
import argparse
import os
from re import M
import numpy as np

import scipy.io as sio
from tqdm import tqdm 
import argparse
import numpy as np

import scipy.io as sio
from tqdm import tqdm 

import os, torch
import numpy as np
from torch.autograd import Variable
from PIL import Image 
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm 
import traceback
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

data_dir = 'data'
web_data_dir = 'web_data'
web_feat_dir = 'web_feat'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def conver_img(img_path, extract=False):
    img = Image.open(img_path)
    img = transform(img)
    if img.shape[0] == 1:
        img = torch.Tensor(np.tile(img, (3,1,1)))
    img = normalize(img)
    if extract:
        # 将img第一维unsqueeze，传入模型
        img = Variable(torch.unsqueeze(img, dim=0).float())
    return img

class DataSet(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list
        assert len(self.img_list) == len(self.label_list)

    # for training
    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = conver_img(img_path)
        return img, label

    def __len__(self):
        return len(self.label_list)
    

def get_model(args):
    if args.model == 'res50':
        model = models.resnet50(pretrained=True)
    elif args.model  == 'res101':
        model = models.resnet101(pretrained=True)
    elif args.model == 'res152':
        model = models.resnet152(pretrained=True)
    del model.fc
    model.fc = lambda x:x
    print('extract feature with pretrained %s' % args.model)
    model.eval()
    return model


# load data 
def get_search_list(name):
    mat = sio.loadmat(os.path.join(data_dir, name+'_data','att_splits.mat'))
    class_name = mat['allclasses_names']
    labels, class_list = [], []
    for idx, cla in enumerate(class_name):
        labels.append(idx)
        cla = cla[0][0]
        if name == 'CUB':
            cla = cla.split('.')[1]
        class_list.append(cla)

    print('DataSet:', name, 'Contains', str(len(class_list))+' classes.')
    print(class_list[:10])
    return class_list, labels
    

# logits:tensor. label:numpy
def get_pred(logits):
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    pred = np.argmax(probs, axis=1)
    return pred

def finetune(args):
    # prepare fine tune last 2 layers
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    params_to_update = []
    model = get_model(args)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    opt = optim.Adam(params_to_update, lr=0.0001, weight_decay=1e-2)
    
    ## prepare data
    class_dir = os.path.join(data_dir, args.ds+'_data')
    image_dir = os.path.join(class_dir, 'images')
    mat = sio.loadmat(os.path.join(class_dir, 'res101.mat'))
    img_files = mat['image_files'].squeeze()
    labels = mat['labels'].squeeze()-1
    img_files = np.array([os.path.join(image_dir, img_file[0].split('Animals_with_Attributes2//')[-1]) for img_file in img_files])
    
    mat = sio.loadmat(os.path.join(class_dir, 'att_splits.mat'))
    train_loc = mat['trainval_loc'].squeeze()-1
    # test_loc = np.hstack((mat['test_seen_loc'].squeeze()-1, mat['test_unseen_loc'].squeeze()-1))
    
    train_files = img_files[train_loc]
    # test_files = img_files[test_loc]
    train_labels = labels[train_loc]
    # test_labels = labels[test_loc]
    
    train_data = DataSet(train_files, train_labels)
    train_loader = DataLoader(
        dataset=train_data, batch_size=256, shuffle=True, num_workers=4
    )
    
    best_loss = float("inf")
    for epoch in tqdm(range(epochs)):
        with torch.set_grad_enabled(True):
            model.train()
            hit, cnt = 0, 0
            losses = 0
            for feat_batch, label_batch in train_loader:
                feat_batch = feat_batch.float().cuda()
                # label_batch = torch.unsqueeze(label_batch.long(), 0).cuda()
                label_batch = label_batch.long().cuda()
                opt.zero_grad()
                logits = model(feat_batch)  # semantic -> visual space
                loss = loss_fn(logits, label_batch)
                hit += np.sum(get_pred(logits) == label_batch.detach().cpu().numpy())
                cnt += label_batch.shape[0]
                loss.backward()
                opt.step()
                losses += loss.item()
            loss_epoch = losses/cnt
            print(epoch, '/',epochs, 'Train Acc:', 1.0*hit/cnt, 'Loss:',loss_epoch)   
            if loss_epoch < best_loss:
                print('Save Model on Loss:', loss_epoch, 'at models/res101.pth')
                torch.save(model.state_dict(), os.path.join('models', 'res101.pth'))

def extract_web(args):
    feats, labels = [], []
    class_list, label_list = get_search_list(args.ds)
    model = get_model(args).cuda()
    # 每次处理一类
    for cla, label in zip(class_list, label_list):
        print(label, '/', len(label_list), cla)
        img_dir = os.path.join(web_data_dir, args.ds, cla)
        img_files = os.listdir(img_dir)
        img_files.sort(key= lambda x:int(x[:-4]))  # 按index排序
        img_files = [os.path.join(img_dir, file) for file in img_files]
        cnt = 0
        for img_path in img_files:
            try:
                img = conver_img(img_path=img_path, extract=True).cuda()
            except:
                print('failed!!!', img_path)
            feat = model(img).detach().cpu().numpy().squeeze()
            if len(feats) == 0:
                feats = feat
            else:
                feats = np.vstack((feats, feat))
            cnt += 1
        label_class = [label]*cnt
        labels.extend(label_class)
        print('feature shape:', feats.shape)    

    print(args.ds)
    print('feature shape:', feats.shape)   
    mat_path = '{}_{}_features.mat'.format(args.model, args.ds)
    sio.savemat(os.path.join(web_feat_dir, mat_path), {
        'features':feats,
        'labels': np.array(labels)
    })
    print('save features on %s' % os.path.join(web_feat_dir, mat_path))
# 'ft_res101_feature_AWA2.mat'

def extract_source(args):
    class_dir = os.path.join(data_dir, args.ds+'_data')
    image_dir = os.path.join(class_dir, 'images')
    mat = sio.loadmat(os.path.join(class_dir, 'res101.mat'))
    img_files = mat['image_files'].squeeze()
    labels = mat['labels'].squeeze()-1
    if args.ds == 'AWA2':
        img_files = np.array([os.path.join(image_dir, img_file[0].split('JPEGImages/')[-1]) for img_file in img_files])
    elif args.ds == 'CUB':
        img_files = np.array([os.path.join(image_dir, img_file[0].split('images/')[-1]) for img_file in img_files])
    elif args.ds == 'FLO':
        img_files = np.array([os.path.join(image_dir, img_file[0].split('jpg/')[-1]) for img_file in img_files])
        
    model = get_model(args).cuda()
    feats = []
    for img_path in tqdm(img_files):
        try:
            img = conver_img(img_path=img_path, extract=True).cuda()
        except:
            print('failed!!!', img_path)
        feat = model(img).detach().cpu().numpy().squeeze()
        if len(feats) == 0:
            feats = feat
        else:
            feats = np.vstack((feats, feat))
    assert feats.shape[0] == labels.shape[0]
    mat_path = '{}_{}_{}_features.mat'.format('pre', args.model, args.ds)
    sio.savemat(os.path.join(class_dir, mat_path), {
        'features':feats,
        'labels': np.array(labels)
    })
    print('save features on %s' % os.path.join(class_dir, mat_path))

def parse_arg():
    parser = argparse.ArgumentParser(description='word embeddign type')
    parser.add_argument('--model', type=str, default='res101',
                        help='word embedding type: [inception, res50]')
    parser.add_argument('--ds', type=str, default='AWA2',
                        help='dataset: [AWA2, CUB, APY, SUN]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device')
    parser.add_argument('--class_num', type=int, default=50)
    parser.add_argument("--options", type=int, default=1, help="script options")
    # parser.add_argument("--ft", action="store_true", default=False, help="whethre extract with fine-tuned model")
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args
    
if __name__ == '__main__':
    args = parse_arg()
    # options: [extra_web, extra_source]
    if args.options == 0:
        extract_web(args)
    elif args.options == 1:
        extract_source(args)
    else:
        raise NotImplementedError("not implemented other options")
        
    