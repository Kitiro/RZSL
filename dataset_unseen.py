#!/usr/bin/env python
# coding=utf-8
"""
Author: Kitiro
Date: 2020-11-03 19:44:27
LastEditTime: 2020-11-05 11:39:32
LastEditors: Kitiro
Description:  将aux feature中的unseen加入到trianing
FilePath: /zzc/exp/Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/dataset.py
"""
from contextlib import ExitStack
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.io as sio
from sklearn import preprocessing as P


def screen_attr(attr_matrix, ratio):
    std = np.std(attr_matrix, axis=0)
    threshold = np.sort(std)[int(std.shape[0] * ratio)]
    attr_matrix = P.scale(attr_matrix, axis=0)
    ret = np.squeeze(attr_matrix[:, np.argwhere(std > threshold)])
    return ret

# 对feature进行距离筛选
def screen_feat(centers, feats, labels, threshold=0.5):
    features_proc = np.array([]).reshape(0, feats.shape[-1])
    labels_proc = np.array([])
    for i in np.unique(labels):
        center = centers[i]
        bat_feats = feats[labels==i]
        dis = np.linalg.norm((bat_feats-np.tile(center, (len(bat_feats),1))), axis=1, ord=2)
        idx = np.argsort(dis)[:int(threshold*len(bat_feats))]
        labels_proc = np.hstack((labels_proc, [i]*len(idx)))
        features_proc = np.vstack((features_proc, bat_feats[idx]))
        
    print('Process features on threshold %.1f. Before:' % threshold, len(feats), 'After:', len(features_proc))
    return features_proc, labels_proc.astype(int)

class DataSet(Dataset):
    def __init__(
        self,
        name="AWA2",
        generative_attribute_num=0,
        screen=False,
        screen_ratio=0.2,
        is_aux_feat=True,
    ):
        self.name = name
        self.data_dir = f"data/{name}_data"
        
        self.generative_attribute_num = generative_attribute_num
        self.screen = screen
        self.screen_ratio = screen_ratio
        # ignore this warnings
        import warnings
        warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
        
        print('dataset:', name)
        if 0:
        #if name == "CUB":
            mat = sio.loadmat(os.path.join(self.data_dir, "googlenet.mat"))
            attribute1 = P.scale(mat["original_att"], axis=0)
            #attribute1 = mat["original_att"]
            self.train_feature = mat["train_feature"]
            self.train_label = mat["train_label"].squeeze().astype(int)
            self.test_feature_unseen = mat["test_feature_unseen"]
            self.test_label_unseen = mat["test_label_unseen"].squeeze().astype(int)

            self.test_feature_seen = mat["test_feature_seen"]
            self.test_label_seen = mat["test_label_seen"].squeeze().astype(int)

                # self.train_feature = P.scale(self.train_feature, axis=1)
                # self.test_feature_unseen = P.scale(self.test_feature_unseen, axis=1)
                # self.test_feature_seen = P.scale(self.test_feature_seen, axis=1)
        else:
            mat_visual = sio.loadmat(os.path.join(self.data_dir, f"pre_res101_{name}_features.mat"))
            features, labels = (
                P.scale(mat_visual["features"], axis=0),  # our extracted_feature 
                mat_visual["labels"].astype(int).squeeze()
                )
            mat_semantic = sio.loadmat(os.path.join(self.data_dir, "att_splits.mat"))
            attribute1 = mat_semantic["att"].T 

            trainval_loc = mat_semantic["trainval_loc"].squeeze() - 1
            test_seen_loc = mat_semantic["test_seen_loc"].squeeze() - 1
            test_unseen_loc = mat_semantic["test_unseen_loc"].squeeze() - 1

            self.train_feature = features[trainval_loc]  # feature
            self.train_label = labels[trainval_loc].astype(int)  # 23527 training samples。
            
            self.train_label_id = np.unique(self.train_label)
            self.tr_cls_centroid = np.zeros((np.unique(labels).shape[0], self.train_feature.shape[1])) # 各个seen类的特征中心
            for i in self.train_label_id:
                self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i], axis=0)
            # self.train_feature, self.train_label = screen_feat(self.tr_cls_centroid, self.train_feature, self.train_label, threshold=threshold_source)
            
            self.test_feature_unseen = features[test_unseen_loc]  # 7913 测试集中的未见类
            self.test_label_unseen = labels[test_unseen_loc].astype(int)

            self.test_feature_seen = features[test_seen_loc]  # 5882  测试集中的已见类
            self.test_label_seen = labels[test_seen_loc].astype(int)

        self.attr1_dim = (attribute1.shape[0], attribute1.shape[1])

        # 是否使用外部特征
        if is_aux_feat:
            print('load aux features.')
            pre_mat_path = os.path.join('web_feat', f'pre_res101_{name}_features.mat')
           
            aux_mat_visual = sio.loadmat(pre_mat_path)
            aux_features, aux_labels = (
                P.scale(aux_mat_visual["features"], axis=0),
                aux_mat_visual["labels"].astype(int).squeeze(), # (36313, )
            )
            
            unseen_features = np.array([]).reshape(0, 2048)
            unseen_labels = np.array([])
            for i in np.unique(self.test_label_unseen):
                feats = aux_features[i==aux_labels][:100]   # 只取前100的samples，可信度较高
                unseen_labels = np.hstack((unseen_labels, [i]*len(feats)))
                unseen_features = np.vstack((unseen_features, feats))
        
            print('unseen_features shape:', unseen_features.shape)   
            self.train_feature = np.vstack((self.train_feature, unseen_features))
            self.train_label = np.hstack((self.train_label, unseen_labels)).astype(int)
            print('Intergrated feature shape:', self.train_feature.shape)
            print('Intergrated label shape:', self.train_label.shape)
            # 全部过滤
            # self.train_feature, self.train_label = screen_feat(self.train_label_id, self.tr_cls_centroid, self.train_feature, self.train_label)
        
        if self.generative_attribute_num != 0:
            attribute2 = np.load(
                f"generate_attributes/generated_attributes_glove/class_attribute_map_{self.name}.npy"
            )
            attribute2 = attribute2[:, :generative_attribute_num]
            self.attr2_dim = (attribute2.shape[0], attribute2.shape[1])
            if screen:
                attribute1 = screen_attr(attribute1, self.screen_ratio, norm=True)
                attribute2 = screen_attr(attribute2, self.screen_ratio, norm=True)
            attribute = np.hstack(
                (attribute1, attribute2)
            )  # concat generated attributes horizontally
        # only adpot original attribute
        else:
            attribute = attribute1
        self.attribute = attribute
        # self.attr_dim = attribute.shape[0]
        self.attr_dim = (attribute.shape[0], attribute.shape[1])
        self.train_att = attribute[self.train_label]  # 23527*85
        self.test_id_unseen = np.unique(self.test_label_unseen)
        self.test_att_map_unseen = attribute[self.test_id_unseen]

    # for training
    def __getitem__(self, index):

        visual_feature = self.train_feature[index]
        semantic_feature = self.train_att[index]
        label = self.train_label[index]

        return visual_feature, semantic_feature, label

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        info = "-{} dataset: \n\t-Training samples: {}\n\t-Test samples: {}\n\t-Visual Dim: {}\n\t-Attribute Dim: {}x{}".format(
            self.name,
            self.train_feature.shape[0],
            self.test_feature_seen.shape[0] + self.test_feature_unseen.shape[0],
            self.train_feature.shape[1],
            self.attr1_dim[0],
            self.attr1_dim[1],
        )
        if self.generative_attribute_num != 0:
            info += "\n\t-Total Attribute Dim: {}x{}".format(
                self.attr_dim[0], self.attr_dim[1]
            )
        return info

def get_screen_feat():
    a = DataSet(name="AWA2", generative_attribute_num=0, screen=False, screen_ratio=0.2, is_aux_feat=True)
    
if __name__ == "__main__":

    get_screen_feat()
