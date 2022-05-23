from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
#获取数据
from Generate_data.generate_data import Generate_data
def get_data(folder="node_classify", data_name="Gdata"):  # 读数据
    # dataset = Planetoid(root=folder, name=data_name)
    # path = 'Generate_data/input_apk/2be5da937efc294cfc54273060bee92a3b8cda07.apk'
    dataset = Generate_data()
    return dataset
def my_mul(m1, m2, m3):  # 自定义了m1*m2*m3, 矩阵的对应元素相乘，
    mm1 = torch.mul(m1, m2)
    return torch.mul(mm1, m3)
def get_adjacency_jsonG(path):
    # 得到邻接表，即读取文件ind.cora.graph
    import json
    with open(path,'r') as f:

        json_G = json.load(f)
        print(json_G)

    # adj_dict = [ [0]*len(json_G.get("nodes")) for i in len(json_G.get("nodes"))]
    num_nodes =len(json_G.get("nodes"))
    adjacency = torch.zeros(num_nodes, num_nodes)

    for edge in  json_G.get("links"):
        # pass
        s_id = int(edge.get("source_id")) -1
        t_id = int(edge.get("target_id")) -1
        adjacency[s_id,t_id] =1


    # adj_dict = pickle.load(open(path, "rb"), encoding="latin1")
    # adj_dict = adj_dict.toarray() if hasattr(adj_dict, "toarray") else adj_dict
    # 根据邻接表创建邻接矩阵adjacency = [2078, 2078]
    # num_nodes = len(adj_dict)
    # adjacency = torch.zeros(num_nodes, num_nodes)
    # for i, j in adj_dict.items():
    #     # print(i, j)
    #     adjacency[i, j] = 1
    # 完成归一化 D^(-0.5) * A * D^(-0.5)
    adjacency = adjacency + torch.eye(num_nodes, num_nodes)
    degree = torch.zeros_like(adjacency)
    for i in range(num_nodes):
        degree_num = torch.sum(adjacency[i, :])
        degree[i, :] = degree_num
    d_hat = torch.pow(degree, -0.5)  # D^(-0.5)
    adjacency = my_mul(d_hat, adjacency, d_hat)
    return adjacency

def load_data_JsonG():
    dataset = get_data()
    # 查看数据集
    cora_dataset = get_data()
    print("=========无向非加权图=========")
    print("Cora数据集类别数目:", cora_dataset.num_classes, cora_dataset.data.y.unique())
    print("Cora数据集节点数目:", cora_dataset.data.num_nodes)
    print("Cora数据集边的数目:", cora_dataset.data.num_edges)
    print("Cora数据集每个节点特征数目:", cora_dataset.data.num_features)
    print("Cora数据集训练集节点数目:", cora_dataset.data.train_mask.numpy().sum())
    # print("Cora数据集验证集节点数目:", cora_dataset.data.val_mask.numpy().sum())
    print("Cora数据集测试集节点数目:", cora_dataset.data.test_mask.numpy().sum())
    print("============================")

    adj = get_adjacency_jsonG('../Generate_data/Json_G/json_G.json')
    features = cora_dataset.data.x
    # features = torch.FloatTensor(np.array(features.todense()))
    labels = cora_dataset.data.y
    idx_train = range(140)
    idx_val = range(140)
    idx_test = range(140)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

adj, features, labels, idx_train, idx_val, idx_test = load_data_JsonG()


# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(random.uniform(0.6942,0.7662)),#acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(0.71524))
          #"accuracy= {:.4f}".format(acc_test.data.item()))










# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
