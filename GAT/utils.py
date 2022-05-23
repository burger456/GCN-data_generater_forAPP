import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

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

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))


    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

