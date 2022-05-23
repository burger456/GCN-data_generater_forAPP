"""

"""
import os

import torch
from torch_geometric.data import InMemoryDataset,download_url
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#import gexf
import networkx as nx
from androguard.core.bytecodes import apk
from androguard.core.bytecodes import dvm
from androguard.core.analysis import analysis
from androguard.misc import AnalyzeAPK
import sys,time,datetime,traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np





"""
重新编码
"""
def encodedMethod_to_string(encodedMethod):
    try:
        # 当函数为内部函数时，获取函数所属类、函数名称、偏移量
        return "%s->%s %s" % (encodedMethod.get_class_name(), encodedMethod.get_name(), encodedMethod.get_code_off())
    except:
        # 当函数为外部函数时，获取函数所属类、函数名称
        return "%s->%s" % (encodedMethod.get_class_name(), encodedMethod.get_name())
"""
获取调用图
"""
def get_apk_graph(a,d,x):
    # pass
    nodes_list=[]
    edges_list=[]
    EG = nx.MultiDiGraph()#定义有向图
    class_list = d.get_classes()
    # print("class_list:",class_list)
    pbar = tqdm(class_list)
    class_list_final = []
    methods_list_final = []
    #获取类和方法
    for class_i in pbar:
        class_name = class_i.name
        methods = class_i.get_methods()


        #暂时不过滤
        if class_name not in class_list_final:
            class_list_final.append(class_name)

        #获取方法
        if methods not in methods_list_final:
            methods_list_final.append(methods)#加入的是list

        #打印进度条
        pbar.set_postfix_str(\
            "Read_class_name:"+class_name)
        #依据方法过滤
        # for method in methods:
        #     raw_code_list = [x for x in method.get_instructions()]
        #     for code_line in raw_code_list:
        #         output_line = code_line.get_output()
        #         #过滤规则
        #         if '???' in output_line:
        #             pass
    pbar.close()
    #生成图
    pbar = tqdm(class_list_final)
    for class_i in pbar:
        EG = x.get_call_graph(class_i)
        G = nx.MultiDiGraph()

        for node in EG.nodes:
            node = encodedMethod_to_string(node)
            nodes_list.append(node)
        for edge in EG.edges:
            edge = (encodedMethod_to_string(edge[0]), encodedMethod_to_string(edge[1]))
            edges_list.append(edge)

        G.add_nodes_from(nodes_list)
        G.add_edges_from(edges_list)
        pbar.set_postfix_str("生成调用图："+class_i)

    pbar.close()
    return G

"""
画图
"""
def plot_G(G):
    labels = {}
    pbar = tqdm(G.nodes())
    for node in pbar:
        labels[node] = node
        pbar.set_postfix_str("画图标记："+node)
    pos = nx.spring_layout(G)                     # 生成节点位置信息
    nx.draw_networkx_nodes(G, pos, node_size=1, node_color='green', node_shape='s', alpha=1)
    nx.draw_networkx_edges(G, pos)# 画边
    # nx.draw_networkx_labels(G, pos, labels, font_size=1)       # 画标签
    plt.show()

"""
读取文件
"""
def read_apk_to_gexf(path):

    # data = nx.read_gexf(path)
    # data = nx.read_gml(path)
    a ,d,x =AnalyzeAPK(path)
    a = apk.APK(path)
    d = dvm.DalvikVMFormat(a.get_dex())
    data = get_apk_graph(a,d,x)
    # plot_G(data)
    # data = Gexf(path)
    # print(data)
    # for node in data.nodes:
    #     print(node)

    return data
"""
生成json图
"""
def generate_json_G(data):
    json_G ={}
    json_G['nodes'] = []
    json_G['links'] = []
    node_count = 1#从一开始
    for node in data.nodes:
        node_dic ={}
        node_dic["id"] = str(node_count)
        node_dic["classname"] = node.split('->')[0]
        method = node.split('->')[1]
        node_dic["method"] = (method.split(' ')[0]  if len(method.split(' ')) >1 else method)
        node_dic["OUT_num"] = (method.split(' ')[1]  if len(method.split(' ')) >1 else '')
        json_G['nodes'].append(node_dic)#加入
        node_count +=1

    for link in data.edges:
        link_dic ={}
        for node in json_G.get("nodes"):
            source = link[0]
            target = link[1]
            get_num = 0
            node_lable =node.get("classname")+"->"+node.get("method")+ ('' if node.get("OUT_num") == '' else ' '+node.get("OUT_num"))
            #获取source
            if node_lable== source:
                link_dic["source_id"] = node.get("id")
                link_dic["source"] = node_lable
                get_num +=1

            if node_lable == target:
                link_dic["target_id"] = node.get("id")
                link_dic["target"] = node_lable
                get_num +=1
            if get_num ==2 :
                link_dic["value"] = str(link[2])
                break
        json_G['links'].append(link_dic)#加入
        # print("link:",link)
    # print("link",json_G.get("links"))

    json_G = json.dumps(json_G,ensure_ascii=False,indent=4)#美观缩进
    # print(json_G)
    return json_G

"""
存json——G
"""
def save_json_G(json_G,save_path):
    with open(save_path,'w') as f:

        f.write(json_G)

"""
加载json
"""
def load_json_G(path):
    with open(path,'r') as f:

        json_G = json.load(f)
        print(json_G)
    return json_G

"""
加载json——G为图data
"""
def load_json_G_to_Gdata(json_G):

    json_G = json.loads(json_G)#使用loads才能正常加载
    # print(json_G)
    #特征矩阵
    X = []
    #图形状
    edge_index=[]
    pbar = tqdm(json_G.get("links"))
    edge_index_i_source = []
    edge_index_i_edge = []
    for link in pbar:
        # edge_index_i_source = []
        # edge_index_i_edge = []
        source_id = int(link.get("source_id"))
        target_id = int(link.get("target_id"))
        edge_index_i_source.append(source_id)
        edge_index_i_edge.append(target_id)
        # edge_index.append(edge_index_i)
        pbar.set_postfix_str("读取json_G")
    edge_index.append(edge_index_i_source)
    edge_index.append(edge_index_i_edge)
    # print("edge_num:",len(json_G.get("links")))
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    pbar.close()
    #生成特征
    # methods = []
    # X = [0 for i in range(len(json_G.get("nodes")))]  # 初始化
    # for node in json_G.get("nodes"):
    #     method = node.get("method")
    #     if method not in methods:
    #         methods.append(method)
    #         #生成特征
    #         index_i = int(node.get("id"))
    #         X[index_i-1] =[len(methods) -1]
    #     elif method in methods:
    #         #生成特征
    #         index_i = int(node.get("id"))
    #         X[index_i - 1] = [methods.index(method)]

    ###生成bert特征
    from BertModel.generate_bertdata import generate_bertdata,generate_model_init
    bert_model,tokenizer = generate_model_init()    #初始化模型
    X = [0 for i in range(len(json_G.get("nodes")))]  # 初始化
    pbar = tqdm(json_G.get("nodes"))
    for node in pbar:
        method = node.get("method")
        classname = node.get("classname")
        method = str(method)
        classname = str(classname)
        str1 =  classname+'->'+method
        index_i = int(node.get("id"))
        sentence_embedding =  generate_bertdata(bert_model,tokenizer,str1=str1)
        X[index_i -1 ] = sentence_embedding.numpy().T.flatten()#我进行了转numpy，[1,768]转置[768,1]然后展平为[768]每个节点768特征
        pbar.set_postfix_str("生成bert特征")
    pbar.close()
    X = torch.tensor(X,dtype=torch.float)

    #生成标签
    # 格式为[[node_num,target]]
    Y = torch.tensor([0 if i%2==0 else 1 for i in range(len(json_G.get("nodes")))],dtype=torch.long)

    #生成tranmask训练集 true的为已知而false为未知需要学习或无效
    train_mask = torch.zeros(Y.size(0),dtype=torch.bool)
    for i in range(int(len(json_G.get("nodes"))/2)  ):
        # train_mask[(Y == i).nonzero(as_tuple=False)[0]] =True
        train_mask[i] = True

    #生成测试集
    test_mask = torch.zeros(Y.size(0), dtype=torch.bool)
    test_mask.fill_(False)
    # test_mask[] = True

    data = Data(x=X,edge_index=edge_index,y=Y,train_mask=train_mask,test_mask=test_mask)
    # print(data.num_nodes)
    # print(data.num_node_features)

    return data

"""
存储data-会导致apk被覆盖
"""
def generate_Gdata_to_save(data,save_path):
    data_list = [data]
    data,slices = InMemoryDataset.collate(data_list)
    torch.save((data,slices),path)

"""
data类
"""
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.url = ''
        self.name = 'Gdata'
        self.BASE_DIR = Path(__file__).resolve().parent.parent




    @property
    def raw_file_names(self):
        return ['Gdata.x', 'Gdata.tx', 'Gdata.allx','Gdata.y',
                'Gdata.ty','Gdata.ally','Gdata.graph','Gdata.test.index']

    @property
    def processed_file_names(self):#这个会自动执行读出
        return ['Gdata.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(self.url, self.raw_dir)
    #     ...

    def process(self):
        # Read data into huge `Data` list.
        # path = 'Generate_data/input_apk/2be5da937efc294cfc54273060bee92a3b8cda07.apk'
        # data = read_apk_to_gexf(path)
        # json_G = generate_json_G(data)
        # #save jsonG
        # save_path = 'Generate_data/Json_G/json_G.json'
        # save_json_G(json_G, save_path)

        import glob
        # print(Path(__file__).parent.parent)
        path = 'Generate_data/input_apk'
        apk_list = glob.glob(path + "/*.apk")
        data_list_temp =[]
        #读取
        for apk in apk_list:
            print(apk)
            data = read_apk_to_gexf(apk)
            json_G = generate_json_G(data)
            # save jsonG
            save_path = 'Generate_data/Json_G/json_G.json'
            save_json_G(json_G, save_path)
            data = load_json_G_to_Gdata(json_G)
            data_list_temp.append(data)
        #生成data
        data_list = data_list_temp





        #生成data
        # data = load_json_G_to_Gdata(json_G)
        # data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Gdata saved!")



    def __repr__(self) -> str:
        return f'{self.name}()'



"""
main返回数据集
"""
def Generate_data(save_flage=False):
    # apk_path ='Generate_data/input_apk/2be5da937efc294cfc54273060bee92a3b8cda07.apk'
    # data = read_apk_to_gexf(apk_path)
    # json_G = generate_json_G(data)
    #
    # if save_flage ==True:
    #     save_path = 'Json_G/json_G.json'
    #     save_json_G(json_G, save_path)
    #
    # data = load_json_G_to_Gdata(json_G)
    # # save_path = "./Gdata.pt"
    b = MyOwnDataset('Gdata')
    # b.process()

    GDATA = b.data
    print("图的节点",GDATA.num_nodes)
    print("图的类",b.num_classes)
    print("图的边数量",GDATA.num_edges, GDATA.is_directed())
    print(GDATA)
    return b


"""
test
"""
#
# path = 'input_apk/2be5da937efc294cfc54273060bee92a3b8cda07.apk'
# with open(path) as f:
#     print(f)
# data = read_apk_to_gexf(path)
# json_G = generate_json_G(data)
# # save_path ='./2be5da937efc294cfc54273060bee92a3b8cda07.json'
# # save_json_G(json_G,save_path)
# # load_json_G(save_path)
#
# data = load_json_G_to_Gdata(json_G)
# save_path="./Gdata.pt"
# # generate_Gdata_to_save(data,save_path)
# b = MyOwnDataset('Gdata')
# b.process()
#
# print(b.data.num_nodes)
# print(b.num_classes)
# print(b.data.num_edges,b.data.is_directed())
#
# loader = DataLoader("Gdata")
#
# print(loader.dataset)

# import glob
# train_path = 'input_apk'
# xml_list = glob.glob(train_path + "/*.apk")
# print(xml_list)
# with open(xml_list[0],'r') as f:
#     print(f)























