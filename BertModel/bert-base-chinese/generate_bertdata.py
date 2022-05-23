"""
产生bert

"""
import os
import numpy as np
import torch
from transformers import DistilBertModel,DistilBertConfig,BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

BASE_DIR = os.path.dirname(__file__)
model_name = 'bert-base-chinese'
MODEL_PATH = os.path.join(BASE_DIR,'bert-base-chinese')#'bert-base-chinese'

# model_name = 'distilbert-base-multilingual-cased'
# MODEL_PATH = os.path.join(BASE_DIR,'distilbert-base-multilingual-cased')#'bert-base-chinese'


"""
模型初始化
"""
def generate_model_init():
    # a. 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # b. 导入配置文件
    model_config = BertConfig.from_pretrained(model_name)
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)
    return bert_model,tokenizer


"""
生成句子表示特征
"""

def generate_bertdata(bert_model,tokenizer,str1='',str2='',str3=''):
    # 激活模型
    bert_model.eval()
    #生成表示
    sen_code = tokenizer.encode_plus(str1)
    # print(sen_code)
    # print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))
    # 对编码进行转换，以便输入Tensor
    tokens_tensor = torch.tensor([sen_code['input_ids']])  # 添加batch维度并,转换为tensor,torch.Size([1, 19])
    segments_tensors = torch.tensor(sen_code['token_type_ids'])  # torch.Size([19]

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        encoded_layers = outputs  # outputs类型为tuple

        # print(encoded_layers[0].shape, encoded_layers[1].shape,
        #       encoded_layers[2][5].shape, encoded_layers[3][0].shape)

        # 句表示
        sentence_embedding = torch.mean(encoded_layers[2][11], 1)
        # print("Our final sentence embedding vector of shape:", sentence_embedding.shape)
    return sentence_embedding#[1,768]








"""
test
"""
# # a. 通过词典导入分词器
# tokenizer = BertTokenizer.from_pretrained(model_name)
# # b. 导入配置文件
# model_config = BertConfig.from_pretrained(model_name)
# # 修改配置
# model_config.output_hidden_states = True
# model_config.output_attentions = True
# # 通过配置和路径导入模型
# bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)
#
# sen_code = tokenizer.encode_plus('/com->hello.aadsfasdf')
# print(sen_code)
# print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))
#
# # 对编码进行转换，以便输入Tensor
# tokens_tensor = torch.tensor([sen_code['input_ids']])       # 添加batch维度并,转换为tensor,torch.Size([1, 19])
# segments_tensors = torch.tensor(sen_code['token_type_ids']) # torch.Size([19]
#
# #激活模型
# bert_model.eval()
#
# with torch.no_grad():
#     outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
#     encoded_layers = outputs  # outputs类型为tuple
#
#     print(encoded_layers[0].shape, encoded_layers[1].shape,
#           encoded_layers[2][5].shape, encoded_layers[3][0].shape)
#
#     #句表示
#     sentence_embedding = torch.mean(encoded_layers[2][11], 1)
#     print("Our final sentence embedding vector of shape:",sentence_embedding.shape)
#
#




