from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
import pandas as pd
import progressbar
import time 
import sklearn.metrics
import torch
from pandas import Series,DataFrame
import pandas as pd
import editdistance

#实例化进度条
bar = progressbar

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

#计算记录标签函数
def compute_pred(score, cos_scores, prefix, test_map):
    for i in range(len(test_map.keys())):
        if cos_scores >= score:
            pred = 1
        else:
            pred = 0
        test_map[prefix + str(i)].append(int(pred))
        score += 0.02

#计算精确度、召回率、F1score函数
def compute_score(label_list,pred_list):
    precision = sklearn.metrics.precision_score(label_list, pred_list)
    recall = sklearn.metrics.recall_score(label_list, pred_list,average='binary')
    f1 = sklearn.metrics.f1_score(label_list, pred_list,average='binary')
    return precision, recall, f1

model.to(device)
model.train()
path_train = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/train_dataset.csv'
#     path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/valid_dataset.csv'
#     path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/test_dataset.csv'
#定义模型
data_type = {"text_a": str, "text_b": str}
train_data = pd.read_csv(path_train, encoding='utf-8',dtype=data_type)
# valid_data = pd.read_csv(path_valid, encoding='utf-8',dtype=data_type)
# test_data = pd.read_csv(path_test, encoding='utf-8',dtype=data_type)

#训练集
train_examples = []
for i in bar.progressbar(range(len(train_data))):
    time.sleep(0.0001)
    text_a = train_data.iloc[i]['text_a']
    text_b = train_data.iloc[i]['text_b']
    text_a = str(text_a)
    text_b = str(text_b)
    text_ab = '[CLS]' + text_a + '[SEP]' + text_b + '[SEP]'
    label_data = train_data.iloc[i]['label']
    label_data = float(label_data)
    train_examples.append(InputExample(texts=[text_ab], label=label_data))
print(InputExample)

#定义数据集，损失函数
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

optim = AdamW(model.parameters(), lr=3e-5)

for epoch in range(3):
    for batch in train_dataloader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
model.eval()

#     #读取并把test所有属性转化成str
#     test_data = pd.read_csv(path_test, encoding='utf-8')
#     test_data['text_a'] = test_data['text_a'].map(lambda x: str(x))
#     test_data['text_b'] = test_data['text_b'].map(lambda x: str(x))

#     #循环创建预测的list字典
#     list_num = 40
#     prefix = 'pred_list_'
#     test_map = {prefix + str(i): [] for i in range(list_num)}
#     label_list = []
#     score = 0.20
#     error_csv = pd.DataFrame(columns=('id','text_a','text_b','cos_scores'))
#     for i in bar.progressbar(range(len(test_data))):
#         time.sleep(0.0001)
#         text_a_embedding = model.encode(test_data.iloc[i]['text_a'], convert_to_tensor=True)
#         text_b_embedding = model.encode(test_data.iloc[i]['text_b'], convert_to_tensor=True)
#         extend_score = 0
#         cos_scores = util.pytorch_cos_sim(text_a_embedding, text_b_embedding)[0]
#         cos_scores = cos_scores.cpu()
#         cos_scores = cos_scores + extend_score
#         #标签list
#         label = test_data.iloc[i]['label']
#         label_list.append(int(label))
#         #记录下错误的数据
#         if cos_scores >= 0.70:
#             pred_test = 1
#         else:
#             pred_test = 0 
#         if pred_test != label:
#             error_text_a = test_data.iloc[i]['text_a']
#             error_text_b = test_data.iloc[i]['text_b']
#             error_cos_scores = cos_scores   
#             error_csv = error_csv.append(pd.DataFrame({'id':[i],'text_a':[error_text_a],'text_b':[error_text_b],'cos_scores':[error_cos_scores]}),ignore_index=True) 
#         #生成预测list
#         compute_pred(score, cos_scores, prefix, test_map)

#     error_csv.to_csv(path_error, index=0)
#     max_f1 = 0
#     target_threshold = 0.01
#     target_precision = 0.01
#     target_recall = 0.01
#     threshold = 0.20
#     #循环输出各种得分结果
#     for i in range(len(test_map.keys())):
#         #循环计算得分
#         precision, recall, f1 = compute_score(label_list, test_map[prefix + str(i)])
#         if f1 >= max_f1:
#             max_f1 = f1
#             target_threshold = threshold
#             target_precision = precision
#             target_recall = recall
#         print('The score > {} result is precision: {}, | recall:{}, | f1: {}'.format(round(threshold,2), precision, recall, f1))
#         threshold += 0.02
#     #输出所有结果
#     print('================dataset_name==================',path_a)
#     print('================threshold:{}, target_precision:{}, target_recall:{}, max_f1:{}'.format(target_threshold, target_precision, target_recall, max_f1))
#     print('================train_time:{}'.format(str(end_time-start_time)))

# if __name__=="__main__":
#     # #Structured——BP-Beer
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=20,evaluation_steps_num=40)

#     # #Structured-iTunes-Amazon
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=6,warmup_steps_num=0,evaluation_steps_num=4)

#     # #Structured-Fodors-Zagats
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/test_dataset.csv'
#     # BertEM(path_a,path_b,path_c,epochs_num=1,warmup_steps_num=0,evaluation_steps_num=0)

#     # #Structured-DBLP-ACM
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/Swap_BP_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=200,evaluation_steps_num=1000)

#     # #Structured-DBLP-GoogleScholar
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/Swap_BP_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=5,warmup_steps_num=1700,evaluation_steps_num=5000)

#     # #Structured-Amazon-Google
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/DA_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=1400, evaluation_steps_num=2000)

#     #Structured-Walmart-Amazon
#     path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/train_dataset.csv'
#     path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/valid_dataset.csv'
#     path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/test_dataset.csv'
#     path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/error_dataset.csv'
#     BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=600,evaluation_steps_num=1200)

#     # #Dirty-iTunes-Amazon
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/DA_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=70,evaluation_steps_num=10)

#     # #Dirty-DBLP-ACM
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=700,evaluation_steps_num=1400)

#     # #Dirty-DBLP-GoogleScholar
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/substitute_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=3400,evaluation_steps_num=2000)

#     # #Dirty-Walmart-Amazon
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/del_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=1200,evaluation_steps_num=1600)

#     # #Textual-Abt-Buy
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=500,evaluation_steps_num=1000)

#     # #Textual-Company
#     # path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_train_dataset.csv'
#     # path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_valid_dataset.csv'
#     # path_c = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_test_dataset.csv'
#     # path_d = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_error_dataset.csv'
#     # BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=6000,evaluation_steps_num=12000)



