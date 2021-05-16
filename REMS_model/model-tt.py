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
import py_entitymatching as em

#以1为正例，0为负例的计算方法，计算记录标签函数
def statistics_pred(score, label, cos_scores, prefix, test_map):
    for i in range(len(test_map.keys())):
        if cos_scores >= score:
            pred = 1
        else:
            pred = 0
        #TP
        if pred == 1 and label == 1:
            test_map[prefix + str(i)][0] += 1
        #FP
        elif pred == 1 and label == 0:
            test_map[prefix + str(i)][1] += 1
        #FN
        elif pred == 0 and label == 1:
            test_map[prefix + str(i)][2] += 1 
        #TN
        elif pred == 0 and label == 0:
            test_map[prefix + str(i)][3] += 1
        else:
            pass
        score += 0.02

#计算精确度、召回率、F1score函数
def compute_score(TP, FP, FN, TN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

#bert模型函数
def BertEM(path_train, path_valid, path_test, path_error,epochs_num, warmup_steps_num, evaluation_steps_num):
    #实例化进度条
    bar = progressbar
    #定义模型
    #model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens',device='cuda:4')
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens',device='cuda:4')
    data_type = {"text_a": str, "text_b": str}
    train_data = pd.read_csv(path_train, encoding='utf-8',dtype=data_type)
    valid_data = pd.read_csv(path_valid, encoding='utf-8',dtype=data_type)
    test_data = pd.read_csv(path_test, encoding='utf-8',dtype=data_type)

    #训练集
    train_examples = []
    for i in bar.progressbar(range(len(train_data))):
        time.sleep(0.0001)
        text_a = train_data.iloc[i]['text_a']
        text_b = train_data.iloc[i]['text_b']
        text_a = str(text_a)
        text_b = str(text_b)
        label_data = train_data.iloc[i]['label']
        label_data = float(label_data)
        train_examples.append(InputExample(texts=[text_a,text_b], label=label_data))
    print(InputExample)

    #验证集
    sentence_a = []
    sentence_b = []
    label_valid = []
    for i in bar.progressbar(range(len(valid_data))):
        time.sleep(0.0001)
        sentence1 = valid_data.iloc[i]['text_a']
        sentence2 = valid_data.iloc[i]['text_b']
        label_valid_t = valid_data.iloc[i]['label']
        label_valid_t = float(label_valid_t)
        sentence_a.append(sentence1)
        sentence_b.append(sentence2)
        label_valid.append(label_valid_t)
    #定义评估器
    #evaluator = evaluation.EmbeddingSimilarityEvaluator(sentence_a, sentence_b, label_valid)
    evaluator = evaluation.BinaryClassificationEvaluator(sentence_a, sentence_b, label_valid)
    #定义数据集，损失函数
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    train_loss = losses.CosineSimilarityLoss(model)

    #计算时间
    start_time = time.clock()
    #训练模型
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs_num, warmup_steps=warmup_steps_num,evaluator=evaluator, evaluation_steps=evaluation_steps_num, use_amp=True)
    end_time = time.clock()

    #=========================================评估过程===================================================
    #读取并把test所有属性转化成str
    test_data = pd.read_csv(path_test, encoding='utf-8')
    test_data['text_a'] = test_data['text_a'].map(lambda x: str(x))
    test_data['text_b'] = test_data['text_b'].map(lambda x: str(x))

    #循环创建预测的list字典
    list_num = 38
    prefix = 'pred_list_'
    test_map = {prefix + str(i): [] for i in range(list_num)}
    for i in range(len(test_map.keys())):
        test_map[prefix + str(i)].append(0)
        test_map[prefix + str(i)].append(0)
        test_map[prefix + str(i)].append(0)
        test_map[prefix + str(i)].append(0)
    label_list = []
    score = 0.20
    #记录错误的dataframe
    error_csv = pd.DataFrame(columns=('id','text_a','text_b','cos_scores'))
    #进入测试集测试
    for i in bar.progressbar(range(len(test_data))):
        time.sleep(0.0001)
        text_a_embedding = model.encode(test_data.iloc[i]['text_a'], convert_to_tensor=True)
        text_b_embedding = model.encode(test_data.iloc[i]['text_b'], convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(text_a_embedding, text_b_embedding)[0]
        cos_scores = cos_scores.cpu()
        #标签list
        label = test_data.iloc[i]['label']
        label = int(label)
        label_list.append(label)
        #记录下错误的数据
        if cos_scores >= 0.80:
            pred_test = 1
        else:
            pred_test = 0 
        if pred_test != label:
            error_text_a = test_data.iloc[i]['text_a']
            error_text_b = test_data.iloc[i]['text_b']
            error_cos_scores = cos_scores   
            error_csv = error_csv.append(pd.DataFrame({'id':[i],'text_a':[error_text_a],'text_b':[error_text_b],'cos_scores':[error_cos_scores]}),ignore_index=True) 
        #生成预测list
        statistics_pred(score, label, cos_scores, prefix, test_map)

    error_csv.to_csv(path_error, index=0)
    max_f1 = 0
    target_threshold = 0.01
    target_precision = 0.01
    target_recall = 0.01
    threshold = 0.20
    #循环所有列表,输出各种得分结果
    for i in range(len(test_map.keys())):
        #循环计算得分
        precision, recall, f1 = compute_score(test_map[prefix + str(i)][0], test_map[prefix + str(i)][1], test_map[prefix + str(i)][2], test_map[prefix + str(i)][3])
        if f1 >= max_f1:
            max_f1 = f1
            target_threshold = threshold
            target_precision = precision
            target_recall = recall
        print('The score > {} result is precision: {}, | recall:{}, | f1: {}'.format(round(threshold,2), precision, recall, f1))
        threshold += 0.02
    #输出所有结果
    print('================dataset_name==================',path_a)
    print('================threshold:{}, target_precision:{}, target_recall:{}, max_f1:{}'.format(target_threshold, target_precision, target_recall, max_f1))
    print('================train_time:{}'.format(str(end_time-start_time)))

if __name__=="__main__":
    # #Structured-Walmart-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_nums=4,warmup_steps_num=80,evaluation_steps_num=160)

    # # #Structured——BP-Beer
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/WordNet_synonym_BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=60,evaluation_steps_num=120)

    # #Structured-iTunes-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=30,evaluation_steps_num=60)

    # #Structured-Fodors-Zagats
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Fodors-Zagats/error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=1,warmup_steps_num=0,evaluation_steps_num=0)

    # #Structured-DBLP-ACM
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/DA_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=1500,evaluation_steps_num=300)

    # #Structured-DBLP-GoogleScholar
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/DA_BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=3400,evaluation_steps_num=1000)

    # #Structured-Amazon-Google
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/Substitute_BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=5,warmup_steps_num=1200, evaluation_steps_num=1000)

    #Structured-Walmart-Amazon
    path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_train_dataset.csv'
    path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_valid_dataset.csv'
    path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_test_dataset.csv'
    path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_error_dataset.csv'
    BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=300,evaluation_steps_num=600)

    # #Dirty-iTunes-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=0,evaluation_steps_num=2)

    # #Dirty-DBLP-ACM
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=500,evaluation_steps_num=1000)

    # #Dirty-DBLP-GoogleScholar
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/test_dataset.csv'
    # BertEM(path_a,path_b,path_c,epochs_num=6,warmup_steps_num=220,evaluation_steps_num=400)

    # #Dirty-Walmart-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/test_dataset.csv'
    # BertEM(path_a,path_b,path_c,warmup_steps_num=80,evaluation_steps_num=160)

    # #Textual-Abt-Buy
    # path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/test_dataset.csv'
    # BertEM(path_a,path_b,path_c,epochs_num=4,warmup_steps_num=500,evaluation_steps_num=1000)

    # #Textual-Company
    # path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Company/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Company/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Textual/Company/test_dataset.csv'
    # BertEM(path_a,path_b,path_c,warmup_steps_num=900,evaluation_steps_num=0)
