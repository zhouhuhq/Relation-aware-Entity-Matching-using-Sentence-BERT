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
def compute_score(pred_list,label_list):
    accuracy = sklearn.metrics.accuracy_score(label_list, pred_list)
    recall = sklearn.metrics.recall_score(label_list, pred_list)
    f1 = sklearn.metrics.f1_score(label_list, pred_list)
    return accuracy, recall, f1

#bert模型函数
def BertEM(path_train, path_valid, path_test, path_error,epochs_num, warmup_steps_num, evaluation_steps_num):
    #实例化进度条
    bar = progressbar
    #定义模型
    #model = SentenceTransformer('bert-large-nli-stsb-mean-tokens',device='cuda:1')
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens',device='cuda:1')
    #model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens',device='cuda:2')
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
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
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
    list_num = 40
    prefix = 'pred_list_'
    test_map = {prefix + str(i): [] for i in range(list_num)}
    label_list = []
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
        label_list.append(int(label))
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
        score = 0.20
        compute_pred(score, cos_scores, prefix, test_map)

    error_csv.to_csv(path_error, index=0)
    max_f1 = 0
    target_threshold = 0.01
    target_accuracy = 0.01
    target_recall = 0.01
    threshold = 0.20
    #循环输出各种得分结果
    for i in range(len(test_map.keys())):
        #循环计算得分
        accuracy, recall, f1 = compute_score(test_map[prefix + str(i)], label_list)
        if f1 >= max_f1:
            max_f1 = f1
            target_threshold = threshold
            target_accuracy = accuracy
            target_recall = recall
        print('The score > {} result is accuracy: {}, | recall:{}, | f1: {}'.format(round(threshold,2), accuracy, recall, f1))
        threshold += 0.02
    #输出所有结果
    print('================dataset_name==================',path_a)
    print('================threshold:{}, target_accuracy:{}, target_recall:{}, max_f1:{}'.format(target_threshold, target_accuracy, target_recall, max_f1))
    print('================train_time:{}'.format(str(end_time-start_time)))

if __name__=="__main__":
    # #Structured-Walmart-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_nums=4,warmup_steps_num=80,evaluation_steps_num=160)

    #Structured——BP-Beer
    path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_train_dataset.csv'
    path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_valid_dataset.csv'
    path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_test_dataset.csv'
    path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/BP_error_dataset.csv'
    BertEM(path_a,path_b,path_c,path_d,epochs_num=3,warmup_steps_num=20,evaluation_steps_num=40)

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
    # BertEM(path_a,path_b,path_c,epochs_num=1,warmup_steps_num=0,evaluation_steps_num=0)

    # #Structured-DBLP-ACM
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=700,evaluation_steps_num=1400)

    # #Structured-DBLP-GoogleScholar
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-GoogleScholar/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=1700,evaluation_steps_num=3000)

    # #Structured-Amazon-Google
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Amazon-Google/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=1,warmup_steps_num=60, evaluation_steps_num=120   )

    # #Structured-Walmart-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=600,evaluation_steps_num=1200)

    # #Dirty-iTunes-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/iTunes-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=30,evaluation_steps_num=60)

    # #Dirty-DBLP-ACM
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-ACM/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=700,evaluation_steps_num=1400)

    # #Dirty-DBLP-GoogleScholar
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=1700,evaluation_steps_num=3000)

    # #Dirty-Walmart-Amazon
    # path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Dirty/Walmart-Amazon/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=600,evaluation_steps_num=1200)

    # #Textual-Abt-Buy
    # path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/BP_error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,epochs_num=4,warmup_steps_num=500,evaluation_steps_num=1000)

    # #Textual-Company
    # path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Company/train_dataset.csv'
    # path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Company/valid_dataset.csv'
    # path_c = '/ssd/zhouhcData/deepmatcherData/Textual/Company/test_dataset.csv'
    # path_d = '/ssd/zhouhcData/deepmatcherData/Textual/Company/error_dataset.csv'
    # BertEM(path_a,path_b,path_c,path_d,warmup_steps_num=6000,evaluation_steps_num=20)


