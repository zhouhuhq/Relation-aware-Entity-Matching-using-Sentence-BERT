import pandas as pd
import progressbar
import time 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import os

os.environ["MODEL_DIR"] = '../model'
#初始化进度条
bar = progressbar

#增强模型
aug = naw.RandomWordAug(action="swap")
#初始化进度条
bar = progressbar

#停用词集合
stop_words = set(stopwords.words('english'))
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.add(w)

#过滤stopwords方法
def filter_stopwords(text):
    text = text.values[0]
    word_tokens = word_tokenize(text)
    filtered_list = []
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_list.append(w)
    filtered_sentence = " ".join(filtered_list)
    return filtered_sentence

#作者截断
def auto_truncate(val):
    return val[:10]

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/tableA.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/tableB.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/tableB_pocess.csv'
path_tmp = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/tmp.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/train_dataset.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/valid_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/test_dataset.csv'
path_train_DA =  '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/substitute_train_dataset.csv'
#随机选择单词的顺序进行调换位置，训练集翻倍
path_train_swap = '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/swap_train_dataset.csv'
#随机删除authors列里的单词，训练集翻倍
path_train_delete_authors =  '/ssd/zhouhcData/deepmatcherData/Structured/DBLP-ACM/delete_authors_train_dataset.csv'
#读取预处理需要的所有表格
train_data = pd.read_csv(path_train, encoding='utf-8')
valid_data = pd.read_csv(path_valid,encoding='utf-8')
test_data = pd.read_csv(path_test,encoding='utf-8')
tableA = pd.read_csv(path_a, encoding='utf-8', converters={'authors':auto_truncate})
tableB = pd.read_csv(path_b, encoding='utf-8', converters={'authors':auto_truncate})

#转化成字符串
tableA['title'] = tableA['title'].map(lambda x: str(x))
tableA['authors'] = tableA['authors'].map(lambda x: str(x))
tableA['venue'] = tableA['venue'].map(lambda x: str(x))
tableA['year'] = tableA['year'].map(lambda x: str(x))
tableB['title'] = tableB['title'].map(lambda x: str(x))
tableB['authors'] = tableB['authors'].map(lambda x: str(x))
tableB['venue'] = tableB['venue'].map(lambda x: str(x))
tableB['year'] = tableB['year'].map(lambda x: str(x))


tableA.insert(tableA.shape[1],'text_a', 0)
tableB.insert(tableB.shape[1],'text_b', 0)
tableA['text_a'] = tableA['title'] + ' - ' + tableA['year'] + ' written by ' + tableA['authors'] + ' posted in ' + tableA['venue'] + ' in ' + tableA['year']
tableB['text_b'] = tableB['title'] + ' - ' + tableB['year'] + ' written by ' + tableB['authors'] + ' posted in ' + tableB['venue'] + ' in ' + tableB['year']

#处理train.csv文件
train_data.insert(train_data.shape[1],'text_a', 0)
train_data.insert(train_data.shape[1],'text_b', 0)
for i in bar.progressbar(range(len(train_data))):
    time.sleep(0.0001)
    ltable_id = train_data.iloc[i]['ltable_id']
    rtable_id = train_data.iloc[i]['rtable_id']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    text_a = filter_stopwords(text_a)
    text_b = filter_stopwords(text_b)
    train_data.iloc[i,3] = text_a
    train_data.iloc[i,4] = text_b
train_data.to_csv(path_train_dataset,index=0)
print(train_data.head())

# #随机删除单词
# aug = naw.RandomWordAug()
# tableA.insert(tableA.shape[1], 'del_authors', 0)
# tableB.insert(tableB.shape[1], 'del_authors', 0)
# for i in bar.progressbar(range(len(tableA))):
#     authors_a = tableA.iloc[i]['authors']
#     del_authors = aug.augment(authors_a)
#     tableA.iloc[i,5] = del_authors
# for i in bar.progressbar(range(len(tableB))):
#     authors_b = tableB.iloc[i]['authors']
#     del_authors = aug.augment(authors_b)
#     tableB.iloc[i,5] = del_authors
# tableA['text_a'] = tableA['title'] + ' ' + tableA['del_authors'] + ' ' + tableA['venue'] + ' ' + tableA['year'] + ' ' + tableA['year']
# tableB['text_b'] = tableB['title'] + ' ' + tableB['del_authors'] + ' ' + tableB['venue'] + ' ' + tableB['year'] + ' ' + tableB['year']

#换位置
for i in bar.progressbar(range(len(train_data))):
    time.sleep(0.0001)
    ltable_id = train_data.iloc[i]['ltable_id']
    rtable_id = train_data.iloc[i]['rtable_id']
    label = train_data.iloc[i]['label']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    text_a = filter_stopwords(text_a)
    text_b = filter_stopwords(text_b)
    augment_data = pd.DataFrame([[ltable_id,rtable_id,label,text_a,text_b]], columns=('ltable_id','rtable_id','label','text_a','text_b'))
    train_data = train_data.append(augment_data, ignore_index=True)
train_data.to_csv(path_train_delete_authors,index=0)
print(train_data.head())

# #普通数据增强
# for i in bar.progressbar(range(len(train_data))):
#     ltable_id = train_data.iloc[i]['ltable_id']
#     rtable_id = train_data.iloc[i]['rtable_id']
#     text_a = train_data.iloc[i]['text_a']
#     text_b = train_data.iloc[i]['text_b']
#     augmented_text_a = aug.augment(text_a)
#     augmented_text_b = aug.augment(text_b)
#     label = train_data.iloc[i]['label']
#     augment_data = pd.DataFrame([[ltable_id,rtable_id,label,augmented_text_a,augmented_text_b]], columns=('ltable_id','rtable_id','label','text_a','text_b'))
#     train_data = train_data.append(augment_data, ignore_index=True)
# train_data.to_csv(path_train_swap,index=0)
# print(len(train_data))

#处理valid.csv文件
valid_data.insert(valid_data.shape[1],'text_a', 0)
valid_data.insert(valid_data.shape[1],'text_b', 0)
for i in bar.progressbar(range(len(valid_data))):
    time.sleep(0.0001)
    ltable_id = valid_data.iloc[i]['ltable_id']
    rtable_id = valid_data.iloc[i]['rtable_id']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    text_a = filter_stopwords(text_a)
    text_b = filter_stopwords(text_b)
    valid_data.iloc[i,3] = text_a
    valid_data.iloc[i,4] = text_b
valid_data.to_csv(path_valid_dataset,index=0)
print(valid_data.head())

#处理test.csv文件
test_data.insert(test_data.shape[1],'text_a', 0)
test_data.insert(test_data.shape[1],'text_b', 0)
for i in bar.progressbar(range(len(test_data))):
    time.sleep(0.0001)
    ltable_id = test_data.iloc[i]['ltable_id']
    rtable_id = test_data.iloc[i]['rtable_id']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    text_a = filter_stopwords(text_a)
    text_b = filter_stopwords(text_b)
    test_data.iloc[i,3] = text_a
    test_data.iloc[i,4] = text_b
test_data.to_csv(path_test_dataset,index=0)
print(test_data.head())