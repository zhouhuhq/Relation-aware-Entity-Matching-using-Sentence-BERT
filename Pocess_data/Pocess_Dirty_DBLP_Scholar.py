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
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert", device='cuda:4')

extend_stopwords = []

with open('/home/zhouhc/Deep_EM/Pocess_table/english_stopwords.txt','r') as f:
    for line in f:
        extend_stopwords.append(line)
#停用词集合
stop_words = set(stopwords.words('english'))
for w in ['!',',','.','?','-s','-ly','</s>','s','nan','mac','(',')']:
    stop_words.add(w)
for w in extend_stopwords:
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

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/tableA.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/tableB.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/tableB_pocess.csv'
path_tmp = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/tmp.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/train_dataset.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/valid_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/test_dataset.csv'
path_train_DA = '/ssd/zhouhcData/deepmatcherData/Dirty/DBLP-GoogleScholar/DA_train_dataset.csv'
#读取预处理需要的所有表格
train_data = pd.read_csv(path_train, encoding='utf-8')
valid_data = pd.read_csv(path_valid,encoding='utf-8')
test_data = pd.read_csv(path_test,encoding='utf-8')
tableA = pd.read_csv(path_a, encoding='utf-8')
tableB = pd.read_csv(path_b, encoding='utf-8')

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
# tableA['text_a'] = tableA['title'] + ' written by ' + tableA['authors'] + ' published in ' + tableA['venue'] + ' in ' + tableA['year']
# tableB['text_b'] = tableB['title'] + ' written by ' + tableB['authors'] + ' published in ' + tableB['venue'] + ' in ' + tableB['year']
tableA['text_a'] = tableA['title'] + ' ' + tableA['authors'] + ' ' + tableA['venue'] + ' ' + tableA['year']
tableB['text_b'] = tableB['title'] + ' ' + tableB['authors'] + ' ' + tableB['venue'] + ' ' + tableB['year']
tableA.to_csv(path_a_pocess)
tableB.to_csv(path_b_pocess)


#处理train.csv文件
train_data.insert(train_data.shape[1],'text_a', 0)
train_data.insert(train_data.shape[1],'text_b', 0)
train_data.to_csv(path_tmp,index=0)
train_dataset = pd.read_csv(path_tmp)
for i in bar.progressbar(range(len(train_dataset))):
    time.sleep(0.0001)
    ltable_id = train_dataset.iloc[i]['ltable_id']
    rtable_id = train_dataset.iloc[i]['rtable_id']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    text_a = filter_stopwords(text_a)
    text_b = filter_stopwords(text_b)
    train_dataset.iloc[i,3] = text_a
    train_dataset.iloc[i,4] = text_b
train_dataset.to_csv(path_train_dataset,index=0)
print(train_dataset.head())

#数据增强
for i in bar.progressbar(range(len(train_dataset))):
    ltable_id = train_dataset.iloc[i]['ltable_id']
    rtable_id = train_dataset.iloc[i]['rtable_id']
    text_a = train_dataset.iloc[i]['text_a']
    text_b = train_dataset.iloc[i]['text_b']
    augmented_text_a = aug.augment(text_a)
    augmented_text_b = aug.augment(text_b)
    label = train_dataset.iloc[i]['label']
    augment_data = pd.DataFrame([[ltable_id,rtable_id,label,augmented_text_a,augmented_text_b]], columns=('ltable_id','rtable_id','label','text_a','text_b'))
    train_dataset = train_dataset.append(augment_data, ignore_index=True)
train_dataset.to_csv(path_train_DA,index=0)
print(len(train_dataset))

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