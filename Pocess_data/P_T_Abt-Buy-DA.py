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

#随机移位
aug = naw.RandomWordAug(action="swap")

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/tableA.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/tableB.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/tableB_pocess.csv'
path_tmp = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/tmp.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/train_dataset.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/valid_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Textual/Abt-Buy/test_dataset.csv'
path_
#读取预处理需要的所有表格
train_data = pd.read_csv(path_train, encoding='utf-8')
valid_data = pd.read_csv(path_valid,encoding='utf-8')
test_data = pd.read_csv(path_test,encoding='utf-8')
tableA = pd.read_csv(path_a, encoding='utf-8')
tableB = pd.read_csv(path_b, encoding='utf-8')

#转化成字符串
tableA['name'] = tableA['name'].map(lambda x: str(x))
tableA['description'] = tableA['description'].map(lambda x: str(x))
tableA['price'] = tableA['price'].map(lambda x: str(x))
tableB['name'] = tableB['name'].map(lambda x: str(x))
tableB['description'] = tableB['description'].map(lambda x: str(x))
tableB['price'] = tableB['price'].map(lambda x: str(x))


tableA.insert(tableA.shape[1],'text_a', 0)
tableB.insert(tableB.shape[1],'text_b', 0)
# tableA['text_a'] = tableA['name'] + ' the detail are ' + tableA['description'] + ' cost ' + tableA['price']
# tableB['text_b'] = tableB['name'] + ' the detail are ' + tableB['description'] + ' cost ' + tableB['price']
tableA['text_a'] = tableA['name'] + ' ' + tableA['description'] + ' ' + tableA['price']
tableB['text_b'] = tableB['name'] + ' ' + tableB['description'] + ' ' + tableB['price']
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
    augmented_text_a = aug.augment(text_a)
    augmented_text_b = aug.augment(text_b)
    train_dataset.iloc[i,3] = augmented_text_a
    train_dataset.iloc[i,4] = augmented_text_b
train_dataset.to_csv(path_train_dataset,index=0)
print(train_dataset.head())