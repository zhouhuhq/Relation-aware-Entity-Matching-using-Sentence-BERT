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

#随机删除
aug = naw.RandomWordAug()

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/tableA.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/tableB.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/tableB_pocess.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/train_dataset.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/valid_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/test_dataset.csv'
del_train_dataset =  '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/del_train_dataset.csv'
gan_train_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Walmart-Amazon/gan_train_dataset.csv'
#读取预处理需要的所有表格
train_data = pd.read_csv(path_train_dataset, encoding='utf-8')
tableA = pd.read_csv(path_a_pocess, encoding='utf-8')
tableB = pd.read_csv(path_b_pocess, encoding='utf-8')

#转化成字符串
tableA['aug_title'] = tableA['title'].map(lambda x: str(x))
tableA['category'] = tableA['category'].map(lambda x: str(x))
tableA['brand'] = tableA['brand'].map(lambda x: str(x))
tableA['modelno'] = tableA['modelno'].map(lambda x: str(x))
tableA['price'] = tableA['price'].map(lambda x: str(x))
tableB['aug_title'] = tableB['title'].map(lambda x: str(x))
tableB['category'] = tableB['category'].map(lambda x: str(x))
tableB['brand'] = tableB['brand'].map(lambda x: str(x))
tableB['modelno'] = tableB['modelno'].map(lambda x: str(x))
tableB['price'] = tableB['price'].map(lambda x: str(x))

#数据增强
for i in bar.progressbar(range(len(train_data))):
    ltable_id = train_data.iloc[i]['ltable_id']
    rtable_id = train_data.iloc[i]['rtable_id']
    text_a = train_data.iloc[i]['text_a']
    text_b = train_data.iloc[i]['text_b']
    aug_text_a = aug.augment(text_a)
    aug_text_b = aug.augment(text_b)
    label = train_data.iloc[i]['label']
    augment_data = pd.DataFrame([[ltable_id,rtable_id,label,aug_text_a,aug_text_b]], columns=('ltable_id','rtable_id','label','text_a','text_b'))
    train_data = train_data.append(augment_data, ignore_index=True)
train_data.to_csv(del_train_dataset,index=0)
print(len(train_data))
