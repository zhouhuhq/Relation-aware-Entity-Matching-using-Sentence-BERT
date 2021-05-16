import pandas as pd
import progressbar
import time 
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
    model_path='distilbert-base-uncased', action="substitute")

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/tableA.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/tableB.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/tableB_pocess.csv'
path_tmp = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/tmp.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/train_dataset.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/valid_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/iTunes-Amazon/test_dataset.csv'

#读取预处理需要的所有表格
train_data = pd.read_csv(path_train, encoding='utf-8')
valid_data = pd.read_csv(path_valid,encoding='utf-8')
test_data = pd.read_csv(path_test,encoding='utf-8')
tableA = pd.read_csv(path_a, encoding='utf-8')
tableB = pd.read_csv(path_b, encoding='utf-8')

#转化成字符串
tableA['Song_Name'] = tableA['Song_Name'].map(lambda x: str(x))
tableA['Artist_Name'] = tableA['Artist_Name'].map(lambda x: str(x))
tableA['Album_Name'] = tableA['Album_Name'].map(lambda x: str(x))
tableA['Genre'] = tableA['Genre'].map(lambda x: str(x))
tableA['Price'] = tableA['Price'].map(lambda x: str(x))
tableA['CopyRight'] = tableA['CopyRight'].map(lambda x: str(x))
tableA['Time'] = tableA['Time'].map(lambda x: str(x))
tableA['Released'] = tableA['Released'].map(lambda x: str(x))
tableB['Song_Name'] = tableB['Song_Name'].map(lambda x: str(x))
tableB['Artist_Name'] = tableB['Artist_Name'].map(lambda x: str(x))
tableB['Album_Name'] = tableB['Album_Name'].map(lambda x: str(x))
tableB['Genre'] = tableB['Genre'].map(lambda x: str(x))
tableB['Price'] = tableB['Price'].map(lambda x: str(x))
tableB['CopyRight'] = tableB['CopyRight'].map(lambda x: str(x))
tableB['Time'] = tableB['Time'].map(lambda x: str(x))
tableB['Released'] = tableB['Released'].map(lambda x: str(x))

tableA.insert(tableA.shape[1],'text_a', 0)
tableB.insert(tableB.shape[1],'text_b', 0)
# tableA['text_a'] = tableA['Song_Name'] + ' created by ' + tableA['Artist_Name'] + ' belongs to ' + tableA['Album_Name'] + ' type is ' + tableA['Genre'] + ' cost ' + tableA['Price']+ ' copyright is ' + tableA['CopyRight'] + ' length is ' + tableA['Time'] + ' released in ' + tableA['Released']
# tableB['text_b'] = tableB['Song_Name'] + ' created by ' + tableB['Artist_Name'] + ' belongs to ' + tableB['Album_Name'] + ' type is ' + tableB['Genre'] + ' cost ' + tableB['Price']+ ' copyright is ' + tableB['CopyRight'] + ' length is ' + tableB['Time'] + ' released in ' + tableB['Released']
tableA['text_a'] = tableA['Song_Name'] + ' ' + tableA['Artist_Name'] + ' ' + tableA['Album_Name'] + ' ' + tableA['Genre'] + ' ' + tableA['Price']+ ' ' + tableA['CopyRight'] + ' ' + tableA['Time'] + ' ' + tableA['Released']
tableB['text_b'] = tableB['Song_Name'] + ' ' + tableB['Artist_Name'] + ' ' + tableB['Album_Name'] + ' ' + tableB['Genre'] + ' ' + tableB['Price']+ ' ' + tableB['CopyRight'] + ' ' + tableB['Time'] + ' ' + tableB['Released']
tableA.to_csv(path_a_pocess)
tableB.to_csv(path_b_pocess)

#处理train.csv文件
train_data.insert(train_data.shape[1],'text_a', 0)
train_data.insert(train_data.shape[1],'text_b', 0)
for i in bar.progressbar(range(len(train_data))):
    time.sleep(0.0001)
    ltable_id = train_data.iloc[i]['ltable_id']
    rtable_id = train_data.iloc[i]['rtable_id']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    train_data.iloc[i,3] = text_a.values[0]
    train_data.iloc[i,4] = text_b.values[0]
train_data.to_csv(path_train_dataset,index=0)
print(train_data.head())

#数据增强
for i in bar.progressbar(range(len(train_data))):
    ltable_id = train_data.iloc[i]['ltable_id']
    rtable_id = train_data.iloc[i]['rtable_id']
    text_a = train_data.iloc[i]['text_a']
    text_b = train_data.iloc[i]['text_b']
    augmented_text_a = aug.augment(text_a)
    augmented_text_b = aug.augment(text_b)
    label = train_data.iloc[i]['label']
    augment_data = pd.DataFrame([[ltable_id,rtable_id,label,augmented_text_a,augmented_text_b]], columns=('ltable_id','rtable_id','label','text_a','text_b'))
    train_data = train_data.append(augment_data, ignore_index=True)
train_data.to_csv(path_train_dataset,index=0)
print(len(train_data))

#处理valid.csv文件
valid_data.insert(valid_data.shape[1],'text_a', 0)
valid_data.insert(valid_data.shape[1],'text_b', 0)
for i in bar.progressbar(range(len(valid_data))):
    time.sleep(0.0001)
    ltable_id = valid_data.iloc[i]['ltable_id']
    rtable_id = valid_data.iloc[i]['rtable_id']
    text_a = tableA.loc[tableA['id']==ltable_id,'text_a']
    text_b = tableB.loc[tableB['id']==rtable_id,'text_b']
    valid_data.iloc[i,3] = text_a.values[0]
    valid_data.iloc[i,4] = text_b.values[0]
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
    test_data.iloc[i,3] = text_a.values[0]
    test_data.iloc[i,4] = text_b.values[0]
test_data.to_csv(path_test_dataset,index=0)
print(test_data.head())