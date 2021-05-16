import pandas as pd
import progressbar
import time 

#初始化进度条
bar = progressbar

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/tableA.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/tableB.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/tableB_pocess.csv'
path_tmp = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/tmp.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/train_dataset.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/valid_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Structured/Beer/test_dataset.csv'

#读取预处理需要的所有表格
train_data = pd.read_csv(path_train, encoding='utf-8')
valid_data = pd.read_csv(path_valid,encoding='utf-8')
test_data = pd.read_csv(path_test,encoding='utf-8')
tableA = pd.read_csv(path_a, encoding='utf-8')
tableB = pd.read_csv(path_b, encoding='utf-8')

#转化成字符串
tableA['Beer_Name'] = tableA['Beer_Name'].map(lambda x: str(x))
tableA['Brew_Factory_Name'] = tableA['Brew_Factory_Name'].map(lambda x: str(x))
tableA['Style'] = tableA['Style'].map(lambda x: str(x))
tableA['ABV'] = tableA['ABV'].map(lambda x: str(x))
tableB['Beer_Name'] = tableB['Beer_Name'].map(lambda x: str(x))
tableB['Brew_Factory_Name'] = tableB['Brew_Factory_Name'].map(lambda x: str(x))
tableB['Style'] = tableB['Style'].map(lambda x: str(x))
tableB['ABV'] = tableB['ABV'].map(lambda x: str(x))

tableA.insert(tableA.shape[1],'text_a', 0)
tableB.insert(tableB.shape[1],'text_b', 0)
# tableA['text_a'] = tableA['Beer_Name'] + ' brewed by ' + tableA['Brew_Factory_Name'] + ' style is ' + tableA['Style'] + ' ABV is ' + tableA['ABV']
# tableB['text_b'] = tableB['Beer_Name'] + ' brewed by ' + tableB['Brew_Factory_Name'] + ' style is ' + tableB['Style'] + ' ABV is ' + tableB['ABV']
tableA['text_a'] = tableA['Beer_Name'] + ' ' + tableA['Brew_Factory_Name'] + ' ' + tableA['Style'] + ' ' + tableA['ABV']
tableB['text_b'] = tableB['Beer_Name'] + ' ' + tableB['Brew_Factory_Name'] + ' ' + tableB['Style'] + ' ' + tableB['ABV']

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
    train_dataset.iloc[i,3] = text_a.values[0]
    train_dataset.iloc[i,4] = text_b.values[0]
train_dataset.to_csv(path_train_dataset,index=0)
print(train_dataset.head())

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