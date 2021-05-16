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

#停用词集合
stop_words = set(stopwords.words('english'))
for w in ['!',',','.','?','-s','-ly','</s>','s','nan','mac']:
    stop_words.add(w)

# aug = naw.RandomWordAug(action='crop')
aug = nas.AbstSummAug(model_path='t5-base', num_beam=3, device='cuda:2')
#过滤stopwords方法
def filter_stopwords(text):
    # text = text.values[0]
    word_tokens = word_tokenize(text)
    # filtered_list = 
    # for w in word_tokens: 
    #     if w not in stop_words: 
    #         filtered_list.append(w)
    # filtered_sentence 
    return " ".join(list(filter(lambda x: x not in stop_words, word_tokens)))

#定义路径
path_train = '/ssd/zhouhcData/deepmatcherData/Textual/Company/train.csv'
path_valid = '/ssd/zhouhcData/deepmatcherData/Textual/Company/valid.csv'
path_test = '/ssd/zhouhcData/deepmatcherData/Textual/Company/test.csv'
path_a = '/ssd/zhouhcData/deepmatcherData/Textual/Company/tableA_tok.csv'
path_b = '/ssd/zhouhcData/deepmatcherData/Textual/Company/tableB_tok.csv'
path_a_pocess = '/ssd/zhouhcData/deepmatcherData/Textual/Company/tableA_pocess.csv'
path_b_pocess = '/ssd/zhouhcData/deepmatcherData/Textual/Company/tableB_pocess.csv'
path_tmp = '/ssd/zhouhcData/deepmatcherData/Textual/Company/tmp.csv'
path_train_dataset = '/ssd/zhouhcData/deepmatcherData/Textual/Company/SummAug_test_dataset.csv'
path_train_dataset_1 = '/ssd/zhouhcData/deepmatcherData/Textual/Company/train_dataset_hwf_1.csv'
path_valid_dataset = '/ssd/zhouhcData/deepmatcherData/Textual/Company/SummAug_test_dataset.csv'
path_test_dataset = '/ssd/zhouhcData/deepmatcherData/Textual/Company/SummAug_test_dataset.csv'
path_train_crop = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_train_dataset.csv'
path_valid_crop = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_valid_dataset.csv'
path_test_crop = '/ssd/zhouhcData/deepmatcherData/Textual/Company/crop_test_dataset.csv'
#读取预处理需要的所有表格
# def auto_truncate(val):
#     return val[:100]
train_dataset = pd.read_csv(path_train_dataset, encoding='utf-8')
valid_dataset = pd.read_csv(path_valid_dataset,encoding='utf-8')
test_dataset = pd.read_csv(path_test_dataset,encoding='utf-8')
# tableA = pd.read_csv(path_a_pocess, encoding='utf-8')
# tableB = pd.read_csv(path_b_pocess, encoding='utf-8')

# #转化成字符串
# train_dataset['text_a'] = train_dataset['text_a'].map(lambda x: str(x))
# train_dataset['text_b'] = train_dataset['text_b'].map(lambda x: str(x))

#删除空值
train_dataset.dropna(axis=0, how='any', inplace=True)
valid_dataset.dropna(axis=0, how='any', inplace=True)
test_dataset.dropna(axis=0, how='any', inplace=True)
train_dataset.insert(train_dataset.shape[1], 'text_a_len', 0)
# for i in bar.progressbar(range(len(train_dataset))):
#     train_dataset.iloc[i]['text_a_len'] = len(train_dataset.iloc[i]['text_a'])
train_dataset['text_a_len'] = train_dataset['text_a'].str.len()
print(train_dataset.head())
# valid_dataset['text_a_len'] = valid_dataset['text_a'].str.len()
# test_dataset['text_a_len'] = test_dataset['text_a'].str.len()
train_dataset.to_csv(path_train_dataset, index=0)
valid_dataset.to_csv(path_valid_dataset, index=0)
test_dataset.to_csv(path_test_dataset, index=0)