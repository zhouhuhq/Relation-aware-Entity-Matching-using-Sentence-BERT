import os
os.environ["MODEL_DIR"] = '../model'
import sklearn.datasets
import re

import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw

def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)

# Load sample data
train_data = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
train_x = train_data.data

# Tokenize input
train_x_tokens = [_tokenizer(x) for x in train_x]

# Train TF-IDF model
tfidf_model = nmw.TfIdf()
tfidf_model.train(train_x_tokens)
tfidf_model.save('.')

# Load TF-IDF augmenter
aug = naw.TfIdfAug(model_path='.', tokenizer=_tokenizer)

texts = [
    'The quick brown fox jumps over the lazy dog',
    'asdasd test apple dog asd asd'
]

for text in texts:
    augmented_text = aug.augment(text)
    
    print('-'*20)
    print('Original Input:{}'.format(text))
    print('Agumented Output:{}'.format(augmented_text))