import numpy as np
import pickle
from sklearn import linear_model
import csv
import re
import os
import jieba
from pypinyin import pinyin, Style
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
import xgboost as xgb


if os.path.exists('/example_module/zh.300.vec.gz'):
    EMBEDDING_PATH = '/example_module/zh.300.vec.gz'
else:
    EMBEDDING_PATH = '/home/wangshuqi/tianchi/securityAI3_submit_demo_v2/example_module/zh.300.vec.gz'

EMBEDDING_DIM = 300
DEFAULT_KEYVEC = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=50000)

def tokenize(text):
    import jieba
    return ' '.join(jieba.cut(text))

def word2vec(word):
    return doc2vec(tokenize(word))

def doc2vec(tokenized):
    tokens = tokenized.split(' ')
    vec = np.full(EMBEDDING_DIM, 1e-10)
    weight = 1e-8
    for _token in tokens:
        try:
            vec += DEFAULT_KEYVEC.get_vector(_token)
            weight += 1.0
        except:
            pass
    return vec / weight

def batch_doc2vec(list_of_tokenized_text):
    return [doc2vec(_text) for _text in list_of_tokenized_text]

def load_dataset(filepath='dataset.csv'):
    vec = np.zeros((EMBEDDING_DIM, 1))
    label = np.zeros(1)
    with open(filepath, 'r') as file:
        content = csv.DictReader(file)
        for line in content:
            vec = np.append(vec, np.reshape(word2vec(line['word']), (300, 1)), axis=1)
            label = np.append(label, [int(line['label'])], axis=0)
    return vec[:, 1:].T, label[1:]

def train_model(data, label):
    reg = linear_model.LogisticRegression()
    reg.fit(data, label)
    with open('classify.pickle', 'wb') as f:
        pickle.dump(reg, f)
    print('success save model')

def regression_model(data, label):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=7)
    reg = linear_model.LogisticRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def xgboost_model(data, label):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=7)
    model = xgb.XGBClassifier(
        max_depth=20,
        learning_rate=0.1,
        n_estimators=2000,
        min_child_weight=5,
        max_delta_step=0,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0,
        reg_lambda=0.4,
        scale_pos_weight=0.8,
        silent=True,
        objective='binary:logistic',
        missing=None,
        eval_metric='auc',
        seed=1440,
        gamma=0)
    model.fit(data, label)
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def load_model():
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    if os.path.exists('/example_module/xgboost.model'):
        booster.load_model('/example_module/xgboost.model')
        clf._Booster = booster
        return clf
    else:
        booster.load_model('/home/wangshuqi/tianchi/tianchi/example_module/xgboost.model')
        clf._Booster = booster
        return clf

def load_model_regression():
    model = None
    if os.path.exists('/example_module/classify.pickle'):
        with open('/example_module/classify.pickle', 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        with open('/home/wangshuqi/tianchi/tianchi/example_module/classify.pickle', 'rb') as f:
            model = pickle.load(f)
        return model
      
def preprocess():
    preprocessres = ""
    REGEX_TO_REMOVE = re.compile(r"[^\u4E00-\u9FA5]")
    with open('raw.txt', 'r') as file:
        for line in file:
            preprocessres += REGEX_TO_REMOVE.sub(r'', line)
    with open('dataset.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, ['word', 'label'])
        for word in jieba.cut(preprocessres):
            writer.writerow({'word': word, 'label': 1})

def preprocess_civil():
    preprocessres = ""
    REGEX_TO_REMOVE = re.compile(r"[^\u4E00-\u9FA5]")
    with open('raw_civil.txt', 'r') as file:
        for line in file:
            preprocessres += REGEX_TO_REMOVE.sub(r'', line)
    preprocessres = preprocessres.replace("子曰", "")
    with open('dataset.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, ['word', 'label'])
        for word in jieba.cut(preprocessres):
            writer.writerow({'word': word, 'label': 0})


def load_chinese_dict():
    chinese_dict = {}
    if os.path.exists('/example_module/dict.pickle'):
        with open('/example_module/dict.pickle', 'rb') as f:
            chinese_dict = pickle.load(f)
        return chinese_dict
    else:
        with open('/home/wangshuqi/tianchi/securityAI3_submit_demo_v2/example_module/dict.pickle', 'rb') as f:
            chinese_dict = pickle.load(f)
            print('load local')
        return chinese_dict
    ### generate
    # with open('chinese3500.txt', 'r') as f:
    #     for line in f:
    #         for word in line:
    #             if word != '\n':
    #                 try:
    #                     word_pinyin = pinyin(word, style=Style.NORMAL)[0][0]
    #                     if word_pinyin not in chinese_dict:
    #                         chinese_dict[word_pinyin] = [word]
    #                     else:
    #                         chinese_dict[word_pinyin] += word
    #                 except Exception:
    #                     continue
    # return chinese_dict

if __name__ == '__main__':
    a, b = load_dataset()
    xgboost_model(a, b)
    # with open('dict.pickle', 'wb') as f:
    #     pickle.dump(load_chinese_dict(), f)
    
        