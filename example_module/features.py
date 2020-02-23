import jieba
import numpy as np
from pypinyin import pinyin, Style
try:
    from example_module.classifier import word2vec, load_model, EMBEDDING_DIM, load_chinese_dict, batch_doc2vec, load_model_regression
except Exception:
    from classifier import word2vec, load_model, EMBEDDING_DIM, load_chinese_dict, batch_doc2vec, load_model_regression
import random
from collections import OrderedDict

chinese_dict = load_chinese_dict()


def tokenize(text):
    return ' '.join(jieba.cut(text))

def replaceToken(src):
    print(pinyin(src, style=Style.NORMAL))
    tgt = ""
    for c in src:
        tgt += c
    return tgt

def batch_embedding_cosine_distance(text_list_a, text_list_b):
    '''
    Compute embedding cosine distances in batches.
    '''
    import numpy as np
    embedding_array_a = np.array(batch_doc2vec(text_list_a))
    embedding_array_b = np.array(batch_doc2vec(text_list_b))
    norm_a = np.linalg.norm(embedding_array_a, axis=1)
    norm_b = np.linalg.norm(embedding_array_b, axis=1)
    cosine_numer = np.multiply(embedding_array_a, embedding_array_b).sum(axis=1)
    cosine_denom = np.multiply(norm_a, norm_b)
    cosine_dist = 1.0 - np.divide(cosine_numer, cosine_denom)
    return cosine_dist.tolist()

def civil(text):
    # translator = Translator()
    res = ""
    model = load_model()
    for word in jieba.cut(text):
        if int(round(model.predict(np.reshape(word2vec(word), (1, EMBEDDING_DIM)))[0])) == 1:
            word_pinyin = pinyin(word, style=Style.NORMAL)
            trans = [ ''.join(item) for item in word_pinyin]
            for item in trans:
                if item in chinese_dict:
                    try:
                        wait_list = chinese_dict[item]
                        distance_dict = {}
                        distances = batch_embedding_cosine_distance([word], wait_list)
                        for idx in range(len(wait_list)):
                            distance_dict[wait_list[idx]] = distances[idx] 
                        distances = sorted(distance_dict, key=distance_dict.get, reverse=True)
                        l = len(wait_list)
                        if l < 3:
                            res += distances[0]
                        elif l < 5:
                            res += distances[random.randint(0,2)]
                        elif l < 10:
                            res += distances[random.randint(0,4)]
                        else:
                            res += distances[random.randint(0,7)]
                    except Exception:
                        res += word
                else:
                    res += word
        else:
            res += word
    return res

def hardcode(text):
    text = text.lower()
    hard_rules = ['sb', 'nmsl', 'wsnd', 'zz', 'mmp', 'wtf', 'fuck', 'nt', 'md', 'nmd', 'nm', 'nc', 'mlgb', 'tm']
    for rule in hard_rules:
        text = text.replace(rule, '')
    text = text.replace('婊', '表')
    text = text.replace('脑残', 'nc')
    text = text.replace('狗', '苟')
    text = text.replace('鸡', 'J')
    text = text.replace('妈', '吗')
    text = text.replace('骨灰', '骨辉')
    text = text.replace('坟', '逢')
    text = text.replace('妈', '吗')
    text = text.replace('屄', 'b')
    text = text.replace('肏', '艹')
    text = text.replace('贱', '建')
    text = text.replace('骚', '搔')
    text = text.replace('杂种', '崽种')
    text = text.replace('干死', '仠死')
    text = text.replace('王八', '王扒')
    text = text.replace('下葬', '吓葬')
    return text

if __name__ == '__main__':
    import csv
    with open('dataset.csv') as file:
        reader = csv.DictReader(file)
        for line in reader:
            if int(line['label']) == 0:
                break
            print(hardcode(civil(line['word'])))