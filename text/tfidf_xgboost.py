from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score,roc_auc_score,recall_score,f1_score,accuracy_score
import numpy as np


'''
tf-idf + xgboost 对文本进行分类。 tf-idf 直接提取文本特征
'''

def get_matrix(pos_path="data/samples/positive.txt", neg_path="data/samples/negative.txt"):
    dataset=[]

    with open(pos_path, encoding='utf8') as f:
        dataset+=[line.split() for line in f if line!='\n']
        pos_len=len(dataset)
        print("positive matrix length",pos_len)
    with open(neg_path, encoding='utf8') as f:
        dataset += [line.split() for line in f if line!='\n']
        neg_len=len(dataset)-pos_len
        print("negative matrix length",neg_len)
    dct=Dictionary(dataset)
    print("dictionary length",len(dct))
    corpus = [dct.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)
    pos_matrix = np.zeros((pos_len, len(dct)))
    neg_matrix = np.zeros((neg_len, len(dct)))
    for i, line in enumerate(model[corpus][:pos_len]):
        for j, n in line:
            pos_matrix[i, j] = n
    for i, line in enumerate(model[corpus][pos_len:]):
        for j, n in line:
            neg_matrix[i, j] = n
    print("get matrix completed")
    return pos_matrix,neg_matrix


def get_matrix_pinyin(pos_path="data/samples/positive.txt", neg_path="data/samples/negative.txt"):
    from xpinyin import Pinyin
    dataset=[]
    pin = Pinyin()
    with open(pos_path, encoding='utf8') as f:
        dataset += [pin.get_pinyin(line, '').split() for line in f if line != '\n']
        pos_len = len(dataset)
        print("positive matrix length", pos_len)
    with open(neg_path, encoding='utf8') as f:
        dataset += [pin.get_pinyin(line, '').split() for line in f if line != '\n']
        neg_len = len(dataset) - pos_len
        print("negative matrix length", neg_len)
    dct=Dictionary(dataset)
    print("dictionary length",len(dct))
    corpus = [dct.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)
    pos_matrix = np.zeros((pos_len, len(dct)))
    neg_matrix = np.zeros((neg_len, len(dct)))
    for i, line in enumerate(model[corpus][:pos_len]):
        for j, n in line:
            pos_matrix[i, j] = n
    for i, line in enumerate(model[corpus][pos_len:]):
        for j, n in line:
            neg_matrix[i, j] = n
    print("get matrix completed")
    return pos_matrix,neg_matrix


def train_test_split(pos_matrix, neg_matrix, train_prop=0.7, repeats=1):
    '''
    repeats:训练集中，正例的复制倍数
    '''
    import numpy as np
    #shuffle
    np.random.shuffle(pos_matrix)
    np.random.shuffle(neg_matrix)
    neg_matrix=neg_matrix[:len(pos_matrix)]
    #split data
    pos_train_size=int(pos_matrix.shape[0]*train_prop)
    neg_train_size=int(neg_matrix.shape[0]*train_prop)

    train_pos=pos_matrix[:pos_train_size]
    train_neg=neg_matrix[:neg_train_size]
    #train_pos=np.tile(train_pos,(repeats,1))
    test_pos=pos_matrix[pos_train_size:]
    test_neg=neg_matrix[neg_train_size:]
    #test_pos=np.tile(test_pos,(6,1))

    x_train=np.vstack((train_pos,train_neg))
    y_train=np.vstack((np.ones((len(train_pos),1)),np.zeros((len(train_neg),1))))
    x_test=np.vstack((test_pos,test_neg))
    y_test=np.vstack((np.ones((len(test_pos),1)),np.zeros((len(test_neg),1))))
    #train=np.hstack((x_train,y_train))
    #np.random.shuffle(train)
    #x_train=train[:,:-1]
    #y_train=train[:,-1:]
    return x_train, y_train, x_test, y_test


def get_metrics(pre_y,y):
    auc_score = roc_auc_score(y,pre_y)
    pre_score = precision_score(y,pre_y)
    rec_score=recall_score(y,pre_y)
    f_score=f1_score(y,pre_y)
    acc_score=accuracy_score(y,pre_y)
    print("xgb_auc_score:",auc_score)
    print("xgb_pre_score:",pre_score)
    print("xgb_rec_score:",rec_score)
    print("xgb_f1_score:",f_score)
    print("acc_scure:",acc_score)
    return auc_score, pre_score, rec_score, f_score,acc_score


pos_matrix, neg_matrix=get_matrix(pos_path="nb/samples/pos_a.txt",neg_path="nb/samples/n.txt")
#pos_matrix, neg_matrix=get_matrix()
x_train, y_train, x_test, y_test=train_test_split(pos_matrix, neg_matrix, train_prop=0.7, repeats=1)

xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
print("xgboost training completed")
pre_train = xgbc.predict(x_train)
get_metrics(pre_train,y_train)
print("----------")
pre_test = xgbc.predict(x_test)
get_metrics(pre_test,y_test)