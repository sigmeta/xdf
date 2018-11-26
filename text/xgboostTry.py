import numpy as np
import sklearn
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score,roc_auc_score,recall_score,f1_score
from gensim.models import Word2Vec


def train_test_split(pos_matrix, neg_matrix, train_prop=0.7, repeats=6):
    '''
    repeats:训练集中，正例的复制倍数
    '''
    import numpy as np
    #shuffle
    np.random.shuffle(pos_matrix)
    np.random.shuffle(neg_matrix)
    #split data
    pos_train_size=int(pos_matrix.shape[0]*train_prop)
    neg_train_size=int(neg_matrix.shape[0]*train_prop)

    train_pos=pos_matrix[:pos_train_size]
    train_neg=neg_matrix[:neg_train_size]
    train_pos=np.tile(train_pos,(repeats,1))
    test_pos=pos_matrix[pos_train_size:]
    test_neg=neg_matrix[neg_train_size:]
    #test_pos=np.tile(test_pos,(6,1))

    x_train=np.vstack((train_pos,train_neg))
    y_train=np.vstack((np.ones((len(train_pos),1)),np.zeros((len(train_neg),1))))
    x_test=np.vstack((test_pos,test_neg))
    y_test=np.vstack((np.ones((len(test_pos),1)),np.zeros((len(test_neg),1))))
    return x_train, y_train, x_test, y_test

def get_metrics(pre_y,y):
    auc_score = roc_auc_score(y,pre_y)
    pre_score = precision_score(y,pre_y)
    rec_score=recall_score(y,pre_y)
    f_score=f1_score(y,pre_y)

    print("xgb_auc_score:",auc_score)
    print("xgb_pre_score:",pre_score)
    print("xgb_rec_score:",rec_score)
    print("xgb_f1_score:",f_score)
    return auc_score, pre_score, rec_score, f_score


def get_matrix(wv="model/word2vec/word2vec",pos_path="data/samples/positive.txt",neg_path="data/samples/negative.txt"):
    #load data
    pos_matrix=np.zeros((0,256))
    neg_matrix=np.zeros((0,256))
    model=Word2Vec.load(wv)
    print("word2vec model loaded.")
    with open(pos_path, encoding='utf8') as f:
        for line in f:
            words=list({w for w in line.split() if w in model})
            if not words:
                continue
            pos_matrix=np.vstack((pos_matrix,model.wv[words].mean(0)))
    print("positive matrix completed.")
    with open(neg_path, encoding='utf8') as f:
        for line in f:
            words=list({w for w in line.split() if w in model})
            if not words:
                continue
            neg_matrix=np.vstack((neg_matrix,model.wv[words].mean(0)))
    print("negative matrix completed.")
    return pos_matrix,neg_matrix


if __name__=='__main__':
    xgbc = XGBClassifier()
    pos_matrix, neg_matrix = get_matrix()

    x_train, y_train, x_test, y_test=train_test_split(pos_matrix, neg_matrix, train_prop=0.7,repeats=1)
    print("Data processing completed")
    xgbc.fit(x_train,y_train)
    print("xgboost training completed")
    #xgbc.save_model("model/xgb/xgb.word2vec.model")

    pre_train = xgbc.predict(x_train)
    get_metrics(pre_train,y_train)
    print("----------")
    pre_test = xgbc.predict(x_test)
    get_metrics(pre_test,y_test)