import fasttext
import random
import sklearn
import os


def process_data(positive_data, negative_data, save_path="data/fasttext", filename="data.txt" ):
    '''
    format the data for fasttext
    :param save_path:
    :return:
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(positive_data, encoding='utf8') as f:
        pos=["__label__POS , "+t for t in f]
        random.shuffle(pos)
    with open(negative_data, encoding='utf8') as f:
        neg = ["__label__NEG , " + t for t in f]
        random.shuffle(neg)
    print(len(pos),len(neg))
    train_num=int(len(pos)*0.7)
    #train_num2=int(train_num)
    train_num2 = int(len(neg)*0.7)
    train_data=pos[:train_num]*6+neg[:train_num2]
    test_data=pos[train_num:]*1+neg[train_num2:]
    #sentences = pos+neg[:len(pos)]
    random.shuffle(train_data)
    random.shuffle(test_data)
    print(train_num*6,train_num2)
    print(len(train_data),len(test_data))
    with open(os.path.join(save_path, "train_data.txt"),'w',encoding='utf8') as f:
        f.writelines(train_data)
    with open(os.path.join(save_path, "test_data.txt"),'w',encoding='utf8') as f:
        f.writelines(test_data)
    return os.path.join(save_path, "train_data.txt"),os.path.join(save_path, "test_data.txt")



if __name__=='__main__':
    pos_file="data/samples/positive.txt"
    neg_file="data/samples/negative.txt"
    train_file, test_file=process_data(pos_file,neg_file)
    classifier = fasttext.supervised(train_file,'model',label_prefix='__label__', )
    result = classifier.test(train_file)
    print("平均F", result.precision)  # 准确率
    print("Number of examples:", result.nexamples)
    tp=fp=fn=tn=0
    with open("data/fasttext/train_data.txt",encoding='utf8') as f:
        for line in f:
            sl=line.split(' , ')
            plabel=classifier.predict([sl[1]])[0][0]
            rlabel=sl[0][-3:]
            #(sl[1])
            if rlabel=='POS' and plabel=='POS':
                tp+=1
            elif rlabel=='POS' and plabel=='NEG':
                fn+=1
            elif rlabel=='NEG' and plabel=='POS':
                fp+=1
            elif rlabel=='NEG' and plabel=='NEG':
                tn+=1
    print(tp,fp,fn,tn)
    precision=tp/(tp+fp)
    recall=tp / (tp + fn)
    print(precision,recall,2*precision*recall/(precision+recall))

    tp = fp = fn = tn = 0
    with open("data/fasttext/test_data.txt", encoding='utf8') as f:
        for line in f:
            sl = line.split(' , ')
            plabel = classifier.predict([sl[1]])[0][0]
            rlabel = sl[0][-3:]
            # (sl[1])
            if rlabel == 'POS' and plabel == 'POS':
                tp += 1
            elif rlabel == 'POS' and plabel == 'NEG':
                fn += 1
            elif rlabel == 'NEG' and plabel == 'POS':
                fp += 1
            elif rlabel == 'NEG' and plabel == 'NEG':
                tn += 1
    print(tp, fp, fn, tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(precision, recall, 2 * precision * recall / (precision + recall))
    result = classifier.test(test_file)
    print("平均F", result.precision)  # 准确率
    print("Number of examples:", result.nexamples)
