import fasttext
import random
import os
from text.extractSamples import extract_samples
from text.textProcess import text_process
from xpinyin import Pinyin


def process_data(positive_data, negative_data, save_path="data/fasttext", correct=6):
    '''
    :param positive_data: positive data path
    :param negative_data: negative data path
    :param save_path: path to save train and test data
    :param correct: Correct the imbalance between positive and negative samples, default by 6
    :return:
    '''
    pin=Pinyin()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(positive_data, encoding='utf8') as f:
        pos=["__label__POS , "+pin.get_pinyin(t,'') for t in f]
        random.shuffle(pos)
    with open(negative_data, encoding='utf8') as f:
        neg = ["__label__NEG , " + pin.get_pinyin(t,'') for t in f]
        random.shuffle(neg)
    print("整体正负样本数量",len(pos),len(neg))
    train_num=int(len(pos)*0.7)
    #train_num2=int(train_num)
    train_num2 = int(len(neg)*0.7)
    train_data=pos[:train_num]*correct+neg[:train_num2]
    test_data=pos[train_num:]+neg[train_num2:]
    #sentences = pos+neg[:len(pos)]
    random.shuffle(train_data)
    random.shuffle(test_data)
    print("调整后训练的正负样本数量",train_num*correct,train_num2)
    print("训练样本数量，测试样本数量", len(train_data),len(test_data))
    with open(os.path.join(save_path, "train_data.txt"),'w',encoding='utf8') as f:
        f.writelines(train_data)
    with open(os.path.join(save_path, "test_data.txt"),'w',encoding='utf8') as f:
        f.writelines(test_data)
    return os.path.join(save_path, "train_data.txt"),os.path.join(save_path, "test_data.txt")


def validate(classifier, file_path):
    tp = fp = fn = tn = 0
    with open(file_path, encoding='utf8') as f:
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
    return 0


if __name__=='__main__':
    # process text
    #text_process(ori_path="//10.200.42.124/videos/denoise_text",save_path="data/text2")
    # extract samples
    #extract_samples(record_path = "//10.200.42.124/videos/segments",ori_path = "data/text2",save_path = "data/samples2")
    # process data
    pos_file="data/samples/positive.txt"
    neg_file="data/samples/negative.txt"
    train_file, test_file=process_data(pos_file,neg_file,correct=2)
    classifier = fasttext.supervised(train_file,'model/fasttext.model',label_prefix='__label__', )
    result = classifier.test(train_file,k=1)
    print("平均F", result.precision)  # 准确率
    print("Number of examples:", result.nexamples)
    validate(classifier, "data/fasttext/train_data.txt")
    print("--------------")
    result = classifier.test(test_file)
    print("平均F", result.precision)  # 准确率
    print("Number of examples:", result.nexamples)
    validate(classifier, "data/fasttext/test_data.txt")
