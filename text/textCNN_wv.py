import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from gensim.models import Word2Vec


#采用word2vec做固定词向量嵌入
# text CNN net
class MultiCNNText(nn.Module):
    def __init__(self, opt):
        super(MultiCNNText, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        kernel_sizes = [1, 2]
        content_convs = [ nn.Sequential(
                                nn.Conv1d(in_channels = opt["embedding_dim"],
                                        out_channels = opt["content_dim"],
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt["content_dim"]),
                                nn.ReLU(inplace=True),

                                nn.Conv1d(in_channels = opt["content_dim"],
                                        out_channels = opt["content_dim"]*2,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt["content_dim"]*2),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(kernel_size = (opt["content_seq_len"] - kernel_size*2 + 2))
                            )
            for kernel_size in kernel_sizes ]

        self.content_convs = nn.ModuleList(content_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*(opt["content_dim"])*2,opt["linear_hidden_size"]),
            #nn.Dropout(0.4, inplace=True),
            nn.BatchNorm1d(opt["linear_hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Linear(opt["linear_hidden_size"],1)
        )

    def forward(self, content):
        content_out = [content_conv(content.permute(0,2,1)) for content_conv in self.content_convs]
        conv_out = torch.cat((content_out),dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return torch.sigmoid(logits)


    def train_test_split(self, pos_matrix, neg_matrix, train_prop=0.7, repeats=6):
        '''
        split train and test set
        :param pos_matrix:
        :param neg_matrix:
        :param train_prop:
        :param repeats: 训练集中，正例的复制倍数
        :return: x_train, y_train, x_test, y_test
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
        train_pos=np.tile(train_pos,(repeats,1,1))
        test_pos=pos_matrix[pos_train_size:]
        test_neg=neg_matrix[neg_train_size:]
        #test_pos=np.tile(test_pos,(6,1))
        print(train_pos.shape,train_neg.shape)
        x_train=np.vstack((train_pos,train_neg))
        y_train=np.vstack((np.ones((len(train_pos),1)),np.zeros((len(train_neg),1))))
        x_test=np.vstack((test_pos,test_neg))
        y_test=np.vstack((np.ones((len(test_pos),1)),np.zeros((len(test_neg),1))))
        return x_train, y_train, x_test, y_test


    def get_tensor(self, wv_path, pos_sample_path, neg_sample_path):
        '''

        :param wv_path:
        :param pos_sample_path:
        :param neg_sample_path:
        :return:
        '''
        # load word2vec model
        model = Word2Vec.load(wv_path)
        print("word2vec model loaded.")

        # load text
        all_words = []
        pos_sentences = []
        neg_sentences = []
        with open(pos_sample_path, encoding='utf8') as f:
            txt = f.read().strip()
            all_words += [w for w in list(set(txt.split())) if w in model]
            pos_sentences = [[w for w in sen.split() if w in model] for sen in txt.split('\n')]
        with open(neg_sample_path, encoding='utf8') as f:
            txt = f.read().strip()
            all_words += [w for w in list(set(txt.split())) if w in model]
            neg_sentences = [[w for w in sen.split() if w in model] for sen in txt.split('\n')]
        all_words = list(set(all_words))
        all_words.sort()
        max_len = max([len(s) for s in pos_sentences + neg_sentences])
        print("最大文本长度：", max_len)

        # get matrix
        pos_matrix = np.zeros((len(pos_sentences), max_len, model.vector_size))  # word2vec维度256
        neg_matrix = np.zeros((len(neg_sentences), max_len, model.vector_size))
        for i, s in enumerate(pos_sentences):
            vec = model.wv[s]
            pos_matrix[i, :vec.shape[0], :] = vec
        for i, s in enumerate(neg_sentences):
            if not s: continue
            vec = model.wv[s]
            neg_matrix[i, :vec.shape[0], :] = vec
        print("get matrix completed.")

        x_train, y_train, x_test, y_test = self.train_test_split(pos_matrix, neg_matrix, train_prop=0.7, repeats=4)
        # convert to pytorch tensor
        self.x_train = torch.from_numpy(x_train)
        self.x_test = torch.from_numpy(x_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_test = torch.from_numpy(y_test)
        # data loader
        trainset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        testset = torch.utils.data.TensorDataset(self.x_test, self.y_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                  shuffle=True, num_workers=2)
        return trainloader, testloader


    def save_tensor(self):
        torch.save(self.x_train, "x_train.pkl")
        torch.save(self.x_test, "x_test.pkl")
        torch.save(self.y_train, "y_train.pkl")
        torch.save(self.y_test, "y_test.pkl")


    def load_tensor(self):
        self.x_train = torch.load("x_train.pkl")
        self.x_test = torch.load("x_test.pkl")
        self.y_train = torch.load("y_train.pkl")
        self.y_test = torch.load("y_test.pkl")
        # data loader
        trainset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        testset = torch.utils.data.TensorDataset(self.x_test, self.y_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=True, num_workers=2)
        print("tensor loaded.")
        return trainloader, testloader

    def get_Fscore(self): # 模型太大时会崩溃，弃用
        step_len=2
        # train f score
        precision=recall=f1=0
        for i in range(0,len(self.y_train),step_len):
            pred = torch.round(net(Variable(self.x_train[i:i+step_len].float().cuda())))
            y=self.y_train[i:i+step_len].float().cuda()
            precision += torch.sum(pred * y) / torch.sum(pred) * len(y)
            recall += torch.sum(pred * y) / torch.sum(y)
            f1 += precision * recall * 2 / (precision + recall)
        precision /= len(self.y_train)
        recall /= len(self.y_train)
        f1 /= len(self.y_train)
        print(f"train precision: {precision}, recall: {recall}, F1 score: {f1}")

        # test f score
        precision = recall = f1 = 0
        for i in range(0, len(self.y_test), step_len):
            pred = torch.round(net(Variable(self.x_test[i:i + step_len].float().cuda())))
            y = self.y_test[i:i + step_len].float().cuda()
            precision += torch.sum(pred * y) / torch.sum(pred) * len(y)
            recall += torch.sum(pred * y) / torch.sum(y)
            f1 += precision * recall * 2 / (precision + recall)
        precision /= len(self.y_test)
        recall /= len(self.y_test)
        f1 /= len(self.y_test)
        print(f"test precision: {precision}, recall: {recall}, F1 score: {f1}")


    def get_metrics(self, loader):
        tp = p = t = 0
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            pred = torch.round(outputs)
            tp += torch.sum(pred * labels)
            p += torch.sum(labels)
            t += torch.sum(pred)
        precision = tp / t
        recall = tp / p
        f1 = precision * recall * 2 / (precision + recall)
        print(f"test precision: {precision}, recall: {recall}, F1 score: {f1}")


    def save(self, path):
        torch.save(self, path)

if __name__=='__main__':
    print("starting ...")
    opt={}
    opt["content_dim"] = 256
    opt["embedding_dim"]=256
    opt["linear_hidden_size"]=128
    opt["content_seq_len"]=225

    net = MultiCNNText(opt)
    if torch.cuda.is_available():
        print("using CUDA to accelerate")
        net.cuda(0)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #从已经分好词的文本中抽取特征矩阵
    trainloader, testloader = net.get_tensor("model/word2vec/word2vec","data/samples/positive.txt","data/samples/negative.txt")
    #保存抽取出来的矩阵（已经转换为pytorch tensor）
    #net.save_tensor()
    #直接读取已经保存的tensor
    #trainloader, testloader = net.load_tensor()

    for epoch in range(30):  # loop over the data set multiple times
        print("training...... epoch:", epoch)
        running_loss = 0.0
        tp = p = t = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            pred = torch.round(outputs)
            tp += torch.sum(pred * labels)
            p += torch.sum(labels)
            t += torch.sum(pred)
        print("loss:", running_loss/i)
        precision = tp / t
        recall = tp / p
        f1 = precision * recall * 2 / (precision + recall)
        print(f"train precision: {precision}, recall: {recall}, F1 score: {f1}")
        net.get_metrics(testloader)

    net.save("net.pkl")
    print('Finished Training')

