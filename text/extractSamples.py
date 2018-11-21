import os
import jieba


def extract_samples(record_path = "//10.200.42.124/videos/segments",
                    ori_path = "data/text",
                    save_path = "data/samples"):
    '''
    将精彩片段抽取出来，正例反例分开
    :param record_path:
    :param ori_path:
    :param save_path:
    :return:
    '''
    #with open("data/stopwords.txt",encoding='utf8') as f:
        #stop_words=set(f.read().split('\n'))
    stop_words = {' ', '，', '？', '囡囡', '囡', '\n', '嗯', '！', '吖', '了', '。', '唉', '呀'}

    count=0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pf=open(os.path.join(save_path,'positive.txt'),'w',encoding='utf8')
    nf=open(os.path.join(save_path,'negative.txt'),'w',encoding='utf8')
    for date_path in os.listdir(ori_path):
        for text_path in os.listdir(os.path.join(ori_path,date_path)):
            print("processing... ",ori_path,date_path,text_path)
            highlights = set([])
            # if exists highlights
            if os.path.exists(os.path.join(record_path,date_path,text_path[:-4],'pos','p')):
                good_frames=os.listdir(os.path.join(record_path,date_path,text_path[:-4],'pos','p'))
                if not good_frames:
                    print("no highlights")
                    continue
                for frame in good_frames:
                    try:
                        minute=((int(frame.split('.')[0])-3)*10+9)//60
                    except Exception as e:
                        print(e)
                        print(frame)
                    else:
                        if minute<0:
                            minute=0
                        highlights.add(minute)
            # no highlights
            else:
                print("no highlights")
                continue
            print("highlights: ",highlights)
            count+=1
            with open(os.path.join(ori_path,date_path,text_path),encoding='utf8') as f:
                for i,line in enumerate(f):
                    if not line: # empty line (minute)
                        continue
                    words=[w for w in jieba.cut(line) if w not in stop_words]
                    if not words:
                        continue
                    if i in highlights:
                        pf.write(' '.join(words)+'\n')
                    else:
                        nf.write(' '.join(words)+'\n')
    pf.close()
    nf.close()
    print("number of valid files:",count)
    return 0


if __name__=='__main__':
    record_path = "//10.200.42.124/videos/segments"
    ori_path = "data/text"
    save_path = "data/samples"
    extract_samples(record_path, ori_path, save_path)