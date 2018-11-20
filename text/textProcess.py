import os
import json


def cut_text_by_minute(txt):
    '''
    :param txt: text that can be transformed to json
    :return: list that contains text cut by minute
    '''
    js = json.loads(txt)
    minutes = (int(js[-1]['bg']) + int(js[-1]['ed'])) // 120000
    mlist = [''] * (minutes + 1)
    print(len(mlist))
    for t in js:
        m = (int(t['bg']) + int(t['ed'])) // 120000
        mlist[m] = mlist[m]+' '+t["onebest"] if mlist[m] else t["onebest"]
    return mlist

def text_process(ori_path, save_path):
    count = 0
    for date_path in os.listdir(ori_path):
        if not os.path.exists(os.path.join(save_path, date_path)):
            os.makedirs(os.path.join(save_path, date_path))
        for video_path in os.listdir(os.path.join(ori_path, date_path)):
            print("processing... ", ori_path, date_path, video_path)
            count += 1
            with open(os.path.join(ori_path, date_path, video_path), encoding='utf8') as f:
                txt = f.read()
            mlist = cut_text_by_minute(txt)
            print(len(mlist))
            with open(os.path.join(save_path, date_path, video_path), 'w', encoding='utf8') as f:
                f.write('\n'.join(mlist))
    print(count)
    return 0

if __name__ == '__main__':
    ori_path = "//10.200.42.124/videos/denoise_text"
    save_path = "data/text"
    text_process(ori_path, save_path)

