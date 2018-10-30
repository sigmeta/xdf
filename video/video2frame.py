import cv2
import os

# 视频所在文件夹
video_path = 'E:\\Py\\xdf\\records_20181023'

#files=[os.path.join(video_path,f) for f in os.listdir(video_path)]
files=[f for f in os.listdir(video_path) if f[-4:]=='.mp4']
if not os.path.exists('image'):
    os.mkdir('image')

#遍历所有视频
for file in files:
    if not os.path.exists(os.path.join('image',file)): #创建图片保存路径
        os.mkdir(os.path.join('image',file))

    vc = cv2.VideoCapture(os.path.join(video_path,file))  # 读入视频文件
    c = 1
    print(file,'读取完毕')

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
        print(file,'视频打开出错')

    timeF = 250  # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作
            print(c)
            cv2.imwrite(os.path.join('image',file) + '/' + str(c) + '.jpg', frame)  # 存储为图像
        c = c + 1
        cv2.waitKey(1)
    vc.release()