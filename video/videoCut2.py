#视频自动切分
#多进程版本
import os
import subprocess
from multiprocessing import Pool
import time


#TODO 需要填写的部分
#ffmpeg所在位置
ffmpeg_path="C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
#图片文件夹位置。该文件夹下是二级文件夹，二级文件夹名是视频名；二级文件夹中是保留的帧图片，图片名是时间（秒）
file_path='//10.200.42.124\\videos\\segments\\records_20181019'
#视频所在的位置
video_path='//10.200.42.105\\videos\\records_20181019'
#输出视频的位置
output_path='C:\\test'
#截取视频长度（秒）
dur_time='60'

#使用ffmpeg切分视频
def cutVideo(ffmpeg_path,input_file,start_time,dur_time,output_name):
    if os.path.exists(output_name):  # 如果已经有该片段了就直接跳过，否则会导致ffmpeg命令询问是否需要覆盖而卡住
        print(output_name, '该片段已存在')
        return 0
    cmd = "%s -i %s -ss %s -t %s %s" % (ffmpeg_path, input_file, start_time, dur_time, output_name)
    try:
        print(output_name)
        subprocess.call(cmd)
        print(output_name, '输出结束')
        return 1
    except Exception as e:
        print(output_name, '截取出错，异常信息如下：')
        print(e)
        return 0

if __name__=='__main__':
    # 所有的二级文件夹，如果一级文件夹下有文件会被跳过，只保留文件夹
    s_paths = [p for p in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, p))]
    # print(s_paths)
    # 判断是否有缺失视频的情况，即帧图片的视频不在视频文件夹中
    if set(s_paths) - set([v[:-4] for v in os.listdir(video_path)]):
        print('！！！以下视频不存在：', set(s_paths) - set([v[:-4] for v in os.listdir(video_path)]))
        s_paths = list(set(s_paths) & set([v[:-4] for v in os.listdir(video_path)]))
    print(s_paths)

    for i,f in enumerate(s_paths):
        loop_start_time = time.time()
        print(loop_start_time)
        print('开始处理视频：'+f,'当前是第'+str(i+1)+'段','共'+str(len(s_paths))+'段视频')
        # 如果输出视频文件夹不存在就创建
        if not os.path.exists(os.path.join(output_path,f)):
            os.makedirs(os.path.join(output_path,f))
        #截取的有效帧放在pos文件夹下
        pic_time_list=[int(pic.split('.')[0])*10-10 for pic in os.listdir(os.path.join(file_path,f,'pos'))]
        start_time_list=[pic_time-pic_time%60 for pic_time in pic_time_list]
        print(sorted(list(set(start_time_list))))
        #多进程同时处理
        p=Pool(4)
        for st in sorted(list(set(start_time_list))):
            start_time=str(st)
            input_file=os.path.join(video_path,f+'.mp4')
            output_name=output_path+'/'+f+'/'+start_time+'.mp4'
            p.apply_async(cutVideo,args=(ffmpeg_path,input_file,start_time,dur_time,output_name))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
        print(time.time()-loop_start_time)

