import os
import subprocess

#TODO 需要填写的部分
#ffmpeg所在位置
ffmpeg_path="E:\\code\\ffmpeg\\bin\\ffmpeg.exe"
#图片文件夹位置。该文件夹下是二级文件夹，二级文件夹名是视频名；二级文件夹中是保留的帧图片，图片名是时间（秒）
file_path='E:\\Python\\xdf\\video\\im'
#视频所在的位置
video_path='E:\\Py\\xdf\\records_20181023'
#输出视频的位置
output_path='E:\\Python\\xdf\\video\\out'
#截取视频长度（秒）
dur_time='60'


#所有的二级文件夹，如果一级文件夹下有文件会被跳过，只保留文件夹
s_paths=[p for p in os.listdir(file_path) if os.path.isdir(os.path.join(file_path,p))]
#print(s_paths)
#判断是否有缺失视频的情况，即帧图片的视频不在视频文件夹中
if set(s_paths)-set([v[:-4] for v in os.listdir(video_path)]):
    print('！！！以下视频不存在：',set(s_paths)-set([v[:-4] for v in os.listdir(video_path)]))
    s_paths=list(set(s_paths)&set([v[:-4] for v in os.listdir(video_path)]))

print(s_paths)
for i,f in enumerate(s_paths):
    print('开始处理视频：'+f,'当前是第'+str(i)+'段','共'+str(len(s_paths))+'段视频')
    # 如果输出视频文件夹不存在就创建
    if not os.path.exists(os.path.join(output_path,f)):
        os.makedirs(os.path.join(output_path,f))
    for pic in os.listdir(os.path.join(file_path,f)):
        input_file=os.path.join(video_path,f+'.mp4')
        pic_time=int(pic.split('.')[0])*10
        start_time=str(pic_time-pic_time%60)
        output_name=output_path+'/'+f+'/'+start_time+'.mp4'
        if os.path.exists(output_name): #如果已经有该片段了就直接跳过，否则会导致ffmpeg命令询问是否需要覆盖而卡住
            print(output_name,'该片段已存在')
            continue
        cmd="%s -i %s -ss %s -t %s %s"%(ffmpeg_path,input_file,start_time,dur_time,output_name)
        try:
            subprocess.call(cmd)
            print(output_name,'输出结束')
        except Exception as e:
            print(output_name,'截取出错，异常信息如下：')
            print(e)
