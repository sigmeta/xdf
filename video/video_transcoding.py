import os
import subprocess
from multiprocessing import Pool


def transcoding(input,output):
    if os.path.exists(output):  # 如果已经有了就直接跳过，否则会导致ffmpeg命令询问是否需要覆盖而卡住
        print(output, '输出视频已存在')
        return False
    cmd = f"{ffmpeg_path} -i {input} -vcodec h264 -s 1280x720 {output}"
    print(cmd)
    try:
        subprocess.call(cmd)
        print(output, '输出结束')
        return True
    except Exception as e:
        print(output, '截取出错，异常信息如下：')
        print(e)
        return False


#ffmpeg所在位置
ffmpeg_path="C:\\xdf\\ffmpeg\\bin\\ffmpeg.exe"
#视频所在的位置
video_path="\\\\10.200.42.105\\video-shuiqing\\records_20181104"
#输出视频的位置
output_path="C:\\xdf\\video\\records_20181104"


if not os.path.exists(output_path):
    os.makedirs(output_path)
videos=os.listdir(video_path)


p = Pool(2)
for v in videos:
    input=os.path.join(video_path,v)
    output=os.path.join(output_path,v)
    p.apply_async(transcoding,args=(input,output))
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')