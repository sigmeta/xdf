import os
import subprocess


#ffmpeg所在位置
ffmpeg_path="C:\\xdf\\ffmpeg\\bin\\ffmpeg.exe"
#视频所在的位置
video_path="\\\\10.200.42.105\\video-shuiqing\\records_20181101"
#输出视频的位置
output_path="C:\\xdf\\video\\records_20181101"


if not os.path.exists(output_path):
    os.makedirs(output_path)
videos=os.listdir(video_path)
for v in videos:
    input=os.path.join(video_path,v)
    output=os.path.join(output_path,v)
    cmd=f"{ffmpeg_path} -i {input} -vcodec h264 {output}"
    try:
        subprocess.call(cmd)
        print(output, '输出结束')
    except Exception as e:
        print(output, '截取出错，异常信息如下：')
        print(e)