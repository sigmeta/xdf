import os
import shutil
ori_path="//10.200.42.124/videos/segments/records_20181023"
for p in os.listdir(ori_path):
    if not os.path.exists(os.path.join(ori_path,p,"pos/p")):
        files=os.listdir(os.path.join(ori_path,p,"pos"))
        os.mkdir(os.path.join(ori_path,p,"pos/p"))
        for f in files:
            shutil.copyfile(os.path.join(ori_path,p,"pos",f),os.path.join(ori_path,p,"pos","p",f))
