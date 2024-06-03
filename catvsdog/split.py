'''
该代码用于将./cat_dog/train目录与./cat_dog/val目录下的猫狗图片分开
需要提前在这两个目录下创建dogs与cats目录
'''

import os
import shutil
paths = ["./cat_dog/train", "./cat_dog/val"]
idx = 1
for path in paths:
    dir_list = os.listdir(path)
    for file_list in dir_list:
        
        if os.path.isfile(path +"/"+ file_list):
            source_path = path + "/" + file_list
            if file_list.startswith("dog"):
                destination_path = path + "/dogs/" + file_list
            else:
                destination_path = path + "/cats/" + file_list

            shutil.move(source_path, destination_path)
            idx += 1
            if idx % 1000 == 0:
                print("move 1000 file")

