
import re
import os
import cv2
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm


def extract_frames(video_path, output_folder):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(output_folder)
    else:
        print("skip:"+os.path.basename(output_folder))
        return

    frame_count = 0
    while True:
        # 逐帧读取视频
        ret, frame = video_capture.read()

        # 如果读取失败，退出循环
        if not ret:
            break

        # 翻转图像（垂直翻转）
        #frame = cv2.flip(frame, 0)

        # 构建帧的输出文件路径
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # 保存帧为图片
        if frame_count != 0:
            cv2.imwrite(frame_filename, frame)
            # 打印保存信息
            # print(f"Saved: {frame_filename}")



        frame_count += 1

    # 释放视频捕捉对象
    video_capture.release()
    # print("Finished extracting frames.")


# csl_data_path = "/remote-home/cs_cs_heli/data/CSL2018-zip-tar/sentence-zip/color-sentence/color"
csl_data_path = "/home/heliy2/lab/data/CSL2018-zip-tar/sentence-zip/color-sentence/color/043/P35_s5_03_3._color.mp4"
# csl_data_path = "/remote-home/cs_cs_heli/data/CSL_video/"
output_folder = "/home/heliy2/lab/data/CSL_video/video10923"  # 输出图片保存文件夹
video_name ='P34_s5_03_3._color.mp4'

# files = [os.path.join(csl_data_path, file) for file in os.listdir(csl_data_path)]
# # files_list = sorted(files,key=lambda x:int(os.path.basename(x)[5:-1]))
# files_list = sorted(files)
# video_count = 0



    #print(file)

extract_frames( csl_data_path,output_folder)









