from tqdm import tqdm
import os







train_txt_path = '/home/heliy2/lab/AdaptSign-main/preprocess/CSL/dev.txt'

import re
f=open(train_txt_path,'r')
alllines=f.readlines()
f.close()
f=open(train_txt_path,'w+')
for eachline in alllines:
    a=re.sub('/disk1/dataset/CSL_Continuous/','/home/heliy2/lab/data/CSL_video/',eachline)
    f.writelines(a)
f.close()
    