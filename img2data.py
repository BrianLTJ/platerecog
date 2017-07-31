import cv2
import numpy as np
import os
import pickle

import argparse

DATA_DIM_X = 20
DATA_DIM_Y = 20

data_mat = []

train_data_percent = 0.6

char_output_dict = {
    "3":[1,0,0,0,0,0,0,0,0,0],
    "4":[0,1,0,0,0,0,0,0,0,0],
    "5":[0,0,1,0,0,0,0,0,0,0],
    "7":[0,0,0,1,0,0,0,0,0,0],
    "a":[0,0,0,0,1,0,0,0,0,0],
    "d":[0,0,0,0,0,1,0,0,0,0],
    "k":[0,0,0,0,0,0,1,0,0,0],
    "l":[0,0,0,0,0,0,0,1,0,0],
    "q":[0,0,0,0,0,0,0,0,1,0],
    "v":[0,0,0,0,0,0,0,0,0,1],
}

def main(img_addr, cur_char):
    print("Loading Image file",img_addr)
    raw_img = cv2.imread(img_addr)
    ret, thr = cv2.threshold(raw_img,127,255,cv2.THRESH_BINARY)
    # cv2.imshow("img", raw_img)
    thr_list = thr.tolist()
    final_mat = []
    for line_item in thr_list:
        for item in line_item:
            if item[0]*0.299+item[1]*0.587+item[2]*0.114 >= 180:
                final_mat.append(1)
            else:
                final_mat.append(0)

    data_mat.append({"char":char_output_dict[cur_char],"data":final_mat})
    # print(final_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="3",help="Source Image file")

    img_char, unparsed = parser.parse_known_args()
    img_char = img_char.img
    imgdir = os.path.join(os.curdir, "imgs", "divided", img_char)
    for f in os.listdir(imgdir):
        if os.path.isfile(os.path.join(imgdir, f)):
            main(os.path.join(imgdir, f),img_char)

    total_number = len(data_mat)
    train_number = total_number*train_data_percent

    if os.path.exists(os.path.join(os.curdir,"data"))==False:
        os.mkdir(os.path.join(os.curdir,"data"))


    with open(os.path.join(os.curdir,"data","data_train_"+img_char+".dat"),'wb') as fp:
        pickle.dump(data_mat[0:int(train_number-1)], fp)


    with open(os.path.join(os.curdir,"data", "data_test_" + img_char + ".dat"), 'wb') as fp:
        pickle.dump(data_mat[int(train_number):], fp)


