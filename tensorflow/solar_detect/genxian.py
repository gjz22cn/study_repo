import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def splitFileFullPath(filename):  
    (filepath,tempfilename) = os.path.split(filename);  
    (shotname,extension) = os.path.splitext(tempfilename);  
    return filepath,shotname,extension

def genXianMixFile(fileName,saveName):
    if os.path.isfile(saveName):
        return

    rgb_black = [0,0,0]
    img = cv2.imread(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    blured = cv2.blur(gray,(5,5))

    step=20
    size=20
    f = 10
    for x in range(0,h,step):
        for y in range(0,w,step):
            start_x = x
            if start_x + size > h:
                start_x = h - size
                
            start_y = y
            if start_y + size > w:
                start_y = w - size
            
            blured_piece = blured[start_x:start_x+size, start_y:start_y+size]
            mean_v = np.mean(blured_piece)
            for i in xrange(size):
                for j in xrange(size):
                    if blured_piece[i,j] > mean_v:
                        blured_piece[i,j] = 255
                    else:
                        if mean_v - blured_piece[i,j] < f:
                            blured_piece[i,j] = 255

    for row in range(0,h):
        for col in range(0,w):
            if blured[row,col] == 255:
                img[row,col] = np.array(rgb_black)

    cv2.imwrite(saveName,img)


def genXianDir(dirPath,dstDir):
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
        
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        file_name = os.path.join(dirPath,list[i])
        if os.path.isfile(file_name):
            fpath,shortname,ext = splitFileFullPath(file_name)
            if not (ext == ".jpg"):
                continue
            
            genXianMixFile(file_name,os.path.join(dstDir,list[i]))

def genOtsuTrainDir(srcDir,dstDir):
    labels = ['0_good',
            '1_yinliesuipian',
            '7_huahen',
            '8_diepiancuowei',
            '10_zhangwu']

    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)

    for label in labels:
        src_dir = os.path.join(srcDir,label)
        if not os.path.isdir(src_dir):
            continue
        
        dst_dir = os.path.join(dstDir,label)
        genXianDir(src_dir,dst_dir)

genOtsuTrainDir("./data_xian", "./xian")
#genOtsuMixFile("./2.jpg","./otsu/2.jpg")
