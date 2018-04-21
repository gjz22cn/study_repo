import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def splitFileFullPath(filename):  
    (filepath,tempfilename) = os.path.split(filename);  
    (shotname,extension) = os.path.splitext(tempfilename);  
    return filepath,shotname,extension

def genKuaiMixFile(fileName,saveName):
    rgb_black = [0,0,0]
    rgb_white = [255,255,255]
    img = cv2.imread(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.blur(gray,(5,5))
    ret,otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    h = img.shape[0]
    w = img.shape[1]
    for row in range(0,h):
        for col in range(0,w):
            if otsu[row,col] == 255:
                img[row,col] = np.array(rgb_black)
    cv2.imwrite(saveName,img)
    #plt.imshow(img)
    #plt.show()


def genKuaiDir(dirPath,dstDir):
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
        
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        file_name = os.path.join(dirPath,list[i])
        if os.path.isfile(file_name):
            fpath,shortname,ext = splitFileFullPath(file_name)
            if not (ext == ".jpg"):
                continue
            
            genKuaiMixFile(file_name,os.path.join(dstDir,list[i]))

def genKuaiTrainDir(srcDir,dstDir):
    labels = ['0_good',
            '2_quejiaobenbian',
            '3_heibanheibian',
            '4_duanluduanlu',
            '5_xuhan',
            '6_dixiao',
            '8_diepiancuowei',
            '9_liangbanbaoguang']

    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)

    for label in labels:
        src_dir = os.path.join(srcDir,label)
        if not os.path.isdir(src_dir):
            continue
        
        dst_dir = os.path.join(dstDir,label)
        genKuaiDir(src_dir,dst_dir)

genKuaiTrainDir("./data_0419", "./kuai")
#genKuaiMixFile("./2.jpg","./otsu/2.jpg")
