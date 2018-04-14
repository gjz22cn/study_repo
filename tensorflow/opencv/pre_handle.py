# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
from PIL import Image  
import os

g_row_fill_out_v = 10000
g_col_fill_out_v = 10000

def printImgGray(image,row,col):
    print('-----------------------------------------')
    for i in range(0,row):
        for j in range(0,col):
            print("%02x "%(image[i,j]), end='')
        print('')

def printRGB(image,row,col):
    print('-----------------------------------------')
    for i in range(0,row):
        for j in range(0,col):
            print("%3d"%(image[i,j,0]), end='')
        print('')


    print('-----------------------------------------')
    for i in range(0,row):
        for j in range(0,col):
            print("%3d"%(image[i,j,1]), end='')
        print('')


    print('-----------------------------------------')
    for i in range(0,row):
        for j in range(0,col):
            print("%3d"%(image[i,j,2]), end='')
        print('')
        
def get_top_margin(img):
    row = img.shape[0]
    for i in range(0,row):
        v = np.sum(img[i])
        if v > g_row_fill_out_v:
            #print("t_v=%d"%(v))
            break
    return i

def get_buttom_margin(img):
    row = img.shape[0]
    got_zero_line = False
    for i in range(row-1,-1,-1):
        v = np.sum(img[i])
        if v == 0:
            got_zero_line = True
        else:
            if got_zero_line:
                if v > g_row_fill_out_v:
                    #print("b_v=%d"%(v))
                    break
        
    return row-i-1

def get_left_margin(img):
    col = img.shape[1]
    for i in range(0,col):
        v = np.sum(img[:,i])
        if v > g_col_fill_out_v:
            #print("l_v=%d"%(v))
            break
    return i

def get_right_margin(img):
    col = img.shape[1]
    for i in range(col-1,-1,-1):
        v = np.sum(img[:,i])
        if v > g_col_fill_out_v:
            #print("r_v=%d"%(v))
            break
    return col-i-1

def handleImage(filename):
    img = cv2.imread(filename)
    #h=img.shape[0]
    #w=img.shape[1]
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #printRGB(img, 100, 100)
    #printImgGray(gray, 100, 100)    
    top_margin = get_top_margin(img)
    buttom_margin = get_buttom_margin(img)
    left_margin = get_left_margin(img)
    right_margin = get_right_margin(img)
    #print("t=%d,b=%d,l=%d,r=%d"%(top_margin,buttom_margin,left_margin,right_margin))
    return top_margin,buttom_margin,left_margin,right_margin

def get_filepath_filename_filext(filename):  
    (filepath,tempfilename) = os.path.split(filename);  
    (shotname,extension) = os.path.splitext(tempfilename);  
    return filepath,shotname,extension

def checkDir(dirPath):
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            if extension == ".jpg":
                top_margin,buttom_margin,left_margin,right_margin = handleImage(filePath)
                if top_margin != 48 or buttom_margin != 106 or left_margin != 48 or right_margin != 48:
                    print("%s:T=%d,B=%d,L=%d,R=%d"%(shortname+extension,top_margin,buttom_margin,left_margin,right_margin))



def main():
    checkDir("F:/opencv2/NG_mark")
    #handleImage("F:/code/opencv/sample3.jpg")
    
if __name__ == '__main__':
    main()
