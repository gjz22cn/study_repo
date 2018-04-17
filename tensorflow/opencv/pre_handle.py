# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
from PIL import Image  
import os
import matplotlib.pyplot as plt

g_row_fill_out_v = 10000
g_col_fill_out_v = 10000


g_labels = ['0_good',
           '1_yinliesuipian', 
           '2_quejiaobenbian', 
           '3_heibanheibian',
           '4_duanluduanlu',
           '5_xuhan',
           '6_dixiao',
           '7_zhangwuhuahen',
           '8_diepiancuowei',
           '9_liangbanbaoguang']

g_labelsNeedRoate = ['1_yinliesuipian',
                     '2_quejiaobenbian',
                     '3_heibanheibian',
                     '5_xuhan',
                     '6_dixiao',
                     '7_zhangwuhuahen',
                     '8_diepiancuowei',
                     '9_liangbanbaoguang']

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
            print("%3d "%(image[i,j,0]), end='')
        print('')


    print('-----------------------------------------')
    for i in range(0,row):
        for j in range(0,col):
            print("%3d "%(image[i,j,1]), end='')
        print('')


    print('-----------------------------------------')
    for i in range(0,row):
        for j in range(0,col):
            print("%3d "%(image[i,j,2]), end='')
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

def resizeImageFile(fileName,saveName):
    image=cv2.imread(fileName)
    new_image = cv2.resize(image, (299,299), interpolation=cv2.INTER_AREA)
    cv2.imwrite(saveName,new_image)

def resizeImageDir(dirPath,dstDir):
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
        
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            if extension == ".jpg":
                resizeImageFile(filePath,dstDir+shortname+extension)

def addRoatedPicForDir(dirPath,angle):
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        if "_rotate_" in list[i]:
            continue
        
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            if os.path.isfile(os.path.join(dirPath,shortname+"_rotate_"+str(angle)+extension)):
                continue
            
            img = Image.open(filePath)
            im_rotate = img.rotate(angle)
            im_rotate.save(dirPath+shortname+"_rotate_"+str(angle)+extension)

def resizeTrainDir(srcPath,dstPath):
    for label in g_labels:
        if not os.path.isdir(dstPath):
            os.mkdir(dstPath)
        
        src_dir = srcPath + label + "/"
        if not os.path.isdir(src_dir):
            continue
        
        resizeImageDir(src_dir, dstPath+label+"/")

def addRotatePicForTrainDir(trainPath):
    for label in g_labelsNeedRoate:   
        src_dir = trainPath + label + "/"
        if not os.path.isdir(src_dir):
            continue
        
        addRoatedPicForDir(src_dir,180)

def show_v_h_r_graph(h_r):
    width = h_r.size
    x_h = np.random.randint(1, size=width)
    for i in range(0,width):
        x_h[i] = i

    plt.plot(x_h,h_r,"b--",linewidth=1)
    plt.title("horizontal")

    plt.show()
    


def test2():
    img1 = cv2.imread(r"F:\opencv2\train\data\0_good\045807_6_2.jpg")
    img2 = cv2.imread(r"F:\opencv2\train\data\0_good\064238_5_3.jpg")
    img3 = cv2.imread(r"F:\opencv2\train\data\1_yinliesuipian\013746_3_2.jpg")
    #image1 = cv2.resize(img1,(299,299))
    #image2 = cv2.resize(img2,(299,299))
    hist1 = cv2.calcHist([img1],[0],None,[256],[0.0,255.0])
    hist2 = cv2.calcHist([img3],[0],None,[256],[0.0,255.0])
    plt.plot(range(256),hist1,'r')
    plt.plot(range(256),hist2,'b')
    plt.show()
    
    degree = 0
    for i in range(len(hist1)): 
        if hist1[i] != hist2[i]: 
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
        else:
            degree = degree + 1
    degree = degree/len(hist1)
    print("degree=%f"%(degree))

def imageLikeDetect(dirPath):
    img_base = cv2.imread(r"F:\opencv2\train\data\0_good\045807_6_2.jpg")
    hist_base = cv2.calcHist([img_base],[0],None,[299],[0.0,255.0])
    
    list = os.listdir(dirPath)
    #for i in range(0,len(list)):
    for i in range(0,10):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            img_check = cv2.imread(filePath)
            hist_check = cv2.calcHist([img_check],[0],None,[299],[0.0,255.0])
            
            plt.plot(range(299),hist_base,'r')
            plt.plot(range(299),hist_check,'b')
            plt.show()
            
            degree = 0
            for i in range(len(hist_base)):
                if hist_base[i] != hist_check[i]:
                    degree = degree + (1 - abs(hist_base[i]-hist_check[i])/max(hist_base[i],hist_check[i]))
                else:
                    degree = degree + 1
            degree = degree/len(hist_base)

            print("%s:\t%f"%(shortname+extension,degree))

def test(fileName):
    image=cv2.imread(fileName)
    row = image.shape[0]
    col = image.shape[1]
    printRGB(image,row,col)
    for i in range(0,row):
        for j in range(0,col):
            if image[i,j,0] > 176:
                image[i,j] = np.array([255, 255, 255])
    plt.imshow(image)
    plt.show()
    cv2.imwrite(r"F:\opencv2\huahen_1.jpg", image)
    '''
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h_mean = np.mean(gray,1)
    for i in range(0,row):
        for j in range(0,col):
            if gray[i,j] > h_mean[i]:
                gray[i,j] = 0
            else:
                gray[i,j] = 255
    
    plt.imshow(gray)
    plt.show()
    '''
    
    #h_r = np.add.reduce(gray,0)
    #im_at_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    #im_at_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    #im_at_mean = cv2.GaussianBlur(gray,(3,3),0)
    
    #show_v_h_r_graph(h_r)
    #plt.imshow(gray)
    #plt.imshow(im_at_mean)
    #plt.show()

def genHuahen(srcDir,dstDir):
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
    
    image_hh=cv2.imread(r"F:\opencv2\huahen_1.jpg")
    h_hh = image_hh.shape[0]
    w_hh = image_hh.shape[1]
    
    list = os.listdir(srcDir)
    for i in range(0,len(list)):        
        filePath = os.path.join(srcDir,list[i])
        if os.path.isfile(filePath):
            print("%s"%(filePath))
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            image=cv2.imread(filePath)
            h = image.shape[0]
            w = image.shape[1]
            for x in range(10,w-40-w_hh,30):
                for y in range(10,h-40-h_hh,30):
                    image_new = image.copy()
                    for x_hh in range(0,h_hh):
                        for y_hh in range(0,w_hh):
                            if image_hh[x_hh,y_hh,0] < 176:
                                image_new[x+x_hh,y+y_hh] = image_hh[x_hh,y_hh]
                    image_new = cv2.GaussianBlur(image_new,(3,3),0)
                    cv2.imwrite(dstDir+shortname+"_human_"+str(x)+"_"+str(y)+extension, image_new)

        
def main():
    #imageLikeDetect("F:/opencv2/train/data/0_good/")
    #imageLikeDetect("F:/opencv2/train/data/1_yinliesuipian/")
    genHuahen("F:/opencv2/data_0417/data_human/0/","F:/opencv2/data_0417/data_human/1/")
    #test(r"F:\opencv2\huahen_1.jpg")
    '''
    test(r"F:\opencv2\train\data\1_yinliesuipian\100616_2_1_rotate_180.jpg")
    test(r"F:\opencv2\train\data\2_quejiaobenbian\LRP503038180300366709_6_1.jpg")
    test(r"F:\opencv2\train\data\3_heibanheibian\LRP503038180300364683_1_4.jpg")
    test(r"F:\opencv2\train\data\5_xuhan\150415_1_7_rotate_180.jpg")
    test(r"F:\opencv2\train\data\9_liangbanbaoguang\5_r90_rotate_180.jpg")
    test(r"F:\opencv2\train\data\0_good\045807_6_2.jpg")
    test(r"F:\opencv2\train\data\0_good\064238_5_3.jpg")
    '''
    #test2()
    #resizeTrainDir("F:/opencv2/data/","F:/opencv2/data_resize/")
    #addRotatePicForTrainDir("F:/opencv2/data/")        
    #checkDir("F:/opencv2/NG_mark")
    #handleImage("F:/code/opencv/sample3.jpg")
    
if __name__ == '__main__':
    main()
