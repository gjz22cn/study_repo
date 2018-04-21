import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

'''
b = 11
v = 2
img1 = cv2.imread('1.jpg',0)
img2 = cv2.imread('2.jpg',0)
img3 = cv2.imread('3.jpg',0)
img4 = cv2.imread('4.jpg',0)
img5 = cv2.imread('5.jpg',0)
img6 = cv2.imread('6.jpg',0)
img7 = cv2.imread('7.jpg',0)
img8 = cv2.imread('8.jpg',0)
#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th3 = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th4 = cv2.adaptiveThreshold(img4,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th5 = cv2.adaptiveThreshold(img5,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th6 = cv2.adaptiveThreshold(img6,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th7 = cv2.adaptiveThreshold(img7,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
th8 = cv2.adaptiveThreshold(img8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)


images = [img1,img2,img3,img4,th1,th2,th3,th4,img5,img6,img7,img8,th5,th6,th7,th8]
plt.figure()
for i in xrange(16):
    plt.subplot(4,4,i+1),plt.imshow(images[i],'gray')
plt.show()
'''

'''
#img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')
img4 = cv2.imread('4.jpg')
img5 = cv2.imread('5.jpg')
img6 = cv2.imread('6.jpg')
img7 = cv2.imread('7.jpg')
img8 = cv2.imread('8.jpg')
img9 = cv2.imread('9.jpg')
img10 = cv2.imread('10.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
retval,otsu1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu7 = cv2.threshold(gray7, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu8 = cv2.threshold(gray8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu9 = cv2.threshold(gray9, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval,otsu10 = cv2.threshold(gray10, 0, 255, cv2.THRESH_OTSU)
images = [img1,img2,img3,img4,img5,otsu1,otsu2,otsu3,otsu4,otsu5,img6,img7,img8,img9,img10,otsu6,otsu7,otsu8,otsu9,otsu10]
plt.figure()
for i in xrange(20):
    plt.subplot(4,5,i+1),plt.imshow(images[i],'gray')
plt.show()
'''

'''
img1 = cv2.imread('good.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print("-----------good------------------")
for i in range(20,50):
    for j in range(20,50):
        print("%3d"%(gray1[i,j])),
    print("")

print("-----------dixiao------------------")
img2 = cv2.imread('dixiao.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
for i in range(20,50):
    for j in range(20,50):
        print("%3d"%(gray2[i,j])),
    print("")
'''

def my_filter(gray):
    h,w = gray.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    return blured


file_names = ['liefeng_3.jpg',
        'quekuai.jpg',
        #'xuhan.jpg',
        'good1.jpg',
        'heibian.jpg']
imgs = []
ths = []
otsus = []
otsu_ths = []
th_otsus = []
blur_otsus = []
my_imgs = []
b = 15 
v = 5
for file_name in file_names:
    img = cv2.imread(file_name)
    h = img.shape[0]
    w = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))


    otsu = copy.copy(gray)

    #retval,otsu = cv2.threshold(otsu, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    step = 200
    size = 200
    for x in range(0,h,step):
        for y in range(0,w,step):
            start_x = x
            if start_x + size > h:
                start_x = h - size
            
            start_y = y
            if start_y + size > w:
                start_y = w - size
            
            gray_piece = otsu[start_x:start_x+size, start_y:start_y+size]
            retval,otsu_piece = cv2.threshold(gray_piece, 0, 255, cv2.THRESH_OTSU)
            for i in xrange(size):
                for j in xrange(size):
                    gray_piece[i,j] = otsu_piece[i,j]
            plt.imshow(gray_piece)
            plt.show()
    '''
    
    my_img = my_filter(gray)
    my_imgs.append(my_img)

    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
    retval,otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #otsu_th = cv2.adaptiveThreshold(th,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,b,v)
    #retval,th_otsu = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #blur = cv2.blur(gray,(3,3),0)    
    #retval,blur_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgs.append(img)
    ths.append(th)
    otsus.append(otsu)
    #otsu_ths.append(otsu_th)
    #th_otsus.append(th_otsu)
    #blur_otsus.append(blur_otsu)

row = len(imgs)
col = 4
plt.figure()
for i in xrange(row):
    plt.subplot(row,col,col*i+1),plt.imshow(imgs[i])
    plt.subplot(row,col,col*i+2),plt.imshow(otsus[i])
    plt.subplot(row,col,col*i+3),plt.imshow(ths[i])
    plt.subplot(row,col,col*i+4),plt.imshow(my_imgs[i])
plt.show()
