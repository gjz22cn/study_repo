import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

g_width = 4896
g_height = 3034
g_top_margin = 48
g_buttom_margin = 104
g_left_margin = 48
g_right_margin = 48
g_rows = 6
g_cols = 10
g_padding = 10

def get_filepath_filename_filext(filename):  
    (filepath,tempfilename) = os.path.split(filename);  
    (shotname,extension) = os.path.splitext(tempfilename);  
    return filepath,shotname,extension

def remove_black_boarder_new(image):
    width=img.shape[1]
    height=img.shape[0]
    zero_row_t_cnt = 0
    zero_row_b_cnt = 0
    zero_col_l_cnt = 0
    zero_col_r_cnt = 0
    ret,output = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    h_r = np.add.reduce(output,0)
    v_r = np.add.reduce(output,1)
    h_r_max = np.max(h_r)
    v_r_max = np.max(v_r)

    print("h_r=",h_r,"h_r.size=%d, h_r_max=%d"%(h_r.size,h_r_max))
    print("v_r=",v_r,"v_r.size=%d, v_r_max=%d"%(v_r.size,v_r_max))
    for col in range(0,width):
        if h_r[col] > 0:
            zero_col_l_cnt = col
            break

    for col in range(width-1, -1, -1):
        if h_r[col] > 0:
            zero_col_r_cnt = width - col -1
            break

    for row in range(0,height):
        if v_r[row] > 0:
            zero_row_t_cnt = row 
            break

    for row in range(height-1,-1,-1):
        if v_r[row] > 0:
            zero_row_b_cnt = height - row -1
            break

    print("r_t=%d, r_b=%d, c_l=%d, c_r=%d"%(zero_row_t_cnt,zero_row_b_cnt,zero_col_l_cnt, zero_col_r_cnt))
    show_v_h_r_graph(h_r,v_r)
    



def remove_black_boarder(image):
    need_break = 0
    zero_row_t_cnt = 0
    zero_row_b_cnt = 0
    zero_col_l_cnt = 0
    zero_col_r_cnt = 0
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    width=img.shape[1]
    height=img.shape[0]

    h_r = np.add.reduce(image,0)
    v_r = np.add.reduce(image,1)
    h_r_max = np.max(h_r)
    v_r_max = np.max(v_r)
    print("h_r=",h_r,"h_r.size=%d, h_r_max=%d"%(h_r.size,h_r_max))
    print("v_r=",v_r,"v_r.size=%d, v_r_max=%d"%(v_r.size,v_r_max))
    

    show_v_h_r_graph(h_r,v_r)

    for row in range(0,height):
        for col in range(0,width):
            if image[row,col] > 0:
                need_break = 1
                break

        if need_break == 1:
            break
        zero_row_t_cnt += 1

    need_break = 0
    for row in range(height-1, -1, -1):
        for col in range(0,width):
            if image[row,col] > 0:
                need_break = 1
                break

        if need_break == 1:
            break
        zero_row_b_cnt += 1

    need_break = 0
    for col in range(0,width):
        for row in range(0,height):
            if image[row,col] > 0:
                need_break = 1
                break

        if need_break == 1:
            break
        zero_col_l_cnt += 1

    need_break = 0
    for col in range(width-1, -1, -1):
        for row in range(0,height):
            if image[row,col] > 0:
                need_break = 1
                break

        if need_break == 1:
            break
        zero_col_r_cnt += 1

    print("r_t=%d, r_b=%d, c_l=%d, c_r=%d"%(zero_row_t_cnt,zero_row_b_cnt,zero_col_l_cnt, zero_col_r_cnt))




def printImgGray(image,height,width):
    print('-----------------------------------------')
    for i in range(0,height):
        for j in range(0,width):
            print("%02x"%(image[i,j])),
        print('')

def printRGB(image):
    h=image.shape[0]
    w=image.shape[1]
    print('-----------------------------------------')
    for i in range(0,h):
        for j in range(0,w):
            print("%3d"%(image[i,j,0])),
        print('')


    print('-----------------------------------------')
    for i in range(0,h):
        for j in range(0,w):
            print("%3d"%(image[i,j,1])),
        print('')


    print('-----------------------------------------')
    for i in range(0,h):
        for j in range(0,w):
            print("%3d"%(image[i,j,2])),
        print('')

def find_value_points(inArray):
    size = inArray.size
    zero_count = []
    zeros = 0
    one_count = []
    ones = 0
    inArray_max = np.max(inArray)
    fill_out_value = int(inArray_max/3)

    for i in range(0,size):
        if inArray[i] < fill_out_value:
            inArray[i] = 0
            zeros += 1
            
            if ones > 0:
                one_count.append(ones)
                ones = 0
        else:
            if zeros > 0:
                zero_count.append(zeros)
                zeros = 0
            ones += 1

    z_c_size = len(zero_count)
    o_c_size = len(one_count)
    z_c_max = np.max(zero_count)
    o_c_max = np.max(one_count)
    return z_c_size,z_c_max,o_c_size,o_c_max

def show_v_h_r_graph(h_r,v_r):
    height = v_r.size
    width = h_r.size
    x_h = np.random.randint(1, size=width)
    x_v = np.random.randint(1, size=height)
    for i in range(0,width):
        x_h[i] = i

    for i in range(0,height):
        x_v[i] = i

    plt.subplot(211)
    plt.plot(x_v,v_r,"b--",linewidth=1)
    plt.title("vertical")

    plt.subplot(212)
    plt.plot(x_h,h_r,"b--",linewidth=1)
    plt.title("horizontal")

    plt.show()

def calc_zero_and_one(inArray,n):
    size = inArray.size
    zero_count = []
    zeros = 0
    one_count = []
    ones = 0
    inArray_max = np.max(inArray)
    fill_out_value = int(inArray_max/n)

    for i in range(0,size):
        if inArray[i] < fill_out_value:
            inArray[i] = 0
            zeros += 1
            
            if ones > 0:
                one_count.append(ones)
                ones = 0
        else:
            if zeros > 0:
                zero_count.append(zeros)
                zeros = 0
            ones += 1

    z_c_size = len(zero_count)
    o_c_size = len(one_count)
    z_c_max = np.max(zero_count)
    o_c_max = np.max(one_count)
    print(zero_count)
    print(one_count)
    return z_c_size,z_c_max,o_c_size,o_c_max

def findSqureLength_np(image):
    h_r = np.add.reduce(output,0)
    v_r = np.add.reduce(output,1)
    h_r_max = np.max(h_r)
    v_r_max = np.max(v_r)
    print("h_r=",h_r,"h_r.size=%d, h_r_max=%d"%(h_r.size,h_r_max))
    print("v_r=",v_r,"v_r.size=%d, v_r_max=%d"%(v_r.size,v_r_max))
    

    show_v_h_r_graph(h_r,v_r)
    h_z_c_size,h_z_c_max,h_o_c_size,h_o_c_max = calc_zero_and_one(v_r,3)
    v_z_c_size,v_z_c_max,v_o_c_size,v_o_c_max = calc_zero_and_one(h_r,3)
    print("h_z_c_size=%d, h_z_c_max=%d, h_o_c_size=%d, h_o_c_max=%d"%(h_z_c_size,h_z_c_max,h_o_c_size,h_o_c_max))
    print("v_z_c_size=%d, v_z_c_max=%d, v_o_c_size=%d, v_o_c_max=%d"%(v_z_c_size,v_z_c_max,v_o_c_size,v_o_c_max))

    show_v_h_r_graph(h_r,v_r)
    return v_z_c_size,v_z_c_max,v_o_c_size,v_o_c_max

def cut_image_and_save(image,save_name,startX,startY,sizeX,sizeY):
    new_img = image[startY-g_padding:startY+sizeY+g_padding, startX-g_padding:startX+sizeX+g_padding]
    cv2.imwrite(save_name, new_img)

def cut_image_and_save_good(image_marked,save_name,startX,startY,sizeX,sizeY):
    marked_cnt = 0
    new_img = image_marked[startY-g_padding:startY+sizeY+g_padding, startX-g_padding:startX+sizeX+g_padding]
    for row in range(0,sizeX):
        for col in range(0,sizeY):
            if (new_img[row, col, 0]<100) and (new_img[row, col, 1]<100) and (new_img[row, col, 2]>200):
                marked_cnt += 1
                if marked_cnt > 10:
                    return False

    cv2.imwrite(save_name, new_img)
    return True

def cut_image_save_good_and_ref(image_marked,goodPrefix,refPrefix,nameTail,startX,startY,sizeX,sizeY):
    marked_cnt = 0
    new_img = image_marked[startY-g_padding:startY+sizeY+g_padding, startX-g_padding:startX+sizeX+g_padding]
    for row in range(0,sizeX):
        for col in range(0,sizeY):
            if (new_img[row, col, 0]<100) and (new_img[row, col, 1]<100) and (new_img[row, col, 2]>200):
                marked_cnt += 1
                if marked_cnt > 10:
                    cv2.imwrite(refPrefix+nameTail, new_img)
                    return False

    cv2.imwrite(goodPrefix+nameTail, new_img)
    return True

def cut_img_360_360(image):
    start_x = 0
    start_y = 0
    for row in range(0,6):
        for col in range(0,12):
            cut_image_and_save(image,"./cut_"+str(row)+"_"+str(col)+".jpg",start_x,start_y,360,360)
            start_x += 360

        start_y += 360
        start_x = 0

def cut_img_299_299(image,step):
    horizontal = image.shape[1]
    vertical = image.shape[0]
    h_c = (horizontal-299)/step
    v_c = (vertical-299)/step
    start_x = 0
    start_y = 0
    for row in range(0,v_c):
        for col in range(0,h_c):
            cut_image_and_save(image,"./cut_"+str(row)+"_"+str(col)+".jpg",start_x,start_y,299,299)
            start_x += step
        
        cut_image_and_save(image,"./cut_"+str(row)+"_"+str(h_c)+".jpg",horizontal-299-1,start_y,299,299)
        start_y += step
        start_x = 0

    start_x = 0
    for col in range(0,h_c):
        cut_image_and_save(image,"./cut_"+str(v_c)+"_"+str(col)+".jpg",start_x,vertical-299-1,299,299)
        start_x += step

    cut_image_and_save(image,"./cut_"+str(v_c)+"_"+str(h_c)+".jpg",horizontal-299-1,vertical-299-1,299,299)

def prepare_good_and_bad_file(dirPath,fileName,step):
    file_ori = dirPath+"/"+fileName+".jpg"
    file_marked = dirPath+"_marked"+"/"+fileName+".jpg"

    if not os.path.isfile(file_ori):
        print("no file: "+file_ori)
        return

    if not os.path.isfile(file_marked):
        print("no file: "+file_marked)
        return

    if not os.path.isdir("./good"):
        print("no dir: ./good")
        return

    if not os.path.isdir("./bad"):
        print("no dir: ./bad")
        return

    if not os.path.isdir("./ref"):
        print("no dir: ./ref")
        return

    print(fileName+".jpg is ready")

    good_prefix = "./good/"+fileName+"_"+str(step)+"_"
    bad_prefix = "./bad/"+fileName+"_"+str(step)+"_"
    ref_prefix = "./ref/"+fileName+"_"+str(step)+"_"

    img_ori = cv2.imread(file_ori)
    img_marked = cv2.imread(file_marked)

    horizontal = img_ori.shape[1]
    vertical = img_ori.shape[0]
    h_c = (horizontal-299)/step
    h_r = np.add.reduce(output,0)
    v_r = np.add.reduce(output,1)
    h_r_max = np.max(h_r)
    v_r_max = np.max(v_r)
    print("h_r=",h_r,"h_r.size=%d, h_r_max=%d"%(h_r.size,h_r_max))
    print("v_r=",v_r,"v_r.size=%d, v_r_max=%d"%(v_r.size,v_r_max))
    

    show_v_h_r_graph(h_r,v_r)
    v_c = (vertical-299)/step
    start_x = 0
    start_y = 0
    for row in range(0,v_c):
        for col in range(0,h_c):
            if not cut_image_and_save_good(img_marked,good_prefix+str(row)+"_"+str(col)+".jpg",start_x,start_y,299,299):
                cut_image_and_save(img_ori,bad_prefix+str(row)+"_"+str(col)+".jpg",start_x,start_y,299,299)
                cut_image_and_save(img_marked,ref_prefix+str(row)+"_"+str(col)+".jpg",start_x,start_y,299,299)
            start_x += step
        
        if not cut_image_and_save_good(img_marked,good_prefix+str(row)+"_"+str(h_c)+".jpg",horizontal-299-1,start_y,299,299):
            cut_image_and_save(img_ori,bad_prefix+str(row)+"_"+str(h_c)+".jpg",horizontal-299-1,start_y,299,299)
            cut_image_and_save(img_marked,ref_prefix+str(row)+"_"+str(h_c)+".jpg",horizontal-299-1,start_y,299,299)
        start_y += step
        start_x = 0

    start_x = 0
    for col in range(0,h_c):
        if not cut_image_and_save_good(img_marked,good_prefix+str(v_c)+"_"+str(col)+".jpg",start_x,vertical-299-1,299,299):
            cut_image_and_save(img_ori,bad_prefix+str(v_c)+"_"+str(col)+".jpg",start_x,vertical-299-1,299,299)
            cut_image_and_save(img_marked,ref_prefix+str(v_c)+"_"+str(col)+".jpg",start_x,vertical-299-1,299,299)
        start_x += step

    if not cut_image_and_save_good(img_marked,good_prefix+str(v_c)+"_"+str(h_c)+".jpg",horizontal-299-1,vertical-299-1,299,299):
        cut_image_and_save(img_ori,bad_prefix+str(v_c)+"_"+str(h_c)+".jpg",horizontal-299-1,vertical-299-1,299,299)
        cut_image_and_save(img_marked,ref_prefix+str(v_c)+"_"+str(h_c)+".jpg",horizontal-299-1,vertical-299-1,299,299)


def prepare_good_and_bad_file_new(dirPath,fileName,width,height,top_margin,buttom_margin,left_margin,right_margin,rows,cols):
    file_ori = dirPath+"/"+fileName+".jpg"
    file_marked = dirPath+"_marked"+"/"+fileName+".jpg"

    if not os.path.isfile(file_ori):
        print("no file: "+file_ori)
        return

    if not os.path.isfile(file_marked):
        print("no file: "+file_marked)
        return

    if not os.path.isdir("./good"):
        os.mkdir("./good")

    if not os.path.isdir("./bad"):
        os.mkdir("./bad")

    if not os.path.isdir("./ref"):
        os.mkdir("./ref")


    good_prefix = "./good/"+fileName+"_"
    bad_prefix = "./bad/"+fileName+"_"
    ref_prefix = "./ref/"+fileName+"_"

    img_ori = cv2.imread(file_ori)
    img_marked = cv2.imread(file_marked)

    img_w = img_ori.shape[1]
    img_h = img_ori.shape[0]

    if img_w != width or img_h != height:
        print("input image is %d*%d, but expected size if %d*%d."%(img_w,img_h,width,height))
        return

    if os.path.isfile(good_prefix+"0_0.jpg") or os.path.isfile(bad_prefix+"0_0.jpg"):
        print(fileName+".jpg is ready.")
        return

    print(fileName+".jpg is on processing...")

    step_x = int((width - left_margin - right_margin)/cols)
    step_y = int((height - top_margin - buttom_margin)/rows)
    start_x = left_margin
    start_y = top_margin
    for row in range(0,rows):
        start_x = left_margin
        
        for col in range(0,cols):
            if not cut_image_save_good_and_ref(img_marked,good_prefix,ref_prefix,str(row)+"_"+str(col)+".jpg",start_x,start_y,step_x,step_y):
                cut_image_and_save(img_ori,bad_prefix+str(row)+"_"+str(col)+".jpg",start_x,start_y,step_x,step_y)
            start_x += step_x
        
        start_y += step_y

def prepare_good_file(dirPath,fileName,width,height,top_margin,buttom_margin,left_margin,right_margin,rows,cols):
    file_ori = dirPath+"/"+fileName+".jpg"

    if not os.path.isfile(file_ori):
        print("no file: "+file_ori)
        return

    if not os.path.isdir("./good"):
        os.mkdir("./good")        

    good_prefix = "./good/"+fileName+"_"

    img_ori = cv2.imread(file_ori)

    img_w = img_ori.shape[1]
    img_h = img_ori.shape[0]

    if img_w != width or img_h != height:
        print("input image is %d*%d, but expected size if %d*%d."%(img_w,img_h,width,height))
        return

    print(fileName+".jpg is ready")

    step_x = int((width - left_margin - right_margin)/cols)
    step_y = int((height - top_margin - buttom_margin)/rows)
    start_x = left_margin
    start_y = top_margin
    for row in range(0,rows):
        start_x = left_margin
        
        for col in range(0,cols):
            cut_image_and_save(img_ori,good_prefix+str(row)+"_"+str(col)+".jpg",start_x,start_y,step_x,step_y)
            start_x += step_x
        
        start_y += step_y


def gen_good_and_bad_dir(dirPath):
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            #prepare_good_and_bad_file(dirPath, list[i][:-4], 250)
            prepare_good_and_bad_file_new(dirPath,list[i][:-4],
                    g_width,g_height,g_top_margin,
                    g_buttom_margin,g_left_margin,g_right_margin,g_rows,g_cols)

def gen_good_dir(dirPath):
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            prepare_good_file(dirPath,list[i][:-4],
                    g_width,g_height,g_top_margin,
                    g_buttom_margin,g_left_margin,g_right_margin,g_rows,g_cols)


def gen_rotate_pic_dir(dirPath,angle):
    dst_dir = "./rotate_" + str(angle) + "/"
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            #img = cv2.imread(filePath)
            img = Image.open(filePath)
            im_rotate = img.rotate(angle)
            #cv2.imwrite(dst_dir+shortname+"_a"+str(angle)+extension, im_rotate)
            im_rotate.save(dst_dir+shortname+"_r"+str(angle)+extension)

# example 1
#img = cv2.imread('./091549.jpg')
#img = cv2.imread('./002.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#h=img.shape[0]
#w=img.shape[1]

#remove_black_boarder_new(img)
#printImgGray(gray, h/2, w/10-6)
#remove_black_boarder(gray)
#print("height=%d,wigth=%d"%(h,w))
#ret,output = cv2.threshold(gray,96,1,cv2.THRESH_BINARY_INV)
#v_z_c_size,v_z_c_max,v_o_c_size,v_o_c_max = findSqureLength_np(output)
#img = cv2.imread('pic_03.jpg')
#img = cv2.imread('pic_0003.jpg')

#h=img.shape[0]
#w=img.shape[1]
#print("height=%d,wigth=%d"%(h,w))
'''
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#printImgGray(gray, h, w)

#cv2.threshold(img, 80, 255, 0, img)
#ret,output = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
ret,output = cv2.threshold(gray,60,1,cv2.THRESH_BINARY_INV)
#printImgGray(output, h, w)
v_z_c_size,v_z_c_max,v_o_c_size,v_o_c_max = findSqureLength_np(output)
'''


#cut_img_360_360(img)

#cut_img_299_299(img,100)
#cut_img_299_299(img,50)

#prepare_good_and_bad("pic_0001", 50)
#findSquareLength(output,h,w)

#plt.imshow(output)  
#plt.show()

# generate good pieces for training form OK dir
# gen_good_dir("./OK")

# generate good and bad pieces for training from NG and NG_marked dir
#gen_good_and_bad_dir("./NG")

# gen rotated pictures from a dir
gen_rotate_pic_dir("./bad_0313_02",90)
gen_rotate_pic_dir("./bad_0313_02",180)
gen_rotate_pic_dir("./bad_0313_02",270)
