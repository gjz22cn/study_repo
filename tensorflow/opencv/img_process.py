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
            print("%3d "%(image[i,j,0])),
        print('')


    print('-----------------------------------------')
    for i in range(0,h):
        for j in range(0,w):
            print("%3d "%(image[i,j,1])),
        print('')


    print('-----------------------------------------')
    for i in range(0,h):
        for j in range(0,w):
            print("%3d "%(image[i,j,2])),
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


def resize_dir_images(oriDir):
    size = 250
    dst_dir = "./resized/"
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    list = os.listdir(oriDir)
    for i in range(0,len(list)):
        filePath = os.path.join(oriDir,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            image=cv2.imread(filePath)
            newImg=cv2.resize(image,(size,size),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_dir+shortname+"_s"+str(size)+extension, newImg)

# generate gray images from a dir
def gray_dir_images(oriDir):
    dst_dir = "./gray/"
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    list = os.listdir(oriDir)
    for i in range(0,len(list)):
        filePath = os.path.join(oriDir,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            image=cv2.imread(filePath)
            grayImg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(dst_dir+shortname+"_gray"+extension, grayImg)


def adjustPiecePic(image_name,save_name):
    max_v = 60
    padding = 1

    image = cv2.imread(image_name)
    row = image.shape[0]
    col = image.shape[1]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h_r = np.add.reduce(gray,1)
    v_r = np.add.reduce(gray,0)
    h_r_fill_out = int(np.max(h_r)/10)
    v_r_fill_out = int(np.max(v_r)/10)

    left=[0,0]
    top=[0,0]
    right = [col-1,col-1]
    buttom = [row-1,row-1]
    min_left=0x10000000
    min_right=0x10000000
    min_top=0x10000000
    min_buttom=0x10000000
    for i in range(0,max_v):
        if h_r[i] <= h_r_fill_out:
            top[0] = i

        if v_r[i] <= v_r_fill_out:
            left[0] = i

    for i in range(row-1,row-1-max_v,-1):
        if h_r[i] <= h_r_fill_out:
            buttom[0] = i

    for i in range(col-1,col-1-max_v,-1):
        print("v_r[%d]=%d"%(i,v_r[i]))
        if v_r[i] <= v_r_fill_out:
            right[0] = i

    for i in range(top[0],max_v):
        if h_r[i] <= min_top:
            min_top = h_r[i]
            top[1] = i

    for i in range(left[0],max_v):
        if v_r[i] <= min_left:
            min_left = v_r[i]
            left[1] = i

    for i in range(buttom[0],row-1-max_v,-1):
        if h_r[i] <= min_buttom:
            min_buttom = h_r[i]
            buttom[1] = i

    for i in range(right[0],col-1-max_v,-1):
        if v_r[i] <= min_right:
            min_right = v_r[i]
            right[1] = i

    #print("L:%d,R:%d,T:%d,B:%d"%(left,right,top,buttom))

    new_img = image[top[1]-padding:buttom[1]+padding, left[1]-padding:right[1]+padding]
    cv2.imwrite(save_name, new_img)

def AdjustPiecePicDir(dirPath, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        filePath = os.path.join(dirPath,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            adjustPiecePic(filePath, dst_dir+shortname+extension)

def genHumain(srcDir,dstDir):
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
    
    #image_hh=cv2.imread("./opencv2_human/heiban_7.jpg")
    image_hh=cv2.imread("./opencv2_human/huahen_9.jpg")
    h_hh = image_hh.shape[0]
    w_hh = image_hh.shape[1]
    
    list = os.listdir(srcDir)
    for i in range(0,len(list)):        
        filePath = os.path.join(srcDir,list[i])
        if os.path.isfile(filePath):
            fpath,shortname,extension = get_filepath_filename_filext(filePath)
            image=cv2.imread(filePath)
            h = image.shape[0]
            w = image.shape[1]
            delta = 1
            for x in range(15+5*i,w-50-w_hh,40):
                for y in range(15+5*i,h-50-h_hh,40):
                    delta = delta + 1
                    image_new = image.copy()
                    for x_hh in range(0,h_hh):
                        for y_hh in range(0,w_hh):
                            if image_hh[x_hh,y_hh,0] < 200:
                                image_new[x+x_hh,y+y_hh] = image_hh[x_hh,y_hh] - np.array([delta,delta,delta])
                    image_new = cv2.GaussianBlur(image_new,(3,3),0)
                    cv2.imwrite(dstDir+shortname+"_human_"+str(x)+"_"+str(y)+extension, image_new)

def test(fileName):
    image=cv2.imread(fileName)
    row = image.shape[0]
    col = image.shape[1]
    print('-----------------------------------------')
    #for k in range(0,15):
    #    for i in range(0,26-k):
    #        image[i,col-k-1] = np.array([255, 255, 255])

    image[0,1] = np.array([255, 255, 255])
    image[2,0] = np.array([255, 255, 255])
    image[3,0] = np.array([255, 255, 255])
    image[4,1] = np.array([255, 255, 255])

    for i in range(0,row):
        for j in range(0,col):
            print("%2x"%(image[i,j,0])),
        print('')
    #cv2.imwrite("./huahen_x.jpg", image)
    #cv2.imwrite(fileName, image)
    return
    for i in range(0,row):
        for j in range(0,col):
            if image[i,j,0] > 0x80:
                image[i,j] = np.array([255, 255, 255])
    plt.imshow(image)
    plt.show()
    #cv2.imwrite("./huahen_x.jpg", image)
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

####################################
#   Cut Image To Small Piece
####################################
def genPiecePerDir(srcDir,dstDir,size,step):
    str_s_s = "_s_"+str(size)+"_"+str(step)+"_"
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
    
    list = os.listdir(srcDir)
    for i in range(0,len(list)):        
        file_name = os.path.join(srcDir,list[i])
        if os.path.isfile(file_name):
            fpath,short_name,ext = get_filepath_filename_filext(file_name)
            if not (ext == ".jpg"):
                continue

            image=cv2.imread(file_name)
            h = image.shape[0]
            w = image.shape[1]
            for x in range(0,h,step):
                for y in range(0,w,step):
                    start_x = x
                    if start_x + size > h:
                        start_x = h - size

                    start_y = y
                    if start_y + size > w:
                        start_y = w - size

                    img_piece = image[start_x:start_x+size, start_y:start_y+size]
                    save_name = os.path.join(dstDir,short_name+str_s_s+str(start_x)+"_"+str(start_y)+ext)
                    cv2.imwrite(save_name, img_piece)

def genTrainPieceDir(srcDir,dstDir):
    lables = ['1_yinliesuipian',
            '2_quejiaobenbian',
            '3_heibanheibian',
            '4_duanluduanlu',
            '5_xuhan',
            '6_dixiao',
            '7_zhangwuhuahen',
            '8_diepiancuowei',
            '9_liangbanbaoguang']

    if not os.path.isdir(srcDir):
        return

    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)

    for lable in lables:
        src_dir = os.path.join(srcDir,lable)
        if os.path.isdir(src_dir):
            genPiecePerDir(src_dir,os.path.join(dstDir,lable),128,128)


#genTrainPieceDir("./../opencv/data_0419","./../opencv/piece_t")
#test("./opencv2_human/huahen_9.jpg")
#genHumain("./good/","./humain/")
#AdjustPiecePicDir("./sample/")
#AdjustPiecePicDir("./data/bad/", "./adjust/bad/")

#adjustPiecePic("./sample2.jpg","./adjust.jpg")

'''
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#printImgGray(gray, h, w)

#cv2.threshold(img, 80, 255, 0, img)
#ret,output = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
ret,output = cv2.threshold(gray,60,1,cv2.THRESH_BINARY_INV)
#printImgGray(output, h, w)
v_z_c_size,v_z_c_max,v_o_c_size,v_o_c_max = findSqureLength_np(output)
image = cv2.imread("sample2.jpg")
mean = np.mean(piece)
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

# resize images in a dir
#resize_dir_images("./data/bad")


# generate gray images from a dir
#gray_dir_images("./data/bad")


####################################
#         Resize Image
####################################
def resizeImageFile(fileName,saveName):
    image=cv2.imread(fileName)
    new_image = cv2.resize(image, (299,299), interpolation=cv2.INTER_AREA)
    cv2.imwrite(saveName,new_image)

def resizeImageDir(dirPath,dstDir):
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
        
    list = os.listdir(dirPath)
    for i in range(0,len(list)):
        file_name = os.path.join(dirPath,list[i])
        if os.path.isfile(file_name):
            fpath,shortname,extension = get_file_name_filename_filext(file_name)
            if extension == ".jpg":
                resizeImageFile(file_name,file_name)

def resizeTrainDir(srcPath,dstPath):
    lables = ['0_good',
            '1_yinliesuipian',
            '2_quejiaobenbian',
            '3_heibanheibian',
            '4_duanluduanlu',
            '5_xuhan',
            '6_dixiao',
            '7_huahen',
            '8_diepiancuowei',
            '9_liangbanbaoguang',
            '10_zhangwu']

    for lable in lables:
        if not os.path.isdir(dstPath):
            os.mkdir(dstPath)
        
        src_dir = os.path.join(srcPath, lable)
        if not os.path.isdir(src_dir):
            continue
        
        resizeImageDir(src_dir, dstPath+label+"/")

####################################
#         Rotate Image
####################################

g_labelsNeedRoate = ['1_yinliesuipian',
                     '2_quejiaobenbian',
                     '3_heibanheibian',
                     '5_xuhan',
                     '7_huahen',
                     '8_diepiancuowei',
                     '9_liangbanbaoguang',
                     '10_zhangwu']

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


def addRotatePicForTrainDir(trainPath):
    for label in g_labelsNeedRoate:   
        src_dir = trainPath + label + "/"
        if not os.path.isdir(src_dir):
            continue
        
        addRoatedPicForDir(src_dir,180)

#addRotatePicForTrainDir("./../opencv/data_0419/")

image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))
sobelcombine = cv2.bitwise_or(sobelx,sobely)
ret,th1=cv2.threshold(sobelcombine,85,255,cv2.THRESH_BINARY)
cv2.imshow('result', np.hstack([th,sobelcombine]))
if(cv2.waitkey(0)==27):
    cv2.destroyAllWindows()




