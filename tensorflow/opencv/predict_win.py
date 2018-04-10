import tensorflow as tf
import sys
import os
import time
import cv2
import numpy as np

g_width = 4896
g_height = 3034
g_top_margin = 48
g_buttom_margin = 104
g_left_margin = 48
g_right_margin = 48
g_rows = 6
g_cols = 10
g_padding = 10

# 程序运行目录
g_run_dir = "E:/ai/test/"
# 需要检测的文件夹
g_image_dir = "E:/el/20180311/NightFlight/NG/"

#=====================================================
g_graph_file = g_run_dir + "output_graph.pb"
g_label_file = g_run_dir + "output_labels.txt"
g_result_dir = g_run_dir + "result/"
g_result_show_dir = g_run_dir + "result/show/"
g_result_show_good_dir = g_result_show_dir + "good/"
g_result_show_bad_dir = g_result_show_dir + "bad/"
g_result_good_dir = g_run_dir + "result/good/"
g_result_bad_dir = g_run_dir + "result/bad/"


def get_filepath_filename_filext(filename):  
    (filepath,tempfilename) = os.path.split(filename);  
    (shotname,extension) = os.path.splitext(tempfilename);  
    return filepath,shotname,extension

# 命令行参数，传入要判断的图片路径
#file_ori = sys.argv[1]
#print(image_file)

# 读取图像
#image = tf.gfile.FastGFile(fileName, 'rb').read()
#img_0 = tf.image.decode_jpeg(image)

# 加载图像分类标签
labels = []
for label in tf.gfile.GFile(g_label_file):
    labels.append(label.rstrip())

# 加载Graph
with tf.gfile.FastGFile(g_graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    '''
    predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
    for index in range(0,predict.shape[1]):
        print(labels[index], predict[0][index])

    # sort
    top = predict[0].argsort()[-len(predict[0]):][::-1]
    for index in top:
        human_string = labels[index]
        score = predict[0][index]
        print(human_string, score)
    '''
    
    def predict_image_piece(image,piece_file_name,startX,startY,sizeX,sizeY):
        bad_idx = 0
        bad_file = ""
        new_img = image[startY-g_padding:startY+sizeY+g_padding, startX-g_padding:startX+sizeX+g_padding]
        good_piece_file = g_result_good_dir + piece_file_name
        bad_piece_file = g_result_bad_dir + piece_file_name

        cv2.imwrite(good_piece_file, new_img)

        input_img = tf.gfile.FastGFile(good_piece_file, 'rb').read()
        
        predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': input_img})
        
        for index in range(0,predict.shape[1]):
            if labels[index] == "bad":
                bad_idx = index
                break
        

        if predict[0][bad_idx] > 0.5:
            print("[%6d, %6d]"%(startX,startY), "%4s"%(labels[bad_idx]), predict[0][bad_idx])
            bad_file = g_result_dir + piece_file_name
            #print("bad_file=%s"%(bad_file))
            os.remove(good_piece_file)
            cv2.imwrite(bad_piece_file, new_img)
            return 1

        return 0

    def predict_file(fileName):
        filepath,shortname,extension = get_filepath_filename_filext(fileName)
        
        if not extension == ".jpg":
            return 0
        
        img_ori = cv2.imread(fileName)

        img_w = img_ori.shape[1]
        img_h = img_ori.shape[0]

        if img_w != g_width or img_h != g_height:
            print("input image is %d*%d, but expected size if %d*%d."%(img_w,img_h,g_width,g_height))
            return

        step_x = int((g_width - g_left_margin - g_right_margin)/g_cols)
        step_y = int((g_height - g_top_margin - g_buttom_margin)/g_rows)
        start_x = g_left_margin
        start_y = g_top_margin
        
        bad_cnt = 0
        is_bad = 0
        
        print("===========predict file: %s============"%(fileName))
        start = time.clock()
        for row in range(0,g_rows):
            start_x = g_left_margin
            
            for col in range(0,g_cols):
                piece_file_name = shortname + "_" + str(row+1) + "_" + str(col+1) + extension
                is_bad = predict_image_piece(img_ori,piece_file_name,start_x,start_y,step_x,step_y)
                if is_bad > 0:
                    bad_cnt += 1 
                    #cv2.rectangle(img_ori, (start_x+20, start_y+20), (start_x+step_x-20, start_y+step_y-20), (0,0,255), 5)
                    cv2.circle(img_ori,(start_x+int(step_x/2),start_y+int(step_y/2)),int(step_x/4),(0,0,255),20)
                start_x += step_x
            
            start_y += step_y
        
        end = time.clock()
        print ("bad_cnt=%d, time=%fs"%(bad_cnt, end-start))

        if bad_cnt > 0:
            cv2.imwrite(g_result_show_bad_dir+shortname+"_result"+extension, img_ori)
            return 1
        else:
            cv2.imwrite(g_result_show_good_dir+shortname+"_result"+extension, img_ori)
            return 0

    def predict_dir(dirPath):
        list = os.listdir(dirPath)
        bad_file = 0
        total_file=len(list)

        for i in range(0,total_file):
            filePath = os.path.join(dirPath,list[i])
            if os.path.isfile(filePath):
                bad_file += predict_file(filePath)
                print("total_file=%d, processed_file=%d, bad_file=%d"%(total_file,i+1,bad_file))


    def prepare_env():
        if not os.path.isdir(g_result_dir):
            os.mkdir(g_result_dir)

        if not os.path.isdir(g_result_good_dir):
            os.mkdir(g_result_good_dir)

        if not os.path.isdir(g_result_bad_dir):
            os.mkdir(g_result_bad_dir)

        if not os.path.isdir(g_result_show_dir):
            os.mkdir(g_result_show_dir)

        if not os.path.isdir(g_result_show_good_dir):
            os.mkdir(g_result_show_good_dir)

        if not os.path.isdir(g_result_show_bad_dir):
            os.mkdir(g_result_show_bad_dir)

    prepare_env()
    predict_dir(g_image_dir)
    #predict_file(filePath)
