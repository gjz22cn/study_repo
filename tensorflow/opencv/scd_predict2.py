from skimage import io,transform
import tensorflow as tf
import numpy as np
 
 
path1 = "E:/data/datasets/flower_photos/daisy/5547758_eea9edfd54_n.jpg"
path2 = "E:/data/datasets/flower_photos/dandelion/7355522_b66e5d3078_m.jpg"
path3 = "E:/data/datasets/flower_photos/roses/394990940_7af082cf8d_n.jpg"
path4 = "E:/data/datasets/flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "E:/data/datasets/flower_photos/tulips/10791227_7168491604.jpg"
 
result_dict = {0:'good', 1:'yinliesuipian', 2:'quejiaobenbian', 
               3:'heibanheibian', 4:'duanluduanlu', 5:'xuhan',
               6:'dixiao', 7:'zhangwuhuahen', 8:'diepiancuowei',
               9:'liangbanbaoguang'}

model_dir = "E:/james/opencv_20180415/solarCellDetect/scd_modle2/"
 
w=100
h=100
c=3
 
def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)
 
with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
 
    saver = tf.train.import_meta_graph(model_dir+'model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint(model_dir))
 
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
 
    logits = graph.get_tensor_by_name("logits_eval:0")
 
    classification_result = sess.run(logits,feed_dict)
 
    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"朵花预测:"+result_dict[output[i]])
