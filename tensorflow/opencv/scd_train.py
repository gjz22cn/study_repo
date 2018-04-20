import tensorflow as tf
import scd_input as inputData
import scd_model as model
import numpy as np
import os

IMAGE_SIZE = 224

def run_training():
    data_dir = 'E:/james/opencv_20180415/train/data/'

    save_mode_dir = "./scd_modle"
    model_name = "scd"
    if not os.path.exists(save_mode_dir):
        os.makedirs(save_mode_dir)
    
    model_saver = tf.train.Saver()

    image,label = inputData.get_files(data_dir)
    image_batches,label_batches = inputData.get_batches(image,label,IMAGE_SIZE,IMAGE_SIZE,64,80)
    

    p = model.mmodel(image_batches)
    cost = model.loss(p,label_batches)
    train_op = model.training(cost,0.001)
    acc = model.get_accuracy(p,label_batches)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    model_path=os.path.join(save_mode_dir,model_name)
    model_saver.restore(sess,model_path)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
        for step in np.arange(10000):
            print("step:%6d "%(step), end='')
            if coord.should_stop():
                break
            _,train_acc,train_loss = sess.run([train_op,acc,cost])
            print("loss:{}\taccuracy:{}".format(train_loss,train_acc))
        
        model_saver.save(sess,os.path.join(save_mode_dir,model_name))
        print("model saved sucessfully")
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

run_training()
