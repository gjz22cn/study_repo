#!/usr/bin/python
#coding=utf-8
import tensorflow as tf
import numpy as np
import hashlib
import sys
import binascii
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', './sha2_logs', 'Summaries directory')


tf.logging.set_verbosity(tf.logging.ERROR)              #日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold=np.nan)                    #打印内容不限制长度

# prepare traning data
count = 10000
gd = 0.001
train_loops = 1000000
layer=15
#input_s = 864
input_s = 1280
output_s = 32

def bitinit(cnt):
    x = np.random.randint(0,2,size=(cnt,1024));
    for row in range(0,cnt):
        for col in range(0,640):
            x[row][col] = float(x[row][col])
        for col in range(640,1024):
            x[row][col] = 0
        x[row][640] = 1
        x[row][1013] = 1
        x[row][1014] = 1
        x[row][1017] = 1
    return x

def bithash(cnt,x):
    print("=============================================================================")
    y = np.random.randint(0,2,size=(cnt,256));
    x1 = [bytearray(80) for _ in range(cnt)]
    y1 = [bytearray(32) for _ in range(cnt)]
    for row in range(0,cnt):
        for col in range(0,80):
            v = 0 + x[row][8*col]*(1<<7)
            v = v + x[row][8*col+1]*(1<<6)
            v = v + x[row][8*col+2]*(1<<5)
            v = v + x[row][8*col+3]*(1<<4)
            v = v + x[row][8*col+4]*(1<<3)
            v = v + x[row][8*col+5]*(1<<2)
            v = v + x[row][8*col+6]*(1<<1)
            v = v + x[row][8*col+7]*(1<<0)
            x1[row][col] = v
        
        hash = hashlib.sha256(hashlib.sha256(x1[row]).digest()).digest()
        #print("hash[%d]-0:"%(row)+hash.encode('hex_codec'))
        #print("hash[%d]-1:"%(row)+hash[::-1].encode('hex_codec'))
        
        for col in range(0,32):
            #v = ord(hash[col])
            #v = ord(h[::-1][col])
            v = hash[::-1][col]
            y[row][8*col] = (v>>7) & 1
            y[row][8*col+1] = (v>>6) & 1
            y[row][8*col+2] = (v>>5) & 1
            y[row][8*col+3] = (v>>4) & 1
            y[row][8*col+4] = (v>>3) & 1
            y[row][8*col+5] = (v>>2) & 1
            y[row][8*col+6] = (v>>1) & 1
            y[row][8*col+7] = (v>>0) & 1
    return y

def printx(cnt,x):
    print("=============================================================================")
    for row in range(0,cnt):
        print("x[%d]:"%(row),end="")
        for col in range(0,80):
            v = 0 + x[row][8*col]*(1<<7)
            v = v + x[row][8*col+1]*(1<<6)
            v = v + x[row][8*col+2]*(1<<5)
            v = v + x[row][8*col+3]*(1<<4)
            v = v + x[row][8*col+4]*(1<<3)
            v = v + x[row][8*col+5]*(1<<2)
            v = v + x[row][8*col+6]*(1<<1)
            v = v + x[row][8*col+7]*(1<<0)
            sys.stdout.write("%02x"%(v))
        print("")

def printy(cnt,y):
    print("=============================================================================")
    for row in range(0,cnt):
        print("y[%d]:"%(row),end="")
        for col in range(0,32):
            v = 0 + y[row][8*col]*(1<<7)
            v = v + y[row][8*col+1]*(1<<6)
            v = v + y[row][8*col+2]*(1<<5)
            v = v + y[row][8*col+3]*(1<<4)
            v = v + y[row][8*col+4]*(1<<3)
            v = v + y[row][8*col+5]*(1<<2)
            v = v + y[row][8*col+6]*(1<<1)
            v = v + y[row][8*col+7]*(1<<0)
            sys.stdout.write("%02x"%(v))
        print("")

#t_x = np.random.randint(0,2,size=(count,640));
#y = np.random.randint(0,2,size=(count,256));
#pre_x = bitinit();
#t_x = pre_x.astype(np.float32)
#printx(t_x)
#t_y = bithash(pre_x).astype(np.float32)
#printy(t_y)

def prepare_test_data():
    x = np.random.randint(0,2,size=(count,input_s));
    y = np.random.randint(0,2,size=(count,32));

    x_t = bitinit(1)
    y_t = bithash(1,x_t)
    printx(x_t)
    printy(y_t)
    
    for col in range(0,32):
        y[0][col] = x_t[0][608+col]
    
    for col in range(0,608):
        x[0][col] = x_t[0][col]
    
    for col in range(0,256):
        x[0][608+col] = y_t[0][col]

    for row in range(1,count):
        for col in range(0,input_s):
            x[row][col] = x[0][col]

        for col in range(0,32):
            y[row][col] = y[0][col]

    return x.astype(np.float32),y.astype(np.float32)

def prepare_training_data():
    x = np.random.randint(0,2,size=(count,input_s));
    y = np.random.randint(0,2,size=(count,32));

    x_t = bitinit(count)
    y_t = bithash(count,x_t)

    for row in range(0,count):
        for col in range(0,1024):
            x[row][col] = x_t[row][col]

        for col in range(0,256):
            x[row][1024+col] = y_t[row][col]

        for col in range(0,32):
            y[row][col] = x[row][608+col]
            x[row][608+col] = 0

    return x.astype(np.float32),y.astype(np.float32)

def prepare_training_data1():
    x = np.random.randint(0,2,size=(count,input_s));
    y = np.random.randint(0,2,size=(count,32));

    x_t = bitinit(count)
    y_t = bithash(count,x_t)

    for row in range(0,count):
        for col in range(0,32):
            y[row][col] = x_t[row][608+col]

        for col in range(0,608):
            x[row][col] = x_t[row][col]

        for col in range(0,256):
            x[row][608+col] = y_t[row][col]

    return x.astype(np.float32),y.astype(np.float32)

t_x,t_y = prepare_training_data()


# define inpuource ~/tensorflow/bin/activate
x = tf.placeholder(tf.float32, shape=[None,input_s])
y_ = tf.placeholder(tf.float32, shape=[None,output_s])


'''
W_l = tf.Variable(tf.random_normal([input_s,32]), name="weights_l")
b_l = tf.Variable(tf.zeros([1,32]), name="biases_l")
prediction  = tf.nn.tanh(tf.matmul(x,W_l) + b_l)
'''

W = tf.Variable(tf.random_normal([layer-1,input_s,input_s]), name="weights")
#W = tf.Variable(tf.constant(1,tf.float32,[layer-1,1024,1024]), name="weights")
b = tf.Variable(tf.zeros([layer-1,input_s]), name="biases")

y_t = tf.nn.sigmoid(tf.matmul(x,W[0]) + b[0])
#y_t = tf.nn.tanh(tf.matmul(x,W[0]) + b[0])
#y_t = tf.nn.relu(tf.matmul(x,W[0]) + b[0])
#y_t = tf.matmul(x,W[0]) + b[0]
for i in range(1,layer-1):
    y_t = tf.nn.sigmoid(tf.matmul(y_t,W[i]) + b[i])
    #y_t = tf.nn.tanh(tf.matmul(y_t,W[i]) + b[i])
    #y_t = tf.nn.relu(tf.matmul(y_t,W[i]) + b[i])
    #y_t = tf.matmul(y_t,W[i]) + b[i]

W_l = tf.Variable(tf.random_normal([input_s,output_s]), name="weights_l")
#W_l = tf.Variable(tf.constant(1,tf.float32,[1024,256]), name="weights_l")
b_l = tf.Variable(tf.zeros([output_s]), name="biases_l")
prediction = tf.nn.sigmoid(tf.matmul(y_t,W_l) + b_l)
#prediction = tf.nn.tanh(tf.matmul(y_t,W_l) + b_l)
#prediction = tf.nn.relu(tf.matmul(y_t,W_l) + b_l)
#prediction = tf.matmul(y_t,W_l) + b_l

loss = tf.reduce_sum(tf.square(y_ - prediction))

#optimizer = tf.train.GradientDescentOptimizer(gd)
#optimizer = tf.train.AdagradOptimizer(gd)
#optimizer = tf.train.MomentumOptimizer(gd)
optimizer = tf.train.AdamOptimizer(gd)
#optimizer = tf.train.FtrlOptimizer(gd)
#optimizer = tf.train.RMSPropOptimizer(gd)
train_op = optimizer.minimize(loss)                     #训练的结果是使得损失函数最小

# Add ops to save and restore all the variables.
#saver = tf.train.Saver({"weights":W,"weights_l":W_l,"biases":b,"biases_l":b_l})
saver = tf.train.Saver()

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def train(stage):
    sess = tf.InteractiveSession()
    loss_min = 5000000
    loss_min_endloop = 5000000
    loss_t = 1
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    #merged = tf.merge_all_summaries()
    #train_writer = tf.train.SummaryWriter(FLAGS.summari{"weights":W,"weights_l":W_l,"biases":b,"biases_l":b_l}es_dir + '/train', sess.graph)
    #test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    tf.initialize_all_variables().run()
    #sess.run(tf.global_variables_initializer())             #变量初始化
    
    # Restore variables from disk.
    if stage>0:
        saver.restore(sess, "./sha2_model.ckpt")
        print("Model restored.")
    
    print(sess.run([W_l[input_s-1],b_l]))
        
    for i in range(1,train_loops):
        sess.run(train_op, feed_dict={x:t_x, y_:t_y})
        loss_t = sess.run(loss,feed_dict={x:t_x, y_:t_y})
        print("i=%-10d, loss=%f"%(i,loss_t))
        if loss_t < loss_min:
            loss_min = loss_t
            
        # Save the variables to disk.
        if i%100 == 0:
            #print(sess.run([W_l[input_s-1],b_l]))
            save_path = saver.save(sess, "./sha2_model.ckpt")
            print("Model saved in file: ", save_path)

            if loss_t < loss_min_endloop:
                loss_min_endloop = loss_t
                save_path = saver.save(sess, "./sha3_min_loss_model.ckpt")
                print("Model min saved in file: ", save_path)

            print("i=%-10d, loss_min=%f, loss_min_endloop=%f"%(i,loss_min,loss_min_endloop))
            print("=============================================================================")

def test():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())             #变量初始化
        
        # Restore variables from disk.
        saver.restore(sess, "./sha2_model.ckpt")
        print("Model restored.")
        
        print(sess.run(loss,feed_dict={x:t_x, y_:t_y}))
        print(sess.run(y,feed_dict={x:t_x, y_:t_y}))

def train_loop():
    #train(0)
    for i in range(0,train_loop_l2):
        print("=============================================================================")
        start1 = time.clock()
        train(1)
        elapsed1 = (time.clock() - start1)
        print("Step: %d"%(i)+", time used:",elapsed1)

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train(0)

if __name__ == '__main__':
  tf.app.run()
