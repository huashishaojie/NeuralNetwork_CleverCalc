import os
import tensorflow as tf
import inference
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib import layers
import data

BATCH_SIZE = 100 #一次训练图像数量
LEARNING_RATE_BASE = 0.015
LEARNING_RATE_DECAY = 0.995
REGULARZTION_RATE = 0.0001
TRAINING_STEPS = 60000 #训练遍数
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [
                        BATCH_SIZE,
                        inference.IMAGE_SIZE,
                        inference.IMAGE_SIZE,
                        inference.NUM_CHANNELS],name='x-input')
    y_ = tf.placeholder(tf.float32,[
                        None,inference.OUTPUT_NODE],name='y-input')
    #规则化可以帮助防止过度配合，提高模型的适用性。（让模型无法完美匹配所有的训练项。）（使用规则来使用尽量少的变量去拟合数据）
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZTION_RATE)
    y = inference.inference(x,True,regularizer)

    global_step = tf.Variable(0,trainable=False)
    #tf.train.ExponentialMovingAverage(decay, steps)
    #这个函数用于更新参数，就是采用滑动平均的方法更新参数。这个函数初始化需要提供一个衰减速率（decay）
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #tf.trainable_variables()返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.count / BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,inference.IMAGE_SIZE,inference.IMAGE_SIZE,inference.NUM_CHANNELS))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})

            if i % 100 == 0:
                print("After %d training step(s), loss is %g" %(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            
        print("After %d training step(s), loss is %g" %(TRAINING_STEPS,loss_value))
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv = None):
    mnist = data.data()
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
