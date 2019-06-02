#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf

# 加载mnist_inference.py 和 mnist_train.py中定义的常量和函数
import inference
import train
import evaldata
import data

# 每10秒加载一次最新的模型， 并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
    with open('./model/frozen_model.pb', "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [
            mnist.count,           # 第一维表示样例的个数
            inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
    
        y = tf.GraphDef().get_tensor_by_name("out:0")
        validate_feed = {x: np.reshape(mnist.Image, (mnist.count, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS)),
                         y_: mnist.Label}
        # 直接通过调用封装好的函数来计算前向传播的结果。
        # 因为测试时不关注正则损失的值，所以这里用于计算正则化损失的函数被设置为None。

        # 使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variable_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        #while True:
    with tf.Session() as sess:
        accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
        print("validation accuracy = %f" % (accuracy_score))
                #else:
                #   print("No checkpoint file found")
                #   return
            #time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = evaldata.evaldata()
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
