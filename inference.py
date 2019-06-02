import tensorflow as tf
INPUT_NODE = 1024  #输入图片大小
OUTPUT_NODE = 31   #输出节点个数

IMAGE_SIZE = 32    #图片规格
NUM_CHANNELS = 1   #图片通道数
NUM_LABELS = 31    #标签个数

CONV1_DEEP = 8 #卷积层1深度
CONV1_SIZE = 5     #卷积核1的大小

CONV2_DEEP = 20   #卷积层2深度
CONV2_SIZE = 5     #卷积核2大小

FC_SIZE = 100   #全连接层节点个数



def inference(input_tensor,train,regularizer):

    with tf.variable_scope('layer1-conv1'):
        #5*5*1*6卷积核，并从截断的正态分布中输出随机值进行赋值，标准偏差为0.1
        conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #将偏置初始化为0
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        
        #实现卷积，input_tensor：输入的图像，
        #它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape
        #batch：图片数量，channels：图片通道数
        #conv1_weights：卷积核结构
        #它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape
        #in_channels：图片通道数，out_channels：输出图像个数，即输出的深度
        #strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
        #padding：当其为‘SAME’时，表示卷积核可以停留在图像边缘
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        #tf.nn.bias_add：将偏置加到卷积上
        #计算激活函数relu，即max(features, 0)。即将矩阵中每行的非最大值置0
        relu1 = tf.nn.elu(tf.nn.bias_add(conv1,conv1_biases))


    with tf.name_scope('layer2-pool1'):
        #最大值池化，ksize：池化窗口大小，strides：滑动步长
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        #同conv1
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights,strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.elu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        #同pool1
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #pool2后输出的形状
    pool_shape = pool2.get_shape().as_list()
    #长，宽，通道，个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #reshape为[x, nodes]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes,FC_SIZE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))

        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.nn.elu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    return logit
