#!/usr/bin/env python
# coding=utf-8

from utils import DataSet
import tensorflow as tf

IMAGE_WIDTH, IMAGE_HEIGHT = 96, 96
N_CLASS = 30
BATCH_SIZE = 64
TRAINING_ITERS = 1000000
DISPLAY_STEP = 20

def create_CNN(Data, Weights, Biases, Dropout):
    
    def conv2d(img, w, b):
        cov_Img = tf.nn.conv2d(img, w, strides = [1, 1, 1, 1], padding = 'SAME')
        add_Res = tf.nn.bias_add(cov_Img, b)
        return tf.nn.relu(add_Res)
    
    def max_pool(img, k):
        return tf.nn.max_pool(img, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

    # Reshape Data 
    Data = tf.reshape(Data, shape = [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    
    # Convolution Network
    conv1 = conv2d(Data, Weights['wc1'], Biases['bc1'])
    conv1 = max_pool(conv1, k = 2)
    conv1 = tf.nn.dropout(conv1, Dropout['dc1'])

    conv2 = conv2d(conv1, Weights['wc2'], Biases['bc2'])
    conv2 = max_pool(conv2, k = 2)
    conv2 = tf.nn.dropout(conv2, Dropout['dc2'])

    conv3 = conv2d(conv2, Weights['wc3'], Biases['bc3'])
    conv3 = max_pool(conv3, k = 2)
    conv3 = tf.nn.dropout(conv3, Dropout['dc3'])

    # Hidden Network
    h1 = tf.reshape(conv3, [-1, Weights['wh1'].get_shape().as_list()[0]])
    h1 = tf.nn.relu(tf.add(tf.matmul(h1, Weights['wh1']), Biases['bh1']))
    h1 = tf.nn.dropout(h1, Dropout['dh1'])

    h2 = tf.nn.relu(tf.add(tf.matmul(h1, Weights['wh2']), Biases['bh2']))
    h2 = tf.nn.dropout(h2, Dropout['dh2'])

    # Output Layer 
    out = tf.add(tf.matmul(h2, Weights['out']), Biases['out'])
    return out


Weight_Dicts = {
    'wc1' : tf.Variable(tf.random_normal([2, 2, 1, 32])),
    'wc2' : tf.Variable(tf.random_normal([2, 2, 32, 64])),
    'wc3' : tf.Variable(tf.random_normal([2, 2, 64, 128])),
    'wh1' : tf.Variable(tf.random_normal([(IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 128, 1000])),
    'wh2' : tf.Variable(tf.random_normal([1000, 1000])),
    'out' : tf.Variable(tf.random_normal([1000, N_CLASS]))
}

Biases_Dict = {
    'bc1' : tf.Variable(tf.random_normal([32])),
    'bc2' : tf.Variable(tf.random_normal([64])),
    'bc3' : tf.Variable(tf.random_normal([128])),
    'bh1' : tf.Variable(tf.random_normal([1000])),
    'bh2' : tf.Variable(tf.random_normal([1000])),
    'out' : tf.Variable(tf.random_normal([N_CLASS]))
}

Dropout_Dict = {
    'dc1' : 0.1,
    'dc2' : 0.2,
    'dc3' : 0.3,
    'dh1' : 0.5,
    'dh2' : 1.0
}

def main():
    imgs = tf.placeholder(tf.float32, [None, IMAGE_WIDTH * IMAGE_HEIGHT])
    keys = tf.placeholder(tf.float32, [None, N_CLASS])

    train_model = create_CNN(imgs, Weight_Dicts, Biases_Dict, Dropout_Dict)
    # Define loss and optimizer 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_model, keys))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    # Evaluate the train model
    correct_model = tf.equal(tf.argmax(train_model, 1), tf.argmax(keys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))

    init = tf.initialize_all_variables()
    
    obj = DataSet()
    obj.load()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * BATCH_SIZE < TRAINING_ITERS:
            batch_imgs, batch_keys = obj.next_batch(BATCH_SIZE)
            sess.run(optimizer, feed_dict = {imgs : batch_imgs, keys : batch_keys})
            if step % DISPLAY_STEP == 0:
                acc = sess.run(accuracy, feed_dict = {imgs : batch_imgs, keys : batch_keys})
                loss = sess.run(cost, feed_dict = {imgs : batch_imgs, keys : batch_keys})
                print "Iter " + str(step * BATCH_SIZE) + ", MiniBatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc)
            step  = step + 1


if __name__ == '__main__':
    main()
