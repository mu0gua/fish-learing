#!/usr/bin/python3
# -*- coding:utf-8 -*-
#   ____    ____    ______________
#  |    |  |    |  |              |
#  |    |  |    |  |_____    _____|
#  |    |__|    |       |    |
#  |     __     |       |    |
#  |    |  |    |       |    |
#  |    |  |    |       |    |
#  |____|  |____|       |____|
#
# fileName:CNN_LSTM 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: {文件作用描述｝
# creatData:2019/5/15

import tensorflow as tf


class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0, num_hidden=100):
        # PLACEHOLDERS
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")  # X - The Data
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout

        l2_loss = tf.constant(0.0)  # Keeping track of l2 regularization loss

        # 1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.compat.v1.name_scope("embedding"):
            self.W = tf.Variable(tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(params=self.W, ids=self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 2. CONVOLUTION LAYER + MAXPOOLING LAYER (per filter) ###############################
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(input=self.embedded_chars_expanded, filters=W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # NON-LINEARITY
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # MAXPOOLING
                pooled = tf.nn.max_pool2d(input=h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # COMBINING POOLED FEATURES
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 3. DROPOUT LAYER ###################################################################
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, 1 - (self.dropout_keep_prob))

        # 4. LSTM LAYER ######################################################################
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        self.h_drop_exp = tf.expand_dims(self.h_drop, -1)
        val, state = tf.compat.v1.nn.dynamic_rnn(cell, self.h_drop_exp, dtype=tf.float32)

        # embed()

        val2 = tf.transpose(a=val, perm=[1, 0, 2])
        last = tf.gather(val2, int(val2.get_shape()[0]) - 1)

        out_weight = tf.Variable(tf.random.normal([num_hidden, num_classes]))
        out_bias = tf.Variable(tf.random.normal([num_classes]))

        with tf.compat.v1.name_scope("output"):
            # lstm_final_output = val[-1]
            # embed()
            self.scores = tf.compat.v1.nn.xw_plus_b(last, out_weight, out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.compat.v1.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.stop_gradient(self.input_y))
            self.loss = tf.reduce_mean(input_tensor=self.losses, name="loss")

        with tf.compat.v1.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(input=self.predictions, axis=1), tf.argmax(input=self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(input_tensor=tf.cast(self.correct_pred, "float"), name="accuracy")

        # embed()
