#!/usr/bin/env python
# coding=utf-8

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""LeNet training"""

import os
import time
import sys
import argparse


import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from npu_bridge.npu_init import *

def get_config(args):
    """
    This function parses the command line arguments
    Args:
        args(str) : The command line arguments

    Returns
        args (dict): The arguments parsed into a dictionary
    """
    
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--random_remove", default='False', help="whether to remove random op in training.")
    parser.add_argument("--data_path", default='MNIST', help="training input data path.")
    parser.add_argument("--output_path", default='output', help="training input data path.")

    parser.add_argument("--batch_size", type=int, help="train batch size.")
    parser.add_argument("--learing_rata", type=float, help="learning rate.")
    parser.add_argument("--steps", type=int, help="training steps")
    parser.add_argument("--ckpt_count", type=int, help="save checkpoiont max counts.")
    parser.add_argument("--epochs", type=int, help="epoch number.")

    args, unknown = parser.parse_known_args(args)

    return args
    
class LeNet(object):

    def __init__(self):
        pass


    def create(self, x):
        x = tf.reshape(x, [(- 1), 28, 28, 1])
        with tf.variable_scope('layer_1') as scope:
            w_1 = tf.get_variable('weights', shape=[5, 5, 1, 6])
            b_1 = tf.get_variable('bias', shape=[6])
        conv_1 = tf.nn.conv2d(x, w_1, strides=[1, 1, 1, 1], padding='SAME')
        act_1 = tf.sigmoid(tf.nn.bias_add(conv_1, b_1))
        max_pool_1 = tf.nn.max_pool(act_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('layer_2') as scope:
            w_2 = tf.get_variable('weights', shape=[5, 5, 6, 16])
            b_2 = tf.get_variable('bias', shape=[16])
        conv_2 = tf.nn.conv2d(max_pool_1, w_2, strides=[1, 1, 1, 1], padding='SAME')
        act_2 = tf.sigmoid(tf.nn.bias_add(conv_2, b_2))
        max_pool_2 = tf.nn.max_pool(act_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        flatten = tf.reshape(max_pool_2, shape=[(- 1), ((7 * 7) * 16)])
        with tf.variable_scope('fc_1') as scope:
            w_fc_1 = tf.get_variable('weight', shape=[((7 * 7) * 16), 120])
            b_fc_1 = tf.get_variable('bias', shape=[120], trainable=True)
        fc_1 = tf.nn.xw_plus_b(flatten, w_fc_1, b_fc_1)
        act_fc_1 = tf.nn.sigmoid(fc_1)
        with tf.variable_scope('fc_2') as scope:
            w_fc_2 = tf.get_variable('weight', shape=[120, 84])
            b_fc_2 = tf.get_variable('bias', shape=[84], trainable=True)
        fc_2 = tf.nn.xw_plus_b(act_fc_1, w_fc_2, b_fc_2)
        act_fc_2 = tf.nn.sigmoid(fc_2)
        with tf.variable_scope('fc_3') as scope:
            w_fc_3 = tf.get_variable('weight', shape=[84, 10])
            b_fc_3 = tf.get_variable('bias', shape=[10], trainable=True)
            tf.summary.histogram('weight', w_fc_3)
            tf.summary.histogram('bias', b_fc_3)
        fc_3 = tf.nn.xw_plus_b(act_fc_2, w_fc_3, b_fc_3)
        return fc_3

def train(args):

    batch_size = 64
    steps = 1000
    epochs = 5

    if args.batch_size is not None and args.batch_size > 0:
        batch_size = args.batch_size

    if args.steps is not None and args.steps > 0: 
        steps = args.steps

    if args.epochs is not None and args.epochs > 0: 
        epochs = args.epochs

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [batch_size, 10])
    le = LeNet()
    y_ = le.create(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    
    #动态loss Scale
    loss_scale_manager_dynamic = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    
    #静态LossScale
    loss_scale_manager_static = FixedLossScaleManager(loss_scale=2**32)
    
    optimizer = tf.train.AdamOptimizer()
    optimizer = NPULossScaleOptimizer(optimizer,loss_scale_manager_dynamic)
    train_op = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    mnist = input_data.read_data_sets(args.data_path, one_hot=True)

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("./dump_output") 
    # enable_dump_debug：是否开启溢出检测功能
    custom_op.parameter_map["enable_dump_debug"].b = True
    # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
    custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all") 
    
    #打印Loss Scale
    lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        for epoch in range(epochs):
            for step in range(steps):
                start_time = time.time()
                (batch_xs, batch_ys) = mnist.train.next_batch(batch_size)
                ( l_s,loss_value, _) = sess.run([lossScale,loss, train_op], feed_dict={x: batch_xs, y: batch_ys})
                cost_time = time.time()-start_time
                print("epoch: %d step: %d loss: %.8f sec/step: %.5f" % (epoch, step, loss_value, cost_time))
                print('loss_scale is: ', l_s)
                i += 1
        test_acc = 0
        test_count = 0
        for _ in range(10):
            (batch_xs, batch_ys) = mnist.test.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            test_acc += acc
            test_count += 1
            
        print('accuracy : {}'.format((test_acc / test_count)))
        saver.save(sess, os.path.join(args.output_path, "mode.ckpt"))
        tf.io.write_graph(sess.graph, args.output_path, 'graph.pbtxt', as_text=True)

def main():
    args = get_config(sys.argv[1:])
    train(args)

if __name__ == '__main__':
    main()
