
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet

from npu_bridge.npu_init import *

#import moxing as mox

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER
    obs_target = 'obs://jeffdingtons/canncamp_model/'#这里填写你的obs地址


    saver = tf.train.Saver()
    #初始化NPU资源
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    for i in range(max_iter):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(lenet.train_accuracy,feed_dict={
                lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
    save_path = saver.save(sess, parameter_path)
    #mox.file.copy_parallel('./checkpoint/',obs_target)

if __name__ == '__main__':
    main()


