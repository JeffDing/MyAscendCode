import tensorflow as tf
from PIL import Image,ImageOps
import numpy as np
from lenet import Lenet
import config as cfg

class inference:
    def __init__(self):
        self.lenet = Lenet()
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
        custom_op.parameter_map["dynamic_input"].b = True
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
        self.sess = tf.Session(config=config)
        self.parameter_path = cfg.PARAMETER_FILE
        self.saver = tf.train.Saver()

    def predict(self,image):
        img = image.convert('L')
        img = img.resize([28, 28], Image.ANTIALIAS)
        image_input = np.array(img, dtype="float32") / 255
        image_input = np.reshape(image_input, [-1, 784])

        self.saver.restore(self.sess,self.parameter_path)
        predition = self.sess.run(self.lenet.prediction, feed_dict={self.lenet.raw_input_image: image_input})
        return predition
