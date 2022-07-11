# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# -*- coding: utf-8 -*-
try:
    from pip._internal import main
    main(["install", "opencv-python", "matplotlib"])
except:
    raise ValueError("Please install opencv-python and matplotlib anually.")
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import shutil
import random
import numpy as np

from mindspore import context, Tensor
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import get_rank, init, get_group_size
from mindspore.train.serialization import export

def switch_precision(net, data_type, config):
    if config.platform == "Ascend":
        net.to_float(data_type)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)


def context_device_init(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)

    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
        if config.run_distribute:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    elif config.platform == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=device_id,
                            save_graphs=False)
        if config.run_distribute:
            context.set_auto_parallel_context(device_num=config.rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True, all_reduce_fusion_config=[140])
            init()
    else:
        raise ValueError("Only support CPU, GPU and Ascend.")


def set_context(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            save_graphs=False)
    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            device_id=config.device_id, save_graphs=False)
    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=config.platform, save_graphs=False)


def delWithCmd(path):
    path = os.path.abspath(path)
    try:
        if os.path.isfile(path):
            cmd = 'rm -rf '+ path
            os.system(cmd)
        if os.path.isdir(path):
            for f in os.listdir(path):
                delWithCmd(os.path.join(path, f))
    except Exception as e:
        print(e)

def export_mindir(net, name):
    input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
    net.set_train(mode=False)
    path = os.path.abspath(f"{name}.mindir")
    export(net, Tensor(input_np), file_name=path, file_format='MINDIR')
    print(f"export {name} MINDIR file at {path}")

def prepare_ckpt(config):
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_0')
    try:
        print("remove pre checkpoint")
        shutil.rmtree(save_ckpt_path)
    except Exception as e:
        delWithCmd(save_ckpt_path)
    if not os.path.isdir(save_ckpt_path):
        os.mkdir(save_ckpt_path)

def read_img(img_path, config):
    img_s = cv2.imread(img_path)
    h, w, c = img_s.shape
    new_w = int(500 / h * w)
    img_s = cv2.resize(img_s, (new_w, 500))
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    img_height = img_s.shape[0]
    img_width = img_s.shape[1]
    img = cv2.resize(img_s, (config.image_width, config.image_height))

    mean = np.array([0.406 * 255, 0.456 * 255, 0.485 * 255]).reshape((1, 1, 3))
    std = np.array([0.225 * 255, 0.224 * 255, 0.229 * 255]).reshape((1, 1, 3))
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1)).reshape((1, 3, config.image_width, config.image_height)).astype(np.float32)
    return img_s, img, img_width, img_height

def predict_from_net(net, img_list, config, show_title="test"):
    for i, file_path in enumerate(img_list):
        img_s, img, img_width, img_height = read_img(file_path, config)
        output = net(Tensor(img)).asnumpy()
        index = output[0].argmax()
        cls = config.map[index]
        plt.subplot(2, 3, i + 1) 
        plt.imshow(np.squeeze(img_s))
        plt.title('class:%s'%cls)
        plt.xticks([])
        plt.axis("off")
    plt.show()
    # predict_decode(output, img_s, config, show_title, save_path=None)

def get_samples_from_eval_dataset(dataset_path, sample_nums=6):
    dirs = []
    for sub_dir in os.listdir(os.path.join(dataset_path, "eval")):
        for file_name in os.listdir(os.path.join(dataset_path, "eval", sub_dir)):
            dirs.append(os.path.join(dataset_path, "eval", sub_dir, file_name))
    return random.sample(dirs, sample_nums)
