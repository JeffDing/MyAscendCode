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
"""
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_dataset
from src.lenet import LeNet5

import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import moxing as mox

os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"]='0'

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

set_seed(1)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_lenet():

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    local_data_url = '/cache/data'
    local_train_url = '/cache/ckpt'

    mox.file.copy_parallel(config.data_url,local_data_url)

    ds_train = create_dataset(os.path.join(local_data_url, "train"), config.batch_size)
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = LeNet5(config.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=local_train_url, config=config_ck)

    if config.device_target != "Ascend":
        if config.device_target == "GPU":
            context.set_context(enable_graph_kernel=True)
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    else:
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")

    print("============== Starting Training ==============")
    model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()])
    mox.file.copy_parallel(local_train_url,config.train_url)

def eval_lenet():

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    local_data_url = '/cache/data'
    local_ckpt_url = '/cache/ckpt'

    mox.file.copy_parallel(config.data_url,local_data_url)

    if config.ckpt_path:
        checkpoint_file=os.path.join(local_ckpt_url,os.path.split(config.ckpt_path)[1])

    mox.file.copy_parallel(config.ckpt_path,checkpoint_file)

    network = LeNet5(config.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # repeat_size = config.epoch_size
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(config.data_path, "test"),
                             config.batch_size,
                             1)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))


if __name__ == "__main__":
    if config.eval == False:
        train_lenet()
    else:
        eval_lenet()