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
"""ResNet50 model train with MindSpore"""
import argparse
import random
import time
import numpy as np
import moxing as mox

from mindspore import context, Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, Callback, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from dataset import create_dataset, device_num
from config import cfg
from resnet import resnet50

random.seed(1)
np.random.seed(1)


class PerformanceCallback(Callback):
    """
    Training performance callback.

    Args:
        batch_size (int): Batch number for one step.
    """

    def __init__(self, batch_size):
        super(PerformanceCallback, self).__init__()
        self.batch_size = batch_size
        self.last_step = 0
        self.epoch_begin_time = 0

    def step_begin(self, run_context):
        self.epoch_begin_time = time.time()

    def step_end(self, run_context):
        params = run_context.original_args()
        cost_time = time.time() - self.epoch_begin_time
        train_steps = params.cur_step_num - self.last_step
        print(f'epoch {params.cur_epoch_num} cost time = {cost_time}, train step num: {train_steps}, '
              f'one step time: {1000*cost_time/train_steps} ms, '
              f'train samples per second of cluster: {device_num*train_steps*self.batch_size/cost_time:.1f}\n')
        self.last_step = run_context.original_args().cur_step_num


def get_lr(global_step,
           total_epochs,
           steps_per_epoch,
           lr_init=0.01,
           lr_max=0.1,
           warmup_epochs=5):
    """
    Generate learning rate array.

    Args:
        global_step (int): Initial step of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.01.
        lr_max (float): Maximum learning rate. Default: 0.1.
        warmup_epochs (int): The number of warming up epochs. Default: 5.

    Returns:
        np.array, learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(int(total_steps)):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max) * base * base
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def resnet50_train(args_opt):
    epoch_size = args_opt.epoch_size
    batch_size = cfg.batch_size
    class_num = cfg.class_num
    loss_scale_num = cfg.loss_scale
    local_data_path = '/cache/data'
    local_ckpt_path = '/cache/ckpt_file'

    # set graph mode and parallel mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_path)

    # create dataset
    print('Create train and evaluate dataset.')
    train_dataset = create_dataset(dataset_path=local_data_path, do_train=True,
                                   repeat_num=epoch_size, batch_size=batch_size)
    train_step_size = train_dataset.get_dataset_size()
    print('Create dataset success.')

    # create model
    net = resnet50(class_num=class_num)
    # reduction='mean' means that apply reduction of mean to loss
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lr = Tensor(get_lr(global_step=0, total_epochs=epoch_size, steps_per_epoch=train_step_size))
    opt = Momentum(net.trainable_params(), lr, momentum=0.9, weight_decay=1e-4, loss_scale=loss_scale_num)
    loss_scale = FixedLossScaleManager(loss_scale_num, False)

    # amp_level="O2" means that the hybrid precision of O2 mode is used for training
    # the whole network except that batchnorm will be cast into float16 format and dynamic loss scale will be used
    # 'keep_batchnorm_fp32 = False' means that use the float16 format
    model = Model(net, amp_level="O2", keep_batchnorm_fp32=False, loss_fn=loss,
                  optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})

    # define performance callback to show ips and loss callback to show loss for every epoch
    time_cb = TimeMonitor(data_size=train_step_size)
    performance_cb = PerformanceCallback(batch_size)
    loss_cb = LossMonitor()
    cb = [time_cb, performance_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * train_step_size,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory=local_ckpt_path, config=config_ck)
    cb += [ckpt_cb]

    print(f'Start run training, total epoch: {epoch_size}.')
    model.train(epoch_size, train_dataset, callbacks=cb)

    # upload checkpoint files
    print('Upload checkpoint.')
    mox.file.copy_parallel(src_url=local_ckpt_path, dst_url=args_opt.train_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')

    args_opt, unknown = parser.parse_known_args()

    resnet50_train(args_opt)
    print('ResNet50 training success!')
