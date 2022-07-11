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
"""Train mobilenetV2 on ImageNet."""

import time

from mindspore import Tensor, nn
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.common import set_seed

from src.dataset import extract_features
from src.lr_generator import get_lr
from src.config import set_config
from src.args import train_parse_args
from src.utils import context_device_init, export_mindir, predict_from_net, get_samples_from_eval_dataset
from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt, get_networks, train

set_seed(1)

if __name__ == '__main__':
    args_opt = train_parse_args()
    config = set_config(args_opt)
    start = time.time()

    # set context and device init
    context_device_init(config)

    # define network
    backbone_net, head_net, net = define_net(config, activation="Softmax")

    # load parameters into backbone net from pre_training checkpoint
    load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)

    # show test img and predict label pre training
    test_list = get_samples_from_eval_dataset(args_opt.dataset_path)
    predict_from_net(net, test_list, config, show_title="pre training")
    
    # catch backbone features
    data, step_size = extract_features(backbone_net, args_opt.dataset_path, config)

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # get learning rate
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size))

    # get optimizer
    opt = Momentum(filter(lambda x: x.requires_grad, head_net.get_parameters()), lr, config.momentum, config.weight_decay)
    
    # define train and eval networks and start training
    train_net, eval_net = get_networks(head_net, loss, opt)
    train(train_net, eval_net, net, data, config)
    print("train total cost {:5.4f} s".format(time.time() - start))
    
    # show test img and predict label after training
    predict_from_net(net, test_list, config, show_title="after training")
    # export mindir file after training
    export_mindir(net, "mobilenetv2")
