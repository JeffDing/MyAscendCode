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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

def set_config(args):
    if not args.run_distribute:
        args.run_distribute = False
    config_cpu = ed({
        "num_classes": 10,
        "image_height": 256,
        "image_width": 256,
        "batch_size": 32,
        "epoch_size": 10,
        "warmup_epochs": 0,
        "lr_init": .0,
        "lr_end": 0.01,
        "lr_max": 0.001,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "label_smooth": 0.1,
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 20,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "run_distribute": args.run_distribute,
        "activation": "Softmax",
        "map":["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    })
    return config_cpu
