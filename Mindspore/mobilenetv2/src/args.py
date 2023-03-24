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
import os
import argparse
import ast

def train_parse_args():
    train_parser = argparse.ArgumentParser(description='Image classification trian')
    train_parser.add_argument('--platform', type=str, default="Ascend", choices=("CPU", "GPU", "Ascend"), \
        help='run platform, only support CPU, GPU and Ascend')
    train_parser.add_argument('--dataset_path', type=str, default="dataset", help='Dataset path')
    train_parser.add_argument('--pretrain_ckpt', type=str, default="mobilenetV2.ckpt", help='Pretrained checkpoint path \
        for fine tune or incremental learning')
    train_parser.add_argument('--freeze_layer', type=str, default="backbone", choices=["", "none", "backbone"], \
        help="freeze the weights of network from start to which layers")
    train_parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    train_parser.add_argument('--filter_head', type=ast.literal_eval, default=False,\
                              help='Filter head weight parameters when load checkpoint, default is False.')
    train_args = train_parser.parse_args()
    train_args.is_training = True
    if train_args.platform == "CPU":
        train_args.run_distribute = False
    train_args.dataset_path = os.path.abspath(train_args.dataset_path)
    return train_args
