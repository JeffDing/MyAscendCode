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
"""train resnet."""
import os
import argparse
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel import set_algo_parameters
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.nn.optim import thor
from mindspore.train.model import Model

from src.resnet import resnet50 as resnet
from src.dataset import create_dataset
from src.crossentropy import CrossEntropy as CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_num', type=int, default=1, help='Device num')
parser.add_argument('--multi_data_url',help='path to multi dataset',default= '/cache/data/')
parser.add_argument('--train_url', help='ckpt folder to save/load',default= '/cache/output/')
parser.add_argument('--ckpt_url', help='model folder to save/load',default= '/cache/output/')
parser.add_argument('--run_qizhi', type=bool,help='run in qizhi',default=False)
args_opt = parser.parse_args()

device_num =args_opt.device_num

if args_opt.device_target == "Ascend":
    from src.config import config
else:
    from src.config import config_gpu as config

set_seed(1)

if args_opt.run_qizhi:
    import moxing as mox
    import json
    import time
    
    ### Copy multiple datasets from obs to training image and unzip###  
    def C2netMultiObsToEnv(multi_data_url, data_dir):
        #--multi_data_url is json data, need to do json parsing for multi_data_url
        multi_data_json = json.loads(multi_data_url)  
        for i in range(len(multi_data_json)):
            zipfile_path = data_dir + "/" + multi_data_json[i]["dataset_name"]
            try:
                mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path) 
                print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],zipfile_path))
                #get filename and unzip the dataset
                filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
                filePath = data_dir + "/" + filename
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                os.system("unzip {} -d {}".format(zipfile_path, filePath))

            except Exception as e:
                print('moxing download {} to {} failed: '.format(
                    multi_data_json[i]["dataset_url"], zipfile_path) + str(e))
        #Set a cache file to determine whether the data has been copied to obs. 
        #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
        f = open("/cache/download_input.txt", 'w')    
        f.close()
        try:
            if os.path.exists("/cache/download_input.txt"):
                print("download_input succeed")
        except Exception as e:
            print("download_input failed")
        return 
    ### Copy the output model to obs ###  
    def EnvToObs(train_dir, obs_train_url):
        try:
            mox.file.copy_parallel(train_dir, obs_train_url)
            print("Successfully Upload {} to {}".format(train_dir,
                                                        obs_train_url))
        except Exception as e:
            print('moxing upload {} to {} failed: '.format(train_dir,
                                                        obs_train_url) + str(e))
        return                                                       
    def DownloadFromQizhi(multi_data_url, data_dir):
        device_num = int(os.getenv('RANK_SIZE'))
        if device_num == 1:
            C2netMultiObsToEnv(multi_data_url,data_dir)
            context.set_context(mode=context.GRAPH_MODE,device_target=args_opt.device_target)
        if device_num > 1:
            # set device_id and init for multi-card training
            context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
            init()
            #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
            local_rank=int(os.getenv('RANK_ID'))
            if local_rank%8==0:
                C2netMultiObsToEnv(multi_data_url,data_dir)
            #If the cache file does not exist, it means that the copy data has not been completed,
            #and Wait for 0th card to finish copying data
            while not os.path.exists("/cache/download_input.txt"):
                time.sleep(1)  
        return
    def UploadToQizhi(train_dir, obs_train_url):
        device_num = int(os.getenv('RANK_SIZE'))
        local_rank=int(os.getenv('RANK_ID'))
        if device_num == 1:
            EnvToObs(train_dir, obs_train_url)
        if device_num > 1:
            if local_rank%8==0:
                EnvToObs(train_dir, obs_train_url)
        return


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def get_thor_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch, decay_epochs=100):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        if epoch >= decay_epochs:
            lr_local = lr_local * 0.5
        if epoch >= decay_epochs + 1:
            lr_local = lr_local * 0.5
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def get_thor_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)
    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now




if __name__ == '__main__':
    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path
    
    if args_opt.run_qizhi:
        ###Initialize and copy data to training image
        data_dir = '/cache/data'  
        train_dir = '/cache/output'

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        DownloadFromQizhi(args_opt.multi_data_url, data_dir)
    else:
        train_dir = args_opt.ckpt_url

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if args_opt.run_distribute:
        if target == "Ascend":
            #device_id = int(os.getenv('DEVICE_ID'))
            #context.set_context(device_id=device_id)
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
            init()
        # GPU target
        else:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)

    # init lr
    lr = get_thor_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    lr = Tensor(lr)

    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    metrics = {"acc"}
    damping = get_thor_damping(0, config.damping_init, config.damping_decay, 70, step_size)
    split_indices = [26, 53]
    opt = thor(net, lr, Tensor(damping), config.momentum, config.weight_decay, config.loss_scale,
               config.batch_size, split_indices=split_indices, frequency=config.frequency)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                  amp_level="O2", keep_batchnorm_fp32=False)

    model = ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                      loss_scale_manager=loss_scale, metrics={'acc'},
                                                      amp_level="O2", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        
        if device_num == 1:
            outputDirectory = train_dir 
        if device_num > 1:
            outputDirectory = train_dir + "/" + str(get_rank()) + "/"
        
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=outputDirectory, config=config_ck)
        cb += [ckpt_cb]

    # train model
    dataset_sink_mode = True
    model.train(config.epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)

    if args_opt.run_qizhi:
        UploadToQizhi(train_dir,args_opt.train_url)