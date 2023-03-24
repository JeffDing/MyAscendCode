# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train"""
import os
import time
import json

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer

### Copy multiple datasets from obs to training image and unzip###  
def init_openi(mode="train"):
    work_dir = "/cache/"

    data_dir = os.path.join(work_dir, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    model_dir = os.path.join(work_dir, "data")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return model_dir, data_dir


def get_git_pretrained_ckpt_file(args):
    current_path = os.path.abspath(__file__)
    code_dir = os.path.dirname(current_path)
    ckpt_file_path = os.path.join(code_dir, "src/configs/{}.ckpt".format(args.arch))
    if not os.path.exists(ckpt_file_path):
        raise ValueError("pretrained ckpt file: {} not exists!".format(ckpt_file_path))
    
    return ckpt_file_path

def get_obs_pretrained_ckpt_file(model_dir):
    pretrained_dir = os.path.join(model_dir, "pretrained")
    if not os.path.exists(pretrained_dir):
        os.mkdir(pretrained_dir)
    pretrained_ckpt_file = os.path.join(pretrained_dir, "swinv2_large_patch4_window16_2560-120_3336.ckpt")

    return pretrained_ckpt_file


def main():
    assert args.crop, f"{args.arch} is only for evaluation"
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)

    model_dir = "/model/"
    data_dir = "/dataset/"
    if args.run_qizhi:
        import moxing as mox
        ### Copy single dataset from obs to training image###
        def ObsToEnv(obs_data_url, data_dir):
            try:     
                mox.file.copy_parallel(obs_data_url, data_dir)
                print("Successfully Download {} to {}".format(obs_data_url, data_dir))
            except Exception as e:
                print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
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
        ### Copy the output to obs###
        def EnvToObs(train_dir, obs_train_url):
            try:
                mox.file.copy_parallel(train_dir, obs_train_url)
                print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
            except Exception as e:
                print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
            return      
        def DownloadFromQizhi(obs_data_url, data_dir):
            device_num = int(os.getenv('RANK_SIZE'))
            if device_num == 1:
                ObsToEnv(obs_data_url,data_dir)
                context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)
            if device_num > 1:
                # set device_id and init for multi-card training
                context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
                init()
                #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
                local_rank=int(os.getenv('RANK_ID'))
                if local_rank%8==0:
                    ObsToEnv(obs_data_url,data_dir)
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
        model_dir, data_dir = init_openi()
        DownloadFromQizhi(args.data_url, data_dir)

    if args.run_openi:
        import moxing as mox
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
                context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)
            if device_num > 1:
                # set device_id and init for multi-card training
                context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
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
        model_dir, data_dir = init_openi()
        DownloadFromQizhi(args.multi_data_url, data_dir)

    train_dir = os.path.join(model_dir, "ckpt_{:04d}".format(rank))
    best_model_dir = os.path.join(model_dir, "ckpt_best_{:04d}".format(rank))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(best_model_dir):
        os.mkdir(best_model_dir)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained_ckpt_file = get_git_pretrained_ckpt_file(args)
        args.pretrained = pretrained_ckpt_file
        pretrained(args, net)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num,
                                 keep_checkpoint_max=args.best_every)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())

    model_prefix = "{}_{:04d}".format(args.arch, rank)
    ckpoint_cb = ModelCheckpoint(prefix=model_prefix, directory=train_dir, config=config_ck)

    print_steps = batch_num // 10
    loss_cb = LossMonitor(per_print_times=print_steps)
    eval_cb = EvaluateCallBack(
        model, eval_dataset=data.val_dataset, src_url=train_dir,
        train_url=os.path.join(args.train_url, "ckpt_{:04d}".format(rank)),
        rank=rank, model_prefix=model_prefix, batch_num=batch_num,
        best_model_dir=best_model_dir, best_freq=args.best_every, save_freq=args.save_every)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb],
                dataset_sink_mode=args.dataset_sink_mode)
    print("train success")

    if args.run_openi or args.run_qizhi:
        UploadToQizhi(best_model_dir,args.train_url)


if __name__ == '__main__':
    main()
