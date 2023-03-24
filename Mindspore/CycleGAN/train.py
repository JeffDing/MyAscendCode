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

"""General-purpose training script for image-to-image translation.
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
Example:
    Train a resnet model:
        python train.py --dataroot ./data/horse2zebra --model ResNet
"""

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from src.utils.args import get_args
from src.utils.reporter import Reporter
from src.utils.tools import get_lr, ImagePool, load_ckpt
from src.dataset.cyclegan_dataset import create_dataset
from src.models.losses import DiscriminatorLoss, GeneratorLoss
from src.models.cycle_gan import get_generator, get_discriminator, Generator, TrainOneStepG, TrainOneStepD

import zipfile
import os

ms.set_seed(1)

args = get_args("train")



def train():
    """Train function."""
    if args.need_profiler:
        from mindspore.profiler.profiling import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)
    ds = create_dataset(args)
    G_A = get_generator(args)
    G_B = get_generator(args)
    D_A = get_discriminator(args)
    D_B = get_discriminator(args)
    if args.load_ckpt:
        load_ckpt(args, G_A, G_B, D_A, D_B)
    imgae_pool_A = ImagePool(args.pool_size)
    imgae_pool_B = ImagePool(args.pool_size)
    generator = Generator(G_A, G_B, args.lambda_idt > 0)

    loss_D = DiscriminatorLoss(args, D_A, D_B)
    loss_G = GeneratorLoss(args, generator, D_A, D_B)
    optimizer_G = nn.Adam(generator.trainable_params(), get_lr(args), beta1=args.beta1)
    optimizer_D = nn.Adam(loss_D.trainable_params(), get_lr(args), beta1=args.beta1)

    net_G = TrainOneStepG(loss_G, generator, optimizer_G)
    net_D = TrainOneStepD(loss_D, optimizer_D)

    data_loader = ds.create_dict_iterator()
    if args.rank == 0:
        reporter = Reporter(args)
        reporter.info('==========start training===============')
    for _ in range(args.max_epoch):
        if args.rank == 0:
            reporter.epoch_start()
        for data in data_loader:
            img_A = data["image_A"]
            img_B = data["image_B"]
            res_G = net_G(img_A, img_B)
            fake_A = res_G[0]
            fake_B = res_G[1]
            res_D = net_D(img_A, img_B, imgae_pool_A.query(fake_A), imgae_pool_B.query(fake_B))
            if args.rank == 0:
                reporter.step_end(res_G, res_D)
                reporter.visualizer(img_A, img_B, fake_A, fake_B)
        if args.rank == 0:
            reporter.epoch_end(net_G)
        if args.need_profiler:
            profiler.analyse()
            break
    if args.rank == 0:
        reporter.info('==========end training===============')


def getZipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

if __name__ == "__main__":
    
    data_dir = '/cache/data'  
    train_dir = args.outputs_dir

    # download dataset
    if args.use_openi:
        import moxing as mox
        import json

        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

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
                context.set_context(mode=context.GRAPH_MODE,device_target=args.platform)
            if device_num > 1:
                # set device_id and init for multi-card training
                context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
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

        ###Initialize and copy data to training image
        DownloadFromQizhi(args.multi_data_url, data_dir)
        ###The dataset path is used here:data_dir + "/MNIST_Data" +"/train"  
    elif args.use_modelarts:
        import moxing as mox
        import json
        import os

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            
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
        DownloadFromQizhi(args.data_url, data_dir)
        
    train()
    
    #Compress result file
    getZipDir(dirpath=train_dir,outFullName=os.path.join(train_dir,'result.zip'))
    
    #uplode result to openi
    if args.use_openi or args.use_modelarts: 
        UploadToQizhi(train_dir,args.train_url)