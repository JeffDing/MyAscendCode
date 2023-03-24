import os
import argparse

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init

import time

from ddm import Unet, GaussianDiffusion, Trainer

parser = argparse.ArgumentParser(description='train afhq dataset')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--local_data_root', default='/cache/data/',
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', metavar='DIR',
                    default='', help='path to dataset')
parser.add_argument('--train_url', metavar='DIR',
                    default='', help='save output')
parser.add_argument('--multi_data_url',help='path to multi dataset',
                    default= '/cache/data/')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--image_size', default=512, type=int,
                    metavar='N', help='img size')
parser.add_argument('--sampling_timesteps', default=250, type=int,
                    metavar='N', help='')
parser.add_argument('--train_num_steps', default=50001, type=int,
                    metavar='N', help='')
parser.add_argument('--save_and_sample_every', default=2000, type=int,
                    metavar='N', help='')
parser.add_argument('--gradient_accumulate_every', default=2, type=int,
                    metavar='N', help='')
parser.add_argument('--ckpt_url', type=str, default=None,
                    help='load ckpt file path')
parser.add_argument('--use_qizhi', type=bool, default=False,
                    help='use qizhi')
parser.add_argument('--use_zhisuan', type=bool, default=False,
                    help='use zhisuan')
args = parser.parse_args()

if args.use_qizhi:
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
    
    data_dir = '/cache/data/'  
    train_dir = '/cache/output/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    ###Initialize and copy data to training image
    DownloadFromQizhi(args.data_url, data_dir)

if args.use_zhisuan:
    import moxing as mox
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
    
    data_dir = '/cache/data/'  
    train_dir = '/cache/output/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    ###Initialize and copy data to training image
    DownloadFromQizhi(args.multi_data_url, data_dir)

path = args.local_data_root

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=args.image_size,
    timesteps=1000,             # number of steps
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    sampling_timesteps=args.sampling_timesteps,
    loss_type='l1'            # L1 or L2
)

if args.use_qizhi == False and args.use_zhisuan == False:
    train_dir=args.train_url
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
trainer = Trainer(
    diffusion,
    path,
    train_batch_size=args.batch_size,
    train_lr=8e-5,
    train_num_steps=args.train_num_steps,         # total training steps
    gradient_accumulate_every=args.gradient_accumulate_every,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp_level='O1',                        # turn on mixed precision
    save_and_sample_every=args.save_and_sample_every,
    results_folder=os.path.join(train_dir, 'results'),
    train_url=train_dir
)

if args.ckpt_url is not None:
    trainer.load(args.ckpt_url)
    print('load ckpt successfully')

trainer.train()

if args.use_qizhi or args.use_zhisuan:
    UploadToQizhi(train_dir,args.train_url)