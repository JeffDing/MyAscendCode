# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
Cybertron tutorial 06: Multi-task with multiple readouts (example 2)
"""

import sys
import time
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

if __name__ == '__main__':

    sys.path.append('..')

    from cybertron import Cybertron
    from cybertron import MolCT
    from cybertron.train import MAE, MLoss
    from cybertron.train import WithLabelLossCell, WithLabelEvalCell
    from cybertron.train import TrainMonitor
    from cybertron.train import TransformerLR

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    data_name = sys.path[0] + '/dataset_qm9_normed_'
    train_file = data_name + 'trainset_1024.npz'
    valid_file = data_name + 'validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    # diplole,polarizability,HOMO,LUMO,gap,R2,zpve,capacity
    idx = [0, 1, 2, 3, 4, 5, 6, 11]

    num_atom = int(train_data['num_atoms'])
    scale = Tensor(train_data['scale'][idx], ms.float32)
    shift = Tensor(train_data['shift'][idx], ms.float32)
    ref = Tensor(train_data['type_ref'][:, idx], ms.float32)

    mod = MolCT(
        cutoff=1,
        n_interaction=3,
        dim_feature=128,
        n_heads=8,
        activation='swish',
        max_cycles=1,
        length_unit='nm',
    )

    net = Cybertron(mod, readout='graph', dim_output=[1, 1, 3, 1, 1, 1],
                    num_atoms=num_atom, length_unit='nm')

    net.print_info()

    tot_params = 0
    for i, param in enumerate(net.get_parameters()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)

    n_epoch = 8
    repeat_time = 1
    batch_size = 16

    ds_train = ds.NumpySlicesDataset(
        {'R': train_data['R'], 'Z': train_data['Z'], 'E': train_data['E'][:, idx]}, shuffle=True)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset(
        {'R': valid_data['R'], 'Z': valid_data['Z'], 'E': valid_data['E'][:, idx]}, shuffle=False)
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    loss_network = WithLabelLossCell('RZE', net, nn.MAELoss())
    eval_network = WithLabelEvalCell(
        'RZE', net, nn.MAELoss(), scale=scale, shift=shift, type_ref=ref)

    lr = TransformerLR(learning_rate=1., warmup_steps=4000, dimension=128)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    eval_mae = 'EvalMAE'
    atom_mae = 'AtomMAE'
    eval_loss = 'Evalloss'
    model = Model(loss_network, optimizer=optim, eval_network=eval_network,
                  metrics={eval_mae: MAE([1, 2], reduce_all_dims=False),
                           atom_mae: MAE([1, 2, 3], reduce_all_dims=False, averaged_by_atoms=True),
                           eval_loss: MLoss(0)},)

    outdir = 'Tutorial_C06'
    outname = outdir + '_' + net.model_name
    record_cb = TrainMonitor(model, outname, per_step=32, avg_steps=32,
                             directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=eval_loss)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=32, keep_checkpoint_max=64, append_info=[net.hyper_param])
    ckpoint_cb = ModelCheckpoint(
        prefix=outname, directory=outdir, config=config_ck)

    np.set_printoptions(linewidth=200)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch, ds_train, callbacks=[
                record_cb, ckpoint_cb], dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print("Training Fininshed!")
    print("Training Time: %02d:%02d:%02d" % (h, m, s))
