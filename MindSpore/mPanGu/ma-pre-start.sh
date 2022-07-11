#!/bin/bash
# Copyright 2022 pcl
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
WORK_DIR=pangu_alpha-r1.3

jieba_file=jieba-0.42.1.tar.gz
LOCAL_DIR=$(cd "$(dirname "$0")";pwd)
echo $LOCAL_DIR
mkdir /home/work/ascend
mkdir /var/log/npu/slog/log
ln -s /var/log/npu/slog/log /home/work/ascend 

python -m pip install --upgrade pip
pip install regex
pip install zhconv
pip install sentencepiece==0.1.94
pip install ${LOCAL_DIR}/${WORK_DIR}/tokenizer/${jieba_file}

LOCAL_HIAI=/usr/local/Ascend
export TBE_IMPL_PATH=$TBE_IMPL_PATH:${LOCAL_HIAI}/ops/op_impl/built-in/ai_core/tbe/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LOCAL_HIAI}/fwkacllib/lib64/:${LOCAL_HIAI}/add-ons/:${LOCAL_HIAI}/driver/lib64/common:${LD_LIBRARY_PATH}
export PATH=$PATH:${LOCAL_HIAI}/fwkacllib/ccec_compiler/bin/:${LOCAL_HIAI}/fwkacllib/bin/:/usr/local/Ascend/toolkit/bin:${PATH}
export PYTHONPATH=$PYTHONPATH:${LOCAL_HIAI}/runtime/opp/op_impl/built-in/ai_core/tbe/:${LOCAL_HIAI}/fwkacllib/python/site-package
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
export GLOG_v=2

sudo mkdir /cache
sudo chown 1101:1101 /cache

cd /cache

sudo mkdir /cache/ckpt
sudo chown 1101:1101 /cache/ckpt

export HCCL_CONNECT_TIMEOUT=1800



