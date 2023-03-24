# **MindSpore SPONGE暑期学校新手教程**

### 环境的配置

在终端依次执行如下命令

```bash
pip install mindspore_gpu-1.8.0-cp37-cp37m-linux_x86_64.whl
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd mindscience/MindSPONGE/
pip install -r requirements.txt
bash build.sh -e gpu -j32 -t on -c on
cd output/
pip install mindscience_sponge_gpu-0.1.0rc1-py3-none-any.whl
pip install mindscience_cybertron-0.1.0rc1-py3-none-any.whl
```

环境配置完成