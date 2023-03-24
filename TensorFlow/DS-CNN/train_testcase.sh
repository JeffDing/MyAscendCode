set -x
export POD_NAME=another0

execpath=${PWD}

output="./output_base_v2"
rm -rf *.pbtxt
ulimit -c 0

rm -rf speech_commands_train
rm -rf summary

start_time=`date +%s`
python3 train.py --model_architecture ds_cnn \
      --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
      --dct_coefficient_count 40 \
      --window_size_ms 30 \
      --window_stride_ms 10 \
      --learning_rate 0.001,0.0001,0.00001 \
      --how_many_training_steps 10000,10000,10000 \
      --batch_size 64 \
      --data_dir /data/speech_dataset/ \
      --summaries_dir ./summary \
      --train_dir ./speech_commands_train > train.log 2>&1
    #--model_dir ./model_path

end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="cross entropy"  #功能检查字
key2="Final test accuracy"  #精度检查字
#key3="val_loss"  #性能检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi


echo execution time was `expr $end_time - $start_time` s.
