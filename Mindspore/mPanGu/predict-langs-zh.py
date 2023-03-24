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
"""
PanGu predict run
"""
import os

import numpy as np

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.model import Model
## from mindspore.train.serialization import load_distributed_checkpoint
from src.serialization import load_distributed_checkpoint
from src.pangu_alpha import PanguAlpha, EvalNet
from src.pangu_alpha_config import PANGUALPHAConfig, set_parse
from src.utils_m53_exp2 import get_args

import time
import moxing as mox
from src.utils_m53_exp2 import download_data, ckpt_copy_tar_new, get_ckpt_file_list

def load_model(args_opt):
    r"""
     The main function for load model
    """
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        
        local_strategy_ckpt_path="/cache/ckpt_strategy.ckpt"
        if rank % 8 == 0:
            os.system('ulimit -s 102400')
            mox.file.copy(src_url=args_opt.strategy_load_ckpt_path, dst_url=local_strategy_ckpt_path)
            steps_name = args_opt.load_ckpt_obs_path.split('/')[-1].split('-')[-1]
            print("steps_name", steps_name)
            name_in = args_opt.load_obs_ckptname
            ckpt_copy_tar_new(args_opt.load_ckpt_obs_path+name_in, target_path=args_opt.load_ckpt_local_path)
            mox.file.copy(f'{args_opt.load_ckpt_obs_path}/{name_in}_word_embedding.npy', f'{args_opt.load_ckpt_local_path}/word_embedding.npy')
            mox.file.copy(f'{args_opt.load_ckpt_obs_path}/{name_in}_position_embedding.npy', f'{args_opt.load_ckpt_local_path}/position_embedding.npy')
            mox.file.copy(f'{args_opt.load_ckpt_obs_path}/{name_in}_top_query_embedding.npy', f'{args_opt.load_ckpt_local_path}/top_query_embedding.npy')
            print("setting env success.")
            # 下载模型文件结束后，写一个文件来表示下载成功
            f = open("/tmp/download_ckpt.txt", 'w')
            f.close()
        # 此处用于阻塞其他进程，直到刷包以及下载数据集完成为止
        while not os.path.exists("/tmp/download_ckpt.txt"):
            time.sleep(1)
        print("\n\n************Checkpoint download succeed!*************\n\n", flush=True)
        if rank % 8 == 0:
            print(os.listdir(args_opt.load_ckpt_local_path), flush=True)
                
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            strategy_ckpt_load_file=local_strategy_ckpt_path,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=local_strategy_ckpt_path)

    use_past = (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        compute_dtype=mstype.float16,
        use_past=use_past,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        eod_reset=False,
        word_emb_dp=True,
        load_ckpt_path=args_opt.load_ckpt_local_path,
        param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    ckpt_name = args_opt.load_ckpt_name
    # Define network
    pangu_alpha = PanguAlpha(config)
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_true, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)
    ##------------------------------------------------------------------------------------------------------
    print("======start load_distributed checkpoint", flush=True)
    ckpt_file_list = get_ckpt_file_list(args_opt.load_ckpt_local_path, device_num=128) 
    # For 2.6B and 13B models, the number of ckpt files is 512.
    ## ckpt_name = 'filerted'
    ## ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"{ckpt_name}_{ckpt_rank}.ckpt") for ckpt_rank in
    ##                   range(0, 512)]
    print(ckpt_file_list)
    print(f"Loading from path {ckpt_file_list[0]}", flush=True)
    # Load checkpoint files
    # print(ckpt_file_list)
    # print(predict_layout)
    load_distributed_checkpoint(eval_net, ckpt_file_list, predict_strategy=predict_layout)
    print("================load param ok=================", flush=True)
    ##-------------------------------------------------------------------------------------------------------
    # from src.serialization import save_checkpoint
    # save_checkpoint(eval_net, '/cache/ckpt_test.ckpt')
    # if rank == 0:
    #     mox.file.copy('/cache/ckpt_test.ckpt', 's3://research-my/ckpt_test.ckpt')
    return model_predict, config

def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(model_predict.predict_network, inputs_np, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(model_predict.predict_network, inputs_np_1, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt):
    """run predict"""
    from src.tokenization_jieba import JIEBATokenizer
    from src.generate import generate, generate_increment
    # Define tokenizer
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab10.vocab'),
                               os.path.join(args_opt.tokenizer_path, 'vocab10.model'))

    # Tokenize input sentence to ids
    sample = "今天是一个好天气"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    generate_func = generate_increment if config.use_past else generate
    output_ids = generate_func(model_predict, input_ids, args_opt)
    # Decode output ids to sentence
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('Output is:', output_samples, flush=True)

def run_predict_langs21(model_predict, config, args_opt):
    """run predict"""
    from tokenizer.tokenizer_spm import SpmTokenizer
    from src.generate import generate, generate_increment
    from tokenizer.tokenizer_spm import langs_ID, translate_ID
    import jieba
    D.init()
    rank = D.get_rank()
    
    work_dir = '/home/work/user-job-dir/pangu_alpha-r1.3'
    # Define tokenizer
    vocab_file = work_dir + '/tokenizer/spm.128k.model.1'
    tokenizer = SpmTokenizer(vocab_file)
    EOT = tokenizer.eot_id
    # inference mode
    generate_func = generate_increment if config.use_past else generate
    
    #------------------------------------------------------------------
    # Tokenize input sentence to ids, example
    sample = "你 今天 中午 吃的 什么 ？"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    print('000000000000'*20)
    print(input_ids)
    output_ids = generate_func(model_predict, input_ids, args_opt, dynamic_generate_length=20)
    # Decode output ids to sentence
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('\nExample output is:', output_samples, flush=True)
    #------------------------------------------------------------------
    
    
    """
    langs_zh = ['vi', 'ko', 'en', 'nl', 'de', 'ms', 'id', 'tl', 'mn', 'my', 'th', 'lo', 'km', 'lt', 'et', 'lv',
                'hu', 'pl', 'cs', 'sk', 'sl', 'hr', 'bs', 'sr', 'bg', 'mk', 'ru', 'uk', 'be', 'sq', 'el', 'ka',
                'hy', 'ro', 'fr', 'es', 'pt', 'fa', 'he', 'ar', 'ps', 'tr', 'hi', 'ta', 'ur',
                'bn', 'si', 'ne']# 51
    
    langs_zh = [['vi', 'ko', 'en', 'nl', 'de'], ['ms', 'id', 'tl', 'mn', 'my'], ['th', 'lo', 'km', 'lt', 'et'], ['lv',
                'hu', 'pl', 'cs', 'sk'], ['sl', 'hr', 'bs', 'sr', 'bg'], ['mk', 'ru', 'uk', 'be', 'sq'], ['el', 'ka',
                'hy', 'ro', 'fr'], ['es', 'pt', 'fa', 'he', 'ar'], ['ps', 'tr'], ['hi', 'ta', 'ur', 'bn', 'si', 'ne']]# 0-8 'ps', 'tr', 
    """
    langs_zh = [['vi', 'ko'], ['en', 'nl'], ['de', 'ms'], ['id', 'tl'], ['mn', 'my'], ['th', 'lo'], ['km', 'lt'], ['et', 'lv'],
                ['hu', 'pl'], ['cs', 'sk'], ['sl', 'hr'], ['bs', 'sr'], ['bg', 'mk'], ['ru', 'uk'], ['be', 'sq'], ['el', 'ka'],
                ['hy', 'ro'], ['fr', 'es'], ['pt', 'fa'], ['he', 'ar'], ['ps', 'tr'], ['hi', 'ta'], ['ur', 'bn'], ['si', 'ne']]# 0-23 'ps', 'tr', 
    # translate_file_path = work_dir + '/tokenizer/datasets/en-zh_matrix--0.txt'
    # langs = translate_file_path.split('/')[-1][:2]
    langs_5 = langs_zh[args_opt.language_idx]
                
    for langs in langs_5:
        try:
            result = []
            translate_file_path = work_dir + f'/tokenizer/langs_53/zh-langs/{langs}-zh_corpus-test.txt'

            obs_sub_dir = args_opt.load_obs_ckptname.split('_')[0] # exp4-31000
            obs_save_dir = f""
            if not mox.file.exists(obs_save_dir):
                print("Creating translate output bueckt dir {}".format(obs_save_dir))
                mox.file.make_dirs(obs_save_dir)

            local_output_save_path = f"/cache/output_zh-{langs}-corpus.txt"
            obs_upload_path =os.path.join(obs_save_dir, f"output_zh-{langs}-corpus.txt")

            src_langs = 'zh'
            tag_langs = langs
            # src_langs = langs
            # tag_langs = 'en'
            with open(translate_file_path, 'r', encoding='utf-8') as f:
                if 'corpus' in translate_file_path:
                    langs_flags_id = langs_ID[langs]

                    for idx, data in enumerate(f.read().split("\n\n")):
                        if data:
                            src_txt, tag_txt = data.split('\t')

                            tokenized_src_langs = tokenizer.tokenize(''.join(jieba.cut(''+src_txt)))
                            src_id = tokenizer.convert_tokens_to_ids(tokenized_src_langs)

                            tokenized_tag_langs = tokenizer.tokenize(''+tag_txt)
                            tag_id = tokenizer.convert_tokens_to_ids(tokenized_tag_langs)
                            # Tokenize input sentence to ids 

                            src_trans2_tag_input = [langs_ID[src_langs], langs_ID[src_langs], langs_ID[src_langs]] +\
                                                    src_id + \
                                                    [translate_ID, translate_ID, translate_ID] + \
                                                    [langs_ID[tag_langs], langs_ID[tag_langs], langs_ID[tag_langs]]

                            tag_trans2_src_input = [langs_ID[tag_langs], langs_ID[tag_langs], langs_ID[tag_langs]] + \
                                                    tag_id + \
                                                    [translate_ID, translate_ID, translate_ID] + \
                                                    [langs_ID[src_langs], langs_ID[src_langs], langs_ID[src_langs]]

                            src_out_max_len = min(len(src_id)+10, 512)
                            tag_out_max_len = min(len(tag_id)+20, 512)

                            # Call inference
                            tag2src_output_ids = generate_func(model_predict, np.array([tag_trans2_src_input]), args_opt, dynamic_generate_length=src_out_max_len).tolist()
                            src2tag_output_ids = generate_func(model_predict, np.array([src_trans2_tag_input]), args_opt, dynamic_generate_length=tag_out_max_len).tolist()
                            # Decode output ids to sentence
                            src_output = tokenizer.convert_ids_to_tokens(src2tag_output_ids[len(src_trans2_tag_input):])
                            tag_output = tokenizer.convert_ids_to_tokens(tag2src_output_ids[len(tag_trans2_src_input):])
                            tag_output.replace(' ', '')
                            result.append(tag_output + '\t' + src_output)
                            if rank == 0:
                                print("----------------------------------------------------------")
                                print(idx, " INPUT is : ", data, "\n")
                                print(idx, " OUTPUT is : " + tag_output + "\n" + src_output)

                            if rank == 0 and idx%20 == 0:
                                with open(local_output_save_path, 'w')as f_output:
                                    for i, i_txt in enumerate(result):
                                        f_output.writelines(str(i) + '\t' + i_txt+"\n")
                                try:
                                    mox.file.copy(local_output_save_path, obs_upload_path)
                                except:
                                    print("Copy to obs Error...")
                    time.sleep(2)
                    if rank == 0:
                        with open(local_output_save_path, 'w')as f_output:
                            for i, i_txt in enumerate(result):
                                f_output.writelines(str(i) + '\t' + i_txt+"\n")
                        mox.file.copy(local_output_save_path, obs_upload_path)
                        print("Copy the output file {} to the obs:{}".format(local_output_save_path, obs_upload_path))
        except:
            print(langs, " error, comtinue...")
                
    
def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)
    model_predict, config = load_model(opt)
    if opt.export:
        export_mindir(model_predict, config)
    else:
        run_predict_langs21(model_predict, config, opt)


if __name__ == "__main__":
    main()
