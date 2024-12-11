# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

from mindspeed_llm import megatron_adaptor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.legacy.model import GPTModel
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from mindspeed_llm.tasks.inference.infer_base import task_factory, add_text_generate_args
from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM

from torch_npu.contrib import transfer_to_npu
from transformers import AutoTokenizer, AutoModel

import torch
import numpy as np
from megatron.training.utils import get_ltor_masks_and_position_ids
import torch.nn.functional as F


def model_provider(pre_process=True, post_process=True) -> Union[GPTModelInfer, GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModelInfer, GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = GPTModel(
            config,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def main():
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    model = GPTModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )
    
    hf_tokenizer = AutoTokenizer.from_pretrained("./model_from_hf/internlm2_5-7b-chat", trust_remote_code=True)

    hf_model = AutoModel.from_pretrained("./model_from_hf/internlm2_5-7b-chat", trust_remote_code=True).half().cuda()

    inputs = torch.tensor([i for i in range(10000, 12048)]).unsqueeze(0).npu()
    with torch.no_grad():
        hf_outputs = hf_model.forward(input_ids=inputs)
        hf_logits = hf_outputs.logits
        print("Huggingface:\n")
        print(hf_logits.shape, hf_logits.dtype)
        print(hf_logits)
        np.save("./forward_out/hf_internlm2_5_7b_chat_logits_fp16.npy", hf_logits.cpu().numpy())
    
    
    input_ids = torch.tensor([i for i in range(10000, 12048)]).unsqueeze(0).npu()
    eod = 0
    reset_position_ids =False
    reset_attention_mask =False
    eod_mask_loss = False
    attention_mask, loss_mask,_= get_ltor_masks_and_position_ids(
        input_ids,
        eod,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss)
    with torch.no_grad():
        modellink_outputs = model.forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask.npu())
        print("ModelLink:")
        print(modellink_outputs.shape, modellink_outputs.dtype)
        print(modellink_outputs)
        
    cosine_similarity =F.cosine_similarity(hf_logits, modellink_outputs, dim=1).mean()
    print(f"Cosine Similarity Between HF and modellink embeddings: {cosine_similarity}")
    
    np.save("./forward_out/modellink_hf_internlm2_5_7b_chat_logits_fp16.npy", modellink_outputs.to(torch.float).cpu().numpy())


if __name__ == "__main__":
    main()
